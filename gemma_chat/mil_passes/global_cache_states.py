"""MIL pass: turn the global KV cache input/output pairs into Core ML state.

Why this is a *post*-materialization pass
-----------------------------------------
The sliding KV caches are bound to Core ML state during StableHLO→MIL
conversion (``stablehlo_coreml``'s ``StateSpec``), because their shape is
static from the start.  The 3 global caches cannot take that route: before
materialization their length dim is symbolic, and Core ML states must have a
concrete shape.

``materialize_symbolic_shape_program`` fixes that — every ``{prefill,decode}_N``
function it emits has fully concrete shapes.  This pass runs right after it and
converts the leftover cache I/O into state:

* the input ``k_4`` becomes an fp16 ``state_tensor_placeholder`` of the same
  (now concrete) shape, keeping its name;
* a ``read_state`` inserted at the top of the function feeds everything that
  used to read the input — it is the first op in the block, so it dominates
  every use;
* the value that used to leave the function as ``k_4_out`` is written back with
  ``coreml_update_state`` at the end of the block, and ``k_4_out`` is dropped
  from the function outputs.

``sliding_pos_ring`` keeps its ordinary int32 I/O: Core ML states must be
floating point.  That falls out of the fp16 filter below rather than being
special-cased by name.

The ``fill_like`` + ``add`` wrapper
-----------------------------------
Handing the updated-cache var straight to ``coreml_update_state`` produces a
model that loads and runs but silently loses the state: on macOS 26 the runtime
turns ``read_state -> <in-place-looking update> -> write_state`` into a write
whose base is not the previous state contents, so every prediction sees a cache
holding only the row it just wrote.  (The same trap is why every cache write in
``decode_coreml`` is a whole-tensor select rather than
``jax.lax.dynamic_update_slice``; see ``tests/test_sliding_state_write.py`` and
``tests/test_global_cache_write.py``.)

Adding a ``fill_like``-produced zero tensor forces the written value into a
tensor of its own and the state then persists.  ``fill_like`` (rather than a
zero ``const``) keeps the addend out of reach of constant folding, and adding
zeros rather than multiplying by zero preserves any NaN/inf already in the
cache — the same trick ``stablehlo_coreml`` uses when a state write would
otherwise be a compile-time constant.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, Var, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

# An input ``X`` and an output ``X_out`` of identical fp16 type are a KV cache
# pair — the exporter names every cache that way (see ``export._kv_export_plan``).
OUTPUT_SUFFIX = "_out"


def _cache_io_pairs(func: Function) -> list[tuple[str, Var]]:
    """Return ``(input_name, output_var)`` for every fp16 ``X`` / ``X_out`` pair.

    Skips inputs that are already state, non-fp16 pairs (``sliding_pos_ring``
    is int32 and must stay I/O), and anything whose shape is still symbolic —
    a state cannot have a flexible shape, so an unmaterialized function is left
    alone rather than turned into a model that fails to load.
    """
    outputs_by_name: dict[str, Var] = {var.name: var for var in func.outputs}
    pairs: list[tuple[str, Var]] = []
    for name, var in func.inputs.items():
        if types.is_state(var.sym_type):
            continue
        out_var = outputs_by_name.get(name + OUTPUT_SUFFIX)
        if out_var is None:
            continue
        if var.dtype != types.fp16 or out_var.dtype != types.fp16:
            continue
        if any_symbolic(var.shape) or any_symbolic(out_var.shape):
            continue
        if out_var is var:
            raise ValueError(
                f"cache {name} is passed through unchanged ({out_var.name} is "
                "the input itself); there is no updated value to write back"
            )
        if tuple(var.shape) != tuple(out_var.shape):
            raise ValueError(
                f"cache pair {name}/{out_var.name} has mismatched shapes "
                f"{tuple(var.shape)} vs {tuple(out_var.shape)}"
            )
        pairs.append((name, out_var))
    return pairs


@block_context_manager
def _statify_function(func: Function, pairs: list[tuple[str, Var]]) -> None:
    """Rewrite one function's cache I/O into state, in place."""
    first_op = next(iter(func.operations), None)
    if first_op is None:
        raise ValueError("cannot convert caches to state in an empty function")

    converted_outputs = {out_var for _, out_var in pairs}
    remaining_outputs = [var for var in func.outputs if var not in converted_outputs]
    if not remaining_outputs:
        raise ValueError(
            "converting every cache to state would leave the function without "
            "outputs, which Core ML rejects"
        )

    for in_name, out_var in pairs:
        old_var = func.inputs[in_name]

        # 1. The input becomes an fp16 state feature of the same concrete shape,
        #    keeping its name so the runtime's cache layout is unchanged.
        placeholder = mb.state_tensor_placeholder(tuple(old_var.shape), dtype=types.fp16)
        placeholder.set_name(in_name)
        state_var = placeholder.outputs[0]
        func.placeholder_inputs[in_name] = placeholder
        func._input_dict[in_name] = state_var

        # 2. Read at function entry, ahead of every op, so the read dominates
        #    all the uses it takes over from the old input.
        read_var = mb.read_state(
            input=state_var, before_op=first_op, name=f"{in_name}_read_state",
        )
        func.replace_uses_of_var_after_op(
            anchor_op=None, old_var=old_var, new_var=read_var,
        )

        # 3. Write the fully-updated cache back at the end of the block.  The
        #    zero add is load-bearing — see the module docstring.
        zeros = mb.fill_like(
            ref_tensor=read_var, value=np.float16(0), name=f"{in_name}_state_zeros",
        )
        value = mb.add(x=zeros, y=out_var, name=f"{in_name}_state_value")
        mb.coreml_update_state(
            state=state_var, value=value, name=f"{in_name}_update_state",
        )

    func.set_outputs(remaining_outputs)


@register_pass(namespace="common")
class global_kv_caches_to_states(AbstractGraphPass):
    """Convert every concrete-shape fp16 ``X`` / ``X_out`` cache pair to state.

    Applied to a materialized program, this turns the 3 global KV caches of
    every ``{prefill,decode}_N`` function into 6 state features (``k_4``,
    ``v_4``, ``k_9``, ``v_9``, ``k_14``, ``v_14``) and removes the matching
    inputs and outputs.  Functions whose caches are still symbolic-shaped are
    left untouched.
    """

    def apply(self, prog: Program) -> None:
        converted: dict[str, list[str]] = {}
        for fname, func in prog.functions.items():
            pairs = _cache_io_pairs(func)
            if not pairs:
                continue
            _statify_function(func, pairs)
            converted[fname] = [name for name, _ in pairs]

        if not converted:
            return
        names = sorted({n for v in converted.values() for n in v})
        print(
            f"  global_kv_caches_to_states: {len(names)} caches → state "
            f"({', '.join(names)}) in {len(converted)} function(s)",
            flush=True,
        )
