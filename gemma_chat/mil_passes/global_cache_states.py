"""MIL pass: turn the global KV cache input/output pairs into Core ML state.

Why this is a *post*-materialization pass
-----------------------------------------
The sliding KV caches are bound to Core ML state during StableHLO→MIL
conversion (``stablehlo_coreml``'s ``StateSpec``), because their shape is
static from the start.  The 3 global caches cannot take that route: before
materialization their length dim is symbolic, and Core ML states must have a
concrete shape.

``materialize_symbolic_shape_program`` fixes that — every per-size function
it emits has fully concrete shapes.  This pass runs right after it and
converts the leftover cache I/O into state:

* the input ``k_4`` becomes an fp16 ``state_tensor_placeholder`` of the same
  (now concrete) shape, keeping its name;
* a ``read_state`` inserted at the top of the function feeds everything that
  used to read the input — it is the first op in the block, so it dominates
  every use;
* the value that used to leave the function as ``k_4_out`` is written back with
  ``coreml_update_state`` right where it is produced, every later reader of it
  reads the write's result instead, and ``k_4_out`` is dropped from the
  function outputs.

Why the write is not a sink
---------------------------
Appending the writes at the end of the block, where nothing reads their
result, looks simpler — and on macOS 27 makes ANECompiler throw ("Exception
thrown: <private>") whenever such a "sink" write lands in an ANE segment, which
fails the whole model load with the misleading "``functionName`` must be nil
unless the model type is ML Program" error.  The sliding caches never had the
problem: the converter already writes them where they are produced and reads
the attention's keys and values back from the write.  This pass does the same.

It is not free on the GPU: with the full model, decode steps of the 1024 and
2048 functions are ~2% slower than with the writes at the end of the block
(MPSGraph; measured with interleaved predict loops, macOS 27, M4 Pro), while
the 512 one is unaffected.

A written value that only reaches the output through an ``identity`` alias
(``updated -> attention``, ``identity(updated) -> k_4_out``) is written where
``updated`` is produced, not after the alias — after it, nothing would read the
write.  And every write the pass inserts must end up with a reader: the pass
fails the export rather than emit a sink.

Read-only caches
----------------
A layer-chunk function (see ``decode_coreml.layer_chunks``) may *read* a cache
another chunk writes — the KV-shared layers read layer 13's and 14's.  Such a
function takes the cache as an input with no ``_out`` partner; the pass turns
it into a state that is only read (``read_state``, no write).  Core ML shares
states across the functions of one package by name, so the reading chunk sees
what the writing chunk stored.

Only inputs named like a cache (``k_<slot>`` / ``v_<slot>``) are touched, so a
chunk's ``hidden`` / ``hidden_out`` pair stays ordinary I/O.

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

import re

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, Var, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

# The exporter names every KV cache ``k_<slot>`` / ``v_<slot>``, and the value
# a function writes back ``k_<slot>_out`` (see ``export._chunk_io_plan``).
CACHE_NAME = re.compile(r"[kv]_\d+")
OUTPUT_SUFFIX = "_out"


def _cache_inputs(func: Function) -> list[tuple[str, Var | None]]:
    """Return ``(input_name, output_var or None)`` for every fp16 cache input.

    ``output_var`` is the ``<name>_out`` output the cache is written back
    from, or None for a cache the function only reads.  Skips inputs that are
    already state and anything whose shape is still symbolic — a state cannot
    have a flexible shape, so an unmaterialized function is left alone rather
    than turned into a model that fails to load.
    """
    outputs_by_name: dict[str, Var] = {var.name: var for var in func.outputs}
    caches: list[tuple[str, Var | None]] = []
    for name, var in func.inputs.items():
        if not CACHE_NAME.fullmatch(name) or types.is_state(var.sym_type):
            continue
        if var.dtype != types.fp16 or any_symbolic(var.shape):
            continue
        out_var = outputs_by_name.get(name + OUTPUT_SUFFIX)
        if out_var is not None:
            if any_symbolic(out_var.shape):
                continue
            if _unaliased(out_var) is var:
                raise ValueError(
                    f"cache {name} is passed through unchanged ({out_var.name} is "
                    "the input itself); there is no updated value to write back"
                )
            if out_var.dtype != types.fp16 or tuple(var.shape) != tuple(out_var.shape):
                raise ValueError(
                    f"cache pair {name}/{out_var.name} has mismatched types "
                    f"{var.sym_type} vs {out_var.sym_type}"
                )
        caches.append((name, out_var))
    return caches


def _unaliased(var: Var) -> Var:
    """``var`` with any chain of ``identity`` ops in front of it stripped."""
    while var.op is not None and var.op.op_type == "identity":
        var = var.op.x
    return var


def _has_live_reader(var: Var, block_outputs: set[Var]) -> bool:
    """Whether anything that survives dead-code elimination reads ``var``.

    An ``identity`` whose own result nobody reads does not count: it is what
    is left of an output alias once the output is dropped.
    """
    for op in var.child_ops:
        if op.op_type != "identity":
            return True
        out = op.outputs[0]
        if out in block_outputs or _has_live_reader(out, block_outputs):
            return True
    return False


@block_context_manager
def _statify_function(func: Function, caches: list[tuple[str, Var | None]]) -> None:
    """Rewrite one function's cache I/O into state, in place."""
    first_op = next(iter(func.operations), None)
    if first_op is None:
        raise ValueError("cannot convert caches to state in an empty function")

    converted_outputs = {out_var for _, out_var in caches if out_var is not None}
    remaining_outputs = [var for var in func.outputs if var not in converted_outputs]
    if not remaining_outputs:
        raise ValueError(
            "converting every cache to state would leave the function without "
            "outputs, which Core ML rejects"
        )

    # Drop the cache outputs before rerouting their readers below: rerouting a
    # block output renames the write's result to ``k_4_out``, and Core ML
    # rejects the program ("Block redefines I/O name").
    func.set_outputs(remaining_outputs)

    for in_name, out_var in caches:
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
        if out_var is None:
            continue  # read-only: no write

        # 3. Write the fully-updated cache back right after it is produced —
        #    behind any output alias — and make every later reader consume the
        #    write's result, so the write is never a sink (see the module
        #    docstring).  The zero add is load-bearing — see there too.
        updated = _unaliased(out_var)
        ops = list(func.operations)
        producer = ops.index(updated.op)
        after = ops[producer + 1] if producer + 1 < len(ops) else None
        zeros = mb.fill_like(
            ref_tensor=read_var, value=np.float16(0),
            name=f"{in_name}_state_zeros", before_op=after,
        )
        value = mb.add(x=zeros, y=updated, name=f"{in_name}_state_value", before_op=after)
        written = mb.coreml_update_state(
            state=state_var, value=value, name=f"{in_name}_update_state", before_op=after,
        )
        func.replace_uses_of_var_after_op(
            anchor_op=written.op, old_var=updated, new_var=written,
        )
        if not _has_live_reader(written, set(func.outputs)):
            raise ValueError(
                f"the write-back of cache {in_name} has no reader; a "
                "coreml_update_state nothing consumes makes ANECompiler fail "
                "the whole model (see the module docstring)"
            )


@register_pass(namespace="common")
class global_kv_caches_to_states(AbstractGraphPass):
    """Convert every concrete-shape fp16 cache input (``k_<slot>`` /
    ``v_<slot>``) to state — written back from its ``_out`` output if it has
    one, read-only otherwise.

    Applied to a materialized program, this turns the global KV caches of
    every per-size function into state features (``k_4``, ``v_4``, ``k_9``,
    ``v_9``, ``k_14``, ``v_14``) and removes the matching inputs and outputs.
    Functions whose caches are still symbolic-shaped are left untouched.
    """

    def apply(self, prog: Program) -> None:
        converted: dict[str, list[str]] = {}
        for fname, func in prog.functions.items():
            caches = _cache_inputs(func)
            if not caches:
                continue
            _statify_function(func, caches)
            converted[fname] = [name for name, _ in caches]

        if not converted:
            return
        names = sorted({n for v in converted.values() for n in v})
        print(
            f"  global_kv_caches_to_states: {len(names)} caches → state "
            f"({', '.join(names)}) in {len(converted)} function(s)",
            flush=True,
        )
