"""MIL pass: replace the global-cache-length input ``N`` with a constant.

Why the input is still there after materialization
--------------------------------------------------
``N`` is the dimension-variable argument JAX adds for the symbolic global cache
length (see ``export._kv_export_plan``).  It is a *value*, not a shape, so
``materialize_symbolic_shape_program`` — which only rewrites shapes — leaves it
as a runtime ``int32[1]`` input in every ``{prefill,decode}_N`` function, even
though every one of those functions has exactly one possible value for it.

The cost is not the input itself but the ops hanging off it.  The global
attention mask is built as ``range_1d(end=N) <= position``, broadcast to
``[1, 8, 1, N]`` against a ``fill(shape=[1, 8, 1, N])`` of ``-10000``.  With
``N`` unknown at compile time all of those stay symbolic, and
``fuse_attention_to_sdpa`` bails on symbolic dimensions — so the global
attention sites (7 of the 35 layers in the full model) run as an unfused
fp32 ``matmul → select(mask) → softmax → matmul`` with 8× GQA tiles and the
transposes that go with them, instead of one ``scaled_dot_product_attention``.

What this pass does
-------------------
For each named function it is given a length for:

1. inserts ``const([length])`` at the top of the block and rewires every use of
   the ``N`` input to it;
2. drops ``N`` from the function signature (the Swift runtime already treats the
   feature as optional — see ``CoreMLModel.classifyIO``);
3. re-runs type inference over everything downstream of ``N``, in program order,
   so the newly-constant shapes propagate.  Only the affected ops are touched:
   re-inferring the whole function would run value inference over the
   ``constexpr_*`` weights and decompress them.

``fuse_attention_to_sdpa`` must be re-run afterwards to collect the winnings;
``materialize.py`` does that right after calling this pass.

The value-inference size cap
----------------------------
Re-inference computes values as well as types, and ``fill``'s value inference
materializes the whole tensor — the ``-10000`` mask of a 65536-long prefill
function is 268 MB of fp32.  Nothing downstream needs those values (the mask
depends on the runtime ``position``, and ``fuse_attention_to_sdpa`` reads the
fill's scalar operand, not its expansion), so any inferred value above
:data:`_MAX_INFERRED_VALUE` elements is dropped again.  Small ones are kept:
shape vectors have to stay constant for the ops they feed to infer a concrete
shape at all, which is the entire point of the pass.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

# The name ``export.py`` gives JAX's dimension-variable argument.
LENGTH_INPUT_NAME = "N"

# Keep inferred values up to this many elements; see the module docstring.
_MAX_INFERRED_VALUE = 8192


def _downstream_ops(var: Var) -> set[int]:
    """Ids of every op reachable from ``var`` by following uses."""
    seen: set[int] = set()
    frontier = [var]
    while frontier:
        for op in frontier.pop().child_ops:
            if id(op) in seen:
                continue
            seen.add(id(op))
            frontier.extend(op.outputs)
    return seen


def _reinfer(func: Function, affected: set[int]) -> None:
    """Re-run type/value inference over ``affected``, in program order."""
    for op in func.operations:
        if id(op) not in affected:
            continue
        op.type_value_inference(overwrite_output=True)
        for out in op.outputs:
            val = out.val
            if val is not None and np.size(val) > _MAX_INFERRED_VALUE:
                out._sym_val = None


@block_context_manager
def _concretize_function(func: Function, length: int) -> None:
    """Replace the ``N`` input of one function with ``const([length])``."""
    old_var = func.inputs[LENGTH_INPUT_NAME]
    first_op = next(iter(func.operations), None)
    if first_op is None:
        raise ValueError("cannot concretize the cache length of an empty function")

    const_var = mb.const(
        val=np.full(old_var.shape, length, dtype=np.int32),
        before_op=first_op,
        name=f"{LENGTH_INPUT_NAME}_const",
    )
    # The child ops change shape, not just value, so the built-in type check
    # would reject the swap; `_reinfer` below re-establishes the types instead.
    func.replace_uses_of_var_after_op(
        anchor_op=None, old_var=old_var, new_var=const_var, no_check_var_types=True,
    )
    if old_var.child_ops:
        raise ValueError(
            f"input {LENGTH_INPUT_NAME!r} still has "
            f"{len(old_var.child_ops)} consumer(s) after replacement"
        )

    del func._input_dict[LENGTH_INPUT_NAME]
    del func.placeholder_inputs[LENGTH_INPUT_NAME]

    _reinfer(func, _downstream_ops(const_var))


@register_pass(namespace="common")
class concretize_cache_length(AbstractGraphPass):
    """Turn the ``N`` cache-length input of each materialized function into a const.

    Supported options:

    - ``function_name_to_length``: ``Dict[str, int]``
        The concrete global cache length of each function to rewrite.  Functions
        that are absent, or that have no ``N`` input, are left alone.
    """

    def __init__(self) -> None:
        self._function_name_to_length: dict[str, int] = {}

    @property
    def function_name_to_length(self) -> dict[str, int]:
        return self._function_name_to_length

    @function_name_to_length.setter
    def function_name_to_length(self, value: dict[str, int]) -> None:
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, int) for k, v in value.items()
        ):
            raise ValueError(
                "function_name_to_length must be a dict of function name → int, "
                f"got {value!r}"
            )
        self._function_name_to_length = value

    def apply(self, prog: Program) -> None:
        rewritten: list[str] = []
        for fname, length in self.function_name_to_length.items():
            func = prog.functions.get(fname)
            if func is None or LENGTH_INPUT_NAME not in func.inputs:
                continue
            _concretize_function(func, length)
            rewritten.append(fname)

        if rewritten:
            print(
                f"  concretize_cache_length: {LENGTH_INPUT_NAME} → const in "
                f"{len(rewritten)} function(s)",
                flush=True,
            )
