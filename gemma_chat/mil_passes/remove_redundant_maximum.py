"""MIL pass: remove redundant ``maximum(-inf, ...)`` in softmax patterns.

The StableHLO→MIL lowering of softmax produces::

    %max = reduce_max(x)
    %clamped = maximum(%max, -inf)        # ensures max ≥ -inf (always true)
    %also_clamped = maximum(-inf, %clamped)  # redundant

The second ``maximum`` is always a no-op because its other operand is
already ≥ −∞.  This pass removes the redundant outer ``maximum``.

Counts: 35 per function × 2 functions = 70 ops eliminated.
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

import numpy as np

# Threshold for detecting -inf constants (covers both -inf and -FLT_MAX).
_NEG_INF_THRESHOLD = -3.0e+38


def _is_neg_inf_const(var) -> bool:
    """True if *var* is a scalar or uniform constant ≤ -3e38."""
    if var.op is None:
        return False
    op = var.op
    if op.op_type not in ("const", "fill"):
        return False
    val = var.val
    if val is None:
        return False
    arr = np.asarray(val)
    if arr.size == 0:
        return False
    return bool(np.all(arr <= _NEG_INF_THRESHOLD))


@block_context_manager
def _remove_in_block(block):
    changed = False
    ops = list(block.operations)
    for op in ops:
        for b in op.blocks:
            changed |= _remove_in_block(b)

        if op.op_type != "maximum":
            continue

        x_var = op.inputs["x"]
        y_var = op.inputs["y"]

        # One operand must be -inf, the other must be the "real" value.
        if _is_neg_inf_const(x_var):
            real_var = y_var
        elif _is_neg_inf_const(y_var):
            real_var = x_var
        else:
            continue

        # Check if the real operand is itself the output of a maximum with
        # -inf — i.e. the value is already clamped.
        parent = real_var.op
        if parent is None or parent.op_type != "maximum":
            continue
        px = parent.inputs["x"]
        py = parent.inputs["y"]
        if not (_is_neg_inf_const(px) or _is_neg_inf_const(py)):
            continue

        # The outer maximum is redundant — replace with the inner result.
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=real_var,
            no_check_var_types=True,
        )
        block.remove_ops([op])
        changed = True

    return changed


@register_pass(namespace="common")
class remove_redundant_maximum(AbstractGraphPass):
    """Remove ``maximum(-inf, maximum(-inf, x))`` — the outer max is a no-op."""

    def apply(self, prog):
        for fname in prog.functions:
            _remove_in_block(prog.functions[fname])
