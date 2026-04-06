"""MIL pass: remove ``tile`` ops used only for broadcasting.

StableHLO requires explicit shape matching, so the lowering inserts
``tile(x, reps=[1,...,N,...,1])`` before binary ops.  MIL's element-wise
ops natively support NumPy-style broadcasting — these tiles are unnecessary.

Only tiles whose *every* consumer is a known-broadcast-capable op are removed.

Counts from multi-chunk.mlpackage:
  decode: 390, prefill: 354 → up to 744 tile ops eliminated.
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

import numpy as np

# Ops that support implicit NumPy-style broadcasting.
_BROADCAST_OPS = frozenset({
    "add", "sub", "mul", "real_div",
    "maximum", "minimum",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or",
    "pow", "floor_div", "mod",
    "select",  # condition broadcasting
})


def _is_single_dim_tile(op) -> bool:
    """True if ``tile`` only replicates along one dimension (rest are 1)."""
    if op.op_type != "tile":
        return False
    reps = op.inputs["reps"].val
    if reps is None:
        return False
    reps = np.asarray(reps).flatten().tolist()
    return sum(1 for r in reps if r != 1) == 1


def _all_consumers_broadcast(op) -> bool:
    """True if every consumer of *op*'s output supports broadcasting."""
    out = op.outputs[0]
    for child_op in out.child_ops:
        if child_op.op_type not in _BROADCAST_OPS:
            return False
    return True


@block_context_manager
def _remove_in_block(block):
    changed = False
    ops = list(block.operations)
    for op in ops:
        for b in op.blocks:
            changed |= _remove_in_block(b)

        if not _is_single_dim_tile(op):
            continue
        if not _all_consumers_broadcast(op):
            continue

        # Replace tile output with its input — consumers will broadcast.
        tile_input = op.inputs["x"]

        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=tile_input,
            no_check_var_types=True,
        )
        block.remove_ops([op])
        changed = True

    return changed


@register_pass(namespace="common")
class remove_broadcast_tiles(AbstractGraphPass):
    """Remove ``tile`` ops that only broadcast for element-wise consumers."""

    def apply(self, prog):
        for fname in prog.functions:
            _remove_in_block(prog.functions[fname])
