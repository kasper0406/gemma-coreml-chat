"""MIL pass: remove ``tile`` ops used only for broadcasting.

StableHLO requires explicit shape matching, so the lowering inserts
``tile(x, reps=[1,...,N,...,1])`` before binary ops.  MIL's element-wise
ops natively support NumPy-style broadcasting — these tiles are unnecessary.

Only tiles whose *every* consumer is a known-broadcast-capable op are removed.
``select`` is excluded: E5RT's multifunction validator fails to propagate
shapes through ``select`` with implicit broadcasting (it reports
"Incompatible Dimension" for the internal lowering).
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

import numpy as np

# Ops that support implicit NumPy-style broadcasting.
# ``select`` is intentionally excluded — E5RT cannot handle implicit
# broadcasting in ``select`` when the model is loaded as multifunction.
_BROADCAST_OPS = frozenset({
    "add", "sub", "mul", "real_div",
    "maximum", "minimum",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or",
    "pow", "floor_div", "mod",
})


def _is_broadcast_tile(op) -> bool:
    """True if ``tile`` replicates along any dimension(s) (non-1 reps)."""
    if op.op_type != "tile":
        return False
    reps = op.inputs["reps"].val
    if reps is None:
        return False
    reps = np.asarray(reps).flatten().tolist()
    return any(r != 1 for r in reps)


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

        if not _is_broadcast_tile(op):
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
