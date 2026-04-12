"""MIL pass: collapse consecutive reshape/squeeze→reshape chains.

StableHLO→MIL lowering often produces ``reshape→reshape`` or
``squeeze→reshape`` chains where the intermediate shape is never used
elsewhere.  A single ``reshape`` to the final target shape is equivalent.

Counts from multi-chunk.mlpackage:
  decode:  39 reshape→reshape, 102 squeeze→reshape
  prefill: 69 reshape→reshape,   1 squeeze→reshape
  Total: ~211 redundant ops eliminated.
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil import Builder as mb


@block_context_manager
def _collapse_in_block(block):
    changed = False
    ops = list(block.operations)
    for op in ops:
        for b in op.blocks:
            changed |= _collapse_in_block(b)

        if op.op_type != "reshape":
            continue

        # The input to this reshape.
        x_var = op.inputs["x"]
        parent_op = x_var.op
        if parent_op is None:
            continue

        # Match reshape→reshape or squeeze→reshape.
        if parent_op.op_type not in ("reshape", "squeeze"):
            continue

        # The original input (before the parent reshape/squeeze).
        original_input = parent_op.inputs["x"]

        # Extract target shape as a Python list so the builder creates a fresh
        # const at the insertion point.  Passing the Var directly would
        # reference a const defined *after* parent_op, causing a "not visible
        # in the block" validation error.
        shape_var = op.inputs["shape"]
        if shape_var.val is None:
            continue  # dynamic shape — skip
        target_shape = shape_var.val.tolist()

        single_consumer = len(x_var.child_ops) == 1

        # Build replacement reshape from original input to final shape.
        new_var = mb.reshape(
            x=original_input,
            shape=target_shape,
            before_op=parent_op,
            name=op.name,
        )

        block.replace_uses_of_var_after_op(
            anchor_op=parent_op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )
        if single_consumer:
            block.remove_ops([op, parent_op])
        else:
            block.remove_ops([op])
        changed = True

    return changed


@register_pass(namespace="common")
class collapse_reshape_chains(AbstractGraphPass):
    """Merge ``reshape→reshape`` and ``squeeze→reshape`` into single reshape."""

    def apply(self, prog):
        for fname in prog.functions:
            _collapse_in_block(prog.functions[fname])
