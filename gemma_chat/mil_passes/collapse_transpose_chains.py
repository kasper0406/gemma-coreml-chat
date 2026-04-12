"""MIL pass: collapse consecutive transpose chains.

StableHLO→MIL lowering often produces ``transpose→transpose`` chains
where the two permutations compose to a simpler single permutation —
or even to the identity (no transpose needed at all).

When the intermediate result has other consumers, the parent transpose
is kept and only the child is replaced (with composed perm or bypassed).

Counts from gemma4-e2b.mlpackage (decode):
  13 single-consumer identity  → remove both
  13 single-consumer compose   → collapse to one
  22 multi-consumer identity   → remove child only
  22 multi-consumer compose    → replace child with composed
  Total: ~70 transpose ops eliminated.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@block_context_manager
def _collapse_in_block(block):
    changed = False
    ops = list(block.operations)
    for op in ops:
        for b in op.blocks:
            changed |= _collapse_in_block(b)

        if op.op_type != "transpose":
            continue

        x_var = op.inputs["x"]
        parent_op = x_var.op
        if parent_op is None or parent_op.op_type != "transpose":
            continue

        p1 = parent_op.inputs["perm"].val
        p2 = op.inputs["perm"].val
        if p1 is None or p2 is None:
            continue

        combined = np.array([p1[p2[i]] for i in range(len(p1))])
        identity = np.arange(len(combined))
        original_input = parent_op.inputs["x"]
        single_consumer = len(x_var.child_ops) == 1

        if np.array_equal(combined, identity):
            # Composed permutation is identity — child produces same as
            # parent's original input.
            block.replace_uses_of_var_after_op(
                anchor_op=parent_op,
                old_var=op.outputs[0],
                new_var=original_input,
                no_check_var_types=True,
            )
            if single_consumer:
                block.remove_ops([op, parent_op])
            else:
                block.remove_ops([op])
        else:
            # Single composed transpose from original input.
            new_var = mb.transpose(
                x=original_input,
                perm=combined.tolist(),
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
class collapse_transpose_chains(AbstractGraphPass):
    """Merge ``transpose→transpose`` into single transpose or identity."""

    def apply(self, prog):
        for fname in prog.functions:
            before = sum(1 for op in prog.functions[fname].operations
                         if op.op_type == "transpose")
            _collapse_in_block(prog.functions[fname])
            after = sum(1 for op in prog.functions[fname].operations
                        if op.op_type == "transpose")
            eliminated = before - after
            if eliminated:
                print(f"  collapse_transpose_chains [{fname}]: "
                      f"eliminated {eliminated} transpose ops "
                      f"({before} → {after})", flush=True)
