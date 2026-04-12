"""MIL pass: collapse consecutive ``cast → cast`` chains.

RMSNorm in Gemma computes in float32, then casts the result back to
float16. When two norms are adjacent (e.g. post-attention norm followed
by pre-FFN norm), the graph contains ``cast(fp32→fp16) → cast(fp16→fp32)``
chains where the intermediate representation is never used by anything
else. Collapsing these avoids pointless precision round-trips.

Rules:
- ``cast(x, A) → cast(_, B)`` where B == x.dtype  →  remove both (identity)
- ``cast(x, A) → cast(_, B)`` otherwise            →  single ``cast(x, B)``
- Only applies when the first cast has a single consumer.
"""

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@block_context_manager
def _collapse_in_block(block):
    changed = False
    ops = list(block.operations)
    removed = set()
    for op in ops:
        for b in op.blocks:
            changed |= _collapse_in_block(b)

        if op in removed or op.op_type != "cast":
            continue

        # The input to this (outer) cast.
        x_var = op.inputs["x"]
        parent_op = x_var.op
        if parent_op is None or parent_op.op_type != "cast":
            continue

        # Only collapse when the inner cast has a single consumer.
        if len(x_var.child_ops) != 1:
            continue

        original_input = parent_op.inputs["x"]
        outer_dtype = op.outputs[0].dtype

        if original_input.dtype == outer_dtype:
            # Round-trip (e.g. fp16→fp32→fp16): replace with identity.
            block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0],
                new_var=original_input, no_check_var_types=True,
            )
            block.remove_ops([op, parent_op])
            removed.update([op, parent_op])
        else:
            # Different target dtype: collapse to single cast.
            new_cast = mb.cast(
                x=original_input, dtype=op.inputs["dtype"].val,
                before_op=parent_op, name=op.name,
            )
            block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0],
                new_var=new_cast, no_check_var_types=True,
            )
            block.remove_ops([op, parent_op])
            removed.update([op, parent_op])

        changed = True

    return changed


@register_pass(namespace="common")
class collapse_cast_chains(AbstractGraphPass):
    """Collapse ``cast → cast`` chains into a single cast or identity."""

    def apply(self, prog):
        for fname in prog.functions:
            func = prog.functions[fname]
            before = sum(1 for op in func.operations if op.op_type == "cast")
            _collapse_in_block(func)
            after = sum(1 for op in func.operations if op.op_type == "cast")
            eliminated = before - after
            if eliminated:
                print(f"  collapse_cast_chains [{fname}]: eliminated "
                      f"{eliminated} cast ops ({before} → {after})", flush=True)
