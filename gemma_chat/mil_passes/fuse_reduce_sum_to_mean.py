"""MIL pass: replace ``reduce_sum`` ÷ N patterns with ``reduce_mean``.

RMSNorm computes ``mean(x²)`` which JAX/XLA lowers in two variants:

1. ``reduce_sum → reshape → mul(1/N)``  →  ``reduce_mean → reshape``
2. ``reduce_sum → add(C) → mul(1/N)``   →  ``reduce_mean → add(C/N)``

Variant 2 arises when XLA restructures ``mean(x²) + ε`` into
``(sum(x²) + ε·N) / N`` to combine the epsilon addition with the
reduction.  Both variants are handled by this pass.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _get_uniform_scalar(var):
    """Return the float value of a uniform const (all elements equal), or None."""
    if var.val is None:
        return None
    arr = np.asarray(var.val)
    if arr.size == 0:
        return None
    first = float(arr.flat[0])
    if arr.size == 1 or np.allclose(arr, first, rtol=1e-5):
        return first
    return None


@block_context_manager
def _fuse_in_block(block):
    changed = False
    ops = list(block.operations)
    removed = set()
    for op in ops:
        for b in op.blocks:
            changed |= _fuse_in_block(b)

        if op in removed or op.op_type != "reduce_sum":
            continue

        rs_out = op.outputs[0]
        axes_var = op.inputs.get("axes")
        if axes_var is None or axes_var.val is None:
            continue
        axes = axes_var.val.tolist()

        # Compute N = product of reduced dimension sizes.
        rs_input = op.inputs["x"]
        input_shape = rs_input.shape
        if input_shape is None:
            continue
        N = 1
        for a in axes:
            dim = input_shape[a]
            if not isinstance(dim, int):
                N = None
                break
            N *= dim
        if N is None or N <= 0:
            continue

        if len(rs_out.child_ops) != 1:
            continue  # reduce_sum result used by multiple ops

        keep_dims_var = op.inputs.get("keep_dims")
        keep_dims = bool(keep_dims_var.val) if keep_dims_var and keep_dims_var.val is not None else False

        consumer = list(rs_out.child_ops)[0]

        # ── Variant 1: reduce_sum → reshape → mul(1/N) ──────────────
        if consumer.op_type == "reshape":
            reshape_op = consumer
            reshape_out = reshape_op.outputs[0]

            mul_ops = [c for c in reshape_out.child_ops if c.op_type == "mul"]
            if len(mul_ops) != 1 or len(reshape_out.child_ops) > 1:
                continue
            mul_op = mul_ops[0]

            scalar_val = None
            for side in ("x", "y"):
                v = mul_op.inputs[side]
                if v is not reshape_out:
                    scalar_val = _get_uniform_scalar(v)
            if scalar_val is None:
                continue
            expected = 1.0 / N
            if abs(scalar_val - expected) / max(abs(expected), 1e-12) > 1e-3:
                continue

            shape_var = reshape_op.inputs["shape"]
            if shape_var.val is None:
                continue

            mean_var = mb.reduce_mean(
                x=rs_input, axes=axes, keep_dims=keep_dims,
                before_op=op, name=op.name + "_mean",
            )
            new_reshape = mb.reshape(
                x=mean_var, shape=shape_var.val.tolist(),
                before_op=op, name=reshape_op.name,
            )

            block.replace_uses_of_var_after_op(
                anchor_op=mul_op, old_var=mul_op.outputs[0],
                new_var=new_reshape, no_check_var_types=True,
            )
            block.remove_ops([mul_op, reshape_op, op])
            removed.update([mul_op, reshape_op, op])
            changed = True

        # ── Variant 2: reduce_sum → add(C) → mul(1/N) ──────────────
        # XLA restructures mean(x²)+ε into (sum(x²)+ε·N)/N.
        elif consumer.op_type == "add":
            add_op = consumer
            add_out = add_op.outputs[0]

            mul_ops = [c for c in add_out.child_ops if c.op_type == "mul"]
            if len(mul_ops) != 1 or len(add_out.child_ops) > 1:
                continue
            mul_op = mul_ops[0]

            # Verify the mul constant ≈ 1/N.
            scalar_val = None
            for side in ("x", "y"):
                v = mul_op.inputs[side]
                if v is not add_out:
                    scalar_val = _get_uniform_scalar(v)
            if scalar_val is None:
                continue
            expected = 1.0 / N
            if abs(scalar_val - expected) / max(abs(expected), 1e-12) > 1e-3:
                continue

            # Get the add constant (the other operand, not the reduce_sum output).
            add_const_var = None
            for side in ("x", "y"):
                v = add_op.inputs[side]
                if v is not rs_out:
                    add_const_var = v
            if add_const_var is None or add_const_var.val is None:
                continue

            # Adjusted constant: C * (1/N) — restores original epsilon.
            adjusted_val = np.asarray(add_const_var.val, dtype=np.float32) * scalar_val

            mean_var = mb.reduce_mean(
                x=rs_input, axes=axes, keep_dims=keep_dims,
                before_op=op, name=op.name + "_mean",
            )
            new_add = mb.add(
                x=mean_var, y=adjusted_val,
                before_op=op, name=add_op.name,
            )

            block.replace_uses_of_var_after_op(
                anchor_op=mul_op, old_var=mul_op.outputs[0],
                new_var=new_add, no_check_var_types=True,
            )
            block.remove_ops([mul_op, add_op, op])
            removed.update([mul_op, add_op, op])
            changed = True

    return changed


@register_pass(namespace="common")
class fuse_reduce_sum_to_mean(AbstractGraphPass):
    """Replace ``reduce_sum`` ÷ N patterns with ``reduce_mean``."""

    def apply(self, prog):
        for fname in prog.functions:
            func = prog.functions[fname]
            before = sum(1 for op in func.operations
                         if op.op_type not in ("const", "constexpr_blockwise_shift_scale"))
            _fuse_in_block(func)
            after = sum(1 for op in func.operations
                        if op.op_type not in ("const", "constexpr_blockwise_shift_scale"))
            eliminated = before - after
            if eliminated:
                rm_count = sum(1 for op in func.operations
                               if op.op_type == "reduce_mean")
                print(f"  fuse_reduce_sum_to_mean [{fname}]: {rm_count} fused, "
                      f"eliminated {eliminated} non-const ops", flush=True)
