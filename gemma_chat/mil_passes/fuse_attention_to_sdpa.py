"""MIL pass: fuse decomposed attention into ``scaled_dot_product_attention``.

After ``replace_decomposed_softmax`` runs, each attention layer appears as::

    K_t = transpose(K, perm=..., -2, -1)         # swap last two dims
    scores = matmul(Q, K_t)                       # [B, H, L, S]
    masked = select(bool_mask, scores, -10000.0)  # causal/sliding mask
    [casted = cast(masked, fp32)]                  # optional fp32 cast
    weights = softmax(casted_or_masked, axis=-1)   # attention weights
    out = matmul(weights, V)                       # [B, H, L, Hd]

CoreML iOS18 has a native ``scaled_dot_product_attention`` op that fuses
the entire chain into a single hardware-accelerated instruction::

    out = sdpa(Q_scaled, K, V, attn_mask)

The op always applies ``1/sqrt(E)`` scaling.  Gemma4 uses ``scale=1.0``
(QK-norm absorbs the scaling), so Q is pre-multiplied by ``sqrt(E)``
to cancel the built-in divisor:  ``(Q·√E) @ K^T / √E  =  Q @ K^T``.

Counts from gemma4-e2b: 35 attention layers per function × 2 functions
= 70 SDPA fusions expected.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

# Mask fill values in the range (-inf, -1000] are treated as "masked out".
_MASK_FILL_THRESHOLD = -1000.0


def _get_scalar_value(var):
    """Return the uniform scalar value of a const/fill var, or None."""
    val = var.val
    if val is not None:
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        first = float(arr.flat[0])
        if arr.size == 1 or np.all(arr == first):
            return first
        return None
    if var.op is not None and var.op.op_type == "fill":
        v = var.op.inputs.get("value")
        if v is not None and v.val is not None:
            return float(v.val)
    return None


def _is_large_negative(var) -> bool:
    """True if var is a scalar/uniform constant ≤ -1000."""
    sv = _get_scalar_value(var)
    return sv is not None and sv <= _MASK_FILL_THRESHOLD


def _peel_transpose(var):
    """If *var* is produced by a transpose that swaps the last two dims,
    return the un-transposed input.  Otherwise return None."""
    if var.op is None or var.op.op_type != "transpose":
        return None
    perm_var = var.op.inputs.get("perm")
    if perm_var is None or perm_var.val is None:
        return None
    perm = list(perm_var.val)
    rank = len(perm)
    if rank < 2:
        return None
    # Check that only the last two dims are swapped.
    expected = list(range(rank))
    expected[-2], expected[-1] = expected[-1], expected[-2]
    if perm != expected:
        return None
    return var.op.inputs["x"]


def _match_attention(matmul2_op):
    """Match a full attention pattern ending at *matmul2_op*.

    Expected chain (working backward from matmul2_op)::

        matmul2(weights, V)                    ← anchor
          weights = softmax(·, axis=-1)
            [· = cast(·, fp32)]                ← optional
              · = select(mask, scores, -BIG)   ← optional mask
                scores = matmul1(Q, K_t)
                  K_t = transpose(K, swap_last_2)

    Returns ``(Q_var, K_var, V_var, mask_var_or_None, all_dead_ops)`` or
    ``None`` if the pattern does not match.
    """
    if matmul2_op.op_type != "matmul":
        return None

    weights_var = matmul2_op.inputs["x"]
    V_var = matmul2_op.inputs["y"]

    # weights must come from softmax
    if weights_var.op is None or weights_var.op.op_type != "softmax":
        return None
    softmax_op = weights_var.op

    # Verify softmax axis is -1 (or last dim)
    axis_var = softmax_op.inputs.get("axis")
    if axis_var is not None and axis_var.val is not None:
        axis = int(axis_var.val)
        sm_input = softmax_op.inputs["x"]
        if sm_input.rank is not None:
            if axis < 0:
                axis += sm_input.rank
            if axis != sm_input.rank - 1:
                return None
        elif axis != -1:
            return None

    # Trace backward through optional cast(fp32)
    pre_softmax = softmax_op.inputs["x"]
    cast_op = None
    if pre_softmax.op is not None and pre_softmax.op.op_type == "cast":
        cast_op = pre_softmax.op
        pre_softmax = cast_op.inputs["x"]

    # Look for select(mask, scores, large_negative) — the masking step
    mask_var = None
    select_op = None
    scores_var = pre_softmax

    if pre_softmax.op is not None and pre_softmax.op.op_type == "select":
        sel = pre_softmax.op
        # select(cond=mask, a=scores, b=fill_value)
        if _is_large_negative(sel.inputs["b"]):
            mask_var = sel.inputs["cond"]
            scores_var = sel.inputs["a"]
            select_op = sel
        # Also handle swapped: select(cond=mask, a=fill_value, b=scores)
        # with inverted mask semantics — unlikely but defensive.
        elif _is_large_negative(sel.inputs["a"]):
            # Inverted: cond=True means masked-out. We'd need to negate.
            # Skip for now — our JAX code always uses (cond, scores, fill).
            pass

    # scores must come from matmul1(Q, K_transposed)
    if scores_var.op is None or scores_var.op.op_type != "matmul":
        return None
    matmul1_op = scores_var.op
    Q_var = matmul1_op.inputs["x"]
    K_t_var = matmul1_op.inputs["y"]

    # Check if matmul1 uses transpose_y=True
    transpose_y = matmul1_op.inputs.get("transpose_y")
    if transpose_y is not None and transpose_y.val is not None and transpose_y.val:
        # K is passed directly with transpose_y=True
        K_var = K_t_var
    else:
        # K_t must be produced by a transpose that swaps last two dims
        K_var = _peel_transpose(K_t_var)
        if K_var is None:
            return None

    # Validate shapes: Q, K, V must be rank >= 3
    for v in (Q_var, K_var, V_var):
        if v.rank is not None and v.rank < 3:
            return None

    # Validate: Q and K must share embedding dim (last dim)
    q_shape = Q_var.shape
    k_shape = K_var.shape
    if (q_shape is not None and k_shape is not None
            and q_shape[-1] is not None and k_shape[-1] is not None
            and q_shape[-1] != k_shape[-1]):
        return None

    # Collect all ops that will be dead after fusion
    dead_ops = {matmul2_op, softmax_op, matmul1_op}
    if cast_op is not None:
        dead_ops.add(cast_op)
    if select_op is not None:
        dead_ops.add(select_op)
    # The K transpose op (if it exists and has no other consumers)
    if K_t_var.op is not None and K_t_var.op.op_type == "transpose":
        k_transpose_op = K_t_var.op
        all_consumers_dead = all(
            child in dead_ops
            for out in k_transpose_op.outputs
            for child in out.child_ops
        )
        if all_consumers_dead:
            dead_ops.add(k_transpose_op)
    # The fill/const for the mask fill value
    if select_op is not None:
        fill_var = select_op.inputs["b"]
        if fill_var.op is not None:
            all_consumers_dead = all(
                child in dead_ops
                for out in fill_var.op.outputs
                for child in out.child_ops
            )
            if all_consumers_dead:
                dead_ops.add(fill_var.op)

    # Verify that intermediate vars have no consumers outside the dead set.
    # If they do, we can't safely remove them.
    for dead_op in list(dead_ops):
        if dead_op is matmul2_op:
            continue  # output of matmul2 will be replaced
        for out_var in dead_op.outputs:
            for consumer in out_var.child_ops:
                if consumer not in dead_ops:
                    # This op's output is used elsewhere — can't fuse.
                    return None

    return Q_var, K_var, V_var, mask_var, dead_ops


@block_context_manager
def _replace_in_block(block):
    changed = False
    ops = list(block.operations)
    removed = set()
    for op in ops:
        for b in op.blocks:
            changed |= _replace_in_block(b)

        if op in removed:
            continue

        result = _match_attention(op)
        if result is None:
            continue
        Q_var, K_var, V_var, mask_var, dead_ops = result

        # Determine embedding dim for pre-scaling Q
        E = Q_var.shape[-1] if Q_var.shape is not None else None
        if E is None or isinstance(E, type(Q_var.shape[-1])) and not isinstance(E, int):
            # Symbolic or unknown — skip fusion
            continue

        # Pre-scale Q by sqrt(E) to cancel SDPA's built-in 1/sqrt(E).
        from coremltools.converters.mil.mil import types as mil_types
        q_dtype = Q_var.dtype
        scale_val = np.sqrt(float(E))
        if q_dtype == mil_types.fp16:
            scale_np = np.float16(scale_val)
        else:
            scale_np = np.float32(scale_val)

        q_scaled = mb.mul(
            x=Q_var,
            y=scale_np,
            before_op=op,
            name=op.name + "_q_prescale",
        )

        sdpa_var = mb.scaled_dot_product_attention(
            query=q_scaled,
            key=K_var,
            value=V_var,
            attn_mask=mask_var,
            before_op=op,
            name=op.name + "_sdpa",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=sdpa_var,
            no_check_var_types=True,
        )

        block.remove_ops(list(dead_ops))
        removed.update(dead_ops)
        changed = True

    return changed


@register_pass(namespace="common")
class fuse_attention_to_sdpa(AbstractGraphPass):
    """Fuse matmul→[mask]→softmax→matmul into ``scaled_dot_product_attention``."""

    def apply(self, prog):
        for fname in prog.functions:
            before = sum(1 for op in prog.functions[fname].operations
                         if op.op_type != "const")
            _replace_in_block(prog.functions[fname])
            after = sum(1 for op in prog.functions[fname].operations
                        if op.op_type != "const")
            sdpa_count = sum(1 for op in prog.functions[fname].operations
                             if op.op_type == "scaled_dot_product_attention")
            if sdpa_count:
                print(f"  fuse_attention_to_sdpa [{fname}]: {sdpa_count} SDPA ops, "
                      f"eliminated {before - after} decomposed ops", flush=True)
