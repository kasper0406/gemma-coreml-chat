"""MIL pass: fuse decomposed attention into ``scaled_dot_product_attention``.

After ``replace_decomposed_softmax`` runs (and with cleanup passes having
removed noop slice_updates, broadcast tiles, and redundant maximums), each
attention layer appears as a 3D batch-folded matmul pattern::

    Q_raw [B,H,L,E]   → reshape → Q_3d [BH,L,E]
    K_raw [H,B,S,E]   → reshape → K_3d [HB,S,E]
    matmul(Q_3d, K_3d, transpose_y=True) → scores_3d [BH,L,S]
      → reshape → transpose → scores_4d [B,H,L,S]
    select(mask, scores_4d, -10000)
    [cast(fp32)]
    softmax(axis=-1) → weights_4d [B,H,L,S]
      → reshape → weights_3d [BH,L,S]
    V_raw [H,B,EV,S]  → reshape → V_3d [HB,EV,S]
    matmul(weights_3d, V_3d, transpose_y=True) → out_3d [BH,L,EV]
      → reshape → transpose → out_4d [B,H,L,EV]

The batch-fold transposes before K and V reshapes have varying permutations
depending on whether GQA repeat (``jnp.repeat``) precedes the layout
transpose.  This pass peels back only through the reshape to get the raw
4D tensors, then inserts explicit transposes to reach SDPA's required
``[B, H, ?, ?]`` layout.

This pass replaces the entire chain with a single SDPA op at rank 4::

    Q_4d  = transpose(Q_raw) if needed     # [B,H,L,E]
    K_4d  = transpose(K_raw, [1,0,2,3])    # [B,H,S,E]
    V_4d  = transpose(V_raw, [1,0,3,2])    # [B,H,S,EV]
    Q_prescaled = Q_4d * sqrt(E)
    out_4d = sdpa(Q_prescaled, K_4d, V_4d, attn_mask=mask)

The ``sqrt(E)`` pre-scaling cancels SDPA's built-in ``1/sqrt(E)`` divisor,
since Gemma4 uses ``scale=1.0`` (QK-norm absorbs the scaling).
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

_MASK_FILL_THRESHOLD = -1000.0


def _is_large_negative(var) -> bool:
    """True if var is a scalar/uniform constant ≤ -1000."""
    val = var.val
    if val is not None:
        arr = np.asarray(val)
        if arr.size == 0:
            return False
        first = float(arr.flat[0])
        if arr.size == 1 or np.all(arr == first):
            return first <= _MASK_FILL_THRESHOLD
        return False
    if var.op is not None and var.op.op_type == "fill":
        v = var.op.inputs.get("value")
        if v is not None and v.val is not None:
            return float(v.val) <= _MASK_FILL_THRESHOLD
    return False


def _peel_back(var, *op_types):
    """Trace backward through a sequence of ops, returning the ultimate input.

    Returns ``(original_input, [op1, op2, ...])`` or ``(var, [])`` if
    the chain doesn't match.  Each op must have its output consumed by the
    next op in the chain (or be the starting var).
    """
    ops = []
    for ot in op_types:
        if var.op is None or var.op.op_type != ot:
            return var, ops
        ops.append(var.op)
        var = var.op.inputs["x"]
    return var, ops


def _peel_fwd_single(var, op_type):
    """If *var* has exactly one consumer of type *op_type*, return its output."""
    consumers = list(var.child_ops)
    if len(consumers) == 1 and consumers[0].op_type == op_type:
        return consumers[0].outputs[0], consumers[0]
    return None, None


def _match_sdpa(softmax_op):
    """Match attention pattern anchored at a softmax op.

    Returns ``(Q_4d, K_4d, V_4d, mask_var, E, out_4d, anchor_op, dead_ops)``
    or ``None``.
    """
    if softmax_op.op_type != "softmax":
        return None

    # ── backward from softmax: optional cast, optional select ────────
    pre = softmax_op.inputs["x"]
    cast_op = None
    if pre.op is not None and pre.op.op_type == "cast":
        cast_op = pre.op
        pre = cast_op.inputs["x"]

    select_op = None
    mask_var = None
    scores_4d = pre
    if pre.op is not None and pre.op.op_type == "select":
        sel = pre.op
        if _is_large_negative(sel.inputs["b"]):
            mask_var = sel.inputs["cond"]
            scores_4d = sel.inputs["a"]
            select_op = sel

    # ── scores_4d ← transpose ← reshape ← matmul_0 ─────────────────
    scores_src, scores_unwrap = _peel_back(scores_4d, "transpose", "reshape")
    if scores_src.op is None or scores_src.op.op_type != "matmul":
        # Try without transpose (scores might be directly from reshape)
        scores_src, scores_unwrap = _peel_back(scores_4d, "reshape")
        if scores_src.op is None or scores_src.op.op_type != "matmul":
            return None

    matmul_0 = scores_src.op
    Q_3d = matmul_0.inputs["x"]
    K_3d = matmul_0.inputs["y"]

    # ── Q_raw ← reshape ← Q_3d ──────────────────────────────────────
    Q_raw, q_prep = _peel_back(Q_3d, "reshape")
    if Q_raw.rank is None or Q_raw.rank < 4:
        return None

    # ── K_raw ← reshape ← K_3d (peel reshape only, not transpose) ──
    K_raw, k_prep = _peel_back(K_3d, "reshape")
    if K_raw.rank is None or K_raw.rank < 4:
        return None

    # ── forward from softmax: reshape → matmul_1 ────────────────────
    sm_out = softmax_op.outputs[0]
    weights_3d, weights_reshape = _peel_fwd_single(sm_out, "reshape")
    if weights_3d is not None:
        # weights_3d → matmul_1
        w_consumers = list(weights_3d.child_ops)
        if len(w_consumers) != 1 or w_consumers[0].op_type != "matmul":
            return None
        matmul_1 = w_consumers[0]
    else:
        # Direct: softmax → matmul_1
        consumers = list(sm_out.child_ops)
        if len(consumers) != 1 or consumers[0].op_type != "matmul":
            return None
        matmul_1 = consumers[0]
        weights_reshape = None

    # ── V_raw ← reshape ← V_3d (peel reshape only, not transpose) ──
    V_3d = matmul_1.inputs["y"]
    V_raw, v_prep = _peel_back(V_3d, "reshape")
    if V_raw.rank is None or V_raw.rank < 4:
        return None

    # ── forward from matmul_1: reshape → transpose → out_4d ─────────
    out_3d = matmul_1.outputs[0]
    out_reshaped, out_reshape_op = _peel_fwd_single(out_3d, "reshape")
    if out_reshaped is None:
        return None
    out_4d, out_transpose_op = _peel_fwd_single(out_reshaped, "transpose")
    if out_4d is None:
        return None

    # ── validate shapes ──────────────────────────────────────────────
    if Q_raw.rank is not None and Q_raw.rank < 3:
        return None
    E = Q_raw.shape[-1] if Q_raw.shape is not None else None
    if E is None or not isinstance(E, int):
        return None

    # K embedding dim must match Q
    K_E = K_raw.shape[-1] if K_raw.shape is not None else None
    if K_E is not None and isinstance(K_E, int) and K_E != E:
        return None

    # ── collect dead ops ─────────────────────────────────────────────
    dead = {matmul_0, matmul_1, softmax_op, out_reshape_op, out_transpose_op}
    for op in scores_unwrap:
        dead.add(op)
    if select_op:
        dead.add(select_op)
    if cast_op:
        dead.add(cast_op)
    if weights_reshape:
        dead.add(weights_reshape)

    # Prep ops only if all their consumers are dead
    for prep_op in list(q_prep) + list(k_prep) + list(v_prep):
        if all(c in dead for o in prep_op.outputs for c in o.child_ops):
            dead.add(prep_op)

    # Fill value const for select
    if select_op:
        fill_var = select_op.inputs["b"]
        if fill_var.op is not None:
            if all(c in dead for o in fill_var.op.outputs for c in o.child_ops):
                dead.add(fill_var.op)

    # Safety: no dead op (except the final output) should have external consumers
    for op in list(dead):
        if op is out_transpose_op:
            continue
        for out in op.outputs:
            for consumer in out.child_ops:
                if consumer not in dead:
                    return None

    return Q_raw, K_raw, V_raw, mask_var, E, out_4d, out_transpose_op, dead


@block_context_manager
def _replace_in_block(block):
    changed = False
    ops = list(block.operations)
    removed = set()
    for op in ops:
        for b in op.blocks:
            changed |= _replace_in_block(b)

        if op in removed or op.op_type != "softmax":
            continue

        result = _match_sdpa(op)
        if result is None:
            continue
        Q_raw, K_raw, V_raw, mask_var, E, out_4d, anchor_op, dead_ops = result

        # ── Fix Q/K/V layouts for SDPA [B, H, ?, ?] ─────────────
        # out_4d is always [B, H, L, EV] — use shape[0] as B reference.
        B_ref = out_4d.shape[0] if out_4d.shape is not None else 1

        # Q_raw: [B, H, L, E] (typical) or [H, B, L, E] (rare)
        if Q_raw.shape[0] != B_ref:
            Q_4d = mb.transpose(
                x=Q_raw, perm=[1, 0, 2, 3],
                before_op=anchor_op,
                name=anchor_op.name + "_q_layout",
            )
        else:
            Q_4d = Q_raw

        # K_raw: [H, B, S, E] from batch-fold reshape
        if K_raw.shape[0] != B_ref:
            K_4d = mb.transpose(
                x=K_raw, perm=[1, 0, 2, 3],
                before_op=anchor_op,
                name=anchor_op.name + "_k_layout",
            )
        else:
            K_4d = K_raw

        # V_raw: [..., EV, S] — always needs S↔EV swap, plus batch swap
        if V_raw.shape[0] != B_ref:
            V_4d = mb.transpose(
                x=V_raw, perm=[1, 0, 3, 2],
                before_op=anchor_op,
                name=anchor_op.name + "_v_layout",
            )
        else:
            V_4d = mb.transpose(
                x=V_raw, perm=[0, 1, 3, 2],
                before_op=anchor_op,
                name=anchor_op.name + "_v_layout",
            )

        from coremltools.converters.mil.mil import types as mil_types
        q_dtype = Q_4d.dtype
        scale_val = np.sqrt(float(E))
        scale_np = np.float16(scale_val) if q_dtype == mil_types.fp16 else np.float32(scale_val)

        q_scaled = mb.mul(
            x=Q_4d, y=scale_np,
            before_op=anchor_op,
            name=anchor_op.name + "_q_prescale",
        )

        sdpa_out = mb.scaled_dot_product_attention(
            query=q_scaled, key=K_4d, value=V_4d,
            attn_mask=mask_var,
            before_op=anchor_op,
            name=anchor_op.name + "_sdpa",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=anchor_op,
            old_var=out_4d,
            new_var=sdpa_out,
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
