"""MIL pass: fuse decomposed attention into ``scaled_dot_product_attention``.

After ``replace_decomposed_softmax`` runs (and with cleanup passes having
removed noop slice_updates, broadcast tiles, and redundant maximums), each
attention layer appears in one of two forms:

**Concrete shapes** (batch-folded, typical for sliding attention)::

    Q_raw [B,H,L,E]   → reshape → Q_3d [BH,L,E]
    K_raw [H,B,S,E]   → reshape → K_3d [HB,S,E]
    matmul(Q_3d, K_3d, transpose_y=True) → scores_3d [BH,L,S]
      → reshape → transpose → scores_4d [B,H,L,S]
    ...softmax...
    V_raw [H,B,EV,S]  → reshape → V_3d [HB,EV,S]
    matmul(weights_3d, V_3d, transpose_y=True) → out_3d [BH,L,EV]
      → reshape → transpose → out_4d [B,H,L,EV]

**Symbolic shapes** (B=1, typical for global attention with dynamic cache)::

    Q_raw [1,H,L,E]   → squeeze → Q_3d [H,L,E]
    K_raw [1,1,S,E]   → squeeze → K_3d [1,S,E]
    matmul(Q_3d, K_3d, transpose_y=True) → scores_3d [H,L,S]
      → expand_dims → transpose → scores_4d [1,H,L,S]
    ...softmax...
    V_raw [1,1,EV,S]  → squeeze → V_3d [1,EV,S]
    matmul(weights_3d, V_3d, transpose_y=True) → out_3d [H,L,EV]
      → expand_dims → transpose → out_4d [1,H,L,EV]

This pass handles two attention scaling styles:

1. **Standard** (Llama, Mistral): explicit ``real_div(scores, sqrt(E))`` or
   ``mul(scores, 1/sqrt(E))`` between the scores matmul and softmax.  The
   pass absorbs this scaling — SDPA divides by ``sqrt(E)`` natively.

2. **Gemma-style** (scale=1.0, QK-norm): no explicit scaling in the graph.
   Q is pre-multiplied by ``sqrt(E)`` to cancel SDPA's built-in divisor.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

_MASK_FILL_THRESHOLD = -1000.0


def _get_scalar_val(var):
    """Extract a Python float from a MIL variable if it's a compile-time scalar
    or a uniform-valued constant (all elements identical)."""
    if var.val is None:
        return None
    arr = np.asarray(var.val).flatten()
    if arr.size == 0:
        return None
    first = float(arr[0])
    if arr.size == 1 or np.all(arr == first):
        return first
    return None


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

    Returns ``(Q_4d, K_4d, V_4d, mask_var, E, out_4d, anchor_op, dead_ops,
    has_explicit_scaling)`` or ``None``.

    ``has_explicit_scaling`` is True if the graph contains an explicit
    ``real_div(scores, sqrt(E))`` or ``mul(scores, 1/sqrt(E))`` between the
    scores matmul and softmax.  When True, SDPA handles scaling natively and
    Q should NOT be pre-scaled.  When False (Gemma-style scale=1.0), Q must
    be pre-scaled by ``sqrt(E)`` to cancel SDPA's built-in divisor.
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

    # ── check for explicit attention scaling ──────────────────────────
    # Standard transformers: real_div(scores, sqrt(E)) or mul(scores, 1/sqrt(E))
    # between scores_4d and the rest of the backward chain.
    has_explicit_scaling = False
    scale_op = None
    if scores_4d.op is not None and scores_4d.op.op_type == "real_div":
        scale_op = scores_4d.op
        scores_4d = scale_op.inputs["x"]
        has_explicit_scaling = True
    elif scores_4d.op is not None and scores_4d.op.op_type == "mul":
        # mul(scores, 1/sqrt(E)) — one operand is a small scalar constant
        candidate = scores_4d.op
        sx = _get_scalar_val(candidate.inputs["x"])
        sy = _get_scalar_val(candidate.inputs["y"])
        if sx is not None and 0 < abs(sx) < 1:
            scale_op = candidate
            scores_4d = candidate.inputs["y"]
            has_explicit_scaling = True
        elif sy is not None and 0 < abs(sy) < 1:
            scale_op = candidate
            scores_4d = candidate.inputs["x"]
            has_explicit_scaling = True

    # ── scores_4d ← transpose ← reshape/expand_dims ← matmul_0 ────
    # Concrete shapes: matmul [BH,L,S] → reshape [B,H,L,S] → transpose
    # Symbolic shapes (B=1): matmul [H,L,S] → expand_dims [H,L,1,S] → transpose
    scores_src, scores_unwrap = _peel_back(scores_4d, "transpose", "reshape")
    if scores_src.op is None or scores_src.op.op_type != "matmul":
        scores_src, scores_unwrap = _peel_back(scores_4d, "transpose", "expand_dims")
        if scores_src.op is None or scores_src.op.op_type != "matmul":
            scores_src, scores_unwrap = _peel_back(scores_4d, "reshape")
            if scores_src.op is None or scores_src.op.op_type != "matmul":
                if scores_4d.op is not None and scores_4d.op.op_type == "matmul":
                    scores_src = scores_4d
                    scores_unwrap = []
                else:
                    return None

    matmul_0 = scores_src.op
    Q_3d = matmul_0.inputs["x"]
    K_3d = matmul_0.inputs["y"]

    # ── Q_raw ← reshape/squeeze ← Q_3d ─────────────────────────────
    Q_raw, q_prep = _peel_back(Q_3d, "reshape")
    if Q_raw.rank is None or Q_raw.rank < 4:
        Q_raw, q_prep = _peel_back(Q_3d, "squeeze")
        if Q_raw.rank is None or Q_raw.rank < 4:
            if Q_3d.rank is not None and Q_3d.rank >= 4:
                Q_raw = Q_3d
                q_prep = []
            else:
                return None

    # ── K_raw ← reshape/squeeze ← K_3d ──────────────────────────────
    K_raw, k_prep = _peel_back(K_3d, "reshape")
    if K_raw.rank is None or K_raw.rank < 4:
        K_raw, k_prep = _peel_back(K_3d, "squeeze")
        if K_raw.rank is None or K_raw.rank < 4:
            if K_3d.rank is not None and K_3d.rank >= 4:
                K_raw = K_3d
                k_prep = []
            else:
                return None

    # ── forward from softmax: reshape/squeeze → matmul_1 ─────────────
    sm_out = softmax_op.outputs[0]
    weights_3d, weights_reshape = _peel_fwd_single(sm_out, "reshape")
    if weights_3d is None:
        weights_3d, weights_reshape = _peel_fwd_single(sm_out, "squeeze")
    if weights_3d is not None:
        w_consumers = list(weights_3d.child_ops)
        if len(w_consumers) != 1 or w_consumers[0].op_type != "matmul":
            return None
        matmul_1 = w_consumers[0]
    else:
        consumers = list(sm_out.child_ops)
        if len(consumers) != 1 or consumers[0].op_type != "matmul":
            return None
        matmul_1 = consumers[0]
        weights_reshape = None

    # ── V_raw ← reshape/squeeze ← V_3d ─────────────────────────────
    V_3d = matmul_1.inputs["y"]
    V_raw, v_prep = _peel_back(V_3d, "reshape")
    if V_raw.rank is None or V_raw.rank < 4:
        V_raw, v_prep = _peel_back(V_3d, "squeeze")
        if V_raw.rank is None or V_raw.rank < 4:
            if V_3d.rank is not None and V_3d.rank >= 4:
                V_raw = V_3d
                v_prep = []
            else:
                return None

    # ── forward from matmul_1: reshape/expand_dims → transpose → out_4d
    out_3d = matmul_1.outputs[0]
    out_reshaped, out_reshape_op = _peel_fwd_single(out_3d, "reshape")
    if out_reshaped is None:
        out_reshaped, out_reshape_op = _peel_fwd_single(out_3d, "expand_dims")
    if out_reshaped is not None:
        out_4d, out_transpose_op = _peel_fwd_single(out_reshaped, "transpose")
        if out_4d is None:
            return None
    else:
        if out_3d.rank is not None and out_3d.rank >= 4:
            out_4d, out_transpose_op = _peel_fwd_single(out_3d, "transpose")
            if out_4d is not None:
                out_reshape_op = None
            else:
                return None
        else:
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

    # ── compute compensating output layout perm ──────────────────────
    # The 3D matmul output is [merged_batch, L, EV].  Reshape inserts B
    # (size B_ref) at some position, giving 4D [d0, d1, d2, EV].  The
    # transpose then reorders to the expected output layout.
    #
    # SDPA always outputs [B, H, L, EV].  If the original output layout
    # differs (e.g. [B, L, H, EV] when the JAX code transposes H↔L after
    # attention), we need a compensating transpose after SDPA.

    if out_reshape_op is not None:
        B_ref_val = out_4d.shape[0] if out_4d.shape is not None else 1
        reshape_4d = list(out_reshaped.shape)[:3]
        out_perm = [int(x) for x in np.asarray(out_transpose_op.inputs["perm"].val)]

        # Find B's position in the 4D reshaped tensor (prefer rightmost).
        b_pos = None
        for i in range(2, -1, -1):
            if reshape_4d[i] == B_ref_val:
                b_pos = i
                break
        if b_pos is None:
            return None

        # Build P: perm mapping SDPA [B,H,L,EV] positions → reshaped positions.
        non_b = [i for i in range(3) if i != b_pos]
        P = [0, 0, 0, 3]
        P[b_pos] = 0       # B (SDPA pos 0) → b_pos in reshaped
        P[non_b[0]] = 1    # H (SDPA pos 1) → first non-B slot
        P[non_b[1]] = 2    # L (SDPA pos 2) → second non-B slot

        # Compensating perm: compose P with the output transpose.
        out_layout_perm = [P[out_perm[i]] for i in range(4)]
    else:
        # No reshape: matmul output is already 4D, just have transpose.
        # The transpose directly maps from matmul output to final layout.
        # SDPA outputs [B,H,L,EV]. The matmul outputs [B,H,L,EV] or similar,
        # and the transpose reorders it. The compensating perm is just the
        # transpose perm itself since there's no reshape to account for.
        out_perm = [int(x) for x in np.asarray(out_transpose_op.inputs["perm"].val)]
        out_layout_perm = out_perm

    # ── collect dead ops ─────────────────────────────────────────────
    dead = {matmul_0, matmul_1, softmax_op, out_transpose_op}
    if out_reshape_op is not None:
        dead.add(out_reshape_op)
    for op in scores_unwrap:
        dead.add(op)
    if select_op:
        dead.add(select_op)
    if cast_op:
        dead.add(cast_op)
    if weights_reshape:
        dead.add(weights_reshape)
    if scale_op:
        dead.add(scale_op)

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

    return (Q_raw, K_raw, V_raw, mask_var, E, out_4d, out_transpose_op,
            dead, has_explicit_scaling, out_layout_perm)


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
        (Q_raw, K_raw, V_raw, mask_var, E, out_4d, anchor_op,
         dead_ops, has_explicit_scaling, out_layout_perm) = result

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

        if has_explicit_scaling:
            # Standard attention: graph already has real_div/mul by sqrt(E),
            # which is absorbed into the fusion. SDPA divides by sqrt(E) natively.
            q_input = Q_4d
        else:
            # Gemma-style: scale=1.0 (QK-norm absorbs scaling). Pre-multiply
            # Q by sqrt(E) to cancel SDPA's built-in 1/sqrt(E) divisor.
            from coremltools.converters.mil.mil import types as mil_types
            q_dtype = Q_4d.dtype
            scale_val = np.sqrt(float(E))
            scale_np = np.float16(scale_val) if q_dtype == mil_types.fp16 else np.float32(scale_val)
            q_input = mb.mul(
                x=Q_4d, y=scale_np,
                before_op=anchor_op,
                name=anchor_op.name + "_q_prescale",
            )

        sdpa_out = mb.scaled_dot_product_attention(
            query=q_input, key=K_4d, value=V_4d,
            attn_mask=mask_var,
            before_op=anchor_op,
            name=anchor_op.name + "_sdpa",
        )

        # Compensate for output layout mismatch.
        # SDPA outputs [B, H, L, EV] but the original graph may expect a
        # different layout (e.g. [B, L, H, EV] for GQA).  Apply the
        # compensating transpose if it's not the identity perm.
        if out_layout_perm != [0, 1, 2, 3]:
            sdpa_out = mb.transpose(
                x=sdpa_out, perm=out_layout_perm,
                before_op=anchor_op,
                name=anchor_op.name + "_sdpa_layout",
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
