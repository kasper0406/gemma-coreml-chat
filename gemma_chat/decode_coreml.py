"""JAX-traceable chunk_prefill and decode_step for CoreML export.

Both functions close over ``params`` and are designed to be lowered with
``jax.jit(...).lower(shape_specs).compiler_ir('stablehlo')`` and converted
to CoreML .mlpackage files via the same pipeline as export.py.

KV cache layout
---------------
Only 15 of the 35 layers store their own KV (layers 15-34 are KV-shared).

- **Sliding layers** (12 caches): ring-buffer shape ``(1, sliding_window_size, nkv, hd)``.
  Slot index = ``position % sliding_window_size``.  A companion
  ``sliding_pos_ring`` array ``(1, sliding_window_size)`` int32 tracks
  which absolute position each slot holds (``-1`` = empty).
- **Global layers** (3 caches): linear shape ``(1, max_seq_len, nkv, hd)``.
  Slot index = absolute position.

Flat KV representation: ``[k0, v0, k1, v1, ..., k14, v14]`` — 30 arrays.
Layer order: 0, 1, 2, 3 (LOCAL_SLIDING), 4 (GLOBAL), 5-8, 9, 10-13, 14.

Tokens: right-padded — real tokens at positions 0..T-1, zeros at T..L-1.
"""

from __future__ import annotations

from typing import List, Tuple

import jax
import jax.numpy as jnp

from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, MAX_SEQ_LEN
from gemma_chat.model import AttentionType, Gemma4Config, _apply_rope, _embed_lookup
from gemma_chat.cache_spec import build_cache_specs, kv_shared_sources


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def kv_non_shared_layers(cfg: Gemma4Config) -> List[int]:
    """Layer indices that own a KV cache slot (0 .. kv_shared_start-1)."""
    return list(range(cfg.num_layers - cfg.num_kv_shared_layers))


def kv_cache_shapes(cfg: Gemma4Config, max_seq_len: int) -> List[Tuple]:
    """One shape per non-shared layer; k and v share the same shape.

    Sliding layers use ``sliding_window_size`` (ring buffer);
    global layers use ``max_seq_len``.
    """
    specs = build_cache_specs(cfg, max_seq_len)
    return [(1, s.cache_len, s.num_kv_heads, s.head_dim) for s in specs]


def empty_kv_cache(
    cfg: Gemma4Config = E2B_CONFIG,
    max_seq_len: int = MAX_SEQ_LEN,
    dtype=jnp.float16,
) -> List[jnp.ndarray]:
    """Return 30 zero JAX arrays: [k0, v0, k1, v1, ..., k14, v14]."""
    flat = []
    for shape in kv_cache_shapes(cfg, max_seq_len):
        flat.append(jnp.zeros(shape, dtype=dtype))
        flat.append(jnp.zeros(shape, dtype=dtype))
    return flat


def empty_pos_ring(cfg: Gemma4Config = E2B_CONFIG) -> jnp.ndarray:
    """Return (1, sliding_window_size) int32 filled with -1 (no entries)."""
    return jnp.full((1, cfg.sliding_window_size), -1, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _rmsnorm(x, scale):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6) * scale.astype(jnp.float32)


def _rmsnorm_noscale(x):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6)


def _ple_for_tokens(params, token_ids, cfg: Gemma4Config):
    """Per-layer input embeddings for token_ids (B, L).

    Returns (B, L, num_layers * per_layer_input_dim).
    """
    B, L = token_ids.shape
    d = cfg.per_layer_input_dim

    ple_table = params['embed_tokens_per_layer']         # (vocab, NL*d)
    ple_embed = _embed_lookup(ple_table, token_ids)      # (B, L, NL*d)
    ple_embed = ple_embed * jnp.sqrt(float(d)).astype(ple_embed.dtype)

    embed_table = params['embed_tokens']                 # (vocab, D)
    x0 = _embed_lookup(embed_table, token_ids)           # (B, L, D)
    x0 = x0 * jnp.sqrt(float(cfg.embed_dim)).astype(x0.dtype)

    W_proj = params['per_layer_model_projection']['kernel']  # (D, NL*d)
    ple_proj = jnp.dot(x0, W_proj) * (cfg.embed_dim ** -0.5)  # (B, L, NL*d)

    # 3D RMSNorm trick (avoids CoreML 4D batch_norm fusion bug)
    NL = B * cfg.num_layers
    scale = params['per_layer_projection_norm']['scale']   # (d,)
    ple_proj_3d = ple_proj.reshape(NL, L, d)
    ple_proj_3d = _rmsnorm(ple_proj_3d, scale)
    ple_proj = ple_proj_3d.reshape(B, L, cfg.num_layers * d)

    return (ple_proj + ple_embed) * (2.0 ** -0.5)


def _ffn(lp, x, hidden_dim: int):
    """GeLU-gated FFN. x: (..., D) → (..., D)."""
    gate = jnp.dot(x, lp['mlp']['gate_proj']['kernel'])
    up   = jnp.dot(x, lp['mlp']['up_proj']['kernel'])
    return jnp.dot(jax.nn.gelu(gate, approximate=False) * up,
                   lp['mlp']['down_proj']['kernel'])


def _ple_gate(lp, x, ple_slice):
    """Per-layer input gate block. x: (..., D), ple_slice: (..., d) → (..., D)."""
    gate = jax.nn.gelu(
        jnp.dot(x, lp['per_layer_input_gate']['kernel']),
        approximate=False,
    ) * ple_slice
    proj = jnp.dot(gate, lp['per_layer_projection']['kernel'])
    proj = _rmsnorm(proj, lp['post_per_layer_input_norm']['scale'])
    return x + proj



# ---------------------------------------------------------------------------
# Single-token decode attention (for decode_step)
# ---------------------------------------------------------------------------

def _attn_decode(lp, x, position, cfg: Gemma4Config, attn_type: str,
                 k_cache, v_cache, shared_kv=None, pos_ring=None):
    """Single-token attention with KV cache read/write.

    x: (1, 1, D)
    position: () int32 traced — absolute position of this new token
    k_cache, v_cache: (1, cache_len, nkv, hd)
        cache_len = sliding_window_size for sliding, max_seq_len for global.
    shared_kv: optional (k_cache, v_cache) from source layer; if given,
               this layer reads from source and does NOT update its own cache.
    pos_ring: (1, sliding_window_size) int32 — absolute position stored
              at each ring-buffer slot.  Required for LOCAL_SLIDING layers.

    Returns (attn_out (1,1,D), k_cache_updated, v_cache_updated).
    For KV-shared layers the returned caches are the source caches (unchanged).
    """
    is_global = attn_type == AttentionType.GLOBAL
    is_sliding = attn_type == AttentionType.LOCAL_SLIDING
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    hd = cfg.effective_head_dim(attn_type)
    rope_frac = cfg.rope_fraction_global if is_global else cfg.rope_fraction_sliding
    base_freq = cfg.global_rope_base_frequency if is_global else cfg.rope_base_frequency
    max_len = k_cache.shape[1]
    sa = lp['self_attn']

    pos_arr = position[jnp.newaxis, jnp.newaxis]  # (1, 1) for RoPE

    q = jnp.dot(x[0, 0], sa['q_proj']['kernel']).reshape(1, 1, num_heads, hd)
    q = _rmsnorm(q, sa['q_norm']['scale'])
    q = _apply_rope(q, pos_arr, base_freq, rope_frac)

    if shared_kv is not None:
        # KV-shared: read source cache directly, no write
        k_full, v_full = shared_kv
        k_updated, v_updated = shared_kv
    else:
        # Compute new K/V
        k_new = jnp.dot(x[0, 0], sa['k_proj']['kernel']).reshape(1, 1, num_kv_heads, hd)
        v_new = jnp.dot(x[0, 0], sa['v_proj']['kernel']).reshape(1, 1, num_kv_heads, hd)
        k_new = _rmsnorm(k_new, sa['k_norm']['scale'])
        v_new = _rmsnorm_noscale(v_new)
        k_new = _apply_rope(k_new, pos_arr, base_freq, rope_frac)

        # Write slot: ring-buffer modular index for sliding, absolute for global.
        write_idx = (position % cfg.sliding_window_size) if is_sliding else position
        k_new_f16 = k_new.astype(jnp.float16)
        v_new_f16 = v_new.astype(jnp.float16)
        k_updated = jax.lax.dynamic_update_slice(k_cache, k_new_f16, (0, write_idx, 0, 0))
        v_updated = jax.lax.dynamic_update_slice(v_cache, v_new_f16, (0, write_idx, 0, 0))
        k_full, v_full = k_updated, v_updated

    # Attention validity mask
    if is_sliding:
        # Ring-buffer mask: use pos_ring to find valid entries.
        pr = pos_ring[0]  # (W,)
        valid = (pr >= 0) & (pr <= position)
    else:
        # Global linear mask: slot index == absolute position.
        valid = jnp.arange(max_len, dtype=jnp.int32) <= position

    kv_rep = num_heads // num_kv_heads
    if kv_rep > 1:
        k_full = jnp.repeat(k_full, kv_rep, axis=2)
        v_full = jnp.repeat(v_full, kv_rep, axis=2)

    qt = jnp.transpose(q, (0, 2, 1, 3))                    # (1, H, 1, hd)
    kt = jnp.transpose(k_full, (0, 2, 1, 3))               # (1, H, max_len, hd)
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))           # (1, H, 1, max_len)
    w = jnp.where(valid[jnp.newaxis, jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w.astype(jnp.float32), axis=-1)

    out = jnp.matmul(w, vt)                                 # (1, H, 1, hd)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(1, 1, num_heads * hd)
    out = jnp.dot(out[0, 0], sa['o_proj']['kernel'])[None, None, :]
    return out, k_updated, v_updated



# ---------------------------------------------------------------------------
# Decode step: single-token forward with KV cache read/write
# ---------------------------------------------------------------------------

def decode_step(
    params,
    token_id,
    position,
    kv_flat,
    sliding_pos_ring,
    cfg: Gemma4Config = E2B_CONFIG,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Single-token autoregressive decode with KV cache.

    Args:
        params:    Flax param tree from load_params().
        token_id:  () int32 — the new token to process.
        position:  () int32 — absolute position of this token (= T + step).
        kv_flat:   List of 30 cache arrays (per-layer shapes, float16).
        sliding_pos_ring: (1, sliding_window_size) int32 — ring position tracker.
        cfg:       Model config.
        max_seq_len: Must match shape of global cache arrays.

    Returns:
        logits: (vocab_size,)
        kv_flat_new: Updated list of 30 cache arrays.
        sliding_pos_ring_new: Updated (1, sliding_window_size) int32.
    """
    token_arr = token_id[jnp.newaxis, jnp.newaxis]  # (1, 1)

    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, token_arr) * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)

    ple_all = _ple_for_tokens(params, token_arr, cfg)  # (1, 1, NL*d)

    # Update sliding_pos_ring for this position (shared by all sliding layers).
    W = cfg.sliding_window_size
    ring_slot = position % W
    ring_mask = (jnp.arange(W, dtype=jnp.int32) == ring_slot)[None, :]
    sliding_pos_ring = jnp.where(ring_mask, position, sliding_pos_ring)

    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    shared_sources = kv_shared_sources(cfg)

    # Unpack flat KV into per-layer dicts
    kv_own = {}   # layer_idx → (k_cache, v_cache) — mutable during this step
    for slot, layer_idx in enumerate(range(kv_shared_start)):
        kv_own[layer_idx] = (kv_flat[slot * 2], kv_flat[slot * 2 + 1])

    attn_types = list(cfg.attention_types)

    for i, attn_type in enumerate(attn_types):
        lp = params[f'layers.{i}']
        is_shared = i >= kv_shared_start

        d = cfg.per_layer_input_dim
        ple_slice = ple_all[:, :, i * d:(i + 1) * d]  # (1, 1, d)

        # Attention sub-layer
        residual = x
        x_ln = _rmsnorm(x, lp['input_layernorm']['scale'])

        if is_shared:
            src = shared_sources[i]
            k_src, v_src = kv_own[src]
            pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
            attn_out, _, _ = _attn_decode(lp, x_ln, position, cfg, attn_type,
                                           k_src, v_src, shared_kv=(k_src, v_src),
                                           pos_ring=pr)
        else:
            k_old, v_old = kv_own[i]
            pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
            attn_out, k_new, v_new = _attn_decode(lp, x_ln, position, cfg, attn_type,
                                                    k_old, v_old, pos_ring=pr)
            kv_own[i] = (k_new, v_new)

        attn_out = _rmsnorm(attn_out, lp['post_attention_layernorm']['scale'])
        x = residual + attn_out

        # FFN sub-layer
        residual = x
        x_ln = _rmsnorm(x, lp['pre_feedforward_layernorm']['scale'])
        ffn_out = _ffn(lp, x_ln, cfg.effective_hidden_dim(i))
        ffn_out = _rmsnorm(ffn_out, lp['post_feedforward_layernorm']['scale'])
        x = residual + ffn_out

        # PLE gate
        x = _ple_gate(lp, x[0, 0], ple_slice[0, 0])[None, None, :]

        x = x * lp['layer_scalar']

    x = _rmsnorm(x, params['norm']['scale'])
    logits = jnp.dot(x[0, 0].astype(jnp.float32),
                     params['embed_tokens'].T.astype(jnp.float32))  # (vocab,)
    if cfg.final_logit_softcap is not None:
        cap = cfg.final_logit_softcap
        logits = jnp.tanh(logits / cap) * cap

    # Repack updated caches into flat list
    kv_flat_new = []
    for layer_idx in range(kv_shared_start):
        k, v = kv_own[layer_idx]
        kv_flat_new.append(k)
        kv_flat_new.append(v)

    return logits, kv_flat_new, sliding_pos_ring


# ---------------------------------------------------------------------------
# Chunked-prefill attention (for chunk_prefill_step)
# ---------------------------------------------------------------------------

def _attn_chunk(lp, x, positions, start_pos, cfg: Gemma4Config, attn_type: str,
                k_cache, v_cache, pos_ring=None):
    """Chunk attention with KV cache read/write.

    x: (1, C, D)  — C = CHUNK_SIZE tokens
    positions: (1, C) int32  — absolute positions
    start_pos: () int32  — absolute position of first token in chunk
    k_cache, v_cache: (1, cache_len, nkv, hd)
    pos_ring: (1, W) int32 — ring position tracker (required for sliding layers)

    Returns (attn_out (1, C, D), k_cache_updated, v_cache_updated).
    """
    C = x.shape[1]
    is_global = attn_type == AttentionType.GLOBAL
    is_sliding = attn_type == AttentionType.LOCAL_SLIDING
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    hd = cfg.effective_head_dim(attn_type)
    rope_frac = cfg.rope_fraction_global if is_global else cfg.rope_fraction_sliding
    base_freq = cfg.global_rope_base_frequency if is_global else cfg.rope_base_frequency
    max_len = k_cache.shape[1]
    sa = lp['self_attn']

    # Q/K/V projections for the chunk
    q = jnp.dot(x, sa['q_proj']['kernel']).reshape(1, C, num_heads, hd)
    k_new = jnp.dot(x, sa['k_proj']['kernel']).reshape(1, C, num_kv_heads, hd)
    v_new = jnp.dot(x, sa['v_proj']['kernel']).reshape(1, C, num_kv_heads, hd)

    q = _rmsnorm(q, sa['q_norm']['scale'])
    k_new = _rmsnorm(k_new, sa['k_norm']['scale'])
    v_new = _rmsnorm_noscale(v_new)

    q = _apply_rope(q, positions, base_freq, rope_frac)
    k_new = _apply_rope(k_new, positions, base_freq, rope_frac)

    k_new_f16 = k_new.astype(jnp.float16)
    v_new_f16 = v_new.astype(jnp.float16)

    if is_sliding:
        # Ring-buffer write via einsum scatter.
        W = cfg.sliding_window_size
        abs_pos = positions[0]  # (C,)
        ring_slots = abs_pos % W  # (C,)
        # write_mask[w, c] = True iff ring_slots[c] == w
        write_mask = (jnp.arange(W, dtype=jnp.int32)[:, None]
                      == ring_slots[None, :])  # (W, C)
        wm_f16 = write_mask.astype(jnp.float16)
        # Scatter chunk values into ring positions.
        k_gathered = jnp.einsum('wc,bchd->bwhd', wm_f16, k_new_f16)
        v_gathered = jnp.einsum('wc,bchd->bwhd', wm_f16, v_new_f16)
        any_written = write_mask.any(axis=1)[None, :, None, None]  # (1, W, 1, 1)
        k_updated = jnp.where(any_written, k_gathered, k_cache)
        v_updated = jnp.where(any_written, v_gathered, v_cache)
    else:
        # Global: linear write via dynamic_update_slice.
        k_updated = jax.lax.dynamic_update_slice(k_cache, k_new_f16, (0, start_pos, 0, 0))
        v_updated = jax.lax.dynamic_update_slice(v_cache, v_new_f16, (0, start_pos, 0, 0))

    # GQA repeat for attention
    kv_rep = num_heads // num_kv_heads
    k_full = jnp.repeat(k_updated, kv_rep, axis=2) if kv_rep > 1 else k_updated
    v_full = jnp.repeat(v_updated, kv_rep, axis=2) if kv_rep > 1 else v_updated

    qt = jnp.transpose(q, (0, 2, 1, 3))         # (1, H, C, hd)
    kt = jnp.transpose(k_full, (0, 2, 1, 3))    # (1, H, cache_len, hd)
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))  # (1, H, C, cache_len)

    pos_q = positions[0]  # (C,)
    if is_sliding:
        # Ring-buffer mask: use pos_ring to determine validity.
        pk = pos_ring[0]  # (W,)
        mask = (pk[jnp.newaxis, :] >= 0) & (pk[jnp.newaxis, :] <= pos_q[:, jnp.newaxis])
    else:
        # Global linear mask.
        pos_k = jnp.arange(max_len, dtype=jnp.int32)
        mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]  # (C, max_len)

    w = jnp.where(mask[jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w.astype(jnp.float32), axis=-1)

    out = jnp.matmul(w, vt)                                    # (1, H, C, hd)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    out = jnp.dot(out, sa['o_proj']['kernel'])
    return out, k_updated, v_updated


def _attn_chunk_shared(lp, x, positions, start_pos, cfg: Gemma4Config, attn_type: str,
                       shared_kv, pos_ring=None):
    """Chunk attention using K/V from a source (KV-shared) layer.

    Only Q is computed from `lp`; K/V come from `shared_kv`.
    pos_ring: required when the source layer is LOCAL_SLIDING (ring buffer).
    """
    C = x.shape[1]
    is_global = attn_type == AttentionType.GLOBAL
    is_sliding = attn_type == AttentionType.LOCAL_SLIDING
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    hd = cfg.effective_head_dim(attn_type)
    rope_frac = cfg.rope_fraction_global if is_global else cfg.rope_fraction_sliding
    base_freq = cfg.global_rope_base_frequency if is_global else cfg.rope_base_frequency
    sa = lp['self_attn']

    q = jnp.dot(x, sa['q_proj']['kernel']).reshape(1, C, num_heads, hd)
    q = _rmsnorm(q, sa['q_norm']['scale'])
    q = _apply_rope(q, positions, base_freq, rope_frac)

    k_src, v_src = shared_kv
    max_len = k_src.shape[1]

    kv_rep = num_heads // num_kv_heads
    k_full = jnp.repeat(k_src, kv_rep, axis=2) if kv_rep > 1 else k_src
    v_full = jnp.repeat(v_src, kv_rep, axis=2) if kv_rep > 1 else v_src

    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k_full, (0, 2, 1, 3))
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))

    pos_q = positions[0]
    if is_sliding:
        pk = pos_ring[0]  # (W,)
        mask = (pk[jnp.newaxis, :] >= 0) & (pk[jnp.newaxis, :] <= pos_q[:, jnp.newaxis])
    else:
        pos_k = jnp.arange(max_len, dtype=jnp.int32)
        mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]

    w = jnp.where(mask[jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w.astype(jnp.float32), axis=-1)

    out = jnp.transpose(jnp.matmul(w, vt), (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    return jnp.dot(out, sa['o_proj']['kernel'])


# ---------------------------------------------------------------------------
# Chunked prefill: process CHUNK_SIZE tokens with KV cache read/write
# ---------------------------------------------------------------------------

def chunk_prefill_step(
    params,
    tokens,
    start_position,
    kv_flat,
    sliding_pos_ring,
    cfg: Gemma4Config = E2B_CONFIG,
    chunk_size: int = CHUNK_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Process a chunk of tokens through the full model, updating KV caches.

    Args:
        params:         Flax param tree from load_params().
        tokens:         (1, chunk_size) int32 — chunk of tokens (right-padded if last).
        start_position: () int32 — absolute position of the first token in this chunk.
        kv_flat:        List of 30 cache arrays (per-layer shapes, float16).
        sliding_pos_ring: (1, sliding_window_size) int32 — ring position tracker.
        cfg:            Model config.
        chunk_size:     Number of tokens per chunk (must match tokens.shape[1]).
        max_seq_len:    Must match shape of global cache arrays.

    Returns:
        logits: (chunk_size, vocab_size) float32 — logits at all chunk positions.
        kv_flat_new: Updated list of 30 cache arrays.
        sliding_pos_ring_new: Updated (1, sliding_window_size) int32.
    """
    C = chunk_size
    positions = start_position + jnp.arange(C, dtype=jnp.int32)
    positions = positions[jnp.newaxis]  # (1, C)

    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, tokens) * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)

    ple_all = _ple_for_tokens(params, tokens, cfg)  # (1, C, NL*d)

    # Update sliding_pos_ring for this chunk (shared by all sliding layers).
    W = cfg.sliding_window_size
    abs_pos = positions[0]  # (C,)
    ring_slots = abs_pos % W  # (C,)
    write_mask = (jnp.arange(W, dtype=jnp.int32)[:, None]
                  == ring_slots[None, :])  # (W, C)
    any_written = write_mask.any(axis=1)  # (W,)
    # Use float16 dot — MPS does not support int32 matmul.
    new_ring_vals = jnp.dot(
        write_mask.astype(jnp.float16), abs_pos.astype(jnp.float16)
    ).astype(jnp.int32)  # (W,)
    sliding_pos_ring = jnp.where(any_written[None, :], new_ring_vals[None, :],
                                 sliding_pos_ring)

    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    shared_sources = kv_shared_sources(cfg)

    # Unpack flat KV into per-layer dicts
    kv_own = {}
    for slot, layer_idx in enumerate(range(kv_shared_start)):
        kv_own[layer_idx] = (kv_flat[slot * 2], kv_flat[slot * 2 + 1])

    attn_types = list(cfg.attention_types)

    for i, attn_type in enumerate(attn_types):
        lp = params[f'layers.{i}']
        is_shared = i >= kv_shared_start

        d = cfg.per_layer_input_dim
        ple_slice = ple_all[:, :, i * d:(i + 1) * d]  # (1, C, d)

        # Attention sub-layer
        residual = x
        x_ln = _rmsnorm(x, lp['input_layernorm']['scale'])

        if is_shared:
            src = shared_sources[i]
            k_src, v_src = kv_own[src]
            pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
            attn_out = _attn_chunk_shared(lp, x_ln, positions, start_position,
                                          cfg, attn_type, (k_src, v_src),
                                          pos_ring=pr)
        else:
            k_old, v_old = kv_own[i]
            pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
            attn_out, k_new, v_new = _attn_chunk(lp, x_ln, positions, start_position,
                                                  cfg, attn_type, k_old, v_old,
                                                  pos_ring=pr)
            kv_own[i] = (k_new, v_new)

        attn_out = _rmsnorm(attn_out, lp['post_attention_layernorm']['scale'])
        x = residual + attn_out

        # FFN sub-layer
        residual = x
        x_ln = _rmsnorm(x, lp['pre_feedforward_layernorm']['scale'])
        ffn_out = _ffn(lp, x_ln, cfg.effective_hidden_dim(i))
        ffn_out = _rmsnorm(ffn_out, lp['post_feedforward_layernorm']['scale'])
        x = residual + ffn_out

        # PLE gate
        x = _ple_gate(lp, x, ple_slice)

        x = x * lp['layer_scalar']

    x = _rmsnorm(x, params['norm']['scale'])
    logits = jnp.dot(x[0].astype(jnp.float32),
                     params['embed_tokens'].T.astype(jnp.float32))  # (C, vocab)
    if cfg.final_logit_softcap is not None:
        cap = cfg.final_logit_softcap
        logits = jnp.tanh(logits / cap) * cap

    # Repack updated caches into flat list
    kv_flat_new = []
    for layer_idx in range(kv_shared_start):
        k, v = kv_own[layer_idx]
        kv_flat_new.append(k)
        kv_flat_new.append(v)

    return logits, kv_flat_new, sliding_pos_ring
