"""JAX-traceable chunk_prefill and decode_step for CoreML export.

Both functions close over ``params`` and are designed to be lowered with
``jax.jit(...).lower(shape_specs).compiler_ir('stablehlo')`` and converted
to CoreML .mlpackage files via the same pipeline as export.py.

KV cache layout
---------------
Only 15 of the 35 layers store their own KV (layers 15-34 are KV-shared).
All 15 caches use the same shape ``(1, max_seq_len, num_kv_heads, head_dim)``
to avoid ring-buffer complexity (simpler static shapes, easier export).

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
from gemma_chat.cache_spec import kv_shared_sources


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def kv_non_shared_layers(cfg: Gemma4Config) -> List[int]:
    """Layer indices that own a KV cache slot (0 .. kv_shared_start-1)."""
    return list(range(cfg.num_layers - cfg.num_kv_shared_layers))


def kv_cache_shapes(cfg: Gemma4Config, max_seq_len: int) -> List[Tuple]:
    """One shape per non-shared layer; k and v share the same shape."""
    attn = list(cfg.attention_types)
    return [
        (1, max_seq_len, cfg.num_kv_heads, cfg.effective_head_dim(attn[li]))
        for li in kv_non_shared_layers(cfg)
    ]


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
                 k_cache, v_cache, shared_kv=None):
    """Single-token attention with static-shape KV cache read/write.

    x: (1, 1, D)
    position: () int32 traced — absolute position of this new token
    k_cache, v_cache: (1, max_seq_len, nkv, hd)
    shared_kv: optional (k_cache, v_cache) from source layer; if given,
               this layer reads from source and does NOT update its own cache.

    Returns (attn_out (1,1,D), k_cache_updated, v_cache_updated).
    For KV-shared layers the returned caches are the source caches (unchanged).
    """
    is_global = attn_type == AttentionType.GLOBAL
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
        # Compute new K/V, write to cache at slot `position` via jnp.where
        k_new = jnp.dot(x[0, 0], sa['k_proj']['kernel']).reshape(1, 1, num_kv_heads, hd)
        v_new = jnp.dot(x[0, 0], sa['v_proj']['kernel']).reshape(1, 1, num_kv_heads, hd)
        k_new = _rmsnorm(k_new, sa['k_norm']['scale'])
        v_new = _rmsnorm_noscale(v_new)
        k_new = _apply_rope(k_new, pos_arr, base_freq, rope_frac)

        # Write at position using jnp.where (avoids dynamic_update_slice).
        # Cast back to float16 so output KV dtype matches input KV dtype;
        # without this, jnp.where promotes fp16 cache + fp32 k_new → fp32,
        # but the CoreML model's KV inputs are fp16, so every round-trip
        # truncates precision and errors compound across decode steps.
        slot = (jnp.arange(max_len, dtype=jnp.int32) == position)[None, :, None, None]
        k_updated = jnp.where(slot, k_new, k_cache).astype(jnp.float16)
        v_updated = jnp.where(slot, v_new, v_cache).astype(jnp.float16)
        k_full, v_full = k_updated, v_updated

    # Attention validity mask
    valid = jnp.arange(max_len, dtype=jnp.int32) <= position  # include just-written pos
    if attn_type == AttentionType.LOCAL_SLIDING:
        win_start = position - (cfg.sliding_window_size - 1)
        valid = valid & (jnp.arange(max_len, dtype=jnp.int32) >= win_start)

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
    cfg: Gemma4Config = E2B_CONFIG,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Single-token autoregressive decode with KV cache.

    Args:
        params:    Flax param tree from load_params().
        token_id:  () int32 — the new token to process.
        position:  () int32 — absolute position of this token (= T + step).
        kv_flat:   List of 30 (1, max_seq_len, nkv, hd) float16 arrays.
        cfg:       Model config.
        max_seq_len: Must match shape of cache arrays.

    Returns:
        logits: (vocab_size,)
        kv_flat_new: Updated list of 30 cache arrays.
    """
    token_arr = token_id[jnp.newaxis, jnp.newaxis]  # (1, 1)

    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, token_arr) * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)

    ple_all = _ple_for_tokens(params, token_arr, cfg)  # (1, 1, NL*d)

    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    shared_sources = kv_shared_sources(cfg)

    # Unpack flat KV into per-layer dicts
    kv_own = {}   # layer_idx → (k_cache, v_cache) — mutable during this step
    for slot, layer_idx in enumerate(range(kv_shared_start)):
        kv_own[layer_idx] = (kv_flat[slot * 2], kv_flat[slot * 2 + 1])

    for i, attn_type in enumerate(cfg.attention_types):
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
            attn_out, _, _ = _attn_decode(lp, x_ln, position, cfg, attn_type,
                                           k_src, v_src, shared_kv=(k_src, v_src))
        else:
            k_old, v_old = kv_own[i]
            attn_out, k_new, v_new = _attn_decode(lp, x_ln, position, cfg, attn_type,
                                                    k_old, v_old)
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

    return logits, kv_flat_new


# ---------------------------------------------------------------------------
# Chunked-prefill attention (for chunk_prefill_step)
# ---------------------------------------------------------------------------

def _attn_chunk(lp, x, positions, start_pos, cfg: Gemma4Config, attn_type: str,
                k_cache, v_cache):
    """Chunk attention with KV cache read/write.

    x: (1, C, D)  — C = CHUNK_SIZE tokens
    positions: (1, C) int32
    start_pos: () int32  — absolute position of first token in chunk
    k_cache, v_cache: (1, max_seq_len, nkv, hd)

    Returns (attn_out (1, C, D), k_cache_updated, v_cache_updated).
    """
    C = x.shape[1]
    is_global = attn_type == AttentionType.GLOBAL
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

    # Write chunk K/V into cache via dynamic_update_slice, cast to fp16
    # to match input cache dtype (same reason as decode: jnp.float32 from
    # RMSNorm would promote the cache, causing precision drift over steps).
    k_new_f16 = k_new.astype(jnp.float16)
    v_new_f16 = v_new.astype(jnp.float16)
    k_updated = jax.lax.dynamic_update_slice(k_cache, k_new_f16, (0, start_pos, 0, 0))
    v_updated = jax.lax.dynamic_update_slice(v_cache, v_new_f16, (0, start_pos, 0, 0))

    # GQA repeat for attention
    kv_rep = num_heads // num_kv_heads
    k_full = jnp.repeat(k_updated, kv_rep, axis=2) if kv_rep > 1 else k_updated
    v_full = jnp.repeat(v_updated, kv_rep, axis=2) if kv_rep > 1 else v_updated

    qt = jnp.transpose(q, (0, 2, 1, 3))         # (1, H, C, hd)
    kt = jnp.transpose(k_full, (0, 2, 1, 3))    # (1, H, max_len, hd)
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))  # (1, H, C, max_len)

    # Causal mask: query at position start+i can attend to cache positions 0..start+i
    pos_q = positions[0]  # (C,)
    pos_k = jnp.arange(max_len, dtype=jnp.int32)
    mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]  # (C, max_len)
    if attn_type == AttentionType.LOCAL_SLIDING:
        win_start = pos_q[:, jnp.newaxis] - (cfg.sliding_window_size - 1)
        mask = mask & (pos_k[jnp.newaxis, :] >= win_start)

    w = jnp.where(mask[jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w.astype(jnp.float32), axis=-1)

    out = jnp.matmul(w, vt)                                    # (1, H, C, hd)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    out = jnp.dot(out, sa['o_proj']['kernel'])
    return out, k_updated, v_updated


def _attn_chunk_shared(lp, x, positions, start_pos, cfg: Gemma4Config, attn_type: str,
                       shared_kv):
    """Chunk attention using K/V from a source (KV-shared) layer.

    Only Q is computed from `lp`; K/V come from `shared_kv`.
    """
    C = x.shape[1]
    is_global = attn_type == AttentionType.GLOBAL
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
    pos_k = jnp.arange(max_len, dtype=jnp.int32)
    mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]
    if attn_type == AttentionType.LOCAL_SLIDING:
        win_start = pos_q[:, jnp.newaxis] - (cfg.sliding_window_size - 1)
        mask = mask & (pos_k[jnp.newaxis, :] >= win_start)

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
    cfg: Gemma4Config = E2B_CONFIG,
    chunk_size: int = CHUNK_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Process a chunk of tokens through the full model, updating KV caches.

    Args:
        params:         Flax param tree from load_params().
        tokens:         (1, chunk_size) int32 — chunk of tokens (right-padded if last).
        start_position: () int32 — absolute position of the first token in this chunk.
        kv_flat:        List of 30 (1, max_seq_len, nkv, hd) float16 arrays.
        cfg:            Model config.
        chunk_size:     Number of tokens per chunk (must match tokens.shape[1]).
        max_seq_len:    Must match shape of cache arrays.

    Returns:
        logits: (chunk_size, vocab_size) float32 — logits at all chunk positions.
        kv_flat_new: Updated list of 30 cache arrays.
    """
    C = chunk_size
    positions = start_position + jnp.arange(C, dtype=jnp.int32)
    positions = positions[jnp.newaxis]  # (1, C)

    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, tokens) * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)

    ple_all = _ple_for_tokens(params, tokens, cfg)  # (1, C, NL*d)

    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    shared_sources = kv_shared_sources(cfg)

    # Unpack flat KV into per-layer dicts
    kv_own = {}
    for slot, layer_idx in enumerate(range(kv_shared_start)):
        kv_own[layer_idx] = (kv_flat[slot * 2], kv_flat[slot * 2 + 1])

    for i, attn_type in enumerate(cfg.attention_types):
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
            attn_out = _attn_chunk_shared(lp, x_ln, positions, start_position,
                                          cfg, attn_type, (k_src, v_src))
        else:
            k_old, v_old = kv_own[i]
            attn_out, k_new, v_new = _attn_chunk(lp, x_ln, positions, start_position,
                                                  cfg, attn_type, k_old, v_old)
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

    return logits, kv_flat_new
