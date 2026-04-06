"""JAX/Flax incremental decoding for Gemma4-E2B with KV caching.

Provides two functions that share the same parameter tree as Gemma4Transformer:

    prefill(params, tokens)  →  (logits_at_last, kv_state)
    decode_step(params, token_id, position, kv_state)  →  (logits, kv_state)

The KV cache uses 15 slots (not 35) because layers 15-34 reuse K/V from
layers 13 and 14 (KV-sharing).  See cache_spec.py for layout details.

Parity guarantee: for greedy (temperature=0) decoding, the token sequence
produced by prefill + repeated decode_step must match the sequence produced
by running Gemma4Transformer from scratch at every step.

Usage::

    from gemma_chat.decode_jax import prefill, decode_step
    from gemma_chat.weight_mapper import load_params
    from gemma_chat.config import E2B_CONFIG

    params = load_params("google/gemma-4-E2B-it", E2B_CONFIG)
    prompt_ids = [1, 2, 3, ...]

    logits, kv = prefill(params, jnp.array(prompt_ids)[None])
    next_id = int(jnp.argmax(logits))

    for _ in range(max_new_tokens):
        logits, kv = decode_step(params, next_id, kv)
        next_id = int(jnp.argmax(logits))
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from gemma_chat.config import E2B_CONFIG, MAX_SEQ_LEN
from gemma_chat.model import (
    AttentionType,
    Gemma4Config,
    Gemma4Transformer,
    _apply_rope,
    _embed_lookup,
)
from gemma_chat.cache_spec import (
    E2B_CACHE_SPECS,
    E2B_KV_SHARED_SOURCES,
    KVState,
    LayerCacheSpec,
    build_cache_specs,
    kv_shared_sources,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ple_for_tokens(params, tokens_ids, cfg: Gemma4Config):
    """Compute per-layer input embeddings for a batch of token IDs.

    tokens_ids: (B, L) int32
    Returns: (B, L, num_layers * per_layer_input_dim)
    """

    B, L = tokens_ids.shape
    d = cfg.per_layer_input_dim

    # PLE token embedding table: shape (vocab_per_layer, num_layers * d)
    ple_table = params['embed_tokens_per_layer']           # (vocab, NL*d)
    ple_embed_flat = _embed_lookup(ple_table, tokens_ids)  # (B, L, NL*d)
    ple_embed_flat = ple_embed_flat * jnp.sqrt(float(d)).astype(ple_embed_flat.dtype)

    # per_layer_model_projection applied to (already scaled) token embeddings.
    # We need the initial scaled embedding x0 = embed_lookup(embed_tokens, ids) * sqrt(D).
    embed_table = params['embed_tokens']                   # (vocab, D)
    x0 = _embed_lookup(embed_table, tokens_ids)            # (B, L, D)
    x0 = x0 * jnp.sqrt(float(cfg.embed_dim)).astype(x0.dtype)

    # Dense projection: (B, L, D) @ (D, NL*d)^T + 0 bias
    W_proj = params['per_layer_model_projection']['kernel']  # (D, NL*d)
    ple_proj_flat = jnp.dot(x0, W_proj) * (cfg.embed_dim ** -0.5)  # (B, L, NL*d)

    # RMSNorm on per_layer_projection_norm — operates on the last dim NL*d.
    # The norm was trained with shape (B*NL, L, d) to avoid CoreML 4D bug,
    # so we must apply it the same way here.
    NL = B * cfg.num_layers
    scale = params['per_layer_projection_norm']['scale']   # (d,)
    ple_proj_3d = ple_proj_flat.reshape(NL, L, d)
    # Manual RMSNorm (same as model.py: float32, eps=1e-6)
    p32 = ple_proj_3d.astype(jnp.float32)
    var = jnp.mean(jnp.square(p32), axis=-1, keepdims=True)
    ple_proj_3d = p32 * jax.lax.rsqrt(var + 1e-6) * scale.astype(jnp.float32)
    ple_proj_flat = ple_proj_3d.reshape(B, L, cfg.num_layers * d)

    return (ple_proj_flat + ple_embed_flat) * (2.0 ** -0.5)


def _attn_with_cache(
    layer_params: dict,
    x: jnp.ndarray,                  # (1, 1, D) — single new token
    position: int,                   # absolute position index
    cfg: Gemma4Config,
    attn_type: str,
    kv_state: KVState,
    layer_idx: int,
    cache_spec: LayerCacheSpec | None,
    shared_kv_from_cache: tuple | None,  # (k, v) arrays from another layer's cache
) -> Tuple[jnp.ndarray, KVState]:
    """Single-token attention using cached K/V from previous positions.

    For non-KV-shared layers: compute new k/v for position, write to ring/global
    cache, then attend over the full valid cache window.
    For KV-shared layers: attend using `shared_kv_from_cache` (same cache as source).

    Returns: (attn_output (1,1,D), updated kv_state)
    """
    is_global = attn_type == AttentionType.GLOBAL
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    head_dim = cfg.effective_head_dim(attn_type)
    rope_fraction = cfg.rope_fraction_global if is_global else cfg.rope_fraction_sliding
    base_freq = cfg.global_rope_base_frequency if is_global else cfg.rope_base_frequency
    D = x.shape[-1]

    lp = layer_params['self_attn']
    pos_arr = jnp.array([[position]], dtype=jnp.int32)  # (1, 1)

    # Project Q for the new token
    q = jnp.dot(x[0, 0], lp['q_proj']['kernel'])  # (num_heads*head_dim,)
    q = q.reshape(1, 1, num_heads, head_dim)

    # Always project K/V from weights (to consume params), even for shared layers
    k_new = jnp.dot(x[0, 0], lp['k_proj']['kernel']).reshape(1, 1, num_kv_heads, head_dim)
    v_new = jnp.dot(x[0, 0], lp['v_proj']['kernel']).reshape(1, 1, num_kv_heads, head_dim)

    # QK-norm
    q = _rmsnorm_apply(q, lp['q_norm']['scale'])
    k_new = _rmsnorm_apply(k_new, lp['k_norm']['scale'])
    v_new = _rmsnorm_noscale(v_new)

    # RoPE
    q = _apply_rope(q, pos_arr, base_freq, rope_fraction)
    k_new = _apply_rope(k_new, pos_arr, base_freq, rope_fraction)

    # For KV-shared layers: use the source layer's cache as K/V
    if shared_kv_from_cache is not None:
        k_cache, v_cache = shared_kv_from_cache
        # Attend over full source cache; valid_len = source layer's seq_len
        # (same kv_state.seq_len, populated during this same step)
        k_full = jnp.array(k_cache)   # (1, cache_len, nkv, hd)
        v_full = jnp.array(v_cache)
    else:
        # Write new k/v into this layer's cache slot
        k_cache_np, v_cache_np = kv_state.caches[layer_idx]
        wp = kv_state.write_pos[layer_idx]

        if attn_type == AttentionType.LOCAL_SLIDING:
            # Ring buffer: write at wp % window_size
            write_idx = wp % cache_spec.cache_len
        else:
            # Global: write at absolute position (capped at cache_len-1)
            write_idx = min(wp, cache_spec.cache_len - 1)

        k_new_np = np.array(k_new)
        v_new_np = np.array(v_new)
        k_cache_np = k_cache_np.copy()
        v_cache_np = v_cache_np.copy()
        k_cache_np[0, write_idx] = k_new_np[0, 0]
        v_cache_np[0, write_idx] = v_new_np[0, 0]

        # Update state (immutable-style: replace entries)
        new_caches = dict(kv_state.caches)
        new_caches[layer_idx] = (k_cache_np, v_cache_np)
        new_write_pos = dict(kv_state.write_pos)
        new_write_pos[layer_idx] = wp + 1
        kv_state = KVState(
            caches=new_caches,
            write_pos=new_write_pos,
            seq_len=kv_state.seq_len,
        )

        k_full = jnp.array(k_cache_np)
        v_full = jnp.array(v_cache_np)

    # How many valid entries in the cache?
    if shared_kv_from_cache is not None:
        # source layer was updated earlier in this step — its write_pos already bumped
        src_layer = _find_source_for_shared(layer_idx, cfg)
        valid_len = min(kv_state.write_pos.get(src_layer, kv_state.seq_len),
                        k_full.shape[1])
    elif attn_type == AttentionType.LOCAL_SLIDING:
        valid_len = min(kv_state.write_pos[layer_idx], cache_spec.cache_len)
    else:
        valid_len = min(kv_state.write_pos[layer_idx], cache_spec.cache_len)

    # GQA expand
    kv_repeat = num_heads // num_kv_heads
    if kv_repeat > 1:
        k_full = jnp.repeat(k_full, kv_repeat, axis=2)
        v_full = jnp.repeat(v_full, kv_repeat, axis=2)

    # Attention: q (1,1,H,hd) attends over k_full (1,cache_len,H,hd)
    # Transpose to (1,H,1,hd) and (1,H,cache_len,hd)
    q_t = jnp.transpose(q, (0, 2, 1, 3))           # (1,H,1,hd)
    k_t = jnp.transpose(k_full, (0, 2, 1, 3))      # (1,H,cache_len,hd)
    v_t = jnp.transpose(v_full, (0, 2, 1, 3))      # (1,H,cache_len,hd)

    attn_w = jnp.matmul(q_t, jnp.swapaxes(k_t, -2, -1))  # (1,H,1,cache_len)

    # Mask out invalid (unwritten) cache positions
    cache_len_total = k_full.shape[1]
    valid_mask = jnp.arange(cache_len_total) < valid_len   # (cache_len,)
    attn_w = jnp.where(
        valid_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
        attn_w,
        jnp.finfo(attn_w.dtype).min,
    )

    attn_w = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1)
    attn_out = jnp.matmul(attn_w, v_t)             # (1,H,1,hd)
    attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(1, 1, num_heads * head_dim)

    result = jnp.dot(attn_out[0, 0], lp['o_proj']['kernel'])[None, None, :]
    return result, kv_state


def _find_source_for_shared(layer_idx: int, cfg: Gemma4Config) -> int:
    """Return the source layer index for a KV-shared layer."""
    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    attn = list(cfg.attention_types)
    prev = attn[:kv_shared_start]
    if attn[layer_idx] == AttentionType.GLOBAL:
        return kv_shared_start - 1 - list(reversed(prev)).index(AttentionType.GLOBAL)
    else:
        return kv_shared_start - 1 - list(reversed(prev)).index(AttentionType.LOCAL_SLIDING)


def _rmsnorm_apply(x, scale):
    """Apply RMSNorm with learnable scale in float32."""
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6) * scale.astype(jnp.float32)


def _rmsnorm_noscale(x):
    """RMSNorm without learnable scale (v_norm)."""
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6)


def _ffn_forward(layer_params: dict, x: jnp.ndarray, hidden_dim: int) -> jnp.ndarray:
    lp = layer_params['mlp']
    gate = jnp.dot(x, lp['gate_proj']['kernel'])
    up   = jnp.dot(x, lp['up_proj']['kernel'])
    hidden = jax.nn.gelu(gate, approximate=True) * up
    return jnp.dot(hidden, lp['down_proj']['kernel'])


def _ple_gate(layer_params: dict, x: jnp.ndarray, per_layer_input: jnp.ndarray,
              cfg: Gemma4Config) -> jnp.ndarray:
    """Apply the per-layer input gate residual block."""
    gate = jax.nn.gelu(
        jnp.dot(x, layer_params['per_layer_input_gate']['kernel']),
        approximate=True,
    ) * per_layer_input
    proj = jnp.dot(gate, layer_params['per_layer_projection']['kernel'])
    proj = _rmsnorm_apply(proj[None, None, :], layer_params['post_per_layer_input_norm']['scale'])[0, 0]
    return x + proj


# ---------------------------------------------------------------------------
# Decode step — single token, updating KV cache
# ---------------------------------------------------------------------------

def _decode_one_layer(
    layer_params: dict,
    x: jnp.ndarray,        # (1, 1, D)
    position: int,
    cfg: Gemma4Config,
    attn_type: str,
    layer_idx: int,
    kv_state: KVState,
    cache_spec: LayerCacheSpec | None,
    shared_kv_from_cache: tuple | None,
    per_layer_input: jnp.ndarray | None,
) -> Tuple[jnp.ndarray, KVState]:
    """One transformer block for a single new token."""
    hidden_dim = cfg.effective_hidden_dim(layer_idx)

    # Attention sub-layer
    residual = x
    x_ln = _rmsnorm_apply(x, layer_params['input_layernorm']['scale'])
    attn_out, kv_state = _attn_with_cache(
        layer_params, x_ln, position, cfg, attn_type,
        kv_state, layer_idx, cache_spec, shared_kv_from_cache,
    )
    attn_out = _rmsnorm_apply(attn_out, layer_params['post_attention_layernorm']['scale'])
    x = residual + attn_out

    # FFN sub-layer
    residual = x
    x_ln = _rmsnorm_apply(x, layer_params['pre_feedforward_layernorm']['scale'])
    ffn_out = _ffn_forward(layer_params, x_ln[0, 0], hidden_dim)
    ffn_out = _rmsnorm_apply(ffn_out[None, None, :],
                             layer_params['post_feedforward_layernorm']['scale'])
    x = residual + ffn_out

    # PLE gate
    x = x + 0  # ensure shape is (1,1,D)
    xi = x[0, 0]
    xi = _ple_gate(layer_params, xi, per_layer_input[0, 0], cfg)
    x = xi[None, None, :]

    # Per-layer scalar
    scalar = layer_params['layer_scalar']
    x = x * scalar

    return x, kv_state


def decode_step(
    params: dict,
    token_id: int,
    kv_state: KVState,
    cfg: Gemma4Config = E2B_CONFIG,
) -> Tuple[jnp.ndarray, KVState]:
    """Generate the next logit distribution given one new token and KV state.

    Args:
        params:    Flax param tree from load_params().
        token_id:  Integer token ID of the new token.
        kv_state:  KVState from prefill() or previous decode_step().
        cfg:       Model configuration (default: E2B_CONFIG).

    Returns:
        (logits, new_kv_state) where logits has shape (vocab_size,).
    """
    position = kv_state.seq_len
    token_arr = jnp.array([[token_id]], dtype=jnp.int32)  # (1, 1)

    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, token_arr) * jnp.sqrt(float(cfg.embed_dim))  # (1,1,D)

    # PLE for single token
    ple_all = _ple_for_tokens(params, token_arr, cfg)  # (1, 1, NL*d)

    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    cache_specs = {s.layer_idx: s for s in build_cache_specs(cfg)}
    shared_sources = kv_shared_sources(cfg)

    for i, attn_type in enumerate(cfg.attention_types):
        layer_params = params[f'layers.{i}']
        is_shared = (cfg.num_kv_shared_layers > 0 and i >= kv_shared_start)
        spec = cache_specs.get(i)

        if is_shared:
            src = shared_sources[i]
            shared_kv = kv_state.caches[src]
        else:
            shared_kv = None

        d = cfg.per_layer_input_dim
        ple_slice = ple_all[:, :, i * d:(i + 1) * d]  # (1,1,d)

        x, kv_state = _decode_one_layer(
            layer_params, x, position, cfg, attn_type,
            i, kv_state, spec, shared_kv, ple_slice,
        )

    x = _rmsnorm_apply(x, params['norm']['scale'])
    logits = jnp.dot(x[0, 0], embed_table.T)
    if cfg.final_logit_softcap is not None:
        cap = cfg.final_logit_softcap
        logits = jnp.tanh(logits / cap) * cap

    new_kv_state = KVState(
        caches=kv_state.caches,
        write_pos=kv_state.write_pos,
        seq_len=kv_state.seq_len + 1,
    )
    return logits, new_kv_state


# ---------------------------------------------------------------------------
# Prefill — process full prompt, return logits at last token + initial KV state
# ---------------------------------------------------------------------------

def prefill(
    params: dict,
    tokens: jnp.ndarray,              # (1, T) int32
    cfg: Gemma4Config = E2B_CONFIG,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[jnp.ndarray, KVState]:
    """Run the full prompt through the model, populating the KV cache.

    Uses the existing Gemma4Transformer full-forward for correctness, then
    extracts and stores K/V values at each cache layer.

    Args:
        params:      Flax param tree.
        tokens:      (1, T) prompt token IDs.
        cfg:         Model configuration.
        max_seq_len: Maximum sequence length.

    Returns:
        (logits, kv_state) where logits is the (vocab_size,) distribution at
        the last prompt token position, and kv_state is ready for decode_step.
    """
    T = tokens.shape[1]
    positions = jnp.arange(T, dtype=jnp.int32)[jnp.newaxis, :]

    # Run full forward to get correct logits (reuses all the existing logic)
    from gemma_chat.weight_mapper import load_params_into_model
    model = Gemma4Transformer(config=cfg, rngs=nnx.Rngs(params=0))
    load_params_into_model(model, params, cfg)
    logits_full = model(tokens)  # (1, T, vocab)
    logits_last = logits_full[0, T - 1, :]                  # (vocab,)

    # Now populate the KV cache by re-running the prefill internals
    # We extract K/V from each cache layer by running a modified forward.
    kv_state = _build_kv_state_from_prefill(model, params, tokens, positions, cfg, max_seq_len)

    return logits_last, kv_state


def _build_kv_state_from_prefill(
    model,
    params: dict,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    cfg: Gemma4Config,
    max_seq_len: int,
) -> KVState:
    """Run prefill through cache layers and store K/V in a KVState."""
    T = tokens.shape[1]
    specs = build_cache_specs(cfg, max_seq_len)
    spec_map = {s.layer_idx: s for s in specs}
    kv_shared_start = cfg.num_layers - cfg.num_kv_shared_layers

    # We need the intermediate hidden states at each cache layer.
    # Run the transformer layer-by-layer, collecting K/V from the attention modules.
    embed_table = params['embed_tokens']
    x = _embed_lookup(embed_table, tokens) * jnp.sqrt(float(cfg.embed_dim))

    ple_all = _ple_for_tokens(params, tokens, cfg)  # (1, T, NL*d)

    shared_sources = kv_shared_sources(cfg)
    stored_kv: dict = {}  # layer_idx → (k_rope, v_norm) arrays shape (1,T,nkv,hd)
    kv_caches: dict = {}  # layer_idx → (k_np, v_np)
    write_pos: dict = {}

    for i, attn_type in enumerate(cfg.attention_types):
        lp = params[f'layers.{i}']
        is_shared = cfg.num_kv_shared_layers > 0 and i >= kv_shared_start

        if is_shared:
            src = shared_sources[i]
            shared_kv_arg = stored_kv[src]
        else:
            shared_kv_arg = None

        d = cfg.per_layer_input_dim
        ple_slice = ple_all[:, :, i * d:(i + 1) * d]

        is_store = i in spec_map

        result = model.layers[i](
            x, positions, ple_slice,
            shared_kv=shared_kv_arg,
            return_kv=is_store,
        )
        if is_store:
            x, (k_rope, v_norm) = result
            stored_kv[i] = (k_rope, v_norm)  # (1, T, nkv, hd)

            # Store into KV cache, handling ring buffer or global truncation
            spec = spec_map[i]
            if attn_type == AttentionType.LOCAL_SLIDING:
                # For sliding window: only keep last `window_size` positions
                win = spec.cache_len
                if T <= win:
                    k_np = np.array(k_rope)      # (1, T, nkv, hd)
                    v_np = np.array(v_norm)
                    # Pad to (1, win, nkv, hd)
                    pad_len = win - T
                    k_pad = np.zeros((1, pad_len, spec.num_kv_heads, spec.head_dim), dtype=k_np.dtype)
                    v_pad = np.zeros_like(k_pad)
                    k_full = np.concatenate([k_np, k_pad], axis=1)
                    v_full = np.concatenate([v_np, v_pad], axis=1)
                    wp = T
                else:
                    # Take the last `win` tokens
                    k_full = np.array(k_rope[:, -win:])
                    v_full = np.array(v_norm[:, -win:])
                    wp = T  # next write index mod win = T % win
            else:
                # Global: truncate at max_seq_len
                keep = min(T, spec.cache_len)
                k_trunc = np.array(k_rope[:, :keep])
                v_trunc = np.array(v_norm[:, :keep])
                pad_len = spec.cache_len - keep
                k_pad = np.zeros((1, pad_len, spec.num_kv_heads, spec.head_dim), dtype=k_trunc.dtype)
                v_pad = np.zeros_like(k_pad)
                k_full = np.concatenate([k_trunc, k_pad], axis=1)
                v_full = np.concatenate([v_trunc, v_pad], axis=1)
                wp = keep

            kv_caches[i] = (k_full, v_full)
            write_pos[i] = wp
        else:
            x = result

    return KVState(caches=kv_caches, write_pos=write_pos, seq_len=T)
