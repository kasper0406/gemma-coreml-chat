"""Gemma4 transformer model matching the google/gemma-4-E2B-it HuggingFace checkpoint.

Architectural features (verified against HF safetensors weight keys and transformers source):
- RMSNorm (pre-norm + post-norm transformer)
- Multi-head attention with GQA + QK-norm
- Rotary Position Embeddings (RoPE):
  - Sliding layers: full RoPE (rope_fraction=1.0)
  - Global layers: partial RoPE (rope_fraction=0.25) at 1M base frequency
- GeGLU gated feedforward (hidden_activation='gelu_pytorch_tanh'):
  - gate = gelu_tanh(gate_proj(x)); output = down_proj(gate * up_proj(x))
- Per-layer input embeddings (PLE), applied AFTER attention+FFW each block:
  - ple_embed = embed_tokens_per_layer(tokens) * sqrt(ple_dim), reshaped to [B,L,layers,ple_dim]
  - ple_proj = per_layer_projection_norm(per_layer_model_projection(x) * embed_dim^-0.5)
  - ple_all = (ple_proj + ple_embed) * 2^-0.5   (combined, one [B,L,ple_dim] slice per layer)
  - In each block: gate = gelu_tanh(gate_proj(x)) * per_layer_input
    x = x + post_per_layer_input_norm(per_layer_projection(gate))
- layer_scalar per transformer block
- Global attention layers use larger head_dim and wider MLP
- Logit soft-capping
- Sliding window + global attention pattern
"""

import dataclasses
from typing import Sequence

from flax import nnx
import jax
import jax.lax as lax
import jax.numpy as jnp


def _embed_lookup(table, tokens):
    """Embedding table lookup with CLIP OOB mode — avoids while_loop fallback.

    jnp.take(table, tokens, axis=0) uses FILL mode which generates an OOB
    validity mask via a custom HLO reduce body.  stablehlo-coreml can't lower
    that to a MIL op and falls back to a while_loop, which crashes Espresso.
    Using lax.gather with CLIP mode instead: OOB indices are clamped to the
    valid range (safe since all token IDs from the tokenizer are in [0, vocab)).
    """
    vocab, dim = table.shape
    dnums = lax.GatherDimensionNumbers(
        offset_dims=(len(tokens.shape),),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    return lax.gather(
        table,
        tokens[..., None],            # [..., 1] integer index
        dnums,
        slice_sizes=(1, dim),
        mode=lax.GatherScatterMode.CLIP,
    )


class AttentionType:
    LOCAL_SLIDING = "local_sliding"
    GLOBAL = "global"


@dataclasses.dataclass(frozen=True)
class Gemma4Config:
    num_embed: int = 100
    embed_dim: int = 64
    # Sliding attention layer dimensions
    hidden_dim: int = 256
    num_heads: int = 2
    head_dim: int = 32
    num_kv_heads: int = 1
    # Global attention layers use a wider head_dim and MLP
    # (0 = same as sliding values)
    global_head_dim: int = 0
    global_hidden_dim: int = 0
    final_logit_softcap: float = 30.0
    attention_types: Sequence[str] = (
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )
    sliding_window_size: int = 64
    rope_base_frequency: float = 10_000.0
    global_rope_base_frequency: float = 1_000_000.0
    # Fraction of head_dim to apply RoPE to
    rope_fraction_sliding: float = 1.0
    rope_fraction_global: float = 0.25
    # Per-layer input embeddings (PLE). 0 = disabled.
    per_layer_input_dim: int = 16

    # Wider MLP for layers >= wide_mlp_from_layer (-1 = disabled)
    wide_mlp_from_layer: int = -1
    wide_hidden_dim: int = 0

    # KV sharing: the last `num_kv_shared_layers` layers reuse K/V from the
    # last non-shared sliding and global layers respectively (Gemma4 E2B).
    # 0 = disabled (all layers compute their own K/V).
    num_kv_shared_layers: int = 0

    # MoE configuration (only used when enable_moe=True)
    enable_moe: bool = False
    num_experts: int = 0
    expert_dim: int = 0
    top_k_experts: int = 2
    moe_dense_hidden_dim: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.attention_types)

    def effective_head_dim(self, attn_type: str) -> int:
        if attn_type == AttentionType.GLOBAL and self.global_head_dim > 0:
            return self.global_head_dim
        return self.head_dim

    def effective_hidden_dim(self, layer_idx: int) -> int:
        if (self.wide_mlp_from_layer >= 0
                and layer_idx >= self.wide_mlp_from_layer
                and self.wide_hidden_dim > 0):
            return self.wide_hidden_dim
        return self.hidden_dim


class RMSNorm(nnx.Module):
    """RMS normalization with learnable scale.

    Computes in float32 to prevent fp16 underflow (epsilon=1e-6 underflows
    to 0 in fp16) and keeps the fp32 result so all downstream activations
    stay in fp32.  Returning fp32 from fp16 input causes JAX to insert an
    explicit cast(fp16→fp32) in StableHLO for the residual add, which
    cast_optimization will NOT fold (we remove that pass) → fp16 weights
    stored compactly while all compute runs in fp32.
    """
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x):
        x32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        x_norm = x32 * jax.lax.rsqrt(variance + 1e-6)
        return x_norm * self.scale[...].astype(jnp.float32)


class RMSNormNoScale(nnx.Module):
    """RMS normalization without a learnable scale parameter.

    Matches Gemma4RMSNorm(with_scale=False) — computes in fp32 and returns fp32.
    """
    def __call__(self, x):
        x32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        return x32 * jax.lax.rsqrt(variance + 1e-6)


def _apply_rope(x, positions, base_frequency, rope_fraction=1.0):
    """Apply rotary position embeddings to input tensor.

    Matches HF Gemma4's rotate_half / apply_rotary_pos_emb semantics:
    - For full RoPE (rope_fraction=1.0, sliding layers):
        pairs (i, i+head_dim//2) — standard rotate_half
    - For proportional RoPE (rope_fraction<1.0, global layers):
        inv_freq has `half_dim` non-trivial + `head_dim//2 - half_dim` zero entries,
        then doubled to shape `head_dim` via cat(freqs, freqs).  Combined with
        rotate_half (pairing i↔i+head_dim//2) this rotates only pairs
        (0, head_dim//2) … (half_dim-1, head_dim//2+half_dim-1); all other
        dims are unchanged (cos=1, sin=0 from zero frequencies).
    """
    head_dim = x.shape[-1]
    rope_dim = int(head_dim * rope_fraction)
    if rope_dim == 0:
        return x

    half_dim = rope_dim // 2   # number of angle pairs that are non-trivial
    head_half = head_dim // 2  # midpoint used by HF's rotate_half

    # HF divides by the full head_dim, not rope_dim, even when using partial RoPE.
    freq_exponent = 2.0 * jnp.arange(half_dim, dtype=jnp.float32) / head_dim
    timescale = base_frequency ** freq_exponent

    # positions: (B, L) -> (B, L, 1)
    positions = positions[..., jnp.newaxis].astype(jnp.float32)
    sinusoid_inp = positions / timescale[jnp.newaxis, jnp.newaxis, :]

    # Expand sin/cos for broadcasting with heads: (B, L, 1, half_dim)
    sin = jnp.sin(sinusoid_inp)[:, :, jnp.newaxis, :]
    cos = jnp.cos(sinusoid_inp)[:, :, jnp.newaxis, :]

    # x1: first half_dim dims;  x2: paired dims at offset head_half
    x1 = x[..., :half_dim]
    x2 = x[..., head_half:head_half + half_dim]

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin

    if rope_dim == head_dim:
        # Full RoPE: half_dim == head_half, no unchanged interior/tail.
        # Avoid zero-size slices that cause shape errors in CoreML tracing.
        return jnp.concatenate([x1_rot, x2_rot], axis=-1)

    # Proportional RoPE: preserve the unchanged dims between x1 and x2,
    # and the unchanged tail after x2.
    return jnp.concatenate([
        x1_rot,
        x[..., half_dim:head_half],        # unchanged middle  (dims half_dim … head_half-1)
        x2_rot,
        x[..., head_half + half_dim:],     # unchanged tail    (dims head_half+half_dim … end)
    ], axis=-1)


class GemmaAttention(nnx.Module):
    """Multi-head attention with GQA, QK-norm, and RoPE, matching Gemma4 patterns.

    Global attention layers use a larger head_dim and partial RoPE (rope_fraction=0.25).
    Sliding attention layers use the standard head_dim and full RoPE.
    """
    def __init__(self, config: Gemma4Config,
                 attn_type: str = AttentionType.LOCAL_SLIDING, *, rngs: nnx.Rngs):
        self.config = config
        self.attn_type = attn_type

        head_dim = config.effective_head_dim(attn_type)
        embed_dim = config.embed_dim

        self.q_proj = nnx.Linear(embed_dim, config.num_heads * head_dim,
                                 use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, config.num_kv_heads * head_dim,
                                 use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, config.num_kv_heads * head_dim,
                                 use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(config.num_heads * head_dim, embed_dim,
                                 use_bias=False, rngs=rngs)

        self.q_norm = RMSNorm(head_dim, rngs=rngs)
        self.k_norm = RMSNorm(head_dim, rngs=rngs)

    def __call__(self, x, positions, shared_kv=None, return_kv=False):
        """
        shared_kv: optional (k_rope, v_norm) pre-computed from an earlier layer.
            When provided (KV-sharing layers), the layer's own k/v projections are
            still applied but their outputs are discarded in favour of the shared KV.
        return_kv: if True, return (attn_out, (k_rope, v_norm)) so the caller can
            store the K/V for downstream KV-sharing layers.
        """
        B, L, D = x.shape
        cfg = self.config
        is_global = self.attn_type == AttentionType.GLOBAL
        num_heads = cfg.num_heads
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.effective_head_dim(self.attn_type)
        rope_fraction = (cfg.rope_fraction_global if is_global
                         else cfg.rope_fraction_sliding)
        base_freq = (cfg.global_rope_base_frequency if is_global
                     else cfg.rope_base_frequency)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, num_heads, head_dim)
        k = k.reshape(B, L, num_kv_heads, head_dim)
        v = v.reshape(B, L, num_kv_heads, head_dim)

        # QK normalization (Gemma4 feature)
        q_normed = self.q_norm(q)
        k_normed = self.k_norm(k)
        # Value normalization without learnable scale (Gemma4 feature)
        v_normed = RMSNormNoScale()(v)

        q_normed = _apply_rope(q_normed, positions, base_freq, rope_fraction)
        k_normed = _apply_rope(k_normed, positions, base_freq, rope_fraction)

        # KV sharing: if caller provides pre-computed K/V from an earlier layer,
        # override this layer's own K/V.
        # Save pre-GQA-expansion copies for return_kv.
        k_rope_pre_gqa = k_normed  # shape: (B, L, num_kv_heads, head_dim)
        v_norm_pre_gqa = v_normed

        if shared_kv is not None:
            k_normed, v_normed = shared_kv

        # GQA: repeat KV heads to match Q heads
        kv_repeat = num_heads // num_kv_heads
        if kv_repeat > 1:
            k_normed = jnp.repeat(k_normed, kv_repeat, axis=2)
            v_normed = jnp.repeat(v_normed, kv_repeat, axis=2)

        # Scaled dot-product attention: (B, L, H, D) -> (B, H, L, D)
        q_t = jnp.transpose(q_normed, (0, 2, 1, 3))
        k_t = jnp.transpose(k_normed, (0, 2, 1, 3))
        v_t = jnp.transpose(v_normed, (0, 2, 1, 3))

        # HF Gemma4 uses scale=1.0 (not head_dim**-0.5) when QK-norm is applied
        attn_weights = jnp.matmul(q_t, jnp.swapaxes(k_t, -2, -1))

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))

        # Sliding window mask for local attention
        if self.attn_type == AttentionType.LOCAL_SLIDING:
            window_mask = jnp.triu(
                jnp.ones((L, L), dtype=jnp.bool_),
                k=-cfg.sliding_window_size + 1
            )
            causal_mask = causal_mask & window_mask

        attn_weights = jnp.where(
            causal_mask[jnp.newaxis, jnp.newaxis, :, :],
            attn_weights,
            -10000.0,
        )

        # Softmax in float32 to prevent fp16 overflow (exp(30) > fp16 max).
        # Keep result in fp32 — same reasoning as RMSNorm above.
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn_out = jnp.matmul(attn_weights, v_t)

        # (B, H, L, D) -> (B, L, H*D)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(B, L, num_heads * head_dim)

        result = self.o_proj(attn_out)
        if return_kv:
            return result, (k_rope_pre_gqa, v_norm_pre_gqa)
        return result


class GemmaFeedForward(nnx.Module):
    """GeGLU gated feedforward, matching Gemma4 hidden_activation='gelu_pytorch_tanh'."""
    def __init__(self, embed_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, embed_dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = jax.nn.gelu(gate, approximate=False) * up
        return self.down_proj(hidden)


class MoE(nnx.Module):
    """Mixture of Experts with top-k routing, matching Gemma4 26B-A4B.

    Each token is routed to top-k experts via learned router logits.
    Experts use GELU-gated FFW (not SwiGLU). Outputs are weighted-summed.
    """
    def __init__(self, features: int, hidden_dim: int, num_experts: int,
                 num_experts_per_tok: int, *, rngs: nnx.Rngs):
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router_norm = RMSNormNoScale()
        self.router_scale = nnx.Param(jnp.ones((features,)))
        self.router_logits = nnx.Param(jnp.zeros((features, num_experts)))
        self.gating_einsum = nnx.Param(
            jnp.zeros((num_experts, 2, hidden_dim, features)))
        self.linear = nnx.Param(
            jnp.zeros((num_experts, hidden_dim, features)))
        self.per_expert_scale = nnx.Param(jnp.ones((num_experts,)))

    def __call__(self, x):
        B, L, D = x.shape
        features = self.features
        num_experts = self.num_experts
        k = self.num_experts_per_tok

        # Router: RMS-norm, scale, compute logits
        router_input = self.router_norm(x)
        root_size = jax.lax.rsqrt(
            jnp.array(features, dtype=router_input.dtype))
        router_input = (
            router_input * root_size
            * self.router_scale[...].astype(router_input.dtype))
        logits = jnp.dot(router_input, self.router_logits[...])  # (B, L, E)

        # Top-k routing with softmax probabilities
        logits_f32 = logits.astype(jnp.float32)
        probs = jax.nn.softmax(logits_f32, axis=-1)  # (B, L, E)
        top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, k)  # (B,L,k)

        # Renormalize weights among selected experts
        indicator = jax.nn.one_hot(
            top_k_indices, num_experts, dtype=probs.dtype)  # (B,L,k,E)
        gate_weights = indicator.sum(axis=-2) * probs  # (B, L, E)
        renorm = jnp.sum(gate_weights, axis=-1, keepdims=True)  # (B, L, 1)
        renorm = jnp.where(renorm > 0.0, renorm, 1.0)
        weights = probs / renorm  # (B, L, E)

        # Gather weights for selected experts
        top_k_weights = jnp.take_along_axis(
            weights, top_k_indices, axis=-1)  # (B, L, k)

        gating_w = self.gating_einsum[...]
        linear_w = self.linear[...]
        per_expert_scale_v = self.per_expert_scale[...]

        # Process each expert's tokens via dense matmuls
        output = jnp.zeros_like(x)  # (B, L, D)

        for ki in range(k):
            expert_idx = top_k_indices[:, :, ki]  # (B, L)
            expert_w = top_k_weights[:, :, ki]    # (B, L)

            # Gather params for selected experts
            gate_params = gating_w[expert_idx]  # (B, L, 2, H, D)
            lin_params = linear_w[expert_idx]    # (B, L, H, D)
            escale = per_expert_scale_v[expert_idx]  # (B, L)

            # GELU-gated FFW: two projections from gate_params
            gate_0 = jnp.einsum(
                'bld,blhd->blh', x, gate_params[:, :, 0, :, :])
            gate_1 = jnp.einsum(
                'bld,blhd->blh', x, gate_params[:, :, 1, :, :])
            activated = jax.nn.gelu(gate_0) * gate_1  # (B, L, H)

            # Project back to embed_dim
            expert_out = jnp.einsum(
                'blh,blhd->bld', activated, lin_params)  # (B, L, D)

            # Scale by per-expert scale and routing weight
            expert_out = expert_out * escale[..., jnp.newaxis]
            expert_out = expert_out * expert_w[..., jnp.newaxis]

            output = output + expert_out

        return output


class GemmaBlock(nnx.Module):
    """Pre-norm transformer block matching google/gemma-4-E2B-it."""
    def __init__(self, config: Gemma4Config,
                 attn_type: str = AttentionType.LOCAL_SLIDING,
                 layer_idx: int = 0, *, rngs: nnx.Rngs):
        cfg = config
        self.config = config
        self.attn_type = attn_type
        self.layer_idx = layer_idx

        embed_dim = cfg.embed_dim

        # Attention sub-block
        self.input_layernorm = RMSNorm(embed_dim, rngs=rngs)
        self.self_attn = GemmaAttention(config=cfg, attn_type=attn_type, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(embed_dim, rngs=rngs)

        # Feed-forward sub-block
        if cfg.enable_moe:
            self.pre_ffw2_norm = RMSNorm(embed_dim, rngs=rngs)
            dense_hidden = cfg.moe_dense_hidden_dim or cfg.hidden_dim
            self.ffw2 = GemmaFeedForward(embed_dim, dense_hidden, rngs=rngs)
            self.post_ffw2_norm = RMSNorm(embed_dim, rngs=rngs)
            self.pre_feedforward_layernorm = RMSNorm(embed_dim, rngs=rngs)
            self.moe = MoE(cfg.embed_dim, cfg.expert_dim, cfg.num_experts,
                           cfg.top_k_experts, rngs=rngs)
            self.post_ffw1_norm = RMSNorm(embed_dim, rngs=rngs)
            self.post_feedforward_layernorm = RMSNorm(embed_dim, rngs=rngs)
        else:
            self.pre_feedforward_layernorm = RMSNorm(embed_dim, rngs=rngs)
            hidden_dim = cfg.effective_hidden_dim(layer_idx)
            self.mlp = GemmaFeedForward(embed_dim, hidden_dim, rngs=rngs)
            self.post_feedforward_layernorm = RMSNorm(embed_dim, rngs=rngs)

        # PLE gate + projection
        ple_dim = cfg.per_layer_input_dim
        self.per_layer_input_gate = nnx.Linear(
            embed_dim, ple_dim, use_bias=False, rngs=rngs)
        self.per_layer_projection = nnx.Linear(
            ple_dim, embed_dim, use_bias=False, rngs=rngs)
        self.post_per_layer_input_norm = RMSNorm(embed_dim, rngs=rngs)

        self.layer_scalar = nnx.Param(jnp.ones((1,)))

    def __call__(self, x, positions, per_layer_input, shared_kv=None, return_kv=False):
        cfg = self.config

        # ── Attention ─────────────────────────────────────────────────────
        residual = x
        x = self.input_layernorm(x)
        attn_result = self.self_attn(x, positions,
                                     shared_kv=shared_kv, return_kv=return_kv)
        if return_kv:
            x, kv = attn_result
        else:
            x = attn_result
        x = self.post_attention_layernorm(x)
        x = residual + x

        # ── Feed-forward ──────────────────────────────────────────────────
        if cfg.enable_moe:
            dense_out = self.pre_ffw2_norm(x)
            dense_out = self.ffw2(dense_out)
            dense_out = self.post_ffw2_norm(dense_out)

            moe_in = self.pre_feedforward_layernorm(x)
            moe_out = self.moe(moe_in)
            moe_out = self.post_ffw1_norm(moe_out)

            ffw_out = dense_out + moe_out
            ffw_out = self.post_feedforward_layernorm(ffw_out)
            x = x + ffw_out
        else:
            residual = x
            x = self.pre_feedforward_layernorm(x)
            x = self.mlp(x)
            x = self.post_feedforward_layernorm(x)
            x = residual + x

        # ── Gated per-layer input embeddings (PLE) ───────────────────────
        residual = x
        gate = jax.nn.gelu(
            self.per_layer_input_gate(x),
            approximate=False,
        ) * per_layer_input
        proj = self.per_layer_projection(gate)
        proj = self.post_per_layer_input_norm(proj)
        x = residual + proj

        # ── Per-layer output scalar ────────────────────────────────────────
        x = x * self.layer_scalar[...]

        if return_kv:
            return x, kv
        return x


class Gemma4Transformer(nnx.Module):
    """Full Gemma4-E2B transformer matching google/gemma-4-E2B-it."""
    def __init__(self, config: Gemma4Config, *, rngs: nnx.Rngs):
        cfg = config
        self.config = config

        # Token embeddings (shared with output projection via weight tying)
        self.embed_tokens = nnx.Param(
            jnp.zeros((cfg.num_embed, cfg.embed_dim)))

        # Per-layer input embeddings (PLE) table
        self.embed_tokens_per_layer = nnx.Param(
            jnp.zeros((cfg.num_embed, cfg.num_layers * cfg.per_layer_input_dim)))

        # PLE projection from initial token embeddings
        self.per_layer_model_projection = nnx.Linear(
            cfg.embed_dim, cfg.num_layers * cfg.per_layer_input_dim,
            use_bias=False, rngs=rngs)

        # PLE projection norm
        self.per_layer_projection_norm = RMSNorm(
            cfg.per_layer_input_dim, rngs=rngs)

        # Transformer blocks
        self.layers = nnx.List([
            GemmaBlock(config=cfg, attn_type=attn_type, layer_idx=i, rngs=rngs)
            for i, attn_type in enumerate(cfg.attention_types)
        ])

        # Final norm
        self.norm = RMSNorm(cfg.embed_dim, rngs=rngs)

    def __call__(self, tokens, return_kv_cache: bool = False):
        """Forward pass.

        Args:
            tokens: (B, L) int32 token IDs.
            return_kv_cache: When True, also return a flat list of KV tensors
                for all non-KV-shared layers (layers 0 .. num_layers-num_kv_shared_layers-1).
                Return value becomes ``(logits, k0, v0, k1, v1, ..., k_N, v_N)``
                where each k/v has shape ``(B, L, num_kv_heads, head_dim)``.
        """
        cfg = self.config
        B, L = tokens.shape

        embedding_table = self.embed_tokens[...]
        x = _embed_lookup(embedding_table, tokens)

        # Scale embeddings (Gemma convention)
        x = x * jnp.sqrt(cfg.embed_dim).astype(x.dtype)

        # Positions for RoPE
        positions = jnp.arange(L, dtype=jnp.int32)[jnp.newaxis, :]
        positions = jnp.broadcast_to(positions, (B, L))

        # Per-layer input embeddings (PLE):
        #   1. PLE token table: [vocab, num_layers * ple_dim], scaled by sqrt(ple_dim)
        #   2. per_layer_model_projection: Linear on initial embeddings, scaled by embed_dim^(-0.5)
        #   3. per_layer_projection_norm: RMSNorm on projected result
        #   4. Combine: (proj_normed + ple_embed) * 2^(-0.5)
        #
        # NOTE: We deliberately avoid a 4D reshape for the RMSNorm.  A 4D input
        # [B, L, num_layers, ple_dim] triggers a coremltools fuse_layernorm_or_instancenorm
        # bug that generates batch_norm ops with wrong output shape [B, L, 1, 1].  Instead
        # we reshape to [B*num_layers, L, ple_dim] so the norm sees a 3D input.
        ple_table = self.embed_tokens_per_layer[...]
        # Scale PLE embeddings by sqrt(per_layer_input_dim), matching
        # HF Gemma4TextScaledWordEmbedding(embed_scale=hidden_size_per_layer_input**0.5)
        ple_embed_flat = _embed_lookup(ple_table, tokens)  # (B, L, num_layers*ple_dim)
        ple_embed_flat = ple_embed_flat * jnp.sqrt(cfg.per_layer_input_dim).astype(ple_embed_flat.dtype)

        # Per-layer projection from initial (scaled) token embeddings.
        # Multiplying by embed_dim**-0.5 cancels the embedding scale so
        # the projection effectively sees un-scaled embeddings.
        ple_proj_flat = self.per_layer_model_projection(x) * (cfg.embed_dim ** -0.5)

        # Reshape to 3D [B*num_layers, L, ple_dim] for RMSNorm to avoid the
        # coremltools batch_norm fusion bug triggered by 4D inputs.
        NL = B * cfg.num_layers
        d = cfg.per_layer_input_dim
        ple_proj_3d = ple_proj_flat.reshape(NL, L, d)
        ple_proj_3d = self.per_layer_projection_norm(ple_proj_3d)
        ple_proj_flat = ple_proj_3d.reshape(B, L, cfg.num_layers * d)

        # Combine embedding and projection, normalise magnitude.
        ple_all_flat = (ple_proj_flat + ple_embed_flat) * (2.0 ** -0.5)

        # Transformer blocks
        # KV sharing: the last `num_kv_shared_layers` layers reuse K/V from the
        # last sliding / global layer before the sharing boundary.
        num_kv_shared = cfg.num_kv_shared_layers
        if num_kv_shared > 0:
            first_kv_shared_idx = cfg.num_layers - num_kv_shared
            prev_types = list(cfg.attention_types[:first_kv_shared_idx])
            last_sliding_src = (first_kv_shared_idx - 1
                                - list(reversed(prev_types)).index(AttentionType.LOCAL_SLIDING))
            last_global_src  = (first_kv_shared_idx - 1
                                - list(reversed(prev_types)).index(AttentionType.GLOBAL))
        else:
            first_kv_shared_idx = cfg.num_layers
            last_sliding_src = last_global_src = -1

        stored_kv: dict = {}
        kv_cache_out: dict = {}

        for i, attn_type in enumerate(cfg.attention_types):
            d = cfg.per_layer_input_dim
            per_layer_input = ple_all_flat[:, :, i * d:(i + 1) * d]

            is_kv_shared = num_kv_shared > 0 and i >= first_kv_shared_idx
            is_store_layer = i in (last_sliding_src, last_global_src)
            need_kv = is_store_layer or (return_kv_cache and not is_kv_shared)

            if is_kv_shared:
                src = last_global_src if attn_type == AttentionType.GLOBAL else last_sliding_src
                shared_kv = stored_kv[src]
            else:
                shared_kv = None

            block_result = self.layers[i](
                x, positions, per_layer_input,
                shared_kv=shared_kv, return_kv=need_kv)

            if need_kv:
                x, kv = block_result
                if is_store_layer:
                    stored_kv[i] = kv
                if return_kv_cache and not is_kv_shared:
                    kv_cache_out[i] = kv
            else:
                x = block_result

        # Final norm
        x = self.norm(x)

        # Output projection (logits) via weight-tied embedding table
        logits = jnp.dot(x, embedding_table.T)

        if cfg.final_logit_softcap is not None:
            cap = cfg.final_logit_softcap
            logits = jnp.tanh(logits / cap) * cap

        if return_kv_cache:
            flat_kv = []
            for i in range(first_kv_shared_idx):
                k, v = kv_cache_out[i]
                flat_kv.append(k)
                flat_kv.append(v)
            return (logits,) + tuple(flat_kv)

        return logits
