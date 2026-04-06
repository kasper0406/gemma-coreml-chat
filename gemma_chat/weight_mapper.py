"""Load Gemma4-E2B weights from HuggingFace and map them into the model.

Provides two loading paths:
  1. ``build_flax_params(hf)`` → nested dict consumed by ``decode_coreml.py``
     (pure-JAX tracing path that operates on raw param dicts).
  2. ``load_params_into_model(model, params)`` → assigns dict values into an NNX
     ``Gemma4Transformer`` instance so it can be called directly.

Key observations from inspecting google/gemma-4-E2B-it model.safetensors:
  - All LM weights live under  model.language_model.*
  - Token embedding:          model.language_model.embed_tokens.weight        [vocab, embed]
  - PLE (packed):             model.language_model.embed_tokens_per_layer.weight  [vocab, 35*256]
  - PLE model projection:     model.language_model.per_layer_model_projection.weight [35*256, embed] → transpose
  - PLE projection norm:      model.language_model.per_layer_projection_norm.weight  [256]
  - Per layer (prefix lm.layers.{i}.):
      input_layernorm.weight                   [embed]
      self_attn.{q,k,v,o}_proj.weight          HF = [out, in]; NNX Linear needs [in, out] → transpose
      self_attn.{q,k}_norm.weight              [head_dim]
      post_attention_layernorm.weight          [embed]
      pre_feedforward_layernorm.weight         [embed]
      mlp.{gate,up}_proj.weight               [hidden, embed] → transpose
      mlp.down_proj.weight                    [embed, hidden] → transpose
      post_feedforward_layernorm.weight        [embed]
      per_layer_projection.weight             [embed, ple_dim] → transpose → [ple_dim, embed]
      per_layer_input_gate.weight             [ple_dim, embed] → transpose → [embed, ple_dim]
      post_per_layer_input_norm.weight        [embed]
      layer_scalar                            [1]
  - Final norm:               model.language_model.norm.weight               [embed]

HF Linear kernels are stored as [out_features, in_features].
NNX Linear kernels are [in_features, out_features].  All Linear weights are transposed.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from gemma_chat.config import E2B_CONFIG, HF_MODEL_ID


def _load_hf_tensors(model_id: str, token: str | None = None,
                     prefix: str = "model.language_model",
                     skip_keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Download (or load from HF cache) safetensors and return flat dict.

    Only loads tensors whose key starts with ``prefix`` to avoid loading
    vision/audio tower weights (the full file is ~9.5 GB but we only need
    the ~5 GB language-model slice).  Pass ``skip_keys`` substrings to
    further prune tensors.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import os, glob

    print(f"Downloading / loading weights for {model_id} …")
    kwargs = dict(
        repo_id=model_id,
        ignore_patterns=["*.bin", "*.pt", "flax_model*", "*.msgpack",
                         "tokenizer*", "*.json", "*.txt", "*.md",
                         "*.jinja"],
    )
    if token:
        kwargs["token"] = token
    local_dir = snapshot_download(**kwargs)

    shards = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"No .safetensors files found in {local_dir}. "
            "Make sure you have accepted the model license on HuggingFace."
        )

    skip_keys = skip_keys or []
    tensors: dict[str, np.ndarray] = {}
    for shard in shards:
        # Import ml_dtypes so numpy can handle bfloat16 tensors
        import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
        with safe_open(os.path.realpath(shard), framework="numpy", device="cpu") as f:
            keys = [
                k for k in f.keys()
                if k.startswith(prefix)
                and not any(s in k for s in skip_keys)
            ]
            for k in keys:
                tensors[k] = f.get_tensor(k)

    print(f"  Loaded {len(tensors)} tensors from {len(shards)} shard(s).")
    return tensors


def _t(arr: np.ndarray) -> np.ndarray:
    """Transpose last two dims: HF Linear [out, in] → Flax Dense [in, out]."""
    return np.swapaxes(arr, -2, -1)


def build_flax_params(hf: dict[str, np.ndarray], config=E2B_CONFIG) -> dict[str, Any]:
    """Map HF tensors to the nested Flax param dict for Gemma4Transformer."""

    lm = "model.language_model"   # shorthand prefix
    params: dict[str, Any] = {}

    # ── Token embedding ────────────────────────────────────────────────────
    params["embed_tokens"] = hf[f"{lm}.embed_tokens.weight"]

    # ── Packed PLE embedding ───────────────────────────────────────────────
    params["embed_tokens_per_layer"] = hf[f"{lm}.embed_tokens_per_layer.weight"]
    params["per_layer_model_projection"] = {
        "kernel": _t(hf[f"{lm}.per_layer_model_projection.weight"])
    }
    params["per_layer_projection_norm"] = {
        "scale": hf[f"{lm}.per_layer_projection_norm.weight"]
    }

    # ── Transformer layers ─────────────────────────────────────────────────
    for i in range(config.num_layers):
        p = f"{lm}.layers.{i}"
        layer: dict[str, Any] = {}

        # -- Attention ---------------------------------------------------
        layer["self_attn"] = {
            "q_proj": {"kernel": _t(hf[f"{p}.self_attn.q_proj.weight"])},
            "k_proj": {"kernel": _t(hf[f"{p}.self_attn.k_proj.weight"])},
            "v_proj": {"kernel": _t(hf[f"{p}.self_attn.v_proj.weight"])},
            "o_proj": {"kernel": _t(hf[f"{p}.self_attn.o_proj.weight"])},
            "q_norm": {"scale": hf[f"{p}.self_attn.q_norm.weight"]},
            "k_norm": {"scale": hf[f"{p}.self_attn.k_norm.weight"]},
        }

        # -- Layer norms -------------------------------------------------
        layer["input_layernorm"]           = {"scale": hf[f"{p}.input_layernorm.weight"]}
        layer["post_attention_layernorm"]  = {"scale": hf[f"{p}.post_attention_layernorm.weight"]}
        layer["pre_feedforward_layernorm"] = {"scale": hf[f"{p}.pre_feedforward_layernorm.weight"]}
        layer["post_feedforward_layernorm"]= {"scale": hf[f"{p}.post_feedforward_layernorm.weight"]}

        # -- MLP (SwiGLU) ------------------------------------------------
        layer["mlp"] = {
            "gate_proj": {"kernel": _t(hf[f"{p}.mlp.gate_proj.weight"])},
            "up_proj":   {"kernel": _t(hf[f"{p}.mlp.up_proj.weight"])},
            "down_proj": {"kernel": _t(hf[f"{p}.mlp.down_proj.weight"])},
        }

        # -- PLE gate, projection, post-norm -----------------------------
        layer["per_layer_input_gate"]    = {"kernel": _t(hf[f"{p}.per_layer_input_gate.weight"])}
        layer["per_layer_projection"]    = {"kernel": _t(hf[f"{p}.per_layer_projection.weight"])}
        layer["post_per_layer_input_norm"] = {"scale": hf[f"{p}.post_per_layer_input_norm.weight"]}

        # -- Layer scalar ------------------------------------------------
        layer["layer_scalar"] = hf[f"{p}.layer_scalar"]

        params[f"layers.{i}"] = layer

    # ── Final norm ─────────────────────────────────────────────────────────
    params["norm"] = {"scale": hf[f"{lm}.norm.weight"]}

    return params


def load_params(model_id: str = HF_MODEL_ID, config=E2B_CONFIG,
                token: str | None = None) -> dict[str, Any]:
    """Download HF weights and return the nested param dict.

    The dict is consumed by ``decode_coreml.py`` (pure-JAX tracing path) and
    by ``load_params_into_model()`` (NNX model loading).
    """
    import os
    hf_token = token or os.environ.get("HF_TOKEN")
    extra_skip: list[str] = []
    hf = _load_hf_tensors(model_id, token=hf_token, skip_keys=extra_skip)
    print("Building param tree …")
    params = build_flax_params(hf, config)
    print("  Done.")
    return params


def load_params_into_model(model, params: dict[str, Any],
                           config=E2B_CONFIG) -> None:
    """Assign weight values from a param dict into an NNX Gemma4Transformer.

    ``params`` is the dict returned by ``build_flax_params()``.  Values are
    assigned directly to the NNX module's ``[...]`` attributes; this is
    zero-copy when both sides share the same backing array type.
    """
    import jax.numpy as jnp

    # Top-level embeddings and projections
    model.embed_tokens[...] = jnp.asarray(params["embed_tokens"])
    model.embed_tokens_per_layer[...] = jnp.asarray(
        params["embed_tokens_per_layer"])
    model.per_layer_model_projection.kernel[...] = jnp.asarray(
        params["per_layer_model_projection"]["kernel"])
    model.per_layer_projection_norm.scale[...] = jnp.asarray(
        params["per_layer_projection_norm"]["scale"])

    # Per-layer weights
    for i in range(config.num_layers):
        lp = params[f"layers.{i}"]
        layer = model.layers[i]

        # Attention projections
        sa = lp["self_attn"]
        layer.self_attn.q_proj.kernel[...] = jnp.asarray(sa["q_proj"]["kernel"])
        layer.self_attn.k_proj.kernel[...] = jnp.asarray(sa["k_proj"]["kernel"])
        layer.self_attn.v_proj.kernel[...] = jnp.asarray(sa["v_proj"]["kernel"])
        layer.self_attn.o_proj.kernel[...] = jnp.asarray(sa["o_proj"]["kernel"])
        layer.self_attn.q_norm.scale[...] = jnp.asarray(sa["q_norm"]["scale"])
        layer.self_attn.k_norm.scale[...] = jnp.asarray(sa["k_norm"]["scale"])

        # Layer norms
        layer.input_layernorm.scale[...] = jnp.asarray(
            lp["input_layernorm"]["scale"])
        layer.post_attention_layernorm.scale[...] = jnp.asarray(
            lp["post_attention_layernorm"]["scale"])
        layer.pre_feedforward_layernorm.scale[...] = jnp.asarray(
            lp["pre_feedforward_layernorm"]["scale"])
        layer.post_feedforward_layernorm.scale[...] = jnp.asarray(
            lp["post_feedforward_layernorm"]["scale"])

        # MLP
        mlp = lp["mlp"]
        layer.mlp.gate_proj.kernel[...] = jnp.asarray(mlp["gate_proj"]["kernel"])
        layer.mlp.up_proj.kernel[...] = jnp.asarray(mlp["up_proj"]["kernel"])
        layer.mlp.down_proj.kernel[...] = jnp.asarray(mlp["down_proj"]["kernel"])

        # PLE gate, projection, post-norm
        layer.per_layer_input_gate.kernel[...] = jnp.asarray(
            lp["per_layer_input_gate"]["kernel"])
        layer.per_layer_projection.kernel[...] = jnp.asarray(
            lp["per_layer_projection"]["kernel"])
        layer.post_per_layer_input_norm.scale[...] = jnp.asarray(
            lp["post_per_layer_input_norm"]["scale"])

        # Layer scalar
        layer.layer_scalar[...] = jnp.asarray(lp["layer_scalar"])

    # Final norm
    model.norm.scale[...] = jnp.asarray(params["norm"]["scale"])


def inspect_hf_keys(model_id: str = HF_MODEL_ID) -> None:
    """Utility: print all HF tensor names and shapes."""
    import os
    hf = _load_hf_tensors(model_id, token=os.environ.get("HF_TOKEN"))
    for name, arr in sorted(hf.items()):
        print(f"  {name:80s}  {arr.shape}  {arr.dtype}")
