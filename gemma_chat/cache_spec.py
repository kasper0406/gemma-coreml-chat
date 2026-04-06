"""KV-cache layout for Gemma4-E2B incremental decoding.

Due to KV-sharing, only **15** caches are needed for the 35-layer E2B model:
  - 12 sliding ring-buffer caches (layers 0-3, 5-8, 10-13)  shape (1, win, nkv, hd_s)
  - 3  global growing caches      (layers 4, 9, 14)           shape (1, Lmax, nkv, hd_g)

Layers 15-34 are KV-shared: they read from layer 13 (sliding) or 14 (global) — no
separate cache entry.  Both k and v are stored post-RoPE, post-QK-norm,
pre-GQA-expansion, matching what `return_kv=True` returns from GemmaAttention.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from gemma_chat.config import E2B_CONFIG, MAX_SEQ_LEN
from gemma_chat.model import AttentionType, Gemma4Config


@dataclasses.dataclass(frozen=True)
class LayerCacheSpec:
    """Describes one layer's KV cache slot."""
    layer_idx: int
    attn_type: str          # AttentionType.LOCAL_SLIDING or GLOBAL
    cache_len: int          # sliding_window_size  OR  max_seq_len
    num_kv_heads: int
    head_dim: int


def build_cache_specs(
    cfg: Gemma4Config = E2B_CONFIG,
    max_seq_len: int = MAX_SEQ_LEN,
) -> List[LayerCacheSpec]:
    """Return one LayerCacheSpec for every layer that has its own KV cache."""
    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    specs = []
    for i, attn_type in enumerate(cfg.attention_types):
        if i >= kv_shared_start:
            break  # layers 15-34 are KV-shared; no cache of their own
        hd = cfg.effective_head_dim(attn_type)
        if attn_type == AttentionType.LOCAL_SLIDING:
            cache_len = cfg.sliding_window_size
        else:
            cache_len = max_seq_len
        specs.append(LayerCacheSpec(
            layer_idx=i,
            attn_type=attn_type,
            cache_len=cache_len,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=hd,
        ))
    return specs


# Singleton cache specs for E2B (computed once at import time).
E2B_CACHE_SPECS: List[LayerCacheSpec] = build_cache_specs()


@dataclasses.dataclass
class KVState:
    """Mutable KV cache state for one forward pass.

    `caches`: dict mapping layer_idx → (k, v) arrays, each shape (1, cache_len, nkv, hd).
    `write_pos`: dict mapping layer_idx → int, the next write position (ring index for
        sliding, absolute token index for global).
    `seq_len`: total number of tokens processed so far (prompt + generated).
    """
    caches: Dict[int, Tuple[np.ndarray, np.ndarray]]
    write_pos: Dict[int, int]
    seq_len: int

    @classmethod
    def empty(
        cls,
        specs: List[LayerCacheSpec] | None = None,
        dtype=jnp.float32,
    ) -> "KVState":
        if specs is None:
            specs = E2B_CACHE_SPECS
        caches = {}
        write_pos = {}
        for s in specs:
            shape = (1, s.cache_len, s.num_kv_heads, s.head_dim)
            caches[s.layer_idx] = (
                np.zeros(shape, dtype=dtype),
                np.zeros(shape, dtype=dtype),
            )
            write_pos[s.layer_idx] = 0
        return cls(caches=caches, write_pos=write_pos, seq_len=0)


def kv_shared_sources(
    cfg: Gemma4Config = E2B_CONFIG,
) -> Dict[int, int]:
    """Return mapping: shared_layer_idx → source_layer_idx for KV-shared layers."""
    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    attn = list(cfg.attention_types)
    prev = attn[:kv_shared_start]

    last_sliding = kv_shared_start - 1 - list(reversed(prev)).index(AttentionType.LOCAL_SLIDING)
    last_global  = kv_shared_start - 1 - list(reversed(prev)).index(AttentionType.GLOBAL)

    result = {}
    for i in range(kv_shared_start, n):
        result[i] = last_global if attn[i] == AttentionType.GLOBAL else last_sliding
    return result


# Precomputed for E2B.
E2B_KV_SHARED_SOURCES: Dict[int, int] = kv_shared_sources()
