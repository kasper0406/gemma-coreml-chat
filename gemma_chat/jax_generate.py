"""Autoregressive token stream using JAX reference decode (decode_jax)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

from gemma_chat.config import MAX_SEQ_LEN
from gemma_chat.decode_jax import decode_step, prefill
from gemma_chat.generate import sample_next_token, truncate_prompt_ids
from gemma_chat.model import Gemma4Config


def generate_jax_stream(
    prompt_ids: list[int],
    params: dict[str, Any],
    cfg: Gemma4Config,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Iterator[int]:
    """Yield generated token IDs (same contract as ``generate_kvcached``)."""
    eos_id = tokenizer.eos_token_id
    prompt_ids = truncate_prompt_ids(
        list(prompt_ids),
        max_seq_len,
        reserve_for_generation=max_new_tokens,
    )
    n_real = len(prompt_ids)
    tokens = jnp.array([prompt_ids], dtype=jnp.int32)
    logits, kv = prefill(params, tokens, cfg, max_seq_len)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)

    max_steps = min(max_new_tokens, max_seq_len - n_real)
    for _ in range(max_steps):
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        yield next_id
        if next_id == eos_id:
            break
        logits, kv = decode_step(params, next_id, kv, cfg)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
