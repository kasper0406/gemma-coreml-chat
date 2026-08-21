"""The sliding KV caches are CoreML state, so their decode-step write had to be
reformulated from ``jax.lax.dynamic_update_slice`` to a whole-tensor select
(a CoreML state update fed by ``slice_update`` does not persist on macOS 26).

These tests are pure JAX / pure Python — no CoreML models are built or loaded.
They pin down two things:

1. ``_sliding_ring_write`` is *numerically* the old ``dynamic_update_slice``
   write, at every position including ring wraparound, and ``decode_step`` as a
   whole is unchanged by the reformulation.
2. ``export.py``'s state mapping points at the right traced argument and result
   indices, which is the part that would silently mis-wire an export.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma_chat import decode_coreml
from gemma_chat.cache_spec import build_cache_specs
from gemma_chat.config import E2B_CONFIG, MAX_SEQ_LEN
from gemma_chat.decode_coreml import _sliding_ring_write, decode_step, empty_pos_ring
from gemma_chat.model import AttentionType, Gemma4Config


def _dus_ring_write(cache, value, position, window: int):
    """The write this replaced: an in-place slice update at the ring slot."""
    return jax.lax.dynamic_update_slice(
        cache, value, (0, position % window, 0, 0)
    )


# ── 1. The write itself ────────────────────────────────────────────────────


@pytest.mark.parametrize("window", [1, 3, 8])
def test_ring_write_matches_dynamic_update_slice(window):
    """Masked select == dynamic_update_slice, including ring wraparound."""
    rng = np.random.default_rng(0)
    nkv, hd = 2, 4
    cache = jnp.asarray(
        rng.standard_normal((1, window, nkv, hd)).astype(np.float16)
    )

    # Walk well past `window` so every slot wraps at least twice.
    for position in range(3 * window + 2):
        value = jnp.asarray(
            rng.standard_normal((1, 1, nkv, hd)).astype(np.float16)
        )
        pos = jnp.int32(position)
        got = _sliding_ring_write(cache, value, pos, window)
        want = _dus_ring_write(cache, value, pos, window)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert got.dtype == cache.dtype
        assert got.shape == cache.shape
        # Feed the update forward so later positions see a non-trivial cache.
        cache = got


def test_ring_write_only_touches_its_own_slot():
    window, nkv, hd = 5, 1, 2
    cache = jnp.arange(window * nkv * hd, dtype=jnp.float16).reshape(
        1, window, nkv, hd
    )
    value = jnp.full((1, 1, nkv, hd), -1.0, dtype=jnp.float16)

    got = np.asarray(_sliding_ring_write(cache, value, jnp.int32(7), window))
    expected = np.asarray(cache).copy()
    expected[0, 7 % window] = -1.0
    np.testing.assert_array_equal(got, expected)


# ── 2. decode_step end-to-end on a tiny random model ───────────────────────


def _tiny_config() -> Gemma4Config:
    """4 sliding + 1 global layer, window 8 so the ring wraps quickly."""
    return dataclasses.replace(Gemma4Config(), sliding_window_size=8)


def _tiny_params(cfg: Gemma4Config, seed: int = 0) -> dict:
    """Random float16 params shaped exactly like `load_params` output."""
    rng = np.random.default_rng(seed)

    def w(*shape):
        return jnp.asarray((0.05 * rng.standard_normal(shape)).astype(np.float16))

    def scale(n):
        return jnp.asarray((1.0 + 0.02 * rng.standard_normal((n,))).astype(np.float16))

    D = cfg.embed_dim
    d = cfg.per_layer_input_dim
    NL = cfg.num_layers
    V = cfg.num_embed

    params = {
        "embed_tokens": w(V, D),
        "embed_tokens_per_layer": w(V, NL * d),
        "per_layer_model_projection": {"kernel": w(D, NL * d)},
        "per_layer_projection_norm": {"scale": scale(d)},
        "norm": {"scale": scale(D)},
    }
    for i, attn_type in enumerate(cfg.attention_types):
        hd = cfg.effective_head_dim(attn_type)
        hidden = cfg.effective_hidden_dim(i)
        params[f"layers.{i}"] = {
            "input_layernorm": {"scale": scale(D)},
            "self_attn": {
                "q_proj": {"kernel": w(D, cfg.num_heads * hd)},
                "k_proj": {"kernel": w(D, cfg.num_kv_heads * hd)},
                "v_proj": {"kernel": w(D, cfg.num_kv_heads * hd)},
                "o_proj": {"kernel": w(cfg.num_heads * hd, D)},
                "q_norm": {"scale": scale(hd)},
                "k_norm": {"scale": scale(hd)},
            },
            "post_attention_layernorm": {"scale": scale(D)},
            "pre_feedforward_layernorm": {"scale": scale(D)},
            "mlp": {
                "gate_proj": {"kernel": w(D, hidden)},
                "up_proj": {"kernel": w(D, hidden)},
                "down_proj": {"kernel": w(hidden, D)},
            },
            "post_feedforward_layernorm": {"scale": scale(D)},
            "per_layer_input_gate": {"kernel": w(D, d)},
            "per_layer_projection": {"kernel": w(d, D)},
            "post_per_layer_input_norm": {"scale": scale(D)},
            "layer_scalar": jnp.asarray(np.float16(1.0)),
        }
    return params


def _run_decode(params, cfg, max_seq_len, steps):
    """Run `steps` decode steps, returning (logits list, caches, pos ring)."""
    specs = build_cache_specs(cfg, max_seq_len)
    kv = []
    for s in specs:
        shape = (1, s.cache_len, s.num_kv_heads, s.head_dim)
        kv.append(jnp.zeros(shape, dtype=jnp.float16))
        kv.append(jnp.zeros(shape, dtype=jnp.float16))
    ring = empty_pos_ring(cfg)

    all_logits = []
    for position in range(steps):
        token = jnp.int32((position * 7 + 3) % cfg.num_embed)
        logits, kv, ring = decode_step(
            params, token, jnp.int32(position), kv, ring, cfg=cfg,
        )
        all_logits.append(np.asarray(logits))
    return all_logits, [np.asarray(c) for c in kv], np.asarray(ring)


def test_decode_step_unchanged_by_the_reformulation(monkeypatch):
    """Same logits and same caches as the dynamic_update_slice version.

    Runs past the sliding window so the ring wraps twice; the global cache
    (which still uses dynamic_update_slice) is exercised at the same time.
    """
    cfg = _tiny_config()
    max_seq_len = 24
    steps = 20  # 2.5 wraps of the 8-slot ring
    params = _tiny_params(cfg)

    new_logits, new_kv, new_ring = _run_decode(params, cfg, max_seq_len, steps)

    monkeypatch.setattr(decode_coreml, "_sliding_ring_write", _dus_ring_write)
    old_logits, old_kv, old_ring = _run_decode(params, cfg, max_seq_len, steps)

    for step, (a, b) in enumerate(zip(new_logits, old_logits)):
        np.testing.assert_array_equal(a, b, err_msg=f"logits differ at step {step}")
    for slot, (a, b) in enumerate(zip(new_kv, old_kv)):
        np.testing.assert_array_equal(a, b, err_msg=f"cache {slot} differs")
    np.testing.assert_array_equal(new_ring, old_ring)

    # Sanity: the caches actually got written (a no-op write would also match).
    assert any(np.any(c != 0) for c in new_kv)


# ── 3. The export-side state mapping ───────────────────────────────────────


def test_kv_export_plan_indices_and_names():
    from gemma_chat.export import _kv_export_plan

    specs = build_cache_specs(E2B_CONFIG, MAX_SEQ_LEN)
    sliding = [i for i, s in enumerate(specs)
               if s.attn_type == AttentionType.LOCAL_SLIDING]
    globals_ = [i for i, s in enumerate(specs)
                if s.attn_type == AttentionType.GLOBAL]
    assert (len(sliding), len(globals_)) == (12, 3)

    states, kv_in, kv_out = _kv_export_plan(specs, has_global=True)

    # Traced args: [N, token, position] + kv_flat + [sliding_pos_ring]
    # Traced outs: [logits] + kv_flat_out + [sliding_pos_ring_out]
    base = 3
    n_args = base + 2 * len(specs) + 1
    n_outs = 1 + 2 * len(specs) + 1

    assert set(states) == {
        base + 2 * slot + half for slot in sliding for half in (0, 1)
    }
    for slot in sliding:
        k_spec = states[base + 2 * slot]
        v_spec = states[base + 2 * slot + 1]
        assert (k_spec.name, k_spec.output) == (f"k_{slot}", 1 + 2 * slot)
        assert (v_spec.name, v_spec.output) == (f"v_{slot}", 2 + 2 * slot)

    assert kv_in == [f"{p}_{slot}" for slot in globals_ for p in ("k", "v")]
    assert kv_out == [f"{n}_out" for n in kv_in]

    # Every argument is either state or a remaining input, exactly once.
    remaining_inputs = ["N", "token_id", "position"] + kv_in + ["sliding_pos_ring"]
    assert len(remaining_inputs) + len(states) == n_args
    # Same for outputs.
    remaining_outputs = ["logits"] + kv_out + ["sliding_pos_ring_out"]
    consumed = {spec.output for spec in states.values()}
    assert len(consumed) == len(states)
    assert len(remaining_outputs) + len(consumed) == n_outs
    assert 0 not in consumed and (n_outs - 1) not in consumed


def test_kv_export_plan_without_global_layers():
    """A truncated all-sliding export has no `N` argument, shifting the base."""
    from gemma_chat.export import _kv_export_plan

    cfg = dataclasses.replace(
        Gemma4Config(),
        attention_types=(AttentionType.LOCAL_SLIDING,) * 2,
    )
    specs = build_cache_specs(cfg, 32)
    states, kv_in, kv_out = _kv_export_plan(specs, has_global=False)

    assert kv_in == [] and kv_out == []
    assert set(states) == {2, 3, 4, 5}
    assert states[2].name == "k_0" and states[2].output == 1
    assert states[5].name == "v_1" and states[5].output == 4
