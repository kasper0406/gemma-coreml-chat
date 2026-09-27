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
from gemma_chat.config import E2B_CONFIG
from gemma_chat.decode_coreml import _sliding_ring_write, decode_step, empty_pos_ring
from gemma_chat.model import AttentionType, Gemma4Config, _embed_lookup


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
        token = jnp.full((1, 1), (position * 7 + 3) % cfg.num_embed, jnp.int32)
        # The rows the host looks up (see ``gemma_chat.host_embeddings``).
        token_embed = _embed_lookup(params["embed_tokens"], token) \
            * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)
        ple_rows = _embed_lookup(params["embed_tokens_per_layer"], token)
        logits, kv, ring = decode_step(
            params, token_embed, ple_rows, jnp.int32(position), kv, ring, cfg=cfg,
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


# ── 3. Layer chunks compose to the whole model ────────────────────────────


def _kv_shared_config() -> Gemma4Config:
    """7 layers, the last two KV-shared (reading layers 3 and 4), window 8."""
    S, G = AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL
    return dataclasses.replace(
        Gemma4Config(), attention_types=(S, S, G, S, G, S, G),
        num_kv_shared_layers=2, sliding_window_size=8,
    )


def test_layer_chunks_cover_the_model_and_share_the_right_caches(monkeypatch):
    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", (0, 3, 5))
    chunks = decode_coreml.layer_chunks(_kv_shared_config())
    assert [(c.layers.start, c.layers.stop) for c in chunks] == [(0, 3), (3, 5), (5, 7)]
    assert [c.writes for c in chunks] == [(0, 1, 2), (3, 4), ()]
    assert [c.reads for c in chunks] == [(), (), (3, 4)]


def test_a_chunk_without_a_global_layer_joins_the_one_before(monkeypatch):
    """Every exported function but the head is one per cache size."""
    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", (0, 3, 4))
    chunks = decode_coreml.layer_chunks(_kv_shared_config())
    # Layer 3 alone is sliding-only.
    assert [(c.layers.start, c.layers.stop) for c in chunks] == [(0, 4), (4, 7)]


def test_the_shipped_chunks_all_hold_a_global_layer():
    chunks = decode_coreml.layer_chunks(E2B_CONFIG)
    assert [i for c in chunks for i in c.layers] == list(range(E2B_CONFIG.num_layers))
    for c in chunks:
        assert any(E2B_CONFIG.attention_types[i] == AttentionType.GLOBAL for i in c.layers)


def test_chunked_decode_matches_a_single_chunk(monkeypatch):
    """Splitting the layers — with the KV-shared ones in a chunk of their own,
    reading caches an earlier chunk wrote — changes nothing."""
    cfg = _kv_shared_config()
    params = _tiny_params(cfg)
    one_logits, one_kv, one_ring = _run_decode(params, cfg, 24, 12)
    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", (0, 3, 5))
    three_logits, three_kv, three_ring = _run_decode(params, cfg, 24, 12)
    for a, b in zip(one_logits, three_logits):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    for a, b in zip(one_kv, three_kv):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-3)
    np.testing.assert_array_equal(one_ring, three_ring)


# ── 4. The export-side signatures ──────────────────────────────────────────


def test_chunk_io_plan_maps_every_argument_and_result():
    from gemma_chat.export import _chunk_io_plan

    cfg = _kv_shared_config()
    specs = build_cache_specs(cfg, 24)
    is_global = {s: spec.attn_type == AttentionType.GLOBAL for s, spec in enumerate(specs)}
    for k, chunk in enumerate(decode_coreml.layer_chunks(cfg)):
        plan = _chunk_io_plan(cfg, chunk, first=k == 0, tokens=1, N=24)
        leading = (["hidden"] if k else []) + ["token_embed", "ple_rows", "position"]
        base = 1 + len(leading)  # N first: every chunk holds a global layer
        assert plan.has_global
        assert plan.input_names[:base] == ["N"] + leading

        # Traced results: [hidden_out] + k/v of every written slot.
        results = ["hidden_out"] + [f"{p}_{s}_out" for s in chunk.writes for p in "kv"]
        for j, slot in enumerate(chunk.slots):
            for half, prefix in enumerate("kv"):
                name, arg = f"{prefix}_{slot}", base + 2 * j + half
                written = results.index(f"{name}_out") if slot in chunk.writes else None
                if is_global[slot]:
                    assert arg not in plan.states and name in plan.input_names
                    assert name in plan.flexible
                else:
                    assert plan.states[arg].name == name
                    assert plan.states[arg].output == written
        state_outputs = {spec.output for spec in plan.states.values()} - {None}
        assert plan.output_names == [r for i, r in enumerate(results) if i not in state_outputs]
        assert plan.input_names[-1] == "sliding_pos_ring"
        # Every traced argument is either state or a named input (``N`` is
        # JAX's own, not in ``arg_specs``).
        assert len(plan.arg_specs) + 1 == len(plan.input_names) + len(plan.states)


def test_state_io_plan_declares_every_cache_read_only():
    from gemma_chat.export import _state_io_plan

    cfg = _kv_shared_config()
    plan = _state_io_plan(cfg, N=24)
    specs = build_cache_specs(cfg, 24)
    sliding = [s for s, spec in enumerate(specs) if spec.attn_type == AttentionType.LOCAL_SLIDING]
    assert sorted(spec.name for spec in plan.states.values()) == sorted(
        f"{p}_{s}" for s in sliding for p in "kv"
    )
    assert all(spec.output is None for spec in plan.states.values())
    assert plan.input_names == ["N"] + [
        f"{p}_{s}" for s in range(len(specs)) if s not in sliding for p in "kv"
    ]
