"""The sliding KV caches are CoreML state, so their write had to be
reformulated from ``jax.lax.dynamic_update_slice`` to a whole-tensor write (a
CoreML state update fed by ``slice_update`` does not persist on macOS 26): a
blend with the host's one-hot write selection (``decode_coreml._cache_write``).

These tests are pure JAX / pure Python — no CoreML models are built or loaded.
They pin down:

1. the ring write is *numerically* the old ``dynamic_update_slice`` write, at
   every position including ring wraparound, and ``decode_step`` as a whole is
   unchanged by the reformulation;
2. the layer chunks compose to the whole model;
3. ``export.py``'s state mapping points at the right traced argument and result
   indices, which is the part that would silently mis-wire an export;
4. the host's masks keep every row's sliding window exact, prefill and decode;
5. the host's RoPE rows are the reference model's, to one fp16 rounding.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma_chat import decode_coreml
from gemma_chat.cache_spec import build_cache_specs, sliding_ring_length
from gemma_chat.config import E2B_CONFIG
from gemma_chat.decode_coreml import (
    _cache_write, chunk_host_inputs, decode_step, empty_pos_ring, host_inputs,
)
from gemma_chat.model import AttentionType, Gemma4Config, _embed_lookup


def _onehot(slots, length):
    """The host's write selection ``(1, length, len(slots))``: row ``l`` of the
    step lands in cache row ``slots[l]``."""
    sel = np.arange(length)[:, None] == np.asarray(slots)[None, :]
    return jnp.asarray(sel.astype(np.float16)[None])


def _dus_ring_write(cache, value, position):
    """The write this replaced: an in-place slice update at the ring slot."""
    return jax.lax.dynamic_update_slice(
        cache, value, (0, position % cache.shape[1], 0, 0)
    )


def _dus_cache_write(cache, value, onehot):
    """``_cache_write`` spelled as the slice updates it replaced, one per row."""
    sel = np.asarray(onehot)[0]
    for row in range(sel.shape[1]):
        for slot in np.nonzero(sel[:, row])[0]:
            cache = jax.lax.dynamic_update_slice(
                cache, value[:, row:row + 1], (0, int(slot), 0, 0)
            )
    return cache


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
        got = _cache_write(cache, value, _onehot([position % window], window))
        want = _dus_ring_write(cache, value, pos)
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

    got = np.asarray(_cache_write(cache, value, _onehot([7 % window], window)))
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


def _empty_caches(cfg, max_seq_len):
    """Zeroed ``{slot: (k, v)}`` in the exported layout: sliding caches are
    ``sliding_ring_length`` rows, global ones ``max_seq_len``."""
    caches = {}
    for slot, s in enumerate(build_cache_specs(cfg, max_seq_len)):
        rows = sliding_ring_length(cfg) if s.attn_type == AttentionType.LOCAL_SLIDING else s.cache_len
        shape = (1, rows, s.num_kv_heads, s.head_dim)
        caches[slot] = (jnp.zeros(shape, jnp.float16), jnp.zeros(shape, jnp.float16))
    return caches


def _host_rows(params, tokens, cfg):
    """The rows the host looks up (see ``gemma_chat.host_embeddings``)."""
    token_embed = _embed_lookup(params["embed_tokens"], tokens) \
        * jnp.sqrt(float(cfg.embed_dim)).astype(jnp.float16)
    return token_embed, _embed_lookup(params["embed_tokens_per_layer"], tokens)


def _run_decode(params, cfg, max_seq_len, steps):
    """Run `steps` decode steps, returning (logits list, caches, pos ring)."""
    kv = [c for pair in _empty_caches(cfg, max_seq_len).values() for c in pair]
    ring = empty_pos_ring(cfg)

    all_logits = []
    for position in range(steps):
        token = jnp.full((1, 1), (position * 7 + 3) % cfg.num_embed, jnp.int32)
        # decode_step builds the host inputs from the position and the ring.
        token_embed, ple_rows = _host_rows(params, token, cfg)
        logits, kv, ring = decode_step(
            params, token_embed, ple_rows, jnp.int32(position), kv, ring, cfg=cfg,
        )
        all_logits.append(np.asarray(logits))
    return all_logits, [np.asarray(c) for c in kv], np.asarray(ring)


@pytest.fixture
def small_ring(monkeypatch):
    """A 4-token prefill chunk, so the tiny models' rings are 8 + 4 rows and
    wrap within a few dozen steps."""
    from gemma_chat import cache_spec
    monkeypatch.setattr(cache_spec, "CHUNK_SIZE", 4)
    return 4


def test_chunk_write_matches_dynamic_update_slice():
    """A prefill chunk wrapping around the ring lands where the slice updates
    put its rows; slots no row claims keep their contents."""
    rng = np.random.default_rng(3)
    R, C, nkv, hd = 12, 4, 1, 2
    cache = jnp.asarray(rng.standard_normal((1, R, nkv, hd)).astype(np.float16))
    value = jnp.asarray(rng.standard_normal((1, C, nkv, hd)).astype(np.float16))
    slots = (10 + np.arange(C)) % R
    got = _cache_write(cache, value, _onehot(slots, R))
    np.testing.assert_array_equal(np.asarray(got),
                                  np.asarray(_dus_cache_write(cache, value, _onehot(slots, R))))
    # A global row past the end of the cache takes no slot.
    got = _cache_write(cache, value, _onehot([R - 2, R - 1, R, R + 1], R))
    want = np.asarray(cache).copy()
    want[0, R - 2:] = np.asarray(value)[0, :2]
    np.testing.assert_array_equal(np.asarray(got), want)


def test_decode_step_unchanged_by_the_reformulation(monkeypatch, small_ring):
    """Same logits and same caches as the dynamic_update_slice version.

    Runs long enough for the 12-row ring to wrap twice; the global cache
    is exercised at the same time.
    """
    cfg = _tiny_config()
    max_seq_len = 32
    steps = 30
    params = _tiny_params(cfg)

    new_logits, new_kv, new_ring = _run_decode(params, cfg, max_seq_len, steps)

    monkeypatch.setattr(decode_coreml, "_cache_write", _dus_cache_write)
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


def test_a_leading_chunk_without_a_global_layer_joins_the_one_after(monkeypatch):
    """A leading sliding-only chunk has no cache-size-dependent input, so it
    used to be exported as a size-less ``decode_c0`` next to ``decode_c1_<N>``
    — a layout the runtime cannot load.  Every chunk must hold a global layer."""
    from gemma_chat.export import _chunk_io_plan

    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", (0, 2, 3))
    cfg = _kv_shared_config()  # S S G S G S G
    chunks = decode_coreml.layer_chunks(cfg)
    assert [(c.layers.start, c.layers.stop) for c in chunks] == [(0, 3), (3, 7)]
    for k, chunk in enumerate(chunks):
        assert _chunk_io_plan(cfg, chunk, first=k == 0, tokens=1, N=24).has_global

    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", (0, 4))
    assert [(c.layers.start, c.layers.stop) for c in decode_coreml.layer_chunks(E2B_CONFIG)] \
        == [(0, E2B_CONFIG.num_layers)]


def test_a_model_without_a_global_layer_is_rejected():
    cfg = dataclasses.replace(
        Gemma4Config(), attention_types=(AttentionType.LOCAL_SLIDING,) * 3,
    )
    with pytest.raises(ValueError, match="no global-attention layer"):
        decode_coreml.layer_chunks(cfg)


def test_the_shipped_chunks_all_hold_a_global_layer():
    chunks = decode_coreml.layer_chunks(E2B_CONFIG)
    assert [i for c in chunks for i in c.layers] == list(range(E2B_CONFIG.num_layers))
    for c in chunks:
        assert any(E2B_CONFIG.attention_types[i] == AttentionType.GLOBAL for i in c.layers)


def test_chunked_decode_matches_a_single_chunk(monkeypatch, small_ring):
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
        host = list(chunk_host_inputs(chunk, cfg))
        leading = (["hidden"] if k else []) + ["token_embed", "ple_rows"] + host
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
        # The global cache length is dim 3 of the global mask, dim 1 of the
        # global write selection.
        if "mask_global" in host:
            assert plan.flexible["mask_global"] == 3
        if "write_global" in host:
            assert plan.flexible["write_global"] == 1
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


# ── 5. The sliding window is exact, prefill and decode alike ──────────────


class _Runtime:
    """The GemmaCore runtime's cache handling (``InferenceEngine`` /
    ``KVCacheState`` / ``CoreMLModel.grownToFit``) over the JAX chunk
    functions, for one conversation.

    * Prefill runs whole chunks from a chunk boundary: the final one is
      right-padded with token 0, and the padding is written into the caches
      and marked in the ring like any token.  Only the last real row's logits
      are kept.
    * Decode writes one position at a time — over the padding, first.
    * The global caches have one of ``sizes`` rows; outgrowing it copies them
      into zeroed caches of the next size (the first rows, as
      ``PredictionBuffer.copyPrefix`` does) and keeps the sliding caches and
      the ring unchanged.
    """

    def __init__(self, params, cfg, chunk_size, sizes):
        self.params, self.cfg, self.chunk_size, self.sizes = params, cfg, chunk_size, sizes
        self.size = sizes[0]
        self.caches = _empty_caches(cfg, self.size)
        self.ring = empty_pos_ring(cfg)
        self.grown = []

    def grow_to_fit(self, needed):
        size = next(s for s in self.sizes if s >= needed)
        if size == self.size:
            return
        fresh = _empty_caches(self.cfg, size)
        for slot, pair in self.caches.items():
            rows = pair[0].shape[1]
            fresh[slot] = tuple(new.at[:, :rows].set(old) for old, new in zip(pair, fresh[slot]))
        self.caches, self.size = fresh, size
        self.grown.append(size)

    def _run(self, step, tokens, start):
        cfg, d = self.cfg, self.cfg.per_layer_input_dim
        positions = start + np.arange(tokens.shape[1])
        self.ring = decode_coreml.ring_with_positions(self.ring, jnp.asarray(positions))
        host = host_inputs(positions, np.asarray(self.ring), self.size, cfg)
        token_embed, ple_rows = _host_rows(self.params, tokens, cfg)
        hidden = token_embed
        for chunk in decode_coreml.layer_chunks(cfg):
            cols = ple_rows[:, :, chunk.layers.start * d:chunk.layers.stop * d]
            hidden, written = step(
                self.params, chunk, hidden, token_embed, cols,
                {n: jnp.asarray(host[n]) for n in chunk_host_inputs(chunk, cfg)},
                {s: self.caches[s] for s in chunk.slots}, cfg,
            )
            self.caches.update(written)
        return hidden

    def prefill(self, ids, offset):
        """``continuePrefill``: ``ids[:, offset:]`` from the chunk holding
        ``offset``; returns the last real token's logits."""
        C, n = self.chunk_size, ids.shape[1]
        padded_len = -(-n // C) * C
        padded = jnp.concatenate([ids, jnp.zeros((1, padded_len - n), jnp.int32)], axis=1)
        self.grow_to_fit(padded_len)
        for start in range(offset // C * C, padded_len, C):
            hidden = self._run(decode_coreml.prefill_chunk, padded[:, start:start + C], start)
        real = n - (padded_len - C)
        return decode_coreml.logits_head(self.params, hidden[:, real - 1:real]).reshape(-1)

    def decode(self, token, position):
        self.grow_to_fit(position + 1)
        hidden = self._run(decode_coreml.decode_chunk, token, position)
        return decode_coreml.logits_head(self.params, hidden).reshape(-1)


@pytest.mark.parametrize("starts", [(0,), (0, 3, 5)])
@pytest.mark.parametrize("prompt", [13, 15, 16])
def test_every_row_attends_exactly_its_window(monkeypatch, small_ring, starts, prompt):
    """A two-turn conversation through the runtime's cache handling against
    the reference model's full-sequence attention, far past the window.

    Turn one prefills ``prompt`` tokens (the final chunk holding 1, 3 or 4
    real rows), decodes over the padding and grows the global caches from 16
    to 32 rows mid-reply; turn two prefills from the chunk its new tokens
    start in and decodes on.  Every position the runtime produces logits for
    is compared.

    A prefill chunk is written into the sliding ring before its rows attend.
    With a ring of exactly ``window`` rows the chunk overwrote positions its
    own first rows still needed, so every row past the window attended to
    fewer than ``window`` positions (logits off by ~1 here).  With
    ``(0, 3, 5)`` the KV-shared layers sit in a later layer chunk and read the
    ring from state after the owning chunk has written it.
    """
    from flax import nnx
    from gemma_chat.model import Gemma4Transformer
    from gemma_chat.weight_mapper import load_params_into_model

    monkeypatch.setattr(decode_coreml, "LAYER_CHUNK_STARTS", starts)
    cfg = _kv_shared_config()  # window 8
    params = _tiny_params(cfg)
    reply, turn_two, reply_two = 9, 6, 5
    total = prompt + reply + turn_two + reply_two
    ids = jnp.asarray(
        np.random.default_rng(1).integers(1, cfg.num_embed, (1, total)), jnp.int32,
    )

    model = Gemma4Transformer(config=cfg, rngs=nnx.Rngs(params=0))
    load_params_into_model(model, params, cfg)
    want = np.asarray(model(ids)[0], np.float32)

    rt = _Runtime(params, cfg, small_ring, sizes=(16, 32, 64))
    got = {}

    def converse(prompt_end, offset, reply_len):
        got[prompt_end - 1] = rt.prefill(ids[:, :prompt_end], offset)
        for p in range(prompt_end, prompt_end + reply_len):
            got[p] = rt.decode(ids[:, p:p + 1], p)

    converse(prompt, 0, reply)
    assert rt.grown == [32], "turn one has to grow the cache mid-reply"
    converse(prompt + reply + turn_two, prompt + reply, reply_two)

    positions = sorted(got)
    # The head's logits are raw; the host soft-caps them, as the model does.
    cap = cfg.final_logit_softcap
    raw = np.stack([np.asarray(got[p], np.float32) for p in positions])
    err = np.abs(cap * np.tanh(raw / cap) - want[positions]).max(axis=-1)
    assert err.max() < 0.1, f"max |logit error| at {positions}: {np.round(err, 3)}"


# ── 6. RoPE rows ───────────────────────────────────────────────────────────


def test_host_rope_rows_are_the_reference_models():
    """``host_inputs``' cos | sin rows against ``model._apply_rope``'s own (fp32
    angles, cos/sin in fp32, cast to fp16), over the whole position range: the
    host rounds fp64 cos/sin of the same fp32 angle once, so the two agree but
    for the odd value that sits on an fp16 rounding boundary — at most one fp16
    step apart, and rarely."""
    from gemma_chat.decode_coreml import _rope_params
    from gemma_chat.model import rope_angles

    cfg = E2B_CONFIG
    positions = np.concatenate([
        np.arange(4096), np.random.default_rng(0).integers(4096, 65536, 4096),
    ])
    host = host_inputs(positions, empty_pos_ring(cfg), 128, cfg)
    for kind, frac in (("sliding", cfg.rope_fraction_sliding), ("global", cfg.rope_fraction_global)):
        base, hd, half = _rope_params(cfg, kind)
        angles = rope_angles(jnp.asarray(positions), base, hd, frac)
        want = np.concatenate([np.asarray(jnp.cos(angles).astype(jnp.float16)),
                               np.asarray(jnp.sin(angles).astype(jnp.float16))], axis=-1)
        got = host[f"rope_{kind}"][0, :, 0, :]
        assert got.shape == (len(positions), 2 * half)
        diff = np.abs(got.astype(np.float32) - want.astype(np.float32))
        step = np.spacing(np.abs(want)).astype(np.float32)
        assert (diff <= step).all(), f"{kind}: {np.max(diff / step)} fp16 steps apart"
        assert (diff > 0).mean() < 1e-3, f"{kind}: {(diff > 0).mean():.2e} of the values differ"
