"""JAX-traceable layer chunks and logit head for CoreML export.

The exported model runs every step as a sequence of **layer chunks** — one
function per contiguous layer range (:func:`layer_chunks`), each taking the
hidden state from the one before — followed by the logit head
(:func:`logits_head`), a function of its own.  :func:`prefill_chunk` and
:func:`decode_chunk` are what ``export.py`` traces for each chunk; they close
over ``params`` and lower with ``jax.jit(...).trace(specs).lower()``.
:func:`decode_step` composes the pieces into the whole model the way the
runtime does, as a reference.

KV cache layout
---------------
Only 15 of the 35 layers store their own KV (layers 15-34 are KV-shared).

- **Sliding layers** (12 caches): ring-buffer shape ``(1, R, nkv, hd)`` with
  ``R = sliding_window_size + CHUNK_SIZE`` rows (``cache_spec.sliding_ring_length``).
  Slot index = ``position % R``.  A companion ``sliding_pos_ring`` array
  ``(1, R)`` int32 tracks which absolute position each slot holds (``-1`` =
  empty), and the attention mask admits a slot only if its position lies in
  the query's window, ``q - W < p <= q``.

  Why ``W + CHUNK_SIZE`` rows rather than ``W``: a prefill call writes its
  whole chunk into the ring before any of its rows attend.  With a ring of
  exactly ``W`` rows, the chunk at positions ``s .. s+C-1`` overwrote positions
  ``s-W .. s-W+C-1`` — history its *first* rows still need (row ``s`` attends
  back to ``s-W+1``) — so every prompt longer than the window lost up to
  ``C - 1`` positions of context per row.  With ``C`` more rows the chunk
  overwrites only ``s-W-C .. s-W-1``, which no row of the chunk can see, and a
  KV-shared layer in a later layer chunk still finds every position it needs
  in the state.  Decode is one row, so it never had the problem; it pays
  ``C/W`` = 25% more sliding keys for sharing the layout.
- **Global layers** (3 caches): linear shape ``(1, max_seq_len, nkv, hd)``.
  Slot index = absolute position.

Flat KV representation: ``[k0, v0, k1, v1, ..., k14, v14]`` — 30 arrays.
Layer order: 0, 1, 2, 3 (LOCAL_SLIDING), 4 (GLOBAL), 5-8, 9, 10-13, 14.

Core ML state
-------------
A chunk takes the caches of every slot it touches and returns the ones it
writes, but ``export.py`` binds the **sliding** ones to Core ML *state* (they
have static shapes, which Core ML states require) so they never cross the model
boundary at run time.  The global caches keep their symbolic dim-1 through
conversion and become state after materialization.  ``sliding_pos_ring`` is
int32 (states must be floating point): the host keeps it up to date
(:func:`ring_with_positions`) and the chunks only read it.

That is why every cache write below is a whole-tensor ``jnp.where`` rather than
``jax.lax.dynamic_update_slice``.  Two independent reasons, one per cache kind:

* *Correctness* (sliding): on macOS 26 a Core ML state write whose value comes
  from ``slice_update`` is applied to a freshly zeroed buffer instead of the
  persisted one, while a ``select``-shaped write persists correctly.
* *Speed* (global): ``slice_update`` with a runtime ``begin`` makes MPSGraph
  read that index back to the CPU mid-encode
  (``GPURegionRuntime::waitAndReadIntTensorData`` → ``waitUntilCompleted``),
  draining the pipeline once per write — 6 times per decode step, ~17 ms.  A
  select is one more whole-cache elementwise op (~0.02 ms at 512 tokens,
  ~2.8 ms at 65536) and never stalls.

Since materialization turns the global caches into state too (see
``mil_passes.global_cache_states``), the first reason now applies to them as
well: there is no ``slice_update`` left anywhere on a cache path.

Inputs: embeddings, not token ids
---------------------------------
Neither function sees a token id.  The two embedding lookups — the token row
and the per-layer-embedding (PLE) row — run on the host, which reads the
int4 block-32 tables the exporter ships next to the model (see
``gemma_chat.host_embeddings``), so the graph takes

* ``token_embed`` ``(1, L, embed_dim)`` fp16 — the token's embedding row,
  already multiplied by ``fp16(sqrt(embed_dim))`` in fp16, and
* ``ple_rows`` ``(1, L, len(chunk.layers) * per_layer_input_dim)`` fp16 — the
  chunk's own columns of the raw PLE row; the ``sqrt(per_layer_input_dim)``
  scaling, the projection of ``token_embed`` and its norm stay in the graph
  (:func:`_ple_from_rows`).

``L`` is 1 for decode and ``CHUNK_SIZE`` for prefill.  The two gathers were
the costliest ops of an ANE plan (~45% of its estimated cost, on the CPU, from
~1.5 GB of tables); a host lookup is one row per token.  The tied logit head
still reads ``params['embed_tokens']``, in :func:`logits_head`.

Prompts are right-padded: real tokens at positions 0..T-1, the rows of the pad
token (id 0) at T..L-1.

Numeric precision
-----------------
Every *stored* activation is fp16 — the residual stream, the attention
inputs/outputs, the KV caches, the MLP activations.  fp32 is used only where a
range genuinely needs it:

* **RMSNorm runs in fp16** (``_rmsnorm``), because the ANE has no fp32: fp32
  norm statistics used to pin ~70% of the graph's ops to the CPU.  The JAX
  function computes the statistics in fp32 — the exact definition — and the
  export turns every norm into one fp16 ``l2_norm`` + ``mul``
  (``stablehlo_coreml``'s ``fuse_rmsnorm``, then ``mil_passes.fp16_l2_norm``).
  A plain fp16 sum of squares would overflow (activations reach ``|x| ~ 1800``,
  ``sum x^2 ~ 7e6``); ``l2_norm`` is range-safe on CPU, GPU and ANE, measured —
  see that pass for the numbers and the ANE's one deviation (it ignores eps,
  which only matters for rows no real activation comes near).  ``l2_norm``
  reduces over the last three axes, so a norm's rows are viewed as
  ``(rows, 1, 1, d)``; decode lays q out as ``(H, 1, 1, hd)`` so that costs it
  no reshape.
* **RoPE angles** (``model._apply_rope``): positions run to 65535, which fp16
  cannot represent exactly; the sinusoid argument, ``sin`` and ``cos`` are
  computed in fp32 and cast to fp16 before the rotation itself.
* **The logits**, from the output matmul onward: fp16 dot against the fp16
  embedding table, fp32 out (upcasting the *weight* instead would put a 1.6 GB
  fp32 constant in the graph — see :func:`logits_head`).

(The prefill ring update was an fp32 scatter for the same range reason as the
RoPE angles; it is gone from the graph now that the host keeps the ring.)

Everything else stays fp16 end to end, which matters because the decode graph is
dispatch-bound: any op that returns fp32 drags its consumers up with it.  A
single fp32 leak in ``_apply_rope`` used to promote q, and through it SDPA, the
attention output and o_proj — which forced the 35 o_proj weights to be
re-materialized as fp32 constants at runtime (528 MB) and put ~1 GB of fp32
attention intermediates in every step.  Long-axis sums inside `matmul` and
``softmax`` are left to the backend, which accumulates them in fp32.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, LAYER_CHUNK_STARTS
from gemma_chat.model import AttentionType, Gemma4Config, _apply_rope
from gemma_chat.cache_spec import kv_shared_sources, sliding_ring_length


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def empty_pos_ring(cfg: Gemma4Config = E2B_CONFIG) -> jnp.ndarray:
    """Return (1, sliding_ring_length) int32 filled with -1 (no entries)."""
    return jnp.full((1, sliding_ring_length(cfg)), -1, dtype=jnp.int32)


def _sliding_mask(pos_ring, pos_q, window: int):
    """Which ring slots each query may attend: ``(len(pos_q), R)`` bool.

    A slot holds absolute position ``p = pos_ring[0, slot]`` (``-1`` when
    empty); query ``q`` sees it iff ``q - window < p <= q`` — the reference
    model's causal sliding window (``model.GemmaAttention``).  The ring holds
    more than ``window`` positions (see the module docstring), so the lower
    bound is what keeps the span exact.
    """
    pk = pos_ring[0][jnp.newaxis, :]    # (1, R)
    q = pos_q[:, jnp.newaxis]           # (Q, 1)
    return (pk >= 0) & (pk <= q) & (pk > q - window)


def _row_write(cache, value, slot):
    """Write the single row ``value`` into row ``slot`` of ``cache``.

    ``cache`` is ``(1, L, nkv, hd)``, ``value`` is ``(1, 1, nkv, hd)`` and
    broadcasts across the length axis; the mask selects the one live row.
    Equivalent to ``jax.lax.dynamic_update_slice(cache, value, (0, slot, 0, 0))``
    but built from a whole-tensor select — see the module docstring.

    ``L`` may be a symbolic dimension (the global caches trace with one).
    """
    length = cache.shape[1]
    mask = (
        jnp.arange(length, dtype=jnp.int32) == slot
    )[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]  # (1, L, 1, 1)
    return jnp.where(mask, value, cache)


def _sliding_ring_write(cache, value, position):
    """Write ``value`` into ring slot ``position % R`` of the ``R``-row ``cache``."""
    return _row_write(cache, value, position % cache.shape[1])


def _chunk_write(cache, value, slots):
    """Scatter the C rows of ``value`` into rows ``slots`` of ``cache``.

    ``cache`` is ``(1, L, nkv, hd)``, ``value`` ``(1, C, nkv, hd)`` and ``slots``
    ``(C,)`` int32 — the destination row of each chunk token.  The one-hot
    matmul places every row at once and the select keeps the rows no token
    claimed, so the whole thing is one whole-tensor write — see the module
    docstring.

    Rows outside ``0 .. L-1`` are simply dropped, where the
    ``dynamic_update_slice`` this replaced would have clamped the whole block
    back inside and written it at the wrong offset.
    """
    length = cache.shape[1]
    # write_mask[l, c] = True iff token c belongs in row l.
    write_mask = (
        jnp.arange(length, dtype=jnp.int32)[:, None] == slots[None, :]
    )  # (L, C)
    gathered = jnp.einsum('lc,bchd->blhd', write_mask.astype(value.dtype), value)
    any_written = write_mask.any(axis=1)[None, :, None, None]  # (1, L, 1, 1)
    return jnp.where(any_written, gathered, cache)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_INV_SQRT2 = float(1.0 / math.sqrt(2.0))


def _gelu_exact(x):
    """Exact GELU in ``x``'s own dtype, spelled so the converter emits one op.

    ``jax.nn.gelu(x, approximate=False)`` writes this as ``0.5 * x * erfc(-x/√2)``,
    and stablehlo-coreml's ``chlo.erfc`` handler builds ``1 - erf(...)`` from an
    fp32 Python literal, which fails to type-check against an fp16 operand.  The
    algebraically identical ``0.5 * x * (1 + erf(x/√2))`` spelling goes through
    the ``chlo.erf`` handler — a bare ``mb.erf`` — and is exactly the form
    coremltools' ``fuse_gelu_exact`` matches, so the chain collapses to a single
    native ``gelu(mode="EXACT")``.

    That op runs in the activation dtype, which is why this is called on fp16
    directly.  The fp32 round-trip it replaces existed to keep the erf
    *polynomial* out of fp16; there is no polynomial left to protect once the
    whole thing is one op, and the cast pair around it cost two dispatches per
    GELU (140 in the full decode graph).  The cancellation in ``1 + erf`` for
    very negative ``x`` never reaches the runtime either — the fused op computes
    the tail itself.
    """
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x * _INV_SQRT2))


_RMSNORM_EPS = 1e-6


def _rmsnorm(x, scale=None):
    """RMSNorm over the last axis, fp16 in and out — see the module docstring.

    Written with fp32 statistics, the exact definition; what the export runs is
    an fp16 ``l2_norm`` (``mil_passes.fp16_l2_norm``).  The row is viewed as
    ``(rows, 1, 1, d)`` because that is the only shape ``fuse_rmsnorm`` turns
    into ``l2_norm``: the op reduces over the last three axes.  Call sites that
    already have that shape (the decode ones) cost no reshape.
    """
    shape = x.shape
    d = shape[-1]
    canonical = x.ndim >= 3 and shape[-2] == 1 and shape[-3] == 1
    x32 = (x if canonical else x.reshape(-1, 1, 1, d)).astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + _RMSNORM_EPS)
    if scale is not None:
        y = y * scale.astype(jnp.float32)
    return y.astype(jnp.float16).reshape(shape)


def _ple_from_rows(params, token_embed, ple_rows, layers: range, cfg: Gemma4Config):
    """Per-layer inputs of ``layers`` from the host-gathered embedding rows.

    token_embed: (B, L, D) fp16, already multiplied by sqrt(D).
    ple_rows:    (B, L, len(layers) * per_layer_input_dim) fp16 — the columns
                 of the raw table rows that belong to ``layers``.

    The projection of ``token_embed`` is sliced to the same columns; its norm
    is per layer (over ``per_layer_input_dim``), so a slice computes exactly
    what the full projection would for those layers.

    Returns (B, L, len(layers) * per_layer_input_dim).
    """
    B, L = token_embed.shape[:2]
    d = cfg.per_layer_input_dim
    n = len(layers)

    ple_embed = ple_rows * jnp.sqrt(float(d)).astype(ple_rows.dtype)

    # numpy slice at trace time: only these columns become a graph constant.
    W_proj = np.asarray(params['per_layer_model_projection']['kernel'])
    W_proj = W_proj[:, layers.start * d:layers.stop * d]              # (D, n*d)
    ple_proj = jnp.dot(token_embed, W_proj) * (cfg.embed_dim ** -0.5)  # (B, L, n*d)

    scale = params['per_layer_projection_norm']['scale']   # (d,)
    ple_proj = _rmsnorm(ple_proj.reshape(B, L, n, d), scale)
    ple_proj = ple_proj.reshape(B, L, n * d)

    return (ple_proj + ple_embed) * (2.0 ** -0.5)


def _ffn(lp, x, hidden_dim: int):
    """GeLU-gated FFN. x: (..., D) → (..., D)."""
    gate = jnp.dot(x, lp['mlp']['gate_proj']['kernel'])
    up   = jnp.dot(x, lp['mlp']['up_proj']['kernel'])
    gate = _gelu_exact(gate)
    return jnp.dot(gate * up, lp['mlp']['down_proj']['kernel'])


def _ple_gate(lp, x, ple_slice):
    """Per-layer input gate block. x: (..., D), ple_slice: (..., d) → (..., D)."""
    gate_proj = jnp.dot(x, lp['per_layer_input_gate']['kernel'])
    gate = _gelu_exact(gate_proj) * ple_slice
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
        cache_len = sliding_ring_length for sliding, max_seq_len for global.
    shared_kv: optional (k_cache, v_cache) from source layer; if given,
               this layer reads from source and does NOT update its own cache.
    pos_ring: (1, sliding_ring_length) int32 — absolute position stored
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

    # (H, 1, 1, hd): each head a row of the shape the fused norm wants (see
    # ``_rmsnorm``); RoPE broadcasts over the leading axis all the same.
    q = jnp.dot(x[0, 0], sa['q_proj']['kernel']).reshape(num_heads, 1, 1, hd)
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
        v_new = _rmsnorm(v_new)
        k_new = _apply_rope(k_new, pos_arr, base_freq, rope_frac)

        # k_new/v_new are already fp16 — the norms and `_apply_rope` are both
        # dtype-preserving — so they go straight into the fp16 caches.
        if is_sliding:
            k_updated = _sliding_ring_write(k_cache, k_new, position)
            v_updated = _sliding_ring_write(v_cache, v_new, position)
        else:
            # Global: linear write, row index = absolute position.
            k_updated = _row_write(k_cache, k_new, position)
            v_updated = _row_write(v_cache, v_new, position)
        k_full, v_full = k_updated, v_updated

    # Attention validity mask
    if is_sliding:
        # Ring-buffer mask: the slots whose position is in this token's window.
        valid = _sliding_mask(pos_ring, position[jnp.newaxis], cfg.sliding_window_size)[0]
    else:
        # Global linear mask: slot index == absolute position.
        valid = jnp.arange(max_len, dtype=jnp.int32) <= position

    # GQA without replicating the cache.  ``jnp.repeat(k, kv_rep, axis=2)``
    # would materialize kv_rep copies of the *entire* cache per layer per token
    # — (1, max_len, H, hd) fp16, which the exporter emits as expand_dims+tile:
    # ~75 MB/step at max_len 512 and ~805 MB/step at 65536, all of it pure
    # memory traffic.  Instead, group the query heads: head h attends to kv
    # head h // kv_rep, so reshaping q to (1, num_kv_heads, kv_rep, hd) makes
    # the batch dims line up with k/v as (1, num_kv_heads, max_len, hd) and the
    # matmul needs no broadcast at all.
    kv_rep = num_heads // num_kv_heads

    qg = q.reshape(1, num_kv_heads, kv_rep, hd)            # (1, G, R, hd)
    kt = jnp.transpose(k_full, (0, 2, 1, 3))               # (1, G, max_len, hd)
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qg, jnp.swapaxes(kt, -2, -1))           # (1, G, R, max_len)
    w = jnp.where(valid[jnp.newaxis, jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w, axis=-1)

    out = jnp.matmul(w, vt)                                 # (1, G, R, hd)
    out = out.reshape(1, 1, num_heads * hd)
    out = jnp.dot(out[0, 0], sa['o_proj']['kernel'])[None, None, :]
    return out, k_updated, v_updated



# ---------------------------------------------------------------------------
# Layer chunks: the unit every exported function covers
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LayerChunk:
    """A contiguous range of layers, exported as one function per phase and size.

    ``writes`` are the cache slots of the chunk's own layers (read and
    written); ``reads`` are the slots its KV-shared layers read but an earlier
    chunk owns.  A slot is the index of its layer (layers 0..14 own slots
    0..14), and the exported state features are named ``k_<slot>`` /
    ``v_<slot>``.
    """
    layers: range
    writes: Tuple[int, ...]
    reads: Tuple[int, ...]

    @property
    def slots(self) -> Tuple[int, ...]:
        """Every cache slot the chunk touches, ascending."""
        return tuple(sorted(self.writes + self.reads))


def layer_chunks(cfg: Gemma4Config) -> List[LayerChunk]:
    """Split ``cfg``'s layers at :data:`config.LAYER_CHUNK_STARTS`.

    A chunk without a global-attention layer would not depend on the cache
    size at all, and every exported function except ``head`` is one per size
    — so such a chunk is merged into the one before it, or, when it leads the
    model, into the one after it.  A model with no global layer at all cannot
    be exported.
    """
    n = cfg.num_layers
    kv_shared_start = n - cfg.num_kv_shared_layers
    sources = kv_shared_sources(cfg)
    starts = [s for s in LAYER_CHUNK_STARTS if 0 < s < n]
    bounds = [0] + starts + [n]
    ranges: List[range] = []
    first = 0  # start of the next chunk: a leading sliding-only range waits here
    for a, b in zip(bounds, bounds[1:]):
        has_global = any(
            cfg.attention_types[i] == AttentionType.GLOBAL for i in range(a, b)
        )
        if has_global:
            ranges.append(range(first, b))
        elif ranges:
            ranges[-1] = range(ranges[-1].start, b)
        else:
            continue
        first = b
    if not ranges:
        raise ValueError(
            "the model has no global-attention layer, so no layer chunk depends "
            "on the cache size; the export needs at least one"
        )

    chunks = []
    for layers in ranges:
        writes = tuple(i for i in layers if i < kv_shared_start)
        reads = tuple(sorted(
            {sources[i] for i in layers if i >= kv_shared_start} - set(writes)
        ))
        chunks.append(LayerChunk(layers=layers, writes=writes, reads=reads))
    return chunks


def ring_with_positions(ring, positions):
    """``sliding_pos_ring`` after tokens at ``positions`` enter the ring.

    The **host** does this before every call (``KVCacheState`` in GemmaCore):
    ring slot ``p % R`` (``R`` the ring's length) records that it now holds
    position ``p``.  The exported functions only read the ring, to build the
    sliding masks.  This is the reference the runtime has to match.
    """
    positions = jnp.atleast_1d(positions)
    return ring.at[0, positions % ring.shape[1]].set(positions)


def _layer(lp, i: int, x, ple_slice, attend, cfg: Gemma4Config):
    """One decoder layer around ``attend(x_ln) -> attn_out``."""
    residual = x
    x_ln = _rmsnorm(x, lp['input_layernorm']['scale'])
    attn_out = _rmsnorm(attend(x_ln), lp['post_attention_layernorm']['scale'])
    x = residual + attn_out

    residual = x
    x_ln = _rmsnorm(x, lp['pre_feedforward_layernorm']['scale'])
    ffn_out = _ffn(lp, x_ln, cfg.effective_hidden_dim(i))
    ffn_out = _rmsnorm(ffn_out, lp['post_feedforward_layernorm']['scale'])
    x = residual + ffn_out

    x = _ple_gate(lp, x, ple_slice)
    return x * lp['layer_scalar']


def _run_chunk(params, chunk: LayerChunk, hidden, token_embed, ple_rows, caches,
               attend_own, attend_shared, cfg: Gemma4Config):
    """The layers of ``chunk``; the final norm too if it is the last one.

    ``caches`` maps every slot in ``chunk.slots`` to its ``(k, v)``; returns
    the new hidden state and ``{slot: (k, v)}`` for ``chunk.writes``.
    """
    ple_all = _ple_from_rows(params, token_embed, ple_rows, chunk.layers, cfg)
    kv_shared_start = cfg.num_layers - cfg.num_kv_shared_layers
    sources = kv_shared_sources(cfg)
    d = cfg.per_layer_input_dim
    caches = dict(caches)

    x = hidden
    for j, i in enumerate(chunk.layers):
        attn_type = cfg.attention_types[i]
        ple_slice = ple_all[:, :, j * d:(j + 1) * d]
        if i >= kv_shared_start:
            src = caches[sources[i]]
            attend = lambda x_ln, lp=params[f'layers.{i}'], t=attn_type, src=src: \
                attend_shared(lp, x_ln, t, src)
        else:
            def attend(x_ln, lp=params[f'layers.{i}'], t=attn_type, i=i):
                out, k, v = attend_own(lp, x_ln, t, *caches[i])
                caches[i] = (k, v)
                return out
        x = _layer(params[f'layers.{i}'], i, x, ple_slice, attend, cfg)

    if chunk.layers.stop == cfg.num_layers:
        x = _rmsnorm(x, params['norm']['scale'])
    return x, {slot: caches[slot] for slot in chunk.writes}


def decode_chunk(params, chunk: LayerChunk, hidden, token_embed, ple_rows,
                 position, caches, sliding_pos_ring, cfg: Gemma4Config = E2B_CONFIG):
    """One decode step through the layers of ``chunk``.

    Args:
        hidden:      (1, 1, D) fp16 — the residual stream entering the chunk
                     (``token_embed`` itself for the first chunk).
        token_embed: (1, 1, D) fp16 — the token's embedding row × sqrt(D).
        ple_rows:    (1, 1, len(chunk.layers) * per_layer_input_dim) fp16 —
                     its raw per-layer-embedding columns for these layers.
        position:    () int32 — absolute position of the token.
        caches:      {slot: (k, v)} for every slot in ``chunk.slots``.
        sliding_pos_ring: (1, R) int32 — already updated for ``position`` by
                     the host (see :func:`ring_with_positions`).

    Returns ``(hidden (1, 1, D), {slot: (k, v)} for chunk.writes)``; the
    hidden state is final-normed when the chunk ends the model.
    """
    def own(lp, x_ln, attn_type, k, v):
        pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
        return _attn_decode(lp, x_ln, position, cfg, attn_type, k, v, pos_ring=pr)

    def shared(lp, x_ln, attn_type, src):
        pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
        out, _, _ = _attn_decode(lp, x_ln, position, cfg, attn_type, *src,
                                 shared_kv=src, pos_ring=pr)
        return out

    return _run_chunk(params, chunk, hidden, token_embed, ple_rows, caches,
                      own, shared, cfg)


def logits_head(params, hidden, cfg: Gemma4Config = E2B_CONFIG):
    """The tied logit head: final-normed hidden (1, 1, D) fp16 → (vocab,) fp32.

    Its own function in the export, shared by decode and prefill (the host
    picks the prefill row it needs).  fp16 matmul, fp32 only from the logits
    onward: upcasting the weight instead would put a [dim, vocab] *fp32*
    constant in the graph — 1.6 GB.  The export stores the weight as int8
    block-32 (see ``mil_passes/quantize_const_weights``).
    """
    logits = jnp.dot(hidden[0, 0], params['embed_tokens'].T).astype(jnp.float32)
    if cfg.final_logit_softcap is not None:
        cap = cfg.final_logit_softcap
        logits = jnp.tanh(logits / cap) * cap
    return logits


def decode_step(params, token_embed, ple_rows, position, kv_flat, sliding_pos_ring,
                cfg: Gemma4Config = E2B_CONFIG):
    """The whole model for one token, as the runtime composes it.

    Ring update (host), every layer chunk in order, then the head.  Only a
    reference for tests: the export traces the pieces.  ``kv_flat`` is
    ``[k_0, v_0, k_1, v_1, ...]``, one pair per cache slot.

    Returns ``(logits (vocab,), kv_flat_new, sliding_pos_ring_new)``.
    """
    ring = ring_with_positions(sliding_pos_ring, position)
    caches = {s: (kv_flat[2 * s], kv_flat[2 * s + 1]) for s in range(len(kv_flat) // 2)}
    d = cfg.per_layer_input_dim
    hidden = token_embed
    for chunk in layer_chunks(cfg):
        cols = ple_rows[:, :, chunk.layers.start * d:chunk.layers.stop * d]
        hidden, written = decode_chunk(
            params, chunk, hidden, token_embed, cols, position,
            {s: caches[s] for s in chunk.slots}, ring, cfg,
        )
        caches.update(written)
    logits = logits_head(params, hidden, cfg)
    kv_new = [c for s in sorted(caches) for c in caches[s]]
    return logits, kv_new, ring


# ---------------------------------------------------------------------------
# Chunked-prefill attention (for chunk_prefill_step)
# ---------------------------------------------------------------------------

def _attn_chunk(lp, x, positions, cfg: Gemma4Config, attn_type: str,
                k_cache, v_cache, pos_ring=None):
    """Chunk attention with KV cache read/write.

    x: (1, C, D)  — C = CHUNK_SIZE tokens
    positions: (1, C) int32  — absolute positions
    k_cache, v_cache: (1, cache_len, nkv, hd)
    pos_ring: (1, R) int32 — ring position tracker (required for sliding layers)

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
    v_new = _rmsnorm(v_new)

    q = _apply_rope(q, positions, base_freq, rope_frac)
    k_new = _apply_rope(k_new, positions, base_freq, rope_frac)

    abs_pos = positions[0]  # (C,)
    # Sliding layers wrap into the ring; global layers write at the absolute
    # position.  Either way one whole-tensor scatter (see module docstring).
    # The ring is W + C rows long, so the chunk overwrites nothing any of its
    # rows still attends to (see the module docstring).
    # k_new/v_new are already fp16 (the norms and `_apply_rope` preserve dtype).
    slots = abs_pos % max_len if is_sliding else abs_pos
    k_updated = _chunk_write(k_cache, k_new, slots)
    v_updated = _chunk_write(v_cache, v_new, slots)

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
        mask = _sliding_mask(pos_ring, pos_q, cfg.sliding_window_size)  # (C, R)
    else:
        # Global linear mask.
        pos_k = jnp.arange(max_len, dtype=jnp.int32)
        mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]  # (C, max_len)

    w = jnp.where(mask[jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w, axis=-1)

    out = jnp.matmul(w, vt)                                    # (1, H, C, hd)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    out = jnp.dot(out, sa['o_proj']['kernel'])
    return out, k_updated, v_updated


def _attn_chunk_shared(lp, x, positions, cfg: Gemma4Config, attn_type: str,
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
        mask = _sliding_mask(pos_ring, pos_q, cfg.sliding_window_size)
    else:
        pos_k = jnp.arange(max_len, dtype=jnp.int32)
        mask = pos_k[jnp.newaxis, :] <= pos_q[:, jnp.newaxis]

    w = jnp.where(mask[jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w, axis=-1)

    out = jnp.transpose(jnp.matmul(w, vt), (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    return jnp.dot(out, sa['o_proj']['kernel'])


# ---------------------------------------------------------------------------
# Chunked prefill: CHUNK_SIZE tokens through one layer chunk
# ---------------------------------------------------------------------------

def prefill_chunk(params, chunk: LayerChunk, hidden, token_embed, ple_rows,
                  start_position, caches, sliding_pos_ring,
                  cfg: Gemma4Config = E2B_CONFIG, chunk_size: int = CHUNK_SIZE):
    """``chunk_size`` tokens through the layers of ``chunk``.

    As :func:`decode_chunk`, with ``L = chunk_size`` rows per input and
    ``start_position`` the absolute position of the first token.  A short
    final prompt chunk is right-padded with the rows of token 0; the padding
    is written into the caches like any token, and the host marks its
    positions in ``sliding_pos_ring`` too, exactly as it does for real ones.

    Returns ``(hidden (1, chunk_size, D), {slot: (k, v)} for chunk.writes)``.
    The runtime runs the head on the one row it needs (the last real token).
    """
    positions = (start_position + jnp.arange(chunk_size, dtype=jnp.int32))[jnp.newaxis]

    def own(lp, x_ln, attn_type, k, v):
        pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
        return _attn_chunk(lp, x_ln, positions, cfg, attn_type, k, v, pos_ring=pr)

    def shared(lp, x_ln, attn_type, src):
        pr = sliding_pos_ring if attn_type == AttentionType.LOCAL_SLIDING else None
        return _attn_chunk_shared(lp, x_ln, positions, cfg, attn_type, src, pos_ring=pr)

    return _run_chunk(params, chunk, hidden, token_embed, ple_rows, caches,
                      own, shared, cfg)
