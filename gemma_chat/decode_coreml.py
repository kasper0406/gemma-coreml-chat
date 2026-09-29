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
  Slot index = ``position % R``.  The host's ``sliding_pos_ring`` ``(1, R)``
  tracks which absolute position each slot holds (``-1`` = empty), and the
  attention mask it builds admits a slot only if its position lies in the
  query's window, ``q - W < p <= q``.

  Why ``W + CHUNK_SIZE`` rows rather than ``W``: a prefill call writes its
  whole chunk into the ring before any of its rows attend.  With a ring of
  exactly ``W`` rows, the chunk at positions ``s .. s+C-1`` overwrote positions
  ``s-W .. s-W+C-1`` — history its *first* rows still need (row ``s`` attends
  back to ``s-W+1``) — so every prompt longer than the window lost up to
  ``C - 1`` positions of context per row.  With ``C`` more rows the chunk
  overwrites only ``s-W-C .. s-W-1``, which no row of the chunk can see, and a
  KV-shared layer in a later layer chunk still finds every position it needs
  in the state.  Decode is one row, so it never had the problem; it pays
  ``C/W`` = 25% more sliding keys for sharing the layout — measured against a
  512-row export, ~0.45 ms per token on the GPU (~4%), ~0.3 ms on the ANE.
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
conversion and become state after materialization.

Every cache write below is a whole-tensor blend with a host-computed one-hot
selection (:func:`_cache_write`) rather than ``jax.lax.dynamic_update_slice``.
Two independent reasons, one per cache kind:

* *Correctness* (sliding): on macOS 26 a Core ML state write whose value comes
  from ``slice_update`` is applied to a freshly zeroed buffer instead of the
  persisted one, while a whole-tensor write persists correctly.
* *Speed* (global): ``slice_update`` with a runtime ``begin`` makes MPSGraph
  read that index back to the CPU mid-encode
  (``GPURegionRuntime::waitAndReadIntTensorData`` → ``waitUntilCompleted``),
  draining the pipeline once per write — 6 times per decode step, ~17 ms.  A
  whole-cache elementwise write never stalls.

Since materialization turns the global caches into state too (see
``mil_passes.global_cache_states``), the first reason now applies to them as
well: there is no ``slice_update`` left anywhere on a cache path.

Host inputs: no integer arithmetic in the graph
-----------------------------------------------
The functions take no position.  Everything a position used to derive in the
graph — the RoPE ``cos``/``sin`` rows, the sliding and global attention masks,
and which cache rows a step writes — the host computes per step and passes in
as fp16 (:func:`host_inputs` is the reference, :data:`HOST_INPUTS` the names).
The Neural Engine has no integer or fp32 arithmetic, so those int32 range /
compare / modulo chains and the fp32 ``sin``/``cos`` ran on the CPU, and every
step hopped CPU↔ANE around them (17 hops per decode step, ~200 CPU ops, the
state reads and layer 0's projections dragged along).  As inputs they cost the
host microseconds and the graph nothing: the masks are *added* to the scores
(the ANE ignores ``scaled_dot_product_attention``'s mask, so attention stays
decomposed — see ``mil_passes.ct_convert_pipeline``), and the writes are
multiplies by 0/1.

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
Every activation is fp16 — the residual stream, the attention inputs/outputs,
the KV caches, the MLP activations, the logits.  The two places with a range
fp16 lacks are handled without fp32 in the graph:

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
* **RoPE angles** (``model.rope_angles``): positions run to 65535, which fp16
  cannot represent exactly; the angle is fp32, and its ``cos``/``sin`` are
  rounded to fp16 before the rotation.  The host computes them
  (:func:`host_inputs`).

Everything stays fp16 end to end, which matters because the decode graph is
dispatch-bound: any op that returns fp32 drags its consumers up with it.  A
single fp32 leak in ``_apply_rope`` used to promote q, and through it SDPA, the
attention output and o_proj — which forced the 35 o_proj weights to be
re-materialized as fp32 constants at runtime (528 MB) and put ~1 GB of fp32
attention intermediates in every step.  Long-axis sums inside `matmul` and
``softmax`` are left to the backend.  The GPU accumulates matmuls in fp32; the
CPU (BNNS, macOS 27) does not for many shapes — an fp16 matmul against an
fp16 or int4 weight at M = 1..8 rows, and the M = 128 attention products, come
back with ~10-40x the error of fp32 accumulation, growing with the contraction
length.  That is why ``cpu-only`` lands further from the float reference than
``cpu-and-gpu`` (last-token KL ~0.002-0.03 against ~0.0001 on long prompts,
prefill and decode alike); the graph has no per-backend precision to ask for.
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
from gemma_chat.model import AttentionType, Gemma4Config, rope_rotate
from gemma_chat.cache_spec import kv_shared_sources, sliding_ring_length


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def empty_pos_ring(cfg: Gemma4Config = E2B_CONFIG) -> jnp.ndarray:
    """Return (1, sliding_ring_length) int32 filled with -1 (no entries)."""
    return jnp.full((1, sliding_ring_length(cfg)), -1, dtype=jnp.int32)


# The additive attention mask of a slot a query may not see.  It works only
# while every score stays far above it and it stays finite in fp16 once a score
# is added.  q and k are RMS-normed, scaled by q_norm / k_norm and rotated
# (which keeps their norm), with no 1/sqrt(hd), so |q.k| <= hd * max|s_q| *
# max|s_k| for every input (:func:`attention_score_bound`): 32.4 for E2B
# (layers 3, 4 and 12).  The largest score measured, over a 1737-token and a
# 55-token conversation on CPU in fp32, was 22.3 (a sliding layer; 19.9
# global).  The export refuses weights past SCORE_BOUND.  After softmax's max
# shift a masked slot is then within MASK_VALUE +- 2 * SCORE_BOUND: finite in
# fp16 (-65504 minus a score overflows to -inf), and exp() of it is exactly 0
# even in fp32 (below -104).  Every row sees at least its own slot
# (:func:`host_inputs`), so no row is ever all mask.
MASK_VALUE = -10000.0
SCORE_BOUND = 34.0


def attention_score_bound(params, cfg: Gemma4Config = E2B_CONFIG) -> float:
    """An input-independent bound on any layer's ``|q.k|``: ``hd * max|s_q|
    * max|s_k|`` (a KV-shared layer's keys are its source layer's), plus 1% for
    the fp16 rounding of q, k and their norms."""
    sources = kv_shared_sources(cfg)
    bound = 0.0
    for i in range(cfg.num_layers):
        def scale(layer, norm):
            sa = params[f'layers.{layer}']['self_attn']
            return float(np.abs(np.asarray(sa[norm]['scale'], np.float32)).max())
        hd = cfg.effective_head_dim(cfg.attention_types[i])
        bound = max(bound, hd * scale(i, 'q_norm') * scale(sources.get(i, i), 'k_norm'))
    return 1.01 * bound

# Every input the host computes per step, in signature order.  A layer chunk
# takes the ones its layers need (:func:`chunk_host_inputs`).
HOST_INPUTS = (
    "rope_sliding", "rope_global",
    "mask_sliding", "mask_global",
    "write_sliding", "write_global",
)


def _rope_params(cfg: Gemma4Config, kind: str) -> tuple[float, int, int]:
    """``(base_frequency, head_dim, half_dim)`` of the ``kind`` RoPE."""
    if kind == "sliding":
        base, hd, frac = cfg.rope_base_frequency, cfg.head_dim, cfg.rope_fraction_sliding
    else:
        base = cfg.global_rope_base_frequency
        hd, frac = cfg.effective_head_dim(AttentionType.GLOBAL), cfg.rope_fraction_global
    return base, hd, int(hd * frac) // 2


def host_input_shape(name: str, cfg: Gemma4Config, tokens: int, cache_len) -> tuple:
    """Shape of host input ``name`` for a step of ``tokens`` rows.

    ``cache_len`` (the global cache length ``N``, symbolic at export) is dim 3
    of ``mask_global`` and dim 1 of ``write_global``; no other input has it.
    """
    kind = name.split("_")[1]
    rows = sliding_ring_length(cfg) if kind == "sliding" else cache_len
    if name.startswith("rope_"):
        return (1, tokens, 1, 2 * _rope_params(cfg, kind)[2])
    if name.startswith("mask_"):
        return (1, 1, tokens, rows)
    return (1, rows, tokens)


def chunk_host_inputs(chunk: "LayerChunk", cfg: Gemma4Config) -> Tuple[str, ...]:
    """The host inputs ``chunk`` takes: RoPE rows and a mask for every attention
    kind among its layers, a write selection for every kind it writes a cache of
    (a KV-shared layer reads its source's cache and writes none)."""
    kinds = {cfg.attention_types[i] for i in chunk.layers}
    writes = {cfg.attention_types[s] for s in chunk.writes}  # slot == layer index
    names = []
    for name in HOST_INPUTS:
        kind = (AttentionType.LOCAL_SLIDING if name.endswith("sliding")
                else AttentionType.GLOBAL)
        if kind in (writes if name.startswith("write_") else kinds):
            names.append(name)
    return tuple(names)


def host_inputs(positions, ring, cache_len: int,
                cfg: Gemma4Config = E2B_CONFIG) -> dict[str, np.ndarray]:
    """What the host feeds one step: the reference ``GemmaCore`` matches.

    ``positions``: the step's absolute positions, one per row (a prefill
    chunk's padding rows included); ``ring``: ``sliding_pos_ring`` *after*
    they were marked (:func:`ring_with_positions`); ``cache_len``: the global
    caches' length.  All fp16:

    * ``rope_<kind>`` ``(1, L, 1, 2 * half)``: ``cos | sin`` of every row's
      RoPE angles.  The angle is computed exactly as ``model.rope_angles``
      does — fp32 ``position / timescale``, with ``timescale`` the fp32
      rounding of ``base ** (2 i / head_dim)`` (what XLA's fp32 pow yields,
      measured for both kinds) — and its cosine and sine in fp64, rounded once
      to fp16.
    * ``mask_<kind>`` ``(1, 1, L, slots)``: additive, 0 where the row may
      attend, :data:`MASK_VALUE` elsewhere.  A sliding slot holding position
      ``p`` (``-1`` = empty) is visible to query ``q`` iff ``q - W < p <= q``
      (the reference model's causal sliding window; the ring holds more than
      ``W`` positions, see the module docstring); a global slot iff
      ``slot <= q``.
    * ``write_<kind>`` ``(1, slots, L)``: 1 where cache row ``slot`` takes the
      step's row ``l`` — ``slot = p % R`` in the ring, ``slot = p`` in a global
      cache (a position past it takes no row) — else 0.
    """
    positions = np.asarray(positions, np.int64).reshape(-1)
    pk = np.asarray(ring, np.int64).reshape(-1)
    q = positions[:, None]
    window = cfg.sliding_window_size
    out = {}
    for kind in ("sliding", "global"):
        base, hd, half = _rope_params(cfg, kind)
        timescale = np.float32(base ** (2.0 * np.arange(half) / hd))
        angles = (positions.astype(np.float32)[:, None] / timescale).astype(np.float64)
        rows = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
        out[f"rope_{kind}"] = rows.astype(np.float16)[None, :, None, :]
    visible = {
        "sliding": (pk >= 0) & (pk <= q) & (pk > q - window),
        "global": np.arange(cache_len) <= q,
    }
    targets = {"sliding": (pk.size, positions % pk.size), "global": (cache_len, positions)}
    for kind in ("sliding", "global"):
        mask = np.where(visible[kind], 0.0, MASK_VALUE)
        out[f"mask_{kind}"] = mask.astype(np.float16)[None, None]
        slots, target = targets[kind]
        write = np.arange(slots)[:, None] == target[None, :]
        out[f"write_{kind}"] = write.astype(np.float16)[None]
    return out


def _cache_write(cache, value, onehot):
    """Write the step's rows ``value`` into ``cache`` where ``onehot`` says.

    ``cache`` is ``(1, S, nkv, hd)``, ``value`` ``(1, L, nkv, hd)``, ``onehot``
    the host's ``(1, S, L)`` write selection (:func:`host_inputs`).  The result
    is ``cache - cache * taken + placed``: ``placed`` the one-hot product
    putting each row in its slot, ``taken`` 1 on the written slots and 0
    elsewhere.  Exact in fp16 — every product is by 0 or 1, ``x - x`` is 0,
    and every sum has at most one nonzero term — and a whole-tensor write, see
    the module docstring.

    No constant meets a tensor of the global caches' symbolic length: JAX
    broadcasts such a constant with a dynamic ``fill``, which the Neural Engine
    does not run (hence ``cache - cache * taken`` rather than
    ``cache * (1 - taken)``), and ``taken`` is a matmul against ones rather
    than a ``jnp.sum``, which JAX would accumulate in fp32.
    """
    length, nkv, hd = cache.shape[1:]
    rows = value.shape[1]
    placed = jnp.matmul(onehot, value.reshape(1, rows, nkv * hd)).reshape(1, length, nkv, hd)
    taken = onehot if rows == 1 else jnp.matmul(onehot, jnp.ones((rows, 1), onehot.dtype))
    return cache - cache * taken[..., jnp.newaxis] + placed


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

    Returns (B, L, len(layers), per_layer_input_dim): each layer's input is a
    slice along axis 2.  (Slicing it out of the flat last axis instead is what
    one decode row cannot do on the Neural Engine — a ``(1, 1, n*d)`` →
    ``(1, 1, d)`` slice fails the ANE compile of the whole function, macOS 27.)
    """
    B, L = token_embed.shape[:2]
    d = cfg.per_layer_input_dim
    n = len(layers)

    ple_embed = (ple_rows * jnp.sqrt(float(d)).astype(ple_rows.dtype)).reshape(B, L, n, d)

    # numpy slice at trace time: only these columns become a graph constant.
    W_proj = np.asarray(params['per_layer_model_projection']['kernel'])
    W_proj = W_proj[:, layers.start * d:layers.stop * d]              # (D, n*d)
    ple_proj = jnp.dot(token_embed, W_proj) * (cfg.embed_dim ** -0.5)  # (B, L, n*d)

    scale = params['per_layer_projection_norm']['scale']   # (d,)
    ple_proj = _rmsnorm(ple_proj.reshape(B, L, n, d), scale)

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

def _kind(attn_type: str) -> str:
    """``"sliding"`` or ``"global"``: which of the host inputs a layer uses."""
    return "global" if attn_type == AttentionType.GLOBAL else "sliding"


def _attn_decode(lp, x, cfg: Gemma4Config, attn_type: str,
                 k_cache, v_cache, host, shared_kv=None):
    """Single-token attention with KV cache read/write.

    x: (1, 1, D)
    k_cache, v_cache: (1, cache_len, nkv, hd)
        cache_len = sliding_ring_length for sliding, max_seq_len for global.
    host: the step's host inputs (:func:`host_inputs`), with each RoPE table
          already split into ``(cos, sin)``.
    shared_kv: optional (k_cache, v_cache) from source layer; if given,
               this layer reads from source and does NOT update its own cache.

    Returns (attn_out (1,1,D), k_cache_updated, v_cache_updated).
    For KV-shared layers the returned caches are the source caches (unchanged).
    """
    kind = _kind(attn_type)
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    hd = cfg.effective_head_dim(attn_type)
    cos, sin = host[f"rope_{kind}"]                 # (1, 1, 1, half) each
    sa = lp['self_attn']

    # (H, 1, 1, hd): each head a row of the shape the fused norm wants (see
    # ``_rmsnorm``); the RoPE rows broadcast over the leading axis all the same.
    q = jnp.dot(x[0, 0], sa['q_proj']['kernel']).reshape(num_heads, 1, 1, hd)
    q = _rmsnorm(q, sa['q_norm']['scale'])
    q = rope_rotate(q, cos, sin)

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
        k_new = rope_rotate(k_new, cos, sin)

        # k_new/v_new are fp16 (the norms and the rotation keep the dtype),
        # so they go straight into the fp16 caches.
        write = host[f"write_{kind}"]
        k_updated = _cache_write(k_cache, k_new, write)
        v_updated = _cache_write(v_cache, v_new, write)
        k_full, v_full = k_updated, v_updated

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
    w = w + host[f"mask_{kind}"]                           # (1, 1, 1, max_len)
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
    position ``p``, and the sliding mask is built from it
    (:func:`host_inputs`).  This is the reference the runtime has to match.
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
    caches = dict(caches)

    x = hidden
    for j, i in enumerate(chunk.layers):
        attn_type = cfg.attention_types[i]
        ple_slice = ple_all[:, :, j]
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


def _split_rope(host):
    """``host`` with every RoPE table split into its ``(cos, sin)`` halves —
    once per chunk, not once per layer."""
    out = dict(host)
    for name, table in host.items():
        if name.startswith("rope_"):
            half = table.shape[-1] // 2
            out[name] = (table[..., :half], table[..., half:])
    return out


def decode_chunk(params, chunk: LayerChunk, hidden, token_embed, ple_rows,
                 host, caches, cfg: Gemma4Config = E2B_CONFIG):
    """One decode step through the layers of ``chunk``.

    Args:
        hidden:      (1, 1, D) fp16 — the residual stream entering the chunk
                     (``token_embed`` itself for the first chunk).
        token_embed: (1, 1, D) fp16 — the token's embedding row × sqrt(D).
        ple_rows:    (1, 1, len(chunk.layers) * per_layer_input_dim) fp16 —
                     its raw per-layer-embedding columns for these layers.
        host:        {name: array} for every name in
                     :func:`chunk_host_inputs` — see :func:`host_inputs`.
        caches:      {slot: (k, v)} for every slot in ``chunk.slots``.

    Returns ``(hidden (1, 1, D), {slot: (k, v)} for chunk.writes)``; the
    hidden state is final-normed when the chunk ends the model.
    """
    host = _split_rope(host)

    def own(lp, x_ln, attn_type, k, v):
        return _attn_decode(lp, x_ln, cfg, attn_type, k, v, host)

    def shared(lp, x_ln, attn_type, src):
        out, _, _ = _attn_decode(lp, x_ln, cfg, attn_type, *src, host, shared_kv=src)
        return out

    return _run_chunk(params, chunk, hidden, token_embed, ple_rows, caches,
                      own, shared, cfg)


# The Neural Engine splits a matmul's output channels across its 16 cores, and
# each core streams its share of the weight: ceil(N_out / 16) * K * bytes.  When
# that lands within ~16 KiB of a multiple of 1 MiB, a DMA erratum roughly halves
# the weight-streaming rate (https://eiln.github.io/posts/ane-dma.html;
# stablehlo-coreml PR #110).  Measured on the full int8 per-channel head
# (262144 x 1536, M4 Pro, macOS 27): 8 vocab slices = 3.000 MiB per core,
# 13.6 ms; 9 slices = 2.667 MiB, 5.66 ms; 10 slices = 2.401 MiB, 5.67 ms.
# Stepping equal slices through the notch one row per core (1.5 KiB) at a
# time, around 1, 2 and 3 MiB alike: 5.6 ms from -16 KiB, 9.2 ms at -12 KiB,
# 15-18 ms from -6 to +1 KiB (worst just below the multiple), 9-10 ms at
# +3..+6 KiB, 5.7 ms again from +8 KiB (+16 KiB at 1 and 2 MiB).
_ANE_CORES = 16
_ANE_NOTCH = 1 << 20
_ANE_NOTCH_GUARD = 64 << 10   # 4x the widest side of the measured slow band
# Rows per head slice, at most.  Core ML's int8 per-channel matmul returns
# garbage on the GPU for N_out >= 65536 once M >= 5 (the head runs at M = 1,
# this is margin), and one ANE weight kernel may not pass 128 MiB (this is
# 48 MiB at K = 1536).
_HEAD_MAX_ROWS = 32768


def ane_core_payload(rows: int, cols: int, bytes_per_weight: int = 1) -> int:
    """Bytes one ANE core streams for a ``[rows, cols]`` matmul weight."""
    return -(-rows // _ANE_CORES) * cols * bytes_per_weight


def in_ane_notch(payload: int) -> bool:
    """Whether a per-core payload sits in the slow band around a (nonzero)
    multiple of 1 MiB."""
    nearest = round(payload / _ANE_NOTCH) * _ANE_NOTCH
    return nearest > 0 and abs(payload - nearest) < _ANE_NOTCH_GUARD


def head_slices(vocab: int, dim: int) -> List[Tuple[int, int]]:
    """Vocab ranges ``[(start, stop), ...]`` the int8 logit head is split into.

    The fewest *equal* slices of at most ``_HEAD_MAX_ROWS`` rows (a multiple
    of 16, so every core gets whole rows) whose per-core payload stays clear
    of the ANE's DMA notch.  Equal, because the head returns them as the rows
    of one ``(slices, rows)`` array: the Neural Engine runs the softcap on
    that, but not on one ``vocab``-long row (MLComputePlan lists only the CPU
    for elementwise ops that wide).  For E2B's 262144 x 1536 head that is 16
    slices of 16384 rows, 1.5 MiB per core: 8 would be exactly 3 MiB.
    """
    for slices in range(-(-vocab // _HEAD_MAX_ROWS), vocab + 1):
        rows, rest = divmod(vocab, slices)
        if rest or (slices > 1 and rows % _ANE_CORES):
            continue
        if not in_ane_notch(ane_core_payload(rows, dim)):
            return [(k * rows, (k + 1) * rows) for k in range(slices)]
    raise ValueError(f"no equal split of a {vocab}-row head clears the ANE's DMA notch")


def logits_head(params, hidden, cfg: Gemma4Config = E2B_CONFIG):
    """The tied logit head: final-normed hidden (1, 1, D) fp16 → fp16 logits
    ``(slices, rows)``, which read row-major are the ``vocab`` logits in order.

    Its own function in the export, shared by decode and prefill (the host
    picks the prefill row it needs).  One fp16 matmul per vocab slice
    (:func:`head_slices`), stacked, and the softcap — all fp16, which is what
    keeps the head on the Neural Engine (it has no fp32): an fp32 softcap put
    the cast and the ``tanh`` on the CPU.  The capped logits lie in
    ``(-30, 30)``, where fp16's spacing is at most 1/64.

    The export stores each slice as int8 with one scale per vocab row
    (``export._export_function(weight_bits=8)``): per-channel scales are what
    the Neural Engine accepts (block-32 ones kept the head on the CPU), and the
    slicing keeps it out of the ANE's DMA notch.  int4 is too lossy here.
    """
    # numpy slices at trace time: each becomes a [dim, rows] graph constant.
    table = np.asarray(params['embed_tokens'])
    h = hidden[0]                                              # (1, D)
    logits = jnp.concatenate([
        jnp.dot(h, table[a:b].T) for a, b in head_slices(*table.shape)
    ])                                                         # (slices, rows)
    cap = cfg.final_logit_softcap
    return logits if cap is None else jnp.tanh(logits / cap) * cap


def decode_step(params, token_embed, ple_rows, position, kv_flat, sliding_pos_ring,
                cfg: Gemma4Config = E2B_CONFIG):
    """The whole model for one token, as the runtime composes it.

    Ring update and host inputs (host), every layer chunk in order, then the
    head.  Only a reference for tests: the export traces the pieces.
    ``kv_flat`` is ``[k_0, v_0, k_1, v_1, ...]``, one pair per cache slot.

    Returns ``(logits (slices, rows), kv_flat_new, sliding_pos_ring_new)``.
    """
    ring = ring_with_positions(sliding_pos_ring, position)
    caches = {s: (kv_flat[2 * s], kv_flat[2 * s + 1]) for s in range(len(kv_flat) // 2)}
    cache_len = max(k.shape[1] for s, (k, _) in caches.items()
                    if cfg.attention_types[s] == AttentionType.GLOBAL)
    host = {n: jnp.asarray(a) for n, a in
            host_inputs(np.atleast_1d(np.asarray(position)), np.asarray(ring), cache_len, cfg).items()}
    d = cfg.per_layer_input_dim
    hidden = token_embed
    for chunk in layer_chunks(cfg):
        cols = ple_rows[:, :, chunk.layers.start * d:chunk.layers.stop * d]
        hidden, written = decode_chunk(
            params, chunk, hidden, token_embed, cols,
            {n: host[n] for n in chunk_host_inputs(chunk, cfg)},
            {s: caches[s] for s in chunk.slots}, cfg,
        )
        caches.update(written)
    logits = logits_head(params, hidden, cfg)
    kv_new = [c for s in sorted(caches) for c in caches[s]]
    return logits, kv_new, ring


# ---------------------------------------------------------------------------
# Chunked-prefill attention (for chunk_prefill_step)
# ---------------------------------------------------------------------------

def _attn_chunk(lp, x, cfg: Gemma4Config, attn_type: str, k_cache, v_cache, host):
    """Chunk attention with KV cache read/write.

    x: (1, C, D)  — C = CHUNK_SIZE tokens
    k_cache, v_cache: (1, cache_len, nkv, hd)
    host: the step's host inputs, RoPE tables split — see :func:`_attn_decode`.

    Returns (attn_out (1, C, D), k_cache_updated, v_cache_updated).
    """
    C = x.shape[1]
    kind = _kind(attn_type)
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    hd = cfg.effective_head_dim(attn_type)
    cos, sin = host[f"rope_{kind}"]                 # (1, C, 1, half) each
    sa = lp['self_attn']

    # Q/K/V projections for the chunk
    q = jnp.dot(x, sa['q_proj']['kernel']).reshape(1, C, num_heads, hd)
    k_new = jnp.dot(x, sa['k_proj']['kernel']).reshape(1, C, num_kv_heads, hd)
    v_new = jnp.dot(x, sa['v_proj']['kernel']).reshape(1, C, num_kv_heads, hd)

    q = _rmsnorm(q, sa['q_norm']['scale'])
    k_new = _rmsnorm(k_new, sa['k_norm']['scale'])
    v_new = _rmsnorm(v_new)

    q = rope_rotate(q, cos, sin)
    k_new = rope_rotate(k_new, cos, sin)

    # Sliding layers wrap into the ring; global layers write at the absolute
    # position — the host's write selection says which (see
    # :func:`host_inputs`).  The ring is W + C rows long, so the chunk
    # overwrites nothing any of its rows still attends to (module docstring).
    write = host[f"write_{kind}"]
    k_updated = _cache_write(k_cache, k_new, write)
    v_updated = _cache_write(v_cache, v_new, write)
    return _chunk_attend(sa, q, k_updated, v_updated, host[f"mask_{kind}"], cfg), \
        k_updated, v_updated


def _chunk_attend(sa, q, k_cache, v_cache, mask, cfg: Gemma4Config):
    """Attention of the chunk's ``q`` (1, C, H, hd) over a whole cache, masked
    by the host's additive ``mask`` (1, 1, C, cache_len); through o_proj."""
    C, num_heads, hd = q.shape[1:]
    kv_rep = num_heads // cfg.num_kv_heads
    k_full = jnp.repeat(k_cache, kv_rep, axis=2) if kv_rep > 1 else k_cache
    v_full = jnp.repeat(v_cache, kv_rep, axis=2) if kv_rep > 1 else v_cache

    qt = jnp.transpose(q, (0, 2, 1, 3))         # (1, H, C, hd)
    kt = jnp.transpose(k_full, (0, 2, 1, 3))    # (1, H, cache_len, hd)
    vt = jnp.transpose(v_full, (0, 2, 1, 3))

    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))  # (1, H, C, cache_len)
    w = jax.nn.softmax(w + mask, axis=-1)

    out = jnp.matmul(w, vt)                                    # (1, H, C, hd)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(1, C, num_heads * hd)
    return jnp.dot(out, sa['o_proj']['kernel'])


def _attn_chunk_shared(lp, x, cfg: Gemma4Config, attn_type: str, shared_kv, host):
    """Chunk attention using K/V from a source (KV-shared) layer.

    Only Q is computed from `lp`; K/V come from `shared_kv`.
    """
    C = x.shape[1]
    kind = _kind(attn_type)
    hd = cfg.effective_head_dim(attn_type)
    cos, sin = host[f"rope_{kind}"]
    sa = lp['self_attn']

    q = jnp.dot(x, sa['q_proj']['kernel']).reshape(1, C, cfg.num_heads, hd)
    q = _rmsnorm(q, sa['q_norm']['scale'])
    q = rope_rotate(q, cos, sin)
    return _chunk_attend(sa, q, *shared_kv, host[f"mask_{kind}"], cfg)


# ---------------------------------------------------------------------------
# Chunked prefill: CHUNK_SIZE tokens through one layer chunk
# ---------------------------------------------------------------------------

def prefill_chunk(params, chunk: LayerChunk, hidden, token_embed, ple_rows,
                  host, caches, cfg: Gemma4Config = E2B_CONFIG):
    """``CHUNK_SIZE`` tokens through the layers of ``chunk``.

    As :func:`decode_chunk`, with ``L = CHUNK_SIZE`` rows per input.  A short
    final prompt chunk is right-padded with the rows of token 0; the padding
    rows take the next positions and are written into the caches like any
    token — the host builds their RoPE rows, masks and writes, and marks them
    in ``sliding_pos_ring``, exactly as it does for real ones.

    Returns ``(hidden (1, chunk_size, D), {slot: (k, v)} for chunk.writes)``.
    The runtime runs the head on the one row it needs (the last real token).
    """
    host = _split_rope(host)

    def own(lp, x_ln, attn_type, k, v):
        return _attn_chunk(lp, x_ln, cfg, attn_type, k, v, host)

    def shared(lp, x_ln, attn_type, src):
        return _attn_chunk_shared(lp, x_ln, cfg, attn_type, src, host)

    return _run_chunk(params, chunk, hidden, token_embed, ple_rows, caches,
                      own, shared, cfg)
