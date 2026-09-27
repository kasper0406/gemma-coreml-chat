"""Integration test for the project's MIL pass pipeline.

Runs the real export flow — ``build_ct_convert_pass_pipeline()`` fed to
``ct.convert`` — over small JAX graphs shaped like the ones
``gemma_chat/decode_coreml.py`` traces, and checks that the fusions the
exported model depends on actually land.

Most of the passes involved live in ``stablehlo_coreml.passes`` and are unit
tested there; what is tested here is the composition: our pipeline, our pass
placement, and the graph shapes this project actually produces.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.scipy.special
import coremltools as ct
from coremltools.converters.mil.mil import types as mil_types
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.decode_coreml import _rmsnorm as rmsnorm
from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline

# ── helpers ──────────────────────────────────────────────────────────────

def _convert(fn, *example_args, load: bool = False, compute_units=ct.ComputeUnit.ALL):
    """Trace ``fn``, run the project pipeline, return ``(mlmodel, mil_program)``."""
    hlo = jax.jit(fn).lower(*example_args).compiler_ir("stablehlo")
    prog = hlo_to_mil(hlo, minimum_deployment_target=ct.target.iOS18)

    pipeline = build_ct_convert_pass_pipeline()
    model = ct.convert(
        prog,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=compute_units,
        skip_model_load=not load,
    )
    return model, model._mil_program


def _ops(prog, fname="main"):
    return list(prog.functions[fname].operations)


def _count(prog, op_type, fname="main"):
    return sum(1 for op in _ops(prog, fname) if op.op_type == op_type)


def _predict(model, *np_inputs):
    """Feed ``np_inputs`` positionally to the model, return the first output."""
    names = [i.name for i in model.get_spec().description.input]
    assert len(names) == len(np_inputs), f"inputs {names} vs {len(np_inputs)} values"
    result = model.predict(dict(zip(names, np_inputs)))
    return np.array(next(iter(result.values())))


def _assert_softmax_not_decomposed(prog):
    """The StableHLO softmax decomposition must be gone, not just rewritten."""
    for op_type in ("reduce_max", "reduce_log_sum_exp", "exp", "reduce_sum"):
        assert _count(prog, op_type) == 0, f"decomposed-softmax leftover: {op_type}"


def _cast_roundtrips(prog):
    """``cast(x, A) → cast(_, x.dtype)`` pairs — pointless precision round-trips."""
    found = 0
    for op in _ops(prog):
        if op.op_type != "cast":
            continue
        parent = op.inputs["x"].op
        if parent is None or parent.op_type != "cast":
            continue
        if parent.inputs["x"].dtype == op.outputs[0].dtype:
            found += 1
    return found


# ── graphs, mirroring gemma_chat/decode_coreml.py ────────────────────────

def _attend(q, k, v, mask, kv_rep):
    """GQA attention exactly as ``decode_coreml._attend_*`` builds it."""
    if kv_rep > 1:
        k = jnp.repeat(k, kv_rep, axis=2)
        v = jnp.repeat(v, kv_rep, axis=2)
    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))
    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))
    w = jnp.where(mask, w, -10000.0)
    w = jax.nn.softmax(w, axis=-1)
    out = jnp.matmul(w, vt)
    B, C, H, hd = q.shape
    return jnp.transpose(out, (0, 2, 1, 3)).reshape(B, C, H * hd)


def chunk_attention(q, k, v, mask):
    """Prefill / chunk attention: query length > 1, mask is (C, S)."""
    return _attend(q, k, v, mask[jnp.newaxis, jnp.newaxis], kv_rep=2)


def decode_attention(q, k, v, valid):
    """Decode attention: query length 1, mask is (S,)."""
    return _attend(q, k, v, valid[jnp.newaxis, jnp.newaxis, jnp.newaxis], kv_rep=2)


def sliding_cache_write(cache, value, slot):
    """``decode_coreml._sliding_ring_write``: masked whole-tensor cache write.

    Both operands broadcast against the ``(1, window, nkv, hd)`` cache — the
    ``(window,)`` slot mask and the ``(1, 1, nkv, hd)`` new entry — so StableHLO
    materializes a ``tile`` for each before the ``select``.
    """
    window = cache.shape[1]
    mask = (jnp.arange(window, dtype=jnp.int32) == slot)[
        jnp.newaxis, :, jnp.newaxis, jnp.newaxis
    ]
    return jnp.where(mask, value, cache)


def exact_gelu(x):
    """The FFN activation from ``decode_coreml._gelu_exact`` — fp16, erf spelling."""
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x * float(1.0 / np.sqrt(2.0))))


def logit_softcap(x):
    """``decode_coreml``'s final logit softcap, cap=30."""
    cap = jnp.float32(30.0)
    return jnp.tanh(x / cap) * cap


def double_rmsnorm(x):
    """Adjacent norms, as every sub-layer boundary of the real graph has them
    (the scales are weight constants there too)."""
    scale = jnp.asarray(np.full((x.shape[-1],), 0.5, np.float16))
    return rmsnorm(rmsnorm(x, scale), scale)


def _attn_args(C, hd, kv_rep=2, H=8, S=128):
    f32 = jnp.float32
    return (
        jnp.ones((1, C, H, hd), f32),
        jnp.ones((1, S, H // kv_rep, hd), f32),
        jnp.ones((1, S, H // kv_rep, hd), f32),
    )


# ── attention fusion ─────────────────────────────────────────────────────

def _assert_attention_decomposed(prog):
    """``matmul -> select(mask) -> softmax -> matmul``, never SDPA.

    The Neural Engine ignores ``scaled_dot_product_attention``'s ``attn_mask``,
    so the pipeline drops ``fuse_attention_to_sdpa`` — see
    ``ct_convert_pipeline``.  The decomposed softmax still collapses to one op.
    """
    assert _count(prog, "scaled_dot_product_attention") == 0
    assert _count(prog, "softmax") == 1
    assert _count(prog, "matmul") == 2
    assert _count(prog, "select") == 1
    _assert_softmax_not_decomposed(prog)


def test_chunk_attention_stays_decomposed():
    q, k, v = _attn_args(C=16, hd=256)
    mask = jnp.ones((16, 128), jnp.bool_)
    _, prog = _convert(chunk_attention, q, k, v, mask)
    _assert_attention_decomposed(prog)


def test_global_chunk_attention_stays_decomposed():
    q, k, v = _attn_args(C=16, hd=512)
    mask = jnp.ones((16, 128), jnp.bool_)
    _, prog = _convert(chunk_attention, q, k, v, mask)
    _assert_attention_decomposed(prog)


def test_decode_attention_stays_decomposed():
    q, k, v = _attn_args(C=1, hd=256)
    valid = jnp.ones((128,), jnp.bool_)
    _, prog = _convert(decode_attention, q, k, v, valid)
    _assert_attention_decomposed(prog)


def test_standalone_softmax_stays_a_softmax():
    def fn(x):
        return jax.nn.softmax(x, axis=-1)

    _, prog = _convert(fn, jnp.ones((1, 8, 1, 128), jnp.float32))

    assert _count(prog, "softmax") == 1
    assert _count(prog, "scaled_dot_product_attention") == 0
    _assert_softmax_not_decomposed(prog)


def test_tile_feeding_select_is_preserved():
    """E5RT's multifunction shape propagation rejects a broadcasting ``select``.

    ``remove_broadcast_tiles`` deliberately excludes ``select`` from the ops it
    strips tiles from: E5RT fails type inference on an implicitly broadcasting
    ``select`` in a multifunction ``.mlpackage``. Exercised on the sliding-cache
    write, the graph that puts a ``select`` in the real exported model.
    """
    cache = jnp.ones((1, 128, 4, 256), jnp.float16)
    value = jnp.ones((1, 1, 4, 256), jnp.float16)
    _, prog = _convert(sliding_cache_write, cache, value, jnp.int32(3))

    assert _count(prog, "select") == 1
    select = next(op for op in _ops(prog) if op.op_type == "select")
    # Both broadcasting operands keep their tile; only ``b`` is already full size.
    tiled = [name for name in ("cond", "a", "b")
             if select.inputs[name].op is not None
             and select.inputs[name].op.op_type == "tile"]
    assert tiled == ["cond", "a"], f"tiles feeding select were removed: kept {tiled}"
    for name in ("cond", "a", "b"):
        assert tuple(select.inputs[name].shape) == (1, 128, 4, 256)


# ── softcap / norm / activation fusion ───────────────────────────────────

def test_logit_softcap_fuses_to_scaled_tanh():
    _, prog = _convert(logit_softcap, jnp.ones((1, 8, 256), jnp.float16))

    assert _count(prog, "scaled_tanh") == 1
    assert _count(prog, "tanh") == 0
    op = next(op for op in _ops(prog) if op.op_type == "scaled_tanh")
    assert abs(float(op.inputs["alpha"].val) - 30.0) < 1e-4
    assert abs(float(op.inputs["beta"].val) - 1.0 / 30.0) < 1e-4


def _rmsnorm_const_scale(x):
    """``rmsnorm`` with the scale as a weight constant, as in the real graph."""
    return rmsnorm(x, jnp.asarray(np.full((x.shape[-1],), 0.5, np.float16)))


def test_rmsnorm_fuses_to_one_fp16_l2_norm():
    """The fp32-statistics norm becomes ``l2_norm`` + one ``mul``, in fp16.

    ``fuse_rmsnorm`` collapses the chain onto ``l2_norm`` and ``fp16_l2_norm``
    drops the casts around it. ``(1, 1, D)`` needs no reshape: ``l2_norm``
    normalizes over the last three dims, which for that shape is exactly the
    last one.
    """
    x = jnp.ones((1, 1, 256), jnp.float16)
    _, prog = _convert(_rmsnorm_const_scale, x)

    assert _count(prog, "l2_norm") == 1
    for op_type in ("reduce_mean", "reduce_sum", "rsqrt", "reshape", "cast"):
        assert _count(prog, op_type) == 0, f"unfused RMSNorm leftover: {op_type}"
    # eps' = d * eps, so that l2_norm's sum of squares matches mean + 1e-6.
    l2 = next(op for op in _ops(prog) if op.op_type == "l2_norm")
    assert l2.inputs["epsilon"].val == np.float16(256 * 1e-6)
    assert l2.outputs[0].dtype == mil_types.fp16
    # l2_norm -> mul(sqrt(d) * scale).
    assert sum(1 for op in _ops(prog) if op.op_type != "const") == 2


def test_rmsnorm_off_canonical_shape_is_viewed_as_rows():
    """Prefill ``(1, L, D)`` and per-head ``(1, L, H, hd)`` norms are viewed as
    ``(rows, 1, 1, d)`` so they fuse too: an unfused fp16 sum of squares
    would overflow, and the fp32 one would pin the norm to the CPU on the ANE."""
    for shape in ((1, 4, 8, 256), (1, 128, 256), (1, 1, 8, 256)):
        _, prog = _convert(_rmsnorm_const_scale, jnp.ones(shape, jnp.float16))
        assert _count(prog, "l2_norm") == 1, shape
        for op_type in ("reduce_mean", "reduce_sum", "rsqrt", "cast"):
            assert _count(prog, op_type) == 0, (shape, op_type)
        assert _count(prog, "reshape") <= 2, shape


def test_a_norm_scale_fp16_cannot_hold_fails_the_conversion():
    """``sqrt(d) * scale`` is narrowed to fp16; one that overflows must not
    silently become ``inf``. At d = 1536 that takes a scale above ~1671."""
    def big_scale_norm(x):
        return rmsnorm(x, jnp.asarray(np.full((x.shape[-1],), 2000.0, np.float16)))

    with pytest.raises(ValueError, match="fp16 cannot represent"):
        _convert(big_scale_norm, jnp.ones((1, 1, 1536), jnp.float16))


def test_exact_gelu_fuses_to_one_fp16_op():
    """``chlo.erf`` is mapped natively and fused by ``fuse_gelu_exact``.

    No cast pair: the fused op runs in the fp16 activation dtype.
    """
    _, prog = _convert(exact_gelu, jnp.ones((1, 8, 256), jnp.float16))

    assert _count(prog, "gelu") == 1
    assert _count(prog, "cast") == 0
    for op_type in ("erf", "erfc", "tanh", "pow"):
        assert _count(prog, op_type) == 0, f"unfused gelu leftover: {op_type}"
    gelu = next(op for op in _ops(prog) if op.op_type == "gelu")
    assert gelu.outputs[0].dtype == mil_types.fp16


def test_rmsnorm_is_fp16_end_to_end():
    """The ANE has no fp32: a norm that upcasts anywhere pins its ops to the CPU.

    Covers every norm shape the model has — the ``(1, 1, D)`` decode norms,
    the chunk and per-head ones — and two norms back to back.
    """
    progs = [_convert(_rmsnorm_const_scale, jnp.ones(shape, jnp.float16))[1]
             for shape in ((1, 1, 1536), (1, 128, 256), (1, 4, 8, 256))]
    progs.append(_convert(double_rmsnorm, jnp.ones((1, 1, 256), jnp.float16))[1])
    for prog in progs:
        assert _count(prog, "cast") == 0
        for op in _ops(prog):
            for out in op.outputs:
                if not mil_types.is_float(out.dtype):
                    continue  # int32 axes constants
                assert out.dtype == mil_types.fp16, (
                    f"{op.op_type} produces {mil_types.builtin_to_string(out.dtype)}"
                )


def _norm_rows():
    """fp16 rows across the range RMSNorm meets, eps-dominated ones included."""
    rng = np.random.RandomState(11)
    rows = {
        "zeros": np.zeros(1536),
        "all 1e-6": np.full(1536, 1e-6),
        "all 1e-4": np.full(1536, 1e-4),
        "all 1e-3": np.full(1536, 1e-3),
        "rms 1": rng.randn(1536),
        "rms 80": rng.randn(1536) * 80.0,
        "rms 2000": rng.randn(1536) * 2000.0,
    }
    outlier = rng.randn(1536) * 300.0
    outlier[7] = 6.0e4  # one huge outlier, as residual streams have
    rows["rms 300 + 6e4 outlier"] = outlier
    return {k: v.astype(np.float16) for k, v in rows.items()}


def _reference_rmsnorm(x16, scale16):
    x = x16.astype(np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * scale16.astype(np.float64)


def test_rmsnorm_matches_an_fp64_reference_across_the_range():
    """The JAX function, fp16 in and out, against fp64 — tiny rows included,
    where ``eps`` dominates and the output must stay small, and rows whose
    fp16 sum of squares would overflow."""
    scale = (1.0 + np.random.RandomState(5).randn(1536) * 0.1).astype(np.float16)
    for name, row in _norm_rows().items():
        # decode (1, 1, D), prefill (1, L, D) and per-head (1, 1, H, hd) rows
        for x in (row.reshape(1, 1, 1536), np.stack([row, row])[None], row.reshape(1, 1, 6, 256)):
            s = scale[:x.shape[-1]]
            out = np.asarray(rmsnorm(jnp.asarray(x), jnp.asarray(s)), np.float64)
            ref = _reference_rmsnorm(x, s)
            shape = x.shape
            assert np.all(np.isfinite(out)), (name, shape)
            np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max() + 1e-12,
                                       err_msg=f"{name} {shape}")


# ── numerical parity ─────────────────────────────────────────────────────

def test_numerical_chunk_attention():
    rng = np.random.RandomState(42)
    C, S, H, kvh, hd = 8, 32, 8, 4, 256
    q = rng.randn(1, C, H, hd).astype(np.float32) * 0.1
    k = rng.randn(1, S, kvh, hd).astype(np.float32) * 0.1
    v = rng.randn(1, S, kvh, hd).astype(np.float32) * 0.1
    mask = np.tril(np.ones((C, S), dtype=np.bool_))

    ref = np.array(chunk_attention(jnp.array(q), jnp.array(k), jnp.array(v), jnp.array(mask)))
    model, prog = _convert(
        chunk_attention,
        jnp.ones_like(q), jnp.ones_like(k), jnp.ones_like(v), jnp.ones((C, S), jnp.bool_),
        load=True,
    )
    _assert_attention_decomposed(prog)
    out = _predict(model, q, k, v, mask.astype(np.float32))

    assert np.max(np.abs(ref - out)) < 1e-3


def test_numerical_decode_attention():
    rng = np.random.RandomState(7)
    S, H, kvh, hd = 32, 8, 4, 256
    q = rng.randn(1, 1, H, hd).astype(np.float32) * 0.1
    k = rng.randn(1, S, kvh, hd).astype(np.float32) * 0.1
    v = rng.randn(1, S, kvh, hd).astype(np.float32) * 0.1
    valid = np.arange(S) < 20

    ref = np.array(decode_attention(jnp.array(q), jnp.array(k), jnp.array(v), jnp.array(valid)))
    model, prog = _convert(
        decode_attention,
        jnp.ones_like(q), jnp.ones_like(k), jnp.ones_like(v), jnp.ones((S,), jnp.bool_),
        load=True,
    )
    _assert_attention_decomposed(prog)
    out = _predict(model, q, k, v, valid.astype(np.float32))

    assert np.max(np.abs(ref - out)) < 1e-3


def test_numerical_logit_softcap():
    x = np.random.RandomState(0).randn(1, 4, 32).astype(np.float16)
    ref = np.array(logit_softcap(jnp.array(x)))

    model, _ = _convert(logit_softcap, jnp.array(x), load=True)
    out = _predict(model, x)

    np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_exported_rmsnorm_matches_an_fp64_reference_on_cpu():
    """The converted norm — an fp16 ``l2_norm`` — on the CPU across the range.

    Its sum of squares must not overflow like a plain fp16 one would. (Measured
    on the GPU and the ANE as well; see ``mil_passes/fp16_l2_norm`` for those
    numbers and the ANE's one deviation, on eps-dominated rows.)
    """
    rows = _norm_rows()
    x = np.stack(list(rows.values()))[:, None, None, :]   # (rows, 1, 1, 1536)
    scale = (1.0 + np.random.RandomState(5).randn(1536) * 0.1).astype(np.float16)
    model, prog = _convert(
        _rmsnorm_const_scale, jnp.asarray(x), load=True, compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    assert _count(prog, "l2_norm") == 1 and _count(prog, "cast") == 0
    out = _predict(model, x).astype(np.float64)
    ref = _reference_rmsnorm(x, np.full(1536, 0.5, np.float16))
    for i, name in enumerate(rows):
        np.testing.assert_allclose(out[i], ref[i], rtol=5e-3, atol=5e-3 * np.abs(ref[i]).max() + 1e-12,
                                   err_msg=name)


def test_numerical_rmsnorm_and_gelu():
    rng = np.random.RandomState(3)
    x = rng.randn(1, 8, 256).astype(np.float16)
    scale = rng.randn(256).astype(np.float16)

    ref_norm = np.array(rmsnorm(jnp.array(x), jnp.array(scale)))
    model, _ = _convert(rmsnorm, jnp.array(x), jnp.array(scale), load=True)
    np.testing.assert_allclose(_predict(model, x, scale), ref_norm, atol=1e-2, rtol=1e-2)

    ref_gelu = np.array(exact_gelu(jnp.array(x)))
    model, _ = _convert(exact_gelu, jnp.array(x), load=True)
    np.testing.assert_allclose(_predict(model, x), ref_gelu, atol=1e-2, rtol=1e-2)


# ── weight quantization: what gets a constexpr and what does not ─────────

def test_logit_projection_is_int8_block32():
    """The [dim, vocab] logit projection is quantized to int8 with block-32 scales.

    Both historical reasons for leaving it fp16 are tied to the old
    [K, N] / ``transpose_y=False`` layout and no longer hold:

    * MPSGraph's ``LowerDequantizeND`` constant-fold (~16 s on the first
      prediction of every process) only happens in that orientation.  Measured
      fresh-process first predict at N=262144: 16.3 s for [K,N]/ty=False vs
      0.06 s for [N,K]/ty=True, the same as fp16.
    * int4 is still too lossy for logits, and buys no speed over int8 here
      (2.02 vs 2.11 ms at M=1).

    Block-32 rather than per-channel: Core ML's int8 per-channel matmul in the
    [N,K]/``transpose_y=True`` orientation returns uncorrelated garbage for
    N >= 65536 once M >= 5 (relRMS 1.0 vs fp16), and blockwise scales keep the
    head off the Neural Engine, where int8 runs ~3x slower than block-32 does
    on the CPU.  This test pins the grouping.

    The other [dim, *] weight in the same graph is the control: it stays int4.
    """
    rng = np.random.RandomState(5)
    vocab = 262144
    hidden = (rng.randn(32, 64) * 0.1).astype(np.float16)
    logit_w = (rng.randn(64, vocab) * 0.02).astype(np.float16)

    def logits_fn(token_embed):
        return jnp.matmul(jnp.matmul(token_embed, jnp.asarray(hidden)),
                          jnp.asarray(logit_w))

    _, prog = _convert(logits_fn, jnp.zeros((1, 1, 32), jnp.float16))

    constexprs = [op for op in _ops(prog)
                  if op.op_type == "constexpr_blockwise_shift_scale"]
    by_shape = {tuple(op.outputs[0].shape): op for op in constexprs}

    head = by_shape.get(logit_w.shape) or by_shape.get(logit_w.T.shape)
    assert head is not None, (
        f"the logit projection was not quantized (constexprs: {list(by_shape)})"
    )

    # int8, not int4: the data must NOT carry the sub-byte tag.
    data = head.inputs["data"].val
    assert data.dtype.metadata is None, "logit head was quantized to int4, not int8"

    # Grouped (block-32) along the contraction axis, i.e. >1 scale per row.
    scale = head.inputs["scale"].val
    assert scale.ndim == 2 and min(scale.shape) > 1, (
        f"logit head scale {scale.shape} is not grouped; per-channel scales "
        "silently corrupt this matmul at M >= 5 for vocab >= 65536"
    )

    assert {hidden.shape, hidden.T.shape} & set(by_shape), "the hidden weight stopped being quantized"

