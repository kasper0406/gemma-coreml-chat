"""Integration test for the project's MIL pass pipeline.

Runs the real export flow — ``build_ct_convert_pass_pipeline()`` minus the
passes ``gemma_chat/export.py:_mil_to_mlpackage`` removes, fed to
``ct.convert`` — over small JAX graphs shaped like the ones
``gemma_chat/decode_coreml.py`` traces, and checks that the fusions the
exported model depends on actually land.

Most of the passes involved live in ``stablehlo_coreml.passes`` and are unit
tested there; what is tested here is the composition: our pipeline, our pass
placement, and the graph shapes this project actually produces.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special
import coremltools as ct
from coremltools.converters.mil.mil import types as mil_types
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline
from gemma_chat.model import _embed_lookup

# Kept in sync with gemma_chat/export.py:_mil_to_mlpackage.
_REMOVED_PASSES = [
    "common::add_fp16_cast",
    "common::fuse_layernorm_or_instancenorm",
    "common::fuse_elementwise_to_batchnorm",
]


# ── helpers ──────────────────────────────────────────────────────────────

def _convert(fn, *example_args, load: bool = False):
    """Trace ``fn``, run the project pipeline, return ``(mlmodel, mil_program)``."""
    hlo = jax.jit(fn).lower(*example_args).compiler_ir("stablehlo")
    prog = hlo_to_mil(hlo, minimum_deployment_target=ct.target.iOS18)

    pipeline = build_ct_convert_pass_pipeline()
    pipeline.remove_passes(_REMOVED_PASSES)
    model = ct.convert(
        prog,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
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


def rmsnorm(x, scale):
    """``decode_coreml._rmsnorm``."""
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + 1e-6) * scale.astype(jnp.float32)).astype(jnp.float16)


def exact_gelu(x):
    """The FFN activation from ``decode_coreml._gelu_exact`` — fp16, erf spelling."""
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x * float(1.0 / np.sqrt(2.0))))


def logit_softcap(x):
    """``decode_coreml``'s final logit softcap, cap=30."""
    cap = jnp.float32(30.0)
    return jnp.tanh(x / cap) * cap


def double_rmsnorm(x, scale_a, scale_b):
    """Adjacent norms — the fp16→fp32 round-trip ``collapse_cast_chains`` targets."""
    return rmsnorm(rmsnorm(x, scale_a), scale_b)


def _attn_args(C, hd, kv_rep=2, H=8, S=128):
    f32 = jnp.float32
    return (
        jnp.ones((1, C, H, hd), f32),
        jnp.ones((1, S, H // kv_rep, hd), f32),
        jnp.ones((1, S, H // kv_rep, hd), f32),
    )


# ── attention fusion ─────────────────────────────────────────────────────

def test_chunk_attention_fuses_to_sdpa():
    q, k, v = _attn_args(C=16, hd=256)
    mask = jnp.ones((16, 128), jnp.bool_)
    _, prog = _convert(chunk_attention, q, k, v, mask)

    assert _count(prog, "scaled_dot_product_attention") == 1
    assert _count(prog, "softmax") == 0
    assert _count(prog, "matmul") == 0
    assert _count(prog, "select") == 0
    _assert_softmax_not_decomposed(prog)


def test_global_attention_fuses_to_sdpa():
    """Global layers use head_dim=512; SDPA pre-scales the query by sqrt(512)."""
    q, k, v = _attn_args(C=16, hd=512)
    mask = jnp.ones((16, 128), jnp.bool_)
    _, prog = _convert(chunk_attention, q, k, v, mask)

    assert _count(prog, "scaled_dot_product_attention") == 1
    assert _count(prog, "softmax") == 0
    scales = [float(op.inputs["y"].val) for op in _ops(prog)
              if op.op_type == "mul" and op.inputs["y"].val is not None
              and np.asarray(op.inputs["y"].val).size == 1]
    assert any(abs(s - np.sqrt(512.0)) < 0.1 for s in scales), scales


def test_decode_attention_fuses_to_sdpa():
    """Decode is the hot path: query length 1 must fuse as completely as prefill.

    stablehlo-coreml 0.1.5 taught ``fuse_attention_to_sdpa`` to handle a unit
    query axis, so the whole block collapses and the mask rides along as SDPA's
    ``attn_mask`` rather than surviving as a ``select``.
    """
    q, k, v = _attn_args(C=1, hd=256)
    valid = jnp.ones((128,), jnp.bool_)
    _, prog = _convert(decode_attention, q, k, v, valid)

    assert _count(prog, "scaled_dot_product_attention") == 1
    assert _count(prog, "softmax") == 0
    assert _count(prog, "matmul") == 0
    assert _count(prog, "select") == 0
    # Only the two GQA repeat tiles (k and v) are left; the mask tile is gone.
    assert _count(prog, "tile") == 2
    _assert_softmax_not_decomposed(prog)

    sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
    assert sdpa.inputs.get("attn_mask") is not None, "the mask was dropped, not absorbed"


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


def test_rmsnorm_fuses_to_l2_norm():
    """``fuse_rmsnorm`` — the eight-op chain becomes ``l2_norm`` + one ``mul``.

    ``(1, 1, D)`` needs no reshape: ``l2_norm`` normalizes over the last three
    dims, which for that shape is exactly the last one.
    """
    x = jnp.ones((1, 1, 256), jnp.float16)
    _, prog = _convert(_rmsnorm_const_scale, x)

    assert _count(prog, "l2_norm") == 1
    for op_type in ("reduce_mean", "reduce_sum", "rsqrt", "reshape"):
        assert _count(prog, op_type) == 0, f"unfused RMSNorm leftover: {op_type}"
    # eps' = d * eps, so that l2_norm's sum-of-squares matches mean + 1e-6.
    l2 = next(op for op in _ops(prog) if op.op_type == "l2_norm")
    assert abs(float(l2.inputs["epsilon"].val) - 256 * 1e-6) < 1e-9
    # cast(fp32) -> l2_norm -> mul(sqrt(d)*scale) -> cast(fp16), nothing else.
    assert sum(1 for op in _ops(prog) if op.op_type != "const") == 4


def test_rmsnorm_off_canonical_shape_is_left_alone():
    """``l2_norm`` reduces the last three dims, so a ``(1, L, H, hd)`` q-norm
    would need reshaping around it — measurably a loss, so the pass skips it."""
    for shape in ((1, 4, 8, 256), (1, 128, 256)):
        _, prog = _convert(_rmsnorm_const_scale, jnp.ones(shape, jnp.float16))
        assert _count(prog, "l2_norm") == 0, shape
        assert _count(prog, "reduce_mean") == 1, shape
        assert _count(prog, "rsqrt") == 1, shape


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


def test_adjacent_rmsnorms_have_no_cast_roundtrip():
    """``collapse_cast_chains`` — coremltools keeps lossy downcast→upcast pairs."""
    x = jnp.ones((1, 1, 256), jnp.float16)
    scale = jnp.ones((256,), jnp.float16)
    _, prog = _convert(double_rmsnorm, x, scale, scale)

    assert _count(prog, "l2_norm") == 2
    assert _cast_roundtrips(prog) == 0


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
    assert _count(prog, "scaled_dot_product_attention") == 1
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
    assert _count(prog, "scaled_dot_product_attention") == 1
    out = _predict(model, q, k, v, valid.astype(np.float32))

    assert np.max(np.abs(ref - out)) < 1e-3


def test_numerical_logit_softcap():
    x = np.random.RandomState(0).randn(1, 4, 32).astype(np.float16)
    ref = np.array(logit_softcap(jnp.array(x)))

    model, _ = _convert(logit_softcap, jnp.array(x), load=True)
    out = _predict(model, x)

    np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)


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

def test_logit_projection_stays_unquantized_fp16():
    """The [dim, vocab] logit projection must reach the model as a plain const.

    Quantizing it at all is a trap in both directions.  int4 cannot carry the
    logits (per-channel scales leave it no grouping to fall back on), and int8
    — the width that could — is what MPSGraph's ``LowerDequantizeND`` pass
    constant-folds on the *first* prediction of every function in every
    process: measured at 16.7 s and an 18 GB transient for the real
    [1536, 262144] tensor on CPU_AND_GPU, never cached to disk.  A plain fp16
    const has no dequantize op for anything to fold.

    The gather table in the same graph is the control: it is the same size and
    over the same threshold, and it *does* get quantized.
    """
    rng = np.random.RandomState(5)
    vocab = 262144
    # Distinct embedding/logit dims, so neither tensor can be mistaken for the
    # other's transpose when the converter picks a matmul orientation.
    table = (rng.randn(vocab, 32) * 0.05).astype(np.float16)
    hidden = (rng.randn(32, 64) * 0.1).astype(np.float16)
    logit_w = (rng.randn(64, vocab) * 0.02).astype(np.float16)

    def fn(tokens):
        embedded = _embed_lookup(jnp.asarray(table), tokens)
        return jnp.matmul(jnp.matmul(embedded, jnp.asarray(hidden)),
                          jnp.asarray(logit_w))

    _, prog = _convert(fn, jnp.zeros((1, 4), jnp.int32))

    def _shapes(op_type):
        return {tuple(op.outputs[0].shape)
                for op in _ops(prog) if op.op_type == op_type}

    quantized = _shapes("constexpr_blockwise_shift_scale")
    assert not ({logit_w.shape, logit_w.T.shape} & quantized), (
        f"the logit projection was quantized (constexprs: {quantized}); "
        "MPSGraph folds int8 dequantizes and int4 is too lossy here"
    )
    assert table.shape in quantized, "the embedding table stopped being quantized"

    # It reaches the model as a plain fp16 const instead.
    logit_consts = [
        op for op in _ops(prog)
        if op.op_type == "const"
        and tuple(op.outputs[0].shape) in (logit_w.shape, logit_w.T.shape)
    ]
    assert len(logit_consts) == 1, [op.name for op in logit_consts]
    assert logit_consts[0].outputs[0].dtype == mil_types.fp16
