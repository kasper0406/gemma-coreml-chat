"""The logit head: int8, one scale per vocab row, in equal slices the ANE streams fast.

``head`` is the only function whose weights are int8 (int4 is too lossy for
logits) and it has to run on the Neural Engine under ``cpu-and-ne``, which
takes per-channel scales but not block-wise ones.  Its vocab is split into
slices sized so that no ANE core's share of a slice lands in the weight-DMA
"notch" around multiples of 1 MiB (``decode_coreml.head_slices``).
"""

from __future__ import annotations

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma_chat.config import E2B_CONFIG
from gemma_chat.decode_coreml import ane_core_payload, head_slices, in_ane_notch, logits_head

MIB = 1 << 20


@pytest.mark.parametrize("payload_mib, slow", [
    # The measured head layouts (M4 Pro, macOS 27; see decode_coreml).
    (3.000, True),    # 8 vocab slices: 13.6 ms
    (2.667, False),   # 9 slices: 5.69 ms
    (2.401, False),   # 10 slices: 5.76 ms
    (1.500, False),   # 8 slices x 2 contracting splits: 5.80 ms
    (1.000, True),    # 8 slices x 3 contracting splits: 13.7 ms
    (0.375, False),   # an int4 layer weight
])
def test_the_notch_matches_the_measurements(payload_mib, slow):
    assert in_ane_notch(round(payload_mib * MIB)) == slow


def test_the_e2b_head_is_sixteen_equal_slices_clear_of_the_notch():
    vocab, dim = E2B_CONFIG.num_embed, E2B_CONFIG.embed_dim
    slices = head_slices(vocab, dim)
    assert slices[0][0] == 0 and slices[-1][1] == vocab
    assert all(a == b for (_, a), (b, _) in zip(slices, slices[1:]))
    # Eight slices of 32768 rows would be the fewest the row cap allows —
    # and exactly 3 MiB per core.
    assert in_ane_notch(ane_core_payload(vocab // 8, dim))
    assert len(slices) == 16
    for a, b in slices:
        assert b - a == 16384
        payload = ane_core_payload(b - a, dim)
        assert not in_ane_notch(payload), (a, b, payload / MIB)


def test_a_small_vocab_is_one_slice():
    assert head_slices(100, 64) == [(0, 100)]


def _convert_head(params, cfg):
    """``head`` the way the export converts it: streaming int8 quantization,
    the export pipeline, then materialization's weight transpose."""
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    from gemma_chat.export import _hlo_to_mil_streaming
    from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline
    from gemma_chat.mil_passes.transpose_matmul_weights import transpose_matmul_weights

    spec = jax.ShapeDtypeStruct((1, 1, cfg.embed_dim), jnp.float16)
    hlo = jax.jit(lambda h: logits_head(params, h)).trace(spec).lower().compiler_ir("stablehlo")
    prog = _hlo_to_mil_streaming(hlo, {}, weight_bits=8)
    prog = ct.convert(
        prog, source="milinternal", minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32, pass_pipeline=build_ct_convert_pass_pipeline(),
        skip_model_load=True,
    )._mil_program
    transpose_matmul_weights().apply(prog)
    PASS_REGISTRY["common::dead_code_elimination"](prog)
    model = ct.convert(
        prog, source="milinternal", minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32, pass_pipeline=ct.PassPipeline.EMPTY,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    return prog, model


def test_the_head_exports_as_int8_per_channel_slices():
    import dataclasses

    cfg = dataclasses.replace(E2B_CONFIG, num_embed=65536, embed_dim=64)
    rng = np.random.default_rng(0)
    table = (rng.standard_normal((cfg.num_embed, cfg.embed_dim)) * 0.05).astype(np.float16)
    table[7] *= np.float16(20)  # a row whose scale is far from the others
    params = {"embed_tokens": table}
    slices = head_slices(cfg.num_embed, cfg.embed_dim)
    assert len(slices) == 2

    prog, model = _convert_head(params, cfg)
    matmuls = [op for op in prog.functions["main"].operations if op.op_type == "matmul"]
    assert len(matmuls) == len(slices)
    for op, (a, b) in zip(matmuls, slices):
        w = op.y.op
        assert w.op_type == "constexpr_blockwise_shift_scale"
        data, scale = w.inputs["data"].val, w.inputs["scale"].val
        # int8 (no int4 tag), [rows, dim] with one scale per vocab row, and
        # the transpose_y orientation the GPU runs fast.
        assert data.dtype == np.int8 and data.dtype.metadata is None
        assert data.shape == (b - a, cfg.embed_dim) and scale.shape == (b - a, 1)
        assert op.transpose_y.val

    hidden = (rng.standard_normal((1, 1, cfg.embed_dim)) * 2).astype(np.float16)
    want = np.asarray(logits_head(params, jnp.asarray(hidden)), np.float32)
    assert want.shape == (len(slices), cfg.num_embed // len(slices))
    want = want.reshape(-1)
    (name,) = [i.name for i in model.get_spec().description.input]
    (got,) = model.predict({name: hidden}).values()
    got = np.asarray(got, np.float32).reshape(-1)
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 0.05 * np.abs(want).max()
    assert got.argmax() == want.argmax()
