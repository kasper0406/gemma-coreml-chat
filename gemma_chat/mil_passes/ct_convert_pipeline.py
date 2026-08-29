"""Build the MIL pass pipeline used after StableHLO→MIL for Gemma export."""

from __future__ import annotations

import coremltools as ct
from stablehlo_coreml import build_pass_pipeline


def build_ct_convert_pass_pipeline() -> ct.PassPipeline:
    """Return upstream's pass pipeline, adjusted for this model.

    The base is ``stablehlo_coreml.build_pass_pipeline()``, which inserts its
    own cleanup, fusion and late-fusion groups into ``ct.PassPipeline.DEFAULT``
    — see that function for what those groups contain. On top of it this adds
    the two passes owned by this repository and drops the three coremltools
    passes the exported model cannot use.
    """
    import gemma_chat.mil_passes.quantize_const_weights  # noqa: F401
    import gemma_chat.mil_passes.collapse_cast_chains  # noqa: F401

    pipeline = build_pass_pipeline()
    # First: weights must be quantized before any pass materializes them.
    pipeline.insert_pass(0, "common::quantize_const_weights")
    # Just before the fusion group, so the fusion passes see fewer casts and
    # the dce entries interleaved with them clean up what this pass orphans.
    pipeline.insert_pass(
        pipeline.passes.index("common::replace_decomposed_softmax"),
        "common::collapse_cast_chains",
    )
    # ``common::fuse_rmsnorm`` is not inserted here: this project's RMSNorm
    # fusion now lives in stablehlo-coreml, which puts it in its own late-fusion
    # group right after ``common::fuse_reduce_mean``. It is not in the 0.1.5
    # release, so this requires the stablehlo-coreml release after 0.1.5.
    pipeline.remove_passes([
        # Callers convert with ``compute_precision=ct.precision.FLOAT32``, which
        # means "leave the dtypes alone", not "compute in fp32": the traced graph
        # already places fp16 and fp32 by hand (see the precision note in
        # ``decode_coreml``), and that is the only setting which preserves that
        # placement — which is why this pass has to go. Left in (or re-run by
        # ct.precision.FLOAT16) it pulls the RMSNorm statistics, the RoPE angles
        # and the ring-position scatter down to fp16 as well; those are fp32 for
        # range and accumulation reasons, and downcasting them is what produced
        # the unstable/garbage-token output seen previously.
        "common::add_fp16_cast",
        # Both of these produce incorrect fusions for this model.
        "common::fuse_layernorm_or_instancenorm",
        "common::fuse_elementwise_to_batchnorm",
    ])
    return pipeline
