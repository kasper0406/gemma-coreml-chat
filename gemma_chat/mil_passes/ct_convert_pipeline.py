"""Build the MIL pass pipeline used after StableHLO→MIL for Gemma export."""

from __future__ import annotations

import coremltools as ct
from stablehlo_coreml import build_pass_pipeline


def build_ct_convert_pass_pipeline() -> ct.PassPipeline:
    """Return upstream's pass pipeline, adjusted for this model.

    The base is ``stablehlo_coreml.build_pass_pipeline()``, which inserts its
    own cleanup, fusion and late-fusion groups into ``ct.PassPipeline.DEFAULT``
    — see that function for what those groups contain. On top of it this adds
    the three passes owned by this repository and drops the passes the exported
    model cannot use.
    """
    import gemma_chat.mil_passes.quantize_const_weights  # noqa: F401
    import gemma_chat.mil_passes.collapse_cast_chains  # noqa: F401
    import gemma_chat.mil_passes.fp16_l2_norm  # noqa: F401

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
    # group right after ``common::fuse_reduce_mean`` (stablehlo-coreml >= 0.1.6).
    # Right behind it, move the fused norms from fp32 to fp16.
    pipeline.insert_pass(
        pipeline.passes.index("common::fuse_rmsnorm") + 1, "common::fp16_l2_norm",
    )
    pipeline.remove_passes([
        # Callers convert with ``compute_precision=ct.precision.FLOAT32``, which
        # means "leave the dtypes alone", not "compute in fp32": the traced graph
        # already places fp16 and fp32 by hand (see the precision note in
        # ``decode_coreml``), and that is the only setting which preserves that
        # placement — which is why this pass has to go. Left in (or re-run by
        # ct.precision.FLOAT16) it pulls the RoPE angles and the ring-position
        # scatter down to fp16 as well; those are fp32 for range reasons, and
        # downcasting them (together with the RMSNorm statistics, which were
        # fp32 then) is what produced the unstable/garbage-token output seen
        # previously.
        "common::add_fp16_cast",
        # Both of these produce incorrect fusions for this model.
        "common::fuse_layernorm_or_instancenorm",
        "common::fuse_elementwise_to_batchnorm",
        # The Neural Engine ignores ``scaled_dot_product_attention``'s
        # ``attn_mask`` (macOS 27, M4 Pro): in the chunk-prefill graph every
        # query attended to all cache slots, empty and future ones included,
        # and ``cpu-and-ne`` / ``all`` produced garbage — additive and boolean
        # masks alike, while the same op on CPU/GPU was correct.  Attention
        # therefore stays ``matmul -> select -> softmax -> matmul``, which the
        # ANE runs correctly.  It costs nothing: prefill time is unchanged on
        # the GPU and the ANE and ~25% lower on the CPU.  (Decode attention
        # never matched the fusion, and the global sites are kept decomposed
        # for two more Apple defects — see ``materialize._concretize_cache_lengths``.)
        "common::fuse_attention_to_sdpa",
    ])
    return pipeline
