"""Build the MIL pass pipeline used after StableHLO→MIL for Gemma export."""

from __future__ import annotations

import coremltools as ct
from stablehlo_coreml import build_pass_pipeline


def build_ct_convert_pass_pipeline() -> ct.PassPipeline:
    """Return the stablehlo-coreml pipeline plus this project's own passes.

    ``stablehlo_coreml.build_pass_pipeline`` supplies the cleanup group
    (``remove_broadcast_tiles``, ``fuse_reduce_keep_dims``, dce,
    ``remove_noop_slice_update``) and the fusion group
    (``replace_decomposed_softmax``, ``fuse_attention_to_sdpa``,
    ``fuse_logit_softcap``, ``fuse_gelu_erfc``) on top of
    ``ct.PassPipeline.DEFAULT``.
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
    # ``common::fuse_rmsnorm`` is no longer inserted here: stablehlo-coreml
    # 0.1.5 ships this project's RMSNorm fusion upstream and already places it
    # in its own late-fusion group, right after ``common::fuse_reduce_mean``.
    return pipeline
