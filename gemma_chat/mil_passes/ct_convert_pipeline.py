"""Build the MIL pass pipeline used after StableHLO→MIL for Gemma export."""

from __future__ import annotations

import copy

import coremltools as ct


_backend_patched = False


def _patch_backend_pipeline():
    """Append replace_scalar_broadcasts to the backend_mlprogram pipeline.

    The backend pipeline runs AFTER the main (DEFAULT) pipeline and contains
    two ``const_elimination`` passes that fold ``fill`` ops back into
    materialized constants.  By appending our pass at the end of the backend
    pipeline we ensure fill ops survive to serialization.
    """
    global _backend_patched
    if _backend_patched:
        return
    import gemma_chat.mil_passes.replace_scalar_broadcasts  # noqa: F401
    from coremltools.converters.mil.mil.passes.pass_pipeline import (
        _BACKEND_MIL_PASSES,
    )
    _BACKEND_MIL_PASSES.append("common::replace_scalar_broadcasts")
    _backend_patched = True


def build_ct_convert_pass_pipeline():
    """Build the MIL pass pipeline for int8 export, then defaults + remove_noop."""
    from stablehlo_coreml.passes.remove_noop_slice_update import (  # noqa: F401
        remove_noop_slice_update,
    )
    import gemma_chat.mil_passes.quantize_const_weights  # noqa: F401
    import gemma_chat.mil_passes.replace_erf_gelu  # noqa: F401
    import gemma_chat.mil_passes.collapse_reshape_chains  # noqa: F401
    import gemma_chat.mil_passes.remove_redundant_maximum  # noqa: F401
    import gemma_chat.mil_passes.remove_broadcast_tiles  # noqa: F401

    _patch_backend_pipeline()

    pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
    pipeline.insert_pass(0, "common::quantize_const_weights")
    pipeline.append_pass("common::remove_noop_slice_update")
    pipeline.append_pass("common::replace_erf_gelu")
    pipeline.append_pass("common::collapse_reshape_chains")
    pipeline.append_pass("common::remove_redundant_maximum")
    pipeline.append_pass("common::remove_broadcast_tiles")
    return pipeline
