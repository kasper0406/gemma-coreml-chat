"""
Monkey-patch stablehlo-coreml StableHloConverter.op_constant for streaming
int8 weights during HLO→MIL (replaces a fork-only _constant_quantizer hook).
Works with PyPI stablehlo-coreml; an upstream optional callback
could replace this later.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

_original_op_constant: Optional[Callable[..., Any]] = None
_streaming_quantizer: Optional[Callable[[np.ndarray, str], Any]] = None
_patch_installed = False


def install_stablehlo_streaming_patch() -> None:
    """Idempotent: replace StableHloConverter.op_constant with a wrapper."""
    global _original_op_constant, _patch_installed
    if _patch_installed:
        return
    from stablehlo_coreml.converter import StableHloConverter

    _original_op_constant = StableHloConverter.op_constant

    def _patched_op_constant(self, context, op):
        from coremltools.converters.mil import Builder as mb

        q = _streaming_quantizer
        if q is None:
            return _original_op_constant(self, context, op)

        constant = np.array(op.value)
        constant = np.reshape(constant, op.result.type.shape)
        result = q(constant, op.result.get_name())
        if result is not None:
            del constant
            context.add_result(op.result, result)
            return
        context.add_result(op.result, mb.const(val=constant))

    StableHloConverter.op_constant = _patched_op_constant

    # The @register_stablehlo_op decorator caches a direct function reference
    # in _stablehlo_ops_registry at class-definition time.  We must update
    # that registry too, otherwise the dispatch table still calls the original.
    from jaxlib.mlir.dialects._stablehlo_ops_gen import ConstantOp
    StableHloConverter._stablehlo_ops_registry[ConstantOp] = _patched_op_constant

    _patch_installed = True


def set_streaming_quantizer(fn: Optional[Callable[[np.ndarray, str], Any]]) -> None:
    global _streaming_quantizer
    _streaming_quantizer = fn


def clear_streaming_quantizer() -> None:
    set_streaming_quantizer(None)
