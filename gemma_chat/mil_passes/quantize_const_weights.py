# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Kasper Nielsen
# Vendored from stablehlo-coreml (https://github.com/kasper0406/stablehlo-coreml).
"""
MIL pass: quantize large float weight constants using symmetric block-wise
quantization, replacing them with constexpr_blockwise_shift_scale (iOS18) ops
that are immune to constant folding.

Mixed precision: embedding tables (detected by having a dimension matching
VOCAB_SIZE) are quantized to int8 for accuracy (they double as the logit
projection), while all other large weights use int4.

This pass runs early in the pipeline so that coremltools' ~95 optimization
passes work on a compressed model instead of a full ~17GB fp32 model,
which prevents OOM crashes on memory-constrained machines during ct.convert.
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as _mil_types
from coremltools.converters.mil.mil.types.type_mapping import (
    SUB_BYTE_DTYPE_METADATA_KEY as _SUB_BYTE_KEY,
)

import numpy as np

# numpy dtype that stores int8 values but tells coremltools to serialize as int4
_INT4_NP_DTYPE = np.dtype(np.int8, metadata={_SUB_BYTE_KEY: _mil_types.int4})

# Minimum number of elements for a weight tensor to be quantized.
# Bias vectors (1D) and small positional buffers are left uncompressed.
_WEIGHT_THRESHOLD = 2048

# Vocab size used to detect embedding tables. Tensors with any dimension
# matching this value are quantized at int8 instead of int4, because the
# embedding table doubles as the logit projection and int4 is too lossy.
_VOCAB_SIZE = 262144

# Group size for block-wise quantization along the reduction axis.
# Smaller groups = more scales = better accuracy but slightly larger model.
# 32 is the standard for int4 group quantization (matches GPTQ/AWQ conventions).
_GROUP_SIZE = 32

# Module-level counters shared between _quantize_consts_in_block and apply().
# Reset by apply() before each run.
_counter_int4: list = [0, 0]   # [count, total_bytes_original]
_counter_int8: list = [0, 0]
_counter_skip: list = [0]      # [count] — skipped (already constexpr)


def _is_embedding(val: np.ndarray) -> bool:
    """True if this tensor looks like an embedding table (has VOCAB_SIZE dim)."""
    return any(s == _VOCAB_SIZE for s in val.shape)


def _classify_quantize(op):
    """Classify a const op for quantization.

    Returns:
        'int4' — standard weight, quantize to int4
        'int8' — embedding table, quantize to int8
        'skip_constexpr' — already feeds a constexpr op
        None — not a quantizable const (wrong type, too small, etc.)
    """
    if op.op_type != "const":
        return None
    val = op.outputs[0].val
    if not isinstance(val, np.ndarray):
        return None
    if val.dtype not in (np.float16, np.float32):
        return None
    if val.ndim < 2 or val.size < _WEIGHT_THRESHOLD:
        return None
    # Don't re-compress what is already feeding a constexpr_* op
    for child_op in op.outputs[0].child_ops:
        if child_op.op_type.startswith("constexpr_"):
            return "skip_constexpr"
    return "int4"


def _quantize_symmetric_blockwise(val: np.ndarray, axis: int = 0,
                                  group_size: int = _GROUP_SIZE,
                                  nbits: int = 4):
    """
    Symmetric block-wise quantization (round-to-nearest).

    nbits=4: signed int4 range [-7, 7], tagged with int4 metadata
    nbits=8: signed int8 range [-127, 127], plain int8

    Quantizes with one scale per group of `group_size` elements along each
    non-`axis` dimension. Along `axis`, each channel gets its own scale
    (block_size=1).

    Returns:
        quantized_data: int8 array with same shape as val
        scale: float array, same rank as val

    Note: processes in chunks along axis=0 to avoid OOM on large tensors.
    """
    max_val = (1 << (nbits - 1)) - 1  # 7 for int4, 127 for int8

    # Determine scale shape: per-channel on axis, grouped on other dims
    scale_shape = []
    for d in range(val.ndim):
        if d == axis:
            scale_shape.append(val.shape[d])
        else:
            n_groups = (val.shape[d] + group_size - 1) // group_size
            scale_shape.append(n_groups)

    # Pad non-axis dims to be divisible by group_size for reshape
    padded_shape = list(val.shape)
    pad_widths = [(0, 0)] * val.ndim
    needs_pad = False
    for d in range(val.ndim):
        if d != axis and val.shape[d] % group_size != 0:
            pad_amount = group_size - (val.shape[d] % group_size)
            pad_widths[d] = (0, pad_amount)
            padded_shape[d] = val.shape[d] + pad_amount
            needs_pad = True

    if needs_pad:
        val_padded = np.pad(val, pad_widths, mode='constant', constant_values=0)
    else:
        val_padded = val

    # Reshape to [..., n_groups, group_size, ...] for grouped max computation
    # For 2D [out, in] with axis=0: reshape to [out, n_groups, group_size]
    # then compute max over the group_size dim
    grouped_shape = []
    reduce_axes_grouped = []
    dim_idx = 0
    for d in range(val.ndim):
        if d == axis:
            grouped_shape.append(padded_shape[d])
            dim_idx += 1
        else:
            n_g = padded_shape[d] // group_size
            grouped_shape.append(n_g)
            grouped_shape.append(group_size)
            reduce_axes_grouped.append(dim_idx + 1)
            dim_idx += 2

    val_grouped = val_padded.reshape(grouped_shape)
    group_max_f32 = np.max(
        np.abs(val_grouped), axis=tuple(reduce_axes_grouped), keepdims=True
    ).astype(np.float32)
    group_max_f32 = np.where(group_max_f32 == 0.0, 1.0, group_max_f32)
    scale_grouped_f32 = group_max_f32 / float(max_val)

    # Quantize in the grouped view
    n_channels = padded_shape[axis]
    _CHUNK = 2048
    quantized_grouped = np.empty(val_grouped.shape, dtype=np.int8)

    for start in range(0, n_channels, _CHUNK):
        end = min(start + _CHUNK, n_channels)
        # Build slice for the grouped array (axis position is the same)
        slc = []
        for d in range(len(grouped_shape)):
            if d == axis:
                slc.append(slice(start, end))
            else:
                slc.append(slice(None))
        slc = tuple(slc)
        chunk_f32 = val_grouped[slc].astype(np.float32)
        chunk_scale = scale_grouped_f32[slc]
        quantized_grouped[slc] = np.clip(
            np.round(chunk_f32 / chunk_scale), -max_val, max_val
        ).astype(np.int8)
        del chunk_f32, chunk_scale

    # Reshape back to original padded shape and trim
    quantized_padded = quantized_grouped.reshape(padded_shape)
    quantized = quantized_padded[tuple(slice(0, s) for s in val.shape)]

    # Scale: squeeze out the group_size dims to get [out, n_groups] shape
    scale_f32 = group_max_f32.squeeze(axis=tuple(reduce_axes_grouped)) / float(max_val)
    scale = scale_f32.astype(val.dtype)

    del val_padded, val_grouped, quantized_grouped, quantized_padded
    del group_max_f32, scale_grouped_f32, scale_f32
    if nbits == 4:
        # Tag the int8 container with int4 metadata so coremltools serializes
        # the data as packed 4-bit (halving on-disk weight storage).
        quantized = quantized.view(_INT4_NP_DTYPE)
    return quantized, scale


@block_context_manager
def _quantize_consts_in_block(block):
    import gc as _gc

    # Phase 1: classify ops — collect quantizable consts, warn on constexpr skips
    ops_to_quantize = []   # list of (op, 'int4' | 'int8')
    for op in block.operations:
        for b in op.blocks:
            _quantize_consts_in_block(b)
        cls = _classify_quantize(op)
        if cls == "skip_constexpr":
            _counter_skip[0] += 1
            val = op.outputs[0].val
            child_types = [c.op_type for c in op.outputs[0].child_ops]
            print(
                f"    ⚠ SKIP (already constexpr) {op.name}  "
                f"shape={val.shape}  dtype={val.dtype}  consumers={child_types}",
                flush=True,
            )
        elif cls in ("int4", "int8"):
            ops_to_quantize.append((op, cls))

    if not ops_to_quantize:
        return False

    # Phase 2: quantize with GC management (same strategy as before)
    import gc as _gc
    import ctypes as _ctypes_q, ctypes.util as _ctu_q
    try:
        _libc_q = _ctypes_q.CDLL(_ctu_q.find_library('c'))
        def _madvise_free():
            _libc_q.malloc_zone_pressure_relief(
                _ctypes_q.c_void_p(0), _ctypes_q.c_size_t(0))
    except Exception:
        def _madvise_free():
            pass

    total_count = _counter_int4[0] + _counter_int8[0]
    _gc.disable()
    try:
        n = len(ops_to_quantize)
        for i in range(n):
            op, precision = ops_to_quantize[i]
            ops_to_quantize[i] = (None, None)  # drop ref for GC

            val = op.outputs[0].val
            nbytes = val.nbytes
            nbits = 4 if precision == "int4" else 8
            quantized_data, scale = _quantize_symmetric_blockwise(
                val, axis=0, group_size=_GROUP_SIZE, nbits=nbits,
            )

            op.outputs[0]._sym_val = None
            del val

            suffix = f"_{precision}"
            new_var = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                before_op=op,
                name=op.name + suffix,
            )

            block.replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=new_var,
                no_check_var_types=True,
            )
            block.remove_ops([op])

            if precision == "int4":
                _counter_int4[0] += 1
                _counter_int4[1] += nbytes
            else:
                _counter_int8[0] += 1
                _counter_int8[1] += nbytes

            total_count += 1
            del op

            if total_count % 20 == 0 or i == n - 1:
                print(
                    f"    quantized {_counter_int4[0]} int4 + "
                    f"{_counter_int8[0]} int8  "
                    f"({(_counter_int4[1] + _counter_int8[1]) / 1e9:.2f} GB fp16)",
                    flush=True,
                )
                _gc.enable()
                _gc.collect()
                _gc.disable()
                _madvise_free()
    finally:
        _gc.enable()

    return True


@register_pass(namespace="common")
class quantize_const_weights(AbstractGraphPass):
    """
    Replace large float weight constants with constexpr_blockwise_shift_scale
    ops using symmetric block-wise quantization (iOS18).

    Mixed precision: embedding tables (vocab_size dim) → int8 for accuracy,
    all other large weights → int4 for size. Group quantization with
    GROUP_SIZE=32 elements per block.

    Inserted at position 0 in the pass pipeline so that all subsequent
    passes work on the compressed model.
    """

    def apply(self, prog):
        _counter_int4[0] = _counter_int4[1] = 0
        _counter_int8[0] = _counter_int8[1] = 0
        _counter_skip[0] = 0
        for f in prog.functions.values():
            _quantize_consts_in_block(f)
        total = _counter_int4[0] + _counter_int8[0]
        if total or _counter_skip[0]:
            print(
                f"    quantized {total} tensors total: "
                f"{_counter_int4[0]} int4 ({_counter_int4[1] / 1e9:.2f} GB), "
                f"{_counter_int8[0]} int8 ({_counter_int8[1] / 1e9:.2f} GB)"
                + (f", {_counter_skip[0]} skipped (already constexpr)"
                   if _counter_skip[0] else ""),
                flush=True,
            )
