# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Kasper Nielsen
# Vendored from stablehlo-coreml (https://github.com/kasper0406/stablehlo-coreml).
"""
MIL pass: quantize large float weight constants to int4 using symmetric
block-wise quantization, replacing them with constexpr_blockwise_shift_scale
(iOS18) ops that are immune to constant folding.

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

# Group size for block-wise quantization along the reduction axis.
# Smaller groups = more scales = better accuracy but slightly larger model.
# 32 is the standard for int4 group quantization (matches GPTQ/AWQ conventions).
_GROUP_SIZE = 32

# Module-level counter shared between _quantize_consts_in_block and apply().
# [count, total_bytes].  Reset by apply() before each run.
_counter: list = [0, 0]


def _should_quantize(op):
    """Return True if this const op should be compressed to int4."""
    if op.op_type != "const":
        return False
    val = op.outputs[0].val
    if not isinstance(val, np.ndarray):
        return False
    # Only compress multi-dimensional float tensors large enough to matter
    if val.dtype not in (np.float16, np.float32):
        return False
    if val.ndim < 2 or val.size < _WEIGHT_THRESHOLD:
        return False
    # Don't re-compress what is already feeding a constexpr_* op
    for child_op in op.outputs[0].child_ops:
        if child_op.op_type.startswith("constexpr_"):
            return False
    return True


def _quantize_symmetric_blockwise(val: np.ndarray, axis: int = 0,
                                  group_size: int = _GROUP_SIZE):
    """
    Symmetric block-wise int4 quantization (round-to-nearest).

    Quantizes to signed int4 (range [-7, 7]) with one scale per group of
    `group_size` elements along each non-`axis` dimension. Along `axis`,
    each channel gets its own scale (block_size=1).

    Returns:
        quantized_data: int8 array with same shape as val (holding int4 values)
        scale: float array, same rank as val, with shape:
            - axis dim: same as val (per-channel)
            - other dims: ceil(val.shape[d] / group_size)

    The implied block_size for constexpr_blockwise_shift_scale is:
        block_size[axis] = 1  (per-channel)
        block_size[d] = group_size  (for other dims, if evenly divisible)

    If a dimension is not evenly divisible by group_size, the last group
    is smaller but the scale still covers it (constexpr_blockwise_shift_scale
    broadcasts the scale to cover remaining elements).

    Note: processes in chunks along axis=0 to avoid OOM on large tensors.
    """
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
    scale_grouped_f32 = group_max_f32 / 7.0

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
            np.round(chunk_f32 / chunk_scale), -7, 7
        ).astype(np.int8)
        del chunk_f32, chunk_scale

    # Reshape back to original padded shape and trim
    quantized_padded = quantized_grouped.reshape(padded_shape)
    quantized = quantized_padded[tuple(slice(0, s) for s in val.shape)]

    # Scale: squeeze out the group_size dims to get [out, n_groups] shape
    scale_f32 = group_max_f32.squeeze(axis=tuple(reduce_axes_grouped)) / 7.0
    scale = scale_f32.astype(val.dtype)

    del val_padded, val_grouped, quantized_grouped, quantized_padded
    del group_max_f32, scale_grouped_f32, scale_f32
    # Tag the int8 container with int4 metadata so coremltools serializes
    # the data as packed 4-bit (halving on-disk weight storage).
    quantized = quantized.view(_INT4_NP_DTYPE)
    return quantized, scale


@block_context_manager
def _quantize_consts_in_block(block):
    import gc as _gc

    # Phase 1: scan block.operations once to find const ops and recurse into
    # any sub-blocks.  We do NOT modify the block here, so it is safe to
    # iterate block.operations directly (no snapshot needed).
    #
    # Collecting only const ops avoids iterating tens of thousands of non-const
    # ops in the main quantize loop (Phase 2).  The previous all-ops snapshot
    # loop continued iterating non-const ops *after* the last const op, which
    # triggered Python's automatic GC at an unsafe callstack position.
    const_ops = []
    for op in block.operations:
        for b in op.blocks:
            _quantize_consts_in_block(b)
        if _should_quantize(op):
            const_ops.append(op)

    if not const_ops:
        return False

    # Phase 2: process only the const ops collected above, using slot-nulling
    # so that gc.collect() can break the op↔var cycles and release the C-level
    # fp16 numpy refs before the int8 data accumulates to OOM levels.
    #
    # gc.disable() prevents Python's automatic GC from firing at unpredictable
    # positions inside C extensions (e.g. replace_uses_of_var_after_op iterates
    # ~50,000 ops, creating hundreds of thousands of Python objects per tensor,
    # easily pushing gen0 past threshold 700 which would trigger auto-GC while
    # MLIR/jaxlib tp_finalize objects are on the C stack).
    #
    # Every 20 tensors we: (1) re-enable GC, (2) call gc.collect() at a safe
    # Python boundary, (3) call malloc_zone_pressure_relief() to evict the just-
    # freed fp16 pages from macOS's malloc large-allocation free list.  Without
    # madvise, freed 42 MB pages stay in the process jetsam footprint as
    # compressed pages — 20 tensors × 42 MB = ~840 MB freed per batch; over the
    # full 317-tensor run that's ~13 GB accumulating even though actual RSS is
    # only ~5 GB.  macOS jetsam kills at ~51 GB compressed footprint.
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

    _gc.disable()
    try:
        n = len(const_ops)
        for i in range(n):
            op = const_ops[i]
            const_ops[i] = None  # drop the list's ref so gc.collect() can collect

            val = op.outputs[0].val
            nbytes = val.nbytes
            quantized_data, scale = _quantize_symmetric_blockwise(
                val, axis=0, group_size=_GROUP_SIZE,
            )

            # Clear the Python-visible sym_val ref (belt-and-suspenders).
            # The C-level ref inside the op object is released only when the
            # op is garbage-collected after gc.collect() breaks the op↔var cycle.
            op.outputs[0]._sym_val = None
            del val

            new_var = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                before_op=op,
                name=op.name + "_int4",
            )

            block.replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=new_var,
                no_check_var_types=True,
            )
            block.remove_ops([op])

            _counter[0] += 1
            _counter[1] += nbytes

            del op  # drop local ref; only the op↔var cycle remains

            # Batched flush (every 20 tensors): write(2) releases the GIL
            # briefly, giving background threads a chance to drop MLIR/jaxlib
            # cycle refs before our explicit gc.collect().

            if _counter[0] % 20 == 0 or i == n - 1:
                print(
                    f"    quantized {_counter[0]} tensors  "
                    f"({_counter[1] / 1e9:.2f} GB fp16 → int4)",
                    flush=True,
                )
                # Collect at a safe Python boundary, then immediately re-disable.
                _gc.enable()
                _gc.collect()
                _gc.disable()
                # Evict the just-freed fp16 pages from macOS's large-allocation
                # free list.  Each batch frees ~840 MB of 42 MB numpy arrays;
                # this call marks those pages as MADV_FREE_REUSABLE so macOS
                # stops counting them in the process's jetsam phys_footprint.
                _madvise_free()
    finally:
        _gc.enable()

    return True


@register_pass(namespace="common")
class quantize_const_weights(AbstractGraphPass):
    """
    Replace large float weight constants with constexpr_blockwise_shift_scale
    ops using symmetric block-wise int4 quantization (iOS18).

    Uses group quantization (GROUP_SIZE=32 elements per block) for accuracy.
    This is inserted at position 0 in the pass pipeline so that all of
    coremltools' subsequent optimization passes work on the compressed model
    rather than a ~17GB fp32 model. This prevents OOM crashes and
    write_fp16_data failures during ct.convert on memory-constrained machines.

    constexpr_blockwise_shift_scale (iOS18) is exempt from constant folding,
    so the int4 representation is preserved through to the final .mlpackage.
    """

    def apply(self, prog):
        _counter[0] = 0
        _counter[1] = 0
        for f in prog.functions.values():
            # Single pass is sufficient — all const ops in Gemma4 are at the
            # top level of the main function block (no nesting that would
            # require a second pass).  A while loop here would call
            # list(block.operations) a second time over thousands of ops
            # (including the newly-created constexpr ops), which can crash or
            # OOM before the pass summary is printed.
            _quantize_consts_in_block(f)
        if _counter[0]:
            print(
                f"    quantized {_counter[0]} tensors total  "
                f"({_counter[1] / 1e9:.2f} GB fp16 → int4)",
                flush=True,
            )
