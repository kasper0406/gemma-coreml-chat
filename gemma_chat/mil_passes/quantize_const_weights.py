# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Kasper Nielsen
# Vendored from stablehlo-coreml (https://github.com/kasper0406/stablehlo-coreml).
"""
MIL pass: quantize large float weight constants, replacing them with
constexpr_blockwise_shift_scale (iOS18) ops that are immune to constant folding.

Everything quantized here is **int4**; the scale granularity is picked by what
consumes the tensor (see :func:`_quantize_weight`):

* **Per-channel int4** for matmul weights — one scale per *output* channel, each
  block spanning the whole input (contraction) axis.  Required for the ANE: the
  CoreML ANE backend refuses to run any op fed by a
  constexpr_blockwise_shift_scale whose scales are grouped along the contraction
  axis (e.g. [D, O/32] for a [D, O] weight), silently pushing the whole model
  onto the CPU.  Scales of shape [1, O] are ANE-eligible.
  These weights are laid out [input_dim, output_dim] — they feed ``matmul`` as
  the ``y`` operand with ``transpose_y=False`` — so the output axis is the last
  one and the scale shape is [1, O].
* **Block-32 int4** for the [VOCAB_SIZE, dim] embedding lookup tables.  They
  feed ``gather``, which never runs on the ANE anyway, so there is nothing to
  gain from per-channel scales and real accuracy to lose: one block would span
  all 262144 vocab rows.  Scale shape stays [V, D/32].
* **Block-32 int8** for the [dim, VOCAB_SIZE] logit projection.  This used to
  be left as plain fp16, for two reasons that no longer hold now that weights
  reach the matmul as [N, K] with ``transpose_y=True`` (see
  ``mil_passes.transpose_matmul_weights``):

  - *"int8 makes MPSGraph constant-fold on every first prediction."*  That is
    specific to the old [K, N] / ``transpose_y=False`` orientation.  Measured on
    a fresh process at N=262144: int8 per-channel takes **16.3 s** to first
    predict in the old orientation and **0.06 s** in the new one, the same as
    fp16.  There is nothing left to avoid.
  - *"int4 is too lossy for logits."*  Still true, and int4 is no faster here:
    2.02 vs 2.11 ms at M=1 and 43.2 vs 43.1 ms at M=128.  int8 it is.

  Block-32 rather than per-channel is **load-bearing for correctness**, not a
  performance choice: in the [N, K] / ``transpose_y=True`` orientation Core ML's
  int8 *per-channel* matmul returns uncorrelated garbage once N >= 65536 and
  M >= 5 (measured relRMS 1.0 against fp16 at N=65536 for M=5 and M=128; correct
  at M <= 4, at N <= 49152, and for block-32 at every M).  Prefill runs this head
  at M = CHUNK_SIZE = 128, so per-channel would silently corrupt every prompt
  while looking ~5x faster.  Block-32 is unaffected.

  Cost: the head shrinks 805 MB -> 403 MB and gets *faster* in both phases —
  3.58 -> 2.11 ms at M=1, 55.5 -> 43.1 ms at M=128.
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

# Vocab size used to detect the embedding lookup tables, which keep block-32
# scales because they feed CPU-side gathers rather than ANE matmuls.
_VOCAB_SIZE = 262144

# Module-level counters shared between _quantize_consts_in_block and apply().
# Reset by apply() before each run.
_counter_int4: list = [0, 0]   # [count, total_bytes_original]

# Set per program in ``apply``: True while quantizing a decode graph, False for
# prefill.  The logit head is int8 only for decode -- see ``_classify_quantize``.
_quantize_logit_head: list = [False]
_counter_skip: list = [0]      # [count] — skipped (already constexpr)


def _is_embedding(val: np.ndarray) -> bool:
    """True if this tensor is an embedding lookup table: [VOCAB_SIZE, dim].

    Only the *leading* dim counts.  The logit projection is [dim, VOCAB_SIZE]
    and feeds a matmul, so it deliberately does not match.
    """
    return val.ndim == 2 and val.shape[0] == _VOCAB_SIZE


def _is_logit_projection(val: np.ndarray) -> bool:
    """True if this tensor is the [dim, VOCAB_SIZE] logit projection.

    Vocab on the *last* axis, so it is the ``y`` operand of the final matmul
    rather than a gather table.  It is left unquantized — see the module
    docstring for why int4 is too lossy and int8 is unlowerable.
    """
    return (val.ndim == 2 and val.shape[-1] == _VOCAB_SIZE
            and val.shape[0] != _VOCAB_SIZE)


def _classify_quantize(op):
    """Classify a const op for quantization.

    Returns:
        'int4' — quantizable weight
        'skip_constexpr' — already feeds a constexpr op
        None — not a quantizable const (wrong type, too small, logit projection)
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
    # The logit projection stays a plain fp16 const.  It is far over the size
    # threshold, so it has to be excluded explicitly — quantizing it at all is
    # what makes the first prediction of every function cost ~17 s under
    # MPSGraph (int8) or the logits inaccurate (int4).  See module docstring.
    if _is_logit_projection(val):
        # Decode only.  In situ the int8 head is worth ~+4% decode but costs
        # ~22% of prefill (measured 1117 -> 870 tok/s at ctx 400), because
        # prefill runs it at M = CHUNK_SIZE while decode runs it at M = 1.
        # Prefill therefore keeps the plain fp16 const.  The two phases are
        # converted separately, so they simply end up with different weights.
        return "int8_logit" if _quantize_logit_head[0] else None
    # Don't re-compress what is already feeding a constexpr_* op
    for child_op in op.outputs[0].child_ops:
        if child_op.op_type.startswith("constexpr_"):
            return "skip_constexpr"
    return "int4"


def _quantize_symmetric_per_channel(val: np.ndarray):
    """
    Symmetric per-channel int4 quantization (round-to-nearest, range [-7, 7]).

    One scale per output channel: the last axis of `val` is the output axis
    (weights feed ``matmul`` as ``y`` with ``transpose_y=False``, i.e. they are
    laid out [input_dim, output_dim]), and each block spans the entire
    remaining (contraction) extent.  For a [D, O] weight the scale is [1, O].
    This is the only granularity the ANE will accept — grouped scales along the
    contraction axis force the whole model onto the CPU.

    Returns:
        quantized_data: int4-tagged int8 array with same shape as val
        scale: float array, same rank as val, shape [1, ..., 1, O]

    Note: processes in chunks along axis 0 to avoid OOM on large tensors.
    """
    max_val = 7

    reduce_axes = tuple(range(val.ndim - 1))
    n_rows = val.shape[0]
    _CHUNK = 2048

    # Pass 1: per-output-channel absmax, accumulated chunk-wise in fp32.
    chan_max_f32 = np.zeros((val.shape[-1],), dtype=np.float32)
    for start in range(0, n_rows, _CHUNK):
        chunk_f32 = val[start:start + _CHUNK].astype(np.float32)
        np.maximum(
            chan_max_f32,
            np.max(np.abs(chunk_f32), axis=reduce_axes),
            out=chan_max_f32,
        )
        del chunk_f32
    chan_max_f32 = np.where(chan_max_f32 == 0.0, 1.0, chan_max_f32)

    # Store the scale in the weight dtype and quantize against that exact
    # value, so dequantization on-device reproduces what we rounded against.
    scale_shape = (1,) * (val.ndim - 1) + (val.shape[-1],)
    scale = (chan_max_f32 / float(max_val)).astype(val.dtype).reshape(scale_shape)
    scale_f32 = scale.astype(np.float32)

    # Pass 2: quantize chunk-wise.
    quantized = np.empty(val.shape, dtype=np.int8)
    for start in range(0, n_rows, _CHUNK):
        end = min(start + _CHUNK, n_rows)
        chunk_f32 = val[start:end].astype(np.float32)
        quantized[start:end] = np.clip(
            np.round(chunk_f32 / scale_f32), -max_val, max_val
        ).astype(np.int8)
        del chunk_f32

    del chan_max_f32, scale_f32
    # Tag the int8 container with int4 metadata so coremltools serializes
    # the data as packed 4-bit (halving on-disk weight storage).
    quantized = quantized.view(_INT4_NP_DTYPE)
    return quantized, scale


def _quantize_symmetric_embedding_blocks(val: np.ndarray, bits: int = 4):
    """
    Symmetric block-wise int4 (or int8) quantization, 32 elements per group.

    One scale per group of 32 elements along the row (embedding-dim) axis, so
    the scale is [V, D/32].  Deliberately *not* per-channel: these tensors feed
    ``gather``, which runs on the CPU no matter how the weights are quantized,
    and a per-channel block would span all 262144 vocab rows.  The block size is
    fixed at 32 — this is the one path that keeps grouped scales, not a knob.

    Returns:
        quantized_data: int4-tagged int8 array with same shape as val
        scale: float array [V, D/32]

    Note: processes in chunks along axis 0 to avoid OOM on large tensors.
    """
    _GROUP = 32
    if val.ndim != 2:
        raise ValueError(f"table must be rank 2, got shape {val.shape}")
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    max_val = 7 if bits == 4 else 127

    n_rows, n_cols = val.shape
    pad = (-n_cols) % _GROUP
    n_groups = (n_cols + pad) // _GROUP
    _CHUNK = 2048

    quantized = np.empty(val.shape, dtype=np.int8)
    scale_f32 = np.empty((n_rows, n_groups), dtype=np.float32)

    for start in range(0, n_rows, _CHUNK):
        end = min(start + _CHUNK, n_rows)
        chunk_f32 = val[start:end].astype(np.float32)
        if pad:
            chunk_f32 = np.pad(chunk_f32, ((0, 0), (0, pad)))
        chunk_f32 = chunk_f32.reshape(end - start, n_groups, _GROUP)
        group_max = np.max(np.abs(chunk_f32), axis=2, keepdims=True)
        group_max = np.where(group_max == 0.0, 1.0, group_max)
        group_scale = group_max / float(max_val)
        q = np.clip(
            np.round(chunk_f32 / group_scale), -max_val, max_val
        ).astype(np.int8)
        quantized[start:end] = q.reshape(end - start, n_groups * _GROUP)[:, :n_cols]
        scale_f32[start:end] = group_scale[:, :, 0]
        del chunk_f32, group_max, group_scale, q

    scale = scale_f32.astype(val.dtype)
    del scale_f32
    if bits == 4:
        quantized = quantized.view(_INT4_NP_DTYPE)
    return quantized, scale


def _quantize_weight(val: np.ndarray):
    """Quantize one weight tensor with the granularity its consumer requires.

    Per-channel for matmul weights, so the ANE will accept the ops they feed;
    block-32 for the [VOCAB_SIZE, dim] embedding tables, whose gathers run on
    the CPU regardless and would only lose accuracy from per-channel scales.
    Always int4 — callers must keep the logit projection away from here (see
    :func:`_is_logit_projection`).
    """
    if _is_logit_projection(val):
        # int8, block-32 along the contraction axis.  NOT per-channel: with the
        # weight in its final [N, K] / transpose_y=True orientation, Core ML's
        # int8 per-channel matmul returns uncorrelated garbage once N >= 65536
        # and M >= 5 (measured: relRMS 1.0 vs fp16 at N=65536/M=5 and M=128,
        # correct at M<=4, at N<=49152, and for block-32 at every M).  Prefill
        # runs this head at M = CHUNK_SIZE = 128, so per-channel would silently
        # corrupt every prompt.
        return _quantize_symmetric_embedding_blocks(val, bits=8)
    if _is_embedding(val):
        return _quantize_symmetric_embedding_blocks(val)
    return _quantize_symmetric_per_channel(val)


@block_context_manager
def _quantize_consts_in_block(block):
    import gc as _gc

    # Phase 1: classify ops — collect quantizable consts, warn on constexpr skips
    ops_to_quantize = []
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
        elif cls in ("int4", "int8_logit"):
            ops_to_quantize.append(op)

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

    _gc.disable()
    try:
        n = len(ops_to_quantize)
        for i in range(n):
            op = ops_to_quantize[i]
            ops_to_quantize[i] = None  # drop ref for GC

            val = op.outputs[0].val
            nbytes = val.nbytes
            quantized_data, scale = _quantize_weight(val)

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

            _counter_int4[0] += 1
            _counter_int4[1] += nbytes
            del op

            if _counter_int4[0] % 20 == 0 or i == n - 1:
                print(
                    f"    quantized {_counter_int4[0]} int4  "
                    f"({_counter_int4[1] / 1e9:.2f} GB fp16)",
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
    ops using symmetric quantization (iOS18).

    Matmul weights get int4 with one scale per output channel (last axis), the
    block spanning the whole contraction axis — the only granularity the ANE
    accepts.  The [vocab_size, dim] embedding tables keep block-32 int4: their
    gathers run on the CPU either way, so per-channel would only cost accuracy.
    The [dim, vocab_size] logit projection is left alone as a plain fp16 const.

    Inserted at position 0 in the pass pipeline so that all subsequent
    passes work on the compressed model.
    """

    def apply(self, prog):
        _counter_int4[0] = _counter_int4[1] = 0
        _counter_skip[0] = 0
        # A decode graph takes a single token, prefill takes a chunk -- that is
        # the only signal here, since both phases convert as "main".  Match on a
        # prefix: at this point in the pipeline the inputs still carry their
        # pre-rename names (`token_id_1d`, `position_1d` for decode;
        # `tokens`, `start_pos_1d` for prefill), so an exact match silently
        # never fires and the head quietly stays fp16 in both phases.
        _quantize_logit_head[0] = any(
            name.startswith("token_id")
            for f in prog.functions.values() for name in f.inputs
        )
        for f in prog.functions.values():
            _quantize_consts_in_block(f)
        if _counter_int4[0] or _counter_skip[0]:
            print(
                f"    quantized {_counter_int4[0]} tensors total: int4 "
                f"({_counter_int4[1] / 1e9:.2f} GB fp16)"
                + (f", {_counter_skip[0]} skipped (already constexpr)"
                   if _counter_skip[0] else ""),
                flush=True,
            )
