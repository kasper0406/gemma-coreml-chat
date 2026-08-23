"""MIL pass: store quantized matmul weights transposed and set ``transpose_y``.

``stablehlo-coreml`` lowers a ``dot_general`` by permuting the rhs into
``[N, K]`` -- emitting an explicit ``mb.transpose`` when the JAX kernel is
stored ``[in, out]`` -- and then calling ``mb.matmul(..., transpose_y=True)``
(``stablehlo_coreml/converter.py``).  ``common::fuse_transpose_matmul``, part
of ``PassPipeline.DEFAULT``, later folds that transpose into the flag, leaving
``matmul(W[K, N], transpose_y=False)``.  Removing a runtime transpose is
normally a win, but when ``W`` is a ``constexpr_blockwise_shift_scale`` output
the resulting orientation is much slower on the GPU backend.  Measured on a
60-pair chain of ``[1536, 12288]`` / ``[12288, 1536]`` int4 weights with the
per-channel scales this project uses (M4 Pro, macOS 26.6.2, ``.cpuAndGPU``)::

    data [K, N], scale [1, N], transpose_y=False    1132 MB    21.49 ms    52.7 GB/s
    data [N, K], scale [N, 1], transpose_y=True     1132 MB     9.96 ms   113.8 GB/s

i.e. **2.16x** on identical weight bytes.  Only the ``[N, K]`` orientation
appears to let the backend fold dequantization into the matmul: in the folded
orientation a 35-layer ``decode_512`` declares ~9.3 GB of dequantized fp16
``constexpr`` output per token, against the 1.7 GB of quantized weights it
actually reads.

End to end on the 35-layer decode graph (ctx 512, measured back to back)::

    baseline                        82.02 ms/token   (~21 GB/s)
    with this pass                  13.58 ms/token   (~127 GB/s)

**6.0x**, which brings the model to within 1.2x of an equivalent MLX
implementation on the same machine.  This pass therefore has to run *after*
``PassPipeline.DEFAULT`` -- see ``materialize.py``, which applies it on the
final graph and then runs dce to drop the ``[K, N]`` originals.

The rewrite is exact, not an approximation.  Transposing ``data`` and moving
the per-channel scale from ``[1, N]`` to ``[N, 1]`` keeps every int4 value
paired with the same scale, so ``W`` dequantizes to the transpose of the
original and ``matmul(x, W_T, transpose_y=True)`` is the same product.
Verified on the full model: 12/12 top-1 agreement, max logit delta 0.0000.

Only per-channel scales (``[1, N]``) are rewritten.  The ``[vocab, blocks]``
block-32 embedding tables are left alone: they feed ``gather``, not ``matmul``,
so the orientation is irrelevant to them.
"""

from __future__ import annotations

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


_counter: list = [0, 0]  # [rewritten, bytes]


_reasons: dict = {}


def _no(reason: str) -> bool:
    _reasons[reason] = _reasons.get(reason, 0) + 1
    return False


def _rewritable(op) -> bool:
    """True for ``matmul(x, constexpr(...), transpose_y=False)`` with a 2-D weight."""
    if op.op_type != "matmul":
        return False

    # Classify by what actually feeds ``y`` BEFORE looking at the flag, so the
    # skip histogram distinguishes "already fast" from "not a weight matmul".
    w = op.y
    producer = "input" if (w is None or w.op is None) else w.op.op_type
    ty = None if op.transpose_y is None else bool(op.transpose_y.val)

    if ty:
        return _no(f"transpose_y already True (y from {producer})")
    if w is None or w.op is None:
        return _no("y has no producer")
    if producer != "constexpr_blockwise_shift_scale":
        return _no(f"y produced by {producer}")
    # The weight must not be shared: rewriting rebuilds it for this consumer.
    if len(w.child_ops) != 1:
        return _no(f"weight shared by {len(w.child_ops)} consumers")

    data = w.op.inputs.get("data")
    scale = w.op.inputs.get("scale")
    if data is None or scale is None:
        return _no("missing data/scale")
    if data.val is None or scale.val is None:
        return _no("data/scale not materialized")
    if len(data.val.shape) != 2:
        return _no(f"weight rank {len(data.val.shape)}")
    # Per-channel only: one scale per output column, broadcast along K.
    if scale.val.shape != (1, data.val.shape[1]):
        return _no(f"scale shape {scale.val.shape} vs data {data.val.shape}")
    # An offset would have to be transposed too; symmetric quant has none.
    off = w.op.inputs.get("offset")
    if off is not None and off.val is not None:
        return _no("has offset")
    return True


@block_context_manager
def _rewrite_block(block) -> int:
    rewritten = 0

    for op in list(block.operations):
        for b in op.blocks:
            rewritten += _rewrite_block(b)

        if not _rewritable(op):
            continue

        cx = op.y.op
        data = cx.inputs["data"].val
        scale = cx.inputs["scale"].val
        n_out = data.shape[1]

        # [K, N] -> [N, K]; the per-channel scale rides along as [N, 1].
        #
        # The int4 tag lives in the numpy dtype's *metadata*
        # (``np.dtype(np.int8, metadata={..: types.int4})`` — see
        # quantize_const_weights._INT4_NP_DTYPE), and transposing produces a
        # plain int8 array that drops it.  Re-view under the original dtype, or
        # the weight silently serializes as int8 at twice the size.
        data_t = np.ascontiguousarray(data.T)
        if data.dtype.metadata:
            data_t = data_t.view(data.dtype)
        scale_t = np.ascontiguousarray(scale.reshape(n_out, 1))

        new_w = mb.constexpr_blockwise_shift_scale(
            data=data_t,
            scale=scale_t,
            before_op=op,
            name=cx.name + "_t",
        )
        new_mm = mb.matmul(
            x=op.x,
            y=new_w,
            transpose_x=op.transpose_x.val if op.transpose_x is not None else False,
            transpose_y=True,
            before_op=op,
            name=op.name + "_ty",
        )
        # ``force_replace`` because the weight is a constexpr, which coremltools
        # otherwise refuses to let go of; dropping it here is the whole point.
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_mm,
            no_check_var_types=True,
            force_replace=True,
        )
        block.remove_ops([op])
        # The original [K, N] weight is now dead. Remove it explicitly rather
        # than leaving it for dce, so its buffer is not serialized.
        if not cx.outputs[0].child_ops:
            block.remove_ops([cx])

        rewritten += 1
        _counter[0] += 1
        _counter[1] += data.nbytes

    return rewritten


@register_pass(namespace="common")
class transpose_matmul_weights(AbstractGraphPass):
    """Flip quantized matmul weights to the ``transpose_y=True`` orientation."""

    def apply(self, prog):
        _counter[0] = 0
        _counter[1] = 0
        _reasons.clear()
        for fname in prog.functions:
            n = _rewrite_block(prog.functions[fname])
            print(
                f"  transpose_matmul_weights [{fname}]: rewrote {n} matmul weight(s)"
                + (f", skipped {sum(_reasons.values())}" if _reasons else ""),
                flush=True,
            )
        for why, k in sorted(_reasons.items(), key=lambda t: -t[1]):
            print(f"      skip: {why} x{k}", flush=True)
