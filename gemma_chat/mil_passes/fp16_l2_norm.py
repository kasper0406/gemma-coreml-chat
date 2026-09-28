"""MIL pass: run the fused RMSNorm ``l2_norm`` on the fp16 activation itself.

``decode_coreml._rmsnorm`` spells RMSNorm with fp32 statistics, which is the
exact definition and what the JAX function computes.  After
``stablehlo_coreml``'s ``fuse_rmsnorm`` every norm in the graph reads::

    %x32 = cast(x=%x, dtype=fp32)          # %x is fp16
    %n   = l2_norm(x=%x32, epsilon=d*eps)
    %s   = mul(x=%n, y=<const sqrt(d)*scale>)
    %y   = cast(x=%s, dtype=fp16)

This pass drops the two casts, so the norm is ``l2_norm`` + ``mul`` in fp16::

    %n   = l2_norm(x=%x, epsilon=d*eps)
    %y   = mul(x=%n, y=<const sqrt(d)*scale, fp16>)

Why: the Neural Engine has no fp32, so the fp32 form pins every norm — and the
segments around it — to the CPU.  Why this is safe: the backends compute
``l2_norm``'s sum of squares internally at a wider range than fp16, where the
plain fp16 ``mul -> reduce_mean`` spelling overflows past ``|x| ~ 256`` (real
activations reach ``|x| ~ 1800`` and ``sum x^2 ~ 7e6``).  Measured on macOS 27
(M4 Pro), with the op placed on each device (checked with ``MLComputePlan``)
and compared against an fp64 reference, rows of width 1536 and 256:

* rms 1, 80, 2000, and rms 300 with a 6e4 outlier (``sum x^2`` up to 6e9):
  CPU, GPU and ANE all within 3e-3 relative;
* all-zero rows: 0 on every backend;
* rows so small that ``eps`` dominates (``mean x^2 <~ 1e-4``): exact on CPU and
  GPU, but the **ANE ignores the epsilon** and returns ``x / rms(x)``.  No real
  activation comes near that: the smallest row across all 2420 norm calls of a
  prompt (BOS included) has ``mean x^2 ~ 1e-3``, where ``eps = 1e-6`` moves the
  result by 0.05%.

It is also the cheapest fp16 spelling on the GPU — the alternative, dividing by
``max|x|`` first so an fp16 sum of squares cannot overflow, costs a second
reduction per norm, ~7% of a decode step.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _const_mul(var):
    """The ``mul`` by a constant that is the sole consumer of ``var``, or None."""
    consumers = var.child_ops
    if len(consumers) != 1 or var in var.op.enclosing_block.outputs:
        return None
    mul = consumers[0]
    if mul.op_type != "mul":
        return None
    other = mul.y if mul.x is var else mul.x
    return mul if other.val is not None else None


def _fp16_scale(value, name: str) -> np.ndarray:
    """The folded ``sqrt(d) * scale`` constant, narrowed to fp16.

    ``l2_norm`` returns unit-norm rows, so the whole RMSNorm magnitude lives in
    this constant: ``sqrt(d)`` (39.2 at ``d = 1536``) times the checkpoint's
    norm scale.  fp16 tops out at 65504, so a scale above ~1671 at that width
    would become ``inf`` and turn every row it touches into inf/NaN.  The
    checkpoint has nothing near that (see the export log), so a site that would
    overflow fails the export instead of getting a special-cased fp32 path.
    """
    wide = np.asarray(value, dtype=np.float32)
    with np.errstate(over="ignore"):
        narrow = wide.astype(np.float16)
    if not np.all(np.isfinite(narrow)):
        raise ValueError(
            f"fp16_l2_norm: the folded RMSNorm scale of {name!r} reaches "
            f"{np.max(np.abs(wide)):.6g}, which fp16 cannot represent"
        )
    _largest_scale[0] = max(_largest_scale[0], float(np.max(np.abs(wide))))
    return narrow


# Largest |folded scale| seen by the current ``apply``, for the log line.
_largest_scale = [0.0]


@block_context_manager
def _narrow_block(block) -> int:
    """Narrow every fp32 ``l2_norm -> mul(const)`` whose input is fp16 underneath.

    The input is either ``cast(fp16 -> fp32)`` or the fp32 result of a norm
    narrowed before it — ``collapse_cast_chains`` removes the fp16 round trip
    between two norms back to back.  The fp32 originals are left for
    dead-code elimination; the ``cast(-> fp16)`` readers are rewired here.
    """
    narrowed = {}  # fp32 var -> its fp16 replacement
    count = 0
    for op in list(block.operations):
        for inner in op.blocks:
            count += _narrow_block(inner)
        if op.op_type != "l2_norm" or op.outputs[0].dtype != types.fp32:
            continue
        x32 = op.x
        if x32 in narrowed:
            x16 = narrowed[x32]
        elif x32.op is not None and x32.op.op_type == "cast" and x32.op.x.dtype == types.fp16:
            x16 = x32.op.x
        else:
            continue
        mul = _const_mul(op.outputs[0])
        if mul is None:
            continue
        scale = mul.y if mul.x is op.outputs[0] else mul.x
        scale16 = _fp16_scale(scale.val, mul.name)
        normalized = mb.l2_norm(
            x=x16, epsilon=np.float16(op.epsilon.val), before_op=mul, name=op.name + "_fp16",
        )
        scaled = mb.mul(x=normalized, y=scale16, before_op=mul, name=mul.name + "_fp16")
        narrowed[mul.outputs[0]] = scaled
        for reader in list(mul.outputs[0].child_ops):
            if reader.op_type == "cast" and reader.outputs[0].dtype == types.fp16:
                block.replace_uses_of_var_after_op(
                    anchor_op=reader, old_var=reader.outputs[0], new_var=scaled,
                )
        count += 1
    return count


@register_pass(namespace="common")
class fp16_l2_norm(AbstractGraphPass):
    """Rewrite ``cast(fp16->fp32) -> l2_norm -> mul(const) -> cast(->fp16)``
    into an fp16 ``l2_norm`` + ``mul``; see the module docstring.  Run it
    before a ``dead_code_elimination``, which drops the fp32 originals."""

    def apply(self, prog):
        _largest_scale[0] = 0.0
        count = sum(_narrow_block(f) for f in prog.functions.values())
        if count:
            print(
                f"  fp16_l2_norm: {count} RMSNorm(s) now run in fp16 "
                f"(largest folded scale {_largest_scale[0]:.4g}, fp16 max 65504)",
                flush=True,
            )
