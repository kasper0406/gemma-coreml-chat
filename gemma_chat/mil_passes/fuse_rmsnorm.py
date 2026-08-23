"""MIL pass: collapse the RMSNorm elementwise chain onto the ``l2_norm`` op.

Gemma's RMSNorm is written in ``decode_coreml`` as fp32 statistics over an
fp16 activation, which the converter lowers to eight ops::

    %x32  = cast(x=%x, dtype=fp32)
    %sq   = mul(x=%x32, y=%x32)
    %var  = reduce_mean(x=%sq, axes=[-1], keep_dims=True)
    %vare = add(x=%var, y=eps)
    %inv  = rsqrt(x=%vare)
    %n    = mul(x=%x32, y=%inv)
    %s    = mul(x=%n, y=const scale)          # absent for the no-scale norms
    %out  = cast(x=%s, dtype=fp16)

Six of those eight ops compute ``x / sqrt(mean(x^2) + eps)``, which is
``l2_norm`` up to a constant:

.. math::
   \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x^2 + \\epsilon}}
   = \\sqrt{d}\\;\\frac{x}{\\sqrt{\\sum x^2 + d\\epsilon}}
   = \\sqrt{d}\\;\\mathrm{l2\\_norm}(x,\\ \\epsilon' = d\\epsilon)

so the whole chain becomes ``l2_norm`` plus one ``mul`` by the precomputed
constant ``sqrt(d) * scale`` — 8 ops down to 4, with the fp32/fp16 placement
of the surrounding casts untouched.  The rewrite is exact algebra on exact
constants; the only numerical difference is a division where the original had
an ``rsqrt`` multiply, both in fp32.

Why not ``layer_norm``?  Every pattern in coremltools'
``fuse_layernorm_or_instancenorm`` begins ``x -> reduce_mean -> sub``, and the
op itself is defined as ``gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta``.
RMSNorm has no mean subtraction, so that op cannot express it without changing
the numerics, and that pass would never match this chain even if it were still
in the pipeline (it is removed — see ``export.py`` — for an unrelated 4D
``batch_norm`` shape bug).

Shape handling: ``l2_norm`` normalizes over the **last three** dimensions and
treats everything before them as batch, so it computes the right reduction only
when ``rank >= 3`` and ``x.shape[-2] == x.shape[-3] == 1``.  Chains whose input
is shaped otherwise are left alone.  In the full 35-layer decode graph that
still covers 171 of the 242 sites: every ``(1, 1, D)`` residual-stream norm and
— because Gemma-4 E2B has a single KV head — every ``(1, 1, 1, hd)`` k/v norm.
The 71 it skips are the ``(1, 1, H, hd)`` q-norms, the rank-1 ``(D,)`` PLE-gate
projection norms and the ``(NL, 1, d)`` PLE norm.

Those could be fused too by reshaping to ``(-1, 1, 1, d)`` and back (8 ops down
to 6), and an earlier revision did.  It was **reverted**: in the prefill
function *every* site is off-canonical (``(1, 128, D)``), so that variant added
84 reshapes of 768 KB fp32 tensors to the 5-layer mini, and an interleaved A/B
on a 128-token chunk put it at 52.94 ms (range 52.74-53.13) against 52.40 ms
(52.17-52.86) for the unfused baseline and 52.34 ms (52.10-52.47) for this
shape-restricted version.  A ~0.5 ms regression, paid for ops that cost nothing
either way.  Fusing only the shapes that need no reshape adds no op anywhere.

What this pass is and is not worth: with the fp16 GELU it ships alongside it
takes the 5-layer mini's ``decode_512`` from 915 to 776 non-const ops (~15%;
~17% projected on the full 35-layer graph), which shortens compile and makes
the graph readable — but it does **not** speed decode up.  The mini is 96%
layer-stack work (10.7 ms/step against 72.5 ms for the 35-layer model: ~2.06 ms
per layer, ~0.4 ms fixed), and cutting those ops moved the median step time by
less than the run-to-run spread on CPU_AND_GPU, ALL and CPU_ONLY alike.  Core
ML's own compute plan says why: across the full ``decode_512`` graph the entire
2403-op ``mul``/``add``/``cast``/``reduce_mean``/``rsqrt`` population carries
**0.117%** of the estimated cost, against 21% for 277 ``matmul``s.  Decode is
not dispatch-bound on this backend, so op-count work of this kind should be
judged on graph size, not on latency.
"""

from __future__ import annotations

import math

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


def _scalar_const(var):
    """Return the value of ``var`` if it is a compile-time uniform scalar."""
    if var is None or var.val is None:
        return None
    val = np.asarray(var.val)
    if val.size != 1:
        return None
    return float(val.reshape(-1)[0])


def _sole_consumer(var):
    """The single op consuming ``var``, or ``None`` if it has 0 or 2+ consumers.

    ``child_ops`` lists an op once per *use*, so an op reading ``var`` twice
    (the ``mul(x, x)`` square) still counts as one consumer.
    """
    consumers = {id(op): op for op in var.child_ops}
    if len(consumers) != 1:
        return None
    return next(iter(consumers.values()))


def _match(rsqrt_op, block):
    """Match the RMSNorm chain ending at ``rsqrt_op``.

    Returns ``(x32, eps, tail_op, scale, dead)``: the fp32 input, the total
    epsilon, the last op of the chain (the scale ``mul``, or the normalize
    ``mul`` when the norm has no learnable scale), that scale's constant value
    or ``None``, and the ops the rewrite replaces, in removal order.
    """
    # rsqrt(x) is 1/sqrt(x + epsilon); fold that epsilon in with the add's.
    eps = _scalar_const(rsqrt_op.epsilon) or 0.0

    add_op = rsqrt_op.x.op
    if add_op is None or add_op.op_type != "add" or add_op.enclosing_block is not block:
        return None
    if _sole_consumer(rsqrt_op.x) is not rsqrt_op:
        return None
    for eps_var, var_var in ((add_op.y, add_op.x), (add_op.x, add_op.y)):
        eps_add = _scalar_const(eps_var)
        if eps_add is not None:
            break
    else:
        return None
    eps += eps_add

    mean_op = var_var.op
    if mean_op is None or mean_op.op_type != "reduce_mean" or mean_op.enclosing_block is not block:
        return None
    if _sole_consumer(var_var) is not add_op:
        return None
    if mean_op.keep_dims is None or mean_op.keep_dims.val is not True:
        return None
    if mean_op.axes is None or mean_op.axes.val is None:
        return None
    axes = list(np.asarray(mean_op.axes.val).reshape(-1))
    rank = mean_op.x.rank
    if len(axes) != 1 or (axes[0] % rank) != rank - 1:
        return None

    square_op = mean_op.x.op
    if square_op is None or square_op.op_type != "mul" or square_op.enclosing_block is not block:
        return None
    if square_op.x is not square_op.y:
        return None
    if _sole_consumer(mean_op.x) is not mean_op:
        return None

    x32 = square_op.x
    if x32.dtype != rsqrt_op.outputs[0].dtype:
        return None
    if x32.rank < 3 or any_symbolic(x32.shape):
        return None
    # l2_norm reduces over the last three dims; only these shapes make that
    # the last dim alone.
    if x32.shape[-2] != 1 or x32.shape[-3] != 1:
        return None

    # The rsqrt result must feed exactly one mul, against x32 itself.
    norm_mul = _sole_consumer(rsqrt_op.outputs[0])
    if norm_mul is None or norm_mul.op_type != "mul" or norm_mul.enclosing_block is not block:
        return None
    if {id(norm_mul.x), id(norm_mul.y)} != {id(x32), id(rsqrt_op.outputs[0])}:
        return None

    # x32 is read by the square (twice) and by the normalize mul, nothing else.
    if {id(op) for op in x32.child_ops} != {id(square_op), id(norm_mul)}:
        return None

    # Optionally absorb a following multiply by a constant scale.
    tail_op, scale = norm_mul, None
    scale_mul = _sole_consumer(norm_mul.outputs[0])
    if scale_mul is not None and scale_mul.op_type == "mul" and scale_mul.enclosing_block is block:
        other = scale_mul.y if scale_mul.x is norm_mul.outputs[0] else scale_mul.x
        if other is not norm_mul.outputs[0] and other.val is not None:
            val = np.asarray(other.val)
            # Must broadcast over the normalized axis only.
            if val.ndim == 0 or (val.shape[-1] in (1, x32.shape[-1])
                                 and all(d == 1 for d in val.shape[:-1])):
                tail_op, scale = scale_mul, val

    # Reverse topological order, so ``remove_ops`` never sees a live consumer.
    dead = [norm_mul, rsqrt_op, add_op, mean_op, square_op]
    if tail_op is not norm_mul:
        dead.insert(0, tail_op)

    return x32, eps, tail_op, scale, dead


@block_context_manager
def _fuse_in_block(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue
        for nested in op.blocks:
            fused += _fuse_in_block(nested)
        if op.blocks or op.op_type != "rsqrt":
            continue

        matched = _match(op, block)
        if matched is None:
            continue
        x32, eps, tail_op, scale, dead = matched

        d = int(x32.shape[-1])
        np_dtype = np.float32 if x32.dtype.__name__ == "fp32" else np.float16

        y = mb.l2_norm(x=x32, epsilon=np_dtype(d * eps), before_op=tail_op,
                       name=tail_op.name + "_l2")
        factor = math.sqrt(d) if scale is None else np.sqrt(d) * scale.astype(np.float64)
        y = mb.mul(x=y, y=np_dtype(factor), before_op=tail_op, name=tail_op.name)

        block.replace_uses_of_var_after_op(
            anchor_op=tail_op, old_var=tail_op.outputs[0], new_var=y,
        )
        block.remove_ops(dead)
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_rmsnorm(AbstractGraphPass):
    """Rewrite the eight-op RMSNorm chain as ``l2_norm`` + a constant ``mul``."""

    def apply(self, prog):
        for fname in prog.functions:
            func = prog.functions[fname]
            before = len(list(func.operations))
            fused = _fuse_in_block(func)
            if fused:
                after = len(list(func.operations))
                print(f"  fuse_rmsnorm [{fname}]: fused {fused} RMSNorm chain(s) "
                      f"({before} → {after} ops)", flush=True)
