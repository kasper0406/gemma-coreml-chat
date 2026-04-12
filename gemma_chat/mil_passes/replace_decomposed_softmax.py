"""MIL pass: fuse decomposed softmax into native ``softmax`` op.

JAX's ``jax.nn.softmax`` lowers to a multi-op chain.  The exact chain
depends on the input dtype:

**float32 input** (no internal casts)::

    reduce_max → maximum(-inf) → reshape → sub → exp →
    reduce_sum → [reshape] → real_div

**float16 input** (JAX promotes reduce_sum to fp32)::

    reduce_max → maximum(-inf) → reshape → sub → exp →
    cast(fp16→fp32) → reduce_sum(fp32) → [reshape] → cast(fp32→fp16) →
    real_div

CoreML MIL has a native ``softmax`` op that computes this in a single
hardware-accelerated instruction on ANE.  This pass matches both
variants and replaces them with ``mb.softmax(x, axis)``.

Dead intermediate ops are collected via backward DCE and removed.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

_NEG_INF_THRESHOLD = -3.0e+38


def _get_scalar_value(var):
    """Return the uniform scalar value of a const/fill var, or None."""
    val = var.val
    if val is not None:
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        first = float(arr.flat[0])
        if arr.size == 1 or np.all(arr == first):
            return first
        return None
    if var.op is not None and var.op.op_type == "fill":
        v = var.op.inputs.get("value")
        if v is not None and v.val is not None:
            return float(v.val)
    return None


def _is_neg_inf(var):
    """True if var is a scalar/uniform constant ≤ -3e38."""
    sv = _get_scalar_value(var)
    return sv is not None and sv <= _NEG_INF_THRESHOLD


def _match_softmax(real_div_op):
    """Match a decomposed softmax chain ending at *real_div_op*.

    Expected chain::

        softmax_input → reduce_max(axis=A) → [maximum(-inf,...)] → [reshape] →
        sub(softmax_input, max_expanded) → exp →
        reduce_sum(exp, axis=A) → [reshape] → real_div(exp, sum_expanded)

    Returns ``(softmax_input_var, axis)`` or ``None``.
    """
    if real_div_op.op_type != "real_div":
        return None

    numerator = real_div_op.inputs["x"]    # exp output
    denominator = real_div_op.inputs["y"]  # reduce_sum output (possibly reshaped)

    # Numerator must be exp(sub(...))
    if numerator.op is None or numerator.op.op_type != "exp":
        return None
    exp_op = numerator.op

    sub_var = exp_op.inputs["x"]
    if sub_var.op is None or sub_var.op.op_type != "sub":
        return None
    sub_op = sub_var.op

    softmax_input = sub_op.inputs["x"]

    # Denominator: peel off reshape and/or cast, expect reduce_sum(exp_output).
    # For fp16 softmax JAX inserts cast(fp32→fp16) between reduce_sum and
    # real_div, so we may need to peel: [reshape] → [cast] → reduce_sum.
    sum_src = denominator
    for _ in range(3):
        if sum_src.op is not None and sum_src.op.op_type in ("reshape", "cast"):
            sum_src = sum_src.op.inputs["x"]
        else:
            break
    if sum_src.op is None or sum_src.op.op_type != "reduce_sum":
        return None
    reduce_sum_op = sum_src.op

    # reduce_sum must consume the same exp output as the numerator.
    # For fp16 softmax JAX inserts cast(fp16→fp32) before reduce_sum,
    # so reduce_sum.inputs["x"] may be cast(numerator) rather than numerator.
    sum_input = reduce_sum_op.inputs["x"]
    if sum_input is not numerator:
        if sum_input.op is not None and sum_input.op.op_type == "cast":
            if sum_input.op.inputs["x"] is not numerator:
                return None
        else:
            return None

    # Trace sub's y input (max value) backward:
    #   [reshape] → [maximum(-inf, ...)] → reduce_max(softmax_input)
    max_val = sub_op.inputs["y"]
    if max_val.op is not None and max_val.op.op_type == "reshape":
        max_val = max_val.op.inputs["x"]

    # Optional maximum(-inf, ...) from JAX's stable softmax
    if max_val.op is not None and max_val.op.op_type == "maximum":
        mx = max_val.op
        if _is_neg_inf(mx.inputs["x"]):
            max_val = mx.inputs["y"]
        elif _is_neg_inf(mx.inputs["y"]):
            max_val = mx.inputs["x"]
        else:
            return None

    # Must end at reduce_max of the same softmax input
    if max_val.op is None or max_val.op.op_type != "reduce_max":
        return None
    if max_val.op.inputs["x"] is not softmax_input:
        return None

    # Get reduction axis from reduce_max
    reduce_max_op = max_val.op
    axes_var = reduce_max_op.inputs.get("axes")
    if axes_var is None or axes_var.val is None:
        return None
    axes = list(np.asarray(axes_var.val).flatten())
    if len(axes) != 1:
        return None
    axis = int(axes[0])

    # Verify reduce_sum uses the same axis
    sum_axes_var = reduce_sum_op.inputs.get("axes")
    if sum_axes_var is None or sum_axes_var.val is None:
        return None
    sum_axes = list(np.asarray(sum_axes_var.val).flatten())
    if len(sum_axes) != 1 or int(sum_axes[0]) != axis:
        return None

    return softmax_input, axis


def _collect_dead_backward(start_op, keep_var):
    """Backward DCE from *start_op*: collect all ops whose outputs are
    consumed only by already-dead ops.  Never crosses *keep_var*."""
    dead = {start_op}
    queue = [start_op]
    while queue:
        op = queue.pop(0)
        for inp_var in op.inputs.values():
            if inp_var is keep_var:
                continue
            parent = inp_var.op
            if parent is None or parent in dead:
                continue
            all_dead = all(
                child in dead
                for out in parent.outputs
                for child in out.child_ops
            )
            if all_dead:
                dead.add(parent)
                queue.append(parent)
    return dead


@block_context_manager
def _replace_in_block(block):
    changed = False
    ops = list(block.operations)
    removed = set()
    for op in ops:
        for b in op.blocks:
            changed |= _replace_in_block(b)

        if op in removed:
            continue

        result = _match_softmax(op)
        if result is None:
            continue
        softmax_input, axis = result

        softmax_var = mb.softmax(
            x=softmax_input,
            axis=axis,
            before_op=op,
            name=op.name + "_softmax",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=softmax_var,
            no_check_var_types=True,
        )

        dead = _collect_dead_backward(op, softmax_input)
        block.remove_ops(list(dead))
        removed.update(dead)
        changed = True

    return changed


@register_pass(namespace="common")
class replace_decomposed_softmax(AbstractGraphPass):
    """Replace JAX's decomposed softmax chain with native ``softmax`` op."""

    def apply(self, prog):
        for fname in prog.functions:
            before = sum(1 for op in prog.functions[fname].operations
                         if op.op_type != "const")
            _replace_in_block(prog.functions[fname])
            after = sum(1 for op in prog.functions[fname].operations
                        if op.op_type != "const")
            sm_count = sum(1 for op in prog.functions[fname].operations
                           if op.op_type == "softmax")
            if sm_count:
                print(f"  replace_decomposed_softmax [{fname}]: {sm_count} softmax ops, "
                      f"eliminated {before - after} decomposed ops", flush=True)
