"""MIL pass: replace JAX erf polynomial approximation with native ``gelu`` op.

JAX's ``jax.lax.erf`` lowers to a ~95-op rational polynomial approximation
of ``erfc``.  The full GELU pattern is::

    0.5 * x * erfc(-x / sqrt(2))   ==   gelu(x, mode="EXACT")

CoreML MIL has a native ``gelu`` op that computes this in a single
instruction.  This pass matches the characteristic fingerprint:

1. ``fill/const(0.5)`` → ``mul(0.5, x)``   (the "half" multiply)
2. ``mul(-1, x)`` → ``mul(neg_x, 0.7071)`` (erf input scaling, 1/√2)
3. Long polynomial chain with ``select`` branch (small-arg vs large-arg)
4. Final ``mul(0.5*x, erf_result)``         (GELU output)

and replaces the entire ~95-op subgraph with ``mb.gelu(x, mode="EXACT")``.
Dead polynomial ops are collected via backward DCE and removed.

Counts from multi-chunk.mlpackage: 5 instances, ~475 ops + ~18.8 MB of
polynomial coefficient constants eliminated.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

_HALF = 0.5
_NEG_ONE = -1.0
_RSQRT2 = 0.7071067690849304  # float32 1/sqrt(2)

_TOL_HALF = 1e-6
_TOL_NEG = 1e-6
_TOL_RSQRT2 = 1e-3


def _get_scalar_value(var):
    """Return the uniform scalar value of a const/fill/constexpr var, or None."""
    val = var.val
    if val is not None:
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        first = float(arr.flat[0])
        if arr.size == 1 or np.all(arr == first):
            return first
        return None
    # fill ops have a symbolic output — read the value input directly.
    if var.op is not None and var.op.op_type == "fill":
        v = var.op.inputs.get("value")
        if v is not None and v.val is not None:
            return float(v.val)
    return None


def _match_gelu(half_mul):
    """If *half_mul* is the ``0.5 * x`` mul of a GELU pattern, return
    ``(gelu_input_var, gelu_output_op)`` or ``None``."""
    if half_mul.op_type != "mul":
        return None

    x_val = _get_scalar_value(half_mul.inputs["x"])
    y_val = _get_scalar_value(half_mul.inputs["y"])
    if x_val is not None and abs(x_val - _HALF) < _TOL_HALF:
        gelu_input = half_mul.inputs["y"]
    elif y_val is not None and abs(y_val - _HALF) < _TOL_HALF:
        gelu_input = half_mul.inputs["x"]
    else:
        return None

    # Verify: gelu_input also feeds mul(-1, gelu_input).
    neg_found = False
    for child in gelu_input.child_ops:
        if child is half_mul or child.op_type != "mul":
            continue
        for side in ("x", "y"):
            sv = _get_scalar_value(child.inputs[side])
            other = "y" if side == "x" else "x"
            if (sv is not None and abs(sv - _NEG_ONE) < _TOL_NEG
                    and child.inputs[other] is gelu_input):
                neg_x_var = child.outputs[0]
                neg_found = True
                break
        if neg_found:
            break
    if not neg_found:
        return None

    # Verify: neg_x feeds mul(neg_x, ~0.7071).
    sqrt2_found = False
    for child in neg_x_var.child_ops:
        if child.op_type != "mul":
            continue
        for side in ("x", "y"):
            sv = _get_scalar_value(child.inputs[side])
            if sv is not None and abs(sv - _RSQRT2) < _TOL_RSQRT2:
                sqrt2_found = True
                break
        if sqrt2_found:
            break
    if not sqrt2_found:
        return None

    # Find the GELU output: the sole mul consumer of 0.5*x.
    half_x_var = half_mul.outputs[0]
    consumers = [c for c in half_x_var.child_ops if c.op_type == "mul"]
    if len(consumers) != 1:
        return None

    return gelu_input, consumers[0]


def _collect_dead_backward(start_op, keep_var):
    """Backward DCE from *start_op*: collect all ops whose outputs are
    consumed only by already-dead ops.  Never crosses *keep_var* (the
    GELU input that must survive)."""
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

        result = _match_gelu(op)
        if result is None:
            continue
        gelu_input, gelu_output = result

        gelu_var = mb.gelu(
            x=gelu_input,
            mode="EXACT",
            before_op=op,
            name=gelu_output.name + "_gelu",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=gelu_output,
            old_var=gelu_output.outputs[0],
            new_var=gelu_var,
            no_check_var_types=True,
            force_replace=True,
        )

        dead = _collect_dead_backward(gelu_output, gelu_input)
        block.remove_ops(list(dead))
        removed.update(dead)
        changed = True

    return changed


@register_pass(namespace="common")
class replace_erf_gelu(AbstractGraphPass):
    """Replace JAX's ~95-op erf polynomial with native ``gelu`` op."""

    def apply(self, prog):
        for fname in prog.functions:
            before = sum(1 for op in prog.functions[fname].operations
                         if op.op_type != "const")
            _replace_in_block(prog.functions[fname])
            after = sum(1 for op in prog.functions[fname].operations
                        if op.op_type != "const")
            gelu_count = sum(1 for op in prog.functions[fname].operations
                             if op.op_type == "gelu")
            if gelu_count:
                print(f"  replace_erf_gelu [{fname}]: {gelu_count} gelu ops, "
                      f"eliminated {before - after} polynomial ops", flush=True)
