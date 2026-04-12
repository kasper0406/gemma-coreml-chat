"""MIL pass: replace JAX erf polynomial approximation with native ``gelu`` op.

JAX's ``jax.lax.erf`` lowers to a ~62-op rational polynomial approximation
of ``erfc``.  The full GELU pattern is::

    gelu(x) = 0.5 * x * erfc(-x / sqrt(2))

CoreML MIL has a native ``gelu`` op that computes this in a single
instruction.  This pass identifies the GELU structure by:

1. Finding ``abs`` ops that mark the entry to the erf polynomial.
2. Verifying the structural signature:
   - ``abs.input`` = ``mul(sub(0, gate), 1/sqrt(2))`` — the erf argument
   - ``mul(abs.input, abs.input)`` → ``sub(0, _)`` → ``exp`` — Gaussian core
3. Finding ``mul(0.5, gate)`` among gate's other consumers (the half multiply).
4. Finding ``mul(0.5*gate, erfc_result)`` as the GELU output.
5. Replacing with ``mb.gelu(x=gate, mode="EXACT")``.

Dead polynomial ops are collected via backward DCE and removed.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

_RSQRT2 = 0.7071067690849304  # float32 1/sqrt(2)
_HALF = 0.5
_TOL = 0.01


def _get_scalar_value(var):
    """Return the uniform scalar value of a const/fill/constexpr var, or None."""
    val = var.val
    if val is not None:
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        first = float(arr.flat[0])
        if arr.size == 1 or np.allclose(arr, first):
            return first
        return None
    if var.op is not None and var.op.op_type == "fill":
        v = var.op.inputs.get("value")
        if v is not None and v.val is not None:
            return float(v.val)
    return None


def _match_gelu(abs_op):
    """If *abs_op* is the ``abs`` at the entry of a GELU erf polynomial,
    return ``(gate_var, gelu_output_op)`` or ``None``.

    Structural signature::

        gate ──┬─ sub(0, gate) ─ mul(_, 1/√2) ─ abs ─ ... polynomial ...
               │                                  └─ mul(x,x) ─ sub(0,_) ─ exp
               └─ mul(0.5, gate) ──── mul(_, erfc_result) ← GELU output
    """
    if abs_op.op_type != "abs":
        return None

    # abs.input = mul(sub(0, gate), 1/sqrt(2))
    working_var = abs_op.inputs["x"]
    working_op = working_var.op
    if working_op is None or working_op.op_type != "mul":
        return None

    # Find sub(0, gate) and scale ≈ 1/sqrt(2) among mul's two inputs.
    gate = None
    for side in ("x", "y"):
        v = working_op.inputs[side]
        other = "y" if side == "x" else "x"
        if v.op is None or v.op.op_type != "sub":
            continue
        sub_x_val = v.op.inputs["x"].val
        if sub_x_val is None or not np.allclose(np.asarray(sub_x_val), 0):
            continue
        scale_val = _get_scalar_value(working_op.inputs[other])
        if scale_val is not None and abs(scale_val - _RSQRT2) < _TOL:
            gate = v.op.inputs["y"]
            break

    if gate is None:
        return None

    # Verify Gaussian core: mul(working, working) → sub(0, _) → exp.
    gaussian_ok = False
    for child in working_var.child_ops:
        if child.op_type != "mul" or child is abs_op:
            continue
        if child.inputs.get("x") is not working_var:
            continue
        if child.inputs.get("y") is not working_var:
            continue
        for s_consumer in child.outputs[0].child_ops:
            if s_consumer.op_type != "sub":
                continue
            sx = s_consumer.inputs.get("x")
            if sx is None or sx.val is None:
                continue
            if not np.allclose(np.asarray(sx.val), 0):
                continue
            for e_consumer in s_consumer.outputs[0].child_ops:
                if e_consumer.op_type == "exp":
                    gaussian_ok = True
                    break
            if gaussian_ok:
                break
        if gaussian_ok:
            break

    if not gaussian_ok:
        return None

    # Find mul(0.5, gate) among gate's consumers.
    half_gate_op = None
    for child in gate.child_ops:
        if child.op_type != "mul":
            continue
        for s in ("x", "y"):
            sv = _get_scalar_value(child.inputs[s])
            oth = "y" if s == "x" else "x"
            if sv is not None and abs(sv - _HALF) < _TOL and child.inputs[oth] is gate:
                half_gate_op = child
                break
        if half_gate_op is not None:
            break

    if half_gate_op is None:
        return None

    # GELU output = the sole mul consumer of 0.5*gate.
    half_gate_var = half_gate_op.outputs[0]
    mul_consumers = [c for c in half_gate_var.child_ops if c.op_type == "mul"]
    if len(mul_consumers) != 1:
        return None

    return gate, mul_consumers[0]


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

        if op in removed or op.op_type != "abs":
            continue

        result = _match_gelu(op)
        if result is None:
            continue
        gate, gelu_output = result

        gelu_var = mb.gelu(
            x=gate,
            mode="EXACT",
            before_op=gelu_output,
            name=gelu_output.name + "_gelu",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=gelu_output,
            old_var=gelu_output.outputs[0],
            new_var=gelu_var,
            no_check_var_types=True,
            force_replace=True,
        )

        dead = _collect_dead_backward(gelu_output, gate)
        block.remove_ops(list(dead))
        removed.update(dead)
        changed = True

    return changed


@register_pass(namespace="common")
class replace_erf_gelu(AbstractGraphPass):
    """Replace JAX's ~62-op erf polynomial with native ``gelu`` op."""

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
