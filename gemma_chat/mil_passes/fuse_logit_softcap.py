# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Kasper Nielsen
"""
MIL pass: fuse ``tanh(x / cap) * cap`` into ``scaled_tanh(x, alpha, beta)``.

This pattern appears in Gemma's logit soft-capping:
    logits = tanh(logits / 30) * 30

CoreML's ``scaled_tanh`` computes ``alpha * tanh(beta * x)``, so the mapping
is ``alpha = cap, beta = 1/cap``.

The pass handles both forms of the input scaling:
    - ``real_div(x, cap)``  →  beta = 1/cap
    - ``mul(x, 1/cap)``     →  beta = mul_const
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil import Builder as mb

import numpy as np


def _get_scalar(var):
    """Extract a Python float from a MIL variable if it's a compile-time scalar
    or a uniform-valued constant (all elements identical — DEFAULT passes
    broadcast scalars into full-size tensors)."""
    if var.val is None:
        return None
    arr = np.asarray(var.val).flatten()
    if arr.size == 0:
        return None
    first = float(arr[0])
    if arr.size > 1 and not np.all(arr == arr[0]):
        return None
    return first


def _sole_consumer(var):
    """Return the single consumer op of *var*, or None if != 1 consumers."""
    children = list(var.child_ops)
    if len(children) != 1:
        return None
    return children[0]


@block_context_manager
def _fuse_in_block(block):
    changed = False
    for op in list(block.operations):
        for b in op.blocks:
            changed |= _fuse_in_block(b)

        if op.op_type != "tanh":
            continue

        # ── Match input: real_div(x, C) or mul(x, beta_const) ──
        input_op = op.inputs["x"].op
        if input_op is None:
            continue

        x_var = None       # the original un-scaled input
        beta = None        # scalar beta for scaled_tanh

        if input_op.op_type == "real_div":
            cap = _get_scalar(input_op.inputs["y"])
            if cap is None or cap == 0.0:
                continue
            beta = 1.0 / cap
            x_var = input_op.inputs["x"]
        elif input_op.op_type == "mul":
            # mul(x, beta_const) — one operand is scalar
            sx = _get_scalar(input_op.inputs["x"])
            sy = _get_scalar(input_op.inputs["y"])
            if sx is not None:
                beta = sx
                x_var = input_op.inputs["y"]
            elif sy is not None:
                beta = sy
                x_var = input_op.inputs["x"]
            else:
                continue
        else:
            continue

        # ── Match output: mul(tanh_out, alpha_const) ──
        consumer = _sole_consumer(op.outputs[0])
        if consumer is None or consumer.op_type != "mul":
            continue

        alpha = None
        sx = _get_scalar(consumer.inputs["x"])
        sy = _get_scalar(consumer.inputs["y"])
        if sx is not None and consumer.inputs["y"] is op.outputs[0]:
            alpha = sx
        elif sy is not None and consumer.inputs["x"] is op.outputs[0]:
            alpha = sy
        else:
            continue

        # ── Verify alpha * beta ≈ 1.0 (soft-cap identity) ──
        if abs(alpha * beta - 1.0) > 1e-4:
            continue

        # ── Replace with scaled_tanh ──
        out_dtype = consumer.outputs[0].dtype
        # Use the same dtype as the output for alpha/beta constants
        np_dtype = np.float16 if "fp16" in str(out_dtype) else np.float32

        new_var = mb.scaled_tanh(
            x=x_var,
            alpha=np_dtype(alpha),
            beta=np_dtype(beta),
            before_op=input_op,
            name=consumer.name + "_softcap",
        )

        consumer.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=consumer,
            old_var=consumer.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )
        # Remove in reverse dependency order: consumer → tanh → input_op
        # Check if tanh was the sole consumer BEFORE removing ops
        tanh_was_sole_consumer = len(list(input_op.outputs[0].child_ops)) == 1
        ops_to_remove = [consumer, op]
        if tanh_was_sole_consumer:
            ops_to_remove.append(input_op)
        block.remove_ops(ops_to_remove)

        changed = True
        print(
            f"    fused softcap: alpha={alpha}, beta={beta:.6f}  "
            f"({input_op.op_type} → tanh → mul  ⟶  scaled_tanh)",
            flush=True,
        )

    return changed


@register_pass(namespace="common")
class fuse_logit_softcap(AbstractGraphPass):
    """Fuse ``tanh(x/cap)*cap`` into ``scaled_tanh(x, alpha=cap, beta=1/cap)``."""

    def apply(self, prog):
        for f in prog.functions.values():
            _fuse_in_block(f)
