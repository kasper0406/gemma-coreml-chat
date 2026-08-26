"""MIL pass: give ``select`` operands the output shape when a dim is symbolic.

E5RT cannot propagate shapes through a ``select`` whose operand broadcasts from
1 against a symbolic dimension.  Loading such a model fails with::

    Failed to PropagateInputTensorShapes: Validation error during type
    inference for select: at unknown location: Incompatible Dimension.

which is what makes a ``--no-materialize`` (RangeDim) export unusable: the
global KV caches carry a symbolic length, and every cache write is a
whole-tensor ``select`` whose *value* operand is a single row of shape
``(1, 1, nkv, hd)`` broadcast across that length.

``stablehlo-coreml``'s ``remove_broadcast_tiles`` already knows about this
failure and deliberately preserves tiles feeding a ``select`` -- but only tiles
that exist.  JAX never emits one for the value operand: ``jnp.where`` broadcasts
implicitly, and an explicit ``jnp.broadcast_to`` in the traced source is folded
away before it reaches MIL (verified -- the emitted graph is byte-identical with
and without it).  So the tile has to be introduced here, on the MIL side.

Rather than a ``tile`` (whose ``reps`` would themselves have to be symbolic),
each under-shaped operand is widened with ``fill_like`` + ``add``: ``fill_like``
takes its shape from a reference tensor that already has the output shape, so no
symbolic arithmetic is needed.  This is the same construction
``global_cache_states`` uses to make a state write persist.  Adding zero is
exact in fp16, so the rewrite does not change numerics.
"""

from __future__ import annotations

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


_counter: list = [0]


def _is_symbolic_dim(d) -> bool:
    return not isinstance(d, (int, np.integer))


def _needs_widening(operand, out_shape) -> bool:
    """True if *operand* broadcasts from 1 into a symbolic output dim."""
    if operand is None or operand.shape is None:
        return False
    shape = operand.shape
    if len(shape) != len(out_shape):
        return True
    for have, want in zip(shape, out_shape):
        if _is_symbolic_dim(want) and not _is_symbolic_dim(have) and have == 1:
            return True
    return False


def _full_shape_reference(op, out_shape):
    """Pick an operand that already has the output shape, to size ``fill_like``."""
    for name in ("cond", "a", "b"):
        var = op.inputs.get(name)
        if var is None or var.shape is None:
            continue
        if len(var.shape) == len(out_shape) and not _needs_widening(var, out_shape):
            return var
    return None


@block_context_manager
def _rewrite_block(block) -> int:
    rewritten = 0

    for op in list(block.operations):
        for b in op.blocks:
            rewritten += _rewrite_block(b)

        if op.op_type != "select":
            continue
        out_shape = op.outputs[0].shape
        if out_shape is None or not any(_is_symbolic_dim(d) for d in out_shape):
            continue

        ref = _full_shape_reference(op, out_shape)
        if ref is None:
            continue

        new_inputs = {}
        for name in ("a", "b"):
            var = op.inputs.get(name)
            if not _needs_widening(var, out_shape):
                continue
            zero = mb.fill_like(ref_tensor=ref, value=0.0, before_op=op,
                                name=op.name + f"_{name}_zeros")
            if zero.dtype != var.dtype:
                zero = mb.cast(x=zero, dtype=types.builtin_to_string(var.dtype),
                               before_op=op, name=op.name + f"_{name}_zeros_cast")
            new_inputs[name] = mb.add(x=var, y=zero, before_op=op,
                                      name=op.name + f"_{name}_wide")

        if not new_inputs:
            continue

        kwargs = {
            "cond": op.inputs["cond"],
            "a": new_inputs.get("a", op.inputs.get("a")),
            "b": new_inputs.get("b", op.inputs.get("b")),
            "before_op": op,
            "name": op.name + "_bcast",
        }
        new_sel = mb.select(**kwargs)
        block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_sel,
            no_check_var_types=True,
        )
        block.remove_ops([op])
        rewritten += 1
        _counter[0] += 1

    return rewritten


@register_pass(namespace="common")
class broadcast_select_operands(AbstractGraphPass):
    """Widen ``select`` operands that broadcast into a symbolic dimension."""

    def apply(self, prog):
        _counter[0] = 0
        for fname in prog.functions:
            n = _rewrite_block(prog.functions[fname])
            if n:
                print(
                    f"  broadcast_select_operands [{fname}]: widened {n} select(s)",
                    flush=True,
                )
