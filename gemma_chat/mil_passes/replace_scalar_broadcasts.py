"""MIL pass: replace uniform-value constants with ``fill`` ops.

MIL's ``const_elimination`` pass eagerly broadcasts scalar values to full
tensor shape.  For example, ``jnp.tanh(logits / 30.0) * 30.0`` creates two
(1, 1024, 262144) fp32 constants — each over 1 GB — containing a single
repeated value.  GELU scalars (0.5, sqrt(2)/2) and attention masks (-10000)
are similarly materialized.

This pass runs at the *end* of the pipeline, after all ``const_elimination``
passes, so the ``fill`` ops it creates won't be folded back.  At inference
time CoreML evaluates ``fill`` near-instantly — it's a memset.

For the Gemma4-E2B prefill function this saves ~3.3 GB (from 8.4 GB to ~5 GB).
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil import Builder as mb

import numpy as np

# Only replace constants larger than this many bytes.
_MIN_BYTES = 4096


def _is_uniform(val: np.ndarray) -> bool:
    """True if every element equals the first element."""
    if val.size <= 1:
        return False
    return bool(np.all(val.flat[0] == val))


def _is_constexpr_input(op) -> bool:
    """True if this const op feeds a constexpr_* consumer."""
    for child in op.outputs[0].child_ops:
        if child.op_type.startswith("constexpr_"):
            return True
    return False


@block_context_manager
def _replace_in_block(block):
    changed = False
    # Snapshot ops list (we'll modify block during iteration).
    ops = list(block.operations)
    for op in ops:
        # Recurse into sub-blocks.
        for b in op.blocks:
            changed |= _replace_in_block(b)

        if op.op_type != "const":
            continue

        val = op.outputs[0].val
        if not isinstance(val, np.ndarray):
            continue
        if val.nbytes < _MIN_BYTES:
            continue
        # fill() supports fp16, fp32, int32, bool.
        if val.dtype not in (np.float16, np.float32, np.int32, np.bool_):
            continue
        # Don't touch const inputs to constexpr ops (they require const).
        if _is_constexpr_input(op):
            continue
        if not _is_uniform(val):
            continue

        scalar = val.flat[0]
        shape = np.array(val.shape, dtype=np.int32)

        fill_var = mb.fill(
            shape=shape,
            value=scalar,
            before_op=op,
            name=op.name + "_fill",
        )

        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=fill_var,
            no_check_var_types=True,
        )
        block.remove_ops([op])
        changed = True

    return changed


@register_pass(namespace="common")
class replace_scalar_broadcasts(AbstractGraphPass):
    """Replace uniform-value ``const`` ops with runtime ``fill`` ops.

    Must run after all ``const_elimination`` passes to avoid being folded
    back into materialized constants.
    """

    def apply(self, prog):
        count_before = [0]

        def _count(func):
            for op in func.operations:
                if op.op_type == "const":
                    val = op.outputs[0].val
                    if (isinstance(val, np.ndarray) and val.nbytes >= _MIN_BYTES
                            and val.dtype in (np.float16, np.float32, np.int32, np.bool_)
                            and not _is_constexpr_input(op) and _is_uniform(val)):
                        count_before[0] += 1

        for fname in prog.functions:
            _count(prog.functions[fname])

        if count_before[0] == 0:
            return

        print(
            f"  replace_scalar_broadcasts: found {count_before[0]} uniform "
            f"const ops to replace with fill",
            flush=True,
        )

        for fname in prog.functions:
            _replace_in_block(prog.functions[fname])
