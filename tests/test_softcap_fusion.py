"""Test logit softcap fusion pass on real StableHLO→MIL graphs.

Traces minimal JAX softcap functions through the actual StableHLO→MIL
pipeline, applies passes, and verifies tanh(x/cap)*cap → scaled_tanh.
"""

import numpy as np
import jax
import jax.numpy as jnp
import coremltools as ct
from stablehlo_coreml.converter import convert as hlo_to_mil
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────────


def _jax_to_mil(fn, *example_args):
    """Trace a JAX function to StableHLO, convert to MIL Program."""
    lowered = jax.jit(fn).lower(*example_args)
    hlo = lowered.compiler_ir("stablehlo")
    return hlo_to_mil(hlo, minimum_deployment_target=ct.target.iOS18)


def _count_ops(prog, op_type, fname="main"):
    return sum(1 for op in prog.functions[fname].operations
               if op.op_type == op_type)


def _apply_cleanup(prog):
    """Apply cleanup passes needed before softcap fusion."""
    import gemma_chat.mil_passes.remove_broadcast_tiles  # noqa: F401
    PASS_REGISTRY["common::remove_broadcast_tiles"].apply(prog)


def _apply_softcap(prog):
    import gemma_chat.mil_passes.fuse_logit_softcap  # noqa: F401
    PASS_REGISTRY["common::fuse_logit_softcap"].apply(prog)


# ── JAX softcap functions ────────────────────────────────────────────────


def softcap_div(x):
    """Softcap using division: tanh(x / 30) * 30."""
    cap = jnp.float32(30.0)
    return jnp.tanh(x / cap) * cap


def softcap_mul(x):
    """Softcap using multiplication: tanh(x * (1/30)) * 30."""
    cap = jnp.float32(30.0)
    return jnp.tanh(x * (1.0 / cap)) * cap


def softcap_different_cap(x):
    """Softcap with a different cap value: tanh(x / 50) * 50."""
    cap = jnp.float32(50.0)
    return jnp.tanh(x / cap) * cap


def softcap_in_larger_fn(x):
    """Softcap embedded in a larger computation (like logit projection)."""
    cap = jnp.float32(30.0)
    # Simulate: linear projection → softcap → output
    y = x * 2.0 + 1.0
    y = jnp.tanh(y / cap) * cap
    return y + x


def not_softcap_mismatched(x):
    """NOT a softcap: tanh(x / 30) * 20 — alpha ≠ 1/beta."""
    return jnp.tanh(x / jnp.float32(30.0)) * jnp.float32(20.0)


# ── Tests ────────────────────────────────────────────────────────────────


def test_softcap_fusion_div():
    """Softcap with real_div form should be fused."""
    x = jnp.ones((1, 8, 256), dtype=jnp.float16)
    prog = _jax_to_mil(softcap_div, x)
    _apply_cleanup(prog)

    assert _count_ops(prog, "tanh") == 1
    assert _count_ops(prog, "scaled_tanh") == 0

    _apply_softcap(prog)

    assert _count_ops(prog, "tanh") == 0
    assert _count_ops(prog, "scaled_tanh") == 1

    # Verify alpha/beta values
    for op in prog.functions["main"].operations:
        if op.op_type == "scaled_tanh":
            alpha = float(op.inputs["alpha"].val)
            beta = float(op.inputs["beta"].val)
            assert abs(alpha - 30.0) < 1e-4
            assert abs(beta - 1.0 / 30.0) < 1e-4


def test_softcap_fusion_mul():
    """Softcap with mul form should be fused (XLA may fold 1/cap)."""
    x = jnp.ones((1, 8, 256), dtype=jnp.float16)
    prog = _jax_to_mil(softcap_mul, x)
    _apply_cleanup(prog)
    _apply_softcap(prog)

    assert _count_ops(prog, "tanh") == 0
    assert _count_ops(prog, "scaled_tanh") == 1


def test_softcap_different_cap():
    """Softcap with cap=50 should work."""
    x = jnp.ones((1, 8, 256), dtype=jnp.float16)
    prog = _jax_to_mil(softcap_different_cap, x)
    _apply_cleanup(prog)
    _apply_softcap(prog)

    assert _count_ops(prog, "tanh") == 0
    assert _count_ops(prog, "scaled_tanh") == 1

    for op in prog.functions["main"].operations:
        if op.op_type == "scaled_tanh":
            alpha = float(op.inputs["alpha"].val)
            beta = float(op.inputs["beta"].val)
            assert abs(alpha - 50.0) < 1e-4
            assert abs(beta - 1.0 / 50.0) < 1e-4


def test_softcap_in_larger_fn():
    """Softcap embedded in a larger function should still fuse."""
    x = jnp.ones((1, 8, 256), dtype=jnp.float16)
    prog = _jax_to_mil(softcap_in_larger_fn, x)
    _apply_cleanup(prog)
    _apply_softcap(prog)

    assert _count_ops(prog, "tanh") == 0
    assert _count_ops(prog, "scaled_tanh") == 1


def test_mismatched_constants_not_fused():
    """tanh(x/30)*20 should NOT be fused (alpha*beta != 1.0)."""
    x = jnp.ones((1, 8, 256), dtype=jnp.float16)
    prog = _jax_to_mil(not_softcap_mismatched, x)
    _apply_cleanup(prog)
    _apply_softcap(prog)

    # Should NOT fuse — tanh stays
    assert _count_ops(prog, "tanh") == 1
    assert _count_ops(prog, "scaled_tanh") == 0


def test_numerical_correctness():
    """Verify scaled_tanh produces the same output as the original graph."""
    x_np = np.random.randn(1, 4, 32).astype(np.float16)
    x_jax = jnp.array(x_np)

    # JAX reference
    ref = np.array(softcap_div(x_jax))

    # CoreML via MIL
    prog = _jax_to_mil(softcap_div, x_jax)
    _apply_cleanup(prog)
    _apply_softcap(prog)
    assert _count_ops(prog, "scaled_tanh") == 1

    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )
    # coremltools renames %arg0 → _arg0
    input_name = model.get_spec().description.input[0].name
    out = model.predict({input_name: x_np})
    result = list(out.values())[0]

    np.testing.assert_allclose(result, ref, atol=1e-2, rtol=1e-2)
