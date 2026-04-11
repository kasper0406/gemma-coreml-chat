"""Test SDPA fusion pass on real StableHLO→MIL attention graphs.

Traces minimal JAX attention functions through the actual StableHLO→MIL
pipeline, applies passes, and verifies the graph is rewritten correctly.
"""

import copy
import numpy as np
import jax
import jax.numpy as jnp
import coremltools as ct
from stablehlo_coreml.converter import convert as hlo_to_mil

# ── helpers ──────────────────────────────────────────────────────────────

def _jax_to_mil(fn, *example_args):
    """Trace a JAX function to StableHLO, convert to MIL Program."""
    lowered = jax.jit(fn).lower(*example_args)
    hlo = lowered.compiler_ir("stablehlo")
    return hlo_to_mil(hlo, minimum_deployment_target=ct.target.iOS18)


def _count_ops(prog, op_type, fname="main"):
    return sum(1 for op in prog.functions[fname].operations
               if op.op_type == op_type)


def _dump_ops(prog, fname="main", skip_const=True):
    """Print all non-const ops in a function with detailed info."""
    for op in prog.functions[fname].operations:
        if skip_const and op.op_type == "const":
            continue
        in_info = {}
        for k, v in op.inputs.items():
            if k == "name":
                continue
            if v.val is not None and np.asarray(v.val).size <= 8:
                in_info[k] = f"{np.asarray(v.val).tolist()}"
            else:
                in_info[k] = f"{list(v.shape) if v.shape else '?'}"
        out_shapes = [list(o.shape) if o.shape else "?" for o in op.outputs]
        print(f"  {op.op_type:25s} {in_info} → {out_shapes}")


def _apply_default_passes(prog):
    """Apply DEFAULT coremltools passes (minus fp16 cast)."""
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY
    pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
    pipeline.remove_passes({"common::add_fp16_cast"})
    for p in pipeline.passes:
        PASS_REGISTRY[p].apply(prog)


def _apply_cleanup_passes(prog):
    """Apply cleanup passes that simplify the graph before fusion."""
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY
    from stablehlo_coreml.passes.remove_noop_slice_update import remove_noop_slice_update  # noqa: F401
    import gemma_chat.mil_passes.collapse_reshape_chains  # noqa: F401
    import gemma_chat.mil_passes.remove_broadcast_tiles  # noqa: F401
    import gemma_chat.mil_passes.remove_redundant_maximum  # noqa: F401
    for name in [
        "common::remove_noop_slice_update",
        "common::collapse_reshape_chains",
        "common::remove_redundant_maximum",
        "common::remove_broadcast_tiles",
    ]:
        PASS_REGISTRY[name].apply(prog)


def _apply_softmax_pass(prog):
    import gemma_chat.mil_passes.replace_decomposed_softmax  # noqa: F401
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY
    PASS_REGISTRY["common::replace_decomposed_softmax"].apply(prog)


def _apply_sdpa_pass(prog):
    import gemma_chat.mil_passes.fuse_attention_to_sdpa  # noqa: F401
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY
    PASS_REGISTRY["common::fuse_attention_to_sdpa"].apply(prog)


def _full_pipeline(prog):
    """Apply the full pass pipeline: DEFAULT + cleanup + softmax + SDPA."""
    _apply_default_passes(prog)
    _apply_cleanup_passes(prog)
    _apply_softmax_pass(prog)
    _apply_sdpa_pass(prog)


# ── JAX attention functions ──────────────────────────────────────────────

def attention_masked(q, k, v, mask):
    """Attention with bool mask, scale=1.0 (like Gemma4 QK-norm)."""
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    scores = jnp.where(mask, scores, jnp.float32(-10000.0))
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    return jnp.matmul(weights, v)


def attention_no_mask(q, k, v):
    """Attention without mask."""
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    return jnp.matmul(weights, v)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_sdpa_fusion_masked():
    """Masked attention → single SDPA op with mask and Q pre-scaling."""
    q = jnp.ones((1, 8, 1, 256), dtype=jnp.float32)
    k = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)
    v = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)
    mask = jnp.ones((1, 1, 1, 128), dtype=jnp.bool_)

    prog = _jax_to_mil(attention_masked, q, k, v, mask)
    _full_pipeline(prog)

    assert _count_ops(prog, "scaled_dot_product_attention") == 1, \
        f"Expected 1 SDPA, got {_count_ops(prog, 'scaled_dot_product_attention')}"
    assert _count_ops(prog, "softmax") == 0, "softmax should be fused away"
    assert _count_ops(prog, "matmul") == 0, "matmuls should be fused away"
    assert _count_ops(prog, "select") == 0, "select should be fused away"

    # Check Q pre-scaling: should have mul(Q, sqrt(256)) = mul(Q, 16.0)
    assert _count_ops(prog, "mul") == 1, "Expected Q pre-scaling mul"
    for op in prog.functions["main"].operations:
        if op.op_type == "mul":
            scale = float(op.inputs["y"].val)
            assert abs(scale - 16.0) < 0.01, f"Expected scale=16.0, got {scale}"

    # Check SDPA shapes
    for op in prog.functions["main"].operations:
        if op.op_type == "scaled_dot_product_attention":
            assert list(op.outputs[0].shape) == [1, 8, 1, 256], \
                f"SDPA output shape wrong: {list(op.outputs[0].shape)}"
            assert op.inputs.get("attn_mask") is not None, "SDPA should have mask"

    print("  ✓ test_sdpa_fusion_masked passed")


def test_sdpa_fusion_no_mask():
    """Unmasked attention → SDPA without mask."""
    q = jnp.ones((1, 8, 1, 256), dtype=jnp.float32)
    k = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)
    v = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)

    prog = _jax_to_mil(attention_no_mask, q, k, v)
    _full_pipeline(prog)

    assert _count_ops(prog, "scaled_dot_product_attention") == 1
    assert _count_ops(prog, "softmax") == 0
    assert _count_ops(prog, "matmul") == 0

    for op in prog.functions["main"].operations:
        if op.op_type == "scaled_dot_product_attention":
            # No mask expected
            mask_input = op.inputs.get("attn_mask")
            # attn_mask may be present but None, or absent
            if mask_input is not None and mask_input.val is not None:
                assert False, "SDPA should not have a mask for unmasked attention"

    print("  ✓ test_sdpa_fusion_no_mask passed")


def test_sdpa_different_head_dim():
    """Global attention head_dim=512 → pre-scale by sqrt(512)≈22.6."""
    q = jnp.ones((1, 8, 1, 512), dtype=jnp.float32)
    k = jnp.ones((1, 8, 128, 512), dtype=jnp.float32)
    v = jnp.ones((1, 8, 128, 512), dtype=jnp.float32)
    mask = jnp.ones((1, 1, 1, 128), dtype=jnp.bool_)

    prog = _jax_to_mil(attention_masked, q, k, v, mask)
    _full_pipeline(prog)

    assert _count_ops(prog, "scaled_dot_product_attention") == 1
    for op in prog.functions["main"].operations:
        if op.op_type == "mul":
            scale = float(op.inputs["y"].val)
            expected = np.sqrt(512.0)
            assert abs(scale - expected) < 0.1, \
                f"Expected scale≈{expected:.1f}, got {scale}"

    print("  ✓ test_sdpa_different_head_dim passed")


def test_softmax_alone():
    """Softmax-only (no second matmul) should not be fused to SDPA."""
    def softmax_only(x):
        return jax.nn.softmax(x, axis=-1)

    x = jnp.ones((1, 8, 1, 128), dtype=jnp.float32)
    prog = _jax_to_mil(softmax_only, x)
    _full_pipeline(prog)

    assert _count_ops(prog, "scaled_dot_product_attention") == 0, \
        "Standalone softmax should not become SDPA"
    assert _count_ops(prog, "softmax") == 1, "Softmax should remain"

    print("  ✓ test_softmax_alone passed")


def test_gqa_attention():
    """GQA with num_kv_heads=1, num_heads=8 (jnp.repeat before attention)."""
    def gqa_attention(q, k, v, mask):
        k = jnp.repeat(k, 8, axis=1)  # [1,1,S,E] → [1,8,S,E]
        v = jnp.repeat(v, 8, axis=1)
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        scores = jnp.where(mask, scores, jnp.float32(-10000.0))
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        return jnp.matmul(weights, v)

    q = jnp.ones((1, 8, 1, 256), dtype=jnp.float32)
    k = jnp.ones((1, 1, 128, 256), dtype=jnp.float32)
    v = jnp.ones((1, 1, 128, 256), dtype=jnp.float32)
    mask = jnp.ones((1, 1, 1, 128), dtype=jnp.bool_)

    prog = _jax_to_mil(gqa_attention, q, k, v, mask)
    _full_pipeline(prog)

    sdpa_count = _count_ops(prog, "scaled_dot_product_attention")
    print(f"  GQA: {sdpa_count} SDPA, {_count_ops(prog, 'softmax')} softmax, "
          f"{_count_ops(prog, 'matmul')} matmul")
    assert sdpa_count == 1, f"Expected 1 SDPA for GQA, got {sdpa_count}"
    assert _count_ops(prog, "softmax") == 0
    assert _count_ops(prog, "matmul") == 0

    print("  ✓ test_gqa_attention passed")


def test_graph_dump():
    """Diagnostic: dump the graph after each pipeline stage."""
    q = jnp.ones((1, 8, 1, 256), dtype=jnp.float32)
    k = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)
    v = jnp.ones((1, 8, 128, 256), dtype=jnp.float32)
    mask = jnp.ones((1, 1, 1, 128), dtype=jnp.bool_)

    prog = _jax_to_mil(attention_masked, q, k, v, mask)

    print("\n--- After DEFAULT + cleanup + softmax ---")
    _apply_default_passes(prog)
    _apply_cleanup_passes(prog)
    _apply_softmax_pass(prog)
    _dump_ops(prog)

    print("\n--- After SDPA fusion ---")
    _apply_sdpa_pass(prog)
    _dump_ops(prog)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    if "--dump" in sys.argv:
        test_graph_dump()
        sys.exit(0)

    test_sdpa_fusion_masked()
    test_sdpa_fusion_no_mask()
    test_sdpa_different_head_dim()
    test_softmax_alone()
    test_gqa_attention()
    print("\nAll tests passed ✓")
