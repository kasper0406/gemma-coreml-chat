"""Folding the global cache length into the materialized functions.

``materialize_symbolic_shape_program`` rewrites *shapes*, so the dimension
variable JAX passes alongside them — renamed ``N`` by ``gemma_chat.export`` —
survives as an ordinary runtime ``int32[1]`` input of every ``{prefill,decode}_S``
function, even though each of those has exactly one possible value for it.

That leaves the global attention mask symbolic (``range_1d(end=N)``,
``fill(shape=[1, H, 1, N])``) even in a function the materializer specialized to
one concrete cache size.  ``mil_passes.concretize_cache_length``, driven by
``materialize._concretize_cache_lengths``, folds ``N`` in so nothing is symbolic
any more and the mask folds to a constant.

Concrete shapes would also let ``fuse_attention_to_sdpa`` — which bails on
symbolic dimensions — finally fuse these global sites.  That fusion is
deliberately *not* re-run (two Apple defects; see
``materialize._concretize_cache_lengths``), so the site must come out of the
pass still decomposed as ``matmul → select → softmax → matmul``.  That is
asserted here, because re-adding the fusion is a one-line change and this test
is where the reason it was removed lives.

The graph below is one global decode-attention site with GQA, the shape the real
export produces at every global layer.  Both halves are checked: the program
rewrite, and — **this test builds, compiles and RUNS a CoreML model** — that the
rewritten model still computes the attention it was traced from.
"""

from __future__ import annotations

import collections
from collections import OrderedDict

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil.passes.defs.symbol_transform import (
    materialize_symbolic_shape_program,
)
from coremltools.converters.mil.mil.types.symbolic import any_symbolic
from jax import export as jax_export
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.materialize import _concretize_cache_lengths
from gemma_chat.mil_passes.concretize_cache_length import LENGTH_INPUT_NAME
from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline

NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 64
CACHE_LEN = 32
# The real export materializes into `{prefill,decode}_{size}` functions; the pass
# keys off the map it is handed, not the name, and a single-function program has
# to be called `main` to compile — so that is what this one materializes into.
FUNC = "main"

# Kept in sync with gemma_chat/export.py:_mil_to_mlpackage.
_REMOVED_PASSES = [
    "common::add_fp16_cast",
    "common::fuse_layernorm_or_instancenorm",
    "common::fuse_elementwise_to_batchnorm",
]


def global_decode_attention(cache_k, cache_v, q, position):
    """One global attention site: mask length comes from the cache, not a const."""
    valid = jnp.arange(cache_k.shape[1], dtype=jnp.int32) <= position[0]
    rep = NUM_HEADS // NUM_KV_HEADS
    k = jnp.repeat(cache_k, rep, axis=2)
    v = jnp.repeat(cache_v, rep, axis=2)
    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))
    w = jnp.matmul(qt, jnp.swapaxes(kt, -2, -1))
    w = jnp.where(valid[jnp.newaxis, jnp.newaxis, jnp.newaxis], w, -10000.0)
    w = jax.nn.softmax(w, axis=-1)
    out = jnp.matmul(w, vt)
    return jnp.transpose(out, (0, 2, 1, 3)).reshape(1, 1, NUM_HEADS * HEAD_DIM)


def _rename_input(func, old: str, new: str) -> None:
    """Rename a pymil function input, order preserved.

    ``export.py`` renames JAX's dimension-variable argument to ``N`` with
    ``ct.utils.rename_feature`` on the saved model, which rewrites the MLProgram
    input names too; ``materialize.py`` then loads that model back, so the pass
    sees the new name.  This does the same thing to an in-memory program.
    """
    placeholder = func.placeholder_inputs[old]
    placeholder.set_name(new)
    func._input_dict[old].set_name(new)
    func.placeholder_inputs = OrderedDict(
        (new if name == old else name, value)
        for name, value in func.placeholder_inputs.items()
    )
    func._input_dict = OrderedDict(
        (new if name == old else name, value)
        for name, value in func._input_dict.items()
    )


def _materialized_program():
    """Trace with a symbolic cache length, convert, then materialize to CACHE_LEN.

    The result is exactly what ``materialize.py`` hands to the pass under test:
    concrete cache shapes, and the length still a runtime input named ``N``.
    """
    (n,) = jax_export.symbolic_shape("N", constraints=["N >= 1"])
    cache_spec = jax.ShapeDtypeStruct((1, n, NUM_KV_HEADS, HEAD_DIM), jnp.float32)
    traced = jax.jit(global_decode_attention).trace(
        cache_spec,
        cache_spec,
        jax.ShapeDtypeStruct((1, 1, NUM_HEADS, HEAD_DIM), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    prog = hlo_to_mil(
        traced.lower().compiler_ir("stablehlo"),
        minimum_deployment_target=ct.target.iOS18,
    )
    pipeline = build_ct_convert_pass_pipeline()
    pipeline.remove_passes(_REMOVED_PASSES)
    model = ct.convert(
        prog,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )
    prog = model._mil_program

    concrete = (1, CACHE_LEN, NUM_KV_HEADS, HEAD_DIM)
    materialize = materialize_symbolic_shape_program()
    materialize.source_function_name = "main"
    materialize.function_name_to_materialization_map = {
        FUNC: {"cache_k": concrete, "cache_v": concrete}
    }
    materialize.apply(prog)

    func = prog.functions[FUNC]
    dim_var = next(
        name for name in func.inputs
        if name not in ("cache_k", "cache_v", "q", "position")
    )
    _rename_input(func, dim_var, LENGTH_INPUT_NAME)
    return prog


def _op_types(prog):
    return collections.Counter(op.op_type for op in prog.functions[FUNC].operations)


@pytest.fixture(scope="module")
def materialized():
    return _materialized_program()


@pytest.fixture(scope="module")
def concretized():
    # A second build rather than a copy: the pass rewrites in place, and both
    # states of the program are needed side by side.
    prog = _materialized_program()
    _concretize_cache_lengths(prog, {FUNC: CACHE_LEN})
    return prog


def test_materialization_leaves_the_length_a_runtime_input(materialized):
    """The starting point: concrete cache, symbolic mask, unfused attention."""
    func = materialized.functions[FUNC]
    assert LENGTH_INPUT_NAME in func.inputs
    assert tuple(func.inputs["cache_k"].shape) == (1, CACHE_LEN, NUM_KV_HEADS, HEAD_DIM)

    counts = _op_types(materialized)
    assert counts["scaled_dot_product_attention"] == 0
    assert counts["softmax"] == 1
    assert counts["matmul"] == 2

    symbolic = [
        op.op_type for op in func.operations
        if any(any_symbolic(out.shape) for out in op.outputs)
    ]
    assert "range_1d" in symbolic and "fill" in symbolic, symbolic


def test_length_leaves_the_signature_and_becomes_a_constant(concretized):
    func = concretized.functions[FUNC]
    assert LENGTH_INPUT_NAME not in func.inputs
    assert LENGTH_INPUT_NAME not in func.placeholder_inputs
    assert list(func.inputs) == ["cache_k", "cache_v", "q", "position"]

    length = next(
        op for op in func.operations
        if op.op_type == "const" and op.name == f"{LENGTH_INPUT_NAME}_const"
    )
    np.testing.assert_array_equal(length.outputs[0].val, np.int32([CACHE_LEN]))


def test_no_shape_is_symbolic_any_more(concretized):
    func = concretized.functions[FUNC]
    leftovers = [
        (op.op_type, out.name, out.shape)
        for op in func.operations
        for out in op.outputs
        if any_symbolic(out.shape)
    ]
    assert leftovers == []


def test_the_global_attention_stays_decomposed(concretized):
    """No SDPA at the global site — see the module docstring for why.

    If someone re-adds ``common::fuse_attention_to_sdpa`` to
    ``_concretize_cache_lengths``, this is the test that fails, and the comment
    at that call site says which two Apple defects have to be fixed first.
    """
    counts = _op_types(concretized)
    assert counts["scaled_dot_product_attention"] == 0
    assert counts["softmax"] == 1
    assert counts["matmul"] == 2
    assert counts["select"] == 1


def test_the_concretized_model_computes_the_attention_it_was_traced_from(concretized):
    """Run the rewritten model and check it against the JAX source, at three
    positions — one where the mask hides almost everything, one in the middle,
    one where it hides nothing.

    The oracle is the JAX function rather than the ``N``-carrying model: with a
    symbolic key axis the mask ``tile`` reaches only the first attention head in
    this small graph, so the pre-pass model is not something to hold the
    concretized one against — the mask it applies is wrong.  (The real export is
    not affected — its ``{prefill,decode}_512`` and ``_1024`` functions agree bit
    for bit on the same tokens — but it is one more reason not to leave the key
    axis symbolic.)
    """
    model = ct.convert(
        concretized,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    rng = np.random.default_rng(0)
    cache_shape = (1, CACHE_LEN, NUM_KV_HEADS, HEAD_DIM)
    inputs = {
        "cache_k": rng.standard_normal(cache_shape).astype(np.float32),
        "cache_v": rng.standard_normal(cache_shape).astype(np.float32),
        "q": rng.standard_normal((1, 1, NUM_HEADS, HEAD_DIM)).astype(np.float32),
    }
    out_name = model._spec.description.output[0].name

    for position in (0, 7, CACHE_LEN - 1):
        pos = np.array([position], np.int32)
        want = np.asarray(global_decode_attention(
            jnp.asarray(inputs["cache_k"]), jnp.asarray(inputs["cache_v"]),
            jnp.asarray(inputs["q"]), jnp.asarray(pos),
        ))
        got = model.predict({**inputs, "position": pos})[out_name]
        np.testing.assert_allclose(
            np.asarray(got).ravel(), want.ravel(), rtol=1e-4, atol=1e-4,
            err_msg=f"at position {position}",
        )
