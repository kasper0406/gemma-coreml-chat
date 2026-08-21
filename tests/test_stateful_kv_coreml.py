"""End-to-end check of the stateful-KV export pipeline, on a synthetic model.

**This test builds, materializes, compiles and RUNS CoreML models** (tiny ones —
a few seconds on CPU_ONLY).  No Gemma weights are involved.

It mirrors the real export in miniature:

* one static-shape "sliding" cache, written with a ring mask + ``jnp.where`` and
  bound to CoreML **state** (a state write fed by ``slice_update`` does not
  persist on macOS 26, which is why the real decode step uses a select);
* one symbolic-dim "global" cache, written with ``dynamic_update_slice`` and
  left as an ordinary input/output (states must be static-shaped);
* ``gemma_chat.materialize`` then clones the dynamic-shape function into two
  concrete sizes, exactly as ``gemma-export`` does.

and then asserts what the Swift runtime relies on: the sliding state persists
across predictions, keeps persisting when the runtime switches to a
differently-sized function on the *same* ``MLState``, and a freshly made state
starts from zero.

Caveat: only the *Swift* CoreML API handles cross-instance ``MLState`` use
reliably.  The coremltools Python proxy segfaults when a state is passed to a
second ``MLModel`` instance of the full Gemma export (tiny models like the ones
here happen to work) — so do not try to "verify" the real artifact's state
sharing through ``MLModel.predict``; use the Swift runtime.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.models.utils import rename_feature
from jax import export as jax_export
from stablehlo_coreml import StateSpec
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.materialize import _materialize_single_function

WINDOW = 4      # sliding cache length (static — this is the state)
HEAD_DIM = 2
SIZES = (8, 16)  # materialized global-cache sizes


def _trace_step():
    """Trace a one-sliding / one-global-cache step function to StableHLO.

    Argument order mirrors the real export: ``[N, pos, sliding, glob]`` — JAX
    prepends a dimension-variable argument for the symbolic global length.
    Result order is ``[total, sliding_out, glob_out]``.
    """
    (N,) = jax_export.symbolic_shape("N", constraints=["N >= 1"])

    def step(pos_1d, sliding, glob):
        pos = pos_1d[0]
        # Written value is (pos + 1), so a sum identifies which writes landed.
        value = jnp.full((1, 1, 1, HEAD_DIM), 1, dtype=jnp.float16) * (
            pos + 1
        ).astype(jnp.float16)
        mask = (jnp.arange(WINDOW, dtype=jnp.int32) == (pos % WINDOW))[
            None, :, None, None
        ]
        sliding_out = jnp.where(mask, value, sliding)
        glob_out = jax.lax.dynamic_update_slice(glob, value, (0, pos, 0, 0))
        total = jnp.sum(sliding_out.astype(jnp.float32)).reshape(1)
        return total, sliding_out, glob_out

    traced = jax.jit(step).trace(
        jax.ShapeDtypeStruct((1,), jnp.int32),                 # pos
        jax.ShapeDtypeStruct((1, WINDOW, 1, HEAD_DIM), jnp.float16),  # sliding
        jax.ShapeDtypeStruct((1, N, 1, HEAD_DIM), jnp.float16),       # glob
    )
    return traced.lower().compiler_ir("stablehlo")


def _apply_flexible_dim1(feature, lower: int, upper: int, default_shape) -> None:
    """Give `feature` a RangeDim on dim 1 — what `export._mil_to_mlpackage` does."""
    arr = feature.type.multiArrayType
    if len(arr.shape) == 0:
        for d in default_shape:
            arr.shape.append(d)
    arr.ClearField("shapeRange")
    for i, dim in enumerate(arr.shape):
        sr = arr.shapeRange.sizeRanges.add()
        sr.lowerBound, sr.upperBound = (lower, upper) if i == 1 else (dim, dim)


def _build_dynamic_package(dest: Path) -> None:
    """Convert + rename + flex-shape + save, mirroring `export.py`."""
    module = _trace_step()

    # arg 2 (`sliding`) becomes state, updated by result 1.
    states = {2: StateSpec(output=1, name="sliding")}
    mil = hlo_to_mil(module, minimum_deployment_target=ct.target.iOS18, states=states)
    model = ct.convert(
        mil,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    spec = model._spec

    # The state argument is gone from the inputs and its updated value is gone
    # from the outputs — that is the whole point, and the rename below would
    # fail loudly if it were not so.
    assert len(spec.description.input) == 3, [i.name for i in spec.description.input]
    assert len(spec.description.output) == 2, [o.name for o in spec.description.output]
    assert [s.name for s in spec.description.state] == ["sliding"]

    for feat, new in zip(list(spec.description.input), ["N", "pos", "glob"]):
        if feat.name != new:
            rename_feature(spec, feat.name, new, rename_outputs=False)
    for feat, new in zip(list(spec.description.output), ["total", "glob_out"]):
        if feat.name != new:
            rename_feature(spec, feat.name, new, rename_inputs=False)

    default_shape = (1, SIZES[0], 1, HEAD_DIM)
    for feat in list(spec.description.input) + list(spec.description.output):
        if feat.name in ("glob", "glob_out"):
            _apply_flexible_dim1(feat, 1, max(SIZES), default_shape)

    if dest.exists():
        shutil.rmtree(dest)
    model.save(str(dest))


def _predict(model, state, pos: int, size: int):
    glob = np.zeros((1, size, 1, HEAD_DIM), dtype=np.float16)
    return model.predict(
        {
            "N": np.array([size], dtype=np.int32),
            "pos": np.array([pos], dtype=np.int32),
            "glob": glob,
        },
        state=state,
    )


@pytest.fixture(scope="module")
def dynamic_package(tmp_path_factory) -> Path:
    dynamic = tmp_path_factory.mktemp("stateful_kv") / "dynamic.mlpackage"
    _build_dynamic_package(dynamic)
    return dynamic


@pytest.fixture(scope="module")
def materialized_package(dynamic_package: Path) -> Path:
    concrete = dynamic_package.parent / "materialized.mlpackage"
    _materialize_single_function(
        dynamic_package, concrete, list(SIZES),
        source_function_name="main", target_prefix="step",
    )
    return concrete


def test_materialize_preserves_state_features(materialized_package):
    """Per-function state survives `materialize_symbolic_shape_program`.

    The feature lands under `description.functions[i].state`, NOT the top-level
    `description.state` — the Swift side reads it via
    `MLModelDescription.stateDescriptionsByName` on the loaded function.
    """
    spec = ct.models.MLModel(
        str(materialized_package), skip_model_load=True,
    )._spec
    by_name = {fd.name: fd for fd in spec.description.functions}
    for size in SIZES:
        fd = by_name[f"step_{size}"]
        assert [s.name for s in fd.state] == ["sliding"]
        assert [i.name for i in fd.input] == ["N", "pos", "glob"]
        assert [o.name for o in fd.output] == ["total", "glob_out"]
        glob = next(i for i in fd.input if i.name == "glob")
        assert list(glob.type.multiArrayType.shape) == [1, size, 1, HEAD_DIM]


def test_sliding_state_persists_across_calls_and_sizes(materialized_package):
    small, large = SIZES
    m_small = ct.models.MLModel(
        str(materialized_package),
        function_name=f"step_{small}",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    m_large = ct.models.MLModel(
        str(materialized_package),
        function_name=f"step_{large}",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # One MLState, made on one function instance and shared with the other —
    # this is exactly how the Swift runtime hands the same sliding caches to
    # `decode_512` and `prefill_1024`.
    state = m_small.make_state()

    # Each step writes (pos + 1) into ring slot (pos % WINDOW), across all
    # HEAD_DIM lanes, so `total` is HEAD_DIM * sum(live ring values).
    r = _predict(m_small, state, pos=0, size=small)
    assert r["total"][0] == pytest.approx(HEAD_DIM * 1)          # [1,0,0,0]

    r = _predict(m_small, state, pos=1, size=small)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (1 + 2))    # [1,2,0,0]

    # Switch to the other size mid-stream: same state, larger global cache.
    r = _predict(m_large, state, pos=2, size=large)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (1 + 2 + 3))  # [1,2,3,0]
    # The global cache is still plain I/O and is written at the absolute row.
    glob_out = np.asarray(r["glob_out"])
    assert glob_out.shape == (1, large, 1, HEAD_DIM)
    np.testing.assert_array_equal(glob_out[0, 2], np.full((1, HEAD_DIM), 3))
    assert not glob_out[0, :2].any(), "global cache is not state — it starts empty"

    # Wrap the ring: pos 4 overwrites slot 0 (which held 1).
    r = _predict(m_large, state, pos=4, size=large)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (5 + 2 + 3))  # [5,2,3,0]

    # A fresh state starts from zero — this is what "new conversation" does.
    fresh = m_large.make_state()
    r = _predict(m_large, fresh, pos=0, size=large)
    assert r["total"][0] == pytest.approx(HEAD_DIM * 1)

    # …and the original state is untouched by the fresh one.
    r = _predict(m_small, state, pos=3, size=small)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (5 + 2 + 3 + 4))


# ── The production layout: merge prefill + decode, then materialize ─────────


@pytest.fixture(scope="module")
def merged_package(dynamic_package: Path) -> Path:
    """What `gemma-export` actually builds: two functions, then materialize."""
    from gemma_chat.materialize import materialize_mlpackage

    from coremltools.models.utils import (
        MultiFunctionDescriptor, save_multifunction,
    )

    desc = MultiFunctionDescriptor()
    desc.add_function(
        str(dynamic_package), src_function_name="main", target_function_name="prefill",
    )
    desc.add_function(
        str(dynamic_package), src_function_name="main", target_function_name="decode",
    )
    desc.default_function_name = "decode"

    combined = dynamic_package.parent / "combined.mlpackage"
    if combined.exists():
        shutil.rmtree(combined)
    save_multifunction(desc, str(combined))

    out = dynamic_package.parent / "combined-mat.mlpackage"
    materialize_mlpackage(combined, out, list(SIZES))
    return out


def test_state_is_shared_across_prefill_and_decode_functions(merged_package):
    """One MLState drives `prefill_<N>` and `decode_<M>` alike.

    This is the layout the Swift runtime loads: a multifunction package whose
    `{prefill,decode}_{size}` functions each declare the same state features.
    The engine makes one state per conversation and hands it to whichever
    function the current cache size resolves to.
    """
    spec = ct.models.MLModel(str(merged_package), skip_model_load=True)._spec
    names = {fd.name for fd in spec.description.functions}
    assert names == {f"{p}_{s}" for p in ("prefill", "decode") for s in SIZES}
    for fd in spec.description.functions:
        assert [s.name for s in fd.state] == ["sliding"], fd.name

    small, large = SIZES
    decode = ct.models.MLModel(
        str(merged_package),
        function_name=f"decode_{small}",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    prefill = ct.models.MLModel(
        str(merged_package),
        function_name=f"prefill_{large}",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    state = decode.make_state()
    r = _predict(prefill, state, pos=0, size=large)
    assert r["total"][0] == pytest.approx(HEAD_DIM * 1)
    r = _predict(decode, state, pos=1, size=small)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (1 + 2))
    r = _predict(prefill, state, pos=2, size=large)
    assert r["total"][0] == pytest.approx(HEAD_DIM * (1 + 2 + 3))
