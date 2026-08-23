"""The global KV caches are written with a whole-tensor select, not a slice.

``jax.lax.dynamic_update_slice`` lowers to a MIL ``slice_update`` whose ``begin``
is a runtime tensor, and MPSGraph's handler for that reads the index back to the
CPU mid-encode (``GPURegionRuntime::waitAndReadIntTensorData`` →
``waitUntilCompleted``).  Six global caches × one such write per step drained the
GPU pipeline six times per decode, ~17 ms of the ~79 ms step.  Both write
helpers in ``gemma_chat.decode_coreml`` therefore build the update out of a mask
and a select instead — the shape of write the sliding caches already used
because a Core ML state update fed by ``slice_update`` silently does not persist
(see ``tests/test_sliding_state_write.py`` and ``mil_passes.global_cache_states``).

Three things are pinned down here:

1. the helpers are *numerically* the ``dynamic_update_slice`` they replaced;
2. the converted graph contains no ``slice_update`` at all — with the old
   formulation converted alongside as a control, to show the check would catch a
   regression;
3. **this test builds, compiles and RUNS a CoreML model**: a select-shaped write
   still persists through ``coreml_update_state``, and rows written by earlier
   calls are still there for later ones.  That last part is the trap the sliding
   caches hit, so it is checked on the real ``_chunk_write`` output rather than a
   hand-built stand-in.
"""

from __future__ import annotations

import collections

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.decode_coreml import _chunk_write, _row_write
from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline
from gemma_chat.mil_passes.global_cache_states import global_kv_caches_to_states

LEN = 8        # "materialized" global cache length
NKV = 1
HEAD_DIM = 2
CHUNK = 2      # tokens per prefill chunk

# Kept in sync with gemma_chat/export.py:_mil_to_mlpackage.
_REMOVED_PASSES = [
    "common::add_fp16_cast",
    "common::fuse_layernorm_or_instancenorm",
    "common::fuse_elementwise_to_batchnorm",
]


# ── 1. The writes themselves ───────────────────────────────────────────────


@pytest.mark.parametrize("length", [1, 3, 8])
def test_row_write_matches_dynamic_update_slice(length):
    rng = np.random.default_rng(0)
    cache = jnp.asarray(
        rng.standard_normal((1, length, NKV, HEAD_DIM)).astype(np.float16)
    )
    for position in range(length):
        value = jnp.asarray(
            rng.standard_normal((1, 1, NKV, HEAD_DIM)).astype(np.float16)
        )
        pos = jnp.int32(position)
        got = _row_write(cache, value, pos)
        want = jax.lax.dynamic_update_slice(cache, value, (0, pos, 0, 0))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert got.shape == cache.shape and got.dtype == cache.dtype
        cache = got


@pytest.mark.parametrize("start", range(LEN - CHUNK + 1))
def test_chunk_write_matches_dynamic_update_slice(start):
    """A contiguous chunk lands exactly where the slice update put it."""
    rng = np.random.default_rng(1)
    cache = jnp.asarray(
        rng.standard_normal((1, LEN, NKV, HEAD_DIM)).astype(np.float16)
    )
    value = jnp.asarray(
        rng.standard_normal((1, CHUNK, NKV, HEAD_DIM)).astype(np.float16)
    )
    slots = jnp.arange(CHUNK, dtype=jnp.int32) + start
    got = _chunk_write(cache, value, slots)
    want = jax.lax.dynamic_update_slice(cache, value, (0, jnp.int32(start), 0, 0))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_chunk_write_drops_rows_past_the_end():
    """Rows no slot claims keep their old contents.

    ``dynamic_update_slice`` would have clamped the whole block back inside the
    cache and written it at the wrong offset; dropping is what the caller wants,
    since a position past the cache is a position nothing can attend to.
    """
    cache = jnp.arange(LEN * NKV * HEAD_DIM, dtype=jnp.float16).reshape(
        1, LEN, NKV, HEAD_DIM
    )
    value = jnp.full((1, CHUNK, NKV, HEAD_DIM), -1.0, jnp.float16)
    slots = jnp.array([LEN - 1, LEN], jnp.int32)

    got = np.asarray(_chunk_write(cache, value, slots))
    expected = np.asarray(cache).copy()
    expected[0, LEN - 1] = -1.0
    np.testing.assert_array_equal(got, expected)


# ── 2. The converted graph ─────────────────────────────────────────────────


def _convert(fn, *args):
    """Trace ``fn`` and run it through the project's export pass pipeline."""
    hlo = jax.jit(fn).lower(*args).compiler_ir("stablehlo")
    prog = hlo_to_mil(hlo, minimum_deployment_target=ct.target.iOS18)
    pipeline = build_ct_convert_pass_pipeline()
    pipeline.remove_passes(_REMOVED_PASSES)
    model = ct.convert(
        prog,
        pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )
    return model._mil_program


def _op_types(prog):
    return collections.Counter(
        op.op_type for op in prog.functions["main"].operations
    )


def _decode_args():
    return (
        jnp.zeros((1, LEN, NKV, HEAD_DIM), jnp.float16),  # cache
        jnp.zeros((1, 1, NKV, HEAD_DIM), jnp.float16),    # value
        jnp.zeros((1,), jnp.int32),                       # position
    )


def _prefill_args():
    return (
        jnp.zeros((1, LEN, NKV, HEAD_DIM), jnp.float16),      # cache
        jnp.zeros((1, CHUNK, NKV, HEAD_DIM), jnp.float16),    # value
        jnp.zeros((1,), jnp.int32),                           # start_position
    )


def _entry_sum(cache):
    """A second reader of the cache, so the write is not its only consumer."""
    return jnp.sum(cache.astype(jnp.float32)).reshape(1)


def decode_write(cache, value, position):
    return _entry_sum(cache), _row_write(cache, value, position[0])


def prefill_write(cache, value, start_position):
    slots = start_position[0] + jnp.arange(CHUNK, dtype=jnp.int32)
    return _entry_sum(cache), _chunk_write(cache, value, slots)


def slice_update_write(cache, value, position):
    """The formulation this replaced — the control for the assertions below."""
    return _entry_sum(cache), jax.lax.dynamic_update_slice(
        cache, value, (0, position[0], 0, 0)
    )


@pytest.mark.parametrize(
    "fn, args",
    [(decode_write, _decode_args()), (prefill_write, _prefill_args())],
    ids=["decode", "prefill"],
)
def test_write_leaves_no_slice_update_in_the_graph(fn, args):
    counts = _op_types(_convert(fn, *args))
    assert counts["slice_update"] == 0
    assert counts["select"] == 1


def test_the_old_formulation_would_have_been_caught():
    """Control: ``dynamic_update_slice`` really does produce the op we banned,
    with the runtime ``begin`` that costs the pipeline stall."""
    prog = _convert(slice_update_write, *_decode_args())
    updates = [
        op for op in prog.functions["main"].operations
        if op.op_type == "slice_update"
    ]
    assert len(updates) == 1
    assert updates[0].inputs["begin"].val is None, (
        "a compile-time begin would not stall; the control has to be a runtime one"
    )


# ── 3. The write as Core ML state, at run time ─────────────────────────────


@pytest.fixture(scope="module")
def state_model():
    """``prefill_write`` with its cache bound to state, compiled for CPU+GPU."""
    prog = _convert(prefill_write, *_prefill_args())
    func = prog.functions["main"]
    # The exporter renames the traced results; ``global_kv_caches_to_states``
    # pairs an input ``X`` with an output ``X_out``.
    func.outputs[0].set_name("entry")
    func.outputs[1].set_name("cache_out")
    global_kv_caches_to_states().apply(prog)
    return ct.convert(
        prog,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )


def test_the_state_is_written_from_the_select(state_model):
    func = state_model._mil_program.functions["main"]
    updates = [op for op in func.operations if op.op_type == "coreml_update_state"]
    assert len(updates) == 1
    add = updates[0].value.op
    assert add.op_type == "add"
    # The zero add is load-bearing; see mil_passes/global_cache_states.
    assert {add.x.op.op_type, add.y.op.op_type} == {"fill_like", "select"}
    assert not any(op.op_type == "slice_update" for op in func.operations)


def test_chunk_rows_accumulate_across_predictions(state_model):
    """Two chunks in a row: the second must see what the first wrote.

    Each call writes ``step + 1`` into its own ``CHUNK`` rows, so the sum the
    *next* call reads on entry grows by ``CHUNK * NKV * HEAD_DIM * (step + 1)``
    — but only if the earlier rows survived the state write.
    """
    state = state_model.make_state()
    running = 0.0
    for step in range(3):
        result = state_model.predict(
            {
                "value": np.full((1, CHUNK, NKV, HEAD_DIM), step + 1, np.float16),
                "start_position": np.array([step * CHUNK], np.int32),
            },
            state=state,
        )
        assert result["entry"][0] == pytest.approx(running), f"at chunk {step}"
        running += CHUNK * NKV * HEAD_DIM * (step + 1)

    # A fresh state starts from zero — this is what "new conversation" does.
    fresh = state_model.make_state()
    result = state_model.predict(
        {
            "value": np.ones((1, CHUNK, NKV, HEAD_DIM), np.float16),
            "start_position": np.array([0], np.int32),
        },
        state=fresh,
    )
    assert result["entry"][0] == pytest.approx(0.0)

    # …and the original state is untouched by the fresh one.
    result = state_model.predict(
        {
            "value": np.zeros((1, CHUNK, NKV, HEAD_DIM), np.float16),
            "start_position": np.array([3 * CHUNK], np.int32),
        },
        state=state,
    )
    assert result["entry"][0] == pytest.approx(running)
