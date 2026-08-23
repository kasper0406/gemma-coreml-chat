"""The post-materialization pass that turns global KV cache I/O into state.

**This test builds, compiles and RUNS a CoreML model** — a tiny synthetic one
holding a single "global" cache written with ``slice_update``, exactly the op
``jax.lax.dynamic_update_slice`` lowers to in the real export.

It pins down both halves of what
``gemma_chat.mil_passes.global_cache_states`` promises:

1. the program-level rewrite — the input becomes an fp16 state feature keeping
   its name, a ``read_state`` at the top of the block takes over *every* use of
   the old input, the updated cache is written back with
   ``coreml_update_state``, and the ``_out`` output is gone;
2. the runtime behaviour — the cache contents survive from one prediction to
   the next, and a fresh state starts from zero.

(2) is the part that a plausible-looking implementation gets wrong: handing the
``slice_update`` var straight to ``coreml_update_state`` yields a model that
loads and predicts but keeps only the row written by the current call.  See the
module docstring of the pass for why the zero-add wrapper is needed.
"""

from __future__ import annotations

import numpy as np
import coremltools as ct
import pytest
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

from gemma_chat.mil_passes.global_cache_states import global_kv_caches_to_states

LEN = 8       # cache length ("materialized" global cache size)
HEAD_DIM = 2


def _build_program():
    """A one-cache step: read the cache, write ``pos + 1`` into row ``pos``.

    Mirrors the materialized export in miniature: ``cache`` in, ``cache_out``
    out, plus a second reader of the input (the sum) so the pass has to rewire
    more than the write itself.
    """

    @mb.program(
        input_specs=[
            mb.TensorSpec((1,), dtype=types.int32),
            mb.TensorSpec((1, LEN, 1, HEAD_DIM), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(pos, cache):
        # What the cache held on entry — the observable proof of persistence.
        entry = mb.reduce_sum(
            x=mb.cast(x=cache, dtype="fp32"), axes=[0, 1, 2, 3], keep_dims=True,
        )
        entry = mb.reshape(x=entry, shape=[1], name="entry")

        value = mb.cast(x=mb.add(x=pos, y=np.int32(1)), dtype="fp16")
        value = mb.tile(
            x=mb.reshape(x=value, shape=[1, 1, 1, 1]), reps=[1, 1, 1, HEAD_DIM],
        )
        begin = mb.concat(
            values=[np.int32([0]), pos, np.int32([0]), np.int32([0])], axis=0,
        )
        end = mb.add(x=begin, y=np.int32([1, 1, 1, HEAD_DIM]))
        cache_out = mb.slice_update(
            x=cache, update=value, begin=begin, end=end, name="cache_out",
        )
        return entry, cache_out

    return prog


@pytest.fixture(scope="module")
def statified_program():
    prog = _build_program()
    global_kv_caches_to_states().apply(prog)
    return prog


def test_input_becomes_a_state_and_output_disappears(statified_program):
    func = statified_program.functions["main"]

    cache_var = func.inputs["cache"]
    assert types.is_state(cache_var.sym_type)
    assert cache_var.dtype == types.fp16
    assert tuple(cache_var.shape) == (1, LEN, 1, HEAD_DIM)
    # Ordinary inputs are untouched, and the input order is preserved.
    assert list(func.inputs) == ["pos", "cache"]
    assert not types.is_state(func.inputs["pos"].sym_type)

    assert [var.name for var in func.outputs] == ["entry"]


def test_read_state_is_first_and_feeds_every_use(statified_program):
    func = statified_program.functions["main"]
    ops = list(func.operations)

    read = ops[0]
    assert read.op_type == "read_state"
    assert read.input is func.inputs["cache"]

    # No op reads the state var except read_state / coreml_update_state, i.e.
    # every consumer of the old input now consumes the (dominating) read.
    consumers = {op.op_type for op in func.inputs["cache"].child_ops}
    assert consumers == {"read_state", "coreml_update_state"}
    # Both the summation and the slice_update took the read var.
    readers = {op.op_type for op in read.outputs[0].child_ops}
    assert {"cast", "slice_update", "fill_like"} <= readers


def test_update_state_writes_the_fully_updated_cache(statified_program):
    func = statified_program.functions["main"]
    updates = [op for op in func.operations if op.op_type == "coreml_update_state"]
    assert len(updates) == 1
    update = updates[0]
    assert update.state is func.inputs["cache"]

    # The written value is the updated cache, kept out of the runtime's
    # in-place slice path by a zero add (see the pass docstring).
    add = update.value.op
    assert add.op_type == "add"
    operands = {add.x.op.op_type, add.y.op.op_type}
    assert operands == {"fill_like", "slice_update"}
    slice_update = add.x.op if add.x.op.op_type == "slice_update" else add.y.op
    assert slice_update.name == "cache_out"


@pytest.fixture(scope="module")
def statified_model(statified_program):
    return ct.convert(
        statified_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )


def test_state_feature_replaces_the_cache_io(statified_model):
    spec = statified_model._spec
    assert [feat.name for feat in spec.description.input] == ["pos"]
    assert [feat.name for feat in spec.description.output] == ["entry"]
    assert [feat.name for feat in spec.description.state] == ["cache"]
    array = spec.description.state[0].type.stateType.arrayType
    assert list(array.shape) == [1, LEN, 1, HEAD_DIM]


def test_cache_contents_persist_across_predictions(statified_model):
    state = statified_model.make_state()

    # Step `pos` writes (pos + 1) into row `pos`, across HEAD_DIM lanes, so the
    # sum seen on entry grows by HEAD_DIM * pos every call — but only if the
    # previous rows are still there.
    running = 0.0
    for pos in range(4):
        result = statified_model.predict(
            {"pos": np.array([pos], dtype=np.int32)}, state=state,
        )
        assert result["entry"][0] == pytest.approx(running), f"at pos {pos}"
        running += HEAD_DIM * (pos + 1)

    # A fresh state starts from zero — this is what "new conversation" does.
    fresh = statified_model.make_state()
    result = statified_model.predict(
        {"pos": np.array([0], dtype=np.int32)}, state=fresh,
    )
    assert result["entry"][0] == pytest.approx(0.0)

    # …and the original state is untouched by the fresh one.
    result = statified_model.predict(
        {"pos": np.array([4], dtype=np.int32)}, state=state,
    )
    assert result["entry"][0] == pytest.approx(running)
