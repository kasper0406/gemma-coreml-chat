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


def _build_program(alias: bool = False, reader: str = "live"):
    """A one-cache step: read the cache, write ``pos + 1`` into row ``pos``.

    Mirrors the materialized export in miniature: ``k_0`` in, ``k_0_out`` out,
    a reader of the input (``entry``, the sum on entry) so the pass has to
    rewire more than the write itself, and — as the attention reads the cache
    it has just written — a reader of the updated cache (``after``).

    ``alias`` routes the output through an ``identity``, the way a converter
    may name an output; ``reader="none"`` leaves the write a sink, and
    ``reader="dead"`` gives it only a reader dead-code elimination removes.
    """

    @mb.program(
        input_specs=[
            mb.TensorSpec((1,), dtype=types.int32),
            mb.TensorSpec((1, LEN, 1, HEAD_DIM), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(pos, k_0):
        # What the cache held on entry — the observable proof of persistence.
        entry = mb.reduce_sum(
            x=mb.cast(x=k_0, dtype="fp32"), axes=[0, 1, 2, 3], keep_dims=True,
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
        updated = mb.slice_update(
            x=k_0, update=value, begin=begin, end=end,
            name="updated" if alias else "k_0_out",
        )
        out = mb.identity(x=updated, name="k_0_out") if alias else updated
        if reader == "none":
            return entry, out
        if reader == "dead":
            mb.cast(x=updated, dtype="fp32", name="unused")
            return entry, out
        after = mb.reduce_sum(
            x=mb.cast(x=updated, dtype="fp32"), axes=[0, 1, 2, 3], keep_dims=True,
        )
        after = mb.reshape(x=after, shape=[1], name="after")
        return entry, after, out

    return prog


@pytest.fixture(scope="module")
def statified_program():
    prog = _build_program()
    global_kv_caches_to_states().apply(prog)
    return prog


def test_input_becomes_a_state_and_output_disappears(statified_program):
    func = statified_program.functions["main"]

    cache_var = func.inputs["k_0"]
    assert types.is_state(cache_var.sym_type)
    assert cache_var.dtype == types.fp16
    assert tuple(cache_var.shape) == (1, LEN, 1, HEAD_DIM)
    # Ordinary inputs are untouched, and the input order is preserved.
    assert list(func.inputs) == ["pos", "k_0"]
    assert not types.is_state(func.inputs["pos"].sym_type)

    assert [var.name for var in func.outputs] == ["entry", "after"]


def test_read_state_is_first_and_feeds_every_use(statified_program):
    func = statified_program.functions["main"]
    ops = list(func.operations)

    read = ops[0]
    assert read.op_type == "read_state"
    assert read.input is func.inputs["k_0"]

    # No op reads the state var except read_state / coreml_update_state, i.e.
    # every consumer of the old input now consumes the (dominating) read.
    consumers = {op.op_type for op in func.inputs["k_0"].child_ops}
    assert consumers == {"read_state", "coreml_update_state"}
    # Both the summation and the slice_update took the read var.
    readers = {op.op_type for op in read.outputs[0].child_ops}
    assert {"cast", "slice_update", "fill_like"} <= readers


def test_update_state_writes_the_fully_updated_cache(statified_program):
    func = statified_program.functions["main"]
    updates = [op for op in func.operations if op.op_type == "coreml_update_state"]
    assert len(updates) == 1
    update = updates[0]
    assert update.state is func.inputs["k_0"]

    # The written value is the updated cache, kept out of the runtime's
    # in-place slice path by a zero add (see the pass docstring).
    add = update.value.op
    assert add.op_type == "add"
    operands = {add.x.op.op_type, add.y.op.op_type}
    assert operands == {"fill_like", "slice_update"}
    slice_update = add.x.op if add.x.op.op_type == "slice_update" else add.y.op
    assert slice_update.name == "k_0_out"


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
    assert [feat.name for feat in spec.description.output] == ["entry", "after"]
    assert [feat.name for feat in spec.description.state] == ["k_0"]
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


# ── The write is consumed, never a sink ─────────────────────────────────────


def test_the_write_follows_its_value_and_feeds_the_later_readers(statified_program):
    """A ``coreml_update_state`` nothing reads makes ANECompiler throw once it
    lands in an ANE segment; the pass must write where the value is produced
    and hand the write's result to everything after it."""
    func = statified_program.functions["main"]
    ops = [op for op in func.operations if op.op_type != "const"]
    update = next(op for op in ops if op.op_type == "coreml_update_state")
    value_op = next(op for op in ops if op.name == "k_0_out")

    i = ops.index(value_op)
    assert [op.op_type for op in ops[i + 1:i + 4]] == ["fill_like", "add", "coreml_update_state"]
    assert ops[i + 3] is update

    written = update.outputs[0]
    assert written.child_ops, "the write is a sink"
    # Nothing after the write reads the unwritten value any more.
    assert [op for op in value_op.outputs[0].child_ops if op is not update.value.op] == []


def test_the_later_readers_see_the_written_cache(statified_model):
    state = statified_model.make_state()
    running = 0.0
    for pos in range(4):
        running += HEAD_DIM * (pos + 1)
        result = statified_model.predict({"pos": np.array([pos], dtype=np.int32)}, state=state)
        assert result["after"][0] == pytest.approx(running), f"at pos {pos}"


@pytest.mark.parametrize("reader", ["none", "dead"])
def test_a_write_nothing_reads_fails_the_pass(reader):
    """Including a write whose only reader is dead code: dead-code elimination
    would remove the unused cast and leave the write a sink."""
    with pytest.raises(ValueError, match="no reader"):
        global_kv_caches_to_states().apply(_build_program(reader=reader))


def test_an_output_alias_is_written_where_the_value_is_produced():
    """``updated -> after`` and ``identity(updated) -> k_0_out``: writing after
    the identity would leave ``after`` reading the unwritten value and the
    write with no reader."""
    prog = _build_program(alias=True)
    global_kv_caches_to_states().apply(prog)
    func = prog.functions["main"]
    update = next(op for op in func.operations if op.op_type == "coreml_update_state")
    assert update.value.op.y.op.name == "updated"
    readers = {op.op_type for op in update.outputs[0].child_ops}
    assert "cast" in readers  # the `after` reduction reads the write
    assert [var.name for var in func.outputs] == ["entry", "after"]


def test_a_cache_without_an_output_becomes_a_read_only_state():
    """A chunk that only reads another chunk's cache (the KV-shared layers)."""

    @mb.program(
        input_specs=[mb.TensorSpec((1, LEN, 1, HEAD_DIM), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(v_3):
        return mb.reduce_sum(x=v_3, axes=[1], keep_dims=False, name="total")

    global_kv_caches_to_states().apply(prog)
    func = prog.functions["main"]
    assert types.is_state(func.inputs["v_3"].sym_type)
    assert [op.op_type for op in func.inputs["v_3"].child_ops] == ["read_state"]
    assert not any(op.op_type == "coreml_update_state" for op in func.operations)


def test_non_cache_io_pairs_are_left_alone():
    """A chunk's ``hidden`` / ``hidden_out`` is fp16 in and out, like a cache."""

    @mb.program(
        input_specs=[mb.TensorSpec((1, 1, 4), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(hidden):
        return mb.add(x=hidden, y=np.float16(1), name="hidden_out")

    global_kv_caches_to_states().apply(prog)
    assert not types.is_state(prog.functions["main"].inputs["hidden"].sym_type)
