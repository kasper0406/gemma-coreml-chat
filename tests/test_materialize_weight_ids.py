"""Same-named consts in different source functions must not share a weight blob.

**This test builds, materializes, compiles and RUNS CoreML models** (tiny ones).

``materialize_symbolic_shape_program`` hands every const it clones a
``weight_id`` derived from the const's *name* when it has none.  Const names are
unique only within a function, so ``prefill`` and ``decode`` each holding a
differently-shaped const of the same name — in the real export both traces name
their leading ``arange`` ``range_1d_0``, one of length ``CHUNK_SIZE`` and one of
``sliding_window_size`` — collide in the weight blob: one value is written,
both functions point at it, and the one whose shape disagrees makes the whole
package unloadable ("Attribute val has incompatible type with operation
output").

The collision only bites once a const is big enough to live in the blob rather
than inline in the proto (10+ elements), which is why raising ``CHUNK_SIZE``
from 8 to 128 is what surfaced it.  ``materialize._scope_auto_weight_ids``
re-scopes those invented ids by source function; this test pins that down at
both ends — the blob offsets differ, and both functions read back their own
constant.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import coremltools as ct
import pytest
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, get_new_symbol, types

from gemma_chat.materialize import materialize_mlpackage

SIZES = (16, 32)          # materialized cache sizes
LENGTHS = {"prefill": 128, "decode": 512}   # per-phase const lengths
INDEX = 100               # both consts are `arange`, so value == index


def _make_function(const_len: int) -> Function:
    """One dynamic-shape function holding a const named ``shared_name``."""
    sym = get_new_symbol()
    with Function(
        inputs={
            "cache": mb.placeholder(shape=(1, sym), dtype=types.fp16),
            "pos": mb.placeholder(shape=(1,), dtype=types.int32),
        },
        opset_version=ct.target.iOS18,
    ) as f:
        cache, pos = f.inputs["cache"], f.inputs["pos"]
        idx = mb.const(val=np.arange(const_len, dtype=np.int32), name="shared_name")
        # Runtime index, so the const survives instead of folding away.
        picked = mb.cast(x=mb.gather(x=idx, indices=pos, axis=0), dtype="fp16")
        f.set_outputs([
            mb.add(x=cache, y=picked, name="sum"),
            mb.identity(x=picked, name="picked"),
        ])
    return f


@pytest.fixture(scope="module")
def materialized(tmp_path_factory) -> Path:
    prog = Program()
    for name, length in LENGTHS.items():
        prog.add_function(name, _make_function(length))
    prog.default_function_name = "decode"
    prog.export_as_multifunction = True
    prog.skip_all_passes = True

    model = _mil_convert(
        prog,
        convert_from="milinternal",
        convert_to="mlprogram",
        specification_version=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    root = tmp_path_factory.mktemp("weight_ids")
    src = root / "dynamic.mlpackage"
    if src.exists():
        shutil.rmtree(src)
    model.save(str(src))

    dst = root / "materialized.mlpackage"
    materialize_mlpackage(src, dst, list(SIZES))
    return dst


def test_same_named_consts_get_separate_blob_entries(materialized):
    spec = ct.models.MLModel(str(materialized), skip_model_load=True)._spec
    offsets: dict[str, set[int]] = {}
    for fname, func in spec.mlProgram.functions.items():
        block = func.block_specializations[func.opset]
        for op in block.operations:
            if op.type != "const" or op.outputs[0].name != "shared_name":
                continue
            value = op.attributes["val"]
            assert value.HasField("blobFileValue"), (
                f"{fname}: const is inline, the collision cannot happen — "
                "this test no longer tests anything"
            )
            declared = [d.constant.size for d in op.outputs[0].type.tensorType.dimensions]
            stored = [d.constant.size for d in value.type.tensorType.dimensions]
            assert declared == stored, f"{fname}: declared {declared}, blob {stored}"
            phase = fname.rsplit("_", 1)[0]
            assert declared == [LENGTHS[phase]], fname
            offsets.setdefault(phase, set()).add(value.blobFileValue.offset)

    assert set(offsets) == set(LENGTHS)
    # All sizes of one phase are clones and keep sharing one blob entry; the two
    # phases must not.
    for phase, offs in offsets.items():
        assert len(offs) == 1, f"{phase} lost weight sharing across sizes: {offs}"
    assert offsets["prefill"] != offsets["decode"]


@pytest.mark.parametrize("phase", sorted(LENGTHS))
def test_each_function_reads_its_own_constant(materialized, phase):
    size = SIZES[0]
    model = ct.models.MLModel(
        str(materialized),
        function_name=f"{phase}_{size}",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    out = model.predict({
        "cache": np.zeros((1, size), dtype=np.float16),
        "pos": np.array([INDEX], dtype=np.int32),
    })
    assert float(np.asarray(out["picked"]).reshape(-1)[0]) == pytest.approx(INDEX)
