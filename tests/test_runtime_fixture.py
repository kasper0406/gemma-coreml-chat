"""A tiny package with the exported model's function set, for GemmaCore's tests.

``swift test`` in ``GemmaCore/`` loads it through ``CoreMLModel.load`` to check
what only the runtime can get wrong — concurrent conversations sharing one
set of functions, position validation — without a multi-GB export.  It has
the signatures ``gemma-export`` produces (``state_<N>``, ``decode_c0_<N>`` and
``prefill_c0_<N>`` for two sizes, ``head``, an ``Embeddings/`` directory) but
none of the model: a chunk marks the rows of its positions in a global cache
state and returns

    hidden_out = token_embed + ple_rows[..., :D] + (rows marked) + (live ring slots)

so every step's output depends on the state, the ring and the inputs, and a
step that ran on the wrong state or a torn input shows up in the logits.

The checked-in copy lives at ``GemmaCore/Tests/GemmaCoreTests/TinyModel.mlpackage``;
regenerate it with ``uv run python tests/test_runtime_fixture.py``.
``test_fixture_is_current`` compares its content — programs, weights and
embedding tables — with a freshly generated one.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from coremltools.proto import Model_pb2

from gemma_chat import host_embeddings

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "GemmaCore" / "Tests" / "GemmaCoreTests" / "TinyModel.mlpackage"
)
D = 64          # token_embed / hidden width
PLE = 96        # ple_rows width (one chunk)
VOCAB = 8
RING = 6        # sliding_pos_ring length
HEAD_DIM = 4
CHUNK = 4       # prefill tokens per call
SIZES = (512, 1024)


def _chunk(tokens: int, size: int):
    @mb.program(
        input_specs=[
            mb.TensorSpec((1, tokens, D), dtype=types.fp16),
            mb.TensorSpec((1, tokens, PLE), dtype=types.fp16),
            mb.TensorSpec((1,), dtype=types.int32),
            mb.TensorSpec((1, RING), dtype=types.int32),
            mb.StateTensorSpec((1, size, 1, HEAD_DIM), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(token_embed, ple_rows, position, sliding_pos_ring, k_1):
        cache = mb.read_state(input=k_1)
        rows = mb.reshape(
            x=mb.range_1d(start=np.int32(0), end=np.int32(size), step=np.int32(1)),
            shape=[1, size, 1, 1],
        )
        first = mb.reshape(x=position, shape=[1, 1, 1, 1])
        mine = mb.logical_and(
            x=mb.greater_equal(x=rows, y=first),
            y=mb.less(x=rows, y=mb.add(x=first, y=np.int32(tokens))),
        )
        written = mb.coreml_update_state(
            state=k_1,
            value=mb.select(cond=mine, a=np.float16(1), b=cache),
        )
        marked = mb.reduce_sum(x=mb.cast(x=written, dtype="fp32"), keep_dims=False)
        live = mb.reduce_sum(
            x=mb.cast(x=mb.greater_equal(x=sliding_pos_ring, y=np.int32(0)), dtype="fp32"),
            keep_dims=False,
        )
        bias = mb.cast(x=mb.add(x=mb.real_div(x=marked, y=np.float32(HEAD_DIM)), y=live), dtype="fp16")
        ple = mb.slice_by_index(x=ple_rows, begin=[0, 0, 0], end=[1, tokens, D])
        return mb.add(x=mb.add(x=token_embed, y=ple), y=bias, name="hidden_out")

    return prog


def _state(size: int):
    @mb.program(
        input_specs=[mb.StateTensorSpec((1, size, 1, HEAD_DIM), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(k_1):
        cache = mb.read_state(input=k_1)
        return mb.reshape(
            x=mb.slice_by_index(x=cache, begin=[0, 0, 0, 0], end=[1, 1, 1, 1]),
            shape=[1], name="probe",
        )

    return prog


def _head():
    weight = (np.random.default_rng(3).standard_normal((D, VOCAB)) * 0.1).astype(np.float16)

    @mb.program(
        input_specs=[mb.TensorSpec((1, 1, D), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(hidden):
        logits = mb.reshape(x=mb.matmul(x=hidden, y=weight), shape=[VOCAB])
        return mb.cast(x=logits, dtype="fp32", name="logits")

    return prog


def write_fixture(out: Path) -> None:
    programs = {"head": _head()}
    for size in SIZES:
        programs[f"state_{size}"] = _state(size)
        programs[f"decode_c0_{size}"] = _chunk(1, size)
        programs[f"prefill_c0_{size}"] = _chunk(CHUNK, size)

    with tempfile.TemporaryDirectory() as tmp:
        desc = MultiFunctionDescriptor()
        for name, prog in programs.items():
            path = Path(tmp) / f"{name}.mlpackage"
            ct.convert(
                prog, minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32, skip_model_load=True,
            ).save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=name)
        desc.default_function_name = f"decode_c0_{SIZES[0]}"
        if out.exists():
            shutil.rmtree(out)
        save_multifunction(desc, str(out))

    rng = np.random.default_rng(2)
    tables = {
        "token_embed": (rng.standard_normal((VOCAB, D)) * 0.05).astype(np.float16),
        "ple_rows": (rng.standard_normal((VOCAB, PLE)) * 0.5).astype(np.float16),
    }
    host_embeddings.write_tables(tables, 1536, out / host_embeddings.DIR_NAME)


def _content_digest(package: Path) -> str:
    """SHA-256 of everything the runtime reads from ``package``.

    The spec (every function's program and signature) is hashed through
    protobuf's deterministic serialization — ``model.mlmodel``'s own bytes
    differ from save to save in map order — and every other file byte for
    byte, except ``Manifest.json``, whose item ids are fresh UUIDs each time.
    """
    digest = hashlib.sha256()
    spec = Model_pb2.Model()
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        rel = path.relative_to(package).as_posix()
        if rel == "Manifest.json":
            continue
        data = path.read_bytes()
        if path.name == "model.mlmodel":
            spec.ParseFromString(data)
            data = spec.SerializeToString(deterministic=True)
        digest.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def test_fixture_is_current(tmp_path):
    fresh = tmp_path / "TinyModel.mlpackage"
    write_fixture(fresh)
    assert _content_digest(fresh) == _content_digest(FIXTURE), (
        "GemmaCore's TinyModel.mlpackage is stale; regenerate it with "
        "`uv run python tests/test_runtime_fixture.py`"
    )


if __name__ == "__main__":
    write_fixture(FIXTURE)
    print(f"wrote {FIXTURE}")
