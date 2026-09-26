"""The host-side embedding lookup reproduces the old in-graph one bit for bit.

The exported functions take embedding rows; the runtime (``GemmaCore``'s
``HostEmbeddings``) dequantizes them from the int4 tables in the package's
``Embeddings/`` directory.  ``_host_rows`` below is a line-for-line Python
rendering of what the Swift reader does with those files.  The reference is
what the graph used to compute: ``_embed_lookup`` on the same block-32 table as
``constexpr_blockwise_shift_scale`` dequantized it, times ``fp16(sqrt(D))``
for the token rows.

The Swift reader is checked against the same reference through a small
fixture (``GemmaCore/Tests/GemmaCoreTests/Fixtures``): this module writes it
and ``test_swift_fixture_is_current`` keeps it in sync; ``swift test`` in
``GemmaCore/`` compares the Swift rows with the expected ones stored there.
Regenerate it with ``uv run python tests/test_host_embeddings.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gemma_chat import host_embeddings
from gemma_chat.mil_passes.quantize_const_weights import (
    _quantize_symmetric_embedding_blocks,
)
from gemma_chat.model import _embed_lookup

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "GemmaCore" / "Tests" / "GemmaCoreTests" / "Fixtures"
)
EMBED_DIM = 1536  # E2B's; sets the token-row multiplier


def _host_rows(tables_dir: Path, name: str, tokens) -> np.ndarray:
    """What ``HostEmbeddings`` does: unpack, fp16(q * scale), fp16(* multiplier)."""
    meta = json.loads((tables_dir / host_embeddings.MANIFEST).read_text())[name]
    rows, cols, group = meta["rows"], meta["cols"], meta["group_size"]
    packed = np.fromfile(tables_dir / f"{name}.int4", np.uint8).reshape(rows, cols // 2)
    scales = np.fromfile(tables_dir / f"{name}.scales", "<f2").reshape(rows, cols // group)

    b = packed[np.asarray(tokens)]
    nibbles = np.empty((len(tokens), cols), np.uint8)
    nibbles[:, 0::2] = b & 0x0F          # even columns: low nibble
    nibbles[:, 1::2] = b >> 4            # odd columns: high nibble
    q = np.where(nibbles >= 8, nibbles.astype(np.int16) - 16, nibbles).astype(np.float32)
    s = np.repeat(scales[np.asarray(tokens)].astype(np.float32), group, axis=1)
    v = (q * s).astype(np.float16)
    mult = np.float16(meta["multiplier"])
    if mult != 1:
        v = (v.astype(np.float32) * np.float32(mult)).astype(np.float16)
    return v


def _graph_rows(table: np.ndarray, tokens, embed_dim: int | None) -> np.ndarray:
    """The old in-graph lookup: gather from the dequantized block-32 table,
    then (token table only) ``* jnp.sqrt(float(D)).astype(float16)``."""
    q, scale = _quantize_symmetric_embedding_blocks(table)
    deq = (q.view(np.int8).astype(np.float32)
           * np.repeat(scale.astype(np.float32), 32, axis=1)).astype(np.float16)
    rows = _embed_lookup(jnp.asarray(deq), jnp.asarray([tokens], jnp.int32))[0]
    if embed_dim is not None:
        rows = rows * jnp.sqrt(float(embed_dim)).astype(jnp.float16)
    return np.asarray(rows)


def _tables(vocab: int, token_cols: int, ple_cols: int, seed: int) -> dict[str, np.ndarray]:
    """Embedding-like fp16 tables with the awkward rows the reader must get right."""
    rng = np.random.default_rng(seed)
    tables = {
        "token_embed": (rng.standard_normal((vocab, token_cols)) * 0.05).astype(np.float16),
        "ple_rows": (rng.standard_normal((vocab, ple_cols)) * 0.5).astype(np.float16),
    }
    for name, t in tables.items():
        t[1, :32] = 0            # an all-zero group (scale falls back to 1/7)
        t[2] *= np.float16(1e-5)  # products in fp16's subnormal range
        # Large rows; the token row overflows to inf once multiplied by sqrt(D).
        t[3] *= np.float16(4e4 if name == "token_embed" else 1e3)
        t[4, ::3] = -t[4, ::3]    # mixed signs: both nibble sign-extensions
        assert np.all(np.isfinite(t))
    return tables


def _check_tables(tmp: Path, tables, tokens):
    host_embeddings.write_tables(tables, EMBED_DIM, tmp)
    for name, table in tables.items():
        host = _host_rows(tmp, name, tokens)
        graph = _graph_rows(table, tokens, EMBED_DIM if name == "token_embed" else None)
        assert host.dtype == graph.dtype == np.float16
        np.testing.assert_array_equal(host.view(np.uint16), graph.view(np.uint16),
                                      err_msg=name)


def test_host_lookup_matches_in_graph_lookup(tmp_path):
    vocab = 300
    tables = _tables(vocab, token_cols=1536, ple_cols=35 * 256 // 4, seed=0)
    _check_tables(tmp_path, tables, tokens=[0, 1, 2, 3, 4, 5, 106, vocab - 1])


def test_multiplier_is_the_graphs_constant():
    graph = jnp.sqrt(float(EMBED_DIM)).astype(jnp.float16)
    assert host_embeddings.table_multipliers(EMBED_DIM)["token_embed"] == float(graph)
    assert host_embeddings.table_multipliers(EMBED_DIM)["ple_rows"] == 1.0


def test_file_sizes_and_manifest(tmp_path):
    tables = _tables(16, token_cols=64, ple_cols=96, seed=1)
    host_embeddings.write_tables(tables, EMBED_DIM, tmp_path)
    manifest = json.loads((tmp_path / host_embeddings.MANIFEST).read_text())
    assert manifest["ple_rows"] == {"rows": 16, "cols": 96, "group_size": 32, "multiplier": 1.0}
    assert (tmp_path / "token_embed.int4").stat().st_size == 16 * 64 // 2
    assert (tmp_path / "token_embed.scales").stat().st_size == 16 * (64 // 32) * 2


# ── the Swift reader's fixture ──────────────────────────────────────────────

_FIXTURE_TOKENS = [0, 1, 2, 3, 4, 7]


def write_swift_fixture(out: Path) -> None:
    """Tiny tables in ``out/Embeddings`` plus the graph's rows in ``expected.json``."""
    tables = _tables(8, token_cols=64, ple_cols=96, seed=2)
    host_embeddings.write_tables(tables, EMBED_DIM, out / host_embeddings.DIR_NAME)
    expected = {"tokens": _FIXTURE_TOKENS}
    for name, table in tables.items():
        rows = _graph_rows(table, _FIXTURE_TOKENS, EMBED_DIM if name == "token_embed" else None)
        expected[name] = rows.view(np.uint16).tolist()  # fp16 bit patterns
    (out / "expected.json").write_text(json.dumps(expected) + "\n")


def test_swift_fixture_is_current(tmp_path):
    write_swift_fixture(tmp_path)

    def files(root: Path) -> dict[str, bytes]:
        return {str(f.relative_to(root)): f.read_bytes()
                for f in sorted(root.rglob("*")) if f.is_file()}

    assert files(tmp_path) == files(FIXTURE_DIR), (
        "GemmaCore's host-embedding fixture is stale; regenerate it with "
        "`uv run python tests/test_host_embeddings.py`"
    )


if __name__ == "__main__":
    import shutil

    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    write_swift_fixture(FIXTURE_DIR)
    print(f"wrote {FIXTURE_DIR}")
