"""The embedding tables the host reads, shipped inside the ``.mlpackage``.

The exported functions take embedding rows, not token ids (see
``decode_coreml``): the runtime looks the rows up itself.  This module writes
the two tables it needs into an ``Embeddings/`` directory in the package, next
to ``Tokenizer/``:

======================  ==================================================
``embeddings.json``     one entry per model input the rows feed
``<input>.int4``        ``rows × cols / 2`` bytes, row-major, packed int4
``<input>.scales``      ``rows × cols / 32`` little-endian fp16, row-major
======================  ==================================================

``embeddings.json`` looks like::

    {"token_embed": {"rows": 262144, "cols": 1536, "group_size": 32,
                     "multiplier": 39.1875},
     "ple_rows":    {"rows": 262144, "cols": 8960, "group_size": 32,
                     "multiplier": 1.0}}

The quantization is exactly the one the graph used for its own in-graph
lookups — symmetric int4 in ``[-7, 7]``, one fp16 scale per 32 consecutive
elements of a row, no offset (``_quantize_symmetric_embedding_blocks``) — so a
host lookup reproduces the old ``gather`` bit for bit:

* element ``c`` of row ``r`` is the nibble ``c % 2`` of byte
  ``r * cols / 2 + c / 2``: **even columns in the low nibble, odd columns in
  the high nibble**, each a 4-bit two's-complement integer;
* its value is ``fp16(q * scale[r, c / 32])`` — the product is exact in fp32,
  so this is a single rounding to fp16, as ``constexpr_blockwise_shift_scale``
  produced it;
* then ``fp16(value * multiplier)``, with ``multiplier`` itself an fp16
  number: ``fp16(sqrt(embed_dim))`` for ``token_embed``, which the graph
  applied right after its gather, and 1 for ``ple_rows``, whose scaling
  stays in the graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gemma_chat.mil_passes.quantize_const_weights import (
    _quantize_symmetric_embedding_blocks,
)

DIR_NAME = "Embeddings"
MANIFEST = "embeddings.json"
GROUP_SIZE = 32

# Rows per packing pass, to bound the transient memory of the 262144-row tables.
_CHUNK_ROWS = 16384


def table_multipliers(embed_dim: int) -> dict[str, float]:
    """The fp16 factor applied after the lookup, per model input.

    ``fp16(sqrt(embed_dim))`` computed the way the graph computed it —
    ``jnp.sqrt(float(d)).astype(float16)``: an fp32 sqrt, rounded to fp16.
    """
    return {
        "token_embed": float(np.float16(np.sqrt(np.float32(embed_dim)))),
        "ple_rows": 1.0,
    }


def pack_int4(q: np.ndarray) -> np.ndarray:
    """``[rows, cols]`` int values in ``[-8, 7]`` → ``[rows, cols / 2]`` uint8,
    even columns in the low nibble."""
    u = q.view(np.int8).astype(np.uint8) & 0x0F
    return u[:, 0::2] | (u[:, 1::2] << 4)


def write_tables(
    tables: dict[str, np.ndarray], embed_dim: int, out_dir: Path,
) -> None:
    """Quantize and write ``{input name: [vocab, cols] table}`` into ``out_dir``.

    ``tables`` must hold exactly ``token_embed`` (the token embedding table)
    and ``ple_rows`` (the per-layer-embedding table), as fp16 — the dtype the
    graph quantized them from.
    """
    multipliers = table_multipliers(embed_dim)
    if set(tables) != set(multipliers):
        raise ValueError(f"expected tables {sorted(multipliers)}, got {sorted(tables)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, table in tables.items():
        rows, cols = table.shape
        if cols % GROUP_SIZE:
            raise ValueError(f"{name}: {cols} columns is not a multiple of {GROUP_SIZE}")
        q, scale = _quantize_symmetric_embedding_blocks(
            np.asarray(table, dtype=np.float16)
        )
        with open(out_dir / f"{name}.int4", "wb") as f:
            for start in range(0, rows, _CHUNK_ROWS):
                f.write(pack_int4(q[start:start + _CHUNK_ROWS]).tobytes())
        scale.astype("<f2").tofile(out_dir / f"{name}.scales")
        del q, scale
        manifest[name] = {
            "rows": rows,
            "cols": cols,
            "group_size": GROUP_SIZE,
            "multiplier": multipliers[name],
        }
        size = sum((out_dir / f"{name}.{ext}").stat().st_size for ext in ("int4", "scales"))
        print(f"  Host embedding table {name}: [{rows}, {cols}] ({size / 1e6:.0f} MB)",
              flush=True)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
