"""The host inputs GemmaCore builds are ``decode_coreml.host_inputs``'.

The exported functions take no position: every step, the runtime
(``GemmaCore``'s ``HostInputs``) computes the RoPE rows, the attention masks
and the cache-write selections itself.  ``decode_coreml.host_inputs`` is the
reference (``tests/test_sliding_state_write.py`` checks it against the
reference model); this module writes what it returns for a handful of steps
into ``GemmaCore/Tests/GemmaCoreTests/HostInputs.json``, and
``swift test`` in ``GemmaCore/`` compares the Swift buffers with it.
``test_swift_fixture_is_current`` keeps the fixture in sync; regenerate it with
``uv run python tests/test_host_inputs.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from gemma_chat.config import E2B_CONFIG
from gemma_chat.decode_coreml import HOST_INPUTS, empty_pos_ring, host_inputs, ring_with_positions

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "GemmaCore" / "Tests" / "GemmaCoreTests" / "HostInputs.json"
)

# (first position, rows, global cache length): a first step, decode past the
# ring's wrap and the window, a chunk running off the end of the cache, and
# positions whose angles are far past fp16's integer range.
CASES = [(0, 1, 512), (700, 1, 1024), (1020, 4, 1024), (65000, 4, 1024)]


def _case(start: int, rows: int, cache_len: int) -> dict:
    # The ring after a conversation that wrote every position from 0.
    ring = np.asarray(ring_with_positions(empty_pos_ring(E2B_CONFIG),
                                          np.arange(start + rows)))
    got = host_inputs(start + np.arange(rows), ring, cache_len, E2B_CONFIG)
    return {
        "start": start, "rows": rows, "cache_length": cache_len,
        "inputs": {
            name: {"shape": list(got[name].shape),
                   "fp16": base64.b64encode(got[name].astype("<f2").tobytes()).decode()}
            for name in HOST_INPUTS
        },
    }


def _fixture() -> str:
    return json.dumps([_case(*c) for c in CASES], indent=1) + "\n"


def test_swift_fixture_is_current():
    assert FIXTURE.read_text() == _fixture(), (
        "GemmaCore's host_inputs.json is stale; regenerate it with "
        "`uv run python tests/test_host_inputs.py`"
    )


def test_masks_and_writes_cover_the_cases_that_matter():
    """Guard the fixture: the ring has wrapped and the window cut in, and a
    chunk ran off the end of the global cache."""
    (_, wrapped, off_end, _) = [host_inputs(s + np.arange(r), np.asarray(ring_with_positions(
        empty_pos_ring(E2B_CONFIG), np.arange(s + r))), n, E2B_CONFIG) for s, r, n in CASES]
    visible = (wrapped["mask_sliding"] == 0).sum()
    assert visible == E2B_CONFIG.sliding_window_size   # the window, not the ring
    assert off_end["write_global"].sum() == 4          # 1020..1023 fit
    assert (off_end["write_global"][0, :, -1] == 0).sum() == 1024 - 1


if __name__ == "__main__":
    FIXTURE.write_text(_fixture())
    print(f"wrote {FIXTURE}")
