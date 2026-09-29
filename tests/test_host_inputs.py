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
import pytest

from gemma_chat.config import E2B_CONFIG
from gemma_chat.decode_coreml import (
    HOST_INPUTS, MASK_VALUE, SCORE_BOUND, _rmsnorm, attention_score_bound, empty_pos_ring,
    host_inputs, ring_with_positions,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "GemmaCore" / "Tests" / "GemmaCoreTests" / "HostInputs.json"
)

# (first position, rows, global cache length): a first step, decode past the
# ring's wrap and the window, a chunk crossing the end of the cache (1022 and
# 1023 fit, 1024 and 1025 do not), and positions whose angles are far past
# fp16's integer range.
CASES = [(0, 1, 512), (700, 1, 1024), (1022, 4, 1024), (65000, 4, 1024)]


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
    chunk crossed the end of the global cache."""
    (_, wrapped, crossing, _) = [host_inputs(s + np.arange(r), np.asarray(ring_with_positions(
        empty_pos_ring(E2B_CONFIG), np.arange(s + r))), n, E2B_CONFIG) for s, r, n in CASES]
    visible = (wrapped["mask_sliding"] == 0).sum()
    assert visible == E2B_CONFIG.sliding_window_size   # the window, not the ring
    write = crossing["write_global"][0]                # [slot, row]
    assert write[1022, 0] == 1 and write[1023, 1] == 1  # 1022 and 1023 fit
    assert write.sum() == 2                            # 1024 and 1025 take no row
    assert not write[:, 2:].any()


def test_the_mask_value_holds_for_every_score_the_bound_allows():
    """``MASK_VALUE`` plus the largest score the export lets through, then
    softmax's shift by a row maximum as large: still finite in fp16, still
    below every unmasked score, and its exp() exactly 0 in fp16 and fp32."""
    mask, bound = np.float16(MASK_VALUE), np.float16(SCORE_BOUND)
    worst = (mask - bound) - bound
    assert worst.dtype == np.float16 and np.isfinite(worst)
    assert mask + bound < -bound
    assert np.exp(np.float32(MASK_VALUE + 2 * SCORE_BOUND)) == 0
    assert np.exp(np.float16(MASK_VALUE + 2 * SCORE_BOUND)) == 0


def test_the_score_bound_is_what_aligned_q_and_k_reach():
    """``attention_score_bound`` is ``hd * max|s_q| * max|s_k|`` (+1%): the
    score of a q and k that are one-hot on the entry both scales peak at."""
    cfg = E2B_CONFIG
    rng = np.random.default_rng(0)
    hd = cfg.effective_head_dim(cfg.attention_types[0])
    j = 17

    params = {}
    for i in range(cfg.num_layers):
        d = cfg.effective_head_dim(cfg.attention_types[i])
        params[f"layers.{i}"] = {"self_attn": {
            "q_norm": {"scale": rng.uniform(-0.5, 0.5, d).astype(np.float32)},
            "k_norm": {"scale": rng.uniform(-0.5, 0.5, d).astype(np.float32)},
        }}
    sq, sk = (params["layers.0"]["self_attn"][n]["scale"] for n in ("q_norm", "k_norm"))
    sq[j], sk[j] = 2.0, -1.5
    bound = attention_score_bound(params, cfg)
    assert bound == pytest.approx(1.01 * hd * 2.0 * 1.5)

    onehot = np.zeros((1, 1, 1, hd), np.float16)
    onehot[..., j] = 1
    q = np.asarray(_rmsnorm(onehot * np.float16(3), sq), np.float32)
    k = np.asarray(_rmsnorm(onehot * np.float16(-7), sk), np.float32)
    score = abs(float((q * k).sum()))
    assert bound / 1.01 * 0.999 <= score <= bound


@pytest.mark.parametrize("start, rows, cache_len", [
    (0, 128, 512),      # a first prefill chunk, padding rows past a short prompt
    (384, 128, 512),    # the chunk that fills the cache
    (1920, 128, 2048),  # the ring long wrapped
    (700, 1, 1024),     # a decode step
])
def test_every_row_sees_its_own_slot(start, rows, cache_len):
    """No row is ever all mask — padding rows are ordinary positions."""
    ring = np.asarray(ring_with_positions(empty_pos_ring(E2B_CONFIG), np.arange(start + rows)))
    got = host_inputs(start + np.arange(rows), ring, cache_len, E2B_CONFIG)
    R = ring.shape[-1]
    for r in range(rows):
        q = start + r
        assert got["mask_sliding"][0, 0, r, q % R] == 0
        assert got["mask_global"][0, 0, r, q] == 0


if __name__ == "__main__":
    FIXTURE.write_text(_fixture())
    print(f"wrote {FIXTURE}")
