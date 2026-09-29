"""``GemmaConfig.swift`` mirrors the architecture constants the host needs for
the per-step inputs (RoPE rows, sliding mask) — no CoreML feature carries
them.  A variant whose values differ would get silently wrong attention, so
every variant must match the one set the runtime has."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gemma_chat.config import VARIANTS
from gemma_chat.model import AttentionType

SWIFT = Path(__file__).resolve().parents[1] / "GemmaCore" / "Sources" / "GemmaCore" / "GemmaConfig.swift"


def _swift_constants() -> dict[str, float]:
    text = SWIFT.read_text()
    return {
        name: float(value.replace("_", ""))
        for name, value in re.findall(r"static let (\w+) = ([0-9_.]+)\s*$", text, re.MULTILINE)
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_swift_host_constants_match_the_config(variant):
    cfg = VARIANTS[variant][0]
    swift = _swift_constants()
    assert swift["slidingWindow"] == cfg.sliding_window_size
    assert swift["slidingRopeBase"] == cfg.rope_base_frequency
    assert swift["slidingHeadDim"] == cfg.head_dim
    assert swift["globalRopeBase"] == cfg.global_rope_base_frequency
    assert swift["globalHeadDim"] == cfg.effective_head_dim(AttentionType.GLOBAL)
