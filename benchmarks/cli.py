"""CLI entry point for the benchmark suite (``uv run gemma-bench``)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmarks.power import check_power_available
from benchmarks.runner import (
    COMPUTE_UNITS, BenchmarkConfig, format_summary, run_benchmark, summarize,
)


def _csv(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure prefill/decode speed and CPU+GPU+ANE rail energy per token",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="gemma4-e2b.mlpackage",
                   help="Comma-separated .mlpackage paths, compared A/B")
    p.add_argument("--compute-units", default="cpu-and-gpu,cpu-and-ne",
                   help=f"Comma-separated, from {', '.join(COMPUTE_UNITS)}")
    p.add_argument("--context-lengths", default="512,2048",
                   help="Comma-separated cache sizes (each must be exported)")
    p.add_argument("--runs", type=int, default=3, help="Repetitions per configuration")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-run timeout in seconds (a first ANE compile takes minutes)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for results.json and the power samples "
                        "(default: benchmarks/results/<timestamp>)")
    args = p.parse_args()

    units = _csv(args.compute_units)
    bad = [u for u in units if u not in COMPUTE_UNITS]
    if bad:
        p.error(f"unknown compute units {bad}")
    if args.runs < 3:
        p.error("--runs must be at least 3")
    models = [Path(m).resolve() for m in _csv(args.models)]
    missing = [str(m) for m in models if not m.exists()]
    if missing:
        p.error(f"no such model package: {', '.join(missing)}")
    if not check_power_available():
        sys.exit("powermetrics needs passwordless sudo — see benchmarks/README.md")

    config = BenchmarkConfig(
        models=[str(m) for m in models],
        compute_units=units,
        context_lengths=[int(n) for n in _csv(args.context_lengths)],
        runs=args.runs,
        timeout_s=args.timeout,
    )
    out_dir = Path(args.output_dir or f"benchmarks/results/{time.strftime('%Y%m%d-%H%M%S')}")
    records = run_benchmark(config, out_dir)
    summary = format_summary(summarize(records))
    (out_dir / "summary.md").write_text(summary + "\n")
    print("\n" + summary)
