"""CLI entry point for the benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.plot import plot_all
from benchmarks.power import check_power_available
from benchmarks.runner import BenchmarkConfig, RunResult, run_benchmark, save_results


# Default context lengths exercise both the short and long ends of the range.
DEFAULT_CONTEXT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192, 16384)


def _parse_int_csv(s: str) -> list[int]:
    return [int(tok) for tok in s.split(",") if tok.strip()]


def _parse_csv(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _print_summary(results: list[RunResult]) -> None:
    if not results:
        print("No results to summarize.")
        return
    import numpy as np
    from collections import defaultdict

    grouped: dict[tuple[str, int], list[RunResult]] = defaultdict(list)
    for r in results:
        if not r.timed_out and r.error is None:
            grouped[(r.backend, r.context_length)].append(r)

    print(f"\n{'Backend':<6} {'Context':>10} {'Prefill tok/s':>18} "
          f"{'Decode tok/s':>18} {'Power W':>10}")
    print("─" * 66)
    for (backend, ctx), runs in sorted(grouped.items()):
        pref = np.array([r.prefill_tokens_per_sec for r in runs])
        dec = np.array([r.decode_tokens_per_sec for r in runs])
        pw = [r.power.get("mean_total_w", 0.0) for r in runs if r.power]
        pref_s = f"{pref.mean():.1f} ± {pref.std():.1f}"
        dec_s = f"{dec.mean():.1f} ± {dec.std():.1f}"
        pow_s = f"{np.mean(pw):.1f}" if pw else "—"
        print(f"{backend:<6} {ctx:>10,} {pref_s:>18} {dec_s:>18} {pow_s:>10}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark CoreML inference via the GemmaBench Swift binary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gemma4-e2b.mlpackage",
                   help="Path to .mlpackage")
    p.add_argument("--backends", default="cpu,gpu,ane,all",
                   help="Comma-separated backends (cpu, ane, gpu, all)")
    p.add_argument("--context-lengths",
                   default=",".join(str(c) for c in DEFAULT_CONTEXT_LENGTHS),
                   help="Comma-separated context lengths")
    p.add_argument("--runs", type=int, default=3,
                   help="Measurement runs per configuration")
    p.add_argument("--decode-tokens", type=int, default=32,
                   help="Decode tokens per run")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-run timeout (seconds)")
    p.add_argument("--output-dir", default="benchmarks/results",
                   help="Output directory for JSON + plots")
    p.add_argument("--no-power", action="store_true",
                   help="Disable power monitoring")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the primer prefill+decode before each measurement")
    args = p.parse_args()

    enable_power = not args.no_power and check_power_available()
    if not args.no_power and not enable_power:
        print("⚠ powermetrics unavailable (needs sudo NOPASSWD) — disabled",
              file=sys.stderr)

    config = BenchmarkConfig(
        model_path=args.model,
        backends=_parse_csv(args.backends),
        context_lengths=_parse_int_csv(args.context_lengths),
        num_runs=args.runs,
        decode_tokens=args.decode_tokens,
        timeout_s=args.timeout,
        enable_power=enable_power,
        no_warmup=args.no_warmup,
    )

    results = run_benchmark(config)

    import time
    tag = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"results-{tag}.json"
    save_results(results, json_path, config)
    try:
        plot_all(json_path, output_dir / f"plots-{tag}")
    except ImportError:
        print("matplotlib not installed — skipping plot generation "
              "(run `uv sync --group dev`)", file=sys.stderr)

    _print_summary(results)
