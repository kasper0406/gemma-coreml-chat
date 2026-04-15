"""CLI entry point for the benchmark suite."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmarks.plot import plot_all
from benchmarks.power import check_power_available
from benchmarks.prompts import (
    DEFAULT_CONTEXT_LENGTHS,
    realistic_prompts,
    synthetic_prompt,
)
from benchmarks.runner import BenchmarkConfig, RunResult, run_benchmark, save_results


def _print_summary(results: list[RunResult]) -> None:
    """Print a summary table to stdout."""
    if not results:
        print("No results to summarize.")
        return

    # Group by (backend, context_length)
    from collections import defaultdict
    import numpy as np

    grouped: dict[tuple[str, int], list[RunResult]] = defaultdict(list)
    for r in results:
        if not r.timed_out and r.error is None:
            grouped[(r.backend, r.context_length)].append(r)

    print(f"\n{'Backend':<10} {'Context':>10} {'Prefill tok/s':>18} {'Decode tok/s':>18} {'Power W':>10}")
    print("─" * 70)

    for (backend, ctx_len), runs in sorted(grouped.items()):
        pref = np.array([r.prefill_tokens_per_sec for r in runs])
        dec = np.array([r.decode_tokens_per_sec for r in runs])
        power_vals = [r.power.get("mean_total_w", 0) for r in runs if r.power]

        pref_str = f"{pref.mean():.1f} ± {pref.std():.1f}"
        dec_str = f"{dec.mean():.1f} ± {dec.std():.1f}"
        pow_str = f"{np.mean(power_vals):.1f}" if power_vals else "—"

        print(f"{backend:<10} {ctx_len:>10,} {pref_str:>18} {dec_str:>18} {pow_str:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CoreML inference (prefill + decode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="gemma4-e2b.mlpackage",
        help="Path to .mlpackage",
    )
    parser.add_argument(
        "--backends", type=str, default="cpu,ane,all",
        help="Comma-separated backends (cpu, ane, all)",
    )
    parser.add_argument(
        "--context-lengths", type=str,
        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated context lengths to benchmark",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of measured runs per configuration",
    )
    parser.add_argument(
        "--decode-tokens", type=int, default=64,
        help="Number of decode tokens per run",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Max seconds per single run before aborting",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Output directory for JSON + plots",
    )
    parser.add_argument(
        "--no-power", action="store_true",
        help="Skip power monitoring",
    )
    parser.add_argument(
        "--prompts", choices=["realistic", "synthetic", "both"], default="both",
        help="Which prompts to use",
    )
    parser.add_argument(
        "--model-id", type=str, default="google/gemma-4-E2B-it",
        help="HuggingFace model ID (for tokenizer)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=65536,
        help="Maximum sequence length (must match model export)",
    )

    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    context_lengths = sorted(int(c.strip()) for c in args.context_lengths.split(","))
    output_dir = Path(args.output_dir)

    # Check power
    enable_power = not args.no_power
    if enable_power:
        if check_power_available():
            print("✓ Power monitoring available (sudo powermetrics)")
        else:
            print("⚠ sudo powermetrics not available — disabling power monitoring")
            print("  To enable: run with sudo, or configure sudoers for powermetrics")
            enable_power = False

    # Load tokenizer
    print("Loading tokenizer…", flush=True)
    from gemma_chat.generate import load_tokenizer
    tokenizer = load_tokenizer(args.model_id)

    # Build prompts for each context length
    prompt_ids_by_length: dict[int, list[int]] = {}

    if args.prompts in ("realistic", "both"):
        for p in realistic_prompts(tokenizer):
            # Find the closest context length bucket
            for cl in context_lengths:
                if abs(p.length - cl) < cl * 0.2:  # within 20%
                    prompt_ids_by_length[cl] = p.token_ids
                    break

    if args.prompts in ("synthetic", "both"):
        for cl in context_lengths:
            if cl not in prompt_ids_by_length:
                sp = synthetic_prompt(tokenizer, cl)
                prompt_ids_by_length[cl] = sp.token_ids

    # Ensure all context lengths have prompts
    for cl in context_lengths:
        if cl not in prompt_ids_by_length:
            sp = synthetic_prompt(tokenizer, cl)
            prompt_ids_by_length[cl] = sp.token_ids

    print(f"\nBenchmark configuration:")
    print(f"  Model:           {args.model}")
    print(f"  Backends:        {backends}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Runs:            {args.runs}")
    print(f"  Decode tokens:   {args.decode_tokens}")
    print(f"  Timeout:         {args.timeout}s")
    print(f"  Power:           {'enabled' if enable_power else 'disabled'}")
    print(f"  Max seq len:     {args.max_seq_len}")

    config = BenchmarkConfig(
        model_path=args.model,
        backends=backends,
        context_lengths=context_lengths,
        num_runs=args.runs,
        decode_tokens=args.decode_tokens,
        timeout_s=args.timeout,
        enable_power=enable_power,
        max_seq_len=args.max_seq_len,
    )

    # Run benchmarks
    start = time.perf_counter()
    results = run_benchmark(config, prompt_ids_by_length)
    elapsed = time.perf_counter() - start

    print(f"\nBenchmark completed in {elapsed:.1f}s")
    _print_summary(results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"benchmark_{timestamp}.json"
    save_results(results, config, results_path)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    try:
        plots = plot_all(results_path, output_dir)
        for p in plots:
            print(f"  Plot: {p}")
    except ImportError:
        print("⚠ matplotlib not available — skipping plots")
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")


if __name__ == "__main__":
    main()
