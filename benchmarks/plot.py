"""Matplotlib plotting for benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class _Series:
    """Aggregated series for one backend."""

    context_lengths: list[int]
    means: list[float]
    stds: list[float]


def _aggregate(
    results: list[dict],
    backends: list[str],
    metric: str,
) -> dict[str, _Series]:
    """Group results by backend, compute mean/std per context length."""
    series: dict[str, _Series] = {}

    for backend in backends:
        backend_runs = [r for r in results if r["backend"] == backend and not r.get("timed_out")]
        if not backend_runs:
            continue

        ctx_lengths = sorted(set(r["context_length"] for r in backend_runs))
        means = []
        stds = []
        valid_lengths = []

        for cl in ctx_lengths:
            vals = [r[metric] for r in backend_runs if r["context_length"] == cl and r.get(metric, 0) > 0]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                valid_lengths.append(cl)

        if valid_lengths:
            series[backend] = _Series(valid_lengths, means, stds)

    return series


def _aggregate_power(
    results: list[dict],
    backends: list[str],
    component: str,  # "mean_cpu_w", "mean_gpu_w", "mean_ane_w", "mean_total_w"
) -> dict[str, _Series]:
    """Aggregate power metrics by backend and context length."""
    series: dict[str, _Series] = {}

    for backend in backends:
        backend_runs = [
            r for r in results
            if r["backend"] == backend and not r.get("timed_out") and r.get("power")
        ]
        if not backend_runs:
            continue

        ctx_lengths = sorted(set(r["context_length"] for r in backend_runs))
        means = []
        stds = []
        valid_lengths = []

        for cl in ctx_lengths:
            vals = [
                r["power"][component]
                for r in backend_runs
                if r["context_length"] == cl and r.get("power", {}).get(component, 0) > 0
            ]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                valid_lengths.append(cl)

        if valid_lengths:
            series[backend] = _Series(valid_lengths, means, stds)

    return series


BACKEND_COLORS = {
    "cpu": "#2196F3",
    "ane": "#4CAF50",
    "all": "#FF9800",
}

BACKEND_LABELS = {
    "cpu": "CPU Only",
    "ane": "CPU + ANE",
    "all": "CPU + GPU + ANE",
}


def _plot_metric(
    series: dict[str, _Series],
    title: str,
    ylabel: str,
    output_path: Path,
    hardware_info: str = "",
    log_x: bool = True,
) -> None:
    """Plot a single metric with mean ± stddev bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for backend, s in series.items():
        color = BACKEND_COLORS.get(backend, "#666666")
        label = BACKEND_LABELS.get(backend, backend)
        means = np.array(s.means)
        stds = np.array(s.stds)

        ax.plot(s.context_lengths, means, "o-", color=color, label=label, linewidth=2)
        ax.fill_between(
            s.context_lengths,
            means - stds,
            means + stds,
            alpha=0.2,
            color=color,
        )

    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(series[next(iter(series))].context_lengths if series else [])
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel(ylabel)
    full_title = title
    if hardware_info:
        full_title += f"\n{hardware_info}"
    ax.set_title(full_title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all(results_path: Path, output_dir: Path) -> list[Path]:
    """Generate all benchmark plots from a JSON results file.

    Returns list of generated plot file paths.
    """
    import matplotlib
    matplotlib.use("Agg")

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]
    hardware = data.get("hardware", {})
    backends = config["backends"]

    hw_info = f"{hardware.get('node', '')} — {hardware.get('machine', '')} — macOS {hardware.get('mac_ver', '')}"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    # Plot 1: Prefill tok/s
    prefill_series = _aggregate(results, backends, "prefill_tokens_per_sec")
    if prefill_series:
        p = output_dir / "prefill_throughput.png"
        _plot_metric(prefill_series, "Prefill Throughput", "Tokens / second", p, hw_info)
        plots.append(p)

    # Plot 2: Decode tok/s
    decode_series = _aggregate(results, backends, "decode_tokens_per_sec")
    if decode_series:
        p = output_dir / "decode_throughput.png"
        _plot_metric(decode_series, "Decode Throughput", "Tokens / second", p, hw_info)
        plots.append(p)

    # Plot 3: Total power (watts)
    power_series = _aggregate_power(results, backends, "mean_total_w")
    if power_series:
        p = output_dir / "power_watts.png"
        _plot_metric(power_series, "Power Consumption", "Watts", p, hw_info)
        plots.append(p)

    # Plot 4: Watts per token (derived)
    # watts_per_tok = mean_total_w / decode_tok_per_sec
    wpt_series: dict[str, _Series] = {}
    for backend in backends:
        backend_runs = [
            r for r in results
            if r["backend"] == backend
            and not r.get("timed_out")
            and r.get("power", {}).get("mean_total_w", 0) > 0
            and r.get("decode_tokens_per_sec", 0) > 0
        ]
        if not backend_runs:
            continue

        ctx_lengths = sorted(set(r["context_length"] for r in backend_runs))
        means = []
        stds = []
        valid = []

        for cl in ctx_lengths:
            vals = [
                r["power"]["mean_total_w"] / r["decode_tokens_per_sec"]
                for r in backend_runs
                if r["context_length"] == cl
                and r["power"].get("mean_total_w", 0) > 0
                and r["decode_tokens_per_sec"] > 0
            ]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                valid.append(cl)

        if valid:
            wpt_series[backend] = _Series(valid, means, stds)

    if wpt_series:
        p = output_dir / "watts_per_token.png"
        _plot_metric(wpt_series, "Energy Efficiency", "Watts / token", p, hw_info)
        plots.append(p)

    # Plot 5: Power breakdown (stacked bar per backend)
    _plot_power_breakdown(results, backends, output_dir, hw_info)
    breakdown_path = output_dir / "power_breakdown.png"
    if breakdown_path.exists():
        plots.append(breakdown_path)

    return plots


def _plot_power_breakdown(
    results: list[dict],
    backends: list[str],
    output_dir: Path,
    hardware_info: str,
) -> None:
    """Stacked bar chart of CPU/GPU/ANE power per backend."""
    import matplotlib.pyplot as plt

    has_power = any(
        r.get("power", {}).get("mean_total_w", 0) > 0
        for r in results if not r.get("timed_out")
    )
    if not has_power:
        return

    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 6), sharey=True)
    if len(backends) == 1:
        axes = [axes]

    for ax, backend in zip(axes, backends):
        backend_runs = [
            r for r in results
            if r["backend"] == backend and not r.get("timed_out") and r.get("power")
        ]
        if not backend_runs:
            continue

        ctx_lengths = sorted(set(r["context_length"] for r in backend_runs))
        cpu_means = []
        gpu_means = []
        ane_means = []

        for cl in ctx_lengths:
            cl_runs = [r for r in backend_runs if r["context_length"] == cl]
            cpu_means.append(np.mean([r["power"].get("mean_cpu_w", 0) for r in cl_runs]))
            gpu_means.append(np.mean([r["power"].get("mean_gpu_w", 0) for r in cl_runs]))
            ane_means.append(np.mean([r["power"].get("mean_ane_w", 0) for r in cl_runs]))

        x = np.arange(len(ctx_lengths))
        w = 0.6

        ax.bar(x, cpu_means, w, label="CPU", color="#2196F3", alpha=0.8)
        ax.bar(x, gpu_means, w, bottom=cpu_means, label="GPU", color="#FF9800", alpha=0.8)
        bottoms = [c + g for c, g in zip(cpu_means, gpu_means)]
        ax.bar(x, ane_means, w, bottom=bottoms, label="ANE", color="#4CAF50", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{cl:,}" for cl in ctx_lengths], rotation=45, ha="right")
        ax.set_xlabel("Context Length")
        ax.set_title(BACKEND_LABELS.get(backend, backend))
        ax.legend()

    axes[0].set_ylabel("Power (Watts)")
    fig.suptitle(f"Power Breakdown by Component\n{hardware_info}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "power_breakdown.png", dpi=150)
    plt.close(fig)
