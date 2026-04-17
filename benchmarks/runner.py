"""Benchmark runner that drives the GemmaBench Swift binary.

The Python layer only orchestrates: loops over the (backend × context × run)
matrix, spawns the Swift binary for each measurement, and optionally records
a `powermetrics` trace alongside each run.  All CoreML work happens inside
the Swift binary, which reuses the `GemmaCore` inference path and benefits
from `.mlmodelc` compilation caching on disk — something Python's coremltools
does not do reliably.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.power import PowerMonitor, PowerTrace


# Path to the Swift bench package under the repo.
_SWIFT_BENCH_PKG = Path(__file__).resolve().parent / "swift" / "bench"


# ---------------------------------------------------------------------------
# Result / config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """One measurement from the Swift bench binary, plus optional power."""

    context_length: int
    backend: str
    run_index: int
    prefill_time_s: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_time_s: float = 0.0
    decode_tokens_per_sec: float = 0.0
    decode_tokens_generated: int = 0
    load_time_s: float = 0.0
    total_time_s: float = 0.0
    power: dict = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None


@dataclass
class BenchmarkConfig:
    model_path: str
    backends: list[str]
    context_lengths: list[int]
    num_runs: int = 5
    decode_tokens: int = 32
    timeout_s: int = 300
    enable_power: bool = True
    no_warmup: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Map the Python-side backend name to what the Swift binary expects.
_BACKEND_TO_SWIFT = {
    "cpu": "cpu",
    "gpu": "cpu-gpu",
    "ane": "cpu-ane",
    "all": "all",
}


# ---------------------------------------------------------------------------
# Swift binary build / invocation
# ---------------------------------------------------------------------------


_BENCH_BUILD_LOCK_MARKER = _SWIFT_BENCH_PKG / ".build" / "bench-built.stamp"


def ensure_swift_bench_built(verbose: bool = True) -> Path:
    """Build the Swift bench binary (once) and return the executable path.

    Requires Xcode (not Command Line Tools) — the package targets
    macOS 15, which the CommandLineTools SDK doesn't provide.
    """
    exe = _SWIFT_BENCH_PKG / ".build" / "release" / "GemmaBench"
    if exe.exists():
        return exe
    if verbose:
        print("Building GemmaBench (swift build -c release) …", flush=True)
    r = subprocess.run(
        ["swift", "build", "-c", "release"],
        cwd=str(_SWIFT_BENCH_PKG),
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        msg = (
            "swift build failed — make sure Xcode is installed and selected:\n"
            "   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
        raise RuntimeError(msg)
    if not exe.exists():
        raise RuntimeError(f"GemmaBench built but not at {exe}")
    _BENCH_BUILD_LOCK_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _BENCH_BUILD_LOCK_MARKER.write_text(time.strftime("%FT%T%z"))
    return exe


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _invoke_bench(
    exe: Path,
    model_path: str,
    backend: str,
    context_length: int,
    decode_tokens: int,
    run_index: int,
    timeout_s: int,
    no_warmup: bool,
) -> dict:
    """Run the Swift bench binary, return its parsed JSON output (1 line)."""
    cu = _BACKEND_TO_SWIFT.get(backend, backend)
    cmd = [
        str(exe),
        "--model", model_path,
        "--compute-units", cu,
        "--context-length", str(context_length),
        "--decode-tokens", str(decode_tokens),
        "--run-index", str(run_index),
    ]
    if no_warmup:
        cmd.append("--no-warmup")
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"__timed_out": True}
    if r.returncode != 0:
        return {"__error": f"exit {r.returncode}: {r.stderr.strip()[:400]}"}
    # Binary emits exactly one JSON line on stdout.
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if not line:
        return {"__error": f"no JSON on stdout. stderr: {r.stderr[:400]}"}
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        return {"__error": f"bad JSON: {e}. line: {line[:400]}"}


def run_single(
    exe: Path,
    config: BenchmarkConfig,
    backend: str,
    context_length: int,
    run_index: int,
) -> RunResult:
    result = RunResult(
        context_length=context_length,
        backend=backend,
        run_index=run_index,
    )

    pm: PowerMonitor | None = None
    if config.enable_power:
        pm = PowerMonitor(sample_ms=200)
        pm.start()

    try:
        out = _invoke_bench(
            exe,
            model_path=config.model_path,
            backend=backend,
            context_length=context_length,
            decode_tokens=config.decode_tokens,
            run_index=run_index,
            timeout_s=config.timeout_s,
            no_warmup=config.no_warmup,
        )
    finally:
        if pm is not None:
            pm.stop()
            if pm.trace.samples:
                result.power = pm.trace.to_dict()

    if out.get("__timed_out"):
        result.timed_out = True
        result.error = f"timeout after {config.timeout_s}s"
        return result
    if "__error" in out:
        result.error = out["__error"]
        return result

    result.prefill_time_s = float(out.get("prefill_time_s", 0.0))
    result.decode_time_s = float(out.get("decode_time_s", 0.0))
    result.decode_tokens_generated = int(out.get("decode_tokens_generated", 0))
    result.prefill_tokens_per_sec = float(out.get("prefill_tokens_per_sec", 0.0))
    result.decode_tokens_per_sec = float(out.get("decode_tokens_per_sec", 0.0))
    result.load_time_s = float(out.get("load_time_s", 0.0))
    result.total_time_s = float(out.get("total_time_s", 0.0))
    return result


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


def run_benchmark(config: BenchmarkConfig) -> list[RunResult]:
    exe = ensure_swift_bench_built()
    results: list[RunResult] = []
    n_cfg = len(config.backends) * len(config.context_lengths) * config.num_runs
    print(
        f"Running {n_cfg} benchmark runs "
        f"({len(config.backends)} backends × {len(config.context_lengths)} "
        f"context lengths × {config.num_runs} runs)",
        flush=True,
    )
    i = 0
    for backend in config.backends:
        for context_length in config.context_lengths:
            for run_index in range(config.num_runs):
                i += 1
                label = (
                    f"[{i}/{n_cfg}] backend={backend} ctx={context_length} "
                    f"run={run_index}"
                )
                print(f"  {label} …", flush=True)
                r = run_single(exe, config, backend, context_length, run_index)
                if r.timed_out or r.error:
                    note = r.error or "timed out"
                    print(f"    ✗ {note}", flush=True)
                else:
                    print(
                        f"    ✓ prefill {r.prefill_tokens_per_sec:.1f} tok/s, "
                        f"decode {r.decode_tokens_per_sec:.1f} tok/s",
                        flush=True,
                    )
                results.append(r)
    return results


def save_results(results: list[RunResult], path: Path, config: BenchmarkConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "timestamp": time.strftime("%FT%T%z"),
        "hardware": {
            "node": platform.node(),
            "machine": platform.machine(),
            "mac_ver": platform.mac_ver()[0],
        },
        "config": config.to_dict(),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(out, indent=2))
    print(f"  saved → {path}", flush=True)
