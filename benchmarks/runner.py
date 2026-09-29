"""Measured runs of the GemmaBench Swift binary, with rail energy per phase.

Method (see benchmarks/README.md for the why):

* **Workload** — fixed in the Swift binary: prefill of 4096 tokens per window
  (whole-cache prompts: 8 × 512 or 2 × 2048), then greedy decode of 256 tokens
  after a 128-token prompt, with ~4 s of idle before and after.  The binary
  reports every phase boundary as a ``CLOCK_UPTIME_RAW`` timestamp.
* **Power** — ``powermetrics -i 100`` (CPU, GPU and ANE rails) runs for the
  whole process; each phase's energy is ``sum(P × overlap)`` of the samples
  with the exact phase window (:mod:`benchmarks.power`).  Idle power is the
  mean of the two idle windows; above-idle energy subtracts it over the
  phase's duration.
* **Quiet-machine gate** — before every run: on AC power, not in Low Power
  Mode, and the other processes' CPU below :data:`QUIET_CPU_PERCENT` of one
  core, with :data:`EXEMPT_PROCESSES` not counted (and logged); otherwise it
  pauses and retries, and gives the run up after :data:`GATE_TIMEOUT_S`.
* **Order** — one unmeasured priming run per configuration (compiles and
  caches it), then ``runs`` repetitions of every configuration, the
  configuration order rotated by one each repetition.
* **Summary** — medians of the kept runs (gate passed, no error), with
  min–max spread.

Each run's power samples are written next to the results, one JSON per run.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.power import RAILS, PowerMonitor

# Path to the Swift bench package under the repo.
_SWIFT_BENCH_PKG = Path(__file__).resolve().parent / "swift" / "bench"

# Other processes may use at most this much CPU, in percent of one core, summed.
QUIET_CPU_PERCENT = 15.0
# Background daemons that are not counted against the gate (but are logged).
EXEMPT_PROCESSES = ("suggestd",)
GATE_RETRY_S = 15
GATE_TIMEOUT_S = 900

COMPUTE_UNITS = ("cpu-and-gpu", "cpu-and-ne", "all", "cpu-only")
PHASES = ("prefill", "decode")


@dataclass
class BenchmarkConfig:
    models: list[str]
    compute_units: list[str]
    context_lengths: list[int]
    runs: int = 3
    timeout_s: int = 1800

    def configurations(self) -> list[tuple[str, str, int]]:
        return [(m, cu, n) for m in self.models for cu in self.compute_units
                for n in self.context_lengths]


@dataclass
class RunRecord:
    model: str
    compute_units: str
    context_length: int
    repetition: int
    started: str = ""
    kept: bool = False
    error: str | None = None
    gate: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)      # anchor diagnostics
    idle_w: dict = field(default_factory=dict)     # per rail, and total
    phases: dict = field(default_factory=dict)     # phase -> metrics
    samples_file: str = ""


# ── Machine state ───────────────────────────────────────────────────────────


def power_source() -> dict:
    """``{"source": "AC Power" | "Battery Power" | ..., "low_power_mode": bool}``."""
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"drawing from '([^']+)'", batt)
    settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    lpm = re.search(r"lowpowermode\s+(\d)", settings)
    return {
        "source": m.group(1) if m else "unknown",
        "low_power_mode": bool(lpm and lpm.group(1) == "1"),
        "battery": batt.strip().splitlines()[-1].strip() if batt.strip() else "",
    }


def other_processes_cpu(own_pids: set[int]) -> tuple[float, list[dict], list[dict]]:
    """CPU use of every other process over one second, in percent of a core.

    ``top``'s second sample is the one-second delta (its first is a since-boot
    average).  Returns the counted total, the counted busy processes and the
    exempt ones.
    """
    out = subprocess.run(
        ["top", "-l", "2", "-s", "1", "-n", "40", "-o", "cpu", "-stats", "pid,cpu,command"],
        capture_output=True, text=True,
    ).stdout
    second = out.split("PID")[-1].splitlines()[1:]
    counted, exempt, total = [], [], 0.0
    for line in second:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, cpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        name = parts[2].strip()
        if pid in own_pids or name.startswith("top") or cpu <= 0.0:
            continue
        entry = {"pid": pid, "cpu": cpu, "command": name}
        if any(name.startswith(e) for e in EXEMPT_PROCESSES):
            exempt.append(entry)
            continue
        counted.append(entry)
        total += cpu
    return total, [c for c in counted if c["cpu"] >= 1.0], exempt


def quiet_gate() -> dict:
    """Wait until the machine is fit to measure; the record says what was seen."""
    own = {os.getpid(), os.getppid()}
    deadline = time.monotonic() + GATE_TIMEOUT_S
    attempts = 0
    while True:
        attempts += 1
        ps = power_source()
        total, busy, exempt = other_processes_cpu(own)
        reasons = []
        if ps["source"] != "AC Power":
            reasons.append(f"on {ps['source']}")
        if ps["low_power_mode"]:
            reasons.append("Low Power Mode")
        if total >= QUIET_CPU_PERCENT:
            reasons.append(f"other processes at {total:.0f}% CPU")
        record = {
            "passed": not reasons, "attempts": attempts, "power_source": ps,
            "other_cpu_percent": round(total, 1), "busy": busy, "exempt": exempt,
            "reasons": reasons,
        }
        if not reasons or time.monotonic() > deadline:
            return record
        print(f"    gate: {', '.join(reasons)} — retrying in {GATE_RETRY_S}s "
              f"({', '.join(f'{b['command']} {b['cpu']:.0f}%' for b in busy[:4])})",
              flush=True)
        time.sleep(GATE_RETRY_S)


def _uptime_to_wall_offset() -> float:
    """``wall - uptime`` (seconds), from the tightest of several paired reads."""
    best = None
    for _ in range(20):
        a = time.time()
        u = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) / 1e9
        b = time.time()
        if best is None or b - a < best[0]:
            best = (b - a, (a + b) / 2 - u)
    return best[1]


# ── Swift binary ────────────────────────────────────────────────────────────


def ensure_swift_bench_built() -> Path:
    """Build the Swift bench binary (always: GemmaCore is the code under test,
    and an up-to-date ``swift build`` is a ~1 s no-op)."""
    exe = _SWIFT_BENCH_PKG / ".build" / "release" / "GemmaBench"
    print("Building GemmaBench (swift build -c release) …", flush=True)
    r = subprocess.run(["swift", "build", "-c", "release"], cwd=str(_SWIFT_BENCH_PKG),
                       capture_output=True, text=True)
    if r.returncode != 0 or not exe.exists():
        raise RuntimeError(
            "swift build failed — make sure Xcode is installed and selected:\n"
            "   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
    return exe


def _invoke_bench(exe: Path, model: str, cu: str, n: int, timeout_s: int) -> dict:
    cmd = [str(exe), "--model", model, "--compute-units", cu, "--context-length", str(n)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout after {timeout_s}s")
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[-400:]}")
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if not line:
        raise RuntimeError(f"no JSON on stdout; stderr: {r.stderr[-400:]}")
    return json.loads(line)


# ── One run ─────────────────────────────────────────────────────────────────


def _phase_metrics(trace, offset: float, start_ns: int, end_ns: int, tokens: int,
                   idle_w: dict) -> dict:
    start, end = start_ns / 1e9 + offset, end_ns / 1e9 + offset
    dur = end - start
    e = trace.energy_mj(start, end)
    total = sum(e.values())
    above = total - idle_w["total"] * 1000 * dur
    return {
        "tokens": tokens,
        "seconds": dur,
        "ms_per_token": dur * 1000 / tokens,
        "w": {r: e[r] / 1000 / dur for r in RAILS} | {"total": total / 1000 / dur},
        "mj_per_token": total / tokens,
        "mj_per_token_above_idle": above / tokens,
    }


def run_one(exe: Path, config: BenchmarkConfig, model: str, cu: str, n: int,
            repetition: int, samples_dir: Path) -> RunRecord:
    rec = RunRecord(model=model, compute_units=cu, context_length=n,
                    repetition=repetition, started=time.strftime("%FT%T%z"))
    rec.gate = quiet_gate()
    if not rec.gate["passed"]:
        rec.error = "quiet-machine gate: " + ", ".join(rec.gate["reasons"])
        return rec

    offset = _uptime_to_wall_offset()
    pm = PowerMonitor()
    pm.start()
    try:
        rec.bench = _invoke_bench(exe, model, cu, n, config.timeout_s)
    except RuntimeError as e:
        rec.error = str(e)
    finally:
        time.sleep(0.3)  # let the sample covering the last window land
        pm.stop()

    name = f"{Path(model).stem}-{cu}-{n}-r{repetition}.json"
    rec.samples_file = str(samples_dir / name)
    samples_dir.mkdir(parents=True, exist_ok=True)
    Path(rec.samples_file).write_text(json.dumps(
        {"uptime_to_wall_offset_s": offset, **pm.trace.to_dict()}))
    if rec.error:
        return rec

    try:
        _, rec.power = pm.trace.windows()
        b = rec.bench
        idle = {}
        for w in ("idle_pre", "idle_post"):
            start = b[w]["start_ns"] / 1e9 + offset
            end = b[w]["end_ns"] / 1e9 + offset
            e = pm.trace.energy_mj(start, end)
            idle[w] = {r: e[r] / 1000 / (end - start) for r in RAILS}
        rec.idle_w = {r: (idle["idle_pre"][r] + idle["idle_post"][r]) / 2 for r in RAILS}
        rec.idle_w["total"] = sum(rec.idle_w.values())
        rec.idle_w["pre_total"] = sum(idle["idle_pre"].values())
        rec.idle_w["post_total"] = sum(idle["idle_post"].values())
        for phase in PHASES:
            p = b[phase]
            rec.phases[phase] = _phase_metrics(
                pm.trace, offset, p["start_ns"], p["end_ns"], p["tokens"], rec.idle_w)
        rec.kept = True
    except (KeyError, ValueError) as e:
        rec.error = f"power analysis: {e}"
    return rec


# ── The matrix ──────────────────────────────────────────────────────────────


def run_benchmark(config: BenchmarkConfig, out_dir: Path) -> list[RunRecord]:
    """Prime every configuration, then run the rotated repetitions, rewriting
    ``out_dir/results.json`` after every run."""
    exe = ensure_swift_bench_built()
    configs = config.configurations()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    print(f"Results → {results_path}", flush=True)

    ps = power_source()
    if ps["source"] != "AC Power" or ps["low_power_mode"]:
        raise SystemExit(f"refusing to measure: {ps['source']}"
                         f"{', Low Power Mode' if ps['low_power_mode'] else ''} — "
                         "connect AC power and turn Low Power Mode off")

    for model, cu, n in configs:
        print(f"  priming {Path(model).name} {cu} {n} …", flush=True)
        try:
            _invoke_bench(exe, model, cu, n, config.timeout_s)
        except RuntimeError as e:
            print(f"    ✗ priming failed: {e}", flush=True)

    records: list[RunRecord] = []
    total = len(configs) * config.runs
    for rep in range(config.runs):
        k = rep % len(configs)
        for model, cu, n in configs[k:] + configs[:k]:
            print(f"  [{len(records) + 1}/{total}] rep {rep} {Path(model).name} {cu} {n} …",
                  flush=True)
            r = run_one(exe, config, model, cu, n, rep, out_dir / "power")
            if r.kept:
                pf, dc = r.phases["prefill"], r.phases["decode"]
                print(f"    ✓ prefill {pf['ms_per_token']:.3f} ms/tok {pf['mj_per_token']:.2f} mJ/tok"
                      f" | decode {dc['ms_per_token']:.2f} ms/tok {dc['mj_per_token']:.1f} mJ/tok"
                      f" | idle {r.idle_w['total']:.2f} W", flush=True)
            else:
                print(f"    ✗ {r.error}", flush=True)
            records.append(r)
            save_results(records, results_path, config)
    return records


def summarize(records: list[RunRecord]) -> list[dict]:
    """Per configuration: medians of the kept runs, with min–max spread."""
    groups: dict[tuple, list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.model, r.compute_units, r.context_length), []).append(r)
    rows = []
    for (model, cu, n), runs in groups.items():
        kept = [r for r in runs if r.kept]
        row = {"model": model, "compute_units": cu, "context_length": n,
               "runs": len(runs), "kept": len(kept),
               "dropped": [r.error for r in runs if not r.kept]}
        if kept:
            def stat(values):
                return {"median": statistics.median(values), "min": min(values), "max": max(values)}
            row["idle_w"] = stat([r.idle_w["total"] for r in kept])
            for phase in PHASES:
                ps = [r.phases[phase] for r in kept]
                row[phase] = {
                    "ms_per_token": stat([p["ms_per_token"] for p in ps]),
                    "mj_per_token": stat([p["mj_per_token"] for p in ps]),
                    "mj_per_token_above_idle": stat([p["mj_per_token_above_idle"] for p in ps]),
                    "w": {k: stat([p["w"][k] for p in ps]) for k in (*RAILS, "total")},
                }
        rows.append(row)
    return rows


def format_summary(rows: list[dict]) -> str:
    """Markdown tables: median [min–max] of the kept runs."""
    def cell(s, fmt):
        return f"{s['median']:{fmt}} [{s['min']:{fmt}}–{s['max']:{fmt}}]"

    out = ["CPU+GPU+ANE rail energy (powermetrics), not whole-system energy. "
           "Median [min–max] of kept runs."]
    for phase in PHASES:
        out += ["", f"### {phase}", "",
                "| model | units | ctx | kept | ms/token | mJ/token | mJ/token above idle "
                "| CPU W | GPU W | ANE W | idle W |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if phase not in r:
                out.append(f"| {Path(r['model']).name} | {r['compute_units']} | "
                           f"{r['context_length']} | 0/{r['runs']} | — | — | — | — | — | — | — |")
                continue
            p = r[phase]
            out.append(
                f"| {Path(r['model']).name} | {r['compute_units']} | {r['context_length']} "
                f"| {r['kept']}/{r['runs']} | {cell(p['ms_per_token'], '.3f')} "
                f"| {cell(p['mj_per_token'], '.2f')} | {cell(p['mj_per_token_above_idle'], '.2f')} "
                f"| {p['w']['cpu']['median']:.2f} | {p['w']['gpu']['median']:.2f} "
                f"| {p['w']['ane']['median']:.2f} | {r['idle_w']['median']:.2f} |")
    return "\n".join(out)


def save_results(records: list[RunRecord], path: Path, config: BenchmarkConfig) -> None:
    """Write the results JSON atomically (temp file in the same dir + rename)."""
    out = {
        "timestamp": time.strftime("%FT%T%z"),
        "energy_scope": "CPU+GPU+ANE rails (powermetrics), not whole-system energy",
        "hardware": {"node": platform.node(), "machine": platform.machine(),
                     "mac_ver": platform.mac_ver()[0]},
        "gate": {"quiet_cpu_percent": QUIET_CPU_PERCENT, "exempt": list(EXEMPT_PROCESSES),
                 "retry_s": GATE_RETRY_S, "timeout_s": GATE_TIMEOUT_S},
        "config": asdict(config),
        "summary": summarize(records),
        "runs": [asdict(r) for r in records],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, path)
