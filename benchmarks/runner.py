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
* **Quiet machine** — once, after priming: an idle-baseline calibration
  (:func:`calibrate`) of the CPU use of every process but the harness (this
  process, its descendants and its parent), in a machine state — AC power,
  no Low Power Mode, display/session (:func:`machine_state`) — that must
  hold throughout it.  Before every run the gate (:func:`quiet_gate`) wants
  that state and quiet samples, and pauses and retries otherwise (giving
  the run up after :data:`GATE_TIMEOUT_S`).  During every run a monitor
  polls the state and samples the CPU from launch to exit; a run is not
  kept if a poll failed or differed from the calibrated state, or if its
  measured span saw a spike or an unsampled stretch, or one of its windows
  drifted (:func:`judge_run`; load and warm-up are exempt from the CPU
  check only).  No other process is exempt; busy ones are logged.
* **Anchor** — a run whose power samples have contradictory stamps, a
  malformed document, or an anchor wider than :data:`ANCHOR_MAX_FRACTION`
  of its shortest window is not kept.
* **Order** — one unmeasured priming run per configuration (compiles and
  caches it; a failed one ends the invocation), then ``runs`` repetitions of every configuration, the
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
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.power import RAILS, PowerMonitor

# Path to the Swift bench package under the repo.
_SWIFT_BENCH_PKG = Path(__file__).resolve().parent / "swift" / "bench"

# Background CPU — every process but the harness, in points of one core (100
# = one core busy), from one-second ``top`` samples — is judged by one rule.
# An idle calibration of CALIBRATION_SAMPLES samples is used only if its mean
# is at most IDLE_CEILING and its busiest sample at most IDLE_CEILING +
# SPIKE_EXCESS; otherwise it is retaken.  Against its mean m, a window whose
# samples average above m + DRIFT_EXCESS has drifted, and a sample above
# m + SPIKE_EXCESS is a spike.  DRIFT_EXCESS is the sensitivity: sustained
# interference of 5 points (a twentieth of a core) or more is caught.
# SPIKE_EXCESS tolerates brief housekeeping — a one-second burst of up to 30
# points — but not a burst of a third of a core.  IDLE_CEILING: an idle Mac
# with its display asleep uses well under a quarter of a core; one that uses
# more is not idle.
CALIBRATION_SAMPLES = 60
IDLE_CEILING = 25.0
DRIFT_EXCESS = 5.0
SPIKE_EXCESS = 30.0
# Longest any one machine-state or process-list command may take.
COMMAND_TIMEOUT_S = 30
# Consecutive samples the gate wants inside the limits before a run starts.
GATE_SAMPLES = 3
GATE_RETRY_S = 15
GATE_TIMEOUT_S = 900
# While a run is measured, no stretch of it may go unsampled for longer than
# this.  A sample covers one second, and ``top`` takes ~1.7 s to return one
# (its first, since-boot pass over every process is the rest), longer on a
# loaded machine; so about 60% of the span is observed, and nothing can hide
# in the unobserved part for longer than this.
MAX_SAMPLE_GAP_S = 2.0
# The power anchor's uncertainty is its width; the midpoint estimate is off by
# at most half of it, which moves each window boundary by that much.  Keeping
# the width under 1% of the shortest measured window bounds the energy error
# from the anchor at 0.5% of any window (a boundary shift δ changes a window's
# energy by at most δ·P_max, against P·duration).
ANCHOR_MAX_FRACTION = 0.01

COMPUTE_UNITS = ("cpu-and-gpu", "cpu-and-ne", "all", "cpu-only")
PHASES = ("prefill", "decode")
WINDOWS = ("idle_pre", *PHASES, "idle_post")


@dataclass
class Configuration:
    """One measured configuration.  ``id`` is unique within an invocation (its
    index comes first), so two packages with the same file name never share
    raw-sample files or summary rows."""

    id: str
    model: str
    compute_units: str
    context_length: int


@dataclass
class BenchmarkConfig:
    models: list[str]
    compute_units: list[str]
    context_lengths: list[int]
    runs: int = 3
    timeout_s: int = 1800

    def configurations(self) -> list[Configuration]:
        combos = [(m, cu, n) for m in self.models for cu in self.compute_units
                  for n in self.context_lengths]
        return [Configuration(f"c{i:02d}-{Path(m).stem}-{cu}-{n}", m, cu, n)
                for i, (m, cu, n) in enumerate(combos)]


@dataclass
class RunRecord:
    config_id: str
    model: str
    compute_units: str
    context_length: int
    repetition: int
    started: str = ""
    kept: bool = False
    error: str | None = None
    gate: dict = field(default_factory=dict)
    background: dict = field(default_factory=dict)  # CPU samples over the run
    bench: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)      # anchor diagnostics
    idle_w: dict = field(default_factory=dict)     # per rail, and total
    phases: dict = field(default_factory=dict)     # phase -> metrics
    samples_file: str = ""


# ── Machine state ───────────────────────────────────────────────────────────


def _output(*cmd: str) -> str:
    """``cmd``'s stdout.  Fails closed: a non-zero exit or a run longer than
    :data:`COMMAND_TIMEOUT_S` raises ``RuntimeError``."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{cmd[0]} timed out after {COMMAND_TIMEOUT_S}s") from None
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}: {r.stderr.strip()[-200:]}")
    return r.stdout


def machine_state() -> dict:
    """Power source, Low Power Mode, display power and screen lock — what
    every run must share with the calibration (battery and Low Power Mode
    change clocks; a lit display keeps WindowServer drawing).  Fails closed:
    a command that fails or times out, or output without the value, raises
    ``RuntimeError`` — an unreadable state is never a reading."""
    def need(value, what: str):
        if not value:
            raise RuntimeError(f"cannot read the {what}")
        return value

    source = need(re.search(r"drawing from '([^']+)'", _output("pmset", "-g", "batt")),
                  "power source")
    # `powermode` (0 automatic, 1 low, 2 high) on Macs that have power modes.
    lpm = need(re.search(r"\b(?:low)?powermode\s+(\d)", _output("pmset", "-g")),
               "Low Power Mode")
    fb = _output("ioreg", "-r", "-d1", "-w0", "-c", "IOMobileFramebufferShim")
    displays = need(re.findall(r'"CurrentPowerState"=(\d+)', fb), "display power state")
    users = _output("ioreg", "-n", "Root", "-d1", "-w0")
    need("IOConsoleUsers" in users, "console session")
    return {
        "power": source.group(1),
        "low_power_mode": lpm.group(1) == "1",
        "display_on": any(int(v) for v in displays),
        "screen_locked": "CGSSessionScreenIsLocked" in users,
    }


def judge_state(states: list[dict], expected: dict) -> list[str]:
    """Reasons the machine did not stay in ``expected`` (empty if every poll
    in ``states`` matches it)."""
    if not states:
        return ["no machine-state polls"]
    changed = [s for s in states if s != expected]
    if not changed:
        return []
    return [f"machine state {changed[0]} ≠ {expected} in {len(changed)} of {len(states)} poll(s)"]


def parse_top(out: str) -> list[tuple[int, float, str]]:
    """``(pid, cpu, command)`` of every process in the last sample of
    ``top -l 2 -stats pid,cpu,command`` output (the first sample is a
    since-boot average).  Raises ``ValueError`` on anything unexpected."""
    lines = out.splitlines()
    headers = [i for i, ln in enumerate(lines) if ln.split()[:2] == ["PID", "%CPU"]]
    if len(headers) < 2:
        raise ValueError(f"top output has {len(headers)} of 2 samples")
    rows = []
    for ln in lines[headers[-1] + 1:]:
        if not ln.strip():
            continue
        parts = ln.split(None, 2)
        try:
            rows.append((int(parts[0]), float(parts[1]), parts[2].strip() if len(parts) > 2 else ""))
        except (ValueError, IndexError):
            raise ValueError(f"unparseable top line {ln!r}") from None
    if not rows:
        raise ValueError("top listed no processes")
    return rows


def _process_tree(root: int) -> set[int]:
    """``root`` and every descendant of it."""
    children: dict[int, list[int]] = {}
    for ln in _output("ps", "-A", "-o", "pid=,ppid=").splitlines():
        pid, ppid = map(int, ln.split())
        children.setdefault(ppid, []).append(pid)
    tree, todo = set(), [root]
    while todo:
        pid = todo.pop()
        tree.add(pid)
        todo += children.get(pid, [])
    return tree


def background_cpu() -> dict:
    """One one-second sample of every process except the harness: this
    process, its descendants (the bench binary, ``powermetrics``, ``top``
    itself) and its parent process (not the parent's other children).
    Fails closed: a ``top`` or ``ps`` failure or timeout, or unparseable
    output, raises ``RuntimeError``."""
    own = _process_tree(os.getpid()) | {os.getppid()}
    proc = subprocess.Popen(
        ["top", "-l", "2", "-s", "1", "-o", "cpu", "-stats", "pid,cpu,command"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        out, err = proc.communicate(timeout=COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"top timed out after {COMMAND_TIMEOUT_S}s") from None
    end = time.time()
    if proc.returncode != 0:
        raise RuntimeError(f"top exited {proc.returncode}: {err.strip()[-200:]}")
    try:
        rows = parse_top(out)
    except ValueError as e:
        raise RuntimeError(f"top: {e}") from None
    own |= _process_tree(os.getpid()) | {proc.pid}
    counted = [{"pid": pid, "cpu": cpu, "command": cmd}
               for pid, cpu, cmd in rows if pid not in own and cpu > 0.0]
    return {
        "end": end,
        "total": round(sum(c["cpu"] for c in counted), 2),
        # Logged, never exempted: who was busy.
        "busy": sorted((c for c in counted if c["cpu"] >= 1.0), key=lambda c: -c["cpu"])[:8],
    }


def idle_limits(totals: list[float]) -> dict:
    """The calibration's mean and busiest sample, and the limits its mean
    sets.  Raises ``ValueError`` for a baseline that is not idle."""
    mean, top = statistics.fmean(totals), max(totals)
    if mean > IDLE_CEILING or top > IDLE_CEILING + SPIKE_EXCESS:
        raise ValueError(f"not idle: background CPU mean {mean:.1f}% (at most {IDLE_CEILING:.0f}), "
                         f"busiest sample {top:.1f}% (at most {IDLE_CEILING + SPIKE_EXCESS:.0f})")
    return {"n": len(totals), "mean": mean, "max": top,
            "drift": mean + DRIFT_EXCESS, "spike": mean + SPIKE_EXCESS}


def _drift(samples: list[dict], limits: dict) -> list[str]:
    mean = statistics.fmean(s["total"] for s in samples)
    if mean > limits["drift"]:
        return [f"background CPU averaged {mean:.1f}% > {limits['drift']:.1f}%"]
    return []


def _spikes(samples: list[dict], limits: dict) -> list[str]:
    spikes = [s for s in samples if s["total"] > limits["spike"]]
    if not spikes:
        return []
    worst = max(spikes, key=lambda s: s["total"])
    who = ", ".join(f"{b['command']} {b['cpu']:.0f}%" for b in worst["busy"][:4])
    return [f"{len(spikes)} background CPU spike(s) up to {worst['total']:.0f}% "
            f"> {limits['spike']:.0f}% ({who})"]


def judge_cpu(samples: list[dict], limits: dict) -> list[str]:
    """Reasons ``samples`` are not quiet against ``limits`` (empty if they are)."""
    if not samples:
        return ["no background CPU samples"]
    return _spikes(samples, limits) + _drift(samples, limits)


def _overlapping(samples: list[dict], start: float, end: float) -> list[dict]:
    """The one-second samples that overlap ``[start, end)``, in time order."""
    return sorted((s for s in samples if s["end"] - 1.0 < end and s["end"] > start),
                  key=lambda s: s["end"])


def judge_run(samples: list[dict], span: dict[str, tuple[float, float]],
              limits: dict) -> list[str]:
    """Reasons a run's background was not quiet: over its measured span
    (``idle_pre`` start to ``idle_post`` end) a spike, or a stretch longer
    than :data:`MAX_SAMPLE_GAP_S` that no sample covers; in any one measured
    window, drift.

    Load and warm-up, before ``idle_pre``, are not judged here (their machine
    state is): nothing is measured there, and the CoreML/ANE daemons that
    load and compile for the bench (``aned``, ``ANECompilerService``, …) are
    outside the harness, so their CPU there is the workload's, not
    background.  Whatever the load leaves running shows up in ``idle_pre``."""
    start, end = span["idle_pre"][0], span["idle_post"][1]
    inside = _overlapping(samples, start, end)
    covered, gap = start, 0.0
    for s in inside:
        gap = max(gap, s["end"] - 1.0 - covered)
        covered = max(covered, s["end"])
    gap = max(gap, end - covered)
    reasons = _spikes(inside, limits)
    if gap > MAX_SAMPLE_GAP_S:
        reasons.append(f"background CPU unsampled for {gap:.1f} s of the measured span")
    for w in WINDOWS:
        in_w = _overlapping(samples, *span[w])
        reasons += [f"{w}: {r}" for r in (_drift(in_w, limits) if in_w
                                          else ["no background CPU samples"])]
    return reasons


def calibrate() -> dict:
    """Measure the idle background: the machine state it was taken in (AC
    power, no Low Power Mode, polled with every sample and unchanged
    throughout) and :data:`CALIBRATION_SAMPLES` one-second samples.  A
    calibration that fails either, or that is not idle (:func:`idle_limits`),
    is retaken, for up to :data:`GATE_TIMEOUT_S`."""
    deadline = time.monotonic() + GATE_TIMEOUT_S
    while True:
        print(f"Calibrating the idle background ({CALIBRATION_SAMPLES} × 1 s) …", flush=True)
        try:
            state = machine_state()
            if state["power"] != "AC Power" or state["low_power_mode"]:
                raise ValueError(f"{state}: needs AC power and Low Power Mode off")
            samples, states = [], []
            for _ in range(CALIBRATION_SAMPLES):
                samples.append(background_cpu())
                states.append(machine_state())
            if changed := judge_state(states, state):
                raise ValueError(changed[0])
            totals = [s["total"] for s in samples]
            limits = idle_limits(totals)
            break
        except (ValueError, RuntimeError) as e:
            if time.monotonic() > deadline:
                raise SystemExit(f"no usable idle calibration: {e}; leave the machine alone")
            print(f"  calibration refused: {e} — retrying in {GATE_RETRY_S}s", flush=True)
            time.sleep(GATE_RETRY_S)
    load: dict[str, float] = {}
    for s in samples:
        for b in s["busy"]:
            load[b["command"]] = load.get(b["command"], 0.0) + b["cpu"] / len(samples)
    print(f"  background CPU mean {limits['mean']:.1f}% (busiest {limits['max']:.1f}%) → "
          f"drift > {limits['drift']:.1f}%, spike > {limits['spike']:.1f}%", flush=True)
    return {
        "state": state, "limits": limits, "totals": totals,
        "mean_busy": dict(sorted(load.items(), key=lambda kv: -kv[1])[:10]),
    }


def quiet_gate(calibration: dict) -> dict:
    """Wait until the machine is as it was calibrated: the same machine state
    (AC power, no Low Power Mode, display/session) and :data:`GATE_SAMPLES`
    background samples within the limits.  The record says what was seen."""
    deadline = time.monotonic() + GATE_TIMEOUT_S
    attempts = 0
    while True:
        attempts += 1
        try:
            state = machine_state()
            reasons = judge_state([state], calibration["state"])
            samples = [background_cpu() for _ in range(GATE_SAMPLES)]
            reasons += judge_cpu(samples, calibration["limits"])
        except RuntimeError as e:
            state, samples, reasons = None, [], [str(e)]
        record = {
            "passed": not reasons, "attempts": attempts, "state": state,
            "samples": samples, "reasons": reasons,
        }
        if not reasons or time.monotonic() > deadline:
            return record
        print(f"    gate: {'; '.join(reasons)} — retrying in {GATE_RETRY_S}s", flush=True)
        time.sleep(GATE_RETRY_S)


class BackgroundMonitor:
    """For the length of a run — launch, load and warm-up included — polls
    :func:`machine_state` and takes a :func:`background_cpu` sample, back to
    back on a thread (a poll every ~2 s), with one last poll after the bench
    exits.  A failed poll or sample ends the monitoring and is reported as
    :attr:`error`, which drops the run: the monitor fails closed."""

    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.states: list[dict] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.states.append(machine_state())
                self.samples.append(background_cpu())
            self.states.append(machine_state())
        except RuntimeError as e:
            self.error = str(e)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=4 * COMMAND_TIMEOUT_S)
        if self._thread.is_alive():
            self.error = "the monitor did not finish its last poll"


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


def run_one(exe: Path, config: BenchmarkConfig, c: Configuration, repetition: int,
            samples_dir: Path, calibration: dict) -> RunRecord:
    rec = RunRecord(config_id=c.id, model=c.model, compute_units=c.compute_units,
                    context_length=c.context_length, repetition=repetition,
                    started=time.strftime("%FT%T%z"))
    rec.gate = quiet_gate(calibration)
    if not rec.gate["passed"]:
        rec.error = "quiet-machine gate: " + "; ".join(rec.gate["reasons"])
        return rec

    offset = _uptime_to_wall_offset()
    pm = PowerMonitor()
    monitor = BackgroundMonitor()
    monitor.start()
    pm.start()
    try:
        rec.bench = _invoke_bench(exe, c.model, c.compute_units, c.context_length,
                                  config.timeout_s)
    except RuntimeError as e:
        rec.error = str(e)
    finally:
        time.sleep(0.3)  # let the sample covering the last window land
        pm.stop()
        monitor.stop()
    rec.background = {"samples": monitor.samples, "states": monitor.states,
                      "error": monitor.error}

    rec.samples_file = str(samples_dir / f"{c.id}-r{repetition}.json")
    samples_dir.mkdir(parents=True, exist_ok=True)
    Path(rec.samples_file).write_text(json.dumps(
        {"uptime_to_wall_offset_s": offset, **pm.trace.to_dict()}))
    if rec.error:
        return rec

    try:
        b = rec.bench
        span = {w: (b[w]["start_ns"] / 1e9 + offset, b[w]["end_ns"] / 1e9 + offset)
                for w in WINDOWS}
        # Machine state over the whole process life; background CPU over the
        # measured windows only (see judge_run for why not load/warm-up).
        reasons = judge_state(monitor.states, calibration["state"])
        reasons += judge_run(monitor.samples, span, calibration["limits"])
        if monitor.error:
            reasons.append(f"background monitor failed: {monitor.error}")
        if reasons:
            raise ValueError("background: " + "; ".join(reasons))

        _, rec.power = pm.trace.windows()
        shortest = min(end - start for start, end in span.values())
        rec.power["shortest_window_s"] = shortest
        if rec.power["anchor_width_s"] > ANCHOR_MAX_FRACTION * shortest:
            raise ValueError(f"anchor width {rec.power['anchor_width_s'] * 1000:.1f} ms exceeds "
                             f"{ANCHOR_MAX_FRACTION:.0%} of the shortest window "
                             f"({shortest:.2f} s)")
        idle = {}
        for w in ("idle_pre", "idle_post"):
            start, end = span[w]
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
    """Prime every configuration, calibrate the idle background, then run the
    rotated repetitions, rewriting ``out_dir/results.json`` after every run."""
    exe = ensure_swift_bench_built()
    configs = config.configurations()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    print(f"Results → {results_path}", flush=True)

    state = machine_state()
    if state["power"] != "AC Power" or state["low_power_mode"]:
        raise SystemExit(f"refusing to measure: {state} — "
                         "connect AC power and turn Low Power Mode off")

    for c in configs:
        print(f"  priming {c.id} ({c.model}) …", flush=True)
        try:
            _invoke_bench(exe, c.model, c.compute_units, c.context_length, config.timeout_s)
        except RuntimeError as e:
            raise SystemExit(f"priming {c.id} failed: {e}")

    calibration = calibrate()
    records: list[RunRecord] = []
    total = len(configs) * config.runs
    for rep in range(config.runs):
        k = rep % len(configs)
        for c in configs[k:] + configs[:k]:
            print(f"  [{len(records) + 1}/{total}] rep {rep} {c.id} …", flush=True)
            r = run_one(exe, config, c, rep, out_dir / "power", calibration)
            if r.kept:
                pf, dc = r.phases["prefill"], r.phases["decode"]
                print(f"    ✓ prefill {pf['ms_per_token']:.3f} ms/tok {pf['mj_per_token']:.2f} mJ/tok"
                      f" | decode {dc['ms_per_token']:.2f} ms/tok {dc['mj_per_token']:.1f} mJ/tok"
                      f" | idle {r.idle_w['total']:.2f} W", flush=True)
            else:
                print(f"    ✗ {r.error}", flush=True)
            records.append(r)
            save_results(records, results_path, config, calibration)
    return records


def summarize(records: list[RunRecord]) -> list[dict]:
    """Per configuration: medians of the kept runs, with min–max spread."""
    groups: dict[str, list[RunRecord]] = {}
    for r in records:
        groups.setdefault(r.config_id, []).append(r)
    rows = []
    for config_id, runs in groups.items():
        kept = [r for r in runs if r.kept]
        first = runs[0]
        row = {"config_id": config_id, "model": first.model,
               "compute_units": first.compute_units, "context_length": first.context_length,
               "runs": len(runs), "kept": len(kept),
               "dropped": [f"rep {r.repetition}: {r.error}" for r in runs if not r.kept]}
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
    """Markdown tables: median [min–max] of the kept runs, which package each
    configuration id ran, and why each dropped run was dropped."""
    def cell(s, fmt):
        return f"{s['median']:{fmt}} [{s['min']:{fmt}}–{s['max']:{fmt}}]"

    out = ["CPU+GPU+ANE rail energy (powermetrics), not whole-system energy. "
           "Median [min–max] of kept runs."]
    for phase in PHASES:
        out += ["", f"### {phase}", "",
                "| config | kept | ms/token | mJ/token | mJ/token above idle "
                "| CPU W | GPU W | ANE W | idle W |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if phase not in r:
                out.append(f"| {r['config_id']} | 0/{r['runs']} | — | — | — | — | — | — | — |")
                continue
            p = r[phase]
            out.append(
                f"| {r['config_id']} | {r['kept']}/{r['runs']} | {cell(p['ms_per_token'], '.3f')} "
                f"| {cell(p['mj_per_token'], '.2f')} | {cell(p['mj_per_token_above_idle'], '.2f')} "
                f"| {p['w']['cpu']['median']:.2f} | {p['w']['gpu']['median']:.2f} "
                f"| {p['w']['ane']['median']:.2f} | {r['idle_w']['median']:.2f} |")
    out += ["", "| config | package | units | ctx |", "|---|---|---|---|"]
    out += [f"| {r['config_id']} | {r['model']} | {r['compute_units']} | {r['context_length']} |"
            for r in rows]
    dropped = [f"- {r['config_id']} {d}" for r in rows for d in r["dropped"]]
    if dropped:
        out += ["", "Dropped runs:", "", *dropped]
    return "\n".join(out)


def save_results(records: list[RunRecord], path: Path, config: BenchmarkConfig,
                 calibration: dict) -> None:
    """Write the results JSON atomically (temp file in the same dir + rename)."""
    out = {
        "timestamp": time.strftime("%FT%T%z"),
        "energy_scope": "CPU+GPU+ANE rails (powermetrics), not whole-system energy",
        "hardware": {"node": platform.node(), "machine": platform.machine(),
                     "mac_ver": platform.mac_ver()[0]},
        "gate": {"calibration": calibration, "gate_samples": GATE_SAMPLES,
                 "max_sample_gap_s": MAX_SAMPLE_GAP_S,
                 "anchor_max_fraction": ANCHOR_MAX_FRACTION,
                 "retry_s": GATE_RETRY_S, "timeout_s": GATE_TIMEOUT_S},
        "config": asdict(config),
        "summary": summarize(records),
        "runs": [asdict(r) for r in records],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, path)
