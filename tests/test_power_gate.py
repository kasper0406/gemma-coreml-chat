"""The benchmark harness's run acceptance: the background-CPU gate and
monitor, the power anchor bound and configuration ids.  No hardware: ``top``,
``powermetrics`` and the bench binary are faked."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import runner
from benchmarks.power import PowerSample, PowerTrace


# ── top parsing: fails closed ───────────────────────────────────────────────

HEADER = "Processes: 3 total\nCPU usage: 1% user\n\nPID    %CPU COMMAND\n"


def _top(first: str, second: str) -> str:
    return HEADER + first + "\n" + HEADER.replace("COMMAND", " COMMAND") + second + "\n"


def test_parse_top_reads_the_second_sample_of_every_process():
    out = _top("1 99.0 since-boot\n", "10 12.5 WindowServer\n11 0.0 idle thing\n12 3.0 Claude Helper (Renderer)")
    assert runner.parse_top(out) == [
        (10, 12.5, "WindowServer"), (11, 0.0, "idle thing"), (12, 3.0, "Claude Helper (Renderer)"),
    ]


@pytest.mark.parametrize("out", [
    "",                                              # nothing at all
    HEADER + "1 5.0 only-one-sample\n",               # a single (since-boot) sample
    _top("1 1.0 a", ""),                             # an empty second sample
    _top("1 1.0 a", "10 12.5 fine\nnot a process line"),  # garbage
])
def test_parse_top_rejects_anything_unexpected(out):
    with pytest.raises(ValueError):
        runner.parse_top(out)


@pytest.fixture(autouse=True)
def _own_tree(monkeypatch):
    """Faked ``top`` below; keep ``ps`` out of it (the harness's own tree is
    just this process)."""
    import os
    monkeypatch.setattr(runner, "_process_tree", lambda root: {os.getpid()})


class _FakePopen:
    def __init__(self, returncode, out):
        self.returncode, self._out, self.pid = returncode, out, 424242

    def communicate(self, timeout=None):
        return self._out, "top: failure"


def test_background_cpu_fails_closed_on_top_exit_status(monkeypatch):
    good = _top("1 1.0 a", "10 12.5 fine")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _FakePopen(1, good))
    with pytest.raises(RuntimeError, match="exited 1"):
        runner.background_cpu()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _FakePopen(0, ""))
    with pytest.raises(RuntimeError, match="top"):
        runner.background_cpu()


def test_background_cpu_counts_every_process_outside_its_own_tree(monkeypatch):
    import os
    own = os.getpid()
    out = _top("1 1.0 a", f"{own} 300.0 python\n424242 5.0 top\n10 12.5 WindowServer\n11 4.0 suggestd")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _FakePopen(0, out))
    sample = runner.background_cpu()
    assert sample["total"] == pytest.approx(16.5)   # suggestd counts: nothing is exempt
    assert [b["command"] for b in sample["busy"]] == ["WindowServer", "suggestd"]


def test_top_is_asked_for_every_process(monkeypatch):
    seen = {}

    def popen(cmd, **kw):
        seen["cmd"] = cmd
        return _FakePopen(0, _top("1 1.0 a", "10 1.0 b"))

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    runner.background_cpu()
    assert "-n" not in seen["cmd"]


# ── the rule: an idle calibration, and limits from its mean ────────────────

def test_the_limits_are_the_calibration_mean_plus_fixed_excesses():
    lim = runner.idle_limits([8.0] * 30 + [12.0] * 30)
    assert lim["mean"] == 10.0 and lim["max"] == 12.0
    assert lim["drift"] == 10.0 + runner.DRIFT_EXCESS
    assert lim["spike"] == 10.0 + runner.SPIKE_EXCESS


@pytest.mark.parametrize("totals", [
    [2.0] * 54 + [90.0] * 6,                      # mean 10.8, but a sample above C + S
    [runner.IDLE_CEILING + 1.0] * 60,             # steady, but not idle
])
def test_a_calibration_that_is_not_idle_is_refused(totals):
    with pytest.raises(ValueError, match="not idle"):
        runner.idle_limits(totals)


LIMITS = {"spike": 60.0, "drift": 55.0}


def _s(end, total, busy=()):
    return {"end": end, "total": total, "busy": [{"command": b, "cpu": total} for b in busy]}


def test_judge_cpu_catches_spikes_and_drift():
    assert runner.judge_cpu([_s(1, 45), _s(2, 48)], LIMITS) == []
    spike = runner.judge_cpu([_s(1, 45), _s(2, 45), _s(3, 70, ["mds_stores"])], LIMITS)
    assert len(spike) == 1 and "spike" in spike[0] and "mds_stores" in spike[0]
    drift = runner.judge_cpu([_s(1, 56), _s(2, 57)], LIMITS)
    assert len(drift) == 1 and "averaged" in drift[0]
    assert runner.judge_cpu([], LIMITS) == ["no background CPU samples"]


SPAN = {"idle_pre": (0.0, 4.0), "prefill": (4.0, 9.0), "decode": (9.0, 36.0),
        "idle_post": (36.0, 40.0)}


def test_a_run_needs_every_window_quiet_no_spike_and_full_coverage():
    quiet = [_s(float(k), 40) for k in range(1, 41)]
    assert runner.judge_run(quiet, SPAN, LIMITS) == []
    # A spike before the measured span does not count; one inside it does.
    assert runner.judge_run([_s(-2.0, 90)] + quiet, SPAN, LIMITS) == []
    spiky = quiet[:20] + [_s(21.0, 90, ["mds_stores"])] + quiet[21:]
    reasons = runner.judge_run(spiky, SPAN, LIMITS)
    assert len(reasons) == 1 and "spike" in reasons[0] and "mds_stores" in reasons[0]
    # A monitor that died half-way leaves the rest unsampled.
    reasons = runner.judge_run(quiet[:20], SPAN, LIMITS)
    assert any("unsampled" in r for r in reasons) and any(r.startswith("idle_post:") for r in reasons)


def test_a_rise_in_the_prefill_window_alone_drops_the_run():
    # 40 s at 40% whose 5 s prefill sits at 59% (drift limit 55): the whole
    # run averages 42.4%, but the prefill window has drifted.
    samples = [_s(float(k), 59 if 4 < k <= 9 else 40) for k in range(1, 41)]
    reasons = runner.judge_run(samples, SPAN, LIMITS)
    assert len(reasons) == 1 and reasons[0].startswith("prefill:") and "averaged 59.0%" in reasons[0]


# ── one run, end to end with fakes ──────────────────────────────────────────

def _tiled(t0, elapsed_ns, n):
    out, t = [], t0
    for _ in range(n):
        t += elapsed_ns / 1e9
        out.append(PowerSample(timestamp=float(int(t)), elapsed_ns=elapsed_ns, cpu_mw=1000.0))
    return out


def _bench(t0):
    """Windows in wall-clock ns (the uptime offset is faked to 0)."""
    edges = {"idle_pre": (1, 5), "prefill": (5, 7), "decode": (8, 13), "idle_post": (14, 18)}
    out = {w: {"start_ns": int((t0 + a) * 1e9), "end_ns": int((t0 + b) * 1e9)}
           for w, (a, b) in edges.items()}
    out["prefill"]["tokens"] = 4096
    out["decode"]["tokens"] = 256
    return out


STATE = {"power": "AC Power", "low_power_mode": False, "display_on": False, "screen_locked": True}


def _fake_run(monkeypatch, tmp_path, samples, background, t0=1000.0, config=None, states=None,
              monitor_error=None):
    class FakePower:
        def __init__(self):
            self.trace = PowerTrace(list(samples))

        def start(self): pass

        def stop(self): pass

    class FakeMonitor:
        def __init__(self):
            self.samples, self.error = background, monitor_error
            self.states = [STATE] * 12 if states is None else states

        def start(self): pass

        def stop(self): pass

    monkeypatch.setattr(runner, "PowerMonitor", FakePower)
    monkeypatch.setattr(runner, "BackgroundMonitor", FakeMonitor)
    monkeypatch.setattr(runner, "_uptime_to_wall_offset", lambda: 0.0)
    monkeypatch.setattr(runner, "quiet_gate", lambda cal: {"passed": True, "reasons": []})
    monkeypatch.setattr(runner, "_invoke_bench", lambda *a: _bench(t0))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    config = config or runner.BenchmarkConfig(models=["/a/m.mlpackage"], compute_units=["cpu-and-ne"],
                                              context_lengths=[512])
    c = config.configurations()[0]
    return runner.run_one(Path("/bin/true"), config, c, 0, tmp_path,
                          {"limits": LIMITS, "state": STATE})


def _quiet(t0=1000.0):
    return [_s(t0 + 0.5 + 1.5 * k, 45) for k in range(15)]


def test_a_clean_run_is_kept_with_its_anchor_width(monkeypatch, tmp_path):
    # ~117 ms samples: the stamps pin the start to milliseconds.
    samples = _tiled(999.7, 117_000_000, 200)
    rec = _fake_run(monkeypatch, tmp_path, samples, _quiet())
    assert rec.kept, rec.error
    assert 0 < rec.power["anchor_width_s"] <= runner.ANCHOR_MAX_FRACTION * rec.power["shortest_window_s"]


def test_contradictory_stamps_drop_the_run(monkeypatch, tmp_path):
    samples = _tiled(999.7, 117_000_000, 200)
    samples[100].timestamp -= 1.0
    rec = _fake_run(monkeypatch, tmp_path, samples, _quiet())
    assert not rec.kept and "disagree" in rec.error


def test_a_wide_anchor_drops_the_run(monkeypatch, tmp_path):
    # Whole-second samples cannot place the start within a second.
    samples = _tiled(999.5, 1_000_000_000, 20)
    rec = _fake_run(monkeypatch, tmp_path, samples, _quiet())
    assert not rec.kept and "anchor width" in rec.error


def test_background_activity_during_the_run_drops_it(monkeypatch, tmp_path):
    samples = _tiled(999.7, 117_000_000, 200)
    busy = _quiet()
    busy[6] = _s(busy[6]["end"], 95, ["mds_stores"])   # inside the decode window
    rec = _fake_run(monkeypatch, tmp_path, samples, busy)
    assert not rec.kept and "spike" in rec.error and "mds_stores" in rec.error


@pytest.mark.parametrize("change", [
    {"power": "Battery Power"}, {"low_power_mode": True}, {"display_on": True},
    {"screen_locked": False},
])
def test_a_machine_state_change_during_the_run_drops_it(monkeypatch, tmp_path, change):
    states = [STATE] * 5 + [STATE | change] + [STATE] * 5
    rec = _fake_run(monkeypatch, tmp_path, _tiled(999.7, 117_000_000, 200), _quiet(),
                    states=states)
    assert not rec.kept and "machine state" in rec.error and "1 of 11" in rec.error
    rec = _fake_run(monkeypatch, tmp_path, _tiled(999.7, 117_000_000, 200), _quiet(), states=[])
    assert not rec.kept and "no machine-state polls" in rec.error


def test_the_monitor_polls_the_state_with_every_sample_and_after_the_run(monkeypatch):
    import time
    polls = iter(range(10**6))
    monkeypatch.setattr(runner, "machine_state", lambda: {"poll": next(polls)})

    def sample():
        time.sleep(0.005)
        return {"end": time.time(), "total": 1.0, "busy": []}

    monkeypatch.setattr(runner, "background_cpu", sample)
    m = runner.BackgroundMonitor()
    m.start()
    time.sleep(0.05)
    m.stop()
    assert m.samples and len(m.states) == len(m.samples) + 1


# ── machine state: fails closed ─────────────────────────────────────────────

GOOD = {
    ("pmset", "-g", "batt"): "Now drawing from 'AC Power'\n",
    ("pmset", "-g"): " displaysleep 10\n powermode            0\n",
    ("ioreg", "-r", "-d1", "-w0", "-c", "IOMobileFramebufferShim"): '"CurrentPowerState"=0\n',
    ("ioreg", "-n", "Root", "-d1", "-w0"): '"IOConsoleUsers" = ({"CGSSessionScreenIsLocked"=Yes})',
}


def _fake_output(outputs):
    def output(*cmd):
        out = outputs[cmd]
        if isinstance(out, Exception):
            raise out
        return out
    return output


def test_machine_state_reads_every_value(monkeypatch):
    monkeypatch.setattr(runner, "_output", _fake_output(GOOD))
    assert runner.machine_state() == STATE
    lpm = GOOD | {("pmset", "-g"): " lowpowermode 1\n"}
    monkeypatch.setattr(runner, "_output", _fake_output(lpm))
    assert runner.machine_state()["low_power_mode"] is True


@pytest.mark.parametrize("cmd, out", [
    (("pmset", "-g", "batt"), RuntimeError("pmset -g batt exited 1")),
    (("pmset", "-g"), " displaysleep 10\n"),              # no Low Power Mode line
    (("ioreg", "-r", "-d1", "-w0", "-c", "IOMobileFramebufferShim"), ""),
    (("ioreg", "-n", "Root", "-d1", "-w0"), "no session"),
])
def test_an_unreadable_machine_state_raises(monkeypatch, cmd, out):
    monkeypatch.setattr(runner, "_output", _fake_output(GOOD | {cmd: out}))
    with pytest.raises(RuntimeError):
        runner.machine_state()


def test_a_failing_or_hanging_command_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="exited 1"):
        runner._output("false")
    monkeypatch.setattr(runner, "COMMAND_TIMEOUT_S", 0.2)
    with pytest.raises(RuntimeError, match="timed out"):
        runner._output("sleep", "5")


def test_an_unreadable_state_refuses_the_calibration_fails_the_gate_and_stops_the_monitor(monkeypatch):
    def unreadable():
        raise RuntimeError("cannot read the Low Power Mode")

    monkeypatch.setattr(runner, "GATE_TIMEOUT_S", -1)
    monkeypatch.setattr(runner, "background_cpu", lambda: _s(1.0, 12.0))
    monkeypatch.setattr(runner, "machine_state", unreadable)
    with pytest.raises(SystemExit, match="Low Power Mode"):
        runner.calibrate()
    gate = runner.quiet_gate({"limits": LIMITS, "state": STATE})
    assert not gate["passed"] and "Low Power Mode" in gate["reasons"][0]
    m = runner.BackgroundMonitor()
    m.start()
    m.stop()
    assert m.error and "Low Power Mode" in m.error


def test_a_failed_monitor_drops_the_run(monkeypatch, tmp_path):
    rec = _fake_run(monkeypatch, tmp_path, _tiled(999.7, 117_000_000, 200), _quiet(),
                    monitor_error="cannot read the display power state")
    assert not rec.kept and "monitor failed" in rec.error


def test_calibration_refuses_a_state_change_or_a_busy_baseline(monkeypatch):
    monkeypatch.setattr(runner, "GATE_TIMEOUT_S", -1)
    totals = iter([2.0] * 54 + [90.0] * 6)
    monkeypatch.setattr(runner, "background_cpu", lambda: _s(1.0, next(totals)))
    monkeypatch.setattr(runner, "machine_state", lambda: STATE)
    with pytest.raises(SystemExit, match="not idle"):
        runner.calibrate()
    polls = iter([STATE] * 30 + [STATE | {"display_on": True}] * 31)
    monkeypatch.setattr(runner, "background_cpu", lambda: _s(1.0, 12.0))
    monkeypatch.setattr(runner, "machine_state", lambda: next(polls))
    with pytest.raises(SystemExit, match="machine state"):
        runner.calibrate()
    monkeypatch.setattr(runner, "machine_state", lambda: STATE)
    cal = runner.calibrate()
    assert cal["state"] == STATE and cal["limits"]["mean"] == 12.0


def test_the_gate_checks_the_machine_state(monkeypatch):
    monkeypatch.setattr(runner, "GATE_TIMEOUT_S", -1)
    monkeypatch.setattr(runner, "background_cpu", lambda: _s(1.0, 45))
    monkeypatch.setattr(runner, "machine_state", lambda: STATE)
    cal = {"limits": LIMITS, "state": STATE}
    assert runner.quiet_gate(cal)["passed"]
    monkeypatch.setattr(runner, "machine_state", lambda: STATE | {"power": "Battery Power"})
    gate = runner.quiet_gate(cal)
    assert not gate["passed"] and "Battery Power" in gate["reasons"][0]


def test_same_named_packages_get_distinct_ids_and_sample_files(monkeypatch, tmp_path):
    config = runner.BenchmarkConfig(
        models=["/baseline/gemma4-e2b.mlpackage", "/experiment/gemma4-e2b.mlpackage"],
        compute_units=["cpu-and-ne"], context_lengths=[512])
    configs = config.configurations()
    ids = [c.id for c in configs]
    assert len(set(ids)) == 2
    files = set()
    for c in configs:
        cfg = runner.BenchmarkConfig(models=[c.model], compute_units=["cpu-and-ne"],
                                     context_lengths=[512])
        monkeypatch.setattr(runner.BenchmarkConfig, "configurations", lambda self, c=c: [c])
        rec = _fake_run(monkeypatch, tmp_path, _tiled(999.7, 117_000_000, 200), _quiet(), config=cfg)
        assert rec.config_id == c.id and rec.model == c.model
        files.add(rec.samples_file)
    assert len(files) == 2 and all(Path(f).exists() for f in files)
    rows = runner.summarize([
        runner.RunRecord(config_id=c.id, model=c.model, compute_units=c.compute_units,
                         context_length=c.context_length, repetition=0) for c in configs])
    assert [r["config_id"] for r in rows] == ids
    assert json.loads(Path(sorted(files)[0]).read_text())["samples"]


def test_the_cli_refuses_a_missing_package_before_measuring(monkeypatch, tmp_path):
    from benchmarks import cli
    monkeypatch.setattr(cli, "check_power_available", lambda: pytest.fail("got past the check"))
    monkeypatch.setattr("sys.argv", ["gemma-bench", "--models", str(tmp_path / "none.mlpackage")])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2


def _record(config_id, kept, rep=0):
    r = runner.RunRecord(config_id=config_id, model="/m.mlpackage", compute_units="cpu-and-ne",
                         context_length=512, repetition=rep, kept=kept,
                         error=None if kept else "background: prefill: drifted")
    if kept:
        r.idle_w = {"total": 1.0}
        stat = {"ms_per_token": 1.0, "mj_per_token": 1.0, "mj_per_token_above_idle": 0.5,
                "w": {k: 1.0 for k in (*runner.RAILS, "total")}}
        r.phases = {p: stat for p in runner.PHASES}
    return r


@pytest.mark.parametrize("kept, fails", [
    ([False, False, False], True),     # every run dropped
    ([True, False, False], True),      # 1 of 3
    ([True, True, False], False),      # 2 of 3: a majority
])
def test_the_cli_fails_unless_every_configuration_keeps_a_majority(monkeypatch, tmp_path, kept, fails):
    from benchmarks import cli
    (tmp_path / "m.mlpackage").mkdir()
    records = [_record("c00", k, rep) for rep, k in enumerate(kept)] + \
              [_record("c01", True, rep) for rep in range(3)]
    monkeypatch.setattr(cli, "check_power_available", lambda: True)
    monkeypatch.setattr(cli, "run_benchmark", lambda config, out: (out.mkdir(), records)[1])
    monkeypatch.setattr("sys.argv", ["gemma-bench", "--models", str(tmp_path / "m.mlpackage"),
                                     "--output-dir", str(tmp_path / "out")])
    if fails:
        with pytest.raises(SystemExit, match="too few kept runs.*c00") as e:
            cli.main()
        assert e.value.code not in (0, None)
    else:
        cli.main()
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "Dropped runs" in summary and "c00 rep 2: background: prefill: drifted" in summary


def test_a_failed_priming_run_ends_the_invocation(monkeypatch, tmp_path):
    def fail(*a):
        raise RuntimeError("exit 1: no such function")

    monkeypatch.setattr(runner, "ensure_swift_bench_built", lambda: Path("/bin/false"))
    monkeypatch.setattr(runner, "machine_state", lambda: STATE)
    monkeypatch.setattr(runner, "_invoke_bench", fail)
    monkeypatch.setattr(runner, "calibrate", lambda: pytest.fail("measured after a failed priming"))
    config = runner.BenchmarkConfig(models=["/a/m.mlpackage"], compute_units=["cpu-and-ne"],
                                    context_lengths=[512])
    with pytest.raises(SystemExit, match="priming .* failed: exit 1"):
        runner.run_benchmark(config, tmp_path)
