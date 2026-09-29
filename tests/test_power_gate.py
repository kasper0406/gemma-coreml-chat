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


# ── limits from the idle baseline ───────────────────────────────────────────

def test_limits_follow_the_baseline_spread():
    totals = [40.0, 42.0, 44.0, 46.0, 48.0, 200.0]   # one outlier barely moves them
    lim = runner.cpu_limits(totals)
    assert lim["median"] == 45.0
    assert lim["sigma"] == pytest.approx(1.4826 * 3.0)
    assert lim["spike"] == pytest.approx(45.0 + runner.SPIKE_SIGMAS * lim["sigma"])
    # m + 2σ = 53.9 is above the 90th percentile + margin: that caps it.
    assert lim["drift"] == pytest.approx(48.0 + runner.DRIFT_MARGIN)


def test_limits_have_a_floor_for_a_perfectly_steady_baseline():
    lim = runner.cpu_limits([30.0] * 60)
    assert lim["sigma"] == runner.SIGMA_FLOOR
    assert lim["drift"] == pytest.approx(30.0 + runner.DRIFT_SIGMAS)


def test_a_bimodal_baseline_is_refused():
    # The review's example: σ from the MAD would be ~30 points, so the limits
    # would be 148.6% / 89.3% and a sustained 80% would pass.
    with pytest.raises(ValueError, match="unsteady"):
        runner.cpu_limits([10.0] * 30 + [50.0] * 30)


def test_limits_never_exceed_the_sensitivity_caps():
    # A busy but steady baseline (median 60, spread 30 < half the median):
    # σ ≈ 11 would put the spike at ~105% and the drift at ~82%.
    totals = [45.0 + 30.0 * k / 59 for k in range(60)]
    lim = runner.cpu_limits(totals)
    med = lim["median"]
    assert lim["spike"] == pytest.approx(med + runner.SPIKE_MAX_EXCESS)
    assert lim["drift"] == pytest.approx(med + runner.DRIFT_MAX_EXCESS)
    assert lim["drift"] < lim["p90"] + runner.DRIFT_MARGIN


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


def test_judge_window_uses_only_samples_overlapping_it_and_requires_coverage():
    # Back-to-back samples every 1.5 s from t=0.5; a spike before the window.
    samples = [_s(0.5, 90)] + [_s(1.0 + 1.5 * k, 45) for k in range(1, 12)]
    assert runner.judge_window(samples, 2.0, 15.0, LIMITS) == []
    # The same window with a spike inside it.
    spiky = samples[:5] + [_s(8.5, 90)] + samples[6:]
    assert any("spike" in r for r in runner.judge_window(spiky, 2.0, 15.0, LIMITS))
    # A monitor that died half-way leaves the rest of the window unsampled.
    dead = samples[:6]
    assert any("unsampled" in r for r in runner.judge_window(dead, 2.0, 15.0, LIMITS))


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


def _fake_run(monkeypatch, tmp_path, samples, background, t0=1000.0, config=None):
    class FakePower:
        def __init__(self):
            self.trace = PowerTrace(list(samples))

        def start(self): pass

        def stop(self): pass

    class FakeMonitor:
        def __init__(self):
            self.samples, self.error = background, None

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
    return runner.run_one(Path("/bin/true"), config, c, 0, tmp_path, {"limits": LIMITS})


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
