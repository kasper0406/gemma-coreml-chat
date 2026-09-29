"""Test the powermetrics plist parser used by the benchmark harness.

Root cause being tested
-----------------------
``powermetrics -f plist`` streams one XML plist per sample, separated by a
NUL byte — the raw stream reads ``…</plist>\\n\\x00<?xml…``.  Three bugs in
the original parser meant the recorded power numbers were mostly fiction:

1. Splitting the stream on ``b"<?xml"`` left the separator NUL dangling on
   the end of every chunk but the last.  ``plistlib.loads`` rejects those,
   and the ``except: continue`` swallowed it — a 3-sample stream parsed as 1.

2. GPU power was read from the top-level ``gpu`` dict, which only carries
   dvfm/frequency/energy entries.  The reading lives at
   ``processor.gpu_power``, so every recorded ``mean_gpu_w`` was 0.0.

3. The reader thread reset its buffer to ``b""`` whenever *any* sample
   parsed, throwing away the partial trailing plist left by a 4 KiB read
   that landed mid-document.  Samples are larger than 4 KiB, so this lost
   roughly half of what survived (1).

The fixtures below reproduce the real stream layout: NUL-separated plists,
each bigger than one 4 KiB read, with the power fields under ``processor``.
"""

import datetime
import io
import plistlib

import pytest

from benchmarks.power import PowerMonitor, PowerSample, PowerTrace, _parse_power_plist, anchor


# ── fixtures ──────────────────────────────────────────────────────────────

READ_SIZE = 4096  # must match PowerMonitor._reader's read size


def _sample_plist(cpu_mw: float, gpu_mw: float, ane_mw: float,
                  stamp: datetime.datetime = datetime.datetime(2026, 9, 29, 7, 0, 0)) -> dict:
    """One powermetrics sample, shaped like the real thing.

    Power lives under ``processor``.  The top-level ``gpu`` dict is present
    but carries no ``gpu_power`` key — exactly the trap the old parser fell
    into.  Padded past ``READ_SIZE`` so a single read can never hold a whole
    sample, as with real output.
    """
    return {
        "is_delta": True,
        "elapsed_ns": 200_000_000,
        "timestamp": stamp,
        "hw_model": "Mac16,10",
        "gpu": {
            "dvfm_states": [{"freq": 444, "used_ratio": 0.5}],
            "freq_hz": 444.0,
            "gpu_energy": 15021,
        },
        "processor": {
            "cpu_power": cpu_mw,
            "gpu_power": gpu_mw,
            "ane_power": ane_mw,
            "package_power": cpu_mw + gpu_mw + ane_mw,
            "combined_power": cpu_mw + gpu_mw + ane_mw,
            "clusters": [
                {"name": f"E-Cluster{i}", "freq_hz": 1000.0 + i, "idle_ratio": 0.9}
                for i in range(40)
            ],
        },
    }


def _stream(samples: list[dict]) -> bytes:
    """Concatenate plists the way powermetrics does: NUL-separated."""
    return b"\x00".join(plistlib.dumps(s) for s in samples)


POWERS = [(1000.0, 75.1, 500.0), (2000.0, 150.2, 250.0), (3000.0, 225.3, 125.0)]
SAMPLES = [_sample_plist(*p) for p in POWERS]


# ── tests ─────────────────────────────────────────────────────────────────

def test_fixture_matches_real_stream_layout():
    """Guard the fixture itself: NUL separators and >4 KiB samples."""
    data = _stream(SAMPLES)
    assert b"</plist>\n\x00<?xml" in data, "fixture must use NUL separators"
    assert len(plistlib.dumps(SAMPLES[0])) > READ_SIZE, \
        "a sample must exceed one read for the buffering test to mean anything"
    print("  ✓ test_fixture_matches_real_stream_layout passed")


def test_parses_every_nul_separated_sample():
    """All three samples parse — the NUL must not kill samples 1..n-1."""
    samples, remainder = _parse_power_plist(_stream(SAMPLES))

    assert len(samples) == len(SAMPLES), \
        f"expected {len(SAMPLES)} samples, got {len(samples)}"
    assert remainder == b"", f"unexpected remainder: {remainder[:40]!r}"
    assert [s.cpu_mw for s in samples] == [p[0] for p in POWERS]
    print("  ✓ test_parses_every_nul_separated_sample passed")


def test_gpu_power_read_from_processor():
    """GPU power comes from processor.gpu_power, not the top-level gpu dict."""
    samples, _ = _parse_power_plist(_stream(SAMPLES))

    assert [s.gpu_mw for s in samples] == [p[1] for p in POWERS]
    assert all(s.gpu_mw > 0 for s in samples), "gpu power must not be 0"
    assert [s.ane_mw for s in samples] == [p[2] for p in POWERS]
    print("  ✓ test_gpu_power_read_from_processor passed")


def test_timestamp_parsed_as_utc_seconds():
    """plistlib hands back naive UTC datetimes; the sample keeps epoch seconds."""
    samples, _ = _parse_power_plist(_stream(SAMPLES))
    expected = datetime.datetime(2026, 9, 29, 7, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert all(s.timestamp == expected for s in samples)
    assert all(s.elapsed_ns == 200_000_000 for s in samples)


def _tiled(t0: float, elapsed_ns: list[int], mw: list[float]) -> list[PowerSample]:
    """Samples as powermetrics stamps them: end time floored to the second."""
    out, t = [], t0
    for e, p in zip(elapsed_ns, mw):
        t += e / 1e9
        out.append(PowerSample(timestamp=float(int(t)), elapsed_ns=e, cpu_mw=p))
    return out


def test_anchor_recovers_start_from_whole_second_stamps():
    """Intersecting every stamp's constraint pins the start to milliseconds."""
    t0 = 1_000.4567
    elapsed = [117_000_000 + 1_000_000 * (i % 3) for i in range(200)]
    t, lo, hi = anchor(_tiled(t0, elapsed, [1.0] * 200))
    assert lo <= t0 < hi
    assert hi - lo < 0.01
    assert abs(t - t0) < 0.005


def test_energy_is_power_times_overlap():
    """A window straddling samples takes each one's share of its interval."""
    # 10 samples of 100 ms from t0 = 50.0 s, power 1000·(k+1) mW.
    trace = PowerTrace(_tiled(50.0, [100_000_000] * 10, [1000.0 * (k + 1) for k in range(10)]))
    trace.samples[-1].timestamp = 51.0  # the last end lands exactly on 51.0
    # 50.15 .. 50.35: half of sample 1, all of sample 2, half of sample 3.
    t, _, _ = anchor(trace.samples)
    e = trace.energy_mj(t + 0.15, t + 0.35)
    assert e["cpu"] == pytest.approx(2000 * 0.05 + 3000 * 0.1 + 4000 * 0.05)
    assert e["gpu"] == 0.0


def test_energy_rejects_uncovered_window():
    trace = PowerTrace(_tiled(50.0, [100_000_000] * 10, [1000.0] * 10))
    t, _, _ = anchor(trace.samples)
    with pytest.raises(ValueError):
        trace.energy_mj(t + 0.5, t + 1.5)


def test_incomplete_tail_is_returned_not_dropped():
    """A document cut mid-stream comes back as the remainder, intact."""
    data = _stream(SAMPLES)
    cut = data.rindex(b"</plist>") - 100  # chop the last sample in half
    samples, remainder = _parse_power_plist(data[:cut])

    assert len(samples) == len(SAMPLES) - 1
    # Feeding the remainder plus the rest recovers the missing sample.
    rest, tail = _parse_power_plist(remainder + data[cut:])
    assert len(rest) == 1
    assert rest[0].gpu_mw == POWERS[-1][1]
    assert tail == b""
    print("  ✓ test_incomplete_tail_is_returned_not_dropped passed")


def test_chunked_reads_lose_nothing():
    """Replay the stream in 4 KiB reads: every sample must survive."""
    data = _stream(SAMPLES)
    buf = b""
    collected = []
    for off in range(0, len(data), READ_SIZE):
        buf += data[off:off + READ_SIZE]
        parsed, buf = _parse_power_plist(buf)
        collected.extend(parsed)

    assert len(collected) == len(SAMPLES), \
        f"chunked reads dropped samples: {len(collected)}/{len(SAMPLES)}"
    assert [s.gpu_mw for s in collected] == [p[1] for p in POWERS]
    print("  ✓ test_chunked_reads_lose_nothing passed")


def test_reader_thread_collects_all_samples():
    """PowerMonitor._reader over a fake stdout — no powermetrics needed."""
    n = 50
    powers = [(100.0 * i, 10.0 * i, i) for i in range(1, n + 1)]
    data = _stream([_sample_plist(*p) for p in powers])

    pm = PowerMonitor()
    pm._reader(io.BytesIO(data))

    assert len(pm.trace.samples) == n, \
        f"reader dropped samples: {len(pm.trace.samples)}/{n}"
    assert [s.gpu_mw for s in pm.trace.samples] == [p[1] for p in powers]
    print("  ✓ test_reader_thread_collects_all_samples passed")


def test_junk_and_partial_documents_are_skipped():
    """Non-plist noise around the stream is ignored, not fatal."""
    data = b"powermetrics: unable to get SMC data\n" + _stream(SAMPLES)
    samples, remainder = _parse_power_plist(data)
    assert len(samples) == len(SAMPLES)
    assert remainder == b""

    # No complete document yet → nothing parsed, everything buffered.
    head = _stream(SAMPLES)[:200]
    samples, remainder = _parse_power_plist(head)
    assert samples == []
    assert remainder == head
    print("  ✓ test_junk_and_partial_documents_are_skipped passed")


def test_stop_without_start_is_a_noop():
    """stop() on a monitor whose powermetrics never launched must not raise."""
    pm = PowerMonitor()
    pm.stop()
    assert pm.trace.samples == []
    print("  ✓ test_stop_without_start_is_a_noop passed")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_fixture_matches_real_stream_layout()
    test_parses_every_nul_separated_sample()
    test_gpu_power_read_from_processor()
    test_timestamp_parsed_as_utc_seconds()
    test_anchor_recovers_start_from_whole_second_stamps()
    test_energy_is_power_times_overlap()
    test_energy_rejects_uncovered_window()
    test_incomplete_tail_is_returned_not_dropped()
    test_chunked_reads_lose_nothing()
    test_reader_thread_collects_all_samples()
    test_junk_and_partial_documents_are_skipped()
    test_stop_without_start_is_a_noop()

    print("\nAll tests passed ✓")
