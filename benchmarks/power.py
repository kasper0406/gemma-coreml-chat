"""CPU / GPU / ANE rail power from ``powermetrics``, integrated over exact windows.

``powermetrics -i 100 -f plist --samplers cpu_power,gpu_power,ane_power``
reports, every ~100 ms, the mean power of each rail over the interval since the
previous sample (``elapsed_ns`` long).  Energy over a phase is therefore
``sum(P_k * overlap(sample_k, phase))``: every sample contributes in proportion
to how much of its interval lies inside the phase window, rather than a mean
power times a duration.

Sample windows.  Samples tile time: sample ``k`` covers
``[E_{k-1}, E_k)`` with ``E_k = T0 + sum(elapsed_ns[:k+1])``.  The one unknown
is ``T0``, and each sample's ``timestamp`` pins it: the stamp is the sample's
end in wall-clock time, truncated to the second (measured: it always equals
``floor`` of the moment the sample is emitted), so ``T0`` lies in
``[ts_k - S_k, ts_k + 1 - S_k)`` for every ``k``.  Intersecting those intervals
over a run of tens of seconds anchors ``T0`` to a few milliseconds
(:func:`anchor`); the width is recorded with every run.  An empty
intersection (stamps that contradict each other: a dropped or malformed
sample, a clock step) raises, and so does a malformed document in the stream:
the tiling is only as good as its every sample.

These are the CPU, GPU and ANE rails only — not DRAM, display or the rest of
the machine — and should be labelled as such.
"""

from __future__ import annotations

import datetime as _dt
import plistlib
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

RAILS = ("cpu", "gpu", "ane")
SAMPLE_MS = 100


@dataclass
class PowerSample:
    """One powermetrics sample: mean rail power (mW) over ``elapsed_ns``."""

    timestamp: float        # wall clock (s), whole seconds: floor of the sample's end
    elapsed_ns: int
    cpu_mw: float = 0.0
    gpu_mw: float = 0.0
    ane_mw: float = 0.0
    arrival: float = 0.0    # wall clock when the harness read it (diagnostic)

    def mw(self, rail: str) -> float:
        return getattr(self, f"{rail}_mw")


@dataclass
class PowerTrace:
    """All samples of one monitoring session, plus their reconstructed windows."""

    samples: list[PowerSample] = field(default_factory=list)
    malformed: int = 0      # plist documents that did not parse into a sample

    def windows(self) -> tuple[list[tuple[float, float]], dict]:
        """Wall-clock ``(start, end)`` of every sample, and the anchor record.

        Raises ``ValueError`` if a document in the stream was malformed or the
        stamps admit no common start (:func:`anchor`)."""
        if self.malformed:
            raise ValueError(f"{self.malformed} malformed powermetrics document(s)")
        t0, lo, hi = anchor(self.samples)
        out, t = [], t0
        for s in self.samples:
            end = t + s.elapsed_ns / 1e9
            out.append((t, end))
            t = end
        late = [s.arrival - w[1] for s, w in zip(self.samples, out) if s.arrival]
        info = {
            "t0": t0,
            "anchor_width_s": hi - lo,
            # How long after its reconstructed end each sample reached us.
            "arrival_lag_s": [min(late), max(late)] if late else None,
        }
        return out, info

    def energy_mj(self, start: float, end: float) -> dict[str, float]:
        """Energy (mJ) per rail inside the wall-clock window ``[start, end)``."""
        windows, _ = self.windows()
        e = dict.fromkeys(RAILS, 0.0)
        covered = 0.0
        for s, (a, b) in zip(self.samples, windows):
            overlap = min(b, end) - max(a, start)
            if overlap <= 0:
                continue
            covered += overlap
            for rail in RAILS:
                e[rail] += s.mw(rail) * overlap
        if covered < (end - start) * 0.999:
            raise ValueError(
                f"power samples cover {covered:.3f} s of a {end - start:.3f} s window"
            )
        return e

    def to_dict(self) -> dict:
        return {"malformed": self.malformed, "samples": [asdict(s) for s in self.samples]}


def anchor(samples: list[PowerSample]) -> tuple[float, float, float]:
    """``(T0, lo, hi)``: the wall-clock start of the first sample, and the
    interval every sample's whole-second stamp confines it to.  Raises
    ``ValueError`` when that interval is empty — the stamps contradict the
    tiling, and no start is right."""
    if not samples:
        raise ValueError("no power samples")
    lo, hi, s_k = -float("inf"), float("inf"), 0.0
    for s in samples:
        s_k += s.elapsed_ns / 1e9
        lo = max(lo, s.timestamp - s_k)
        hi = min(hi, s.timestamp + 1.0 - s_k)
    if hi <= lo:
        raise ValueError(f"sample stamps disagree by {lo - hi:.3f} s: no common start")
    return (lo + hi) / 2, lo, hi


_PLIST_START = b"<?xml"
_PLIST_END = b"</plist>"


def _parse_power_plist(
    data: bytes, arrival: float = 0.0,
) -> tuple[list[PowerSample], bytes, int]:
    """Parse a ``powermetrics -f plist`` byte stream into PowerSamples.

    ``powermetrics`` emits one XML plist per sample, separated by a NUL byte
    (the stream reads ``…</plist>\\n\\x00<?xml…``), so NULs are stripped before
    the documents are handed to :mod:`plistlib`.

    Documents are cut on the ``</plist>`` boundary and the trailing bytes that
    do not yet form a complete document are returned alongside the samples.  A
    streaming caller must keep that remainder and prepend it to its next read —
    a single sample is larger than a 4 KiB read, so dropping the remainder
    loses roughly every second sample.

    The third value counts documents that did not make a sample — unparseable,
    or without a timestamp or a positive ``elapsed_ns``.  Text that is not a
    document at all (a warning line) is not counted.
    """
    samples: list[PowerSample] = []
    malformed = 0
    pos = 0
    while (end := data.find(_PLIST_END, pos)) >= 0:
        end += len(_PLIST_END)
        doc = data[pos:end].replace(b"\x00", b"")
        pos = end
        start = doc.find(_PLIST_START)
        if start < 0:
            continue
        try:
            d = plistlib.loads(doc[start:])
        except Exception:
            malformed += 1
            continue

        # Every power field lives under "processor"; the top-level "gpu" dict
        # only carries dvfm/frequency/energy entries.
        proc = d.get("processor", {})
        stamp = d.get("timestamp")
        if isinstance(stamp, _dt.datetime):
            # plistlib returns naive UTC datetimes.
            stamp = stamp.replace(tzinfo=_dt.timezone.utc).timestamp()
        elapsed = d.get("elapsed_ns")
        if not isinstance(stamp, (int, float)) or not isinstance(elapsed, int) or elapsed <= 0:
            malformed += 1
            continue
        samples.append(PowerSample(
            timestamp=float(stamp),
            elapsed_ns=elapsed,
            cpu_mw=float(proc.get("cpu_power", 0.0) or 0.0),
            gpu_mw=float(proc.get("gpu_power", 0.0) or 0.0),
            ane_mw=float(proc.get("ane_power", 0.0) or 0.0),
            arrival=arrival,
        ))
    # Drop the separator (newline + NUL) so a caller's buffer is left empty
    # when the stream ends on a document boundary.
    return samples, data[pos:].lstrip(b"\x00 \t\r\n"), malformed


class PowerMonitor:
    """Runs ``sudo powermetrics`` in the background for one measured run."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.trace = PowerTrace()

    def start(self) -> None:
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "-i", str(SAMPLE_MS),
            "-f", "plist",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(
            target=self._reader, args=(self._proc.stdout,), daemon=True,
        )
        self._thread.start()

    def _reader(self, stdout) -> None:
        """Background thread: sole reader of stdout, until EOF.

        ``stdout`` is passed in rather than reached for through ``self._proc``
        so that ``stop()`` clearing that attribute cannot race this thread.
        """
        buf = b""
        # read1: return what is there rather than wait for a full buffer, so
        # the arrival stamp is taken when a sample lands.
        read = getattr(stdout, "read1", stdout.read)
        while chunk := read(4096):
            buf += chunk
            samples, buf, malformed = _parse_power_plist(buf, arrival=time.time())
            self.trace.samples.extend(samples)
            self.trace.malformed += malformed

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        proc.stdout.close()  # type: ignore[union-attr]
        self._proc = None


def check_power_available() -> bool:
    """Whether ``sudo powermetrics`` runs without a password prompt."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "-f", "plist",
             "--samplers", "cpu_power"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
