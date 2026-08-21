"""Power metrics collection via macOS powermetrics."""

from __future__ import annotations

import plistlib
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class PowerSample:
    """A single power sample (milliwatts)."""

    timestamp: float  # time.monotonic()
    cpu_mw: float = 0.0
    gpu_mw: float = 0.0
    ane_mw: float = 0.0

    @property
    def total_mw(self) -> float:
        return self.cpu_mw + self.gpu_mw + self.ane_mw

    @property
    def total_w(self) -> float:
        return self.total_mw / 1000.0


@dataclass
class PowerTrace:
    """Aggregated power trace from a monitoring session."""

    samples: list[PowerSample] = field(default_factory=list)

    @property
    def mean_cpu_w(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.cpu_mw for s in self.samples) / len(self.samples) / 1000.0

    @property
    def mean_gpu_w(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.gpu_mw for s in self.samples) / len(self.samples) / 1000.0

    @property
    def mean_ane_w(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.ane_mw for s in self.samples) / len(self.samples) / 1000.0

    @property
    def mean_total_w(self) -> float:
        return self.mean_cpu_w + self.mean_gpu_w + self.mean_ane_w

    def to_dict(self) -> dict:
        return {
            "n_samples": len(self.samples),
            "mean_cpu_w": round(self.mean_cpu_w, 3),
            "mean_gpu_w": round(self.mean_gpu_w, 3),
            "mean_ane_w": round(self.mean_ane_w, 3),
            "mean_total_w": round(self.mean_total_w, 3),
        }


_PLIST_START = b"<?xml"
_PLIST_END = b"</plist>"


def _parse_power_plist(data: bytes) -> tuple[list[PowerSample], bytes]:
    """Parse a ``powermetrics -f plist`` byte stream into PowerSamples.

    ``powermetrics`` emits one XML plist per sample, separated by a NUL byte
    (the stream reads ``…</plist>\\n\\x00<?xml…``), so NULs are stripped before
    the documents are handed to :mod:`plistlib`.

    Documents are cut on the ``</plist>`` boundary and the trailing bytes that
    do not yet form a complete document are returned alongside the samples.  A
    streaming caller must keep that remainder and prepend it to its next read —
    a single sample is larger than a 4 KiB read, so dropping the remainder
    loses roughly every second sample.
    """
    samples: list[PowerSample] = []
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
            continue

        # Every power field lives under "processor"; the top-level "gpu" dict
        # only carries dvfm/frequency/energy entries, so reading gpu_power
        # from it always yielded 0.
        proc = d.get("processor", {})
        samples.append(PowerSample(
            timestamp=time.monotonic(),
            cpu_mw=float(proc.get("cpu_power", proc.get("package_power", 0.0)) or 0.0),
            gpu_mw=float(proc.get("gpu_power", d.get("gpu_power", 0.0)) or 0.0),
            ane_mw=float(proc.get("ane_power", d.get("ane_power", 0.0)) or 0.0),
        ))
    # Drop the separator (newline + NUL) so a caller's buffer is left empty
    # when the stream ends on a document boundary.
    return samples, data[pos:].lstrip(b"\x00 \t\r\n")


class PowerMonitor:
    """Context manager that runs ``sudo powermetrics`` in the background.

    Usage::

        with PowerMonitor(sample_ms=200) as pm:
            run_inference()
        trace = pm.trace
    """

    def __init__(self, sample_ms: int = 200):
        self.sample_ms = sample_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._trace = PowerTrace()

    @property
    def trace(self) -> PowerTrace:
        return self._trace

    def __enter__(self) -> "PowerMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "--sample-rate", str(self.sample_ms),
            "-f", "plist",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("⚠ powermetrics not found — power monitoring disabled")
            return

        self._thread = threading.Thread(
            target=self._reader, args=(self._proc.stdout,), daemon=True,
        )
        self._thread.start()

    def _reader(self, stdout) -> None:
        """Background thread: sole reader of stdout, until EOF.

        ``stdout`` is passed in rather than reached for through ``self._proc``
        so that ``stop()`` clearing that attribute cannot race this thread.
        The loop runs to EOF, which ``stop()`` produces by terminating the
        process; anything left buffered at that point is parsed here.
        """
        buf = b""
        while chunk := stdout.read(4096):
            buf += chunk
            samples, buf = _parse_power_plist(buf)
            self._trace.samples.extend(samples)

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
        # The reader thread owns stdout — let it drain to EOF and finish
        # before anything else touches the pipe or clears self._proc.
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        proc.stdout.close()  # type: ignore[union-attr]
        self._proc = None


def check_power_available() -> bool:
    """Check if sudo powermetrics can run without a password prompt."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "-f", "plist"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
