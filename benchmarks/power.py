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


def _parse_power_plist(data: bytes) -> list[PowerSample]:
    """Parse powermetrics plist output into PowerSample list."""
    samples: list[PowerSample] = []
    try:
        # powermetrics outputs concatenated plists — split on the XML header.
        chunks = data.split(b"<?xml")
        for chunk in chunks:
            if not chunk.strip():
                continue
            xml = b"<?xml" + chunk
            try:
                d = plistlib.loads(xml)
            except Exception:
                continue

            cpu_mw = 0.0
            gpu_mw = 0.0
            ane_mw = 0.0

            # CPU power — look in processor dict
            proc = d.get("processor", {})
            cpu_mw = proc.get("cpu_power", proc.get("package_power", 0.0))

            # GPU power
            gpu_entries = d.get("gpu", [])
            if isinstance(gpu_entries, list):
                for g in gpu_entries:
                    gpu_mw += g.get("gpu_power", 0.0)
            elif isinstance(gpu_entries, dict):
                gpu_mw = gpu_entries.get("gpu_power", 0.0)

            # ANE power
            ane_mw = d.get("ane_power", proc.get("ane_power", 0.0))

            samples.append(PowerSample(
                timestamp=time.monotonic(),
                cpu_mw=cpu_mw,
                gpu_mw=gpu_mw,
                ane_mw=ane_mw,
            ))
    except Exception:
        pass
    return samples


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
        self._stop = threading.Event()

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

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Background thread: read stdout and parse samples."""
        assert self._proc is not None
        buf = b""
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            buf += chunk
            # Try to parse completed plist entries
            samples = _parse_power_plist(buf)
            if samples:
                self._trace.samples.extend(samples)
                buf = b""

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            # Read any remaining data
            remaining = self._proc.stdout.read()  # type: ignore[union-attr]
            if remaining:
                samples = _parse_power_plist(remaining)
                self._trace.samples.extend(samples)
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


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
