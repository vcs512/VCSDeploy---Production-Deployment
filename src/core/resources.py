"""System resource monitoring for the evaluation service."""

import threading
import time

import numpy as np
import psutil

from src.schemas.evaluation import ResourceStats, SystemResourceState

_PERCENTILES = (50, 90, 95, 99)


def _percentiles(values: list[float]) -> dict:
    """Compute peak and percentile statistics for a sample series.

    Args:
        values: Ordered sample values.

    Returns:
        Dictionary with a ``peak`` field and one entry per percentile point.
    """
    if not values:
        return {"peak": 0.0}
    array = np.asarray(values, dtype=float)
    stats = {"peak": float(np.max(array))}
    for point in _PERCENTILES:
        stats[f"p{point}"] = float(np.percentile(array, point))
    return stats


class _GpuContext:
    """Lazy wrapper around pynvml for GPU sampling (best-effort)."""

    def __init__(self) -> None:
        self._handle = None
        self._enabled = False
        self._init()

    def _init(self) -> None:
        """Initialize the NVML handle when a CUDA device is present."""
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                pynvml.nvmlShutdown()
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._enabled = True
        except (ImportError, OSError, RuntimeError):
            self._handle = None
            self._enabled = False

    def read(self) -> tuple[float, float]:
        """Read current GPU utilization and used memory.

        Returns:
            A tuple with the utilization (percent) and used memory (bytes).
        """
        if not self._enabled or self._handle is None:
            return 0.0, 0.0
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return float(util.gpu), float(memory.used)
        except (OSError, RuntimeError):
            return 0.0, 0.0


class ResourceMonitor:
    """Sample CPU, RAM, VRAM and GPU usage in a background thread.

    Attributes:
        sample_freq_hz: Sampling frequency in hertz.
    """

    def __init__(self, sample_freq_hz: float = 2.0) -> None:
        """Initialize the monitor.

        Args:
            sample_freq_hz: Sampling frequency in hertz.
        """
        self.sample_freq_hz = sample_freq_hz
        self._samples: list[SystemResourceState] = []
        self._process = psutil.Process()
        self._gpu = _GpuContext()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def start(self) -> None:
        """Begin sampling in a daemon background thread."""
        self._start_time = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _sample_loop(self) -> None:
        """Sample resources until the stop event is set."""
        interval = 1.0 / max(self.sample_freq_hz, 0.1)
        while not self._stop.is_set():
            self._samples.append(self._snapshot())
            self._stop.wait(interval)

    def _snapshot(self) -> SystemResourceState:
        """Capture a single resource sample.

        Returns:
            A snapshot of the current system resource usage.
        """
        cpu_percent = self._process.cpu_percent(interval=None)
        ram_bytes = float(self._process.memory_info().rss)
        gpu_util, vram_bytes = self._gpu.read()
        return SystemResourceState(
            cpu_percent=float(cpu_percent),
            ram_used_bytes=ram_bytes,
            vram_used_bytes=vram_bytes,
            gpu_util_percent=gpu_util,
            time_seconds=time.monotonic() - self._start_time,
        )

    def summarize(self) -> ResourceStats:
        """Summarise the collected samples with peak and percentiles.

        Returns:
            Resource statistics computed from all collected samples.
        """
        cpu = [sample.cpu_percent for sample in self._samples]
        ram = [sample.ram_used_bytes for sample in self._samples]
        vram = [sample.vram_used_bytes for sample in self._samples]
        gpu = [sample.gpu_util_percent for sample in self._samples]
        elapsed = (
            self._samples[-1].time_seconds if self._samples else 0.0
        )
        return ResourceStats(
            cpu_percent=_percentiles(cpu),
            ram_used_bytes=_percentiles(ram),
            vram_used_bytes=_percentiles(vram),
            gpu_util_percent=_percentiles(gpu),
            elapsed_seconds=elapsed,
        )
