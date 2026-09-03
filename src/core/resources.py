"""System resource monitoring for the evaluation service."""

import glob
import os
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


def summarize_latencies(times: list[float]) -> dict:
    """Summarise per-inference latencies with peak and percentiles.

    Args:
        times: Per-inference wall-clock durations in seconds.

    Returns:
        Dictionary with a ``peak`` field and one entry per percentile point.
    """
    return _percentiles(times)


class _GpuContext:
    """Lazy GPU sampler using NVML, falling back to the DRM interface."""

    def __init__(self) -> None:
        self._handle = None
        self._nvml_errors = (OSError, RuntimeError, ImportError)
        self._drm_card = None
        self._enabled = False
        self._init()

    def _init(self) -> None:
        """Initialize the NVML handle or the DRM card when available."""
        try:
            import pynvml

            if hasattr(pynvml, "NVMLError_LibraryNotFound"):
                self._nvml_errors = (
                    pynvml.NVMLError_LibraryNotFound,
                    OSError,
                    RuntimeError,
                    ImportError,
                )
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                pynvml.nvmlShutdown()
                self._init_drm()
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._enabled = True
        except Exception as err:
            if not isinstance(err, self._nvml_errors):
                raise
            self._handle = None
            self._init_drm()

    def _init_drm(self) -> None:
        """Locate the first DRM card exposing a readable hwmon metric."""
        for card in glob.glob("/sys/class/drm/card*"):
            hwmon = os.path.join(card, "device", "hwmon")
            candidates = glob.glob(os.path.join(hwmon, "hwmon*"))
            if any(
                os.access(os.path.join(metric_dir, name), os.R_OK)
                for metric_dir in candidates
                for name in ("utilization", "mem_used_mb")
            ):
                self._drm_card = card
                self._enabled = True
                return

    def read(self) -> tuple[float, float]:
        """Read current GPU utilization and used memory.

        Returns:
            A tuple with the utilization (percent) and used memory (bytes).
        """
        if not self._enabled:
            return 0.0, 0.0
        if self._handle is not None:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return float(util.gpu), float(memory.used)
            except Exception as err:
                if not isinstance(err, self._nvml_errors):
                    raise
                return 0.0, 0.0
        if self._drm_card is not None:
            try:
                return _read_drm_metrics(self._drm_card)
            except (OSError, ValueError):
                return 0.0, 0.0
        return 0.0, 0.0


def _read_drm_metrics(card: str) -> tuple[float, float]:
    """Read utilization and used memory from a DRM card.

    Args:
        card: Absolute path to a /sys/class/drm/card* entry.

    Returns:
        A tuple with the utilization (percent) and used memory (bytes).
    """
    util = 0.0
    memory = 0.0
    for metric_dir in glob.glob(os.path.join(card, "device", "hwmon", "hwmon*")):
        utilization = os.path.join(metric_dir, "utilization")
        if os.access(utilization, os.R_OK):
            with open(utilization, encoding="utf-8") as handle:
                util = float(handle.read().strip())
        mem_used = os.path.join(metric_dir, "mem_used_mb")
        if os.access(mem_used, os.R_OK):
            with open(mem_used, encoding="utf-8") as handle:
                memory = float(handle.read().strip()) * 1024 * 1024
    return util, memory


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

    def summarize(self, inference_seconds: dict | None = None) -> ResourceStats:
        """Summarise the collected samples with peak and percentiles.

        Args:
            inference_seconds: Percentile distribution of per-batch latency.

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
            inference_seconds=inference_seconds or {},
            elapsed_seconds=elapsed,
        )
