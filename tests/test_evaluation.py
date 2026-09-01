"""Unit tests for the evaluation routine (CPU)."""

import pandas as pd
import pytest

from src.core.loader import read_image_processor_stats
from src.core.resources import ResourceMonitor, _percentiles, summarize_latencies
from src.core.writer import write_evaluation_csv
from src.schemas.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    ResourceStats,
)


def test_evaluation_config_loads():
    with open("configs/evaluation.json", encoding="utf-8") as handle:
        config = EvaluationConfig.model_validate_json(handle.read())
    assert config.model_path == "experiments/best"
    assert config.split == "test"
    assert config.device == "auto"
    assert config.resolved_device() in ("cuda", "cpu")


def test_percentiles_peak_and_points():
    values = list(range(100))
    stats = _percentiles(values)
    assert stats["peak"] == 99.0
    assert stats["p50"] == 49.5
    assert stats["p90"] == pytest.approx(89.1)
    assert stats["p95"] == pytest.approx(94.05)
    assert stats["p99"] == pytest.approx(98.01)


def test_percentiles_empty():
    assert _percentiles([]) == {"peak": 0.0}


def test_summarize_latencies_percentiles():
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats["peak"] == 4.0
    assert stats["p50"] == 2.5
    assert stats["p90"] == pytest.approx(3.7)


def test_summarize_latencies_empty():
    assert summarize_latencies([]) == {"peak": 0.0}


def test_resource_monitor_summarize():
    monitor = ResourceMonitor(sample_freq_hz=50.0)
    monitor._samples.append(
        type(
            "Sample",
            (),
            {
                "cpu_percent": 10.0,
                "ram_used_bytes": 100.0,
                "vram_used_bytes": 200.0,
                "gpu_util_percent": 30.0,
                "time_seconds": 0.0,
            },
        )()
    )
    stats = monitor.summarize()
    assert stats.cpu_percent["peak"] == 10.0
    assert stats.vram_used_bytes["peak"] == 200.0
    assert stats.gpu_util_percent["peak"] == 30.0
    assert stats.elapsed_seconds == 0.0


def test_evaluation_result_defaults():
    result = EvaluationResult()
    assert result.num_samples == 0
    assert result.metrics == {}
    assert result.resources.cpu_percent == {}


def _sample_result() -> EvaluationResult:
    return EvaluationResult(
        num_samples=128,
        num_batches=16,
        metrics={"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75},
        resources=ResourceStats(
            cpu_percent={"peak": 90.0, "p50": 50.0},
            ram_used_bytes={"peak": 100.0, "p50": 80.0},
            vram_used_bytes={"peak": 200.0, "p50": 150.0},
            gpu_util_percent={"peak": 60.0, "p50": 40.0},
            inference_seconds={"peak": 1.0, "p50": 0.5},
            elapsed_seconds=5.0,
        ),
    )


def test_to_flat_record_includes_backend_and_stats():
    result = _sample_result()
    record = result.to_flat_record(
        backend="pytorch",
        device="cuda",
        model_path="experiments/best",
        dataset_name="AI-Lab-Makerere/beans",
        split="test",
    )
    assert record["backend"] == "pytorch"
    assert record["device"] == "cuda"
    assert record["model_path"] == "experiments/best"
    assert record["accuracy"] == 0.9
    assert record["cpu_peak"] == 90.0
    assert record["vram_peak"] == 200.0
    assert record["gpu_p50"] == 40.0
    assert record["infer_peak"] == 1.0
    assert record["infer_p50"] == 0.5
    assert record["elapsed_seconds"] == 5.0


def test_write_evaluation_csv(tmp_path):
    config = EvaluationConfig(model_path="experiments/best", output_dir=str(tmp_path))
    result = _sample_result()
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="pytorch",
        device=config.resolved_device(),
    )
    assert report_path.exists()
    assert report_path.name.startswith("evaluation_pytorch_")
    assert report_path.suffix == ".csv"
    frame = pd.read_csv(report_path)
    assert len(frame) == 1
    assert frame.loc[0, "backend"] == "pytorch"
    assert frame.loc[0, "device"] == config.resolved_device()
    assert frame.loc[0, "accuracy"] == 0.9
    assert frame.loc[0, "vram_peak"] == 200.0
    assert frame.loc[0, "infer_peak"] == 1.0


def test_read_image_processor_stats_from_json():
    stats = read_image_processor_stats("experiments/best")
    assert stats["image_size"] == 256
    assert stats["image_mean"] == pytest.approx((0.485, 0.456, 0.406))
    assert stats["image_std"] == pytest.approx((0.229, 0.224, 0.225))
