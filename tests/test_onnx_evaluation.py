"""Unit tests for the ONNX evaluation routine."""

import numpy as np
import onnxruntime as ort
import pytest

from src.core.onnx_evaluation import build_session
from src.core.writer import write_evaluation_csv
from src.schemas.evaluation import EvaluationResult, ResourceStats
from src.schemas.onnx_evaluation import OnnxEvaluationConfig

ONNX_PATH = "experiments/best/export/onnx/model.onnx"


def test_onnx_config_loads():
    with open("configs/onnx_evaluation.json", encoding="utf-8") as handle:
        config = OnnxEvaluationConfig.model_validate_json(handle.read())
    assert config.onnx_path == ONNX_PATH
    assert config.split == "test"
    assert config.provider == "cuda"
    assert config.provider in ("cuda", "cpu")


def test_resolved_provider_falls_back_to_cpu():
    config = OnnxEvaluationConfig(onnx_path=ONNX_PATH, provider="cuda")
    available = ort.get_available_providers()
    assert config.resolved_provider() in available
    if "CUDAExecutionProvider" not in available:
        assert config.resolved_provider() == "CPUExecutionProvider"


def test_build_session_runs_cpu_inference():
    session = build_session(OnnxEvaluationConfig(onnx_path=ONNX_PATH, provider="cpu"))
    pixel_values = np.zeros((1, 3, 256, 256), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: pixel_values})[0]
    assert logits.shape == (1, 3)


def test_write_onnx_csv_uses_backend_column(tmp_path):
    config = OnnxEvaluationConfig(onnx_path=ONNX_PATH, output_dir=str(tmp_path))
    result = EvaluationResult(
        num_samples=64,
        num_batches=8,
        metrics={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        resources=ResourceStats(inference_seconds={"peak": 2.0, "p50": 1.0}),
    )
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="onnx",
        device=config.resolved_device_value(),
    )
    assert report_path.name.startswith("evaluation_onnx_")
    import pandas as pd

    frame = pd.read_csv(report_path)
    assert frame.loc[0, "backend"] == "onnx"
    assert frame.loc[0, "device"] == config.resolved_device_value()
    assert pytest.approx(frame.loc[0, "accuracy"]) == 1.0
    assert frame.loc[0, "infer_peak"] == 2.0


def test_resolved_device_value():
    config = OnnxEvaluationConfig(onnx_path=ONNX_PATH, provider="cpu")
    assert config.resolved_device_value() == "cpu"
