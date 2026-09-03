"""Unit tests for the OpenVINO evaluation routine."""

import subprocess

import numpy as np
import pytest

from src.core.openvino_evaluation import build_compiled_model
from src.core.writer import write_evaluation_csv
from src.schemas.evaluation import EvaluationResult, ResourceStats
from src.schemas.openvino_evaluation import OpenVinoEvaluationConfig

OPENVINO_PATH = "experiments/best/export/openvino/model.xml"

_HAS_AVX2 = "avx2" in subprocess.run(
    ["grep", "-o", r"\bavx2\b", "/proc/cpuinfo"],
    capture_output=True,
    text=True,
    check=False,
).stdout


def test_openvino_config_loads():
    with open("configs/openvino_evaluation.json", encoding="utf-8") as handle:
        config = OpenVinoEvaluationConfig.model_validate_json(handle.read())
    assert config.openvino_path == OPENVINO_PATH
    assert config.split == "test"
    assert config.ov_device == "gpu"


def test_resolved_ov_device_falls_back_to_cpu():
    config = OpenVinoEvaluationConfig(openvino_path=OPENVINO_PATH, ov_device="gpu")
    assert config.resolved_ov_device() in ("GPU", "CPU")
    if "GPU" not in config.resolved_ov_device():
        assert config.resolved_ov_device() == "CPU"


def test_resolved_device_value():
    config = OpenVinoEvaluationConfig(openvino_path=OPENVINO_PATH, ov_device="cpu")
    assert config.resolved_device_value() == "cpu"


@pytest.mark.skipif(
    not _HAS_AVX2,
    reason="OpenVINO CPU inference is impractical without AVX2 support",
)
def test_build_compiled_model_runs_cpu_inference():
    compiled = build_compiled_model(
        OpenVinoEvaluationConfig(openvino_path=OPENVINO_PATH, ov_device="cpu")
    )
    pixel_values = np.zeros((1, 3, 256, 256), dtype=np.float32)
    logits = compiled({compiled.input(0): pixel_values})[compiled.output(0)]
    assert np.asarray(logits).shape == (1, 3)


def test_write_openvino_csv_uses_backend_column(tmp_path):
    config = OpenVinoEvaluationConfig(openvino_path=OPENVINO_PATH, output_dir=str(tmp_path))
    result = EvaluationResult(
        num_samples=64,
        num_batches=8,
        metrics={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        resources=ResourceStats(inference_seconds={"peak": 2.0, "p50": 1.0}),
    )
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="openvino",
        device=config.resolved_device_value(),
    )
    assert report_path.name.startswith("evaluation_openvino_")
    import pandas as pd

    frame = pd.read_csv(report_path)
    assert frame.loc[0, "backend"] == "openvino"
    assert frame.loc[0, "device"] == config.resolved_device_value()
    assert pytest.approx(frame.loc[0, "accuracy"]) == 1.0
    assert frame.loc[0, "infer_peak"] == 2.0

