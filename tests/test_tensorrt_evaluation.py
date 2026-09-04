"""Unit tests for the TensorRT evaluation routine."""

import pandas as pd
import pytest
import torch

from src.core.tensorrt_evaluation import build_engine
from src.core.writer import write_evaluation_csv
from src.schemas.evaluation import EvaluationResult, ResourceStats
from src.schemas.tensorrt_evaluation import TensorrtEvaluationConfig

TENSORRT_PATH = "experiments/best/export/tensorrt/model.engine"


def test_tensorrt_config_loads():
    with open("configs/tensorrt_evaluation.json", encoding="utf-8") as handle:
        config = TensorrtEvaluationConfig.model_validate_json(handle.read())
    assert config.tensorrt_path == TENSORRT_PATH
    assert config.split == "test"
    assert config.engine_batch_size == 8


def test_resolved_device_value_is_cuda():
    config = TensorrtEvaluationConfig(tensorrt_path=TENSORRT_PATH)
    assert config.resolved_device_value() == "cuda"


def test_build_engine_runs_cuda_inference():
    assert torch.cuda.is_available(), "TensorRT evaluation requires CUDA"
    engine, context = build_engine(
        TensorrtEvaluationConfig(tensorrt_path=TENSORRT_PATH)
    )
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = tuple(context.get_tensor_shape(input_name))
    input_tensor = torch.zeros(
        input_shape, dtype=torch.float32, device=torch.device("cuda")
    )
    output_tensor = torch.empty(
        tuple(context.get_tensor_shape(output_name)),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("TensorRT execution failed.")
    torch.cuda.current_stream().synchronize()
    assert output_tensor.cpu().numpy().shape == tuple(
        context.get_tensor_shape(output_name)
    )


def test_write_tensorrt_csv_uses_backend_column(tmp_path):
    config = TensorrtEvaluationConfig(
        tensorrt_path=TENSORRT_PATH, output_dir=str(tmp_path)
    )
    result = EvaluationResult(
        num_samples=64,
        num_batches=8,
        metrics={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        resources=ResourceStats(inference_seconds={"peak": 2.0, "p50": 1.0}),
    )
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="tensorrt",
        device=config.resolved_device_value(),
    )
    assert report_path.name.startswith("evaluation_tensorrt_")
    frame = pd.read_csv(report_path)
    assert frame.loc[0, "backend"] == "tensorrt"
    assert frame.loc[0, "device"] == "cuda"
    assert pytest.approx(frame.loc[0, "accuracy"]) == 1.0
    assert frame.loc[0, "infer_peak"] == 2.0
