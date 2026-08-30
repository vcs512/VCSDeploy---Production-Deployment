"""Unit tests for the export routine (CPU)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.core.exporter import (
    build_dummy_input,
    check_onnx,
    check_openvino,
    check_tensorrt,
    compute_reference_logits,
    export_onnx,
    export_openvino,
    export_tensorrt,
    resolve_image_size,
    run_export,
)
from src.schemas.export import ExportConfig, ExportTarget


class DummyClassifier(nn.Module):
    """Minimal classifier mimicking the Transformers forward contract."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, pixel_values):
        return self.classifier(self.flatten(pixel_values))


class DummyRefModel(nn.Module):
    """Minimal model whose forward returns an object with ``logits``."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(192, 3)

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.linear(pixel_values.flatten(start_dim=1)))


def build_config(tmp_path: Path) -> tuple[ExportConfig, torch.Tensor, DummyClassifier]:
    """Return a small export config, input tensor and dummy model."""
    config = ExportConfig(
        model_path=str(tmp_path),
        image_size=32,
        batch_size=1,
        num_channels=3,
        onnx_opset=13,
        targets=[ExportTarget.onnx],
    )
    input_tensor = build_dummy_input(
        image_size=32,
        batch_size=1,
        num_channels=3,
        device=torch.device("cpu"),
        seed=42,
    )
    return config, input_tensor, DummyClassifier(num_classes=3).eval()


def test_export_config_loads():
    with open("configs/export.json", encoding="utf-8") as handle:
        config = ExportConfig.model_validate_json(handle.read())
    assert config.model_path == "experiments/best"
    assert config.targets == [
        ExportTarget.onnx,
        ExportTarget.tensorrt,
        ExportTarget.openvino,
    ]
    assert config.tensorrt_precision == "fp16"
    assert config.openvino_dir() == Path("experiments/best/export/openvino")


def test_resolve_image_size_from_model_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"image_size": 224}))
    config = ExportConfig(model_path=str(tmp_path))
    assert resolve_image_size(config=config) == 224
    explicit = ExportConfig(model_path=str(tmp_path), image_size=300)
    assert resolve_image_size(config=explicit) == 300


def test_compute_reference_logits():
    model = DummyRefModel().eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    logits = compute_reference_logits(model=model, input_tensor=input_tensor)
    assert logits.shape == (1, 3)


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnx

    config, input_tensor, model = build_config(tmp_path)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        config=config,
        model=model,
        input_tensor=input_tensor,
        onnx_path=onnx_path,
    )
    assert onnx_path.exists()
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    with torch.no_grad():
        ref_logits = model(pixel_values=input_tensor)
    outcome = check_onnx(
        onnx_path=onnx_path,
        input_tensor=input_tensor,
        ref_logits=ref_logits,
    )
    assert outcome["argmax_match"] is True
    assert outcome["max_abs_diff"] <= 1e-4


def test_openvino_convert_tiny_onnx(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("openvino")

    config, input_tensor, model = build_config(tmp_path)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        config=config,
        model=model,
        input_tensor=input_tensor,
        onnx_path=onnx_path,
    )
    out_dir = tmp_path / "ir"
    export_openvino(config=config, onnx_path=onnx_path, out_dir=out_dir)
    assert (out_dir / "model.xml").exists()
    assert (out_dir / "model.bin").exists()

    with torch.no_grad():
        ref_logits = model(pixel_values=input_tensor)
    outcome = check_openvino(
        out_dir=out_dir,
        input_tensor=input_tensor,
        ref_logits=ref_logits,
    )
    assert outcome["argmax_match"] is True


def test_run_export_tensorrt_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("requires a CPU-only environment")
    config = ExportConfig(targets=[ExportTarget.tensorrt])
    with pytest.raises(RuntimeError, match="CUDA"):
        run_export(config=config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_tensorrt_engine_matches_pytorch(tmp_path):
    pytest.importorskip("tensorrt")

    config = ExportConfig(
        model_path=str(tmp_path),
        image_size=32,
        batch_size=1,
        num_channels=3,
        onnx_opset=13,
        targets=[ExportTarget.onnx],
        tensorrt_precision="fp16",
    )
    model = DummyClassifier(num_classes=3).eval().cuda()
    input_tensor = build_dummy_input(
        image_size=32,
        batch_size=1,
        num_channels=3,
        device=torch.device("cuda"),
        seed=42,
    )
    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        config=config,
        model=model,
        input_tensor=input_tensor,
        onnx_path=onnx_path,
    )
    with torch.no_grad():
        ref_logits = model(pixel_values=input_tensor)
    engine_bytes = export_tensorrt(
        config=config,
        onnx_path=onnx_path,
        engine_path=tmp_path / "model.engine",
    )
    outcome = check_tensorrt(
        engine_bytes=engine_bytes,
        input_tensor=input_tensor,
        ref_logits=ref_logits,
    )
    assert outcome["argmax_match"] is True
