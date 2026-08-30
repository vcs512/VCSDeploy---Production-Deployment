"""Export configuration schema for the model-conversion routine."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ExportTarget(str, Enum):
    """Export backends supported by the conversion service.

    Attributes:
        onnx: ONNX graph for the ONNX Runtime GPU backend.
        tensorrt: TensorRT engine built from the ONNX graph.
        openvino: OpenVINO IR converted from the ONNX graph.
    """

    onnx = "onnx"
    tensorrt = "tensorrt"
    openvino = "openvino"


class ExportConfig(BaseModel):
    """Configuration for converting a trained classifier.

    Attributes:
        model_path: Directory of the trained Transformers model.
        output_subdir: Subdirectory under model_path for converted artifacts.
        image_size: Square input resolution; when null read from the model.
        batch_size: Fixed batch dimension used by the TensorRT engine.
        num_channels: Number of input image channels.
        targets: Backends to convert to.
        onnx_opset: ONNX operator-set version used during export.
        tensorrt_precision: TensorRT engine precision ('fp32' or 'fp16').
        tensorrt_workspace_mb: TensorRT builder workspace memory in MiB.
        openvino_precision: OpenVINO IR precision ('fp32' or 'fp16').
        run_validation: Compare converted outputs against the PyTorch model.
        seed: Random seed used for validation inputs.
    """

    model_path: str = Field(default="experiments/best")
    output_subdir: str = Field(default="export")
    image_size: int | None = Field(default=None)
    batch_size: int = Field(default=1)
    num_channels: int = Field(default=3)
    targets: list[ExportTarget] = Field(
        default=[ExportTarget.onnx, ExportTarget.tensorrt, ExportTarget.openvino]
    )
    onnx_opset: int = Field(default=17)
    tensorrt_precision: str = Field(default="fp16")
    tensorrt_workspace_mb: int = Field(default=2048)
    openvino_precision: str = Field(default="fp16")
    run_validation: bool = Field(default=True)
    seed: int = Field(default=42)

    def onnx_dir(self) -> Path:
        """Return the directory where the ONNX artifact is written.

        Returns:
            Absolute path to the ONNX output directory.
        """
        return Path(self.model_path) / self.output_subdir / "onnx"

    def tensorrt_dir(self) -> Path:
        """Return the directory where the TensorRT engine is written.

        Returns:
            Absolute path to the TensorRT output directory.
        """
        return Path(self.model_path) / self.output_subdir / "tensorrt"

    def openvino_dir(self) -> Path:
        """Return the directory where the OpenVINO IR is written.

        Returns:
            Absolute path to the OpenVINO output directory.
        """
        return Path(self.model_path) / self.output_subdir / "openvino"


class ExportResult(BaseModel):
    """Artifacts produced by the conversion routine.

    Attributes:
        onnx_path: Path to the exported ONNX graph.
        tensorrt_path: Path to the TensorRT engine (null when skipped).
        openvino_dir: Directory of the OpenVINO IR (null when skipped).
        validation: Per-backend outcome of the lightweight sanity check.
    """

    onnx_path: Path
    tensorrt_path: Path | None = None
    openvino_dir: Path | None = None
    validation: dict = Field(default_factory=dict)
