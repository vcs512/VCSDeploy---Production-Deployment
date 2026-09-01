"""Configuration schema for the ONNX evaluation service."""

from typing import Literal

from pydantic import Field

from src.schemas.evaluation import EvaluationConfig


class OnnxEvaluationConfig(EvaluationConfig):
    """Configuration for evaluating an exported ONNX classifier.

    Attributes:
        onnx_path: Path to the exported ONNX graph.
        provider: Preferred ONNX Runtime provider ('cuda' or 'cpu'); falls
            back to CPU when the preferred provider is unavailable.
    """

    onnx_path: str = Field(default="experiments/best/export/onnx/model.onnx")
    provider: Literal["cuda", "cpu"] = Field(default="cuda")

    def resolved_provider(self) -> str:
        """Resolve the ONNX Runtime provider name.

        Returns:
            'CUDAExecutionProvider' when CUDA is requested and available,
            otherwise 'CPUExecutionProvider'.
        """
        import onnxruntime as ort

        available = ort.get_available_providers()
        preferred = (
            "CUDAExecutionProvider" if self.provider == "cuda" else "CPUExecutionProvider"
        )
        if preferred in available:
            return preferred
        return "CPUExecutionProvider"

    def resolved_device_value(self) -> str:
        """Resolve the runtime device used for inference.

        Returns:
            'cuda' when the resolved provider runs on CUDA, otherwise 'cpu'.
        """
        return "cuda" if self.resolved_provider() == "CUDAExecutionProvider" else "cpu"
