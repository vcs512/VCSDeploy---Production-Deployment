"""Configuration schema for the TensorRT evaluation service."""

from pydantic import Field

from src.schemas.evaluation import EvaluationConfig


class TensorrtEvaluationConfig(EvaluationConfig):
    """Configuration for evaluating a serialized TensorRT engine.

    Attributes:
        tensorrt_path: Path to the serialized TensorRT engine.
        engine_batch_size: Fixed batch dimension compiled into the engine.
    """

    tensorrt_path: str = Field(default="experiments/best/export/tensorrt/model.engine")
    engine_batch_size: int = Field(default=8)

    def resolved_device_value(self) -> str:
        """Resolve the runtime device used for inference.

        Returns:
            'cuda' because the serialized TensorRT engine only runs on a
            CUDA device and has no CPU fallback.
        """
        return "cuda"
