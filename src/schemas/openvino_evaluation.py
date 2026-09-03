"""Configuration schema for the OpenVINO evaluation service."""

from typing import Literal

import openvino as ov
from pydantic import Field

from src.schemas.evaluation import EvaluationConfig


class OpenVinoEvaluationConfig(EvaluationConfig):
    """Configuration for evaluating an exported OpenVINO IR.

    Attributes:
        openvino_path: Path to the OpenVINO IR XML file.
        ov_device: Preferred OpenVINO device ('auto', 'gpu' or 'cpu');
            falls back to CPU when the preferred device is unavailable.
    """

    openvino_path: str = Field(default="experiments/best/export/openvino/model.xml")
    ov_device: Literal["auto", "gpu", "cpu"] = Field(default="cpu")

    def resolved_ov_device(self) -> str:
        """Resolve the OpenVINO device plugin name.

        Returns:
            'GPU' when a GPU is requested and available, otherwise 'CPU'.
        """
        available = ov.Core().available_devices
        preferred = "GPU" if self.ov_device == "gpu" else "CPU"
        if self.ov_device == "auto" and "GPU" in available:
            return "GPU"
        if preferred in available:
            return preferred
        return "CPU"

    def resolved_device_value(self) -> str:
        """Resolve the runtime device used for inference.

        Returns:
            'gpu' when the resolved plugin device runs on a GPU, otherwise
            'cpu'.
        """
        return "gpu" if self.resolved_ov_device() == "GPU" else "cpu"