"""Configuration and result schemas for the evaluation service."""

from typing import Literal

from pydantic import BaseModel, Field

PercentilePoints = Literal[50, 90, 95, 99]


class EvaluationConfig(BaseModel):
    """Configuration for evaluating a trained image classifier.

    Attributes:
        model_path: Directory of the trained Transformers model.
        dataset_name: HuggingFace ``datasets`` dataset identifier.
        dataset_cache_dir: Directory where the dataset is cached (persisted).
        label_column: Name of the dataset label column.
        split: Dataset split used for evaluation.
        image_size: Square input resolution; when null read from the model.
        batch_size: Per-device batch size.
        num_workers: DataLoader worker processes.
        device: Device used for inference ('auto', 'cuda' or 'cpu').
        sample_freq_hz: Sampling frequency of the resource monitor.
        output_dir: Directory where result reports are written.
        seed: Random seed for reproducibility.
    """

    model_path: str = Field(default="experiments/best")
    dataset_name: str = Field(default="AI-Lab-Makerere/beans")
    dataset_cache_dir: str = Field(default="data")
    label_column: str = Field(default="labels")
    split: str = Field(default="test")
    image_size: int | None = Field(default=None)
    batch_size: int = Field(default=8)
    num_workers: int = Field(default=0)
    device: str = Field(default="auto")
    sample_freq_hz: float = Field(default=2.0)
    output_dir: str = Field(default="experiments")
    seed: int = Field(default=42)

    def resolved_device(self) -> str:
        """Resolve the target device name.

        Returns:
            'cuda' when available and requested, otherwise 'cpu'.
        """
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class ResourceStats(BaseModel):
    """Summarised resource usage during evaluation.

    Attributes:
        cpu_percent: Percentile distribution and peak of CPU utilization.
        ram_used_bytes: Percentile distribution and peak of process RAM.
        vram_used_bytes: Percentile distribution and peak of GPU memory.
        gpu_util_percent: Percentile distribution and peak of GPU utilization.
        elapsed_seconds: Total elapsed evaluation time.
    """

    cpu_percent: dict = Field(default_factory=dict)
    ram_used_bytes: dict = Field(default_factory=dict)
    vram_used_bytes: dict = Field(default_factory=dict)
    gpu_util_percent: dict = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0.0)


class EvaluationResult(BaseModel):
    """Outcome of the evaluation routine.

    Attributes:
        num_samples: Number of evaluated examples.
        num_batches: Number of processed batches.
        metrics: Classification metrics (accuracy, precision, recall, f1).
        resources: Summarised resource usage.
    """

    num_samples: int = Field(default=0)
    num_batches: int = Field(default=0)
    metrics: dict = Field(default_factory=dict)
    resources: ResourceStats = Field(default_factory=ResourceStats)

    def to_flat_record(
        self,
        backend: str,
        model_path: str,
        dataset_name: str,
        split: str,
    ) -> dict:
        """Flatten the result into a single summary record.

        Args:
            backend: Name of the inference runtime (e.g. 'pytorch').
            model_path: Path of the evaluated model.
            dataset_name: Identifier of the evaluated dataset.
            split: Dataset split that was evaluated.

        Returns:
            A flat dictionary with one value per CSV column.
        """
        record = {
            "backend": backend,
            "model_path": model_path,
            "dataset_name": dataset_name,
            "split": split,
            "num_samples": self.num_samples,
            "num_batches": self.num_batches,
            "elapsed_seconds": self.resources.elapsed_seconds,
        }
        record.update(
            {f"{key}": value for key, value in self.metrics.items()}
        )
        for prefix, stats in (
            ("cpu", self.resources.cpu_percent),
            ("ram", self.resources.ram_used_bytes),
            ("vram", self.resources.vram_used_bytes),
            ("gpu", self.resources.gpu_util_percent),
        ):
            for name, value in stats.items():
                record[f"{prefix}_{name}"] = value
        return record


class SystemResourceState(BaseModel):
    """A single sampled snapshot of system resources.

    Attributes:
        cpu_percent: Process CPU utilization as a percentage.
        ram_used_bytes: Process resident memory in bytes.
        vram_used_bytes: Process GPU memory in bytes.
        gpu_util_percent: GPU utilization as a percentage.
        time_seconds: Timestamp of the sample relative to the run start.
    """

    cpu_percent: float = Field(default=0.0)
    ram_used_bytes: float = Field(default=0.0)
    vram_used_bytes: float = Field(default=0.0)
    gpu_util_percent: float = Field(default=0.0)
    time_seconds: float = Field(default=0.0)
