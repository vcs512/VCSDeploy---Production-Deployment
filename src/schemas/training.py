"""Training configuration schema for the image-classification routine."""

import os

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for fine-tuning a local image classifier.

    Attributes:
        model_local_path: Path to the local pretrained model directory.
        dataset_name: HuggingFace ``datasets`` dataset identifier.
        dataset_cache_dir: Directory where the dataset is cached (persisted).
        image_size: Square input resolution fed to the model.
        batch_size: Per-device batch size.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        freeze_backbone: When true only the classification head is trained.
        gradient_accumulation_steps: Optimizer steps before a weight update.
        mixed_precision: Enable AMP when a CUDA device is available.
        mlflow_tracking_uri: MLflow backend store URI.
        mlflow_experiment_name: MLflow experiment name.
        experiments_dir: Root directory for exported artifacts.
        output_model_subdir: Subdirectory (under experiments_dir) for the best model.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker processes.
    """

    model_local_path: str = Field(
        default="checkpoints/swinv2-tiny-patch4-window16-256",
    )
    dataset_name: str = Field(default="AI-Lab-Makerere/beans")
    dataset_cache_dir: str = Field(default="data")
    label_column: str = Field(default="labels")
    image_size: int = Field(default=224)
    batch_size: int = Field(default=8)
    num_epochs: int = Field(default=5)
    learning_rate: float = Field(default=2e-4)
    freeze_backbone: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=2)
    mixed_precision: bool = Field(default=True)
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment_name: str = Field(default="swinv2-beans-clf")
    experiments_dir: str = Field(default="experiments")
    output_model_subdir: str = Field(default="best")
    seed: int = Field(default=42)
    num_workers: int = Field(default=2)

    def model_dir(self) -> str:
        """Return the directory where the best model is written.

        Returns:
            Absolute path to the best-model output directory.
        """
        return os.path.join(self.experiments_dir, self.output_model_subdir)
