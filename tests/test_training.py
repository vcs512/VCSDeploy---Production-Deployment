"""Unit tests for the image-classification training routine (CPU)."""

import os

import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import nn

from src.core.dataset import build_collate_fn, build_transforms
from src.core.metrics import compute_metrics
from src.core.model import freeze_backbone
from src.schemas.training import TrainingConfig

CONFIG_PATH = os.path.join("configs", "training.json")


def test_training_config_loads():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = TrainingConfig.model_validate_json(handle.read())
    assert config.dataset_name == "AI-Lab-Makerere/beans"
    assert config.model_dir().endswith(os.path.join("experiments", "best"))


def test_compute_metrics_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 0, 1, 2]).numpy()
    preds = labels.copy()
    metrics = compute_metrics(predictions=preds, labels=labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_compute_metrics_matches_sklearn():
    labels = torch.tensor([0, 1, 2, 0, 1, 2]).numpy()
    preds = torch.tensor([0, 1, 2, 1, 1, 0]).numpy()
    metrics = compute_metrics(predictions=preds, labels=labels)
    assert metrics["accuracy"] == accuracy_score(labels, preds)


def test_build_transforms_output_shape():
    transform = build_transforms(
        image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    fake_image = Image.new("RGB", (64, 64))
    out = transform(fake_image)
    assert out.shape == (3, 32, 32)


def test_freeze_backbone_leaves_classifier_trainable():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 3)

        def forward(self, x):
            return self.classifier(self.backbone(x))

    model = DummyModel()
    freeze_backbone(model=model)
    assert model.backbone.weight.requires_grad is False
    assert model.classifier.weight.requires_grad is True


def test_build_collate_fn_produces_batched_tensors():
    transform = build_transforms(16, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    collate_fn = build_collate_fn(transform=transform, label_column="labels")
    batch = {
        "image": [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))],
        "labels": [0, 2],
    }
    out = collate_fn(batch)
    assert out["pixel_values"].shape == (2, 3, 16, 16)
    assert out["labels"].shape == (2,)
    assert out["labels"].dtype == torch.long
