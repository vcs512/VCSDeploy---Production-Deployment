"""Model construction for the local image classifier."""


import torch
from transformers import AutoModelForImageClassification

from src.schemas.training import TrainingConfig


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze every parameter except the classification head.

    Args:
        model: The model whose backbone should be frozen.
    """
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def build_model(
    config: TrainingConfig,
    num_labels: int,
    id2label: dict,
    label2id: dict,
) -> torch.nn.Module:
    """Load the local model and adapt its classification head.

    Args:
        config: Resolved training configuration.
        num_labels: Number of target classes.
        id2label: Identifier-to-label-name mapping.
        label2id: Label-name-to-identifier mapping.

    Returns:
        The prepared model (optionally with a frozen backbone).
    """
    model = AutoModelForImageClassification.from_pretrained(
        config.model_local_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if config.freeze_backbone:
        freeze_backbone(model=model)
    return model
