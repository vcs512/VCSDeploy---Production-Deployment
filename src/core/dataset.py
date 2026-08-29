"""Dataset loading and preprocessing for image classification."""

from collections.abc import Callable

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from src.schemas.training import TrainingConfig


def build_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Build the image transformation pipeline.

    Args:
        image_size: Square target resolution.
        mean: Per-channel normalization mean.
        std: Per-channel normalization standard deviation.

    Returns:
        Composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_collate_fn(
    transform: transforms.Compose,
    label_column: str,
) -> Callable:
    """Build a collate function that transforms a batch into tensors.

    Args:
        transform: Torchvision transform applied per image.
        label_column: Name of the dataset label column.

    Returns:
        A collate function returning a dict with ``pixel_values`` and ``labels``.
    """

    def collate_fn(batch):
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch[label_column]
        else:
            images = [example["image"] for example in batch]
            labels = [example[label_column] for example in batch]
        if isinstance(images, list):
            pixel_values = torch.stack([transform(image) for image in images])
        else:
            pixel_values = transform(images).unsqueeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def load_classification_dataset(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, int, dict, dict]:
    """Load the toy dataset and wrap it into train/validation loaders.

    Args:
        config: Resolved training configuration.

    Returns:
        A tuple with the training loader, validation loader, number of
        classes, the id-to-label mapping and the label-to-id mapping.
    """
    image_processor = AutoImageProcessor.from_pretrained(config.model_local_path)
    transform = build_transforms(
        image_size=config.image_size,
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )
    collate_fn = build_collate_fn(
        transform=transform, label_column=config.label_column
    )

    raw: DatasetDict = load_dataset(
        config.dataset_name, cache_dir=config.dataset_cache_dir
    )
    label_feature = raw["train"].features[config.label_column]
    num_labels = label_feature.num_classes
    id2label = {i: name for i, name in enumerate(label_feature.names)}
    label2id = {name: i for i, name in id2label.items()}

    train_loader = DataLoader(
        raw["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        raw["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, num_labels, id2label, label2id
