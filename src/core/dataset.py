"""Image preprocessing transforms and collation for classification."""

from collections.abc import Callable

import torch
from torchvision import transforms


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
