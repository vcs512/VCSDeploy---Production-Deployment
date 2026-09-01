"""Transformers-free data loader for evaluation, shared across runtimes."""

import json
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.core.dataset import build_collate_fn, build_transforms
from src.schemas.evaluation import EvaluationConfig


def read_image_processor_stats(model_path: str) -> dict:
    """Read image size, mean and std from the model processor JSON config.

    Args:
        model_path: Directory of the model (config.json and
            preprocessor_config.json).

    Returns:
        Dictionary with ``image_size``, ``image_mean`` and ``image_std``.
    """
    model_dir = Path(model_path)
    with open(model_dir / "preprocessor_config.json", encoding="utf-8") as handle:
        processor = json.load(handle)
    image_size = processor.get("size")
    if isinstance(image_size, dict):
        image_size = image_size.get("height") or image_size.get(
            "shortest_edge"
        )
    if image_size is None:
        with open(model_dir / "config.json", encoding="utf-8") as handle:
            image_size = json.load(handle).get("image_size", 256)
    return {
        "image_size": int(image_size),
        "image_mean": tuple(processor["image_mean"]),
        "image_std": tuple(processor["image_std"]),
    }


def build_split_loader(
    config: EvaluationConfig,
    image_size: int,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
) -> tuple[DataLoader, int, list[str]]:
    """Build a data loader for the configured dataset split.

    Args:
        config: Resolved evaluation configuration.
        image_size: Square input resolution.
        image_mean: Per-channel normalization mean.
        image_std: Per-channel normalization standard deviation.

    Returns:
        A tuple with the data loader, the number of classes and the names
        of the classes in identifier order.
    """
    transform = build_transforms(
        image_size=image_size,
        mean=image_mean,
        std=image_std,
    )
    collate_fn = build_collate_fn(
        transform=transform, label_column=config.label_column
    )

    dataset = load_dataset(
        config.dataset_name,
        cache_dir=config.dataset_cache_dir,
        split=config.split,
    )
    label_feature = dataset.features[config.label_column]
    num_classes = label_feature.num_classes
    id2label = [name for name in label_feature.names]

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader, num_classes, id2label
