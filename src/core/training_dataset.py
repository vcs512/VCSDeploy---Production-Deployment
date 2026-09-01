"""Training dataset loading with a transformers image processor."""

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from src.core.dataset import build_collate_fn, build_transforms
from src.schemas.training import TrainingConfig


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
