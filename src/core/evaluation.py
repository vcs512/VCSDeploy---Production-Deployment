"""Evaluation loop for the trained image classifier."""

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.core.dataset import build_collate_fn, build_transforms
from src.core.metrics import compute_metrics
from src.core.resources import ResourceMonitor
from src.schemas.evaluation import EvaluationConfig, EvaluationResult


def build_split_loader(
    config: EvaluationConfig,
) -> tuple[DataLoader, int, list[str]]:
    """Build a data loader for the configured dataset split.

    Args:
        config: Resolved evaluation configuration.

    Returns:
        A tuple with the data loader, the number of classes and the names
        of the classes in identifier order.
    """
    image_processor = AutoImageProcessor.from_pretrained(config.model_path)
    transform = build_transforms(
        image_size=_resolve_image_size(config, image_processor),
        mean=image_processor.image_mean,
        std=image_processor.image_std,
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


def _resolve_image_size(
    config: EvaluationConfig,
    image_processor: AutoImageProcessor,
) -> int:
    """Resolve the square input resolution for evaluation.

    Args:
        config: Resolved evaluation configuration.
        image_processor: Processor used to derive the default resolution.

    Returns:
        The configured image size or the one from the processor.
    """
    if config.image_size is not None:
        return config.image_size
    size = getattr(image_processor, "size", None)
    if size is not None:
        height = getattr(size, "height", None) or size.get("height", None)
        if height is not None:
            return int(height)
        shortest = (
            getattr(size, "shortest_edge", None)
            or size.get("shortest_edge", None)
        )
        if shortest is not None:
            return int(shortest)
    return 224


def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    """Evaluate the trained model on the configured split.

    Args:
        config: Resolved evaluation configuration.

    Returns:
        The classification metrics and resource usage summary.
    """
    device_name = config.resolved_device()
    device = torch.device(device_name)

    model = AutoModelForImageClassification.from_pretrained(
        config.model_path
    ).to(device).eval()
    model.compile()

    loader, _, _ = build_split_loader(config)
    use_amp = device.type == "cuda"

    monitor = ResourceMonitor(sample_freq_hz=config.sample_freq_hz)
    monitor.start()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    num_batches = 0
    try:
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Processing batches"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                with autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    outputs = model(pixel_values=pixel_values)
                all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                num_batches += 1
    finally:
        monitor.stop()

    predictions = np.concatenate(all_preds)
    ground_truth = np.concatenate(all_labels)
    metrics = compute_metrics(predictions=predictions, labels=ground_truth)

    return EvaluationResult(
        num_samples=int(predictions.shape[0]),
        num_batches=num_batches,
        metrics=metrics,
        resources=monitor.summarize(),
    )
