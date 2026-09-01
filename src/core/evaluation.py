"""Evaluation loop for the trained image classifier."""

import time

import numpy as np
import torch
import tqdm
from torch.amp import autocast
from transformers import AutoModelForImageClassification

from src.core.loader import build_split_loader, read_image_processor_stats
from src.core.metrics import compute_metrics
from src.core.resources import ResourceMonitor, summarize_latencies
from src.schemas.evaluation import EvaluationConfig, EvaluationResult


def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    """Evaluate the trained model on the configured split.

    Args:
        config: Resolved evaluation configuration.

    Returns:
        The classification metrics and resource usage summary.
    """
    device_name = config.resolved_device()
    device = torch.device(device_name)

    processor = read_image_processor_stats(config.model_path)
    image_size = config.image_size or processor["image_size"]

    model = AutoModelForImageClassification.from_pretrained(
        config.model_path
    ).to(device).eval()
    model.compile()

    loader, _, _ = build_split_loader(
        config,
        image_size=image_size,
        image_mean=processor["image_mean"],
        image_std=processor["image_std"],
    )
    use_amp = device.type == "cuda"

    monitor = ResourceMonitor(sample_freq_hz=config.sample_freq_hz)
    monitor.start()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    latencies: list[float] = []
    num_batches = 0
    try:
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Processing batches"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                started = num_batches >= 1
                start = time.perf_counter() if started else 0.0
                with autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    outputs = model(pixel_values=pixel_values)
                if started:
                    latencies.append(time.perf_counter() - start)
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
        resources=monitor.summarize(
            inference_seconds=summarize_latencies(latencies),
        ),
    )
