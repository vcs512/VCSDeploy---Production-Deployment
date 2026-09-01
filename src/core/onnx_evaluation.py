"""Evaluation loop for the exported ONNX classifier."""

import time

import numpy as np
import onnxruntime as ort
import tqdm

from src.core.loader import build_split_loader, read_image_processor_stats
from src.core.metrics import compute_metrics
from src.core.resources import ResourceMonitor, summarize_latencies
from src.schemas.evaluation import EvaluationResult
from src.schemas.onnx_evaluation import OnnxEvaluationConfig


def build_session(config: OnnxEvaluationConfig) -> ort.InferenceSession:
    """Create an ONNX Runtime session for the exported graph.

    Args:
        config: Resolved ONNX evaluation configuration.

    Returns:
        An inference session bound to the preferred available provider.
    """
    provider = config.resolved_provider()
    return ort.InferenceSession(
        config.onnx_path,
        providers=[provider],
    )


def run_onnx_evaluation(config: OnnxEvaluationConfig) -> EvaluationResult:
    """Evaluate the exported ONNX model on the configured split.

    Args:
        config: Resolved ONNX evaluation configuration.

    Returns:
        The classification metrics and resource usage summary.
    """
    session = build_session(config)
    input_name = session.get_inputs()[0].name

    processor = read_image_processor_stats(config.model_path)
    image_size = config.image_size or processor["image_size"]
    loader, _, _ = build_split_loader(
        config,
        image_size=image_size,
        image_mean=processor["image_mean"],
        image_std=processor["image_std"],
    )

    monitor = ResourceMonitor(sample_freq_hz=config.sample_freq_hz)
    monitor.start()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    latencies: list[float] = []
    num_batches = 0
    try:
        for batch in tqdm.tqdm(loader, desc="Processing batches"):
            pixel_values = batch["pixel_values"].numpy()
            labels = batch["labels"].numpy()
            started = num_batches >= 1
            start = time.perf_counter() if started else 0.0
            logits = session.run(None, {input_name: pixel_values})[0]
            if started:
                latencies.append(time.perf_counter() - start)
            all_preds.append(np.argmax(logits, axis=-1))
            all_labels.append(labels)
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
