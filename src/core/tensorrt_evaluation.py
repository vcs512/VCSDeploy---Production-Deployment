"""Evaluation loop for the serialized TensorRT engine."""

import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import tqdm

from src.core.loader import build_split_loader, read_image_processor_stats
from src.core.metrics import compute_metrics
from src.core.resources import ResourceMonitor, summarize_latencies
from src.schemas.evaluation import EvaluationResult
from src.schemas.tensorrt_evaluation import TensorrtEvaluationConfig


def build_engine(
    config: TensorrtEvaluationConfig,
) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """Deserialize the engine and create its execution context.

    Args:
        config: Resolved TensorRT evaluation configuration.

    Returns:
        A tuple with the deserialized engine and its execution context.

    Raises:
        RuntimeError: When no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT evaluation requires a CUDA device.")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(config.tensorrt_path).read_bytes())
    context = engine.create_execution_context()
    return engine, context


def _resolve_image_size(config: TensorrtEvaluationConfig, processor: dict) -> int:
    """Resolve the image size from the config or the model processor.

    Args:
        config: Resolved TensorRT evaluation configuration.
        processor: Dictionary with ``image_size``, ``image_mean`` and
            ``image_std`` from the model processor config.

    Returns:
        The square input resolution fed to the engine.
    """
    return config.image_size or processor["image_size"]


def _pad_to_batch(batch: dict, engine_batch_size: int) -> tuple[torch.Tensor, int]:
    """Pad a batch to the fixed engine batch by repeating the last image.

    Args:
        batch: A single loader batch with ``pixel_values`` and ``labels``.
        engine_batch_size: Fixed batch dimension compiled into the engine.

    Returns:
        A tuple with the padded float32 input tensor and the valid count.
    """
    pixel_values = batch["pixel_values"]
    valid = pixel_values.shape[0]
    if valid == engine_batch_size:
        return pixel_values.to(torch.float32), valid
    repeat = [1] * pixel_values.ndim
    repeat[0] = engine_batch_size - valid
    last = pixel_values[-1:].repeat(repeat)
    padded = torch.cat([pixel_values, last], dim=0).to(torch.float32)
    return padded, valid


def run_tensorrt_evaluation(
    config: TensorrtEvaluationConfig,
) -> EvaluationResult:
    """Evaluate the serialized TensorRT engine on the configured split.

    Args:
        config: Resolved TensorRT evaluation configuration.

    Returns:
        The classification metrics and resource usage summary.
    """
    engine, context = build_engine(config)
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    processor = read_image_processor_stats(config.model_path)
    image_size = _resolve_image_size(config, processor)
    loader, _, _ = build_split_loader(
        config,
        image_size=image_size,
        image_mean=processor["image_mean"],
        image_std=processor["image_std"],
    )

    device = torch.device("cuda")
    input_shape = (
        config.engine_batch_size,
        3,
        image_size,
        image_size,
    )
    output_shape = (config.engine_batch_size, 3)
    input_tensor = torch.empty(input_shape, dtype=torch.float32, device=device)
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
    context.set_input_shape(input_name, tuple(input_shape))

    monitor = ResourceMonitor(sample_freq_hz=config.sample_freq_hz)
    monitor.start()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    latencies: list[float] = []
    num_batches = 0
    try:
        stream_handle = torch.cuda.current_stream().cuda_stream
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        for batch in tqdm.tqdm(loader, desc="Processing batches"):
            padded, valid = _pad_to_batch(batch, config.engine_batch_size)
            labels = batch["labels"].numpy().astype(np.int64)
            started = num_batches >= 1
            start = time.perf_counter() if started else 0.0
            input_tensor.copy_(padded)
            if not context.execute_async_v3(stream_handle):
                raise RuntimeError("TensorRT execution failed.")
            torch.cuda.current_stream().synchronize()
            if started:
                latencies.append(time.perf_counter() - start)
            logits = output_tensor.cpu().numpy()
            all_preds.append(np.argmax(logits[:valid], axis=-1))
            all_labels.append(labels[:valid])
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
