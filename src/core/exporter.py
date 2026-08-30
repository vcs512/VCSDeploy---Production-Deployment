"""Export of a trained classifier to ONNX, TensorRT and OpenVINO."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForImageClassification

from src.schemas.export import ExportConfig, ExportResult, ExportTarget


def resolve_image_size(config: ExportConfig) -> int:
    """Resolve the square input resolution for the model.

    Args:
        config: Resolved export configuration.

    Returns:
        The configured image size or the one stored in the model config.
    """
    if config.image_size is not None:
        return config.image_size
    config_path = Path(config.model_path) / "config.json"
    with open(config_path, encoding="utf-8") as handle:
        model_config = json.load(handle)
    return int(model_config.get("image_size", 256))


def load_export_model(config: ExportConfig) -> nn.Module:
    """Load the trained classifier from the model directory.

    Args:
        config: Resolved export configuration.

    Returns:
        The trained image-classification model.
    """
    return AutoModelForImageClassification.from_pretrained(config.model_path)


def build_dummy_input(
    image_size: int,
    batch_size: int,
    num_channels: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Build a reproducible random input tensor for tracing and validation.

    Args:
        image_size: Square input resolution.
        batch_size: Batch dimension of the input tensor.
        num_channels: Number of input image channels.
        device: Device where the tensor is allocated.
        seed: Random seed for reproducibility.

    Returns:
        Random input tensor of shape (batch, channels, height, width).
    """
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        num_channels,
        image_size,
        image_size,
        device=device,
    )


def compute_reference_logits(
    model: nn.Module, input_tensor: torch.Tensor
) -> torch.Tensor:
    """Compute reference logits with the PyTorch model.

    Args:
        model: The PyTorch model in evaluation mode.
        input_tensor: Input tensor used for inference.

    Returns:
        Logits tensor produced by the reference model.
    """
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
    return outputs.logits


def export_onnx(
    config: ExportConfig,
    model: nn.Module,
    input_tensor: torch.Tensor,
    onnx_path: Path,
) -> None:
    """Export the model to an ONNX graph with a dynamic batch axis.

    Args:
        config: Resolved export configuration.
        model: The PyTorch model in evaluation mode.
        input_tensor: Input tensor used for tracing.
        onnx_path: Destination path for the ONNX file.
    """
    import onnx

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Legacy exporter keeps the stable dynamic-axes/opset contract for TRT.
    torch.onnx.export(
        model,
        (input_tensor,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=config.onnx_opset,
        do_constant_folding=True,
        dynamo=False,
    )
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)


def _to_fp16_onnx(onnx_path: Path, out_dir: Path) -> Path:
    """Convert an ONNX graph to half precision keeping float I/O.

    Args:
        onnx_path: Path to the source (float32) ONNX graph.
        out_dir: Directory where the half-precision graph is written.

    Returns:
        Path to the half-precision ONNX graph.
    """
    import onnx
    from onnxruntime.transformers import float16

    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = out_dir / "model_fp16.onnx"
    model = float16.convert_float_to_float16(
        onnx.load(onnx_path),
        keep_io_types=True,
    )
    onnx.save(model, fp16_path)
    return fp16_path


def export_tensorrt(
    config: ExportConfig,
    onnx_path: Path,
    engine_path: Path,
) -> bytes:
    """Build a fixed-shape TensorRT engine from the ONNX graph.

    Args:
        config: Resolved export configuration.
        onnx_path: Path to the ONNX graph.
        engine_path: Destination path for the serialized engine.

    Returns:
        The serialized TensorRT engine bytes.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    source_path = onnx_path
    if config.tensorrt_precision == "fp16" and fp16_flag is None:
        source_path = _to_fp16_onnx(
            onnx_path=onnx_path,
            out_dir=engine_path.parent,
        )
    with open(source_path, "rb") as handle:
        if not parser.parse(handle.read()):
            details = "\n".join(
                parser.get_error(i).desc() for i in range(parser.num_errors)
            )
            raise RuntimeError(f"Failed to parse ONNX graph: {details}")

    image_size = resolve_image_size(config=config)
    input_tensor = network.get_input(0)
    input_tensor.shape = (
        config.batch_size,
        config.num_channels,
        image_size,
        image_size,
    )

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        config.tensorrt_workspace_mb * 1024 * 1024,
    )
    if config.tensorrt_precision == "fp16" and fp16_flag is not None:
        builder_config.set_flag(fp16_flag)

    engine_raw = builder.build_serialized_network(network, builder_config)
    engine_bytes = engine_raw if isinstance(engine_raw, bytes) else bytes(engine_raw)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    return engine_bytes


def export_openvino(
    config: ExportConfig,
    onnx_path: Path,
    out_dir: Path,
) -> None:
    """Convert the ONNX graph to an OpenVINO IR.

    Args:
        config: Resolved export configuration.
        onnx_path: Path to the ONNX graph.
        out_dir: Destination directory for the XML and BIN IR files.
    """
    import openvino as ov

    ir_model = ov.convert_model(str(onnx_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "model.xml"
    ov.save_model(
        ir_model,
        str(xml_path),
        compress_to_fp16=config.openvino_precision == "fp16",
    )


def validate_logits(pred_logits: np.ndarray, ref_logits: torch.Tensor) -> dict:
    """Compare predicted logits against the PyTorch reference.

    Args:
        pred_logits: Logits produced by a converted backend.
        ref_logits: Logits produced by the PyTorch model.

    Returns:
        Dictionary with the argmax match and the maximum absolute difference.
    """
    reference = ref_logits.detach().cpu().numpy()
    pred_classes = np.argmax(pred_logits, axis=-1)
    ref_classes = np.argmax(reference, axis=-1)
    return {
        "argmax_match": bool(np.array_equal(pred_classes, ref_classes)),
        "max_abs_diff": float(np.max(np.abs(pred_logits - reference))),
    }


def check_onnx(
    onnx_path: Path,
    input_tensor: torch.Tensor,
    ref_logits: torch.Tensor,
) -> dict:
    """Validate the ONNX graph against the PyTorch reference.

    Args:
        onnx_path: Path to the ONNX graph.
        input_tensor: Input tensor used for inference.
        ref_logits: Reference logits produced by the PyTorch model.

    Returns:
        Dictionary with the argmax match and the maximum absolute difference.
    """
    import onnxruntime as ort

    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: input_tensor.cpu().numpy()})[0]
    return validate_logits(pred_logits=predictions, ref_logits=ref_logits)


def check_tensorrt(
    engine_bytes: bytes,
    input_tensor: torch.Tensor,
    ref_logits: torch.Tensor,
) -> dict:
    """Validate the TensorRT engine against the PyTorch reference.

    Args:
        engine_bytes: Serialized TensorRT engine bytes.
        input_tensor: Input tensor allocated on the CUDA device.
        ref_logits: Reference logits produced by the PyTorch model.

    Returns:
        Dictionary with the argmax match and the maximum absolute difference.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, tuple(input_tensor.shape))

    output_shape = tuple(context.get_tensor_shape(output_name))
    output_tensor = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=input_tensor.device,
    )
    stream_handle = torch.cuda.current_stream().cuda_stream
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())
    if not context.execute_async_v3(stream_handle):
        raise RuntimeError("TensorRT execution failed.")
    torch.cuda.current_stream().synchronize()

    return validate_logits(
        pred_logits=output_tensor.cpu().numpy(),
        ref_logits=ref_logits,
    )


def check_openvino(
    out_dir: Path,
    input_tensor: torch.Tensor,
    ref_logits: torch.Tensor,
) -> dict:
    """Validate the OpenVINO IR against the PyTorch reference.

    Args:
        out_dir: Directory of the OpenVINO IR files.
        input_tensor: Input tensor used for inference.
        ref_logits: Reference logits produced by the PyTorch model.

    Returns:
        Dictionary with the argmax match and the maximum absolute difference.
    """
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(str(out_dir / "model.xml"), "CPU")
    predictions = compiled([input_tensor.cpu().numpy()])
    return validate_logits(
        pred_logits=predictions[compiled.output(0)],
        ref_logits=ref_logits,
    )


def run_export(config: ExportConfig) -> ExportResult:
    """Export the trained model to each configured backend.

    Args:
        config: Resolved export configuration.

    Returns:
        Result describing the produced artifacts and validation outcomes.
    """
    if ExportTarget.tensorrt in config.targets and not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires a CUDA device; run the export service "
            "through docker compose with GPU reservations."
        )

    image_size = resolve_image_size(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_export_model(config=config).to(device).eval()
    input_tensor = build_dummy_input(
        image_size=image_size,
        batch_size=config.batch_size,
        num_channels=config.num_channels,
        device=device,
        seed=config.seed,
    )

    onnx_path = config.onnx_dir() / "model.onnx"
    export_onnx(
        config=config,
        model=model,
        input_tensor=input_tensor,
        onnx_path=onnx_path,
    )
    ref_logits = compute_reference_logits(model=model, input_tensor=input_tensor)

    validation: dict = {}
    tensorrt_path = None
    openvino_dir = None

    if ExportTarget.onnx in config.targets and config.run_validation:
        validation["onnx"] = check_onnx(
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            ref_logits=ref_logits,
        )

    if ExportTarget.tensorrt in config.targets:
        engine_path = config.tensorrt_dir() / "model.engine"
        engine_bytes = export_tensorrt(
            config=config,
            onnx_path=onnx_path,
            engine_path=engine_path,
        )
        tensorrt_path = engine_path
        if config.run_validation:
            validation["tensorrt"] = check_tensorrt(
                engine_bytes=engine_bytes,
                input_tensor=input_tensor,
                ref_logits=ref_logits,
            )

    if ExportTarget.openvino in config.targets:
        out_dir = config.openvino_dir()
        export_openvino(
            config=config,
            onnx_path=onnx_path,
            out_dir=out_dir,
        )
        openvino_dir = out_dir
        if config.run_validation:
            validation["openvino"] = check_openvino(
                out_dir=out_dir,
                input_tensor=input_tensor,
                ref_logits=ref_logits,
            )

    return ExportResult(
        onnx_path=onnx_path,
        tensorrt_path=tensorrt_path,
        openvino_dir=openvino_dir,
        validation=validation,
    )
