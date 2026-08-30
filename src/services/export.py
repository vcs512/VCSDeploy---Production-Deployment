"""Export service entrypoint for model conversion."""

import argparse

from src.core.exporter import run_export
from src.schemas.export import ExportConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert a trained model to ONNX, TensorRT and OpenVINO.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/export.json",
        help="Path to the export JSON configuration.",
    )
    return parser.parse_args()


def print_result(result) -> None:
    """Print the produced artifacts to stdout.

    Args:
        result: Export result returned by the conversion routine.
    """
    print(f"ONNX:     {result.onnx_path}")
    print(f"TensorRT: {result.tensorrt_path}")
    print(f"OpenVINO: {result.openvino_dir}")
    if result.validation:
        print("Validation:")
        for backend, outcome in result.validation.items():
            print(f"  {backend}: {outcome}")


def main() -> None:
    """Load the configuration and run the export routine."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = ExportConfig.model_validate_json(handle.read())
    result = run_export(config=config)
    print_result(result)


if __name__ == "__main__":
    main()
