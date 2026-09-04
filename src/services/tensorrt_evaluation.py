"""TensorRT evaluation service entrypoint."""

import argparse

from src.core.reporting import print_result
from src.core.tensorrt_evaluation import run_tensorrt_evaluation
from src.core.writer import write_evaluation_csv
from src.schemas.tensorrt_evaluation import TensorrtEvaluationConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a serialized TensorRT engine and report resources.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tensorrt_evaluation.json",
        help="Path to the TensorRT evaluation JSON configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configuration and run the TensorRT evaluation routine."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = TensorrtEvaluationConfig.model_validate_json(handle.read())
    result = run_tensorrt_evaluation(config=config)
    print_result(result)
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="tensorrt",
        device=config.resolved_device_value(),
    )
    print(f"CSV report: {report_path}")


if __name__ == "__main__":
    main()
