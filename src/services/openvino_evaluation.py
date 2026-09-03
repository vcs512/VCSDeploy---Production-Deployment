"""OpenVINO evaluation service entrypoint."""

import argparse

from src.core.openvino_evaluation import run_openvino_evaluation
from src.core.reporting import print_result
from src.core.writer import write_evaluation_csv
from src.schemas.openvino_evaluation import OpenVinoEvaluationConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an exported OpenVINO classifier and report resources.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/openvino_evaluation.json",
        help="Path to the OpenVINO evaluation JSON configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configuration and run the OpenVINO evaluation routine."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = OpenVinoEvaluationConfig.model_validate_json(handle.read())
    result = run_openvino_evaluation(config=config)
    print_result(result)
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="openvino",
        device=config.resolved_device_value(),
    )
    print(f"CSV report: {report_path}")


if __name__ == "__main__":
    main()