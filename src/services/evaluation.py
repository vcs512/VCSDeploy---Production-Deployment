"""Evaluation service entrypoint for the trained image classifier."""

import argparse

from src.core.evaluation import run_evaluation
from src.core.writer import write_evaluation_csv
from src.schemas.evaluation import EvaluationConfig, EvaluationResult


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier and report resource usage.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.json",
        help="Path to the evaluation JSON configuration.",
    )
    return parser.parse_args()


def _format_bytes(value: float) -> str:
    """Format a byte count into a human readable string.

    Args:
        value: Byte count to format.

    Returns:
        Human readable byte string with units.
    """
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _print_stat(label: str, stats: dict, display_unit: bool = True) -> None:
    """Print a labelled resource statistic series.

    Args:
        label: Name of the resource metric.
        stats: Dictionary with peak and percentile values.
        display_unit: When true format bytes with units.
    """
    if display_unit:
        formatted = {k: _format_bytes(v) for k, v in stats.items()}
    else:
        formatted = {k: f"{v:.1f}%" for k, v in stats.items()}
    print(f"  {label}: {formatted}")


def print_result(result: EvaluationResult) -> None:
    """Print the evaluation outcome to stdout.

    Args:
        result: Evaluation result to display.
    """
    print("Classification metrics:")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key}: {result.metrics.get(key, 0.0):.4f}")
    print(f"samples={result.num_samples} batches={result.num_batches}")

    resources = result.resources
    print("Resources:")
    _print_stat("cpu% ", resources.cpu_percent, display_unit=False)
    _print_stat("ram  ", resources.ram_used_bytes)
    _print_stat("vram ", resources.vram_used_bytes)
    _print_stat("gpu% ", resources.gpu_util_percent, display_unit=False)
    print(f"  elapsed: {resources.elapsed_seconds:.2f} s")


def main() -> None:
    """Load the configuration and run the evaluation routine."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = EvaluationConfig.model_validate_json(handle.read())
    result = run_evaluation(config=config)
    print_result(result)
    report_path = write_evaluation_csv(
        result=result,
        config=config,
        backend="pytorch",
    )
    print(f"CSV report: {report_path}")


if __name__ == "__main__":
    main()
