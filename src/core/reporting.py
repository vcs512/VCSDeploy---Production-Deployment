"""Shared console reporting for evaluation results."""

from collections.abc import Callable

from src.schemas.evaluation import EvaluationResult


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


def _format_percent(value: float) -> str:
    """Format a fraction as a percentage string.

    Args:
        value: Fractional value to format.

    Returns:
        Percentage string with a single decimal.
    """
    return f"{value:.1f}%"


def _format_seconds(value: float) -> str:
    """Format a duration in seconds.

    Args:
        value: Duration in seconds.

    Returns:
        Seconds string with three decimals.
    """
    return f"{value:.3f} s"


def _print_stat(
    label: str,
    stats: dict,
    formatter: Callable[[float], str] = _format_bytes,
) -> None:
    """Print a labelled statistic series.

    Args:
        label: Name of the statistic metric.
        stats: Dictionary with peak and percentile values.
        formatter: Callable formatting each value into a display string.
    """
    formatted = {k: formatter(v) for k, v in stats.items()}
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
    _print_stat("cpu% ", resources.cpu_percent, _format_percent)
    _print_stat("ram  ", resources.ram_used_bytes)
    _print_stat("vram ", resources.vram_used_bytes)
    _print_stat("gpu% ", resources.gpu_util_percent, _format_percent)
    _print_stat("infer", resources.inference_seconds, _format_seconds)
    print(f"  elapsed: {resources.elapsed_seconds:.2f} s")
