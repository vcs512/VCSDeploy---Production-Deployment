"""CSV report writer for evaluation results."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.schemas.evaluation import EvaluationConfig, EvaluationResult


def _build_report_path(output_dir: str, backend: str) -> Path:
    """Build the timestamped report path for a backend.

    Args:
        output_dir: Directory where the report is written.
        backend: Name of the inference runtime.

    Returns:
        Absolute path to the CSV report file.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{backend}_{timestamp}.csv"
    return Path(output_dir) / filename


def write_evaluation_csv(
    result: EvaluationResult,
    config: EvaluationConfig,
    backend: str,
    device: str,
) -> Path:
    """Write the evaluation result summary to a CSV file.

    Args:
        result: Evaluation result to persist.
        config: Resolved evaluation configuration.
        backend: Name of the inference runtime (e.g. 'pytorch').
        device: Runtime device used for inference ('cpu' or 'cuda').

    Returns:
        Path to the written CSV report.
    """
    record = result.to_flat_record(
        backend=backend,
        device=device,
        model_path=config.model_path,
        dataset_name=config.dataset_name,
        split=config.split,
    )
    report_path = _build_report_path(config.output_dir, backend)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(report_path, index=False)
    return report_path
