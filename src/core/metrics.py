"""Evaluation metrics for image classification."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute classification metrics from predictions and ground truth.

    Args:
        predictions: Predicted class identifiers.
        labels: Ground-truth class identifiers.

    Returns:
        Dictionary with accuracy, precision, recall and f1 (macro averaged).
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
