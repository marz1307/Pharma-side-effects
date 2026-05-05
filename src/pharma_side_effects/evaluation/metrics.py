"""Evaluation metrics for classification pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    zero_division: int = 0,
) -> dict[str, Any]:
    """Compute a compact classification summary.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional explicit label ordering.
        zero_division: Behaviour when a metric is undefined.

    Returns:
        Dictionary with ``accuracy``, ``macro_f1``, ``weighted_f1``, and
        the full ``report`` dict from scikit-learn.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=zero_division,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        "report": report,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Compute a confusion matrix with explicit label ordering.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class ordering.

    Returns:
        Square confusion matrix as a numpy array.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
