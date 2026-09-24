"""Held-out test metrics (MLR-006, EXPERIMENT_PLAN §8.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int], positive_label: int | None = None
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro precision/recall/F1 (``zero_division=0``) and the confusion matrix.

    ``positive_label`` (binary datasets: 1 = disease) adds that class's precision, recall and F1; otherwise
    those fields are ``None``. Confusion-matrix rows are true labels, columns predicted, in ``labels`` order.
    """
    labels = [int(label) for label in labels]
    macro = {"labels": labels, "average": "macro", "zero_division": 0}
    record: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, **macro)),
        "recall_macro": float(recall_score(y_true, y_pred, **macro)),
        "f1_macro": float(f1_score(y_true, y_pred, **macro)),
        "positive_class_precision": None,
        "positive_class_recall": None,
        "positive_class_f1": None,
    }
    if positive_label is not None:
        binary = {"pos_label": positive_label, "average": "binary", "zero_division": 0}
        record["positive_class_precision"] = float(precision_score(y_true, y_pred, **binary))
        record["positive_class_recall"] = float(recall_score(y_true, y_pred, **binary))
        record["positive_class_f1"] = float(f1_score(y_true, y_pred, **binary))
    record["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    record["labels"] = labels
    return record
