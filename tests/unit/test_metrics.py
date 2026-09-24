"""UT-16: test-metric computation on hand-made predictions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pso_rf.evaluation import compute_metrics


def test_multiclass_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert m["precision_macro"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert m["recall_macro"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert m["f1_macro"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["labels"] == [0, 1, 2]
    assert m["positive_class_precision"] is None
    assert m["positive_class_recall"] is None and m["positive_class_f1"] is None


def test_zero_division_is_zero_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_metrics(np.array([0, 1, 2]), np.array([0, 0, 0]), labels=[0, 1, 2])
    assert m["precision_macro"] == pytest.approx((1 / 3) / 3)


def test_binary_positive_class_metrics() -> None:
    m = compute_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), labels=[0, 1], positive_label=1)
    assert m["positive_class_precision"] == pytest.approx(1.0)
    assert m["positive_class_recall"] == pytest.approx(0.5)
    assert m["positive_class_f1"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]


@pytest.mark.filterwarnings("ignore:A single label was found")  # from balanced_accuracy on 1 class
def test_confusion_matrix_keeps_absent_labels() -> None:
    m = compute_metrics(np.array([0, 0]), np.array([0, 0]), labels=[0, 1, 2])
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]
