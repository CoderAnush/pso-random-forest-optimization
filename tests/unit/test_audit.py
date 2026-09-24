"""UT-21: the class-ratio gate, and the audit record (RESULTS_SCHEMA §4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pso_rf.datasets import DatasetBundle, load_dataset
from pso_rf.datasets.audit import audit, count_duplicate_rows, select_fitness_metric

AUDIT_FIELDS = [
    "dataset",
    "n_samples",
    "n_features",
    "n_classes",
    "class_counts",
    "class_ratio",
    "fitness_metric",
    "missing_per_feature",
    "n_duplicate_rows",
    "duplicates_removed",
    "feature_names",
    "class_names",
]


def _bundle(counts: tuple[int, ...], seed: int = 0) -> DatasetBundle:
    """A synthetic bundle with the given class counts and unique continuous rows."""
    y = np.repeat(np.arange(len(counts)), counts).astype(np.int64)
    X = np.random.default_rng(seed).normal(size=(len(y), 3))
    return DatasetBundle("synthetic", X, y, ["f0", "f1", "f2"], [f"c{k}" for k in range(len(counts))], {})


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((50, 50, 50), "accuracy"),
        ((30, 20), "accuracy"),  # ratio exactly 1.5: not above the gate
        ((31, 20), "balanced_accuracy"),  # 1.55
        ((100, 10), "balanced_accuracy"),
    ],
)
def test_gate_selects_balanced_accuracy_only_above_the_ratio(counts: tuple[int, ...], expected: str) -> None:
    record = audit(_bundle(counts), class_ratio_gate=1.5, configured_metric="accuracy")
    assert record["class_ratio"] == pytest.approx(max(counts) / min(counts))
    assert record["fitness_metric"] == expected


def test_explicit_metric_overrides_the_gate_only_when_the_gate_is_disabled() -> None:
    imbalanced = _bundle((100, 10))
    assert audit(imbalanced, 1.5, "accuracy")["fitness_metric"] == "balanced_accuracy"
    assert audit(imbalanced, None, "accuracy")["fitness_metric"] == "accuracy"
    balanced = _bundle((50, 50))
    assert audit(balanced, 1.5, "balanced_accuracy")["fitness_metric"] == "balanced_accuracy"
    assert select_fitness_metric(1.5, 1.5, "accuracy") == "accuracy"
    assert select_fitness_metric(1.51, 1.5, "accuracy") == "balanced_accuracy"
    assert select_fitness_metric(9.0, None, "balanced_accuracy") == "balanced_accuracy"


def test_audit_record_fields_missing_values_and_duplicates() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    X[0, 1] = np.nan
    X[1, 1] = np.nan
    y = np.tile([0, 1], 100).astype(np.int64)
    # append an exact copy of row 0 (NaN included) and a relabelled copy of row 5
    X = np.vstack([X, X[0], X[5]])
    y = np.concatenate([y, [y[0], 1 - y[5]]]).astype(np.int64)
    record = audit(DatasetBundle("synthetic", X, y, ["f0", "f1"], ["neg", "pos"], {}))
    assert list(record) == AUDIT_FIELDS
    assert (record["dataset"], record["n_samples"], record["n_features"], record["n_classes"]) == (
        "synthetic",
        202,
        2,
        2,
    )
    assert record["class_counts"] == {"0": int((y == 0).sum()), "1": int((y == 1).sum())}
    assert record["missing_per_feature"] == {"f0": 0, "f1": 3}
    assert record["n_duplicate_rows"] == 1  # NaNs compare equal; the relabelled copy is not a duplicate
    assert record["duplicates_removed"] is False  # 1 of 202 rows is below the 1% threshold
    assert (record["feature_names"], record["class_names"]) == (["f0", "f1"], ["neg", "pos"])


def test_duplicates_above_one_percent_are_never_silently_kept() -> None:
    X = np.vstack([np.arange(20.0).reshape(10, 2)] * 2)
    bundle = DatasetBundle("duplicated", X, np.zeros(20, dtype=np.int64), ["a", "b"], ["only"], {})
    assert count_duplicate_rows(bundle.X, bundle.y) == 10
    with pytest.raises(NotImplementedError, match="DR-005"):
        audit(bundle)


def test_audit_rejects_a_class_without_samples() -> None:
    bundle = DatasetBundle(
        "empty_class", np.zeros((4, 1)), np.zeros(4, dtype=np.int64), ["a"], ["c0", "c1"], {}
    )
    with pytest.raises(ValueError, match="no samples"):
        audit(bundle)


@pytest.mark.parametrize("name", ["iris", "digits", "heart_cleveland"])
def test_no_project_dataset_triggers_the_gate(data_dir: Path, name: str) -> None:
    """CONTEXT §4 V4: every dataset stays at or below the 1.5 ratio, so fitness is accuracy."""
    record = audit(load_dataset(name, data_dir), 1.5, "accuracy")
    assert record["class_ratio"] <= 1.5
    assert record["fitness_metric"] == "accuracy"
    assert record["duplicates_removed"] is False
