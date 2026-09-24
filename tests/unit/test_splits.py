"""UT-14 (outer folds) and UT-15 (the held-out test-set guard)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pso_rf.datasets import DatasetBundle
from pso_rf.evaluation import (
    HeldOutTestSet,
    OptimizationData,
    OptimizationPhase,
    TestSetAccessError,
    full_data,
    optimization_phase_active,
    outer_folds,
)


def _imbalanced_bundle() -> DatasetBundle:
    """103 rows, 3 classes of 50 / 33 / 20: fold sizes and class shares do not divide evenly."""
    y = np.repeat([0, 1, 2], [50, 33, 20]).astype(np.int64)
    X = np.random.default_rng(7).normal(size=(len(y), 2))
    return DatasetBundle("imbalanced", X, y, ["a", "b"], ["c0", "c1", "c2"], {})


# ------------------------------------------------------------------------------------------------ UT-14


@pytest.mark.parametrize("bundle_name", ["synthetic", "iris", "heart"])
def test_outer_folds_partition_the_rows_with_stratification(
    bundle_name: str, iris_bundle: DatasetBundle, heart_bundle: DatasetBundle
) -> None:
    bundle = {"synthetic": _imbalanced_bundle(), "iris": iris_bundle, "heart": heart_bundle}[bundle_name]
    n = len(bundle.y)
    folds = outer_folds(bundle, n_splits=5, seed=42)
    assert [fold.fold_index for fold in folds] == [0, 1, 2, 3, 4]
    test_sets = [set(fold.test.indices.tolist()) for fold in folds]
    for a, b in itertools.combinations(test_sets, 2):
        assert not a & b  # pairwise disjoint
    assert set().union(*test_sets) == set(range(n))  # every row is tested exactly once
    class_totals = np.bincount(bundle.y)
    for fold in folds:
        assert not set(fold.opt.indices.tolist()) & set(fold.test.indices.tolist())
        assert len(fold.opt.indices) + len(fold.test) == n
        _, y_test = fold.test.reveal()
        counts = np.bincount(y_test, minlength=len(class_totals))
        proportional = class_totals * len(fold.test) / n
        assert np.all(np.abs(counts - proportional) <= 1), (counts, proportional)


def test_outer_folds_are_deterministic_per_seed(heart_bundle: DatasetBundle) -> None:
    first = outer_folds(heart_bundle, 5, 42)
    second = outer_folds(heart_bundle, 5, 42)
    other = outer_folds(heart_bundle, 5, 43)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.test.indices, b.test.indices)
        assert np.array_equal(a.opt.indices, b.opt.indices)
    assert any(not np.array_equal(a.test.indices, c.test.indices) for a, c in zip(first, other, strict=True))


def test_fold_arrays_are_the_bundle_rows(heart_bundle: DatasetBundle) -> None:
    fold = outer_folds(heart_bundle, 5, 42)[0]
    np.testing.assert_array_equal(fold.opt.X, heart_bundle.X[fold.opt.indices])  # NaNs compare equal here
    np.testing.assert_array_equal(fold.opt.y, heart_bundle.y[fold.opt.indices])
    X_test, y_test = fold.test.reveal()
    np.testing.assert_array_equal(X_test, heart_bundle.X[fold.test.indices])
    np.testing.assert_array_equal(y_test, heart_bundle.y[fold.test.indices])


def test_optimization_data_is_read_only_and_full_data_covers_every_row(iris_bundle: DatasetBundle) -> None:
    data = full_data(iris_bundle)
    assert len(data.y) == len(iris_bundle.y)
    np.testing.assert_array_equal(data.indices, np.arange(len(iris_bundle.y)))
    assert data.indices.dtype == np.int64
    for array in (data.X, data.y, data.indices):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        data.X[0, 0] = 1.0
    assert iris_bundle.X.flags.writeable  # the bundle itself is left untouched
    with pytest.raises(ValueError):
        OptimizationData(np.zeros((3, 2)), np.zeros(2, dtype=np.int64), np.arange(3))


# ------------------------------------------------------------------------------------------------ UT-15


@pytest.fixture
def sealed() -> HeldOutTestSet:
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    return HeldOutTestSet(X, y, np.array([10, 11, 12, 13, 14, 15]))


@pytest.mark.isolation
def test_reveal_raises_inside_the_optimization_phase(sealed: HeldOutTestSet) -> None:
    assert not optimization_phase_active()
    with OptimizationPhase():
        assert optimization_phase_active()
        with pytest.raises(TestSetAccessError):
            sealed.reveal()
    X, y = sealed.reveal()  # allowed once the phase is closed
    np.testing.assert_array_equal(X, np.arange(12.0).reshape(6, 2))
    np.testing.assert_array_equal(y, [0, 1, 0, 1, 0, 1])


@pytest.mark.isolation
def test_reveal_raises_in_nested_phases_until_the_outermost_closes(sealed: HeldOutTestSet) -> None:
    with OptimizationPhase():
        with OptimizationPhase():
            with pytest.raises(TestSetAccessError):
                sealed.reveal()
        assert optimization_phase_active()  # still inside the outer phase
        with pytest.raises(TestSetAccessError):
            sealed.reveal()
    assert not optimization_phase_active()
    sealed.reveal()


@pytest.mark.isolation
def test_phase_state_is_restored_after_an_exception(sealed: HeldOutTestSet) -> None:
    with pytest.raises(ZeroDivisionError):
        with OptimizationPhase():
            with OptimizationPhase():
                _ = 1 / 0
    assert not optimization_phase_active()
    sealed.reveal()


@pytest.mark.isolation
def test_test_arrays_are_reachable_only_through_reveal(sealed: HeldOutTestSet) -> None:
    public_arrays = [
        name
        for name in dir(sealed)
        if not name.startswith("_") and isinstance(getattr(sealed, name), np.ndarray)
    ]
    assert public_arrays == ["indices"]
    assert not sealed.indices.flags.writeable
    with pytest.raises(AttributeError):
        sealed.indices = np.arange(6)  # type: ignore[misc]
    assert len(sealed) == 6
    assert repr(sealed) == "HeldOutTestSet(n=6, sealed)"


@pytest.mark.isolation
def test_reveal_returns_copies_that_cannot_alter_the_sealed_data(sealed: HeldOutTestSet) -> None:
    X, y = sealed.reveal()
    X[:] = -1.0
    y[:] = 9
    X_again, y_again = sealed.reveal()
    np.testing.assert_array_equal(X_again, np.arange(12.0).reshape(6, 2))
    np.testing.assert_array_equal(y_again, [0, 1, 0, 1, 0, 1])
