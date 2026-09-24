"""UT-20: preprocessing leakage. The imputer is a pipeline step that learns only from its fit rows."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from pso_rf.datasets import DatasetBundle
from pso_rf.preprocessing.pipeline import PreprocessingSpec, build_steps

HEART_SPEC = PreprocessingSpec(impute="most_frequent")
N_TRAIN = 200  # rows 0..199 play the training part, rows 200..302 the held-out part


def _column_modes(X: np.ndarray) -> np.ndarray:
    """Most frequent non-NaN value per column; ties go to the smallest value, as in SimpleImputer."""
    modes = []
    for column in X.T:
        values, counts = np.unique(column[~np.isnan(column)], return_counts=True)
        modes.append(values[counts == counts.max()].min())
    return np.array(modes)


def _pipeline(spec: PreprocessingSpec) -> Pipeline:
    return Pipeline([*build_steps(spec), ("clf", DummyClassifier())])


def test_imputer_statistics_come_from_the_training_rows_only(heart_bundle: DatasetBundle) -> None:
    X_train, y_train = heart_bundle.X[:N_TRAIN], heart_bundle.y[:N_TRAIN]
    X_test = heart_bundle.X[N_TRAIN:]
    assert np.isnan(X_train).any() and np.isnan(X_test).any()  # both parts contain missing cells
    fitted = _pipeline(HEART_SPEC).fit(X_train, y_train)
    np.testing.assert_array_equal(fitted.named_steps["impute"].statistics_, _column_modes(X_train))


def test_distinctive_values_in_the_test_rows_do_not_reach_the_statistics(heart_bundle: DatasetBundle) -> None:
    X_train, y_train = heart_bundle.X[:N_TRAIN], heart_bundle.y[:N_TRAIN]
    columns = [heart_bundle.feature_names.index("ca"), heart_bundle.feature_names.index("thal")]
    poisoned = heart_bundle.X[N_TRAIN:].copy()
    poisoned[:, columns] = 99.0  # a sentinel that never occurs in the training rows
    assert not (X_train[:, columns] == 99.0).any()

    fitted = _pipeline(HEART_SPEC).fit(X_train, y_train)
    fitted.predict(poisoned)  # scoring the held-out rows must not refit anything
    statistics = fitted.named_steps["impute"].statistics_
    np.testing.assert_array_equal(statistics, _column_modes(X_train))
    assert 99.0 not in statistics

    # Control: had those rows been part of the fit, the sentinel would have become the statistic.
    leaky = SimpleImputer(strategy="most_frequent").fit(np.vstack([X_train, poisoned, poisoned]))
    assert (leaky.statistics_[columns] == 99.0).all()


def test_imputer_is_refit_on_each_cv_training_part(heart_bundle: DatasetBundle) -> None:
    X, y = heart_bundle.X, heart_bundle.y
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    result = cross_validate(_pipeline(HEART_SPEC), X, y, cv=folds, return_estimator=True)
    for (train_index, _), estimator in zip(folds, result["estimator"], strict=True):
        np.testing.assert_array_equal(
            estimator.named_steps["impute"].statistics_, _column_modes(X[train_index])
        )


def test_heart_pipeline_fits_and_predicts_with_nans_present(heart_bundle: DatasetBundle) -> None:
    X_train, y_train = heart_bundle.X[:N_TRAIN], heart_bundle.y[:N_TRAIN]
    X_test = heart_bundle.X[N_TRAIN:]
    model = Pipeline(
        [*build_steps(HEART_SPEC), ("rf", RandomForestClassifier(n_estimators=10, random_state=0))]
    )
    model.fit(X_train, y_train)
    assert not np.isnan(model[:-1].transform(X_test)).any()  # the forest never sees a NaN
    assert model.predict(X_test).shape == (len(X_test),)


def test_no_steps_without_imputation_and_a_no_op_on_complete_data(iris_bundle: DatasetBundle) -> None:
    assert build_steps(PreprocessingSpec()) == []
    assert not np.isnan(iris_bundle.X).any()
    imputer = build_steps(HEART_SPEC)[0][1].fit(iris_bundle.X)
    np.testing.assert_array_equal(imputer.transform(iris_bundle.X), iris_bundle.X)


def test_build_steps_returns_fresh_unfitted_steps() -> None:
    first, second = build_steps(HEART_SPEC), build_steps(HEART_SPEC)
    assert [name for name, _ in first] == ["impute"]
    assert first[0][1] is not second[0][1]
    assert first[0][1].get_params()["strategy"] == "most_frequent"
    assert not hasattr(first[0][1], "statistics_")


def test_unknown_imputation_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="impute must be"):
        PreprocessingSpec(impute="constant")
