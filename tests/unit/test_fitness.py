"""UT-09 (fitness calculation), UT-10 (fitness cache) and UT-12 (failure handling)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import pso_rf.evaluation.fitness as fitness_module
from pso_rf.evaluation import FitnessEvaluator, FoldData
from pso_rf.models import build_model
from pso_rf.preprocessing.pipeline import PreprocessingSpec

SMALL = {"n_estimators": 50, "max_depth": 4, "min_samples_split": 3}
HEART = PreprocessingSpec(impute="most_frequent")
PLAIN = PreprocessingSpec()


def _evaluator(fold: FoldData, spec: PreprocessingSpec, **options: object) -> FitnessEvaluator:
    settings = {"cv_folds": 5, "seed": 0, "metric": "accuracy", "n_jobs_folds": 1, **options}
    return FitnessEvaluator(fold.opt, preprocessing=spec, **settings)  # type: ignore[arg-type]


@pytest.fixture
def fit_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Spy on ``Pipeline.fit``: records the row count of every fit (folds run in-process with n_jobs = 1)."""
    calls: list[int] = []
    original_fit = Pipeline.fit

    def spy_fit(self: Pipeline, X: np.ndarray, y: np.ndarray | None = None, **params: object) -> Pipeline:
        calls.append(len(X))
        return original_fit(self, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    return calls


# ------------------------------------------------------------------------------------------------ UT-09


def test_fitness_equals_the_mean_of_a_manual_cross_val_score(heart_fold0: FoldData) -> None:
    evaluator = _evaluator(heart_fold0, HEART)
    result = evaluator(SMALL)
    manual = cross_val_score(
        build_model(SMALL, 0, HEART), heart_fold0.opt.X, heart_fold0.opt.y, cv=list(evaluator.folds)
    )
    assert result.status == "ok" and result.error is None
    assert result.cv_scores == tuple(manual.tolist())
    assert result.fitness == float(np.mean(manual))
    assert result.cv_std == float(np.std(manual))  # ddof = 0
    assert result.config == SMALL
    assert set(result.diagnostics) == {"balanced_accuracy", "f1_macro"}
    assert result.cache_hit is False and result.fit_time_s > 0


def test_inner_folds_are_drawn_once_from_the_seed(heart_fold0: FoldData) -> None:
    evaluator = _evaluator(heart_fold0, HEART, seed=3)
    before = [(train.copy(), validation.copy()) for train, validation in evaluator.folds]
    evaluator(SMALL)
    evaluator({**SMALL, "max_depth": 6})
    expected = StratifiedKFold(n_splits=5, shuffle=True, random_state=3).split(
        heart_fold0.opt.X, heart_fold0.opt.y
    )
    for (train, validation), (b_train, b_validation), (e_train, e_validation) in zip(
        evaluator.folds, before, expected, strict=True
    ):
        assert np.array_equal(train, b_train) and np.array_equal(
            validation, b_validation
        )  # fixed across calls
        assert np.array_equal(train, e_train) and np.array_equal(validation, e_validation)  # seeded
        assert not train.flags.writeable and not validation.flags.writeable
    validation_rows = np.concatenate([validation for _, validation in evaluator.folds])
    assert np.array_equal(np.sort(validation_rows), np.arange(len(heart_fold0.opt.y)))


def test_the_metric_switch_changes_the_fitness_source(heart_fold0: FoldData) -> None:
    accuracy = _evaluator(heart_fold0, HEART, metric="accuracy")(SMALL)
    balanced_evaluator = _evaluator(heart_fold0, HEART, metric="balanced_accuracy")
    balanced = balanced_evaluator(SMALL)
    manual = cross_val_score(
        build_model(SMALL, 0, HEART),
        heart_fold0.opt.X,
        heart_fold0.opt.y,
        cv=list(balanced_evaluator.folds),
        scoring="balanced_accuracy",
    )
    assert balanced.cv_scores == tuple(manual.tolist())
    assert balanced.fitness == float(np.mean(manual))
    assert (
        balanced.fitness == accuracy.diagnostics["balanced_accuracy"]
    )  # same folds and models, other metric
    with pytest.raises(ValueError, match="metric"):
        _evaluator(heart_fold0, HEART, metric="f1_macro")


# ------------------------------------------------------------------------------------------------ UT-10


def test_a_repeated_configuration_is_served_from_the_cache(
    iris_fold0: FoldData, fit_calls: list[int]
) -> None:
    evaluator = _evaluator(iris_fold0, PLAIN)
    first = evaluator(SMALL)
    assert len(fit_calls) == 5  # one fit per inner fold
    second = evaluator(dict(SMALL))
    assert len(fit_calls) == 5  # no refit
    assert second.cache_hit is True and second.fit_time_s == 0.0
    assert first.cache_hit is False and first.fit_time_s > 0
    same = ("fitness", "cv_scores", "cv_std", "diagnostics", "status", "error", "config")
    assert all(getattr(second, name) == getattr(first, name) for name in same)
    assert evaluator.n_unique_fits == 1

    second.config["n_estimators"] = 999  # a caller mutating a result cannot corrupt the cache
    assert evaluator(SMALL).config == SMALL

    evaluator({**SMALL, "max_depth": 5})
    assert len(fit_calls) == 10 and evaluator.n_unique_fits == 2  # only misses count


def test_the_cache_can_be_disabled(iris_fold0: FoldData, fit_calls: list[int]) -> None:
    evaluator = _evaluator(iris_fold0, PLAIN, cache=False)
    first, second = evaluator(SMALL), evaluator(SMALL)
    assert len(fit_calls) == 10
    assert second.cache_hit is False and second.fitness == first.fitness
    assert evaluator.n_unique_fits == 2


# ------------------------------------------------------------------------------------------------ UT-12


def test_an_invalid_configuration_is_scored_minus_inf(
    iris_fold0: FoldData, caplog: pytest.LogCaptureFixture
) -> None:
    evaluator = _evaluator(iris_fold0, PLAIN)
    invalid = {"n_estimators": 50, "max_depth": 4, "min_samples_split": 1}
    with caplog.at_level(logging.WARNING, logger="pso_rf"):
        result = evaluator(invalid)
    assert result.fitness == -math.inf and result.status == "failed"
    assert result.error is not None and "min_samples_split" in result.error
    assert result.cv_scores == () and result.diagnostics == {} and math.isnan(result.cv_std)
    assert any(
        record.levelno == logging.WARNING and "failed" in record.getMessage() for record in caplog.records
    )

    again = evaluator(invalid)  # failures are cached: the fitness is deterministic
    assert again.cache_hit is True and again.status == "failed" and evaluator.n_unique_fits == 1
    assert evaluator(SMALL).status == "ok"  # the evaluator keeps working after a failure


def test_a_non_finite_score_is_a_failure(iris_fold0: FoldData, monkeypatch: pytest.MonkeyPatch) -> None:
    def nan_cross_validate(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        scores = np.array([0.9, np.nan, 0.9, 0.9, 0.9])
        return {"test_accuracy": scores, "test_balanced_accuracy": scores, "test_f1_macro": scores}

    monkeypatch.setattr(fitness_module, "cross_validate", nan_cross_validate)
    result = _evaluator(iris_fold0, PLAIN)(SMALL)
    assert result.status == "failed" and result.fitness == -math.inf
    assert result.error is not None and "non-finite" in result.error


def test_config_keys_are_checked_before_any_fit(iris_fold0: FoldData, fit_calls: list[int]) -> None:
    evaluator = _evaluator(iris_fold0, PLAIN)
    with pytest.raises(ValueError, match="config keys"):
        evaluator({"n_estimators": 50, "max_depth": 4})
    assert fit_calls == [] and evaluator.n_unique_fits == 0


@pytest.mark.isolation
def test_the_evaluator_refuses_held_out_test_data(iris_fold0: FoldData) -> None:
    with pytest.raises(TypeError, match="OptimizationData"):
        FitnessEvaluator(iris_fold0.test, 5, 0, "accuracy", PLAIN)  # type: ignore[arg-type]
    X_test, y_test = iris_fold0.test.reveal()
    with pytest.raises(TypeError, match="OptimizationData"):
        FitnessEvaluator((X_test, y_test), 5, 0, "accuracy", PLAIN)  # type: ignore[arg-type]
