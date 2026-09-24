"""IT-08 (partial, Phase 6): running the inner folds in parallel gives exactly the serial scores (NFR-007)."""

from __future__ import annotations

import pytest

from pso_rf.evaluation import FitnessEvaluator, FoldData
from pso_rf.preprocessing.pipeline import PreprocessingSpec

CONFIGS = [
    {"n_estimators": 60, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 90, "max_depth": 12, "min_samples_split": 7},
]


@pytest.mark.parametrize(
    ("fold_fixture", "spec"),
    [("iris_fold0", PreprocessingSpec()), ("heart_fold0", PreprocessingSpec(impute="most_frequent"))],
)
@pytest.mark.parametrize("config", CONFIGS)
def test_parallel_and_serial_folds_give_identical_scores(
    request: pytest.FixtureRequest, fold_fixture: str, spec: PreprocessingSpec, config: dict
) -> None:
    fold: FoldData = request.getfixturevalue(fold_fixture)
    serial = FitnessEvaluator(fold.opt, 5, 0, "accuracy", spec, n_jobs_folds=1)(config)
    parallel = FitnessEvaluator(fold.opt, 5, 0, "accuracy", spec, n_jobs_folds=5)(config)
    assert serial.status == parallel.status == "ok"
    assert parallel.cv_scores == serial.cv_scores
    assert parallel.fitness == serial.fitness
    assert parallel.diagnostics == serial.diagnostics
