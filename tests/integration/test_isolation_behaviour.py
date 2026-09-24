"""Test-data isolation proofs IT-05 (row-hash spy), IT-06 (label invariance) and IT-07 (feature invariance).

Together with IT-04 (indices), UT-15 (sealed test set) and UT-22 (layering) they show that test data never
enters the PSO fitness loop.
"""

from __future__ import annotations

import functools
import hashlib
from collections import Counter

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from pso_rf.evaluation import optimization_phase_active
from pso_rf.experiments.runner import run_fold
from tests.integration.helpers import read_json, read_rows, with_test_data

pytestmark = pytest.mark.isolation


def row_hashes(X) -> list[str]:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in X]


@pytest.mark.parametrize("dataset", ["iris", "heart_cleveland"])
def test_it05_spy_sees_no_test_row_during_optimization(
    dataset, request, tmp_path, test_config, monkeypatch
) -> None:
    bundle = request.getfixturevalue(f"{'heart' if dataset == 'heart_cleveland' else 'iris'}_bundle")
    fold = request.getfixturevalue(f"{'heart' if dataset == 'heart_cleveland' else 'iris'}_fold0")
    seen = {"inside": [], "fit_after": [], "predict_after": []}
    original_fit, original_predict = Pipeline.fit, Pipeline.predict

    @functools.wraps(original_fit)
    def spy_fit(self, X, y=None, **params):
        seen["inside" if optimization_phase_active() else "fit_after"].append(row_hashes(X))
        return original_fit(self, X, y, **params)

    @functools.wraps(original_predict)
    def spy_predict(self, X, **params):
        seen["inside" if optimization_phase_active() else "predict_after"].append(row_hashes(X))
        return original_predict(self, X, **params)

    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    monkeypatch.setattr(Pipeline, "predict", spy_predict)
    assert test_config.fitness.n_jobs_folds == 1  # folds run in-process, so the spy sees every call

    for method in ("baseline", "random_search", "pso"):
        seen["fit_after"].clear()
        seen["predict_after"].clear()
        run_fold(test_config, bundle, fold, method, "accuracy", tmp_path / method, "it05")
        X_test, _ = fold.test.reveal()
        # after the phase: exactly one refit on the optimization rows, one prediction on the test rows
        assert [Counter(h) for h in seen["fit_after"]] == [Counter(row_hashes(fold.opt.X))]
        assert [Counter(h) for h in seen["predict_after"]] == [Counter(row_hashes(X_test))]

    X_test, _ = fold.test.reveal()
    test_only = set(row_hashes(X_test)) - set(row_hashes(fold.opt.X))  # rows duplicated across folds excluded
    inside = {h for call in seen["inside"] for h in call}
    assert inside, "the spy recorded nothing during optimization"
    assert test_only, "no test-only rows to check"
    assert inside.isdisjoint(test_only)


def _trajectory(path):
    return read_rows(path / "evaluations.csv"), read_rows(path / "iterations.csv")


def test_it06_test_label_invariance(tmp_path, test_config, iris_bundle, iris_fold0) -> None:
    _, y_test = iris_fold0.test.reveal()
    changed = with_test_data(iris_fold0, y=(y_test + 1) % 3)  # every label different
    a = run_fold(test_config, iris_bundle, iris_fold0, "pso", "accuracy", tmp_path / "a", "it06")
    b = run_fold(test_config, iris_bundle, changed, "pso", "accuracy", tmp_path / "b", "it06")
    assert _trajectory(tmp_path / "a") == _trajectory(tmp_path / "b")
    assert a["best_hyperparameters"] == b["best_hyperparameters"]
    pa, pb = read_rows(tmp_path / "a" / "predictions.csv"), read_rows(tmp_path / "b" / "predictions.csv")
    assert [r["y_pred"] for r in pa] == [r["y_pred"] for r in pb]  # the same model
    assert all(ra["y_true"] != rb["y_true"] for ra, rb in zip(pa, pb, strict=True))
    assert a["test_metrics"]["accuracy"] != b["test_metrics"]["accuracy"]


def test_it07_test_feature_invariance(tmp_path, test_config, heart_bundle, heart_fold0) -> None:
    X_test, _ = heart_fold0.test.reveal()
    noise = np.random.default_rng(7).normal(size=X_test.shape)
    changed = with_test_data(heart_fold0, X=noise)
    a = run_fold(test_config, heart_bundle, heart_fold0, "pso", "accuracy", tmp_path / "a", "it07")
    b = run_fold(test_config, heart_bundle, changed, "pso", "accuracy", tmp_path / "b", "it07")
    assert _trajectory(tmp_path / "a") == _trajectory(tmp_path / "b")
    assert a["best_hyperparameters"] == b["best_hyperparameters"]
    assert read_json(tmp_path / "a" / "final.json")["best_validation_fitness"] == b["best_validation_fitness"]
