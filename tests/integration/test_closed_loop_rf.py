"""IT-03 (real RF closed loop), IT-10 (baseline), IT-12 (convergence tracking), IT-13 (random search)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pso_rf.evaluation import FitnessEvaluator
from pso_rf.experiments.config import load_config
from pso_rf.experiments.runner import run_fold
from pso_rf.models.baseline import BASELINE_CONFIG
from pso_rf.models.random_forest import build_model
from tests.integration.helpers import read_json, read_rows

pytestmark = pytest.mark.closed_loop

NAMES = ("n_estimators", "max_depth", "min_samples_split")
BOUNDS = {"n_estimators": (50, 200), "max_depth": (2, 20), "min_samples_split": (2, 10)}


def test_it03_pso_closed_loop_with_random_forest(
    tmp_path, test_config, iris_bundle, iris_fold0, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="pso_rf")
    record = run_fold(test_config, iris_bundle, iris_fold0, "pso", "accuracy", tmp_path, "it03")
    n, t = test_config.pso.n_particles, test_config.pso.max_iter
    evaluations = read_rows(tmp_path / "evaluations.csv")
    iterations = read_rows(tmp_path / "iterations.csv")
    assert len(evaluations) == n * (t + 1)
    assert len(iterations) == t + 1
    for row in evaluations:
        for name in NAMES:
            low, high = BOUNDS[name]
            assert low <= int(row[name]) <= high
        assert 0.0 <= float(row["fitness"]) <= 1.0
    assert (tmp_path / "final.json").exists() and (tmp_path / "predictions.csv").exists()
    assert set(record["test_metrics"]) >= {"accuracy", "f1_macro", "confusion_matrix"}
    assert record["n_evaluations"] == n * (t + 1)
    trace = [r.getMessage() for r in caplog.records if " iter " in r.getMessage()]
    assert len(trace) == t + 1
    assert all(line.startswith("[iris|fold 0|pso|seed 0]") for line in trace)


def test_it10_baseline_protocol(tmp_path, test_config, heart_bundle, heart_fold0) -> None:
    record = run_fold(test_config, heart_bundle, heart_fold0, "baseline", "accuracy", tmp_path, "it10")
    settings = test_config.for_dataset("heart_cleveland")
    evaluator = FitnessEvaluator(
        heart_fold0.opt, test_config.split.inner_folds, 0, "accuracy", settings.preprocessing, n_jobs_folds=1
    )
    assert record["best_hyperparameters"] == dict(BASELINE_CONFIG)
    assert record["best_validation_fitness"] == evaluator(dict(BASELINE_CONFIG)).fitness
    assert record["n_evaluations"] == 1 and record["boundary_hits"] == []
    # final model = refit on the whole optimization portion with random_state = run seed
    model = build_model(dict(BASELINE_CONFIG), 0, settings.preprocessing).fit(
        heart_fold0.opt.X, heart_fold0.opt.y
    )
    X_test, _ = heart_fold0.test.reveal()
    predictions = read_rows(tmp_path / "predictions.csv")
    assert [int(r["y_pred"]) for r in predictions] == model.predict(X_test).tolist()
    assert [int(r["row_index"]) for r in predictions] == heart_fold0.test.indices.tolist()
    assert record["test_metrics"]["positive_class_recall"] is not None  # binary: disease = 1


def test_it12_convergence_tracking(tmp_path, configs_dir, iris_bundle, iris_fold0) -> None:
    cfg = load_config([configs_dir / "default.yaml", configs_dir / "test.yaml"], ["pso.max_iter=8"])
    record = run_fold(cfg, iris_bundle, iris_fold0, "pso", "accuracy", tmp_path, "it12")
    evaluations = read_rows(tmp_path / "evaluations.csv")
    iterations = read_rows(tmp_path / "iterations.csv")
    n = cfg.pso.n_particles
    running = -np.inf
    for row in evaluations:
        running = max(running, float(row["fitness"]))
        assert float(row["best_so_far_fitness"]) == running
    unique = 0
    for it in iterations:
        t = int(it["iteration"])
        upto = [r for r in evaluations if int(r["iteration"]) <= t]
        assert float(it["gbest_fitness"]) == max(float(r["fitness"]) for r in upto)
        unique = sum(r["cache_hit"] == "false" for r in upto)
        assert int(it["cumulative_unique_fits"]) == unique
        assert int(it["cumulative_evaluations"]) == n * (t + 1)
        this = [r for r in evaluations if int(r["iteration"]) == t]
        assert int(it["n_cache_hits"]) == sum(r["cache_hit"] == "true" for r in this)
    last_improved = max(int(it["iteration"]) for it in iterations if it["gbest_improved"] == "true")
    assert record["convergence_iteration"] == last_improved
    assert record["n_unique_fits"] == unique
    assert record["convergence_status"] in {"plateaued", "improving_at_end"}


def test_it13_random_search_through_runner(tmp_path, test_config, iris_bundle, iris_fold0) -> None:
    record = run_fold(test_config, iris_bundle, iris_fold0, "random_search", "accuracy", tmp_path, "it13")
    budget = test_config.for_dataset("iris").random_search_budget
    evaluations = read_rows(tmp_path / "evaluations.csv")
    assert len(evaluations) == budget == record["n_evaluations"]
    for row in evaluations:
        assert row["iteration"] == row["particle_id"] == row["pbest_fitness"] == row["gbest_fitness"] == ""
        assert all(row[f"pos_{name}"] == "" and row[f"vel_{name}"] == "" for name in NAMES)
    assert not (tmp_path / "iterations.csv").exists()
    final = read_json(tmp_path / "final.json")
    assert final["stop_reason"] == "budget" and final["n_iterations"] is None
    assert final["convergence_status"] is None
    assert final["method"] == "random_search"
