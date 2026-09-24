"""UT-19: the recorder writes exactly the RESULTS_SCHEMA columns, in order, with the right value formats."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pso_rf.experiments.recorder import Recorder, RunContext, evaluation_columns, iteration_columns
from pso_rf.optimization import Evaluation, IntParam, PSOConfig, PSOOptimizer, RandomSearch, SearchSpace

NAMES = ("n_estimators", "max_depth", "min_samples_split")


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        [
            IntParam("n_estimators", 50, 200),
            IntParam("max_depth", 2, 20),
            IntParam("min_samples_split", 2, 10),
        ]
    )


def objective(config: dict[str, int]) -> Evaluation:
    if config["max_depth"] == 2:
        return Evaluation(
            -math.inf, {"cv_scores": [], "cv_std": math.nan, "status": "failed", "error": "E: x"}
        )
    fitness = 1.0 - abs(config["max_depth"] - 9) / 20
    info = {
        "cv_scores": [fitness, fitness],
        "cv_std": 0.0,
        "diag_balanced_accuracy": fitness,
        "diag_f1_macro": fitness,
        "cache_hit": False,
        "status": "ok",
        "error": None,
        "fit_time_s": 0.01,
    }
    return Evaluation(fitness, info)


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def test_expected_column_lists() -> None:
    cols = evaluation_columns(NAMES)
    assert cols[:9] == [
        "exp_id",
        "dataset",
        "method",
        "run_id",
        "outer_fold",
        "seed",
        "eval_index",
        "iteration",
        "particle_id",
    ]
    assert len(cols) == 32 and cols[-1] == "timestamp"
    assert len(iteration_columns(NAMES)) == 25


def test_pso_recording(tmp_path: Path, space: SearchSpace) -> None:
    context = RunContext("exp", "iris", "pso", "fold_0", 0, 0, "accuracy")
    recorder = Recorder(
        context, NAMES, tmp_path / "evaluations.csv", tmp_path / "iterations.csv", total_iterations=3
    )
    config = PSOConfig(n_particles=5, max_iter=3)
    PSOOptimizer(space, objective, config, np.random.default_rng(0)).run([recorder])

    assert header(tmp_path / "evaluations.csv") == evaluation_columns(NAMES)
    assert header(tmp_path / "iterations.csv") == iteration_columns(NAMES)
    ev = pd.read_csv(tmp_path / "evaluations.csv", keep_default_na=False)
    raw = pd.read_csv(tmp_path / "evaluations.csv", dtype=str, keep_default_na=False)
    it = pd.read_csv(tmp_path / "iterations.csv")
    assert len(ev) == 20 and len(it) == 4
    assert ev["eval_index"].tolist() == list(range(20))
    for column in ("iteration", "particle_id", *NAMES):
        assert pd.api.types.is_integer_dtype(ev[column])
    assert set(raw["cache_hit"]) <= {"true", "false"}
    assert set(raw["status"]) <= {"ok", "failed"}
    assert (ev["fitness_metric"] == "accuracy").all()
    assert ev["timestamp"].str.endswith("Z").all()
    assert it["gbest_improved"].dtype == bool
    assert it["cumulative_evaluations"].tolist() == [5, 10, 15, 20]


def test_failed_rows_and_random_search_rows(tmp_path: Path, space: SearchSpace) -> None:
    context = RunContext("exp", "iris", "random_search", "fold_0", 0, 0, "accuracy")
    recorder = Recorder(context, NAMES, tmp_path / "evaluations.csv", None, flush_every=7)
    RandomSearch(space, objective, 60, np.random.default_rng(1)).run([recorder])
    ev = pd.read_csv(tmp_path / "evaluations.csv", dtype=str, keep_default_na=False)
    assert len(ev) == 60
    pso_only = [
        "iteration",
        "particle_id",
        *(f"pos_{n}" for n in NAMES),
        *(f"vel_{n}" for n in NAMES),
        "pbest_fitness",
        "gbest_fitness",
    ]
    assert (ev[pso_only] == "").all().all()
    failed = ev[ev["status"] == "failed"]
    assert len(failed) > 0
    assert (failed["fitness"] == "-inf").all()
    assert (failed["cv_scores"] == "[]").all() and (failed["cv_std"] == "nan").all()
    assert (failed["error"] == "E: x").all()
    assert not (tmp_path / "iterations.csv").exists()
