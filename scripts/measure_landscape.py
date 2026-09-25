"""Measure the real fitness landscape F_0 on a grid, for the web playground (ADR-027).

For each dataset, evaluates the inner 5-fold CV accuracy of every configuration on the grid
n_estimators ∈ {50, 60, …, 200} × max_depth ∈ {2, …, 20} × min_samples_split ∈ {2, 5, 10}, on outer fold 0's
optimization portion. That is exactly the fitness PSO maximizes in run 0 (same folds, seed, pipeline). The
held-out test fold is never touched. Output: ``results/landscape/<dataset>_fold0.json``.

Usage: python scripts/measure_landscape.py [--datasets iris digits heart_cleveland]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pso_rf.datasets import load_dataset
from pso_rf.datasets.audit import audit
from pso_rf.evaluation import FitnessEvaluator, OptimizationPhase, outer_folds
from pso_rf.experiments.config import load_config
from pso_rf.utils.io import utc_timestamp, write_json_atomic
from pso_rf.utils.manifest import git_state, package_versions

ROOT = Path(__file__).resolve().parents[1]
N_GRID = list(range(50, 201, 10))
D_GRID = list(range(2, 21))
S_GRID = [2, 5, 10]


def measure(name: str) -> Path:
    cfg = load_config([ROOT / "configs" / "default.yaml"])
    settings = cfg.for_dataset(name)
    bundle = load_dataset(name, ROOT / "data")
    metric = audit(bundle, settings.fitness.class_ratio_gate, settings.fitness.metric)["fitness_metric"]
    fold = outer_folds(bundle, cfg.split.outer_folds, cfg.split.outer_seed)[0]
    seed = cfg.split.run_seeds[0]
    evaluator = FitnessEvaluator(
        fold.opt,
        cfg.split.inner_folds,
        seed,
        metric,
        settings.preprocessing,
        n_jobs_folds=settings.fitness.n_jobs_folds,
    )
    start = time.perf_counter()
    values: dict[str, list[list[float]]] = {}
    with OptimizationPhase():  # the test fold stays sealed while measuring
        for s in S_GRID:
            values[str(s)] = [
                [
                    evaluator({"n_estimators": n, "max_depth": d, "min_samples_split": s}).fitness
                    for n in N_GRID
                ]
                for d in D_GRID
            ]
            print(f"{name}: min_samples_split={s} done ({time.perf_counter() - start:.0f}s)", flush=True)
    commit, dirty = git_state(ROOT)
    out = ROOT / "results" / "landscape" / f"{name}_fold0.json"
    write_json_atomic(
        out,
        {
            "dataset": name,
            "description": "Inner 5-fold CV accuracy F_0 on outer fold 0's optimization portion "
            "(never test data).",
            "metric": metric,
            "outer_fold": 0,
            "seed": seed,
            "inner_folds": cfg.split.inner_folds,
            "n_estimators": N_GRID,
            "max_depth": D_GRID,
            "min_samples_split": S_GRID,
            "fitness": values,  # fitness[s][depth_index][n_index]
            "n_opt_samples": len(fold.opt.y),
            "elapsed_s": time.perf_counter() - start,
            "created_at": utc_timestamp(),
            "git_commit": commit,
            "git_dirty": dirty,
            "packages": package_versions(),
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["iris", "heart_cleveland", "digits"])
    for name in parser.parse_args().datasets:
        print(measure(name), flush=True)


if __name__ == "__main__":
    main()
