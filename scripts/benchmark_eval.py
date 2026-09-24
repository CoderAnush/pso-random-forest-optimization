"""Phase 6 timing gate (ET-06): measure t_eval per dataset and project the full experiment's runtime.

Usage:
    python scripts/benchmark_eval.py [--config CONFIG] [--data-dir DATA_DIR] [--reps REPS]

For each dataset, on the optimization portion of outer fold 0, with the configured inner folds and fold
parallelism: one untimed warm-up evaluation (it starts the loky worker pool), then REPS timed evaluations of
the worst-case configuration (200, 20, 2) and of a mid-range one (125, 11, 6), each on a fresh evaluator
with the cache off; the median is used. Projection per dataset: evaluations = outer folds x (PSO budget +
random-search budget) + deployment budget; upper bound = evaluations x t_worst, typical = evaluations x t_mid.
The baseline's cross-validation and the final refits (a few fits per fold) are not included.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

from pso_rf.datasets import LOADERS, load_dataset
from pso_rf.evaluation import FitnessEvaluator, OptimizationData, outer_folds
from pso_rf.experiments.config import DatasetSettings, ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
WORST = {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2}
MID = {"n_estimators": 125, "max_depth": 11, "min_samples_split": 6}
GATE_SECONDS = 2 * 3600  # ADR-005: keep 5-fold inner CV if the upper bound is at most 2 hours


def _evaluator(
    opt: OptimizationData, config: ExperimentConfig, settings: DatasetSettings
) -> FitnessEvaluator:
    return FitnessEvaluator(
        opt,
        cv_folds=config.split.inner_folds,
        seed=config.split.run_seeds[0],
        metric=settings.fitness.metric,
        preprocessing=settings.preprocessing,
        n_jobs_folds=settings.fitness.n_jobs_folds,
        cache=False,
        rf_n_jobs=config.random_forest.n_jobs,
    )


def time_evaluations(
    opt: OptimizationData, config: ExperimentConfig, settings: DatasetSettings, hyper: dict, reps: int
) -> list[float]:
    """Wall-clock seconds of ``reps`` evaluations of ``hyper``, each on a fresh, cache-off evaluator."""
    times = []
    for _ in range(reps):
        evaluator = _evaluator(opt, config, settings)
        start = time.perf_counter()
        result = evaluator(hyper)
        times.append(time.perf_counter() - start)
        if result.status != "ok":
            raise RuntimeError(f"benchmark evaluation failed: {result.error}")
    return times


def evaluations_per_dataset(config: ExperimentConfig, settings: DatasetSettings) -> int:
    """outer folds x (PSO budget + random-search budget) + deployment budget, for the configured methods."""
    pso_budget = settings.pso.n_particles * (settings.pso.max_iter + 1)
    n_folds = len(config.experiment.folds) if config.experiment.folds else config.split.outer_folds
    methods = config.experiment.methods
    per_fold = pso_budget * ("pso" in methods) + settings.random_search_budget * ("random_search" in methods)
    deployment = pso_budget if config.experiment.deployment_run else 0
    return n_folds * per_fold + deployment


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark, print the table, the projection and the ADR-005 decision."""
    parser = argparse.ArgumentParser(description="Phase 6 timing gate: t_eval and projected runtime.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args(argv)
    config = load_config([args.config])

    print("Timing gate (Phase 6): t_eval = one fitness evaluation on outer fold 0's optimization portion")
    print(
        f"inner folds {config.split.inner_folds}, n_jobs_folds {config.fitness.n_jobs_folds}, "
        f"RF n_jobs {config.random_forest.n_jobs}, median of {args.reps} reps, cache off"
    )
    print(
        f"{platform.platform()}, {os.cpu_count()} CPUs, Python {platform.python_version()}, "
        f"numpy {np.__version__}, scikit-learn {sklearn.__version__}"
    )
    print()
    header = (
        f"{'dataset':<16}{'n_opt':>6}{'t_worst (s)':>13}{'t_mid (s)':>11}{'evals':>7}"
        f"{'upper (min)':>13}{'typical (min)':>15}"
    )
    print(header)
    print("-" * len(header))
    total_upper = total_typical = 0.0
    reps_lines = []
    for name in LOADERS:
        settings = config.for_dataset(name)
        opt = outer_folds(
            load_dataset(name, args.data_dir), config.split.outer_folds, config.split.outer_seed
        )[0].opt
        _evaluator(opt, config, settings)(MID)  # untimed warm-up: starts the worker pool
        worst = time_evaluations(opt, config, settings, WORST, args.reps)
        mid = time_evaluations(opt, config, settings, MID, args.reps)
        t_worst, t_mid = statistics.median(worst), statistics.median(mid)
        evals = evaluations_per_dataset(config, settings)
        upper, typical = evals * t_worst, evals * t_mid
        total_upper += upper
        total_typical += typical
        print(
            f"{name:<16}{len(opt.y):>6}{t_worst:>13.3f}{t_mid:>11.3f}{evals:>7}"
            f"{upper / 60:>13.1f}{typical / 60:>15.1f}"
        )
        worst_text = ", ".join(f"{t:.3f}" for t in worst)
        mid_text = ", ".join(f"{t:.3f}" for t in mid)
        reps_lines.append(f"  {name}: worst {worst_text} s; mid {mid_text} s")
    print("-" * len(header))
    print(f"{'total':<16}{'':>6}{'':>13}{'':>11}{'':>7}{total_upper / 60:>13.1f}{total_typical / 60:>15.1f}")
    print()
    print("individual repetitions:")
    print("\n".join(reps_lines))
    print()
    upper_text = f"{total_upper / 60:.1f} min ({total_upper / 3600:.2f} h)"
    print(f"projected upper bound: {upper_text}; typical: {total_typical / 60:.1f} min")
    if total_upper <= GATE_SECONDS:
        print("decision: upper bound <= 2 h, so keep 5-fold inner CV (ADR-005)")
    else:
        print("decision: upper bound > 2 h, so fall back to 3-fold inner CV and record it in ADR-005")
    return 0


if __name__ == "__main__":
    sys.exit(main())
