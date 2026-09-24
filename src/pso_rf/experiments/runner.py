"""The runner: the only place where the optimizer and the Random Forest evaluator are wired together.

For one dataset × outer fold × method, :func:`run_fold`:

1. builds the fitness evaluator on the fold's optimization portion (inner 5-fold CV, fixed folds);
2. opens an :class:`~pso_rf.evaluation.OptimizationPhase` (the test fold is sealed) and runs the method.
   For PSO this is the closed loop: PSO → configuration → RF → validation → fitness → PSO update;
3. closes the phase, refits the chosen configuration on the whole optimization portion and scores the held-out
   test fold exactly once (:func:`~pso_rf.evaluation.final.final_evaluate`);
4. writes ``final.json`` and ``predictions.csv`` (the recorder has written the evaluation logs).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pso_rf.datasets import DatasetBundle
from pso_rf.evaluation import FitnessEvaluator, FitnessResult, FoldData, OptimizationData, OptimizationPhase
from pso_rf.evaluation.final import final_evaluate
from pso_rf.experiments.config import DatasetSettings, ExperimentConfig
from pso_rf.experiments.recorder import Recorder, RunContext
from pso_rf.experiments.seeding import RunSeeds, make_rng
from pso_rf.models.baseline import BASELINE_CONFIG
from pso_rf.optimization import (
    Evaluation,
    IntParam,
    Objective,
    OptimizationResult,
    PSOOptimizer,
    RandomSearch,
    SearchSpace,
)
from pso_rf.utils.io import append_csv_rows, utc_timestamp, write_json_atomic
from pso_rf.utils.log import RunLoggerAdapter

RUN_FILES = ("evaluations.csv", "iterations.csv", "final.json", "predictions.csv", "deployment.json")
PREDICTION_COLUMNS = ("row_index", "y_true", "y_pred")

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------- wiring


def make_objective(evaluator: FitnessEvaluator) -> Objective:
    """Adapter from the evaluator to the optimizer: only ``fitness`` is fed back; the rest is recorded."""

    def objective(config: dict[str, int]) -> Evaluation:
        result: FitnessResult = evaluator(config)
        return Evaluation(
            fitness=result.fitness,
            info={
                "cv_scores": list(result.cv_scores),
                "cv_std": result.cv_std,
                "diag_balanced_accuracy": result.diagnostics.get("balanced_accuracy"),
                "diag_f1_macro": result.diagnostics.get("f1_macro"),
                "cache_hit": result.cache_hit,
                "status": result.status,
                "error": result.error,
                "fit_time_s": result.fit_time_s,
            },
        )

    return objective


def build_search_space(cfg: ExperimentConfig) -> SearchSpace:
    """The configured hyperparameter box, in configuration order."""
    return SearchSpace([IntParam(name, b.low, b.high) for name, b in cfg.search_space.params.items()])


def positive_label_for(bundle: DatasetBundle) -> int | None:
    """Binary datasets report positive-class metrics for label 1 (Heart: disease); others report none."""
    return 1 if len(bundle.class_names) == 2 else None


def baseline_config(cfg: ExperimentConfig) -> dict[str, int | None]:
    """The default-RF configuration (``RandomForestClassifier()`` defaults) plus optional overrides."""
    return {**BASELINE_CONFIG, **cfg.baseline.params}


# ------------------------------------------------------------------------------------------- one run


@dataclass(frozen=True)
class _Outcome:
    """What a method chose, before the test fold is opened."""

    config: dict[str, int | None]
    fitness: float
    cv_scores: list[float]
    result: OptimizationResult | None
    n_unique_fits: int


def run_fold(
    cfg: ExperimentConfig,
    bundle: DatasetBundle,
    fold: FoldData,
    method: str,
    metric: str,
    out_dir: Path,
    exp_id: str,
) -> dict[str, Any]:
    """Run one method on one outer fold and return its ``final.json`` record (also written to ``out_dir``)."""
    start = time.perf_counter()
    settings = cfg.for_dataset(bundle.name)
    k = fold.fold_index
    seeds = RunSeeds.from_run_seed(cfg.split.run_seeds[k])
    out_dir = _fresh_run_dir(out_dir)
    log = RunLoggerAdapter(_log, bundle.name, k, method, seeds.run)
    context = RunContext(exp_id, bundle.name, method, f"fold_{k}", k, seeds.run, metric)

    evaluator = _make_evaluator(cfg, settings, fold.opt, seeds, metric, log)
    with OptimizationPhase():
        outcome = _optimize(method, cfg, settings, evaluator, context, out_dir, seeds, log)
    optimization_finished_at = utc_timestamp()

    final = final_evaluate(
        outcome.config,
        fold.opt,
        fold.test,
        seeds.rf,
        settings.preprocessing,
        labels=list(range(len(bundle.class_names))),
        positive_label=positive_label_for(bundle),
        n_jobs=cfg.random_forest.n_jobs,
    )
    test_evaluated_at = utc_timestamp()
    append_csv_rows(
        out_dir / "predictions.csv",
        (
            {"row_index": int(i), "y_true": int(t), "y_pred": int(p)}
            for i, t, p in zip(final.row_index, final.y_true, final.y_pred, strict=True)
        ),
        PREDICTION_COLUMNS,
    )
    record = {
        **context.ids(),
        **_selection_fields(method, cfg, outcome, metric),
        "test_metrics": final.metrics,
        "n_opt_samples": len(fold.opt.y),
        "n_test_samples": len(fold.test),
        "optimization_finished_at": optimization_finished_at,
        "test_evaluated_at": test_evaluated_at,
        "total_time_s": time.perf_counter() - start,
    }
    write_json_atomic(out_dir / "final.json", record)
    log.info("test accuracy %.4f (evaluated once, after optimization)", final.metrics["accuracy"])
    return record


def run_deployment(
    cfg: ExperimentConfig,
    bundle: DatasetBundle,
    opt: OptimizationData,
    metric: str,
    out_dir: Path,
    exp_id: str,
) -> dict[str, Any]:
    """PSO on the full dataset (no test fold exists); returns the ``deployment.json`` fields it can fill.

    The caller adds ``performance_estimate`` (the outer-CV mean of the PSO method) once the summary exists.
    """
    settings = cfg.for_dataset(bundle.name)
    seeds = RunSeeds.from_run_seed(cfg.split.deployment_seed)
    out_dir = _fresh_run_dir(out_dir)
    log = RunLoggerAdapter(_log, bundle.name, None, "pso", seeds.run)
    context = RunContext(exp_id, bundle.name, "pso", "deployment", None, seeds.run, metric)
    evaluator = _make_evaluator(cfg, settings, opt, seeds, metric, log)
    with OptimizationPhase():
        outcome = _optimize("pso", cfg, settings, evaluator, context, out_dir, seeds, log)
    result = outcome.result
    assert result is not None
    return {
        "exp_id": exp_id,
        "dataset": bundle.name,
        "seed": seeds.run,
        "recommended_hyperparameters": outcome.config,
        "validation_fitness": outcome.fitness,
        "fitness_metric": metric,
        "performance_estimate": None,
        "n_samples": len(opt.y),
        "n_evaluations": result.n_evaluations,
        "n_unique_fits": outcome.n_unique_fits,
        "convergence_iteration": result.convergence_iteration,
        "note": (
            "No held-out test data exists for the deployment run; performance is estimated by the outer "
            "5-fold CV of the same procedure. validation_fitness is optimistically biased."
        ),
    }


# ------------------------------------------------------------------------------------------------ helpers


def _make_evaluator(
    cfg: ExperimentConfig,
    settings: DatasetSettings,
    opt: OptimizationData,
    seeds: RunSeeds,
    metric: str,
    log: logging.LoggerAdapter,
) -> FitnessEvaluator:
    """A fresh evaluator (fresh cache) for one method on one optimization portion."""
    return FitnessEvaluator(
        opt,
        cv_folds=cfg.split.inner_folds,
        seed=seeds.inner_cv,
        metric=metric,
        preprocessing=settings.preprocessing,
        n_jobs_folds=settings.fitness.n_jobs_folds,
        cache=settings.fitness.cache,
        rf_n_jobs=cfg.random_forest.n_jobs,
        logger=log,
    )


def _optimize(
    method: str,
    cfg: ExperimentConfig,
    settings: DatasetSettings,
    evaluator: FitnessEvaluator,
    context: RunContext,
    out_dir: Path,
    seeds: RunSeeds,
    log: logging.LoggerAdapter,
) -> _Outcome:
    """Select a configuration with ``method``; runs inside the optimization phase (test fold sealed)."""
    if method == "baseline":
        config = baseline_config(cfg)
        scored = evaluator(config)
        if scored.status != "ok":
            raise RuntimeError(f"the baseline configuration failed to evaluate: {scored.error}")
        log.info("baseline %s: validation %s %.4f", config, context.fitness_metric, scored.fitness)
        return _Outcome(config, scored.fitness, list(scored.cv_scores), None, evaluator.n_unique_fits)

    space = build_search_space(cfg)
    objective = make_objective(evaluator)
    if method == "pso":
        recorder = Recorder(
            context,
            space.names,
            out_dir / "evaluations.csv",
            out_dir / "iterations.csv",
            logger=log,
            total_iterations=settings.pso.max_iter,
        )
        result = PSOOptimizer(space, objective, settings.pso, make_rng(seeds.pso)).run([recorder])
    elif method == "random_search":
        recorder = Recorder(context, space.names, out_dir / "evaluations.csv", None, logger=log)
        budget = settings.random_search_budget
        result = RandomSearch(space, objective, budget, make_rng(seeds.random_search)).run([recorder])
    else:
        raise ValueError(f"unknown method {method!r}")
    cv_scores = list(result.best_info.get("cv_scores", ()))
    return _Outcome(dict(result.best_config), result.best_fitness, cv_scores, result, evaluator.n_unique_fits)


def _selection_fields(method: str, cfg: ExperimentConfig, outcome: _Outcome, metric: str) -> dict[str, Any]:
    """The ``final.json`` fields describing how the configuration was chosen (RESULTS_SCHEMA §7)."""
    result = outcome.result
    fields: dict[str, Any] = {
        "best_hyperparameters": outcome.config,
        "best_validation_fitness": outcome.fitness,
        "validation_cv_scores": outcome.cv_scores,
        "fitness_metric": metric,
        "best_found_at_eval": None,
        "best_found_at_iteration": None,
        "n_evaluations": 1,
        "n_unique_fits": outcome.n_unique_fits,
        "n_iterations": None,
        "stop_reason": None,
        "convergence_iteration": None,
        "convergence_status": None,
        "n_ties_with_best": None,
        "boundary_hits": [],
    }
    if result is None:  # baseline: sklearn defaults, outside the search space by design
        return fields
    fields.update(
        best_found_at_eval=result.best_found_at_eval,
        best_found_at_iteration=result.best_found_at_iteration,
        n_evaluations=result.n_evaluations,
        n_iterations=result.n_iterations,
        stop_reason=result.stop_reason,
        convergence_iteration=result.convergence_iteration,
        convergence_status=_convergence_status(result),
        n_ties_with_best=result.n_ties_with_best,
        boundary_hits=boundary_hits(outcome.config, cfg),
    )
    return fields


def _convergence_status(result: OptimizationResult) -> str | None:
    """PSO only: ``improving_at_end`` if gbest improved in the last 3 iterations, else ``plateaued``."""
    if result.n_iterations is None or result.convergence_iteration is None:
        return None
    return "improving_at_end" if result.convergence_iteration >= result.n_iterations - 2 else "plateaued"


def boundary_hits(config: Mapping[str, int | None], cfg: ExperimentConfig) -> list[str]:
    """Names of hyperparameters whose chosen value equals a search-space bound."""
    return [
        name
        for name, bounds in cfg.search_space.params.items()
        if config.get(name) in (bounds.low, bounds.high)
    ]


def _fresh_run_dir(path: Path) -> Path:
    """Create ``path``; remove only this module's own output files left by an earlier run there."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (path / name).unlink(missing_ok=True)
    return path
