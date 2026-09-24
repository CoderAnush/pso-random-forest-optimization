"""The fitness evaluator: the measurement block of the closed loop (CONTEXT §12, ADR-005, ADR-016).

A configuration's fitness is the mean score of the configured metric over inner stratified k-fold CV of
the optimization portion. The folds are drawn once per evaluator and stay fixed, so the fitness is
deterministic and the cache is exact. This module never imports ``pso_rf.optimization`` (ADR-023): the
adapter from :class:`FitnessResult` to the optimizer's ``Evaluation`` is
``pso_rf.experiments.runner.make_objective``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from pso_rf.evaluation.splits import HeldOutTestSet, OptimizationData
from pso_rf.models.random_forest import HYPERPARAMETER_NAMES, build_model
from pso_rf.preprocessing.pipeline import PreprocessingSpec

FITNESS_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy")
DIAGNOSTIC_METRICS: tuple[str, ...] = ("balanced_accuracy", "f1_macro")  # logged, never fed back
SCORING: dict[str, Any] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitnessResult:
    """One evaluation of a configuration.

    ``fitness`` is the mean of ``cv_scores`` (the configured metric on each inner fold), and ``cv_std``
    their standard deviation with ddof = 0. ``diagnostics`` holds the mean balanced accuracy and macro F1,
    which are logged but never fed back. A failed evaluation has ``fitness = -inf``, ``status = "failed"``,
    the error text, empty ``cv_scores`` and ``diagnostics``, and ``cv_std = nan``.
    """

    config: dict[str, int | None]
    fitness: float
    cv_scores: tuple[float, ...]
    cv_std: float
    diagnostics: dict[str, float]
    cache_hit: bool
    fit_time_s: float
    status: str
    error: str | None


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _cache_key(config: Mapping[str, int | None]) -> tuple[int | None, ...]:
    if set(config) != set(HYPERPARAMETER_NAMES):
        raise ValueError(f"config keys must be exactly {list(HYPERPARAMETER_NAMES)}, got {sorted(config)}")
    return tuple(config[name] for name in HYPERPARAMETER_NAMES)


class FitnessEvaluator:
    """Scores configurations on one optimization portion with fixed inner folds, a cache and failure handling.

    Only :class:`OptimizationData` is accepted, so held-out test data can never reach the fitness (ADR-024).
    ``n_unique_fits`` counts cache misses (failed evaluations included). ``logger`` may be a run-context
    adapter, so that warnings carry the run prefix.
    """

    def __init__(
        self,
        opt: OptimizationData,
        cv_folds: int,
        seed: int,
        metric: str,
        preprocessing: PreprocessingSpec,
        n_jobs_folds: int = 5,
        cache: bool = True,
        rf_n_jobs: int = 1,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        if isinstance(opt, HeldOutTestSet) or not isinstance(opt, OptimizationData):
            raise TypeError(
                f"FitnessEvaluator accepts only OptimizationData, got {type(opt).__name__}: "
                "held-out test data can never be used for fitness (ADR-004, ADR-024)"
            )
        if metric not in FITNESS_METRICS:
            raise ValueError(f"metric must be one of {list(FITNESS_METRICS)}, got {metric!r}")
        self._opt = opt
        self._seed = seed
        self._metric = metric
        self._preprocessing = preprocessing
        self._n_jobs_folds = n_jobs_folds
        self._use_cache = cache
        self._rf_n_jobs = rf_n_jobs
        self._log = logger if logger is not None else log
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        self._folds = tuple(
            (_read_only(train), _read_only(validation)) for train, validation in splitter.split(opt.X, opt.y)
        )
        self._cache: dict[tuple[int | None, ...], FitnessResult] = {}
        self.n_unique_fits = 0

    @property
    def folds(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """The fixed inner folds as read-only ``(train_index, validation_index)`` pairs."""
        return self._folds

    @property
    def metric(self) -> str:
        """The metric whose inner-CV mean is the fitness."""
        return self._metric

    def __call__(self, config: Mapping[str, int | None]) -> FitnessResult:
        """Fitness of one configuration: from the cache if seen before, otherwise by inner CV."""
        key = _cache_key(config)
        if self._use_cache and key in self._cache:
            cached = self._cache[key]
            return dataclasses.replace(
                cached,
                config=dict(cached.config),
                diagnostics=dict(cached.diagnostics),
                cache_hit=True,
                fit_time_s=0.0,
            )
        result = self._evaluate(dict(config))
        self.n_unique_fits += 1
        if self._use_cache:  # store copies, so that callers cannot alter the cached result
            self._cache[key] = dataclasses.replace(
                result, config=dict(result.config), diagnostics=dict(result.diagnostics)
            )
        return result

    def _evaluate(self, config: dict[str, int | None]) -> FitnessResult:
        start = time.perf_counter()
        try:
            model = build_model(config, self._seed, self._preprocessing, self._rf_n_jobs)
            scores = cross_validate(
                model,
                self._opt.X,
                self._opt.y,
                cv=self._folds,
                scoring=SCORING,
                n_jobs=self._n_jobs_folds,
                error_score="raise",
            )
            fold_scores = np.asarray(scores[f"test_{self._metric}"], dtype=np.float64)
            fitness = float(fold_scores.mean())
            if not math.isfinite(fitness):
                raise ValueError(f"non-finite mean {self._metric} ({fitness})")
        except Exception as exc:  # one failing candidate must not stop the run (NFR-006)
            elapsed = time.perf_counter() - start
            error = f"{type(exc).__name__}: {exc}"
            self._log.warning("evaluation of %s failed and is scored -inf: %s", config, error)
            return FitnessResult(config, -math.inf, (), math.nan, {}, False, elapsed, "failed", error)
        elapsed = time.perf_counter() - start
        return FitnessResult(
            config=config,
            fitness=fitness,
            cv_scores=tuple(float(score) for score in fold_scores),
            cv_std=float(fold_scores.std(ddof=0)),
            diagnostics={name: float(np.mean(scores[f"test_{name}"])) for name in DIAGNOSTIC_METRICS},
            cache_hit=False,
            fit_time_s=elapsed,
            status="ok",
            error=None,
        )
