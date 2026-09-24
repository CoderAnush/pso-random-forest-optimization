"""The recorder: optimizer events → ``evaluations.csv`` / ``iterations.csv`` and the live trace.

Columns follow RESULTS_SCHEMA §5–§6.

Rows are buffered and flushed at every PSO iteration end (every ``flush_every`` evaluations for random search,
and at the end), so a crashed run keeps its history. Recording never influences the search: the recorder
only reads events.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pso_rf.optimization.events import Callback, EvaluationEvent, IterationSummary, OptimizationResult
from pso_rf.utils.io import append_csv_rows, utc_timestamp

ID_COLUMNS: tuple[str, ...] = ("exp_id", "dataset", "method", "run_id", "outer_fold", "seed")
TIMING_COLUMNS: frozenset[str] = frozenset({"timestamp", "fit_time_s", "elapsed_s"})


def evaluation_columns(names: Sequence[str]) -> list[str]:
    """``evaluations.csv`` columns in schema order (§5) for hyperparameters ``names``."""
    return [
        *ID_COLUMNS,
        "eval_index",
        "iteration",
        "particle_id",
        *(f"pos_{name}" for name in names),
        *(f"vel_{name}" for name in names),
        *names,
        "fitness",
        "fitness_metric",
        "cv_scores",
        "cv_std",
        "diag_balanced_accuracy",
        "diag_f1_macro",
        "cache_hit",
        "status",
        "error",
        "pbest_fitness",
        "gbest_fitness",
        "best_so_far_fitness",
        "fit_time_s",
        "timestamp",
    ]


def iteration_columns(names: Sequence[str]) -> list[str]:
    """``iterations.csv`` columns in schema order (§6) for hyperparameters ``names``."""
    return [
        *ID_COLUMNS,
        "iteration",
        "gbest_fitness",
        *(f"gbest_{name}" for name in names),
        "gbest_improved",
        "mean_fitness",
        "std_fitness",
        "min_fitness",
        "max_fitness",
        "diversity",
        "n_unique_configs",
        "n_cache_hits",
        "n_ties_with_gbest",
        "n_failed",
        "cumulative_evaluations",
        "cumulative_unique_fits",
        "no_improve_count",
        "elapsed_s",
    ]


@dataclass(frozen=True)
class RunContext:
    """Identification columns shared by every row of one run."""

    exp_id: str
    dataset: str
    method: str
    run_id: str
    outer_fold: int | None
    seed: int
    fitness_metric: str

    def ids(self) -> dict[str, Any]:
        """The six identification columns."""
        return {
            "exp_id": self.exp_id,
            "dataset": self.dataset,
            "method": self.method,
            "run_id": self.run_id,
            "outer_fold": self.outer_fold,
            "seed": self.seed,
        }


class Recorder(Callback):
    """Writes one CSV row per evaluation (and per PSO iteration) and logs the per-iteration trace line."""

    def __init__(
        self,
        context: RunContext,
        names: Sequence[str],
        evaluations_path: Path,
        iterations_path: Path | None = None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
        total_iterations: int | None = None,
        flush_every: int = 10,
    ) -> None:
        self.context = context
        self.names = tuple(names)
        self.evaluations_path = Path(evaluations_path)
        self.iterations_path = None if iterations_path is None else Path(iterations_path)
        self.log = logger if logger is not None else logging.getLogger(__name__)
        self.total_iterations = total_iterations
        self.flush_every = flush_every
        self._eval_columns = evaluation_columns(self.names)
        self._iter_columns = iteration_columns(self.names)
        self._buffer: list[dict[str, Any]] = []
        self._iteration_cache_hits = 0
        self.unique_fits = 0

    # ------------------------------------------------------------------------------------------- callbacks

    def on_evaluation(self, event: EvaluationEvent) -> None:
        """Buffer the event as an ``evaluations.csv`` row."""
        info = event.info or {}
        cache_hit = bool(info.get("cache_hit", False))
        if cache_hit:
            self._iteration_cache_hits += 1
        else:
            self.unique_fits += 1
        row = self.context.ids()
        row.update(
            eval_index=event.eval_index,
            iteration=event.iteration,
            particle_id=event.particle_id,
        )
        for d, name in enumerate(self.names):
            row[f"pos_{name}"] = None if event.position is None else event.position[d]
            row[f"vel_{name}"] = None if event.velocity is None else event.velocity[d]
        for name in self.names:
            row[name] = event.config[name]
        cv_std = info.get("cv_std")
        row.update(
            fitness=event.fitness,
            fitness_metric=self.context.fitness_metric,
            cv_scores=list(info.get("cv_scores", ())),
            cv_std=math.nan if cv_std is None else cv_std,
            diag_balanced_accuracy=info.get("diag_balanced_accuracy"),
            diag_f1_macro=info.get("diag_f1_macro"),
            cache_hit=cache_hit,
            status=info.get("status", "ok"),
            error=info.get("error"),
            pbest_fitness=event.pbest_fitness,
            gbest_fitness=event.gbest_fitness,
            best_so_far_fitness=event.best_so_far_fitness,
            fit_time_s=float(info.get("fit_time_s", 0.0)),
            timestamp=utc_timestamp(),
        )
        self._buffer.append(row)
        if event.iteration is None and len(self._buffer) >= self.flush_every:
            self.flush()

    def on_iteration_end(self, summary: IterationSummary) -> None:
        """Flush this iteration's rows, append its ``iterations.csv`` row and log the trace line."""
        self.flush()
        if self.iterations_path is not None:
            row = self.context.ids()
            row.update(iteration=summary.iteration, gbest_fitness=summary.gbest_fitness)
            for name in self.names:
                row[f"gbest_{name}"] = summary.gbest_config.get(name)
            row.update(
                gbest_improved=summary.gbest_improved,
                mean_fitness=summary.mean_fitness,
                std_fitness=summary.std_fitness,
                min_fitness=summary.min_fitness,
                max_fitness=summary.max_fitness,
                diversity=summary.diversity,
                n_unique_configs=summary.n_unique_configs,
                n_cache_hits=self._iteration_cache_hits,
                n_ties_with_gbest=summary.n_ties_with_gbest,
                n_failed=summary.n_failed,
                cumulative_evaluations=summary.cumulative_evaluations,
                cumulative_unique_fits=self.unique_fits,
                no_improve_count=summary.no_improve_count,
                elapsed_s=summary.elapsed_s,
            )
            append_csv_rows(self.iterations_path, [row], self._iter_columns)
        total = "?" if self.total_iterations is None else self.total_iterations
        config = ",".join(str(summary.gbest_config.get(name)) for name in self.names)
        self.log.info(
            "iter %d/%s  gbest (%s)=%.4f  mean=%.4f  unique fits %d%s",
            summary.iteration,
            total,
            config,
            summary.gbest_fitness,
            summary.mean_fitness,
            self.unique_fits,
            "  *" if summary.gbest_improved else "",
        )
        self._iteration_cache_hits = 0

    def on_finish(self, result: OptimizationResult) -> None:
        """Flush any remaining rows (random search) and log the outcome."""
        self.flush()
        config = ",".join(str(result.best_config.get(name)) for name in self.names)
        self.log.info(
            "done: best (%s)=%.4f after %d evaluations, %d unique fits",
            config,
            result.best_fitness,
            result.n_evaluations,
            self.unique_fits,
        )

    def flush(self) -> None:
        """Append buffered evaluation rows to ``evaluations.csv``."""
        if self._buffer:
            append_csv_rows(self.evaluations_path, self._buffer, self._eval_columns)
            self._buffer = []
