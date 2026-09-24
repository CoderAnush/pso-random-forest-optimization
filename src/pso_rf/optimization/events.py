"""What an objective returns, what optimizers emit, and what they return (CONTEXT §9).

The optimizer uses only ``Evaluation.fitness``. ``Evaluation.info`` (fold scores, cache flag, timing …) is
forwarded untouched to callbacks, so recording never changes the search.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evaluation:
    """The feedback for one configuration: ``fitness`` (fed back) and ``info`` (recorded only)."""

    fitness: float
    info: Mapping[str, Any] = field(default_factory=dict)


Objective = Callable[[dict[str, int]], "Evaluation | float"]


def as_evaluation(value: Evaluation | float) -> Evaluation:
    """Accept an ``Evaluation`` or a plain number; NaN becomes ``-inf`` (a failed evaluation)."""
    if isinstance(value, Evaluation):
        evaluation = value
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        evaluation = Evaluation(float(value))
    else:
        raise TypeError(f"an objective must return an Evaluation or a number, got {type(value).__name__}")
    fitness = float(evaluation.fitness)
    if math.isnan(fitness):
        return Evaluation(-math.inf, evaluation.info)
    return evaluation if fitness == evaluation.fitness else Evaluation(fitness, evaluation.info)


@dataclass(frozen=True)
class EvaluationEvent:
    """One proposed configuration and its feedback, in evaluation order.

    PSO fills ``iteration``, ``particle_id``, ``position``, ``velocity`` (the velocity with which the particle
    arrived; the initial velocity at t = 0) and ``pbest_fitness`` / ``gbest_fitness`` (after this iteration's
    memory update). Random search leaves them ``None``. ``best_so_far_fitness`` is the running maximum of
    fitness over evaluations up to and including this one.
    """

    eval_index: int
    iteration: int | None
    particle_id: int | None
    position: tuple[float, ...] | None
    velocity: tuple[float, ...] | None
    config: dict[str, int]
    fitness: float
    info: Mapping[str, Any]
    pbest_fitness: float | None
    gbest_fitness: float | None
    best_so_far_fitness: float


@dataclass(frozen=True)
class IterationSummary:
    """Swarm statistics at the end of iteration ``t`` (after the memory update)."""

    iteration: int
    gbest_fitness: float
    gbest_config: dict[str, int]
    gbest_improved: bool
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    diversity: float
    n_unique_configs: int
    n_ties_with_gbest: int
    n_failed: int
    cumulative_evaluations: int
    no_improve_count: int
    elapsed_s: float


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of one optimizer run; ``best_info`` is the ``info`` of the evaluation that found it."""

    method: str
    best_config: dict[str, int]
    best_fitness: float
    best_position: tuple[float, ...] | None
    best_info: Mapping[str, Any]
    best_found_at_eval: int
    best_found_at_iteration: int | None
    n_evaluations: int
    n_iterations: int | None
    stop_reason: str
    convergence_iteration: int | None
    n_ties_with_best: int
    history: tuple[IterationSummary, ...] = ()


class Callback:
    """Receives optimizer events; every hook is optional and defaults to doing nothing."""

    def on_evaluation(self, event: EvaluationEvent) -> None:
        """Called once per evaluated configuration, in ``eval_index`` order."""

    def on_iteration_end(self, summary: IterationSummary) -> None:
        """Called at the end of each PSO iteration (PSO only)."""

    def on_finish(self, result: OptimizationResult) -> None:
        """Called once when the run has finished successfully."""


def notify(callbacks: tuple[Any, ...], hook: str, payload: Any) -> None:
    """Call ``hook`` on every callback that defines it (callbacks need not subclass :class:`Callback`)."""
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if method is not None:
            method(payload)
