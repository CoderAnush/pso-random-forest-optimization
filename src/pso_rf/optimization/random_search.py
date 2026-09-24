"""Random search: the open-loop comparator (MATHEMATICAL_FORMULATION §8, ADR-008).

Configurations are drawn i.i.d. uniformly from the integer search space. The next proposal never depends
on any fitness value; the feedback is used only to pick the winner at the end (earliest on ties). With the
same objective and budget as PSO, any systematic difference between the two is attributable to feedback.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pso_rf.optimization.events import (
    Evaluation,
    EvaluationEvent,
    OptimizationResult,
    as_evaluation,
    notify,
)
from pso_rf.optimization.search_space import SearchSpace


class RandomSearch:
    """Evaluate ``budget`` uniformly sampled configurations and return the best one."""

    def __init__(
        self,
        space: SearchSpace,
        objective: Callable[[dict[str, int]], Evaluation | float],
        budget: int,
        rng: np.random.Generator,
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.space = space
        self.objective = objective
        self.budget = budget
        self.rng = rng

    def run(self, callbacks: Sequence[Any] = ()) -> OptimizationResult:
        """Sample, evaluate and record ``budget`` configurations; the best is the earliest maximum."""
        callbacks = tuple(callbacks)
        best_fit, best_cfg, best_info, best_eval = -math.inf, None, {}, 0
        best_so_far = -math.inf
        seen: dict[tuple[int, ...], float] = {}
        for j in range(self.budget):
            config = self.space.sample(self.rng)  # independent of every fitness value (open loop)
            evaluation = as_evaluation(self.objective(dict(config)))
            fitness = evaluation.fitness
            seen.setdefault(self.space.key(config), fitness)
            if fitness > best_fit:
                best_fit, best_cfg, best_info, best_eval = fitness, config, evaluation.info, j
            best_so_far = max(best_so_far, fitness)
            notify(
                callbacks,
                "on_evaluation",
                EvaluationEvent(
                    eval_index=j,
                    iteration=None,
                    particle_id=None,
                    position=None,
                    velocity=None,
                    config=dict(config),
                    fitness=fitness,
                    info=evaluation.info,
                    pbest_fitness=None,
                    gbest_fitness=None,
                    best_so_far_fitness=best_so_far,
                ),
            )
        if best_cfg is None or not math.isfinite(best_fit):
            raise RuntimeError("every random-search evaluation failed; there is no best configuration")
        result = OptimizationResult(
            method="random_search",
            best_config=dict(best_cfg),
            best_fitness=best_fit,
            best_position=None,
            best_info=best_info,
            best_found_at_eval=best_eval,
            best_found_at_iteration=None,
            n_evaluations=self.budget,
            n_iterations=None,
            stop_reason="budget",
            convergence_iteration=None,
            n_ties_with_best=sum(1 for value in seen.values() if value == best_fit),
        )
        notify(callbacks, "on_finish", result)
        return result
