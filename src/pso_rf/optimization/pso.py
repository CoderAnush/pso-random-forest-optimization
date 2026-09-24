"""Particle Swarm Optimization, implemented from scratch (MATHEMATICAL_FORMULATION §7, ADR-011 to ADR-017).

NumPy and the standard library only (UT-22). The optimizer knows nothing about Random Forests or data: it
proposes configurations through a :class:`~pso_rf.optimization.search_space.SearchSpace` and receives only a
fitness value back from the objective. That feedback is what closes the loop.

Random draws happen in a fixed order from one ``numpy.random.Generator``: initial positions, initial
velocities, then ``R1`` and ``R2`` once per iteration. The same seed therefore reproduces the same run.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pso_rf.optimization.events import (
    Evaluation,
    EvaluationEvent,
    IterationSummary,
    OptimizationResult,
    as_evaluation,
    notify,
)
from pso_rf.optimization.search_space import SearchSpace


@dataclass(frozen=True)
class PSOConfig:
    """PSO parameters; the defaults are the design values (ADR-012 to ADR-015).

    ``v_max_frac`` and ``v_init_frac`` are fractions of each dimension's range (upper - lower).
    The patience rule is off by default; when on, the run stops once gbest has improved by less than
    ``patience_tol`` over ``patience_iterations`` iterations.
    """

    n_particles: int = 10
    max_iter: int = 20
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    v_max_frac: float = 0.2
    v_init_frac: float = 0.1
    boundary: str = "absorb"
    topology: str = "gbest"
    update: str = "synchronous"
    patience_enabled: bool = False
    patience_tol: float = 1e-4
    patience_iterations: int = 5


# ------------------------------------------------------------------------------------ pure update functions


def velocity_update(
    V: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    g: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    v_max: np.ndarray,
) -> np.ndarray:
    """§7.3: ``clip(w·V + c1·R1·(P − X) + c2·R2·(g − X), −v_max, v_max)`` (element-wise)."""
    raw = w * V + c1 * R1 * (P - X) + c2 * R2 * (g - X)
    return np.clip(raw, -v_max, v_max)


def apply_absorbing_bounds(
    X_new: np.ndarray, V: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """§7.4: clip positions into the box and zero every velocity component whose position was clipped."""
    outside = (X_new < lower) | (X_new > upper)
    return np.clip(X_new, lower, upper), np.where(outside, 0.0, V)


def update_personal_best(
    P: np.ndarray, P_fit: np.ndarray, X: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§7.5: replace a personal best only on strict improvement (``F > P_fit``); ties keep the incumbent.

    Returns the new ``(P, P_fit, improved_mask)``; the inputs are not modified.
    """
    improved = F > P_fit
    return np.where(improved[:, None], X, P), np.where(improved, F, P_fit), improved


def best_index(P_fit: np.ndarray) -> int:
    """Index of the best personal best; ties go to the lowest particle index."""
    return int(np.argmax(P_fit))


def swarm_diversity(X: np.ndarray, lower: np.ndarray, range_: np.ndarray) -> float:
    """Mean Euclidean distance of the range-normalized positions from their centroid."""
    Z = (X - lower) / range_
    return float(np.mean(np.linalg.norm(Z - Z.mean(axis=0), axis=1)))


# ------------------------------------------------------------------------------------------------ optimizer


class PSOOptimizer:
    """Global-best, synchronous PSO over a :class:`SearchSpace` (the controller of the closed loop)."""

    def __init__(
        self,
        space: SearchSpace,
        objective: Callable[[dict[str, int]], Evaluation | float],
        config: PSOConfig,
        rng: np.random.Generator,
    ) -> None:
        if config.boundary != "absorb" or config.topology != "gbest" or config.update != "synchronous":
            raise NotImplementedError(
                "only boundary='absorb', topology='gbest' and update='synchronous' are implemented"
            )
        if config.n_particles < 1 or config.max_iter < 0:
            raise ValueError("n_particles must be >= 1 and max_iter >= 0")
        self.space = space
        self.objective = objective
        self.config = config
        self.rng = rng

    def run(self, callbacks: Sequence[Any] = ()) -> OptimizationResult:
        """Run the swarm for ``max_iter`` iterations (or until patience stops it); return the best found."""
        callbacks = tuple(callbacks)
        cfg, space, rng = self.config, self.space, self.rng
        N, D = cfg.n_particles, space.dim
        lower, upper, range_ = space.lower, space.upper, space.range_
        v_max = cfg.v_max_frac * range_
        v_init = cfg.v_init_frac * range_
        start = time.perf_counter()

        # §7.2 initialization (t = 0)
        X = rng.uniform(lower, upper, size=(N, D))
        V = rng.uniform(-v_init, v_init, size=(N, D))
        configs, F, infos = self._evaluate_swarm(X)

        P, P_fit = X.copy(), F.copy()
        P_cfg, P_info = list(configs), list(infos)
        P_eval = [i for i in range(N)]  # eval_index at which each personal best was found

        state = _RunState(start=start)
        self._update_global(state, P, P_fit, P_cfg, P_info, P_eval, t=0)
        self._emit(callbacks, state, t=0, X=X, V=V, configs=configs, F=F, infos=infos, P_fit=P_fit)

        stop_reason = "max_iter"
        t_last = 0
        for t in range(1, cfg.max_iter + 1):
            R1 = rng.random((N, D))
            R2 = rng.random((N, D))
            V = velocity_update(V, X, P, state.g, R1, R2, cfg.w, cfg.c1, cfg.c2, v_max)
            X, V = apply_absorbing_bounds(X + V, V, lower, upper)
            configs, F, infos = self._evaluate_swarm(X)

            P, P_fit, improved = update_personal_best(P, P_fit, X, F)
            for i in np.flatnonzero(improved):
                P_cfg[i], P_info[i], P_eval[i] = configs[i], infos[i], t * N + int(i)
            self._update_global(state, P, P_fit, P_cfg, P_info, P_eval, t=t)
            self._emit(callbacks, state, t=t, X=X, V=V, configs=configs, F=F, infos=infos, P_fit=P_fit)
            t_last = t

            if cfg.patience_enabled and t >= cfg.patience_iterations:
                gain = state.gbest_history[t] - state.gbest_history[t - cfg.patience_iterations]
                if gain < cfg.patience_tol:
                    stop_reason = "patience"
                    break

        if not math.isfinite(state.g_fit):
            raise RuntimeError("every PSO evaluation failed; there is no best configuration")
        result = OptimizationResult(
            method="pso",
            best_config=dict(state.g_cfg),
            best_fitness=state.g_fit,
            best_position=tuple(float(v) for v in state.g),
            best_info=state.g_info,
            best_found_at_eval=state.g_eval,
            best_found_at_iteration=state.g_iter,
            n_evaluations=N * (t_last + 1),
            n_iterations=t_last,
            stop_reason=stop_reason,
            convergence_iteration=state.convergence_iteration,
            n_ties_with_best=state.history[-1].n_ties_with_gbest,
            history=tuple(state.history),
        )
        notify(callbacks, "on_finish", result)
        return result

    # ---------------------------------------------------------------------------------------------- helpers

    def _evaluate_swarm(self, X: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray, list[Any]]:
        """Decode every particle and query the objective: the feedback step of the loop."""
        configs, fitness, infos = [], [], []
        for x in X:
            config = self.space.decode(x)
            evaluation = as_evaluation(self.objective(dict(config)))
            configs.append(config)
            fitness.append(evaluation.fitness)
            infos.append(evaluation.info)
        return configs, np.array(fitness, dtype=np.float64), infos

    @staticmethod
    def _update_global(
        state: _RunState,
        P: np.ndarray,
        P_fit: np.ndarray,
        P_cfg: list[dict[str, int]],
        P_info: list[Any],
        P_eval: list[int],
        t: int,
    ) -> None:
        """§7.5: gbest changes only if the best personal best strictly beats it (lowest index on ties)."""
        i = best_index(P_fit)
        if state.g is None or P_fit[i] > state.g_fit:
            improved = P_fit[i] > state.g_fit
            state.g, state.g_fit, state.g_cfg = P[i].copy(), float(P_fit[i]), dict(P_cfg[i])
            state.g_info, state.g_eval, state.g_iter = P_info[i], P_eval[i], t
            if improved:
                state.improved = True
                state.convergence_iteration = t
                state.no_improve = 0
                state.gbest_history.append(state.g_fit)
                return
        state.improved = False
        state.no_improve += 1
        state.gbest_history.append(state.g_fit)

    def _emit(
        self,
        callbacks: tuple[Any, ...],
        state: _RunState,
        t: int,
        X: np.ndarray,
        V: np.ndarray,
        configs: list[dict[str, int]],
        F: np.ndarray,
        infos: list[Any],
        P_fit: np.ndarray,
    ) -> None:
        """Send one event per particle, then the iteration summary, and record the summary."""
        N = len(F)
        for i in range(N):
            config = configs[i]
            state.seen.setdefault(self.space.key(config), float(F[i]))
            state.best_so_far = max(state.best_so_far, float(F[i]))
            event = EvaluationEvent(
                eval_index=t * N + i,
                iteration=t,
                particle_id=i,
                position=tuple(float(v) for v in X[i]),
                velocity=tuple(float(v) for v in V[i]),
                config=dict(config),
                fitness=float(F[i]),
                info=infos[i],
                pbest_fitness=float(P_fit[i]),
                gbest_fitness=state.g_fit,
                best_so_far_fitness=state.best_so_far,
            )
            notify(callbacks, "on_evaluation", event)

        finite = F[np.isfinite(F)]
        empty = finite.size == 0
        ties = (
            sum(1 for value in state.seen.values() if value == state.g_fit)
            if math.isfinite(state.g_fit)
            else 0
        )
        summary = IterationSummary(
            iteration=t,
            gbest_fitness=state.g_fit,
            gbest_config=dict(state.g_cfg),
            gbest_improved=state.improved,
            mean_fitness=math.nan if empty else float(finite.mean()),
            std_fitness=math.nan if empty else float(finite.std(ddof=0)),
            min_fitness=math.nan if empty else float(finite.min()),
            max_fitness=math.nan if empty else float(finite.max()),
            diversity=swarm_diversity(X, self.space.lower, self.space.range_),
            n_unique_configs=len({self.space.key(c) for c in configs}),
            n_ties_with_gbest=ties,
            n_failed=int(N - finite.size),
            cumulative_evaluations=N * (t + 1),
            no_improve_count=state.no_improve,
            elapsed_s=time.perf_counter() - state.start,
        )
        state.history.append(summary)
        notify(callbacks, "on_iteration_end", summary)


class _RunState:
    """Mutable bookkeeping of one run: global best, convergence and tie tracking."""

    def __init__(self, start: float) -> None:
        self.start = start
        self.g: np.ndarray | None = None
        self.g_fit = -math.inf
        self.g_cfg: dict[str, int] = {}
        self.g_info: Any = {}
        self.g_eval = 0
        self.g_iter = 0
        self.improved = False
        self.convergence_iteration = 0
        self.no_improve = 0
        self.best_so_far = -math.inf
        self.gbest_history: list[float] = []
        self.seen: dict[tuple[int, ...], float] = {}
        self.history: list[IterationSummary] = []
