"""UT-02 … UT-08: PSO initialization, update equations, bounds, memory, stopping (stub objectives only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pso_rf.optimization import Callback, Evaluation, IntParam, PSOConfig, PSOOptimizer, SearchSpace
from pso_rf.optimization.pso import (
    apply_absorbing_bounds,
    best_index,
    update_personal_best,
    velocity_update,
)

W, C = 0.7298, 1.49618


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        [
            IntParam("n_estimators", 50, 200),
            IntParam("max_depth", 2, 20),
            IntParam("min_samples_split", 2, 10),
        ]
    )


class Collector(Callback):
    def __init__(self) -> None:
        self.events = []
        self.summaries = []

    def on_evaluation(self, event) -> None:
        self.events.append(event)

    def on_iteration_end(self, summary) -> None:
        self.summaries.append(summary)


def quadratic(space: SearchSpace, optimum=(140, 9, 5)):
    target = np.array(optimum, dtype=float)

    def objective(config: dict[str, int]) -> float:
        theta = np.array([config[name] for name in space.names], dtype=float)
        return -float(np.sum(((theta - target) / space.range_) ** 2))

    return objective


def run(space, objective, seed=0, **overrides):
    collector = Collector()
    result = PSOOptimizer(space, objective, PSOConfig(**overrides), np.random.default_rng(seed)).run(
        [collector]
    )
    return result, collector


# ------------------------------------------------------------------------------------------ UT-02 init


def test_initialization_within_bounds_and_seeded(space: SearchSpace) -> None:
    _, a = run(space, quadratic(space), seed=3, max_iter=0)
    _, b = run(space, quadratic(space), seed=3, max_iter=0)
    _, c = run(space, quadratic(space), seed=4, max_iter=0)
    assert len(a.events) == 10
    X = np.array([e.position for e in a.events])
    V = np.array([e.velocity for e in a.events])
    assert X.shape == V.shape == (10, 3)
    assert np.all((X >= space.lower) & (X <= space.upper))
    assert np.all(np.abs(V) <= 0.1 * space.range_)
    assert [e.position for e in a.events] == [e.position for e in b.events]
    assert [e.velocity for e in a.events] == [e.velocity for e in b.events]
    assert [e.position for e in a.events] != [e.position for e in c.events]


# ---------------------------------------------------------------------- UT-03 worked example (§9)


def test_velocity_update_reproduces_worked_example(space: SearchSpace) -> None:
    X = np.array([[120.4, 8.2, 4.6]])
    V = np.array([[5.0, -1.0, 0.5]])
    P = np.array([[130.0, 10.0, 3.0]])
    g = np.array([180.0, 15.0, 3.0])
    R1 = np.array([[0.5, 0.2, 0.9]])
    R2 = np.array([[0.3, 0.7, 0.1]])
    raw = velocity_update(V, X, P, g, R1, R2, W, C, C, np.full(3, np.inf))
    np.testing.assert_allclose(raw[0], [37.5824, 6.9306, -2.0290], atol=1e-4)
    v_max = 0.2 * space.range_
    clamped = velocity_update(V, X, P, g, R1, R2, W, C, C, v_max)
    np.testing.assert_allclose(clamped[0], [30.0, 3.6, -1.6])
    X_new, V_new = apply_absorbing_bounds(X + clamped, clamped, space.lower, space.upper)
    np.testing.assert_allclose(X_new[0], [150.4, 11.8, 3.0])
    np.testing.assert_allclose(V_new[0], clamped[0])
    assert space.decode(X[0]) == {"n_estimators": 120, "max_depth": 8, "min_samples_split": 5}
    assert space.decode(X_new[0]) == {"n_estimators": 150, "max_depth": 12, "min_samples_split": 3}


# ------------------------------------------------------------------------------------- UT-04 clamp


def test_velocity_clamp(space: SearchSpace) -> None:
    rng = np.random.default_rng(0)
    shape = (200, 3)
    X = rng.uniform(space.lower, space.upper, size=shape)
    P = rng.uniform(space.lower, space.upper, size=shape)
    g = rng.uniform(space.lower, space.upper)
    V = rng.uniform(-100, 100, size=shape)
    R1, R2 = rng.random(shape), rng.random(shape)
    v_max = 0.2 * space.range_
    raw = velocity_update(V, X, P, g, R1, R2, W, C, C, np.full(3, np.inf))
    clamped = velocity_update(V, X, P, g, R1, R2, W, C, C, v_max)
    assert np.all(np.abs(clamped) <= v_max)
    inside = np.abs(raw) <= v_max
    np.testing.assert_array_equal(clamped[inside], raw[inside])


# ------------------------------------------------------------------------------ UT-05 absorbing wall


def test_absorbing_boundary(space: SearchSpace) -> None:
    X = np.array([[100.0, 10.0, 2.5]])
    V = np.array([[5.0, 1.0, -1.6]])
    X_new, V_new = apply_absorbing_bounds(X + V, V, space.lower, space.upper)
    np.testing.assert_allclose(X_new[0], [105.0, 11.0, 2.0])
    np.testing.assert_allclose(V_new[0], [5.0, 1.0, 0.0])
    X_up, V_up = apply_absorbing_bounds(
        np.array([[250.0, 25.0, 9.0]]), np.ones((1, 3)), space.lower, space.upper
    )
    np.testing.assert_allclose(X_up[0], [200.0, 20.0, 9.0])
    np.testing.assert_allclose(V_up[0], [0.0, 0.0, 1.0])


def test_positions_always_inside_the_box(space: SearchSpace) -> None:
    _, collector = run(space, quadratic(space, optimum=(200, 20, 2)), seed=1, max_iter=15, v_max_frac=0.9)
    X = np.array([e.position for e in collector.events])
    assert np.all((X >= space.lower) & (X <= space.upper))


# ---------------------------------------------------------------------------------- UT-06 pbest


def test_personal_best_strict_improvement() -> None:
    P = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    P_fit = np.array([0.5, 0.5, 0.5])
    X = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])
    F = np.array([0.6, 0.5, 0.4])  # improves, ties, worsens
    P_new, P_fit_new, improved = update_personal_best(P, P_fit, X, F)
    np.testing.assert_array_equal(improved, [True, False, False])
    np.testing.assert_array_equal(P_new, [[9.0, 9.0], [2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_array_equal(P_fit_new, [0.6, 0.5, 0.5])
    np.testing.assert_array_equal(P, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # inputs untouched


def test_pbest_fitness_is_running_max_per_particle(space: SearchSpace) -> None:
    _, collector = run(space, quadratic(space), seed=2)
    for particle in range(10):
        events = [e for e in collector.events if e.particle_id == particle]
        running = -math.inf
        for event in events:
            running = max(running, event.fitness)
            assert event.pbest_fitness == running


# ---------------------------------------------------------------------------------- UT-07 gbest


def test_best_index_ties_go_to_lowest_index() -> None:
    assert best_index(np.array([0.1, 0.9, 0.9, 0.2])) == 1


def test_gbest_monotone_equals_max_pbest_and_changes_only_on_improvement(space: SearchSpace) -> None:
    result, collector = run(space, quadratic(space), seed=5)
    history = [s.gbest_fitness for s in collector.summaries]
    assert history == sorted(history)
    for summary in collector.summaries:
        events = [e for e in collector.events if e.iteration == summary.iteration]
        assert summary.gbest_fitness == max(e.pbest_fitness for e in events)
        assert all(e.gbest_fitness == summary.gbest_fitness for e in events)
    for previous, current in zip(collector.summaries, collector.summaries[1:], strict=False):
        if not current.gbest_improved:
            assert current.gbest_config == previous.gbest_config
    assert result.best_fitness == history[-1]


def test_equal_fitness_never_replaces_gbest(space: SearchSpace) -> None:
    result, collector = run(space, lambda config: 1.0, seed=0)
    first = collector.events[0].config  # particle 0 wins the initial tie (lowest index)
    assert all(s.gbest_config == first for s in collector.summaries)
    assert [s.gbest_improved for s in collector.summaries] == [True] + [False] * 20
    assert result.best_found_at_eval == 0


# -------------------------------------------------------------------------------- UT-08 stopping


def test_default_budget_is_210_evaluations(space: SearchSpace) -> None:
    calls = []

    def objective(config):
        calls.append(config)
        return quadratic(space)(config)

    result, collector = run(space, objective, seed=0)
    assert len(calls) == 210 == result.n_evaluations
    assert result.n_iterations == 20 and result.stop_reason == "max_iter"
    assert len(collector.summaries) == 21
    last_improved = max(s.iteration for s in collector.summaries if s.gbest_improved)
    assert result.convergence_iteration == last_improved


def test_patience_stops_a_constant_objective(space: SearchSpace) -> None:
    result, collector = run(space, lambda config: 0.5, seed=0, patience_enabled=True, patience_iterations=5)
    assert result.stop_reason == "patience"
    assert result.n_iterations == 5
    assert result.n_evaluations == 10 * 6
    assert len(collector.summaries) == 6


def test_patience_does_not_stop_while_improving(space: SearchSpace) -> None:
    result, _ = run(
        space, quadratic(space), seed=0, patience_enabled=True, patience_iterations=5, patience_tol=0.0
    )
    assert result.n_iterations <= 20


# ------------------------------------------------------------------------ failures and feedback types


def test_failed_evaluations_are_tolerated(space: SearchSpace) -> None:
    def objective(config):
        if config["max_depth"] < 8:
            return Evaluation(-math.inf, {"status": "failed"})
        return quadratic(space)(config)

    result, collector = run(space, objective, seed=0)
    assert math.isfinite(result.best_fitness)
    assert result.best_config["max_depth"] >= 8
    assert any(s.n_failed > 0 for s in collector.summaries)


def test_all_failed_raises(space: SearchSpace) -> None:
    with pytest.raises(RuntimeError, match="every PSO evaluation failed"):
        run(space, lambda config: float("nan"), seed=0, max_iter=2)


def test_evaluation_info_is_forwarded_untouched(space: SearchSpace) -> None:
    def objective(config):
        return Evaluation(1.0, {"tag": config["n_estimators"]})

    _, collector = run(space, objective, seed=0, max_iter=1)
    assert all(e.info == {"tag": e.config["n_estimators"]} for e in collector.events)


def test_unsupported_variants_are_rejected(space: SearchSpace) -> None:
    with pytest.raises(NotImplementedError):
        PSOOptimizer(space, lambda c: 0.0, PSOConfig(boundary="reflect"), np.random.default_rng(0))


# ------------------------------------------------------------------ optional starting position (demo option)


def test_start_places_particle_zero_and_keeps_the_rest(space: SearchSpace) -> None:
    _, plain = run(space, quadratic(space), seed=4, max_iter=0)
    _, seeded = run(space, quadratic(space), seed=4, max_iter=0, start=(60.0, 3.0, 10.0))
    assert seeded.events[0].position == (60.0, 3.0, 10.0)
    assert seeded.events[0].config == {"n_estimators": 60, "max_depth": 3, "min_samples_split": 10}
    # same random stream: every other particle and every velocity is unchanged
    assert [e.position for e in seeded.events[1:]] == [e.position for e in plain.events[1:]]
    assert [e.velocity for e in seeded.events] == [e.velocity for e in plain.events]


def test_start_none_is_the_standard_run(space: SearchSpace) -> None:
    a, ca = run(space, quadratic(space), seed=9)
    b, cb = run(space, quadratic(space), seed=9, start=None)
    assert [e.position for e in ca.events] == [
        e.position for e in cb.events
    ] and a.best_config == b.best_config


def test_start_is_clipped_and_checked(space: SearchSpace) -> None:
    _, c = run(space, quadratic(space), seed=0, max_iter=0, start=(500.0, 0.0, 5.0))
    assert c.events[0].position == (200.0, 2.0, 5.0)
    with pytest.raises(ValueError):
        PSOOptimizer(space, quadratic(space), PSOConfig(start=(60.0, 3.0)), np.random.default_rng(0))
