"""IT-01 (closed loop on a stub system) and IT-02 (feedback ablation), without any ML."""

from __future__ import annotations

import numpy as np
import pytest

from pso_rf.optimization import Callback, IntParam, PSOConfig, PSOOptimizer, SearchSpace

pytestmark = pytest.mark.closed_loop

OPTIMUM = np.array([140.0, 9.0, 5.0])


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        [
            IntParam("n_estimators", 50, 200),
            IntParam("max_depth", 2, 20),
            IntParam("min_samples_split", 2, 10),
        ]
    )


def true_fitness(space: SearchSpace, config: dict[str, int]) -> float:
    theta = np.array([config[name] for name in space.names], dtype=float)
    return -float(np.sum(((theta - OPTIMUM) / space.range_) ** 2))


class Trace(Callback):
    def __init__(self) -> None:
        self.events = []

    def on_evaluation(self, event) -> None:
        self.events.append(event)

    def positions(self, iteration: int) -> np.ndarray:
        return np.array([e.position for e in self.events if e.iteration == iteration])

    def configs(self, iteration: int) -> list[dict[str, int]]:
        return [e.config for e in self.events if e.iteration == iteration]


def run(space, objective, seed=0):
    trace = Trace()
    result = PSOOptimizer(space, objective, PSOConfig(), np.random.default_rng(seed)).run([trace])
    return result, trace


def test_it01_closed_loop_converges_on_stub_system(space: SearchSpace) -> None:
    received = []

    def system(config):  # the "plant": receives decoded integer parameters, returns the measurement
        received.append(config)
        return true_fitness(space, config)

    result, trace = run(space, system)
    # every call received a decoded integer configuration proposed by the optimizer
    assert received == [e.config for e in trace.events]
    assert all(space.contains(config) for config in received)
    # gbest non-decreasing and strictly better than the best initial particle
    gbest = [s.gbest_fitness for s in result.history]
    assert gbest == sorted(gbest)
    best_initial = max(e.fitness for e in trace.events if e.iteration == 0)
    assert result.best_fitness > best_initial
    # converged near the interior optimum (normalized distance)
    found = np.array([result.best_config[name] for name in space.names], dtype=float)
    assert np.linalg.norm((found - OPTIMUM) / space.range_) < 0.1
    # the system parameters actually change between iterations
    for t in range(result.n_iterations):
        assert trace.configs(t) != trace.configs(t + 1)


def test_it02_constant_feedback_never_moves_gbest(space: SearchSpace) -> None:
    result, trace = run(space, lambda config: 0.0)
    assert all(s.gbest_config == trace.events[0].config for s in result.history)
    assert [s.gbest_improved for s in result.history].count(True) == 1


def test_it02_trajectory_depends_on_feedback(space: SearchSpace) -> None:
    _, informed = run(space, lambda config: true_fitness(space, config))
    _, constant = run(space, lambda config: 0.0)
    np.testing.assert_array_equal(informed.positions(0), constant.positions(0))  # same seed, same start
    assert not np.allclose(informed.positions(2), constant.positions(2))


def test_it02_shuffled_feedback_steers_elsewhere(space: SearchSpace) -> None:
    def reflect(config):
        return {p.name: p.low + p.high - config[p.name] for p in space.params}

    true_result, true_trace = run(space, lambda config: true_fitness(space, config))
    shuffled_result, shuffled_trace = run(space, lambda config: true_fitness(space, reflect(config)))
    assert not np.allclose(true_trace.positions(3), shuffled_trace.positions(3))
    # judged by the TRUE objective, the swarm fed wrong feedback ends up worse
    assert true_fitness(space, shuffled_result.best_config) < true_fitness(space, true_result.best_config)
