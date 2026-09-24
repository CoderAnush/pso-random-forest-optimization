"""UT-17: random search is uniform, respects its budget, and ignores feedback when proposing."""

from __future__ import annotations

import numpy as np
import pytest

from pso_rf.optimization import Callback, IntParam, RandomSearch, SearchSpace


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

    def on_evaluation(self, event) -> None:
        self.events.append(event)


def run(space, objective, budget=50, seed=0):
    collector = Collector()
    result = RandomSearch(space, objective, budget, np.random.default_rng(seed)).run([collector])
    return result, collector


def test_budget_and_samples_in_space(space: SearchSpace) -> None:
    calls = []

    def objective(config):
        calls.append(config)
        return float(config["max_depth"])

    result, collector = run(space, objective, budget=37)
    assert len(calls) == 37 == result.n_evaluations == len(collector.events)
    assert all(space.contains(config) for config in calls)
    assert [e.eval_index for e in collector.events] == list(range(37))
    assert result.stop_reason == "budget" and result.n_iterations is None


def test_proposals_are_independent_of_feedback(space: SearchSpace) -> None:
    noise = np.random.default_rng(99)
    _, constant = run(space, lambda config: 0.0, seed=3)
    _, random = run(space, lambda config: float(noise.random()), seed=3)
    assert [e.config for e in constant.events] == [e.config for e in random.events]


def test_best_is_earliest_maximum(space: SearchSpace) -> None:
    result, _ = run(space, lambda config: 1.0)
    assert result.best_found_at_eval == 0
    result, collector = run(space, lambda config: float(config["n_estimators"]))
    best = max(e.fitness for e in collector.events)
    first = next(e for e in collector.events if e.fitness == best)
    assert result.best_found_at_eval == first.eval_index
    assert result.best_config == first.config


def test_events_have_no_pso_fields_and_running_best(space: SearchSpace) -> None:
    _, collector = run(space, lambda config: float(config["max_depth"]))
    running = -np.inf
    for event in collector.events:
        assert event.iteration is None and event.particle_id is None
        assert event.position is None and event.velocity is None
        assert event.pbest_fitness is None and event.gbest_fitness is None
        running = max(running, event.fitness)
        assert event.best_so_far_fitness == running


def test_all_failed_raises(space: SearchSpace) -> None:
    with pytest.raises(RuntimeError, match="every random-search evaluation failed"):
        run(space, lambda config: float("nan"), budget=5)


def test_invalid_budget(space: SearchSpace) -> None:
    with pytest.raises(ValueError):
        RandomSearch(space, lambda config: 0.0, 0, np.random.default_rng(0))
