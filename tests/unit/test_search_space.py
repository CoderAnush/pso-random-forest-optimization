"""UT-01: search space, integer conversion (decode) and sampling."""

from __future__ import annotations

import numpy as np
import pytest

from pso_rf.optimization import IntParam, SearchSpace


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        [
            IntParam("n_estimators", 50, 200),
            IntParam("max_depth", 2, 20),
            IntParam("min_samples_split", 2, 10),
        ]
    )


def test_bounds_and_size(space: SearchSpace) -> None:
    assert space.names == ("n_estimators", "max_depth", "min_samples_split")
    np.testing.assert_array_equal(space.lower, [50, 2, 2])
    np.testing.assert_array_equal(space.upper, [200, 20, 10])
    np.testing.assert_array_equal(space.range_, [150, 18, 8])
    assert space.size == 151 * 19 * 9 == 25_821


def test_decode_rounds_half_to_even(space: SearchSpace) -> None:
    assert space.decode(np.array([124.5, 8.5, 2.5])) == {
        "n_estimators": 124,
        "max_depth": 8,
        "min_samples_split": 2,
    }


def test_decode_clips_to_bounds(space: SearchSpace) -> None:
    assert space.decode(np.array([200.4, 20.49, 10.2])) == {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 10,
    }
    assert space.decode(np.array([49.6, 1.2, 1.9])) == {
        "n_estimators": 50,
        "max_depth": 2,
        "min_samples_split": 2,
    }


def test_decode_returns_python_ints_inside_bounds(space: SearchSpace) -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(space.lower - 30, space.upper + 30, size=(500, 3))
    for position in positions:
        config = space.decode(position)
        assert all(type(value) is int for value in config.values())
        assert space.contains(config)


def test_decode_rejects_bad_positions(space: SearchSpace) -> None:
    with pytest.raises(ValueError):
        space.decode(np.array([100.0, 5.0]))
    with pytest.raises(ValueError):
        space.decode(np.array([100.0, np.nan, 3.0]))


def test_sample_is_uniform_over_the_integer_grid(space: SearchSpace) -> None:
    rng = np.random.default_rng(1)
    samples = [space.sample(rng) for _ in range(9000)]
    assert all(space.contains(s) for s in samples)
    counts = np.bincount([s["min_samples_split"] for s in samples], minlength=11)[2:]
    assert np.all(np.abs(counts - 1000) < 150)  # 9 values × 1000 expected each
    depths = {s["max_depth"] for s in samples}
    assert depths == set(range(2, 21))


def test_invalid_params_are_rejected() -> None:
    with pytest.raises(ValueError):
        IntParam("x", 5, 5)
    with pytest.raises(ValueError):
        IntParam("x", 1.5, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SearchSpace([IntParam("x", 0, 1), IntParam("x", 0, 2)])
    with pytest.raises(ValueError):
        SearchSpace([])
