"""Seed policy (ADR-018): one integer per run drives every random component of that run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunSeeds:
    """The seeds of one run. In run *k* all equal ``run_seeds[k]``, which pairs the methods by fold."""

    run: int
    pso: int
    random_search: int
    inner_cv: int
    rf: int

    @classmethod
    def from_run_seed(cls, seed: int) -> RunSeeds:
        """Every component uses the run seed (outer fold k → seed k; deployment → 5)."""
        return cls(run=seed, pso=seed, random_search=seed, inner_cv=seed, rf=seed)


def make_rng(seed: int) -> np.random.Generator:
    """A fresh, independent generator; the only source of randomness an optimizer receives."""
    return np.random.default_rng(seed)
