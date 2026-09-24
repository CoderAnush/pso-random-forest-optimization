"""The integer search space and the decode operator (MATHEMATICAL_FORMULATION §3–§4, ADR-011).

PSO moves in the continuous box ``[lower, upper]``; a position becomes a configuration only when it is
evaluated, through ``decode(x) = clip(rint(x), lower, upper)`` (round half to even).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntParam:
    """One integer decision variable with inclusive bounds ``low < high``."""

    name: str
    low: int
    high: int

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"parameter name must be an identifier, got {self.name!r}")
        for bound in (self.low, self.high):
            if not isinstance(bound, int) or isinstance(bound, bool):
                raise ValueError(f"{self.name}: bounds must be integers, got {self.low!r}, {self.high!r}")
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be < high, got {self.low} >= {self.high}")


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class SearchSpace:
    """An ordered box of integer parameters; the order is the dimension order of a particle position."""

    def __init__(self, params: Sequence[IntParam]) -> None:
        params = tuple(params)
        if not params:
            raise ValueError("a search space needs at least one parameter")
        names = tuple(p.name for p in params)
        if len(set(names)) != len(names):
            raise ValueError(f"parameter names must be unique, got {names}")
        self.params = params
        self.names = names
        self.lower = _read_only(np.array([p.low for p in params], dtype=np.float64))
        self.upper = _read_only(np.array([p.high for p in params], dtype=np.float64))
        self.range_ = _read_only(self.upper - self.lower)
        self._low_int = np.array([p.low for p in params], dtype=np.int64)
        self._high_int = np.array([p.high for p in params], dtype=np.int64)

    @property
    def dim(self) -> int:
        """Number of decision variables."""
        return len(self.params)

    @property
    def size(self) -> int:
        """Number of integer configurations, e.g. 151 × 19 × 9 = 25,821 for the default space."""
        return math.prod(p.high - p.low + 1 for p in self.params)

    def clip(self, position: np.ndarray) -> np.ndarray:
        """Component-wise clip of a position into the box."""
        return np.clip(np.asarray(position, dtype=np.float64), self.lower, self.upper)

    def decode(self, position: np.ndarray) -> dict[str, int]:
        """The configuration a position stands for: ``clip(rint(position))`` as Python ints."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (self.dim,):
            raise ValueError(f"position must have shape ({self.dim},), got {position.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"position must be finite, got {position}")
        values = np.clip(np.rint(position), self.lower, self.upper)
        return {name: int(value) for name, value in zip(self.names, values, strict=True)}

    def sample(self, rng: np.random.Generator) -> dict[str, int]:
        """One configuration drawn uniformly from the integer grid (random search)."""
        values = rng.integers(self._low_int, self._high_int + 1)
        return {name: int(value) for name, value in zip(self.names, values, strict=True)}

    def key(self, config: Mapping[str, int]) -> tuple[int, ...]:
        """A hashable key for a configuration, in dimension order."""
        return tuple(config[name] for name in self.names)

    def contains(self, config: Mapping[str, int]) -> bool:
        """True if ``config`` has exactly this space's names with integer values inside the bounds."""
        if set(config) != set(self.names):
            return False
        return all(isinstance(config[p.name], int) and p.low <= config[p.name] <= p.high for p in self.params)

    def __repr__(self) -> str:
        bounds = ", ".join(f"{p.name}=[{p.low}, {p.high}]" for p in self.params)
        return f"SearchSpace({bounds})"
