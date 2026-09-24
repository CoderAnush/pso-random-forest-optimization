"""The dataset bundle returned by every loader (CONTEXT §9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class DatasetError(RuntimeError):
    """A dataset file is missing, or fails its SHA-256 check."""


@dataclass(frozen=True, eq=False)
class DatasetBundle:
    """A loaded dataset: ``X`` (n, d) float64 with NaN for missing values, ``y`` (n,) int64 labels 0..C-1.

    ``meta`` holds ``source``, ``citation``, ``sha256`` (of the raw file; None for scikit-learn's bundled
    data) and ``sha256_arrays``; :func:`pso_rf.datasets.load_dataset` adds a gate-independent ``audit``
    summary. Equality is identity (``eq=False``) because arrays have no single truth value.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        X, y = self.X, self.y
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype != np.float64:
            raise ValueError(f"{self.name}: X must be a 2-D float64 array")
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.dtype != np.int64:
            raise ValueError(f"{self.name}: y must be a 1-D int64 array")
        if len(y) != X.shape[0]:
            raise ValueError(f"{self.name}: X has {X.shape[0]} rows but y has {len(y)}")
        if len(self.feature_names) != X.shape[1]:
            raise ValueError(f"{self.name}: {len(self.feature_names)} feature names for {X.shape[1]} columns")
        if len(y) and (y.min() < 0 or y.max() >= len(self.class_names)):
            raise ValueError(f"{self.name}: labels must lie in 0..{len(self.class_names) - 1}")
