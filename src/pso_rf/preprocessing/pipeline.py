"""Preprocessing steps declared per dataset (FR-018, ADR-010).

Every step that learns from data is returned as a pipeline member, so it is refit on the training part of each
fit (every inner-CV split, the final refit, the baseline) and never sees held-out rows (MLR-004, UT-20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.impute import SimpleImputer

IMPUTE_STRATEGIES: tuple[str, ...] = ("most_frequent", "mean", "median")


@dataclass(frozen=True)
class PreprocessingSpec:
    """Declared preprocessing for one dataset; ``impute=None`` means no step (the model sees raw features)."""

    impute: str | None = None

    def __post_init__(self) -> None:
        if self.impute is not None and self.impute not in IMPUTE_STRATEGIES:
            raise ValueError(f"impute must be None or one of {list(IMPUTE_STRATEGIES)}, got {self.impute!r}")


def build_steps(spec: PreprocessingSpec) -> list[tuple[str, Any]]:
    """Fresh, unfitted pipeline steps for ``spec``: ``[("impute", SimpleImputer(...))]`` or ``[]``."""
    if spec.impute is None:
        return []
    return [("impute", SimpleImputer(strategy=spec.impute))]
