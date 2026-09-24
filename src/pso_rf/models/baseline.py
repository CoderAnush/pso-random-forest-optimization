"""The default Random Forest baseline (MLR-007, ADR-008): ``RandomForestClassifier()`` defaults.

It lies outside the search space by design (``max_depth=None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

BASELINE_CONFIG: Mapping[str, int | None] = MappingProxyType(
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
)
