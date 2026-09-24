"""Dataset loading, checksum verification and auditing (FR-001).

``load_dataset(name, data_dir)`` is the single entry point. The returned bundle's ``meta["audit"]`` holds the
gate-independent audit facts (sizes, class counts, missing values, duplicates).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from pso_rf.datasets.audit import describe
from pso_rf.datasets.base import DatasetBundle, DatasetError
from pso_rf.datasets.digits import load_digits
from pso_rf.datasets.heart_cleveland import load_heart_cleveland
from pso_rf.datasets.iris import load_iris

LOADERS: dict[str, Callable[[Path], DatasetBundle]] = {
    "iris": load_iris,
    "digits": load_digits,
    "heart_cleveland": load_heart_cleveland,
}


def load_dataset(name: str, data_dir: Path) -> DatasetBundle:
    """Load a registered dataset (``data_dir`` is the ``data/`` directory) and attach its audit summary."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; known: {list(LOADERS)}")
    bundle = LOADERS[name](Path(data_dir))
    return dataclasses.replace(bundle, meta={**bundle.meta, "audit": describe(bundle)})


__all__ = ["LOADERS", "DatasetBundle", "DatasetError", "load_dataset"]
