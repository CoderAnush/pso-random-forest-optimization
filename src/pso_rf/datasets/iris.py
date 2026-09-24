"""Iris: scikit-learn's bundled copy of the UCI Iris data (3 classes, 4 features)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn import datasets as sk_datasets

from pso_rf.datasets.base import DatasetBundle
from pso_rf.utils.hashing import sha256_arrays

SOURCE = "sklearn.datasets.load_iris (UCI Iris)"
CITATION = "Fisher, R. A. Iris [Dataset]. UCI Machine Learning Repository (copy bundled with scikit-learn)."


def load_iris(data_dir: Path | None = None) -> DatasetBundle:
    """Load Iris; ``data_dir`` is unused (the data ships with scikit-learn)."""
    raw = sk_datasets.load_iris()
    X = np.asarray(raw.data, dtype=np.float64)
    y = np.asarray(raw.target, dtype=np.int64)
    return DatasetBundle(
        name="iris",
        X=X,
        y=y,
        feature_names=[str(name) for name in raw.feature_names],
        class_names=[str(name) for name in raw.target_names],
        meta={"source": SOURCE, "citation": CITATION, "sha256": None, "sha256_arrays": sha256_arrays(X, y)},
    )
