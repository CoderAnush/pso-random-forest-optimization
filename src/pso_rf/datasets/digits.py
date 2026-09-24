"""Digits: scikit-learn's bundled copy of the UCI Optical Recognition of Handwritten Digits test set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn import datasets as sk_datasets

from pso_rf.datasets.base import DatasetBundle
from pso_rf.utils.hashing import sha256_arrays

SOURCE = "sklearn.datasets.load_digits (UCI Optical Recognition of Handwritten Digits, test set)"
CITATION = (
    "Alpaydin, E. & Kaynak, C. Optical Recognition of Handwritten Digits [Dataset]. "
    "UCI Machine Learning Repository (test set, bundled with scikit-learn)."
)


def load_digits(data_dir: Path | None = None) -> DatasetBundle:
    """Load Digits (8×8 images as 64 features, 10 classes); ``data_dir`` is unused."""
    raw = sk_datasets.load_digits()
    X = np.asarray(raw.data, dtype=np.float64)
    y = np.asarray(raw.target, dtype=np.int64)
    return DatasetBundle(
        name="digits",
        X=X,
        y=y,
        feature_names=[str(name) for name in raw.feature_names],
        class_names=[str(name) for name in raw.target_names],
        meta={"source": SOURCE, "citation": CITATION, "sha256": None, "sha256_arrays": sha256_arrays(X, y)},
    )
