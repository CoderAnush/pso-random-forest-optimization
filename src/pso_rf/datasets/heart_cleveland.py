"""UCI Heart Disease, Cleveland subset (``processed.cleveland.data``), vendored in ``data/raw/`` (ADR-009)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pso_rf.datasets.base import DatasetBundle, DatasetError
from pso_rf.utils.hashing import sha256_arrays, sha256_file

COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)
FEATURES = COLUMNS[:-1]
CLASS_NAMES = ("no_disease", "disease")
RAW_FILE = "processed.cleveland.data"
MANIFEST_FILE = "MANIFEST.json"
DOWNLOAD_HINT = "run `python scripts/download_cleveland.py` to download it"


def load_heart_cleveland(data_dir: Path) -> DatasetBundle:
    """Load Cleveland from ``<data_dir>/raw/``, verifying its SHA-256 against ``MANIFEST.json``.

    ``?`` becomes NaN (DR-004) and the target is ``num > 0 → 1`` (disease), ``num = 0 → 0`` (DR-003).
    Raises :class:`DatasetError` if the file or manifest is missing or the checksum does not match.
    """
    raw_dir = Path(data_dir) / "raw"
    raw_path = raw_dir / RAW_FILE
    manifest_path = raw_dir / MANIFEST_FILE
    if not raw_path.is_file():
        raise DatasetError(f"Cleveland data file not found: {raw_path}; {DOWNLOAD_HINT}.")
    if not manifest_path.is_file():
        raise DatasetError(f"checksum manifest not found: {manifest_path}; {DOWNLOAD_HINT}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    actual = sha256_file(raw_path)
    if actual != expected:
        raise DatasetError(
            f"SHA-256 mismatch for {raw_path}: expected {expected} (from {MANIFEST_FILE}), got {actual}. "
            f"The file was modified or corrupted; restore it from git or {DOWNLOAD_HINT} again."
        )

    frame = pd.read_csv(raw_path, header=None, names=list(COLUMNS), na_values="?")
    if frame["num"].isna().any():
        raise DatasetError(f"{raw_path}: the target column 'num' has missing values")
    X = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    y = (frame["num"].to_numpy() > 0).astype(np.int64)
    return DatasetBundle(
        name="heart_cleveland",
        X=X,
        y=y,
        feature_names=list(FEATURES),
        class_names=list(CLASS_NAMES),
        meta={
            "source": f"UCI Heart Disease (id {manifest.get('uci_id')}), {RAW_FILE}",
            "source_url": manifest.get("source_url_used"),
            "citation": manifest.get("citation"),
            "license": manifest.get("license"),
            "sha256": actual,
            "sha256_arrays": sha256_arrays(X, y),
            "target": "num > 0 -> 1 (disease); num = 0 -> 0 (no disease)",
        },
    )
