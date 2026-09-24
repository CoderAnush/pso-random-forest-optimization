"""Helpers shared by the integration tests (reading result files, rebuilding modified folds)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pso_rf.evaluation import FoldData, HeldOutTestSet
from pso_rf.experiments.recorder import TIMING_COLUMNS


def read_rows(path: Path, drop_timing: bool = True) -> list[dict[str, str]]:
    """CSV rows as dicts of strings; timing columns (which may differ between re-runs) are dropped."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if drop_timing:
        rows = [{k: v for k, v in row.items() if k not in TIMING_COLUMNS} for row in rows]
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def with_test_data(fold: FoldData, X: np.ndarray | None = None, y: np.ndarray | None = None) -> FoldData:
    """The same fold with its held-out test features and/or labels replaced (optimization data unchanged)."""
    X_test, y_test = fold.test.reveal()
    test = HeldOutTestSet(X_test if X is None else X, y_test if y is None else y, fold.test.indices)
    return FoldData(fold.fold_index, fold.opt, test)
