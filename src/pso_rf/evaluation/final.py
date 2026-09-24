"""Final evaluation: refit on the optimization portion, then score the held-out test fold once (MLR-005).

This is the only code path that reads test data, and it refuses to run inside an optimization phase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pso_rf.evaluation.metrics import compute_metrics
from pso_rf.evaluation.splits import (
    HeldOutTestSet,
    OptimizationData,
    TestSetAccessError,
    optimization_phase_active,
)
from pso_rf.models.random_forest import build_model
from pso_rf.preprocessing.pipeline import PreprocessingSpec


@dataclass(frozen=True)
class FinalEvaluation:
    """Test metrics plus the per-row predictions (``row_index`` = original dataset index)."""

    metrics: dict[str, Any]
    row_index: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def final_evaluate(
    config: Mapping[str, int | None],
    opt: OptimizationData,
    test: HeldOutTestSet,
    seed: int,
    preprocessing: PreprocessingSpec,
    labels: Sequence[int],
    positive_label: int | None = None,
    n_jobs: int = 1,
) -> FinalEvaluation:
    """Fit ``config`` on all of ``opt`` (``random_state = seed``), predict ``test``, compute its metrics."""
    if optimization_phase_active():
        raise TestSetAccessError("final_evaluate must run after the optimization phase has closed")
    model = build_model(config, seed, preprocessing, n_jobs).fit(opt.X, opt.y)
    X_test, y_test = test.reveal()
    y_pred = model.predict(X_test)
    return FinalEvaluation(
        metrics=compute_metrics(y_test, y_pred, labels, positive_label),
        row_index=test.indices.copy(),
        y_true=y_test,
        y_pred=np.asarray(y_pred),
    )
