"""Experiment summaries built only from saved per-fold files (RESULTS_SCHEMA §9–§10, ER-003, ER-004).

``summary_folds.csv`` has one row per dataset × method × fold; ``summary.csv`` one row per dataset × method,
with fold-paired deltas and win/tie/loss counts against the baseline (and PSO against random search). The
statistics are descriptive only: no significance tests (ADR-020).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pso_rf.utils.io import append_csv_rows

METHOD_ORDER: tuple[str, ...] = ("baseline", "random_search", "pso")
HYPERPARAMETERS: tuple[str, ...] = ("n_estimators", "max_depth", "min_samples_split")
TEST_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)

FOLD_COLUMNS: tuple[str, ...] = (
    "exp_id",
    "dataset",
    "method",
    "outer_fold",
    "seed",
    *HYPERPARAMETERS,
    "validation_fitness",
    *(f"test_{m}" for m in TEST_METRICS),
    "delta_accuracy_vs_baseline",
    "delta_f1_macro_vs_baseline",
    "delta_accuracy_vs_random_search",
    "n_evaluations",
    "n_unique_fits",
    "convergence_iteration",
    "total_time_s",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "exp_id",
    "dataset",
    "method",
    "n_folds",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_balanced_accuracy_mean",
    "test_balanced_accuracy_std",
    "test_f1_macro_mean",
    "test_f1_macro_std",
    "pooled_test_accuracy",
    "validation_fitness_mean",
    "delta_accuracy_mean",
    "delta_accuracy_std",
    "wins",
    "ties",
    "losses",
    "wins_vs_rs",
    "ties_vs_rs",
    "losses_vs_rs",
    "n_unique_fits_mean",
    "total_time_s_sum",
)


def load_final_records(root: Path) -> list[dict[str, Any]]:
    """Every ``<dataset>/fold_<k>/<method>/final.json`` under an experiment directory."""
    records = []
    for path in sorted(Path(root).glob("*/fold_*/*/final.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def load_predictions(root: Path, dataset: str, method: str) -> list[tuple[int, int, int]]:
    """All ``(row_index, y_true, y_pred)`` of one dataset and method, concatenated over folds."""
    rows = []
    for path in sorted(Path(root).glob(f"{dataset}/fold_*/{method}/predictions.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            rows += [(int(r["row_index"]), int(r["y_true"]), int(r["y_pred"])) for r in csv.DictReader(fh)]
    return rows


def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def _wtl(deltas: Sequence[float]) -> tuple[int, int, int]:
    return (sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas))


def fold_rows(records: Sequence[dict[str, Any]], dataset_order: Sequence[str]) -> list[dict[str, Any]]:
    """``summary_folds.csv`` rows, sorted by dataset (config order), method (run order) and fold."""
    by_key = {(r["dataset"], r["method"], r["outer_fold"]): r for r in records}
    rows = []
    for record in sorted(
        records,
        key=lambda r: (
            list(dataset_order).index(r["dataset"]),
            METHOD_ORDER.index(r["method"]),
            r["outer_fold"],
        ),
    ):
        dataset, method, fold = record["dataset"], record["method"], record["outer_fold"]
        metrics = record["test_metrics"]
        baseline = by_key.get((dataset, "baseline", fold))
        random_search = by_key.get((dataset, "random_search", fold))
        row = {
            "exp_id": record["exp_id"],
            "dataset": dataset,
            "method": method,
            "outer_fold": fold,
            "seed": record["seed"],
            **{name: record["best_hyperparameters"].get(name) for name in HYPERPARAMETERS},
            "validation_fitness": record["best_validation_fitness"],
            **{f"test_{m}": metrics[m] for m in TEST_METRICS},
            "delta_accuracy_vs_baseline": None,
            "delta_f1_macro_vs_baseline": None,
            "delta_accuracy_vs_random_search": None,
            "n_evaluations": record["n_evaluations"],
            "n_unique_fits": record["n_unique_fits"],
            "convergence_iteration": record["convergence_iteration"],
            "total_time_s": record["total_time_s"],
        }
        if baseline is not None:
            row["delta_accuracy_vs_baseline"] = metrics["accuracy"] - baseline["test_metrics"]["accuracy"]
            row["delta_f1_macro_vs_baseline"] = metrics["f1_macro"] - baseline["test_metrics"]["f1_macro"]
        if method == "pso" and random_search is not None:
            row["delta_accuracy_vs_random_search"] = (
                metrics["accuracy"] - random_search["test_metrics"]["accuracy"]
            )
        rows.append(row)
    return rows


def summary_rows(root: Path, folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """``summary.csv`` rows (one per dataset × method), from the fold rows and the saved predictions."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in folds:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    rows = []
    for (dataset, method), items in groups.items():
        accuracy = [r["test_accuracy"] for r in items]
        predictions = load_predictions(root, dataset, method)
        pooled = float(np.mean([t == p for _, t, p in predictions])) if predictions else math.nan
        deltas = [
            r["delta_accuracy_vs_baseline"] for r in items if r["delta_accuracy_vs_baseline"] is not None
        ]
        vs_rs = [
            r["delta_accuracy_vs_random_search"]
            for r in items
            if r["delta_accuracy_vs_random_search"] is not None
        ]
        wins = ties = losses = None
        if method != "baseline" and deltas:
            wins, ties, losses = _wtl(deltas)
        wins_rs = ties_rs = losses_rs = None
        if method == "pso" and vs_rs:
            wins_rs, ties_rs, losses_rs = _wtl(vs_rs)
        rows.append(
            {
                "exp_id": items[0]["exp_id"],
                "dataset": dataset,
                "method": method,
                "n_folds": len(items),
                "test_accuracy_mean": float(np.mean(accuracy)),
                "test_accuracy_std": _std(accuracy),
                "test_balanced_accuracy_mean": float(np.mean([r["test_balanced_accuracy"] for r in items])),
                "test_balanced_accuracy_std": _std([r["test_balanced_accuracy"] for r in items]),
                "test_f1_macro_mean": float(np.mean([r["test_f1_macro"] for r in items])),
                "test_f1_macro_std": _std([r["test_f1_macro"] for r in items]),
                "pooled_test_accuracy": pooled,
                "validation_fitness_mean": float(np.mean([r["validation_fitness"] for r in items])),
                "delta_accuracy_mean": float(np.mean(deltas)) if deltas else None,
                "delta_accuracy_std": _std(deltas) if deltas else None,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wins_vs_rs": wins_rs,
                "ties_vs_rs": ties_rs,
                "losses_vs_rs": losses_rs,
                "n_unique_fits_mean": float(np.mean([r["n_unique_fits"] for r in items])),
                "total_time_s_sum": float(np.sum([r["total_time_s"] for r in items])),
            }
        )
    return rows


def write_summaries(
    root: Path, dataset_order: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build both summaries from the files under ``root`` and (re)write the two CSVs; return the rows."""
    root = Path(root)
    folds = fold_rows(load_final_records(root), dataset_order)
    summary = summary_rows(root, folds)
    for name, rows, columns in (
        ("summary_folds.csv", folds, FOLD_COLUMNS),
        ("summary.csv", summary, SUMMARY_COLUMNS),
    ):
        path = root / name
        path.unlink(missing_ok=True)
        append_csv_rows(path, rows, columns)
    return folds, summary
