"""Reading saved experiments for the dashboard and replay pages (files only; nothing is recomputed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_LABEL = {"iris": "Iris", "digits": "Digits", "heart_cleveland": "Heart Disease (Cleveland)"}
DATASET_TASK = {
    "iris": "3-class flower species",
    "digits": "10-class handwritten digits (8×8)",
    "heart_cleveland": "binary heart-disease diagnosis",
}


@dataclass(frozen=True)
class Experiment:
    """One saved experiment directory ``results/<exp_id>/``."""

    root: Path
    config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def exp_id(self) -> str:
        return self.root.name

    @property
    def complete(self) -> bool:
        return self.manifest.get("status") == "completed"

    @property
    def datasets(self) -> list[str]:
        return [d for d in self.config["experiment"]["datasets"] if (self.root / d).is_dir()]

    def summary(self) -> pd.DataFrame:
        path = self.root / "summary.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def summary_folds(self) -> pd.DataFrame:
        path = self.root / "summary_folds.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def folds_of(self, dataset: str) -> list[int]:
        return sorted(int(p.name.split("_")[1]) for p in (self.root / dataset).glob("fold_*") if p.is_dir())

    def run_dir(self, dataset: str, fold: int | None, method: str) -> Path:
        if fold is None:
            return self.root / dataset / "deployment" / method
        return self.root / dataset / f"fold_{fold}" / method

    def evaluations(self, dataset: str, fold: int | None, method: str) -> pd.DataFrame:
        path = self.run_dir(dataset, fold, method) / "evaluations.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def iterations(self, dataset: str, fold: int | None, method: str = "pso") -> pd.DataFrame:
        path = self.run_dir(dataset, fold, method) / "iterations.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def final(self, dataset: str, fold: int, method: str) -> dict[str, Any] | None:
        path = self.run_dir(dataset, fold, method) / "final.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def deployment(self, dataset: str) -> dict[str, Any] | None:
        path = self.root / dataset / "deployment" / "pso" / "deployment.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def audit(self, dataset: str) -> dict[str, Any] | None:
        path = self.root / dataset / "audit.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def predictions(self, dataset: str, method: str) -> pd.DataFrame:
        paths = sorted(self.root.glob(f"{dataset}/fold_*/{method}/predictions.csv"))
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()

    def all_evaluations(self, dataset: str, method: str) -> pd.DataFrame:
        paths = sorted(self.root.glob(f"{dataset}/fold_*/{method}/evaluations.csv"))
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()

    def all_iterations(self, dataset: str) -> pd.DataFrame:
        paths = sorted(self.root.glob(f"{dataset}/fold_*/pso/iterations.csv"))
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def list_experiments(results_dir: Path) -> list[Experiment]:
    """Experiment directories (those with a manifest), newest first; the ``live`` folder is excluded."""
    found = []
    if not results_dir.is_dir():
        return found
    for root in sorted(results_dir.iterdir(), reverse=True):
        if root.name == "live" or not (root / "manifest.json").exists():
            continue
        try:
            config = json.loads((root / "config.resolved.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found.append(Experiment(root, config, manifest))
    return found
