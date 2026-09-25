"""Checks on saved experiment files: completeness, temporal isolation, and run-to-run reproducibility.

* :func:`verify_results` is the experiment-level audit (ET-02 completeness, ET-03 summary consistency,
  ET-04 temporal isolation). It returns a list of problems; an empty list means the results pass.
* :func:`compare_runs` compares two result directories, ignoring identity and timing fields (IT-08, RR-005).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pso_rf.experiments.config import ExperimentConfig
from pso_rf.experiments.summary import (
    FOLD_COLUMNS,
    SUMMARY_COLUMNS,
    fold_rows,
    load_final_records,
    summary_rows,
)
from pso_rf.utils.hashing import sha256_json
from pso_rf.utils.io import format_csv_value

IGNORED_KEYS: frozenset[str] = frozenset(
    {"exp_id", "timestamp", "fit_time_s", "elapsed_s", "total_time_s", "total_time_s_sum"}
)
SKIPPED_FILES: frozenset[str] = frozenset({"run.log", "manifest.json"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- verification


def verify_results(root: Path) -> list[str]:
    """Audit one experiment directory against its own ``config.resolved.json``; return the problems found."""
    root = Path(root)
    problems: list[str] = []
    try:
        raw_config = _read_json(root / "config.resolved.json")
        manifest = _read_json(root / "manifest.json")
    except FileNotFoundError as exc:
        return [f"missing {exc.filename}"]
    cfg = ExperimentConfig.from_dict(raw_config)
    if sha256_json(raw_config) != manifest.get("config_hash"):
        problems.append("config.resolved.json does not match the manifest's config_hash")
    if manifest.get("status") != "completed":
        problems.append(f"manifest status is {manifest.get('status')!r}, not 'completed'")

    folds = cfg.experiment.folds if cfg.experiment.folds is not None else tuple(range(cfg.split.outer_folds))
    for dataset in cfg.experiment.datasets:
        settings = cfg.for_dataset(dataset)
        pso_budget = settings.pso.n_particles * (settings.pso.max_iter + 1)
        if not (root / dataset / "audit.json").exists():
            problems.append(f"{dataset}: audit.json missing")
        for k in folds:
            for method in cfg.experiment.methods:
                problems += _check_run(root / dataset / f"fold_{k}" / method, method, settings, pso_budget)
        problems += _check_coverage(root, dataset, cfg.experiment.methods, folds, cfg.split.outer_folds)
        if cfg.experiment.deployment_run:
            deployment = root / dataset / "deployment" / "pso"
            for name in ("deployment.json", "evaluations.csv", "iterations.csv"):
                if not (deployment / name).exists():
                    problems.append(f"{dataset}: deployment/{name} missing")
    problems += _check_summaries(root, cfg)
    problems += _check_log(root / "run.log")
    return problems


def _check_run(run_dir: Path, method: str, settings: Any, pso_budget: int) -> list[str]:
    label = "/".join(run_dir.parts[-3:])
    final_path = run_dir / "final.json"
    if not final_path.exists():
        return [f"{label}: final.json missing"]
    problems = []
    final = _read_json(final_path)
    predictions = run_dir / "predictions.csv"
    if not predictions.exists():
        problems.append(f"{label}: predictions.csv missing")
    elif len(_read_csv(predictions)) != final["n_test_samples"]:
        problems.append(f"{label}: predictions.csv rows != n_test_samples")
    if not final["test_evaluated_at"] > final["optimization_finished_at"]:
        problems.append(f"{label}: test evaluated before the optimization phase ended (ET-04)")
    if method in ("random_search", "pso"):
        rows = _read_csv(run_dir / "evaluations.csv") if (run_dir / "evaluations.csv").exists() else []
        expected = settings.random_search_budget if method == "random_search" else pso_budget
        if method == "pso" and settings.pso.patience_enabled:
            expected = settings.pso.n_particles * (final["n_iterations"] + 1)
        if len(rows) != expected or final["n_evaluations"] != expected:
            problems.append(f"{label}: {len(rows)} evaluation rows, expected {expected}")
    if method == "pso":
        path = run_dir / "iterations.csv"
        n = len(_read_csv(path)) if path.exists() else 0
        if n != final["n_iterations"] + 1:
            problems.append(f"{label}: iterations.csv has {n} rows, expected {final['n_iterations'] + 1}")
    return problems


def _check_coverage(root: Path, dataset: str, methods: Any, folds: Any, n_folds: int) -> list[str]:
    """With every outer fold run, each method's test predictions must cover each row exactly once."""
    if len(folds) != n_folds or not (root / dataset / "audit.json").exists():
        return []
    n_samples = _read_json(root / dataset / "audit.json")["n_samples"]
    problems = []
    for method in methods:
        indices = []
        for k in folds:
            path = root / dataset / f"fold_{k}" / method / "predictions.csv"
            if path.exists():
                indices += [int(r["row_index"]) for r in _read_csv(path)]
        if sorted(indices) != list(range(n_samples)):
            problems.append(f"{dataset}/{method}: test folds do not cover every row exactly once")
    return problems


def _check_summaries(root: Path, cfg: ExperimentConfig) -> list[str]:
    """ET-03: both summary CSVs must equal a fresh recomputation from the per-fold files."""
    problems = []
    folds = fold_rows(load_final_records(root), cfg.experiment.datasets)
    expected = {
        "summary_folds.csv": (folds, FOLD_COLUMNS),
        "summary.csv": (summary_rows(root, folds), SUMMARY_COLUMNS),
    }
    for name, (rows, columns) in expected.items():
        path = root / name
        if not path.exists():
            problems.append(f"{name} missing")
            continue
        saved = _read_csv(path)
        fresh = [{c: format_csv_value(row[c]) for c in columns} for row in rows]
        if saved != fresh:
            problems.append(f"{name} differs from a recomputation from the per-fold files")
    return problems


def _check_log(path: Path) -> list[str]:
    """ET-04: the run log must never show the test set revealed while an optimization phase is open."""
    if not path.exists():
        return ["run.log missing"]
    depth, problems = 0, []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if "optimization phase opened" in line:
                depth += 1
            elif "optimization phase closed" in line:
                depth -= 1
            elif "held-out test set revealed" in line and depth > 0:
                problems.append(f"run.log line {number}: test set revealed inside an optimization phase")
    return problems


# ---------------------------------------------------------------------------------------------- comparison


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in IGNORED_KEYS and not k.endswith("_at")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def compare_runs(a: Path, b: Path) -> list[str]:
    """Differences between two result directories, ignoring identity, timestamps and timing fields."""
    a, b = Path(a), Path(b)
    files_a = {
        p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file() and p.name not in SKIPPED_FILES
    }
    files_b = {
        p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file() and p.name not in SKIPPED_FILES
    }
    differences = [f"only in {a.name}: {f}" for f in sorted(files_a - files_b)]
    differences += [f"only in {b.name}: {f}" for f in sorted(files_b - files_a)]
    for name in sorted(files_a & files_b):
        pa, pb = a / name, b / name
        if name.endswith(".csv"):
            ra, rb = _strip(_read_csv(pa)), _strip(_read_csv(pb))
        elif name.endswith(".json"):
            ra, rb = _strip(_read_json(pa)), _strip(_read_json(pb))
            if name == "config.resolved.json":
                ra["experiment"].pop("name", None), rb["experiment"].pop("name", None)
        else:
            ra, rb = pa.read_bytes(), pb.read_bytes()
        if ra != rb:
            differences.append(f"differs: {name}")
    return differences
