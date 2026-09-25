"""ET-01 … ET-04 on one small experiment (test.yaml, outer fold 0, all datasets, with deployment runs),
plus run-level reproducibility (IT-08) and seed sensitivity (IT-09)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pso_rf.experiments.cli import main
from pso_rf.experiments.config import load_config
from pso_rf.experiments.runner import run_experiment, run_fold
from pso_rf.experiments.verify import compare_runs, verify_results
from tests.integration.helpers import read_rows

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def experiment(tmp_path_factory, configs_dir) -> Path:
    results = tmp_path_factory.mktemp("results")
    cfg = load_config(
        [configs_dir / "default.yaml", configs_dir / "test.yaml"],
        [
            "experiment.folds=[0]",
            "experiment.deployment_run=true",
            f"experiment.results_dir={results.as_posix()}",
        ],
    )
    return run_experiment(cfg, command="pytest", exp_id="et")


def test_et01_cli_smoke(tmp_path, configs_dir) -> None:
    code = main(
        [
            "run",
            "--config",
            str(configs_dir / "default.yaml"),
            "--config",
            str(configs_dir / "test.yaml"),
            "--datasets",
            "iris",
            "--folds",
            "1",
            "--methods",
            "baseline",
            "pso",
            "--set",
            f"experiment.results_dir={tmp_path.as_posix()}",
        ]
    )
    assert code == 0
    (root,) = tmp_path.iterdir()
    assert sorted(p.name for p in (root / "iris" / "fold_1").iterdir()) == ["baseline", "pso"]
    assert main(["verify", "--results", str(root)]) == 0


def test_et02_completeness_and_et04_temporal_isolation(experiment: Path) -> None:
    assert verify_results(experiment) == []
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert set(manifest["datasets"]) == {"iris", "digits", "heart_cleveland"}
    assert manifest["datasets"]["heart_cleveland"]["sha256_raw"]
    for dataset in ("iris", "digits", "heart_cleveland"):
        deployment = json.loads(
            (experiment / dataset / "deployment" / "pso" / "deployment.json").read_text("utf-8")
        )
        assert deployment["performance_estimate"]["source"] == "summary.csv"
        assert (
            deployment["n_samples"]
            == json.loads((experiment / dataset / "audit.json").read_text("utf-8"))["n_samples"]
        )


def test_et02_detects_a_missing_file(experiment: Path, tmp_path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(experiment, copy)
    (copy / "iris" / "fold_0" / "pso" / "predictions.csv").unlink()
    assert any("predictions.csv missing" in p for p in verify_results(copy))


def test_et03_summary_consistency(experiment: Path) -> None:
    folds = read_rows(experiment / "summary_folds.csv")
    summary = read_rows(experiment / "summary.csv")
    assert len(summary) == 9 and len(folds) == 9
    for row in folds:
        final = json.loads(
            (experiment / row["dataset"] / "fold_0" / row["method"] / "final.json").read_text("utf-8")
        )
        base = json.loads(
            (experiment / row["dataset"] / "fold_0" / "baseline" / "final.json").read_text("utf-8")
        )
        assert float(row["test_accuracy"]) == final["test_metrics"]["accuracy"]
        delta = final["test_metrics"]["accuracy"] - base["test_metrics"]["accuracy"]
        assert float(row["delta_accuracy_vs_baseline"]) == delta
    for row in summary:
        with (experiment / row["dataset"] / "fold_0" / row["method"] / "predictions.csv").open(
            encoding="utf-8"
        ) as fh:
            preds = list(csv.DictReader(fh))
        pooled = sum(p["y_true"] == p["y_pred"] for p in preds) / len(preds)
        assert float(row["pooled_test_accuracy"]) == pytest.approx(pooled)
        if row["method"] != "baseline":
            delta = float(
                next(f for f in folds if f["dataset"] == row["dataset"] and f["method"] == row["method"])[
                    "delta_accuracy_vs_baseline"
                ]
            )
            assert (int(row["wins"]), int(row["ties"]), int(row["losses"])) == (
                delta > 0,
                delta == 0,
                delta < 0,
            )


def test_it08_rerun_is_identical_and_it09_seed_matters(
    tmp_path, configs_dir, iris_bundle, iris_fold0
) -> None:
    layers = [configs_dir / "default.yaml", configs_dir / "test.yaml"]
    subset = [
        "experiment.datasets=[iris]",
        "experiment.folds=[0]",
        f"experiment.results_dir={tmp_path.as_posix()}",
    ]
    a = run_experiment(load_config(layers, subset), command="pytest", exp_id="a")
    b = run_experiment(load_config(layers, subset), command="pytest", exp_id="b")
    assert compare_runs(a, b) == []
    # a different run seed must change the trajectory (guards against an ignored seed)
    cfg = load_config(layers, ["split.run_seeds=[1, 1, 2, 3, 4]"])
    run_fold(cfg, iris_bundle, iris_fold0, "pso", "accuracy", tmp_path / "seed1", "s")
    keys = ("n_estimators", "max_depth", "min_samples_split", "fitness")

    def trajectory(path: Path) -> list[tuple[str, ...]]:
        return [tuple(row[k] for k in keys) for row in read_rows(path)]

    seed0 = trajectory(a / "iris" / "fold_0" / "pso" / "evaluations.csv")
    assert trajectory(tmp_path / "seed1" / "evaluations.csv") != seed0


def test_et05_plots_from_files(experiment: Path, tmp_path) -> None:
    import ast

    import pso_rf.visualization.plots as plots_module
    from pso_rf.visualization import make_all_plots

    out = make_all_plots(experiment, tmp_path / "plots")
    sources = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    assert sources["config_hash"] == manifest["config_hash"]
    expected = {f"F{n}" for n in range(3, 12)}
    assert {name.split("_")[0] for name in sources["figures"]} == expected
    for name, files in sources["figures"].items():
        assert (out / name).stat().st_size > 0
        assert files and all((experiment / f).exists() for f in files)
    tree = ast.parse(Path(plots_module.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = ("sklearn", "pso_rf.models", "pso_rf.optimization", "pso_rf.evaluation", "pso_rf.experiments")
    assert not any(m.startswith(forbidden) for m in imported)
