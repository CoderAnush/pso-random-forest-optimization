"""UT-18: configuration layering, validation, JSON round trip and config-hash stability."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml

from pso_rf.experiments.config import (
    KNOWN_DATASETS,
    ConfigError,
    ExperimentConfig,
    load_config,
    parse_override,
)
from pso_rf.optimization.pso import PSOConfig
from pso_rf.preprocessing.pipeline import PreprocessingSpec


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def default_yaml(configs_dir: Path) -> Path:
    return configs_dir / "default.yaml"


def test_default_yaml_equals_builtin_defaults(default_yaml: Path) -> None:
    cfg = load_config([default_yaml])
    assert cfg == ExperimentConfig().resolved()
    assert cfg == load_config([])
    assert cfg.config_hash() == load_config([]).config_hash()


def test_defaults_are_the_design_values(default_yaml: Path) -> None:
    cfg = load_config([default_yaml])
    assert cfg.pso == PSOConfig()
    assert (cfg.pso.n_particles, cfg.pso.max_iter) == (10, 20)
    assert (cfg.pso.w, cfg.pso.c1, cfg.pso.c2) == (0.7298, 1.49618, 1.49618)
    assert (cfg.pso.v_max_frac, cfg.pso.v_init_frac) == (0.2, 0.1)
    assert (cfg.pso.boundary, cfg.pso.topology, cfg.pso.update) == ("absorb", "gbest", "synchronous")
    assert (cfg.pso.patience_enabled, cfg.pso.patience_tol, cfg.pso.patience_iterations) == (False, 1e-4, 5)
    bounds = {name: (b.low, b.high) for name, b in cfg.search_space.params.items()}
    assert bounds == {"n_estimators": (50, 200), "max_depth": (2, 20), "min_samples_split": (2, 10)}
    assert list(bounds) == ["n_estimators", "max_depth", "min_samples_split"]  # position dimension order
    assert (cfg.split.outer_folds, cfg.split.inner_folds, cfg.split.outer_seed) == (5, 5, 42)
    assert (cfg.split.run_seeds, cfg.split.deployment_seed) == ((0, 1, 2, 3, 4), 5)
    assert (cfg.fitness.metric, cfg.fitness.class_ratio_gate) == ("accuracy", 1.5)
    assert (cfg.fitness.cache, cfg.fitness.n_jobs_folds) == (True, 5)
    assert cfg.fitness.diagnostics == ("balanced_accuracy", "f1_macro")
    assert cfg.random_search.budget == 10 * (20 + 1)
    assert (cfg.random_forest.n_jobs, cfg.baseline.params) == (1, {})
    assert cfg.experiment.datasets == KNOWN_DATASETS
    assert cfg.experiment.methods == ("baseline", "random_search", "pso")
    assert cfg.experiment.folds is None and cfg.experiment.deployment_run is True
    assert cfg.for_dataset("heart_cleveland").preprocessing == PreprocessingSpec(impute="most_frequent")
    assert cfg.for_dataset("iris").preprocessing == PreprocessingSpec()
    assert cfg.for_dataset("digits").preprocessing == PreprocessingSpec()


def test_preprocessing_can_be_overridden_per_dataset(default_yaml: Path) -> None:
    cfg = load_config([default_yaml], ["datasets.heart_cleveland.preprocessing.impute=median"])
    assert cfg.for_dataset("heart_cleveland").preprocessing == PreprocessingSpec(impute="median")
    off = load_config([default_yaml], ["datasets.heart_cleveland.preprocessing.impute=null"])
    assert off.for_dataset("heart_cleveland").preprocessing == PreprocessingSpec()


def test_layer_order_is_default_then_file_then_dataset_then_cli(default_yaml: Path, tmp_path: Path) -> None:
    layer = _write(
        tmp_path / "experiment.yaml",
        {"pso": {"max_iter": 7, "w": 0.6}, "datasets": {"digits": {"pso": {"max_iter": 9}}}},
    )
    cfg = load_config([default_yaml, layer])
    assert cfg.pso.max_iter == 7  # file over default
    assert cfg.pso.n_particles == 10  # untouched default
    assert cfg.for_dataset("iris").pso.max_iter == 7  # no dataset override: global value
    assert cfg.for_dataset("digits").pso.max_iter == 9  # dataset block over file
    assert cfg.for_dataset("digits").pso.w == 0.6  # keys the block does not set fall through

    cli = load_config([default_yaml, layer], ["pso.max_iter=11"])
    assert cli.pso.max_iter == 11
    assert cli.for_dataset("digits").pso.max_iter == 11  # CLI over dataset block

    targeted = load_config([default_yaml, layer], ["datasets.digits.pso.max_iter=13"])
    assert targeted.pso.max_iter == 7
    assert targeted.for_dataset("digits").pso.max_iter == 13


def test_later_files_override_earlier_files(default_yaml: Path, tmp_path: Path) -> None:
    first = _write(tmp_path / "a.yaml", {"pso": {"n_particles": 8, "max_iter": 4}})
    second = _write(tmp_path / "b.yaml", {"pso": {"n_particles": 12}})
    cfg = load_config([default_yaml, first, second])
    assert (cfg.pso.n_particles, cfg.pso.max_iter) == (12, 4)


def test_demo_and_test_layers(configs_dir: Path, test_config: ExperimentConfig) -> None:
    demo = load_config([configs_dir / "default.yaml", configs_dir / "demo.yaml"])
    assert demo.experiment.name == "demo"
    assert (demo.experiment.datasets, demo.experiment.folds) == (("iris",), (0,))
    assert demo.experiment.deployment_run is False
    assert (demo.pso.n_particles, demo.pso.max_iter) == (6, 5)
    assert demo.random_search.budget == 6 * (5 + 1)

    assert test_config.experiment.name == "test"
    assert test_config.experiment.datasets == KNOWN_DATASETS
    assert test_config.experiment.deployment_run is False
    assert (test_config.pso.n_particles, test_config.pso.max_iter) == (4, 2)
    assert (test_config.split.inner_folds, test_config.fitness.n_jobs_folds) == (3, 1)
    assert test_config.random_search.budget == 4 * (2 + 1)


def test_budget_auto_follows_pso_and_an_explicit_budget_wins(default_yaml: Path, tmp_path: Path) -> None:
    assert load_config([default_yaml], ["pso.n_particles=6"]).random_search.budget == 6 * 21
    assert load_config([default_yaml], ["random_search.budget=50"]).random_search.budget == 50

    layer = _write(tmp_path / "per_dataset.yaml", {"datasets": {"digits": {"pso": {"max_iter": 30}}}})
    cfg = load_config([default_yaml, layer])
    assert cfg.random_search.budget == 210
    assert cfg.for_dataset("iris").random_search_budget == 210
    assert cfg.for_dataset("digits").random_search_budget == 10 * 31  # equal budget per dataset
    assert cfg.to_dict()["datasets"]["digits"]["random_search"] == {"budget": 310}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("search_space.max_depth.low=20", "low must be < high"),
        ("search_space.max_depth.high=1", "low must be < high"),
        ("search_space.n_estimators.low=50.5", "low and high must be integers"),
        ("split.run_seeds=[0, 1, 2]", "exactly outer_folds"),
        ("split.outer_folds=3", "exactly outer_folds"),
        ("split.inner_folds=1", "split.inner_folds"),
        ("fitness.metric=f1_macro", "fitness.metric"),
        ("fitness.class_ratio_gate=0.5", "fitness.class_ratio_gate"),
        ("pso.n_particles=1", "pso.n_particles"),
        ("pso.max_iter=0", "pso.max_iter"),
        ("pso.w=0", "pso.w"),
        ("pso.c1=-0.5", "pso.c1"),
        ("pso.boundary=reflect", "pso.boundary"),
        ("experiment.datasets=[iris, mnist]", "unknown name(s) ['mnist']"),
        ("experiment.methods=[pso, grid_search]", "unknown name(s) ['grid_search']"),
        ("experiment.folds=[5]", "experiment.folds"),
        ("random_search.budget=0", "random_search.budget"),
        ("datasets.digits.pso.n_particles=1", "datasets.digits.pso.n_particles"),
        ("datasets.heart_cleveland.preprocessing.impute=constant", "preprocessing.impute"),
        ("datasets.mnist.pso.max_iter=3", "unknown dataset"),
    ],
)
def test_validation_rejects_bad_values(default_yaml: Path, override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config([default_yaml], [override])


@pytest.mark.parametrize(
    "override",
    ["pso.max_iters=30", "fitnes.metric=accuracy", "split.run_seed=1", "datasets.iris.search_space.x=1"],
)
def test_unknown_keys_are_rejected(default_yaml: Path, override: str) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config([default_yaml], [override])


def test_all_problems_are_reported_together(default_yaml: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_config([default_yaml], ["pso.n_particles=1", "fitness.metric=log_loss"])
    assert "pso.n_particles" in str(info.value) and "fitness.metric" in str(info.value)


def test_exponent_without_decimal_point_gets_a_hint(default_yaml: Path) -> None:
    with pytest.raises(ConfigError, match=re.escape("1.0e-4")):
        load_config([default_yaml], ["pso.patience.tol=1e-4"])
    assert load_config([default_yaml], ["pso.patience.tol=1.0e-3"]).pso.patience_tol == 1e-3


def test_resolved_config_round_trips_through_json(configs_dir: Path, tmp_path: Path) -> None:
    for layer in (None, "demo.yaml", "test.yaml"):
        paths = [configs_dir / "default.yaml"] + ([configs_dir / layer] if layer else [])
        cfg = load_config(paths)
        text = json.dumps(cfg.to_dict())
        restored = ExperimentConfig.from_dict(json.loads(text))
        assert restored == cfg
        assert restored.config_hash() == cfg.config_hash()
        # config.resolved.json is itself a complete config: loading it alone reproduces the experiment
        resolved_file = tmp_path / "config.resolved.json"
        resolved_file.write_text(text, encoding="utf-8")
        assert load_config([resolved_file]) == cfg


def test_config_hash_is_stable_and_sensitive(default_yaml: Path, tmp_path: Path) -> None:
    digest = load_config([default_yaml]).config_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert load_config([default_yaml]).config_hash() == digest

    data = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    reordered = {key: data[key] for key in reversed(list(data))}
    reordered["pso"] = {key: data["pso"][key] for key in reversed(list(data["pso"]))}
    assert load_config([_write(tmp_path / "reordered.yaml", reordered)]).config_hash() == digest

    assert load_config([default_yaml], ["pso.w=0.72"]).config_hash() != digest
    as_int = load_config([default_yaml], ["pso.w=1"])
    assert as_int.pso.w == 1.0 and isinstance(as_int.pso.w, float)
    assert as_int.config_hash() == load_config([default_yaml], ["pso.w=1.0"]).config_hash()


def test_override_values_are_parsed_as_yaml(default_yaml: Path) -> None:
    cfg = load_config(
        [default_yaml],
        [
            "experiment.datasets=[iris, digits]",
            "fitness.class_ratio_gate=null",
            "experiment.deployment_run=false",
        ],
    )
    assert cfg.experiment.datasets == ("iris", "digits")
    assert cfg.fitness.class_ratio_gate is None  # gate disabled
    assert cfg.experiment.deployment_run is False
    assert parse_override("pso.max_iter=30") == (["pso", "max_iter"], 30)
    assert parse_override("experiment.name=a=b") == (["experiment", "name"], "a=b")


@pytest.mark.parametrize("bad", ["pso.max_iter", "=3", "pso..max_iter=3", "pso.max_iter=[1,"])
def test_malformed_overrides_are_rejected(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_methods_run_in_canonical_order(default_yaml: Path) -> None:
    cfg = load_config([default_yaml], ["experiment.methods=[pso, baseline]"])
    assert cfg.experiment.methods == ("baseline", "pso")


def test_for_dataset_rejects_unknown_names(test_config: ExperimentConfig) -> None:
    with pytest.raises(ConfigError, match="unknown dataset"):
        test_config.for_dataset("mnist")


def test_yaml_layers_must_be_mappings(default_yaml: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config([default_yaml, empty]) == load_config([default_yaml])
    as_list = tmp_path / "list.yaml"
    as_list.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config([default_yaml, as_list])


def test_config_is_frozen(test_config: ExperimentConfig) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        test_config.pso = PSOConfig()  # type: ignore[misc]
