"""Experiment configuration: layered YAML, validation and frozen dataclasses (ARCHITECTURE §H, ADR-025).

Layers, lowest precedence first:

1. built-in defaults, identical to ``configs/default.yaml``;
2. each YAML file given to :func:`load_config`, in order (deep-merged);
3. the ``datasets.<name>`` block, merged into the global ``pso`` / ``fitness`` / ``random_search`` sections by
   :meth:`ExperimentConfig.for_dataset`;
4. CLI overrides ``a.b.c=value`` (the value is parsed with ``yaml.safe_load``). An override of a global
   ``pso``, ``fitness`` or ``random_search`` key also replaces that key in every dataset block that sets
   it, so the CLI layer wins over dataset blocks.

A loaded configuration is validated (CONTEXT §16) and resolved: ``random_search.budget: auto`` becomes
``n_particles × (max_iter + 1)`` (per dataset when a dataset overrides PSO), float fields hold floats, and
``experiment.methods`` is put in the canonical run order baseline → random_search → pso (CONTEXT §7).
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from pso_rf.optimization.pso import PSOConfig
from pso_rf.preprocessing.pipeline import IMPUTE_STRATEGIES, PreprocessingSpec
from pso_rf.utils.hashing import sha256_json

KNOWN_DATASETS: tuple[str, ...] = ("iris", "digits", "heart_cleveland")
KNOWN_METHODS: tuple[str, ...] = ("baseline", "random_search", "pso")  # canonical run order
FITNESS_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy")
DIAGNOSTIC_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "f1_macro")
PSO_BOUNDARIES: tuple[str, ...] = ("absorb",)  # "reflect" is a documented extension, not implemented
PSO_TOPOLOGIES: tuple[str, ...] = ("gbest",)
PSO_UPDATES: tuple[str, ...] = ("synchronous",)
DATASET_SECTIONS: tuple[str, ...] = ("pso", "fitness", "random_search", "preprocessing")

_CLI_WINS_SECTIONS = ("pso", "fitness", "random_search")
_PATIENCE_KEYS = ("enabled", "tol", "iterations")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_SEED = 2**32 - 1


class ConfigError(ValueError):
    """An invalid configuration file, override or value."""


# --------------------------------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class ExperimentSection:
    """What to run and where to write it. ``folds=None`` runs every outer fold."""

    name: str = "default"
    datasets: tuple[str, ...] = KNOWN_DATASETS
    methods: tuple[str, ...] = KNOWN_METHODS
    deployment_run: bool = True
    results_dir: str = "results"
    plots_dir: str = "plots"
    data_dir: str = "data"
    folds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SplitConfig:
    """Outer/inner cross-validation and the seeds (CONTEXT §17)."""

    outer_folds: int = 5
    outer_seed: int = 42
    inner_folds: int = 5
    run_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    deployment_seed: int = 5


@dataclass(frozen=True)
class FitnessConfig:
    """Fitness metric, class-ratio gate (``None`` disables it), diagnostics, cache and fold parallelism."""

    metric: str = "accuracy"
    class_ratio_gate: float | None = 1.5
    diagnostics: tuple[str, ...] = ("balanced_accuracy", "f1_macro")
    cache: bool = True
    n_jobs_folds: int = 5


@dataclass(frozen=True)
class IntBounds:
    """Inclusive integer bounds of one hyperparameter."""

    low: int
    high: int


def _default_params() -> dict[str, IntBounds]:
    return {
        "n_estimators": IntBounds(50, 200),
        "max_depth": IntBounds(2, 20),
        "min_samples_split": IntBounds(2, 10),
    }


@dataclass(frozen=True)
class SearchSpaceConfig:
    """Ordered hyperparameter bounds; the order is the dimension order of a particle position."""

    params: dict[str, IntBounds] = field(default_factory=_default_params)


@dataclass(frozen=True)
class RandomSearchConfig:
    """Random search budget; ``"auto"`` resolves to the PSO budget ``n_particles × (max_iter + 1)``."""

    budget: int | str = "auto"


@dataclass(frozen=True)
class RFConfig:
    """Random Forest settings that are not decision variables (all others stay at scikit-learn defaults)."""

    n_jobs: int = 1


@dataclass(frozen=True)
class BaselineConfig:
    """Baseline Random Forest parameters; ``{}`` means ``RandomForestClassifier()`` defaults."""

    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetOverride:
    """A ``datasets.<name>`` block: partial ``pso`` / ``fitness`` / ``random_search`` overrides, plus the
    dataset's own ``preprocessing`` (there is no global preprocessing section)."""

    pso: dict[str, Any] = field(default_factory=dict)
    fitness: dict[str, Any] = field(default_factory=dict)
    random_search: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSettings:
    """Effective settings for one dataset, as returned by :meth:`ExperimentConfig.for_dataset`."""

    name: str
    pso: PSOConfig
    fitness: FitnessConfig
    preprocessing: PreprocessingSpec
    random_search_budget: int


def _default_datasets() -> dict[str, DatasetOverride]:
    return {
        "iris": DatasetOverride(),
        "digits": DatasetOverride(),
        "heart_cleveland": DatasetOverride(preprocessing={"impute": "most_frequent"}),
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """The complete experiment configuration. Build it with :func:`load_config`."""

    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    split: SplitConfig = field(default_factory=SplitConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    random_search: RandomSearchConfig = field(default_factory=RandomSearchConfig)
    random_forest: RFConfig = field(default_factory=RFConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    datasets: dict[str, DatasetOverride] = field(default_factory=_default_datasets)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        """Build a configuration from a nested mapping in the YAML layout (missing sections take defaults)."""
        raw = _mapping(raw, "config")
        _check_keys(raw, [f.name for f in fields(cls)], "config")
        return cls(**{name: _SECTION_PARSERS[name](value) for name, value in raw.items()})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable nested dict in the YAML layout (content of ``config.resolved.json``)."""
        return {
            "experiment": _section_to_raw(self.experiment),
            "split": _section_to_raw(self.split),
            "fitness": _section_to_raw(self.fitness),
            "search_space": {n: {"low": b.low, "high": b.high} for n, b in self.search_space.params.items()},
            "pso": _pso_to_raw(self.pso),
            "random_search": _section_to_raw(self.random_search),
            "random_forest": _section_to_raw(self.random_forest),
            "baseline": _section_to_raw(self.baseline),
            "datasets": {name: _override_to_raw(o) for name, o in self.datasets.items()},
        }

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`to_dict` (RESULTS_SCHEMA §2)."""
        return sha256_json(self.to_dict())

    def for_dataset(self, name: str) -> DatasetSettings:
        """Effective PSO, fitness, preprocessing and random-search budget for one dataset."""
        if name not in KNOWN_DATASETS:
            raise ConfigError(f"unknown dataset {name!r}; known: {list(KNOWN_DATASETS)}")
        override = self.datasets.get(name, DatasetOverride())
        pso = _floats(_effective_pso(self, name, override))
        fitness = _floats(_effective_fitness(self, name, override))
        budget = _resolve_budget(override.random_search.get("budget", self.random_search.budget), pso)
        return DatasetSettings(
            name=name,
            pso=pso,
            fitness=fitness,
            preprocessing=_preprocessing_spec(name, override),
            random_search_budget=budget,
        )

    def validate(self) -> None:
        """Check every value (CONTEXT §16); raise :class:`ConfigError` listing all problems found."""
        errors = _Errors()
        _validate_experiment(self.experiment, self.split, errors)
        _validate_split(self.split, errors)
        _validate_fitness(self.fitness, "fitness", errors)
        _validate_search_space(self.search_space, errors)
        _validate_pso(self.pso, "pso", errors)
        _validate_budget(self.random_search.budget, "random_search.budget", errors)
        n_jobs = self.random_forest.n_jobs
        errors.check(_is_n_jobs(n_jobs), "random_forest.n_jobs", "must be a positive integer or -1", n_jobs)
        params = self.baseline.params
        errors.check(
            isinstance(params, dict) and all(isinstance(key, str) for key in params),
            "baseline.params",
            "must be a mapping of RandomForestClassifier parameters ({} = defaults)",
            params,
        )
        for name, override in self.datasets.items():
            _validate_override(self, name, override, errors)
        if errors:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))

    def resolved(self) -> ExperimentConfig:
        """Return the canonical form of a validated config.

        Budgets become integers, float fields hold floats and methods follow the run order. A dataset whose
        effective budget differs from the global one records its own budget in its block.
        """
        pso = _floats(self.pso)
        budget = _resolve_budget(self.random_search.budget, pso)
        datasets = {}
        for name, override in self.datasets.items():
            random_search = dict(override.random_search)
            raw_budget = random_search.get("budget", self.random_search.budget)
            dataset_budget = _resolve_budget(raw_budget, _effective_pso(self, name, override))
            if "budget" in random_search or dataset_budget != budget:
                random_search["budget"] = dataset_budget
            datasets[name] = replace(override, random_search=random_search)
        methods = tuple(method for method in KNOWN_METHODS if method in self.experiment.methods)
        return replace(
            self,
            experiment=replace(self.experiment, methods=methods),
            fitness=_floats(self.fitness),
            pso=pso,
            random_search=replace(self.random_search, budget=budget),
            datasets=datasets,
        )


# --------------------------------------------------------------------------------------------------- loading


def load_config(paths: Sequence[Path | str], overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load, merge, validate and resolve a configuration (layer order in the module docstring)."""
    merged = ExperimentConfig().to_dict()
    for path in paths:
        merged = deep_merge(merged, _read_yaml(Path(path)))
    for item in overrides:
        parts, value = parse_override(item)
        _apply_override(merged, parts, value)
    config = ExperimentConfig.from_dict(merged)
    config.validate()
    return config.resolved()


def deep_merge(base: Mapping[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` updated with ``layer``: mappings merge recursively, other values replace."""
    merged = copy.deepcopy(dict(base))
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(item: str) -> tuple[list[str], Any]:
    """Split a CLI override ``a.b.c=value`` into its key path and its YAML-parsed value."""
    key, sep, text = item.partition("=")
    parts = [part.strip() for part in key.split(".")]
    if not sep or not all(parts):
        raise ConfigError(f"override {item!r}: expected key.path=value, e.g. pso.max_iter=30")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {item!r}: the value is not valid YAML ({exc})") from exc
    return parts, value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return dict(data)


def _apply_override(merged: dict[str, Any], parts: list[str], value: Any) -> None:
    _set_path(merged, parts, value)
    if parts[0] in _CLI_WINS_SECTIONS:
        for block in (merged.get("datasets") or {}).values():
            if isinstance(block, dict) and _has_path(block, parts):
                _set_path(block, parts, value)


def _set_path(root: dict[str, Any], parts: list[str], value: Any) -> None:
    node = root
    for depth, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise ConfigError(f"override {'.'.join(parts)}: {'.'.join(parts[: depth + 1])} is not a mapping")
        node = child
    leaf = parts[-1]
    if isinstance(value, Mapping) and isinstance(node.get(leaf), Mapping):
        node[leaf] = deep_merge(node[leaf], value)
    else:
        node[leaf] = copy.deepcopy(value)


def _has_path(root: Mapping[str, Any], parts: list[str]) -> bool:
    node: Any = root
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


# --------------------------------------------------------------------------------------------------- parsing


def _mapping(value: Any, where: str, *, allow_none: bool = False) -> dict[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__} {value!r}")
    return dict(value)


def _check_keys(raw: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = [key for key in raw if key not in allowed]
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {list(allowed)}")


def _freeze(value: Any) -> Any:
    """Lists become tuples (dataclass fields stay immutable); mappings are deep-copied."""
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze` for JSON: tuples become lists, mappings become plain dicts."""
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    return value


def _section_from_raw(cls: type, raw: Any, where: str) -> Any:
    raw = _mapping(raw, where)
    _check_keys(raw, [f.name for f in fields(cls)], where)
    return cls(**{key: _freeze(value) for key, value in raw.items()})


def _section_to_raw(section: Any) -> dict[str, Any]:
    return {f.name: _thaw(getattr(section, f.name)) for f in fields(section)}


def _pso_from_raw(raw: Any, where: str) -> PSOConfig:
    raw = _mapping(raw, where)
    patience = _mapping(raw.pop("patience", {}), f"{where}.patience", allow_none=True)
    _check_keys(patience, _PATIENCE_KEYS, f"{where}.patience")
    plain = [f.name for f in fields(PSOConfig) if not f.name.startswith("patience_")]
    _check_keys(raw, [*plain, "patience"], where)
    kwargs = dict(raw)
    kwargs.update({f"patience_{key}": value for key, value in patience.items()})
    return PSOConfig(**kwargs)


def _pso_to_raw(pso: PSOConfig) -> dict[str, Any]:
    raw = {f.name: getattr(pso, f.name) for f in fields(pso) if not f.name.startswith("patience_")}
    raw["patience"] = {
        "enabled": pso.patience_enabled,
        "tol": pso.patience_tol,
        "iterations": pso.patience_iterations,
    }
    return raw


def _search_space_from_raw(raw: Any, where: str = "search_space") -> SearchSpaceConfig:
    raw = _mapping(raw, where)
    params = {}
    for name, bounds in raw.items():
        bounds = _mapping(bounds, f"{where}.{name}")
        _check_keys(bounds, ("low", "high"), f"{where}.{name}")
        missing = [key for key in ("low", "high") if key not in bounds]
        if missing:
            raise ConfigError(f"{where}.{name}: missing {missing}")
        params[str(name)] = IntBounds(low=bounds["low"], high=bounds["high"])
    return SearchSpaceConfig(params=params)


def _datasets_from_raw(raw: Any, where: str = "datasets") -> dict[str, DatasetOverride]:
    raw = _mapping(raw, where)
    overrides = {}
    for name, block in raw.items():
        block = _mapping(block, f"{where}.{name}", allow_none=True)
        _check_keys(block, DATASET_SECTIONS, f"{where}.{name}")
        sections = {
            key: _mapping(value, f"{where}.{name}.{key}", allow_none=True) for key, value in block.items()
        }
        overrides[str(name)] = DatasetOverride(**copy.deepcopy(sections))
    return overrides


def _override_to_raw(override: DatasetOverride) -> dict[str, Any]:
    return {f.name: _thaw(getattr(override, f.name)) for f in fields(override) if getattr(override, f.name)}


_SECTION_PARSERS = {
    "experiment": lambda raw: _section_from_raw(ExperimentSection, raw, "experiment"),
    "split": lambda raw: _section_from_raw(SplitConfig, raw, "split"),
    "fitness": lambda raw: _section_from_raw(FitnessConfig, raw, "fitness"),
    "search_space": _search_space_from_raw,
    "pso": lambda raw: _pso_from_raw(raw, "pso"),
    "random_search": lambda raw: _section_from_raw(RandomSearchConfig, raw, "random_search"),
    "random_forest": lambda raw: _section_from_raw(RFConfig, raw, "random_forest"),
    "baseline": lambda raw: _section_from_raw(BaselineConfig, raw, "baseline"),
    "datasets": _datasets_from_raw,
}


def _effective_pso(config: ExperimentConfig, name: str, override: DatasetOverride) -> PSOConfig:
    return _pso_from_raw(deep_merge(_pso_to_raw(config.pso), override.pso), f"datasets.{name}.pso")


def _effective_fitness(config: ExperimentConfig, name: str, override: DatasetOverride) -> FitnessConfig:
    merged = deep_merge(_section_to_raw(config.fitness), override.fitness)
    return _section_from_raw(FitnessConfig, merged, f"datasets.{name}.fitness")


def _preprocessing_spec(name: str, override: DatasetOverride) -> PreprocessingSpec:
    where = f"datasets.{name}.preprocessing"
    _check_keys(override.preprocessing, ("impute",), where)
    try:
        return PreprocessingSpec(**override.preprocessing)
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _resolve_budget(budget: int | str, pso: PSOConfig) -> int:
    return pso.n_particles * (pso.max_iter + 1) if budget == "auto" else budget


def _floats(section: Any) -> Any:
    """Copy of a dataclass with ints in float-annotated fields turned into floats (canonical JSON/hash)."""
    changes = {
        f.name: float(getattr(section, f.name))
        for f in fields(section)
        if "float" in str(f.type) and _is_int(getattr(section, f.name))
    }
    return replace(section, **changes) if changes else section


# ------------------------------------------------------------------------------------------------ validation


class _Errors(list):
    """Collects validation messages so that every problem is reported at once."""

    def check(self, ok: bool, where: str, message: str, value: Any) -> None:
        if not ok:
            self.append(f"{where}: {message} (got {value!r}){_hint(value)}")


def _hint(value: Any) -> str:
    """Explain the common YAML 1.1 trap: ``1e-4`` (no decimal point) is read as a string, not a float."""
    if not isinstance(value, str):
        return ""
    try:
        float(value)
    except ValueError:
        return ""
    if "e" in value.lower():
        return "; YAML reads an exponent number without a decimal point as text: write e.g. 1.0e-4"
    return "; write the number without quotes"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_seed(value: Any) -> bool:
    return _is_int(value) and 0 <= value <= _MAX_SEED


def _is_n_jobs(value: Any) -> bool:
    return _is_int(value) and (value >= 1 or value == -1)


def _validate_names(values: Any, allowed: Sequence[str], where: str, errors: _Errors) -> None:
    ok = isinstance(values, tuple) and len(values) > 0 and all(isinstance(value, str) for value in values)
    errors.check(ok, where, "must be a non-empty list of names", values)
    if ok:
        unknown = [value for value in values if value not in allowed]
        errors.check(not unknown, where, f"unknown name(s) {unknown}; allowed: {list(allowed)}", values)
        errors.check(len(set(values)) == len(values), where, "must not repeat a name", values)


def _validate_experiment(experiment: ExperimentSection, split: SplitConfig, errors: _Errors) -> None:
    name = experiment.name
    errors.check(
        isinstance(name, str) and bool(_NAME_RE.match(name)),
        "experiment.name",
        "must be a name made of letters, digits, '_', '.' or '-'",
        name,
    )
    _validate_names(experiment.datasets, KNOWN_DATASETS, "experiment.datasets", errors)
    _validate_names(experiment.methods, KNOWN_METHODS, "experiment.methods", errors)
    errors.check(
        isinstance(experiment.deployment_run, bool),
        "experiment.deployment_run",
        "must be true or false",
        experiment.deployment_run,
    )
    for key in ("results_dir", "plots_dir", "data_dir"):
        value = getattr(experiment, key)
        errors.check(
            isinstance(value, str) and value.strip() != "", f"experiment.{key}", "must be a path", value
        )
    folds = experiment.folds
    if folds is not None:
        ok = isinstance(folds, tuple) and len(folds) > 0 and all(_is_int(k) for k in folds)
        ok = ok and len(set(folds)) == len(folds)
        errors.check(
            ok, "experiment.folds", "must be null (all folds) or a list of distinct fold indices", folds
        )
        if ok and _is_int(split.outer_folds):
            in_range = all(0 <= k < split.outer_folds for k in folds)
            errors.check(
                in_range, "experiment.folds", f"indices must lie in 0..{split.outer_folds - 1}", folds
            )


def _validate_split(split: SplitConfig, errors: _Errors) -> None:
    for key in ("outer_folds", "inner_folds"):
        value = getattr(split, key)
        errors.check(_is_int(value) and value >= 2, f"split.{key}", "must be an integer >= 2", value)
    for key in ("outer_seed", "deployment_seed"):
        value = getattr(split, key)
        errors.check(_is_seed(value), f"split.{key}", "must be an integer seed in [0, 2**32 - 1]", value)
    seeds = split.run_seeds
    seeds_ok = isinstance(seeds, tuple) and all(_is_seed(seed) for seed in seeds)
    errors.check(seeds_ok, "split.run_seeds", "must be a list of integer seeds in [0, 2**32 - 1]", seeds)
    if seeds_ok and _is_int(split.outer_folds):
        errors.check(
            len(seeds) == split.outer_folds,
            "split.run_seeds",
            f"must have exactly outer_folds = {split.outer_folds} entries",
            seeds,
        )


def _validate_fitness(fitness: FitnessConfig, where: str, errors: _Errors) -> None:
    errors.check(
        fitness.metric in FITNESS_METRICS,
        f"{where}.metric",
        f"must be one of {list(FITNESS_METRICS)}",
        fitness.metric,
    )
    gate = fitness.class_ratio_gate
    errors.check(
        gate is None or (_is_number(gate) and gate >= 1),
        f"{where}.class_ratio_gate",
        "must be null (gate disabled) or a number >= 1",
        gate,
    )
    diagnostics = fitness.diagnostics
    ok = isinstance(diagnostics, tuple) and all(d in DIAGNOSTIC_METRICS for d in diagnostics)
    errors.check(
        ok and len(set(diagnostics)) == len(diagnostics),
        f"{where}.diagnostics",
        f"must be a list of distinct names from {list(DIAGNOSTIC_METRICS)}",
        diagnostics,
    )
    errors.check(isinstance(fitness.cache, bool), f"{where}.cache", "must be true or false", fitness.cache)
    errors.check(
        _is_n_jobs(fitness.n_jobs_folds),
        f"{where}.n_jobs_folds",
        "must be a positive integer or -1",
        fitness.n_jobs_folds,
    )


def _validate_search_space(space: SearchSpaceConfig, errors: _Errors) -> None:
    errors.check(
        len(space.params) > 0, "search_space", "must define at least one hyperparameter", space.params
    )
    for name, bounds in space.params.items():
        where = f"search_space.{name}"
        errors.check(name.isidentifier(), where, "the name must be a Python identifier", name)
        ints = _is_int(bounds.low) and _is_int(bounds.high)
        errors.check(ints, where, "low and high must be integers", (bounds.low, bounds.high))
        if ints:
            errors.check(bounds.low < bounds.high, where, "low must be < high", (bounds.low, bounds.high))


def _validate_pso(pso: PSOConfig, where: str, errors: _Errors) -> None:
    errors.check(
        _is_int(pso.n_particles) and pso.n_particles >= 2,
        f"{where}.n_particles",
        "must be an integer >= 2",
        pso.n_particles,
    )
    errors.check(
        _is_int(pso.max_iter) and pso.max_iter >= 1,
        f"{where}.max_iter",
        "must be an integer >= 1",
        pso.max_iter,
    )
    errors.check(_is_number(pso.w) and pso.w > 0, f"{where}.w", "must be a number > 0", pso.w)
    for key in ("c1", "c2"):
        value = getattr(pso, key)
        errors.check(_is_number(value) and value >= 0, f"{where}.{key}", "must be a number >= 0", value)
    errors.check(
        _is_number(pso.v_max_frac) and pso.v_max_frac > 0,
        f"{where}.v_max_frac",
        "must be a number > 0",
        pso.v_max_frac,
    )
    errors.check(
        _is_number(pso.v_init_frac) and pso.v_init_frac >= 0,
        f"{where}.v_init_frac",
        "must be a number >= 0",
        pso.v_init_frac,
    )
    for key, allowed in (("boundary", PSO_BOUNDARIES), ("topology", PSO_TOPOLOGIES), ("update", PSO_UPDATES)):
        value = getattr(pso, key)
        errors.check(
            value in allowed,
            f"{where}.{key}",
            f"must be one of {list(allowed)} (others are not implemented)",
            value,
        )
    errors.check(
        isinstance(pso.patience_enabled, bool),
        f"{where}.patience.enabled",
        "must be true or false",
        pso.patience_enabled,
    )
    errors.check(
        _is_number(pso.patience_tol) and pso.patience_tol >= 0,
        f"{where}.patience.tol",
        "must be a number >= 0",
        pso.patience_tol,
    )
    errors.check(
        _is_int(pso.patience_iterations) and pso.patience_iterations >= 1,
        f"{where}.patience.iterations",
        "must be an integer >= 1",
        pso.patience_iterations,
    )


def _validate_budget(budget: Any, where: str, errors: _Errors) -> None:
    errors.check(
        budget == "auto" or (_is_int(budget) and budget >= 1),
        where,
        "must be 'auto' or a positive integer",
        budget,
    )


def _validate_override(
    config: ExperimentConfig, name: str, override: DatasetOverride, errors: _Errors
) -> None:
    where = f"datasets.{name}"
    if name not in KNOWN_DATASETS:
        errors.append(f"{where}: unknown dataset; known: {list(KNOWN_DATASETS)}")
        return
    try:
        _check_keys(override.random_search, ("budget",), f"{where}.random_search")
        _check_keys(override.preprocessing, ("impute",), f"{where}.preprocessing")
        pso = _effective_pso(config, name, override)
        fitness = _effective_fitness(config, name, override)
    except ConfigError as exc:
        errors.append(str(exc))
        return
    if override.pso:
        _validate_pso(pso, f"{where}.pso", errors)
    if override.fitness:
        _validate_fitness(fitness, f"{where}.fitness", errors)
    if "budget" in override.random_search:
        _validate_budget(override.random_search["budget"], f"{where}.random_search.budget", errors)
    impute = override.preprocessing.get("impute")
    errors.check(
        impute is None or impute in IMPUTE_STRATEGIES,
        f"{where}.preprocessing.impute",
        f"must be null or one of {list(IMPUTE_STRATEGIES)}",
        impute,
    )
