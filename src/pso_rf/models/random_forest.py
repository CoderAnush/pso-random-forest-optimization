"""Random Forest pipeline builder: the decision variables become constructor arguments (MLR-001..003)."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from pso_rf.preprocessing.pipeline import PreprocessingSpec, build_steps

HYPERPARAMETER_NAMES: tuple[str, ...] = ("n_estimators", "max_depth", "min_samples_split")


def build_model(
    config: Mapping[str, int | None], seed: int, preprocessing: PreprocessingSpec, n_jobs: int = 1
) -> Pipeline:
    """Build an unfitted pipeline: the preprocessing steps, then ``("rf", RandomForestClassifier(...))``.

    ``config`` must have exactly the keys in :data:`HYPERPARAMETER_NAMES`; every other forest parameter keeps
    its scikit-learn default, except ``random_state = seed`` and ``n_jobs``. Values are not checked here:
    scikit-learn validates them at fit time, so an invalid value surfaces as a failed evaluation.
    """
    if set(config) != set(HYPERPARAMETER_NAMES):
        raise ValueError(f"config keys must be exactly {list(HYPERPARAMETER_NAMES)}, got {sorted(config)}")
    forest = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        random_state=seed,
        n_jobs=n_jobs,
    )
    return Pipeline([*build_steps(preprocessing), ("rf", forest)])
