"""UT-11: the RF builder passes exactly the decision variables; the baseline is the default forest."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from pso_rf.models import BASELINE_CONFIG, HYPERPARAMETER_NAMES, build_model
from pso_rf.preprocessing.pipeline import PreprocessingSpec

CONFIG = {"n_estimators": 137, "max_depth": 9, "min_samples_split": 4}
SKLEARN_DEFAULTS = RandomForestClassifier().get_params()


def test_rf_gets_exactly_the_decision_variables_seed_and_n_jobs() -> None:
    pipeline = build_model(CONFIG, seed=3, preprocessing=PreprocessingSpec())
    assert [name for name, _ in pipeline.steps] == ["rf"]
    params = pipeline.named_steps["rf"].get_params()
    expected = {**CONFIG, "random_state": 3, "n_jobs": 1}
    for key, value in expected.items():
        assert params[key] == value, key
    assert set(params) == set(SKLEARN_DEFAULTS)
    for key in set(params) - set(expected):
        assert params[key] == SKLEARN_DEFAULTS[key], key  # every other parameter keeps its default


def test_heart_spec_adds_the_imputer_and_other_datasets_do_not() -> None:
    heart = build_model(CONFIG, seed=0, preprocessing=PreprocessingSpec(impute="most_frequent"))
    assert [name for name, _ in heart.steps] == ["impute", "rf"]
    assert isinstance(heart.named_steps["impute"], SimpleImputer)
    assert heart.named_steps["impute"].strategy == "most_frequent"
    plain = build_model(CONFIG, seed=0, preprocessing=PreprocessingSpec())
    assert [name for name, _ in plain.steps] == ["rf"]


def test_n_jobs_is_passed_through_and_each_call_builds_fresh_objects() -> None:
    assert build_model(CONFIG, 0, PreprocessingSpec(), n_jobs=2).named_steps["rf"].n_jobs == 2
    first = build_model(CONFIG, 0, PreprocessingSpec())
    second = build_model(CONFIG, 0, PreprocessingSpec())
    assert first.named_steps["rf"] is not second.named_steps["rf"]


@pytest.mark.parametrize(
    "config",
    [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "max_features": "sqrt"},
    ],
)
def test_config_keys_must_be_exactly_the_decision_variables(config: dict) -> None:
    with pytest.raises(ValueError, match="config keys"):
        build_model(config, 0, PreprocessingSpec())


def test_baseline_is_the_scikit_learn_default_forest_and_is_read_only() -> None:
    assert dict(BASELINE_CONFIG) == {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
    assert dict(BASELINE_CONFIG) == {key: SKLEARN_DEFAULTS[key] for key in HYPERPARAMETER_NAMES}
    with pytest.raises(TypeError):
        BASELINE_CONFIG["n_estimators"] = 5  # type: ignore[index]
    baseline_rf = build_model(BASELINE_CONFIG, seed=4, preprocessing=PreprocessingSpec()).named_steps["rf"]
    assert baseline_rf.get_params() == RandomForestClassifier(random_state=4, n_jobs=1).get_params()
