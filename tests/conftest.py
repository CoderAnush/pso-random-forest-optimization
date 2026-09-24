"""Shared test fixtures (TESTING_STRATEGY: fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pso_rf.experiments.config import ExperimentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """The ``configs/`` directory."""
    return repo_root / "configs"


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    """The ``data/`` directory (``data/raw/`` holds the Cleveland file and its manifest)."""
    return repo_root / "data"


@pytest.fixture
def test_config(configs_dir: Path) -> ExperimentConfig:
    """The minimal-budget configuration used by tests: default.yaml ← test.yaml."""
    return load_config([configs_dir / "default.yaml", configs_dir / "test.yaml"])
