"""The experiment manifest: environment, git state and dataset checksums (RESULTS_SCHEMA §3, RR-004)."""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scikit-learn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("pyyaml", "PyYAML"),
    ("joblib", "joblib"),
    ("scipy", "scipy"),
)


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages the results depend on (``None`` if one is missing)."""
    versions: dict[str, str | None] = {}
    for key, dist in PACKAGES:
        try:
            versions[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state(cwd: Path, exclude: Sequence[str] = ()) -> tuple[str | None, bool | None]:
    """``(HEAD commit, dirty?)`` for the repository at ``cwd``.

    Paths in ``exclude`` (the results and plots directories, which the run itself writes) are ignored when
    deciding whether the tree is dirty. Both values are ``None`` outside a git repository.
    """
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return None, None
    pathspec = [".", *(f":(exclude){path}" for path in exclude)]
    status = _git(["status", "--porcelain", "--", *pathspec], cwd)
    return commit, None if status is None else status != ""


def build_manifest(
    exp_id: str,
    created_at: str,
    command: str,
    config_hash: str,
    datasets: Mapping[str, Mapping[str, Any]],
    repo_dir: Path,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``manifest.json`` record at the start of a run (``status = "running"``)."""
    commit, dirty = git_state(repo_dir, exclude)
    return {
        "exp_id": exp_id,
        "created_at": created_at,
        "finished_at": None,
        "command": command,
        "config_hash": config_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "datasets": {name: dict(info) for name, info in datasets.items()},
        "status": "running",
    }
