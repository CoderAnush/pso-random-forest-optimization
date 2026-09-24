"""UT-22: import layering (ADR-023).

``pso_rf.optimization`` imports only NumPy, the standard library and itself; ``pso_rf.evaluation`` never
imports ``pso_rf.optimization``; the optimizers' constructors take no data.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import pso_rf

pytestmark = pytest.mark.isolation

PACKAGE_DIR = Path(pso_rf.__file__).resolve().parent
STDLIB = frozenset(sys.stdlib_module_names)


def imported_modules(source: str, module: str, is_package: bool) -> list[str]:
    """Absolute names a source file imports; ``from a import b`` also yields ``a.b`` (b may be a module)."""
    package = module if is_package else module.rpartition(".")[0]
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def _is_optimization(name: str) -> bool:
    return name == "pso_rf.optimization" or name.startswith("pso_rf.optimization.")


def _allowed_in_optimization(name: str) -> bool:
    top = name.split(".")[0]
    return top == "numpy" or top in STDLIB or _is_optimization(name)


def _violations(subpackage: str, is_bad: Callable[[str], bool]) -> list[str]:
    files = sorted((PACKAGE_DIR / subpackage).rglob("*.py"))
    assert files, f"no Python files found under pso_rf/{subpackage}/"
    found = []
    for path in files:
        parts = list(path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        for name in imported_modules(path.read_text(encoding="utf-8"), module, is_package):
            if is_bad(name):
                found.append(f"{module}: imports {name}")
    return found


def test_optimization_imports_only_numpy_stdlib_and_itself() -> None:
    assert _violations("optimization", lambda name: not _allowed_in_optimization(name)) == []


def test_evaluation_never_imports_optimization() -> None:
    assert _violations("evaluation", _is_optimization) == []


@pytest.mark.parametrize(
    "source",
    [
        "import sklearn",
        "from sklearn.ensemble import RandomForestClassifier",
        "import pandas as pd",
        "import scipy.stats",
        "from pso_rf.evaluation import fitness",
        "from pso_rf import models",
        "from ..datasets import load_dataset",
        "from ..experiments import runner",
        "from .. import utils",
    ],
)
def test_scanner_flags_forbidden_imports(source: str) -> None:
    names = imported_modules(source, "pso_rf.optimization.pso", is_package=False)
    assert not all(_allowed_in_optimization(name) for name in names)


def test_scanner_accepts_allowed_imports() -> None:
    source = (
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from dataclasses import dataclass\n"
        "from collections.abc import Sequence\n"
        "from . import events\n"
        "from .search_space import SearchSpace\n"
        "from pso_rf.optimization.events import Evaluation\n"
    )
    names = imported_modules(source, "pso_rf.optimization.pso", is_package=False)
    assert all(_allowed_in_optimization(name) for name in names)


def test_scanner_flags_optimization_imports_from_evaluation() -> None:
    for source in ("from pso_rf import optimization", "from ..optimization.pso import PSOConfig"):
        names = imported_modules(source, "pso_rf.evaluation.fitness", is_package=False)
        assert any(_is_optimization(name) for name in names)


@pytest.mark.parametrize(
    ("module_name", "class_name", "expected"),
    [
        ("pso_rf.optimization.pso", "PSOOptimizer", ["space", "objective", "config", "rng"]),
        ("pso_rf.optimization.random_search", "RandomSearch", ["space", "objective", "budget", "rng"]),
    ],
)
def test_optimizer_constructors_take_no_data(module_name: str, class_name: str, expected: list[str]) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        pytest.skip(f"{module_name} is added in a later phase")
    cls = getattr(module, class_name, None)
    if cls is None:
        pytest.skip(f"{module_name}.{class_name} is added in a later phase")
    parameters = [name for name in inspect.signature(cls.__init__).parameters if name != "self"]
    assert parameters == expected
