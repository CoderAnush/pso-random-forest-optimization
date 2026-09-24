"""IT-04: index disjointness of the optimization and test portions on the real datasets."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from pso_rf.datasets import load_dataset
from pso_rf.evaluation import outer_folds
from pso_rf.experiments.config import KNOWN_DATASETS, load_config

pytestmark = pytest.mark.isolation


@pytest.mark.parametrize("name", KNOWN_DATASETS)
def test_optimization_and_test_indices_never_overlap(name: str, data_dir: Path, configs_dir: Path) -> None:
    split = load_config([configs_dir / "default.yaml"]).split
    bundle = load_dataset(name, data_dir)
    n = len(bundle.y)
    folds = outer_folds(bundle, split.outer_folds, split.outer_seed)
    assert len(folds) == split.outer_folds == 5
    for fold in folds:
        opt, test = set(fold.opt.indices.tolist()), set(fold.test.indices.tolist())
        assert not opt & test  # no row is both optimized on and tested
        assert opt | test == set(range(n))  # together they are the whole dataset
    test_sets = [set(fold.test.indices.tolist()) for fold in folds]
    for a, b in itertools.combinations(test_sets, 2):
        assert not a & b  # the test folds of different runs are disjoint
    assert sorted(itertools.chain.from_iterable(test_sets)) == list(range(n))  # each row tested exactly once
