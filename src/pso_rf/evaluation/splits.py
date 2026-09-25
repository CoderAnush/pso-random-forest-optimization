"""Outer folds and the sealed held-out test set (ADR-007, ADR-024).

The test arrays of each outer fold live inside a :class:`HeldOutTestSet` and are reachable only through
:meth:`HeldOutTestSet.reveal`, which raises :class:`TestSetAccessError` while an :class:`OptimizationPhase`
is open.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pso_rf.datasets.base import DatasetBundle

log = logging.getLogger(__name__)

# Open OptimizationPhase contexts, counted per thread: a phase guards the flow of control that opened it.
# Streamlit serves several browser sessions from one process on separate threads, so a process-wide flag would
# let one session's search block another session's legitimate final evaluation (ADR-024).
_state = threading.local()


def _depth() -> int:
    return getattr(_state, "depth", 0)


class TestSetAccessError(RuntimeError):
    """Held-out test data was requested while an optimization phase was open."""

    __test__ = False  # not a pytest test class, despite the name


def optimization_phase_active() -> bool:
    """True while at least one :class:`OptimizationPhase` context is open in the current thread."""
    return _depth() > 0


class OptimizationPhase:
    """Context manager for the optimization phase: :meth:`HeldOutTestSet.reveal` raises inside it.

    The phase is per thread (it guards the code that opened it); nesting is safe (a depth counter), and the
    state is restored even when the block raises.
    """

    def __enter__(self) -> OptimizationPhase:
        _state.depth = _depth() + 1
        log.debug("optimization phase opened (depth %d)", _state.depth)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if _depth() <= 0:
            raise RuntimeError("OptimizationPhase exited more times than it was entered")
        _state.depth = _depth() - 1
        log.debug("optimization phase closed (depth %d)", _state.depth)


def _read_only(array: np.ndarray, dtype: type | None = None) -> np.ndarray:
    """A read-only view of ``array`` (the caller's array and its flags are left untouched)."""
    view = np.asarray(array, dtype=dtype).view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, eq=False)
class OptimizationData:
    """The optimization portion of a fold: read-only ``X``, ``y`` and the original row ``indices``."""

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", _read_only(self.X))
        object.__setattr__(self, "y", _read_only(self.y))
        object.__setattr__(self, "indices", _read_only(self.indices, np.int64))
        if self.X.ndim != 2 or self.y.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("OptimizationData needs a 2-D X and 1-D y and indices")
        if not len(self.X) == len(self.y) == len(self.indices):
            raise ValueError("X, y and indices must have the same number of rows")


class HeldOutTestSet:
    """The held-out test fold. Its arrays are private; :meth:`reveal` is the only way to read them."""

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
        self.__X = _read_only(np.array(X, copy=True))
        self.__y = _read_only(np.array(y, copy=True))
        self.__indices = _read_only(np.array(indices, dtype=np.int64, copy=True))
        if self.__X.ndim != 2 or self.__y.ndim != 1 or self.__indices.ndim != 1:
            raise ValueError("HeldOutTestSet needs a 2-D X and 1-D y and indices")
        if not len(self.__X) == len(self.__y) == len(self.__indices):
            raise ValueError("X, y and indices must have the same number of rows")

    @property
    def indices(self) -> np.ndarray:
        """Original row indices of the test fold (read-only)."""
        return self.__indices

    def __len__(self) -> int:
        return len(self.__indices)

    def reveal(self) -> tuple[np.ndarray, np.ndarray]:
        """Copies of ``(X_test, y_test)``; raises :class:`TestSetAccessError` inside an optimization phase."""
        if optimization_phase_active():
            raise TestSetAccessError(
                f"held-out test data (n={len(self)}) requested while an OptimizationPhase is open; "
                "test data may be read only by the final evaluation, after optimization ends"
            )
        log.debug("held-out test set revealed (n=%d)", len(self))
        return self.__X.copy(), self.__y.copy()

    def __repr__(self) -> str:
        return f"HeldOutTestSet(n={len(self)}, sealed)"


@dataclass(frozen=True)
class FoldData:
    """One outer fold: its index, the optimization portion and the sealed test fold."""

    fold_index: int
    opt: OptimizationData
    test: HeldOutTestSet


def outer_folds(bundle: DatasetBundle, n_splits: int, seed: int) -> list[FoldData]:
    """Split with ``StratifiedKFold(n_splits, shuffle=True, random_state=seed)``; fold k tests part k."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (opt_index, test_index) in enumerate(splitter.split(bundle.X, bundle.y)):
        opt = OptimizationData(bundle.X[opt_index], bundle.y[opt_index], opt_index)
        test = HeldOutTestSet(bundle.X[test_index], bundle.y[test_index], test_index)
        folds.append(FoldData(fold_index=k, opt=opt, test=test))
    return folds


def full_data(bundle: DatasetBundle) -> OptimizationData:
    """All rows as optimization data (the deployment run, which has no test fold)."""
    n = len(bundle.y)
    return OptimizationData(bundle.X.copy(), bundle.y.copy(), np.arange(n, dtype=np.int64))
