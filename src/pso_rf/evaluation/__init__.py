"""Outer folds, test-set guard, fitness evaluator and final metrics.

This package must never import ``pso_rf.optimization`` (ADR-023, enforced by UT-22).
"""

from pso_rf.evaluation.splits import (
    FoldData,
    HeldOutTestSet,
    OptimizationData,
    OptimizationPhase,
    TestSetAccessError,
    full_data,
    optimization_phase_active,
    outer_folds,
)

__all__ = [
    "FoldData",
    "HeldOutTestSet",
    "OptimizationData",
    "OptimizationPhase",
    "TestSetAccessError",
    "full_data",
    "optimization_phase_active",
    "outer_folds",
]
