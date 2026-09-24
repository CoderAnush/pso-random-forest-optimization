"""Outer folds, test-set guard, fitness evaluator and final metrics.

This package must never import ``pso_rf.optimization`` (ADR-023, enforced by UT-22).
"""

from pso_rf.evaluation.final import FinalEvaluation, final_evaluate
from pso_rf.evaluation.fitness import FitnessEvaluator, FitnessResult
from pso_rf.evaluation.metrics import compute_metrics
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
    "FinalEvaluation",
    "FitnessEvaluator",
    "FitnessResult",
    "FoldData",
    "HeldOutTestSet",
    "OptimizationData",
    "OptimizationPhase",
    "TestSetAccessError",
    "compute_metrics",
    "final_evaluate",
    "full_data",
    "optimization_phase_active",
    "outer_folds",
]
