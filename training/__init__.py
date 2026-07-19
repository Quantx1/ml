"""
PR 128 — unified training pipeline.

Per the locked memory directive (project_unified_training_plan_2026_04_19):
every ML feature ships a *trainer module* under ``ml/training/trainers/``.
Trainers do not run inline at PR time; they're discovered and executed by
``ml/training/runner.py`` in a single end-to-end GPU run after the PR flow
finishes.

Public surface:

    from ml.training import discover_trainers, Trainer

    for t in discover_trainers():
        t.train(out_dir=...)

The runner additionally calls ``evaluate`` and ``register`` per trainer.
"""

from .base import Trainer, TrainerError, TrainResult
from .cpcv import CPCVConfig, combinatorial_purged_split, extract_backtest_paths, n_paths
from .discovery import discover_trainers
from .optuna_search import OptunaConfig, SearchSpace, run_optuna_search
from .wfcv import WFCVConfig, aggregate_fold_metrics, walk_forward_split

__all__ = [
    "CPCVConfig",
    "OptunaConfig",
    "SearchSpace",
    "Trainer",
    "TrainerError",
    "TrainResult",
    "WFCVConfig",
    "aggregate_fold_metrics",
    "combinatorial_purged_split",
    "discover_trainers",
    "extract_backtest_paths",
    "n_paths",
    "run_optuna_search",
    "walk_forward_split",
]
