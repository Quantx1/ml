"""Declarative per-engine contract for the canonical training spine.

An engine = one EngineSpec (these dataclasses) + the PipelineTrainer hooks
(build_features/build_labels/make_model/...). The spine reads the spec to
parameterize the shared stages (EDA thresholds, CV windows, eval gates) so
every engine produces a uniform metrics dict in model_versions.metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class CVSpec:
    """Purged walk-forward CV windows (passed to purged_walk_forward_by_date)."""
    n_folds: int = 5
    test_days: int = 63
    embargo_days: int = 20
    train_days: int = 378


@dataclass
class EvalSpec:
    """What 'good' means + the promote/quality gate thresholds."""
    task: str = "ranking"            # "ranking" | "classification" | "regression"
    primary_metric: str = "rank_ic_mean"
    min_ic: float = 0.02             # ranking: OOS mean rank-IC floor
    min_icir: float = 0.5            # ranking: IC information ratio floor
    min_deflated_sharpe: float = 0.0  # 0 disables the DSR gate (reported regardless)
    max_pbo: float = 1.0             # 1 disables the PBO gate (reported regardless)
    # Column IC/decile-spread/HPO rank predictions against. None (default) =
    # the raw fwd_return_col. Engines whose LABEL is a transformed return
    # (e.g. beta-residualized) must point this at that column, or the gate
    # grades the model on a target it was never trained to rank — an
    # unfalsifiable experiment. fwd_return_col itself always stays RAW: the
    # portfolio backtest reads it for realized money.
    ic_target_col: Optional[str] = None
    # --- classification gates (task="classification" only; defaults = off) --
    min_auc: float = 0.0              # mean OOS AUC floor (0 disables)
    min_fold_auc: float = 0.5         # no single fold may score below this
    min_tercile_lift: float = 0.0     # top-pred-tercile win rate - base rate
    require_brier_beats_climatology: bool = False  # Brier <= r*(1-r) of base rate


@dataclass
class EDASpec:
    """Stage-1 pre-train audit thresholds (ml/preprocessing/eda.py)."""
    max_nan_pct: float = 0.50
    min_abs_ic: float = 0.005
    max_leakage_corr: float = 0.95
    run_ic_leakage: bool = True       # ranking/regression: IC + leakage gates
    check_class_balance: bool = False  # classification only
    min_class_pct: float = 0.05
    expected_classes: Optional[Sequence[Any]] = None
    max_constant_features: int = 5     # Stage-2 audit_feature_matrix fatal cap


@dataclass
class EngineSpec:
    """The full declarative contract for one engine."""
    name: str
    horizon: int = 20
    label_col: str = "relevance"      # the y column build_labels emits
    fwd_return_col: str = "fwd_return"  # the realized-return column for evaluation
    hpo_trials: int = 0               # >0 enables Optuna; momentum sets e.g. 30
    cv: CVSpec = field(default_factory=CVSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    eda: EDASpec = field(default_factory=EDASpec)


__all__ = ["CVSpec", "EvalSpec", "EDASpec", "EngineSpec"]
