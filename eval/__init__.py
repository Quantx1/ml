"""
PR 162 — model-eval primitives.

Trainer-facing evaluation layer separate from the full strategy
backtester in ``ml/backtest/``. Trainers call these functions to
compute Sharpe, drawdown, Calmar, win-rate, profit-factor, and benchmark-
relative metrics on a per-fold basis. The promote gate (PR 167) reads
the aggregated values to decide whether ``is_prod=TRUE`` is allowed.

The full backtester engine (`ml/backtest/engine.py`) handles strategy
simulation with slippage/brokerage/STT and is used outside of training
(scanner-lab walk-throughs, manual user backtests). This module is
intentionally lightweight: a frozen series of predictions + realized
returns + a benchmark goes in, a dict of metrics comes out.
"""

from .backtest_eval import (
    BacktestEvalConfig,
    compute_backtest_metrics,
    metrics_from_returns,
    promote_gate_passes,
)
from .drift import (
    DriftConfig,
    feature_drift_report,
    ks_drift,
    psi,
    summarize_report,
)
from .impact_cost import (
    DEFAULT_BASE_BPS,
    DEFAULT_IMPACT_COEF,
    ImpactCostConfig,
    apply_impact_to_returns,
    impact_cost_bps,
)
from .kelly import (
    DEFAULT_FRACTIONAL,
    MAX_KELLY_FRACTION,
    kelly_fraction,
    kelly_from_metrics,
)
from .spa import (
    family_wise_t_correction,
    hansen_spa_test,
)
from .overfitting import (
    deflated_sharpe_ratio,
    dsr_pbo_from_fold_returns,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)
from .lambdarank_ic import (
    lambdarank_ic_objective,
    rank_ic_metric,
)

__all__ = [
    "BacktestEvalConfig",
    "DEFAULT_BASE_BPS",
    "DEFAULT_FRACTIONAL",
    "DEFAULT_IMPACT_COEF",
    "DriftConfig",
    "ImpactCostConfig",
    "MAX_KELLY_FRACTION",
    "apply_impact_to_returns",
    "compute_backtest_metrics",
    "deflated_sharpe_ratio",
    "dsr_pbo_from_fold_returns",
    "expected_max_sharpe",
    "family_wise_t_correction",
    "feature_drift_report",
    "hansen_spa_test",
    "impact_cost_bps",
    "kelly_fraction",
    "kelly_from_metrics",
    "ks_drift",
    "lambdarank_ic_objective",
    "rank_ic_metric",
    "metrics_from_returns",
    "probability_of_backtest_overfitting",
    "promote_gate_passes",
    "psi",
    "summarize_report",
]
