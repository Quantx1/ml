"""
PR 162 — backtest-driven metrics for trainer evaluation.

Per-fold metrics that prove a model would actually have made money in
the test window. Promote-gate (PR 167) reads these to allow / deny
``is_prod=TRUE``.

Two entry points:

    metrics_from_returns(strategy_returns, benchmark_returns) -> dict
        Pure metric calculation. Caller has already simulated the
        strategy (e.g. RL env, regime-conditional position sizer).

    compute_backtest_metrics(predictions, forward_returns, benchmark_returns, cfg) -> dict
        End-to-end: turn discrete predictions (BUY=1, SELL=-1, HOLD=0)
        or probabilities into a strategy return series, apply
        per-trade transaction costs, then call metrics_from_returns.

Metric contract (all annualized to 252 trading days where applicable):

    sharpe              float  Risk-adjusted return; > 1 is decent, > 2 great
    max_drawdown_pct    float  Most-negative equity drop; -0.25 = -25 percent
    calmar              float  annualized return / abs(max_dd); > 0.5 to ship
    cagr                float  Compound annual growth rate of strategy
    annualized_vol      float  Stddev x sqrt(252) on daily returns
    total_return_pct    float  Strategy cum-return over test window
    benchmark_return_pct float Same for Nifty buy-and-hold
    excess_return_pct   float  Strategy minus benchmark
    win_rate            float  Fraction of trades with positive PnL
    profit_factor       float  Gross profit / gross loss; > 1.5 to ship
    n_trades            int
    avg_holding_days    float
    primary_metric      str    'sharpe' so promote_gate has a named field
    primary_value       float

All values are JSONB-safe (float | int | str | list of those).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Trading days per year for annualization.
TRADING_DAYS = 252

# Default per-trade round-trip cost (matches ml/backtest/engine.py and
# ml/rl/env.py). Legacy value was 13 bps round-trip but that under-counts
# real NSE costs by ~13 bps — the proper SEBI/exchange breakdown for a
# typical retail discount-broker swing trade (₹100k notional, delivery
# product) is ~27 bps round-trip:
#   buy:  ₹20 brokerage + ₹15 stamp duty + ₹3 exch + ₹4 GST + ₹50 slippage = 9.2 bps
#   sell: ₹20 brokerage + ₹100 STT + ₹3 exch + ₹4 GST + ₹50 slippage      = 17.7 bps
#
# See ml/eval/nse_costs.py for the SEBI/exchange-fee breakdown and
# round_trip_cost_bps() for size-aware computation. Backtest harness
# should default to the SEBI-accurate number; only legacy regressions
# may override down to the old 13 bps assumption.
DEFAULT_COST_BPS = 27.0  # NSE swing trade, ₹100k delivery, discount broker


@dataclass
class BacktestEvalConfig:
    """Configuration for ``compute_backtest_metrics``.

    cost_bps:
        Round-trip transaction cost in basis points. 13 bps = 0.13 percent.
        Applied on every position change.

    confidence_threshold:
        For probability predictions: only trade when |p - 0.5| > threshold.
        Default 0.05 (i.e. p > 0.55 = LONG, p < 0.45 = SHORT).

    direction_neutral:
        If True, predictions are treated as discrete LONG/SHORT/FLAT
        (-1/0/+1). If False, predictions are treated as probabilities
        in [0, 1] for binary up/down models.

    max_position:
        Cap on absolute position size (0.0 = flat, 1.0 = full long,
        -1.0 = full short). Mirrors AutoPilot's per-symbol cap.
    """

    cost_bps: float = DEFAULT_COST_BPS
    confidence_threshold: float = 0.05
    direction_neutral: bool = True
    max_position: float = 1.0


# ============================================================================
# Pure metric calculator
# ============================================================================


def metrics_from_returns(
    strategy_returns: np.ndarray | pd.Series | Iterable[float],
    benchmark_returns: Optional[np.ndarray | pd.Series | Iterable[float]] = None,
) -> dict:
    """Compute Sharpe / drawdown / Calmar / CAGR / vol from a return series.

    ``strategy_returns`` is an array of per-period (typically daily)
    simple returns (e.g. 0.012 = +1.2 percent). Caller is responsible for
    deducting transaction costs upstream.

    ``benchmark_returns`` is the same series for Nifty / NIFTY 50 buy-and-
    hold. When provided, adds excess_return_pct + benchmark fields.
    """
    arr = np.asarray(list(strategy_returns), dtype=float)
    if arr.size == 0:
        return _empty_metrics()
    n = arr.size

    # Annualized stats — Phase 1.7 audit-fix #1
    # The legacy `(1 + daily_mean) ** 252 - 1` formula explodes when daily
    # returns have outliers (e.g. RL policies producing ±100% bars). We saw
    # annualized_return = 5.4e12 on a wiped-out RL holdout because the
    # formula has no upper clamp. Two defenses:
    #   1. Sanitize: drop non-finite bars, clamp per-bar returns to a sane
    #      range before computing mean/std. ±50% per bar is realistic NSE
    #      circuit-limit ceiling; anything beyond is upstream corruption.
    #   2. Use CAGR (from cumulative equity) as the authoritative annualized
    #      return; the (1+mean)^252 formula is reported separately but also
    #      clamped to [-1, 100] (i.e. -100% to +10,000% — saner range than
    #      "trillion-fold" garbage).
    arr_clean = arr[np.isfinite(arr)]
    if arr_clean.size == 0:
        return _empty_metrics()
    arr_clipped = np.clip(arr_clean, -0.50, 0.50)
    n_clipped = int(np.sum(arr_clipped != arr_clean))
    daily_mean = float(arr_clipped.mean())
    daily_std = float(arr_clipped.std(ddof=1)) if arr_clipped.size > 1 else 0.0

    # Equity curve uses UNCLIPPED returns so drawdown reflects reality
    equity = np.cumprod(1.0 + arr_clean)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_dd = float(drawdowns.min()) if drawdowns.size > 0 else 0.0

    # CAGR from total cumulative — immune to mean-explosion
    total_return = float(equity[-1] - 1.0) if equity.size > 0 else 0.0
    years = arr_clean.size / TRADING_DAYS
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and total_return > -1.0 else -1.0

    # Old-style annualized_return, but CLAMPED so it can't explode into
    # garbage. A 100x annual return is already wildly unrealistic; anything
    # claiming more is a sign of corrupted return series.
    if daily_mean > -1.0:
        raw_ann = (1.0 + daily_mean) ** TRADING_DAYS - 1.0
        ann_return = float(np.clip(raw_ann, -1.0, 100.0))   # cap at 10,000% per year
    else:
        ann_return = -1.0
    ann_vol = daily_std * np.sqrt(TRADING_DAYS)

    # Sharpe: mean / std x sqrt(252). Computed on the CLIPPED series so
    # per-bar outliers from corrupted upstream returns don't artificially
    # inflate the mean/std ratio (the "Sharpe 3 but wiped out" pathology).
    sharpe = float((daily_mean / daily_std) * np.sqrt(TRADING_DAYS)) if daily_std > 1e-12 else 0.0

    # Phase 1.7 audit fix #2.5 — clip asymmetry diagnostic. Sharpe is
    # computed on the clipped series while equity / drawdown use the
    # unclipped one. That's intentional (Sharpe defends against
    # corruption; drawdown must reflect what actually happened), but
    # the asymmetry can hide a corrupted input series. Emit the
    # unclipped Sharpe + clip count alongside so a reviewer can spot it.
    # If the gap between clipped and unclipped Sharpe exceeds 0.3 it's
    # the smoking gun for outlier-driven inflation.
    if arr_clean.size > 1:
        unclipped_std = float(arr_clean.std(ddof=1))
        unclipped_mean = float(arr_clean.mean())
        sharpe_unclipped = (
            float((unclipped_mean / unclipped_std) * np.sqrt(TRADING_DAYS))
            if unclipped_std > 1e-12 else 0.0
        )
    else:
        sharpe_unclipped = 0.0

    # Calmar uses the CAGR (geometrically-faithful) over |max_dd|, not
    # the explosive arithmetic-mean annualized return.
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0
    n = int(arr_clean.size)

    out = {
        "sharpe": round(sharpe, 4),
        "sharpe_unclipped": round(sharpe_unclipped, 4),
        "n_clipped_bars": n_clipped,
        "max_drawdown_pct": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "cagr": round(cagr, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_vol": round(ann_vol, 4),
        "total_return_pct": round(total_return, 4),
        "n_periods": int(n),
        "primary_metric": "sharpe",
        "primary_value": round(sharpe, 4),
    }
    # Surface the clipped-vs-unclipped gap so a reviewer can spot
    # outlier-driven Sharpe inflation. A delta > 0.3 reliably signals
    # upstream corruption (RL log-return amplification, missing T+1
    # shift, label-mapping bug) that the clip is concealing.
    if abs(sharpe - sharpe_unclipped) > 0.3 or n_clipped > max(1, int(n * 0.005)):
        out["sharpe_clip_warning"] = (
            f"clipped_sharpe={sharpe:.2f} unclipped={sharpe_unclipped:.2f} "
            f"clipped_bars={n_clipped}/{n} — check for upstream return corruption"
        )

    # Benchmark-relative
    if benchmark_returns is not None:
        bench = np.asarray(list(benchmark_returns), dtype=float)
        if bench.size == arr.size:
            bench_eq = np.cumprod(1.0 + bench)
            bench_total = float(bench_eq[-1] - 1.0)
            out["benchmark_return_pct"] = round(bench_total, 4)
            out["excess_return_pct"] = round(total_return - bench_total, 4)
            # Information ratio: excess return / tracking error
            excess = arr - bench
            te = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
            out["information_ratio"] = (
                round(float(excess.mean() / te) * np.sqrt(TRADING_DAYS), 4)
                if te > 1e-12
                else 0.0
            )

    return out


# ============================================================================
# End-to-end: predictions -> strategy returns -> metrics
# ============================================================================


def compute_backtest_metrics(
    predictions: np.ndarray | pd.Series | Iterable[float],
    forward_returns: np.ndarray | pd.Series | Iterable[float],
    benchmark_returns: Optional[np.ndarray | pd.Series | Iterable[float]] = None,
    cfg: Optional[BacktestEvalConfig] = None,
) -> dict:
    """End-to-end backtest metric computation.

    Args:
        predictions:
            Aligned with ``forward_returns``. Either:
            - Discrete: -1, 0, +1 (SELL, HOLD, BUY) when cfg.direction_neutral
            - Probability: [0, 1] when cfg.direction_neutral=False (binary up)

        forward_returns:
            Realized next-period returns (same length as predictions).

        benchmark_returns:
            Nifty buy-and-hold returns. Same length as forward_returns.

        cfg:
            BacktestEvalConfig.

    Returns: metrics dict from metrics_from_returns() + n_trades, win_rate,
    profit_factor, avg_holding_days.
    """
    cfg = cfg or BacktestEvalConfig()
    preds = np.asarray(list(predictions), dtype=float)
    rets = np.asarray(list(forward_returns), dtype=float)
    if preds.shape != rets.shape:
        raise ValueError(
            f"predictions.shape {preds.shape} != forward_returns.shape {rets.shape}",
        )
    if preds.size == 0:
        return _empty_metrics()

    # Convert predictions to position signals in [-1, 1].
    if cfg.direction_neutral:
        positions = np.clip(preds, -cfg.max_position, cfg.max_position)
    else:
        # Probability gate around 0.5 + threshold.
        upper = 0.5 + cfg.confidence_threshold
        lower = 0.5 - cfg.confidence_threshold
        positions = np.where(
            preds > upper, cfg.max_position,
            np.where(preds < lower, -cfg.max_position, 0.0),
        )

    # Strategy daily return = position x forward_return - cost on changes.
    pos_change = np.abs(np.diff(positions, prepend=0.0))
    cost_per_period = pos_change * (cfg.cost_bps / 10_000.0)
    strategy_returns = positions * rets - cost_per_period

    base = metrics_from_returns(strategy_returns, benchmark_returns)

    # Trade-level metrics: a "trade" is each entry/exit (position changes).
    trades = _extract_trades(positions, rets, cfg.cost_bps)
    base["n_trades"] = int(len(trades))
    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        base["win_rate"] = round(len(wins) / len(trades), 4)
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        base["profit_factor"] = (
            round(gross_profit / gross_loss, 4) if gross_loss > 1e-12 else 0.0
        )
        base["avg_holding_days"] = round(
            float(np.mean([t["holding_days"] for t in trades])), 2
        )
    else:
        base["win_rate"] = 0.0
        base["profit_factor"] = 0.0
        base["avg_holding_days"] = 0.0

    # Average daily turnover (proxy for transaction-cost drag).
    base["avg_daily_turnover"] = round(float(np.mean(pos_change)), 4)
    return base


# Backwards-compat alias name kept off the module surface; the package
# __init__ re-exports the canonical name.
backtest_eval = compute_backtest_metrics


# ============================================================================
# Promote gate
# ============================================================================


# Default thresholds — set conservatively for v1 real-money launch.
# Calibrated from publicly-available NSE strategy backtests; any model
# that doesn't beat these is worse than tier-1 retail systematic
# strategies.
DEFAULT_PROMOTE_THRESHOLDS = {
    # Real-money production bars (Phase 1.0 reframe). These match the
    # acceptance criteria in the build plan; any model promoted under
    # these thresholds is shippable to live capital.
    "min_sharpe": 1.0,
    "max_drawdown_pct": -0.25,          # drawdown shallower than -25 percent
    "min_calmar": 0.5,
    "min_profit_factor": 1.5,
    "min_n_trades": 30,                  # avoid lucky-streak promotes
    "min_excess_return_pct": 0.05,       # beat Nifty by >= 5 percent over test window
    # PR 174 — Bailey/López de Prado overfit defenses. DSR adjusts Sharpe
    # for multiple-testing bias (N hyperparameter trials inflate Sharpe);
    # PBO is the probability the in-sample-best model under-performs OOS.
    # Phase 1.0: tightened to 0.5 / 0.4 and made mandatory via
    # `require_overfit_defenses` below.
    "min_deflated_sharpe": 0.5,
    "max_pbo": 0.4,
    # W2 (pre-training audit 2026-05-19) — flipped to True. Production
    # money-running trainers MUST clear DSR + PBO. Zero-shot models
    # (momentum_timesfm) and validation-tier shadows (finrl_x_*) can
    # opt-out via their own promote_thresholds class attr:
    #   promote_thresholds = {"require_overfit_defenses": False}
    # PROD set (lgbm_signal_gate, qlib_alpha158, tft_swing, intraday_lstm)
    # inherits this True default — no manual opt-in needed.
    "require_overfit_defenses": True,
    # Regime-stratified Sharpe floor. Models that work only in bull
    # markets are a liability for live capital. Phase 1.0: enforce a
    # minimum Sharpe in EACH regime that has > 50 backtest days.
    # Trainers emit metrics["regime_sharpe"] = {"bull": x, "sideways": y, "bear": z}.
    # When absent, the check is informational-only (no block) until the
    # walk-forward harness backfills it for every trainer.
    "min_regime_stratified_sharpe": 0.5,
}


def promote_gate_passes(
    metrics: dict,
    thresholds: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """Decide whether a model's metrics warrant ``is_prod=TRUE``.

    Returns (passed, failure_reasons). When ``passed`` is False the
    runner records the reasons in ``model_versions.notes`` so the
    trainer team knows why it was held back.

    The thresholds dict can be overridden per trainer when a model has
    a known different risk profile (e.g. AutoPilot's drawdown ceiling
    is tighter than swing's because it leverages the portfolio).
    """
    th = {**DEFAULT_PROMOTE_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []

    sharpe = float(metrics.get("sharpe_mean", metrics.get("sharpe", 0.0)))
    if sharpe < th["min_sharpe"]:
        reasons.append(f"Sharpe {sharpe:.2f} < min {th['min_sharpe']}")

    max_dd = float(metrics.get("max_drawdown_pct_mean", metrics.get("max_drawdown_pct", 0.0)))
    if max_dd < th["max_drawdown_pct"]:
        reasons.append(
            f"Max drawdown {max_dd*100:.1f} percent deeper than allowed {th['max_drawdown_pct']*100:.0f} percent",
        )

    calmar = float(metrics.get("calmar_mean", metrics.get("calmar", 0.0)))
    if calmar < th["min_calmar"]:
        reasons.append(f"Calmar {calmar:.2f} < min {th['min_calmar']}")

    pf = float(metrics.get("profit_factor_mean", metrics.get("profit_factor", 0.0)))
    if pf < th["min_profit_factor"]:
        reasons.append(f"Profit factor {pf:.2f} < min {th['min_profit_factor']}")

    n_trades = float(metrics.get("n_trades_mean", metrics.get("n_trades", 0)))
    if n_trades < th["min_n_trades"]:
        reasons.append(f"n_trades {n_trades:.0f} < min {th['min_n_trades']}")

    excess = float(metrics.get("excess_return_pct_mean", metrics.get("excess_return_pct", 0.0)))
    if excess < th["min_excess_return_pct"]:
        reasons.append(
            f"Excess vs Nifty {excess*100:.2f} percent < min {th['min_excess_return_pct']*100:.0f} percent",
        )

    # PR 174 + Phase 1.0 — Bailey/López de Prado overfit defenses.
    # When require_overfit_defenses=True (default for real-money), the
    # gate BLOCKS promotion if either metric is missing. Models that
    # genuinely can't emit DSR/PBO (zero-shot foundation models) must
    # explicitly override require_overfit_defenses=False.
    require_defenses = bool(th.get("require_overfit_defenses", False))

    if th.get("min_deflated_sharpe") is not None:
        if "deflated_sharpe" in metrics:
            dsr = float(metrics["deflated_sharpe"])
            if dsr < th["min_deflated_sharpe"]:
                reasons.append(
                    f"Deflated Sharpe {dsr:.3f} < min {th['min_deflated_sharpe']:.2f} "
                    f"(curve-fit risk — significance p={1-dsr:.3f})",
                )
        elif require_defenses:
            reasons.append(
                "Deflated Sharpe missing — promote-gate requires DSR for real-money. "
                "Run trainer with Optuna search + CPCV, or override "
                "promote_thresholds={'require_overfit_defenses': False} if zero-shot."
            )

    if th.get("max_pbo") is not None:
        if "probability_backtest_overfitting" in metrics:
            pbo = float(metrics["probability_backtest_overfitting"])
            if pbo > th["max_pbo"]:
                reasons.append(
                    f"PBO {pbo:.3f} > max {th['max_pbo']:.2f} "
                    f"(in-sample-best ranks below median OOS — overfit risk)",
                )
        elif require_defenses:
            reasons.append(
                "PBO missing — promote-gate requires probability-of-backtest-overfitting "
                "for real-money. Run ml.training.cpcv to compute, or override."
            )

    # Phase 1.0 — Regime-stratified Sharpe floor. A model that works in
    # bull and fails in bear is a real-money liability. Block if any
    # regime with > 50 days has Sharpe below the floor.
    min_regime_sharpe = th.get("min_regime_stratified_sharpe")
    if min_regime_sharpe is not None and "regime_sharpe" in metrics:
        rs = metrics.get("regime_sharpe", {})
        rn = metrics.get("regime_n_days", {})
        for regime in ("bull", "sideways", "bear"):
            ns = float(rn.get(regime, 0) or 0)
            if ns < 50:
                continue   # too few days to judge
            ss = float(rs.get(regime, 0) or 0)
            if ss < min_regime_sharpe:
                reasons.append(
                    f"Regime-stratified Sharpe in {regime}={ss:.2f} "
                    f"(n={ns:.0f} days) < min {min_regime_sharpe:.2f} "
                    f"(model unreliable in this market state)",
                )

    return len(reasons) == 0, reasons


# ============================================================================
# Internals
# ============================================================================


def _extract_trades(positions: np.ndarray, returns: np.ndarray, cost_bps: float) -> list[dict]:
    """Walk the position array bar-by-bar; emit a trade per close."""
    trades: list[dict] = []
    cur_pos = 0.0
    entry_idx: Optional[int] = None
    pnl = 0.0
    cost_unit = cost_bps / 10_000.0
    for i, pos in enumerate(positions):
        if cur_pos == 0.0 and pos != 0.0:
            # Open a new trade
            entry_idx = i
            cur_pos = float(pos)
            pnl = -abs(cur_pos) * cost_unit  # entry cost
        elif cur_pos != 0.0 and pos != cur_pos:
            # Close (or flip) — accumulate today's return then book the trade
            pnl += cur_pos * float(returns[i]) - abs(cur_pos - float(pos)) * cost_unit
            trades.append({
                "entry_idx": int(entry_idx) if entry_idx is not None else i,
                "exit_idx": int(i),
                "holding_days": int(i - (entry_idx or i)),
                "pnl": float(pnl),
            })
            # Open a new trade if pos != 0 (flip case)
            cur_pos = float(pos)
            entry_idx = i if pos != 0.0 else None
            pnl = -abs(cur_pos) * cost_unit if pos != 0.0 else 0.0
        else:
            # Holding the same position; accrue return
            pnl += cur_pos * float(returns[i])
    if cur_pos != 0.0 and entry_idx is not None:
        # Close any open position at the last bar
        trades.append({
            "entry_idx": int(entry_idx),
            "exit_idx": int(len(positions) - 1),
            "holding_days": int(len(positions) - 1 - entry_idx),
            "pnl": float(pnl),
        })
    return trades


def _empty_metrics() -> dict:
    """Sentinel for empty input — keeps downstream JSON shape stable."""
    return {
        "sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar": 0.0,
        "cagr": 0.0,
        "annualized_return": 0.0,
        "annualized_vol": 0.0,
        "total_return_pct": 0.0,
        "n_periods": 0,
        "n_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_holding_days": 0.0,
        "avg_daily_turnover": 0.0,
        "primary_metric": "sharpe",
        "primary_value": 0.0,
    }
