"""
PR 197 — Kelly fraction sizing for per-model position limits.

A model that passes the promote gate is "tradeable" but not all
tradeable models are tradeable at the same size. Kelly's criterion
gives the optimal bet fraction:

    full_kelly = (p * b - q) / b
        p = win rate
        q = 1 - p (loss rate)
        b = win/loss ratio (avg_win / avg_loss)

Full Kelly is mathematically optimal for log-utility maximization but
empirically too aggressive — Wall Street practice is "fractional Kelly"
at 10-25 percent of full because:

  1. We're ESTIMATING p and b from finite data → estimation error
     amplifies disastrously at full Kelly
  2. Drawdowns under full Kelly can exceed -50 percent which is
     psychologically and practically unworkable
  3. Half / quarter Kelly captures most of the geometric growth with
     a fraction of the drawdown variance

This module computes Kelly fractions from a strategy's backtest stats
(win rate, profit factor, avg win/loss) and writes them to the
model_versions metadata. AutoPilot reads at runtime to scale its
per-model position limits.

Public surface:

    from ml.eval.kelly import kelly_fraction, kelly_from_metrics

    f = kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, fractional=0.25)
    # → 0.0625 (6.25 percent of capital per signal)

    f = kelly_from_metrics({"win_rate": 0.55, "profit_factor": 1.8, ...})

References:
  Kelly (1956), "A New Interpretation of Information Rate"
  Optimal Kelly under risk constraints (scirp.org 2025)
  Thorp (2006), "The Kelly Criterion in Blackjack, Sports Betting,
  and the Stock Market"
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Wall Street convention: 25 percent of full Kelly is the most common
# institutional choice. Half Kelly (50 percent) is aggressive; quarter
# Kelly (25 percent) is "we trust the estimates moderately."
DEFAULT_FRACTIONAL = 0.25

# Hard ceiling regardless of what Kelly says. Even a model with apparent
# 80 percent win rate shouldn't size more than 20 percent of capital
# per signal — concentration risk.
MAX_KELLY_FRACTION = 0.20


def kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    *,
    fractional: float = DEFAULT_FRACTIONAL,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Compute the position fraction for one signal/trade.

    Args:
        win_rate: probability of a winning trade in [0, 1].
        win_loss_ratio: avg_win / avg_loss (b in Kelly's formula).
                        Must be > 0.
        fractional: scaling factor on full Kelly (0 < fractional <= 1).
                    0.25 = quarter Kelly (institutional default).
        max_fraction: hard ceiling on the returned size.

    Returns:
        Fraction of capital to allocate, in [0, max_fraction]. Returns
        0 when full Kelly would be negative (bad-edge model — don't
        trade) or when inputs are degenerate.
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0
    if not (0.0 < fractional <= 1.0):
        raise ValueError(f"fractional must be in (0, 1], got {fractional}")

    p = win_rate
    q = 1.0 - p
    b = win_loss_ratio
    full = (p * b - q) / b
    if full <= 0:
        return 0.0
    return float(min(max_fraction, fractional * full))


def kelly_from_metrics(
    metrics: dict,
    *,
    fractional: float = DEFAULT_FRACTIONAL,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Compute Kelly fraction from a TrainResult.metrics-shaped dict.

    Looks for either:
      - explicit (win_rate, win_loss_ratio) keys
      - (win_rate, profit_factor) — derives win/loss ratio from profit
         factor: PF = (win_rate * avg_win) / ((1-win_rate) * avg_loss),
         so avg_win / avg_loss = PF * (1-win_rate) / win_rate.

    Returns 0 when neither shape is available.
    """
    win_rate = float(metrics.get("win_rate_mean", metrics.get("win_rate", 0.0)))
    if win_rate <= 0:
        return 0.0

    if "win_loss_ratio" in metrics:
        wlr = float(metrics["win_loss_ratio"])
    elif "profit_factor" in metrics or "profit_factor_mean" in metrics:
        pf = float(metrics.get("profit_factor_mean", metrics.get("profit_factor", 0.0)))
        if pf <= 0 or win_rate >= 1:
            return 0.0
        wlr = pf * (1 - win_rate) / win_rate
    else:
        return 0.0

    return kelly_fraction(
        win_rate=win_rate, win_loss_ratio=wlr,
        fractional=fractional, max_fraction=max_fraction,
    )


__all__ = [
    "DEFAULT_FRACTIONAL",
    "MAX_KELLY_FRACTION",
    "kelly_fraction",
    "kelly_from_metrics",
]
