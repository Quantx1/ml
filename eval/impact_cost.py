"""
PR 196 — market-impact cost model (Almgren-Chriss square-root law).

The default cost model in compute_backtest_metrics is a flat 13bps
round-trip. That's correct for HDFCBANK in normal volume but wildly
wrong for ZOMATO at meaningful size or for anything in a small-cap
universe where a single trade can be 5-20% of daily ADV.

Almgren-Chriss (2000) decomposed execution cost into:
  permanent impact ∝ trade_size / ADV     (price walks against you)
  temporary impact ∝ √(trade_size / ADV)   (square-root law from
                                             order-book depth scaling)

Combining the two with a per-stock σ (vol) scaling gives the standard
quant-fund execution cost formula:

  cost_bps = base_bps + impact_coef * vol_pct * sqrt(trade_size / ADV)

Where:
  base_bps = exchange fees + STT + brokerage (~5-8 bps round-trip on NSE)
  impact_coef ≈ 0.5-1.0 (calibrated; institutional traders use ~0.7)
  vol_pct = realized 20-day daily vol
  trade_size / ADV = participation rate

This module exposes:

    impact_cost_bps(trade_size, adv, vol_pct, base_bps=8.0, impact_coef=0.7)
    apply_impact_to_returns(positions, raw_returns, adv, vol_pct, ...)

Used by compute_backtest_metrics' cost layer and by AutoPilot's
position-sizing logic at runtime to estimate slippage before placing
orders.

References:
  Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
  Square-root impact law: empirically validated on every major equity
  market including NSE (Tóth et al. 2011).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Conservative defaults calibrated from public NSE execution data:
# - SBI Cap, Kotak Inst, ICICI Sec broker reports (institutional)
# - Retail NSE participation rate < 0.1% of ADV → impact ≈ 1-3 bps
# - Institutional participation 5% of ADV → impact ≈ 25-40 bps
DEFAULT_BASE_BPS = 8.0     # exchange fees + STT + brokerage round-trip
DEFAULT_IMPACT_COEF = 0.7  # √-law slope at unit volatility


@dataclass
class ImpactCostConfig:
    """Almgren-Chriss square-root impact cost config.

    base_bps:
        Fixed round-trip cost (fees + STT + brokerage). NSE retail:
        ~8-13 bps; institutional: ~5 bps.

    impact_coef:
        Coefficient on the temporary-impact √-law. 0.7 is a
        widely-quoted institutional estimate (Tóth et al. 2011).

    vol_units:
        Whether vol_pct passed into impact_cost_bps is a fraction
        ("decimal", 0.02 = 2 percent) or basis points ("bps", 200 =
        2 percent). Default decimal.

    max_bps:
        Safety cap on per-trade cost. Avoids runaway estimates when a
        trade size approaches 100% of ADV (where the model breaks down
        anyway — no real trade goes there).
    """

    base_bps: float = DEFAULT_BASE_BPS
    impact_coef: float = DEFAULT_IMPACT_COEF
    vol_units: str = "decimal"
    max_bps: float = 200.0


def impact_cost_bps(
    trade_size: float,
    adv: float,
    vol_pct: float,
    cfg: Optional[ImpactCostConfig] = None,
) -> float:
    """Per-trade round-trip impact cost in basis points.

    Args:
        trade_size: rupee value of the order (₹).
        adv: average daily traded value of the symbol (₹).
        vol_pct: per-day realized volatility (decimal by default).
        cfg: ImpactCostConfig — overrides defaults.

    Returns:
        Cost in basis points (1 bp = 0.01 percent). Capped at cfg.max_bps.

    Edge cases: if adv ≤ 0 or trade_size ≤ 0 → returns base_bps only
    (no impact). vol_pct ≤ 0 → impact term is zero.
    """
    cfg = cfg or ImpactCostConfig()
    if trade_size <= 0 or adv <= 0:
        return cfg.base_bps
    participation = trade_size / adv
    vol = vol_pct if cfg.vol_units == "decimal" else vol_pct / 10_000.0
    if vol <= 0:
        return cfg.base_bps
    # Square-root impact in PRICE space → convert to bps via vol_pct
    # × √participation, then × impact_coef.
    impact_pct = cfg.impact_coef * vol * np.sqrt(participation)
    impact_bps = impact_pct * 10_000.0
    total_bps = cfg.base_bps + impact_bps
    return float(min(cfg.max_bps, total_bps))


def apply_impact_to_returns(
    positions: np.ndarray | Iterable[float],
    raw_returns: np.ndarray | Iterable[float],
    adv: float,
    vol_pct: float,
    capital: float,
    cfg: Optional[ImpactCostConfig] = None,
) -> np.ndarray:
    """Apply per-bar impact-cost adjustment to a strategy return series.

    For each bar where the position changes, the trade size is the
    capital × |position_change|. Cost is deducted from that bar's
    return.

    Args:
        positions: position weights at each bar in [-1, 1].
        raw_returns: per-bar simple returns (e.g. forward 1-day).
        adv: average daily traded value of the symbol.
        vol_pct: realized vol (decimal).
        capital: total capital allocated to the strategy.
        cfg: impact-cost config.

    Returns:
        Adjusted return array, same length as inputs.
    """
    cfg = cfg or ImpactCostConfig()
    pos = np.asarray(list(positions), dtype=float)
    rets = np.asarray(list(raw_returns), dtype=float)
    if pos.size == 0 or pos.size != rets.size:
        return rets.copy()
    pos_changes = np.abs(np.diff(pos, prepend=0.0))
    out = rets.copy()
    for i, change in enumerate(pos_changes):
        if change <= 0:
            continue
        trade_size = float(capital) * float(change)
        cost_bps_i = impact_cost_bps(trade_size, adv, vol_pct, cfg)
        out[i] = out[i] - cost_bps_i / 10_000.0
    return out


__all__ = [
    "DEFAULT_BASE_BPS",
    "DEFAULT_IMPACT_COEF",
    "ImpactCostConfig",
    "apply_impact_to_returns",
    "impact_cost_bps",
]
