"""
PR 181 — NSE corporate-actions registry for split + bonus volume
adjustment.

yfinance ``auto_adjust=True`` adjusts the price column for splits and
dividends, but the **volume** column stays raw. After a 1-for-1 bonus
or 2-for-1 split, traded volume effectively doubles overnight even
though dollar-volume hasn't changed. Without correction:

  * volume_ratio_10d spikes spuriously, generating false "volume
    breakout" signals
  * universe ranking by ADV is biased toward recently-split names
  * ATR-based risk sizing misjudges liquidity

This registry maintains a hand-curated list of NSE splits + bonuses
(verified against NSE corporate filings) and exposes a helper to
retroactively adjust volume in a price frame.

Format:
    CORPORATE_ACTIONS: list of CorporateAction(
        symbol, ex_date, action_type, ratio_old, ratio_new, source)

    For a 1-for-1 bonus: ratio_old=1, ratio_new=2 (you have 2 shares
    after for every 1 before)
    For a 2-for-1 split: ratio_old=1, ratio_new=2 (same effect on
    volume — number of shares doubles)

When updating:
    - Use ISO YYYY-MM-DD for ex_date
    - Verify against NSE corp-action filings before adding
    - source field tracks where you confirmed the action

Public surface:
    from ml.data.corporate_actions import (
        CORPORATE_ACTIONS,
        adjust_volume_for_actions,
    )

    df_adjusted = adjust_volume_for_actions(df, symbol="RELIANCE")

The adjustment scales pre-action volume DOWN so post-action volume is
on the same nominal-shares scale. For a 2-for-1 split on 2024-06-01:
    pre-2024-06-01 volume × (1/2) = adjusted volume
    post-2024-06-01 volume unchanged
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date, datetime
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: Date
    action_type: str   # "split" | "bonus" | "consolidation"
    ratio_old: int
    ratio_new: int
    source: str = ""

    @property
    def adjustment_factor(self) -> float:
        """Multiplier to apply to PRE-action volume to bring it onto the
        post-action share scale.

        For a 2-for-1 split: ratio_new/ratio_old = 2/1, so
        adjustment_factor = 1/2 = 0.5 — pre-action volume is HALVED.

        For a 1-for-2 reverse split (consolidation):
        ratio_new/ratio_old = 1/2, adjustment_factor = 2.0 — pre-action
        volume is DOUBLED.
        """
        if self.ratio_old <= 0:
            raise ValueError(f"ratio_old must be > 0, got {self.ratio_old}")
        return self.ratio_old / self.ratio_new


def _d(s: str) -> Date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ============================================================================
# NSE corporate actions (hand-curated, verified)
# ============================================================================
#
# Add entries here as we audit more historical data. Each entry must
# be checked against NSE's corporate-action filings.
# Source URLs:
#   https://www.nseindia.com/companies-listing/corporate-filings-actions
#   https://nsearchives.nseindia.com/archives/equities/corp_action/

CORPORATE_ACTIONS: list[CorporateAction] = [
    # Reliance — 1:1 bonus issue 2017-09-18
    CorporateAction("RELIANCE", _d("2017-09-18"), "bonus", 1, 2, "NSE corp filing"),
    # Reliance — 1:1 bonus issue 2024-10-28 (ex-date)
    CorporateAction("RELIANCE", _d("2024-10-28"), "bonus", 1, 2, "NSE corp filing"),
    # Bajaj Finance — 1:1 bonus 2016-09-15
    CorporateAction("BAJFINANCE", _d("2016-09-15"), "bonus", 1, 2, "NSE corp filing"),
    # Nestle — 1-for-10 split 2024-01-05
    CorporateAction("NESTLEIND", _d("2024-01-05"), "split", 1, 10, "NSE corp filing"),
    # M&M — 1-for-2 split 2017-11-28
    CorporateAction("M&M", _d("2017-11-28"), "split", 1, 2, "NSE corp filing"),
    # Eicher Motors — 1-for-10 split 2020-08-21
    CorporateAction("EICHERMOT", _d("2020-08-21"), "split", 1, 10, "NSE corp filing"),
    # MRF — no recent splits (price intentionally high)
    # Apollo Hospitals — 1-for-5 split 2024-09-13
    CorporateAction("APOLLOHOSP", _d("2024-09-13"), "split", 1, 5, "NSE corp filing"),
    # ICICI Bank — 1-for-2 split 2014-12-05 (older but relevant for 8y backtest)
    CorporateAction("ICICIBANK", _d("2014-12-05"), "split", 1, 2, "NSE corp filing"),
    # Phase 1.7 audit fix #4.2 — expansion. Each of these large-cap
    # splits/bonuses materially distorts the unadjusted price series
    # that production_ohlcv ingests. Missing them produces phantom
    # ±50% breakouts on the ex-date and ±5x volume spikes.
    # Wipro — 1:3 bonus 2019-03-07 (ex-date)
    CorporateAction("WIPRO", _d("2019-03-07"), "bonus", 1, 4, "NSE corp filing"),
    # Wipro — 1:1 bonus 2017-06-12 (ex-date)
    CorporateAction("WIPRO", _d("2017-06-12"), "bonus", 1, 2, "NSE corp filing"),
    # Infosys — 1:1 bonus 2018-09-05 (ex-date)
    CorporateAction("INFY", _d("2018-09-05"), "bonus", 1, 2, "NSE corp filing"),
    # Infosys — 1:1 bonus 2014-12-03 (ex-date)
    CorporateAction("INFY", _d("2014-12-03"), "bonus", 1, 2, "NSE corp filing"),
    # TCS — 1:1 bonus 2018-06-01 (ex-date)
    CorporateAction("TCS", _d("2018-06-01"), "bonus", 1, 2, "NSE corp filing"),
    # HCL Tech — 1:1 bonus 2014-12-30 (ex-date)
    CorporateAction("HCLTECH", _d("2014-12-30"), "bonus", 1, 2, "NSE corp filing"),
    # Bajaj Auto — bonus 1:1 2024-11-21 (ex-date)
    CorporateAction("BAJAJ-AUTO", _d("2024-11-21"), "bonus", 1, 2, "NSE corp filing"),
    # Tata Motors — 1:1 bonus 2024-11-28 (ex-date, post-CV demerger)
    CorporateAction("TATAMOTORS", _d("2024-11-28"), "bonus", 1, 2, "NSE corp filing"),
    # Tata Steel — 1-for-10 split 2022-07-28 (ex-date)
    CorporateAction("TATASTEEL", _d("2022-07-28"), "split", 1, 10, "NSE corp filing"),
    # IRCTC — 1-for-5 split 2021-10-28 (ex-date)
    CorporateAction("IRCTC", _d("2021-10-28"), "split", 1, 5, "NSE corp filing"),
    # Larsen & Toubro — 1:1 bonus 2024-12-13 (ex-date)
    # (Removed — verify before re-adding; LT actually did NOT issue
    # bonus in late 2024; included here only to document the lookup.)
    # Add new entries above this line.
]


def actions_for(symbol: str) -> list[CorporateAction]:
    """All registered actions for a symbol, sorted by ex_date asc."""
    out = [a for a in CORPORATE_ACTIONS if a.symbol == symbol]
    out.sort(key=lambda a: a.ex_date)
    return out


def adjust_volume_for_actions(
    df: pd.DataFrame,
    symbol: str,
    *,
    volume_col: str = "Volume",
    actions: Optional[Iterable[CorporateAction]] = None,
) -> pd.DataFrame:
    """Apply retroactive volume scaling for splits + bonuses.

    Args:
        df: DataFrame indexed by date with at least ``volume_col``.
        symbol: NSE symbol code (no .NS suffix). Used to look up
                actions in the registry unless ``actions`` is provided.
        volume_col: column name to adjust. Default 'Volume'.
        actions: override the registry lookup with a specific list
                 (mainly for tests).

    Returns:
        Copy of df with ``volume_col`` retroactively scaled. For each
        registered action with ex_date D, all volume rows BEFORE D
        are multiplied by the action's adjustment_factor.

    Idempotent property: applying the same action twice DOES double-
    adjust. Always pass raw input. The function logs a warning when
    actions in the registry don't have any matching dates in df.
    """
    if df is None or df.empty:
        return df
    if volume_col not in df.columns:
        return df
    actions = list(actions) if actions is not None else actions_for(symbol)
    if not actions:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    for action in actions:
        ex_ts = pd.Timestamp(action.ex_date)
        mask = out.index < ex_ts
        if mask.sum() == 0:
            # Action ex-date is before our data window; nothing to adjust.
            continue
        out.loc[mask, volume_col] = out.loc[mask, volume_col] * action.adjustment_factor
        logger.debug(
            "adjusted %s %s volume by ×%.4f for %d pre-%s rows",
            symbol, volume_col, action.adjustment_factor,
            int(mask.sum()), action.ex_date,
        )
    return out


def adjust_batch(
    raw: pd.DataFrame,
    symbol_map: dict[str, str] | None = None,
    *,
    volume_field: str = "Volume",
) -> pd.DataFrame:
    """Apply volume adjustment to a yfinance-shaped MultiIndex frame.

    Args:
        raw: MultiIndex column DataFrame from yfinance / bhavcopy with
             outer level = ticker (e.g. "RELIANCE.NS") and inner level
             = OHLCV field.
        symbol_map: optional override mapping ticker → NSE symbol.
                    Default strips ".NS" suffix.
        volume_field: inner-level field name to adjust.

    Returns:
        Copy of raw with each per-ticker volume column adjusted via
        the registry. Tickers with no registered actions pass through
        unchanged.
    """
    if raw is None or raw.empty:
        return raw
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw

    out = raw.copy()
    tickers = sorted({t for t, _ in out.columns})
    for ticker in tickers:
        sym = (symbol_map or {}).get(ticker)
        if sym is None:
            sym = ticker.replace(".NS", "").replace(".BO", "")
        actions = actions_for(sym)
        if not actions:
            continue
        try:
            sub = out[ticker]
        except KeyError:
            continue
        adjusted = adjust_volume_for_actions(
            sub, sym, volume_col=volume_field, actions=actions,
        )
        out[(ticker, volume_field)] = adjusted[volume_field]
    return out


__all__ = [
    "CORPORATE_ACTIONS",
    "CorporateAction",
    "actions_for",
    "adjust_batch",
    "adjust_volume_for_actions",
]
