"""Regime-detection feature builder (NIFTY-level, daily).

Design principles (shared across ``ml/regime``):
  * Regimes are LATENT — 100% real-time accuracy is impossible. The module
    optimizes for (1) high agreement with hindsight-smoothed labels,
    (2) minimal turning-point lag, (3) no whipsaw, and (4) UTILITY: the
    gated momentum/swing books must beat their ungated versions (utility is
    evaluated elsewhere, in the walk-forward backtests).
  * STRICT NO-LOOKAHEAD: every feature here is trailing — rolling windows,
    ``cummax`` drawdowns and lagged diffs only. ``build_regime_features`` on a
    truncated history reproduces exactly the same rows as on the full history
    (tested), so the online path ``fit(history)`` -> ``filter_next(x_t)``
    never sees information from after ``t``.

Conventions mirror ``ml.features.momentum_features``: ``pct_change`` always
uses ``fill_method=None`` (a gap yields NaN, never a fabricated 0%), and
per-symbol panel ops use ``groupby(...).transform`` (index-aligned, no
cross-symbol leakage).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: Output feature columns, in order. NOTE: the model-input matrix built by
#: ``ml.regime.ensemble`` puts ``ret_21d`` FIRST (models relabel states by the
#: mean of feature 0), which is independent of this display order.
REGIME_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "realized_vol_21",
    "realized_vol_63",
    "vol_of_vol_21",
    "drawdown",
    "drawdown_21d_change",
    "trend_200",
    "trend_50",
    "sma50_slope_21",
    "breadth_200",
    "breadth_change_21",
]

#: Longest trailing warmup any feature consumes (trend_200's 200-bar SMA).
REGIME_WARMUP_BARS = 200


def _breadth_200(universe_panel: pd.DataFrame) -> pd.Series:
    """% of universe symbols trading above their own 200-day SMA, per date.

    Trailing only: each symbol's SMA200 uses that symbol's own past closes.
    Symbols still inside their 200-bar warmup contribute nothing to that
    date's breadth (NaN indicator rows are skipped by the mean), so early
    dates reflect only symbols with enough history — fail-soft, no lookahead.
    """
    up = universe_panel[["date", "symbol", "close"]].dropna()
    up = up.sort_values(["symbol", "date"])
    sma200 = up.groupby("symbol")["close"].transform(lambda s: s.rolling(200).mean())
    above = pd.Series(np.where(sma200.isna(), np.nan, (up["close"] > sma200).astype(float)),
                      index=up.index)
    return above.groupby(up["date"]).mean()  # NaN indicators skipped


def build_regime_features(
    nifty: pd.DataFrame, universe_panel: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Build the daily regime feature frame from NIFTY closes (+ optional breadth).

    Args:
        nifty: DataFrame ``['date', 'close']`` — daily NIFTY closes, e.g. from
            ``ml.data.benchmark.load_nifty_benchmark``.
        universe_panel: optional long panel ``['date', 'symbol', 'close']``
            (e.g. from ``ml.data.data_loader.load_ohlcv``) used for breadth.
            When absent, the two breadth columns are emitted as NaN
            (fail-soft — downstream models simply drop the columns).

    Returns:
        DataFrame ``['date', *REGIME_FEATURES]`` sorted by date. The first
        ~``REGIME_WARMUP_BARS`` rows contain NaN (long-window warmup); callers
        drop NaN on the model-feature subset before fitting.

    All features are strictly trailing (no lookahead) — see module docstring.
    """
    df = (
        nifty[["date", "close"]]
        .dropna()
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    c = df["close"].astype(float)
    r1 = c.pct_change(fill_method=None)

    df["ret_1d"] = r1
    df["ret_5d"] = c.pct_change(5, fill_method=None)
    df["ret_21d"] = c.pct_change(21, fill_method=None)

    df["realized_vol_21"] = r1.rolling(21).std()
    df["realized_vol_63"] = r1.rolling(63).std()
    df["vol_of_vol_21"] = df["realized_vol_21"].rolling(21).std()

    dd = c / c.cummax() - 1.0  # running-peak drawdown; cummax is causal
    df["drawdown"] = dd
    df["drawdown_21d_change"] = dd.diff(21)

    sma200 = c.rolling(200).mean()
    sma50 = c.rolling(50).mean()
    df["trend_200"] = c / sma200 - 1.0
    df["trend_50"] = c / sma50 - 1.0
    df["sma50_slope_21"] = sma50.pct_change(21, fill_method=None)

    if universe_panel is not None and not universe_panel.empty:
        breadth = _breadth_200(universe_panel)
        df["breadth_200"] = df["date"].map(breadth)
        df["breadth_change_21"] = df["breadth_200"].diff(21)
    else:
        df["breadth_200"] = np.nan
        df["breadth_change_21"] = np.nan

    return df[["date", *REGIME_FEATURES]]


__all__ = ["REGIME_FEATURES", "REGIME_WARMUP_BARS", "build_regime_features"]
