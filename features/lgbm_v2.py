"""
lgbm_v2 — canonical feature pipeline for the v2 LGBM signal gate.

This module is the **single source of truth** for the LGBM Verdict
engine's feature set. Both the trainer (`ml/training/trainers/
lgbm_signal_gate.py`) and the live inference path (`backend/
services/feature_engineering.py`) import from here so the two stay
in lock-step.

Feature schema (30 features, in order):

  4 returns:        ret_1d, ret_5d, ret_10d, ret_20d
  6 OHLCV tech:     rsi_14, macd_diff, ema_20_dist, ema_50_dist,
                    atr_14_pct, volume_ratio_10d
  1 Bollinger:      bb_percent
  2 52-week:        high_52w_dist, low_52w_dist
  2 Stochastic:     stoch_k, stoch_d
  4 FII/DII flow:   fii_5d_sum, dii_5d_sum, fii_5d_z, dii_5d_z
  2 Sentiment:      sentiment_5d_mean, sentiment_5d_count
  8 Fundamentals:   eps_yoy_growth, revenue_yoy_growth, margin_trend_4q,
                    promoter_delta_4q, fii_delta_4q, debt_to_equity,
                    book_value_yoy, fundamentals_age_days
  1 FFD:            log_close_ffd_04


Why this module exists (PR-E audit, 2026-05-19): the trainer wrote
30 features into its sidecar `.meta.json` but the live path only emitted
15 legacy keys. After training, `LGBMGate.predict()` would KeyError on
every call. This module ensures parity.
"""

from __future__ import annotations

import os as _os
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical FEATURE_ORDER — both trainer and live inference read THIS list.
# ---------------------------------------------------------------------------
FEATURE_ORDER_BASE: list[str] = [
    # Returns
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # OHLCV technicals
    "rsi_14", "macd_diff",
    "ema_20_dist", "ema_50_dist",
    "atr_14_pct",
    "volume_ratio_10d",
    # Bollinger / 52w / stoch
    "bb_percent",
    "high_52w_dist", "low_52w_dist",
    "stoch_k", "stoch_d",
    # FII/DII flow
    "fii_5d_sum", "dii_5d_sum", "fii_5d_z", "dii_5d_z",
    # Sentiment
    "sentiment_5d_mean", "sentiment_5d_count",
    # Fundamentals
    "eps_yoy_growth", "revenue_yoy_growth", "margin_trend_4q",
    "promoter_delta_4q", "fii_delta_4q", "debt_to_equity",
    "book_value_yoy", "fundamentals_age_days",
    # Fractionally-differentiated log close
    "log_close_ffd_04",
]


FEATURE_ORDER: list[str] = list(FEATURE_ORDER_BASE)


# ---------------------------------------------------------------------------
# Pure OHLCV-derived features (no external data). Used by both training
# loop (per-symbol DataFrame) and inference (per-bar dict).
# ---------------------------------------------------------------------------
def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 15 OHLCV-derived features for one symbol's daily frame.

    Expects columns: Close, High, Low, Volume (Yfinance-shape, capitalized).

    Returns: DataFrame indexed by the same dates as df, with the
    OHLCV-block columns from FEATURE_ORDER plus _atr_raw + _fwd_return
    private helpers used by the trainer's labeling step.

    NaN rows from rolling-window warmup are kept; caller drops them
    after labeling.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # ── Returns ──
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # ── RSI(14) Wilder's ──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

    # ── MACD diff (12-26 EMA - 9 EMA signal) ──
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_diff"] = macd - macd_signal

    # ── EMA distance (% from price) ──
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    out["ema_20_dist"] = (close - ema_20) / ema_20
    out["ema_50_dist"] = (close - ema_50) / ema_50

    # ── ATR(14) as % of price (vol-normalized) ──
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_14_pct"] = (atr / close).fillna(0)
    out["_atr_raw"] = atr  # Private — used by trainer labeling

    # ── Volume ratio ──
    out["volume_ratio_10d"] = (
        volume / volume.rolling(10).mean().replace(0, np.nan)
    )

    # ── Bollinger %B ──
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    out["bb_percent"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ── 52-week high/low distance ──
    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    out["high_52w_dist"] = (close - high_52w) / high_52w.replace(0, np.nan)
    out["low_52w_dist"] = (close - low_52w) / low_52w.replace(0, np.nan)

    # ── Stochastic ──
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3).mean()

    # ── Fractionally-differentiated log close (AFML Ch.5, d=0.4) ──
    from ml.features.frac_diff import frac_diff_ffd  # local import — heavy dep
    log_close = np.log(close.replace(0, np.nan))
    out["log_close_ffd_04"] = frac_diff_ffd(log_close, d=0.4, thresh=1e-3)

    return out


# ---------------------------------------------------------------------------
# Live inference helper — produces one feature row (all 30 keys) for one
# symbol at one date. Inference path consumes this; LGBMGate.predict()
# receives the dict.
# ---------------------------------------------------------------------------
def compute_inference_features(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    flow_features_df: Optional[pd.DataFrame] = None,
    sentiment_features_df: Optional[pd.DataFrame] = None,
    fundamentals_features: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Build the 30-feature row used at live inference time.

    Args:
        symbol: NSE symbol (no .NS suffix).
        ohlcv_df: daily OHLCV history for the symbol (Yfinance shape:
            Close, High, Low, Volume — capitalized). Must end at or
            after as_of_date.
        as_of_date: the bar-close date to score.
        flow_features_df: pre-computed FII/DII flow features indexed by
            date with columns fii_5d_sum, dii_5d_sum, fii_5d_z, dii_5d_z.
            Optional — if None, the four flow features are zero-filled.
        sentiment_features_df: per-symbol sentiment features indexed by
            (symbol, date) with columns sentiment_5d_mean,
            sentiment_5d_count. Optional — zero-filled if None.
        fundamentals_features: dict with the 8 fundamentals keys.
            Optional — zero-filled if None or symbol not covered.

    Returns:
        A dict with EXACTLY the keys in FEATURE_ORDER, each value a
        Python float (no NaN). Callers feed this into LGBMGate.predict().
    """
    # ── 1. Compute OHLCV block ──
    ohlcv = compute_ohlcv_features(ohlcv_df)

    # Pick the row corresponding to as_of_date (or last row <= as_of_date).
    if as_of_date in ohlcv.index:
        ohlcv_row = ohlcv.loc[as_of_date]
    else:
        # Use the most recent row at or before as_of_date.
        valid = ohlcv.loc[ohlcv.index <= as_of_date]
        if valid.empty:
            raise ValueError(
                f"compute_inference_features: no OHLCV rows on or before {as_of_date}"
            )
        ohlcv_row = valid.iloc[-1]

    out: Dict[str, float] = {}
    # OHLCV-block keys from FEATURE_ORDER_BASE (excluding flow / sentiment
    # / fundamentals which come from other sources).
    _OHLCV_KEYS = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "rsi_14", "macd_diff",
        "ema_20_dist", "ema_50_dist",
        "atr_14_pct",
        "volume_ratio_10d",
        "bb_percent",
        "high_52w_dist", "low_52w_dist",
        "stoch_k", "stoch_d",
        "log_close_ffd_04",
    ]
    for key in _OHLCV_KEYS:
        v = ohlcv_row.get(key)
        out[key] = float(v) if v is not None and not pd.isna(v) else 0.0

    # ── 2. FII/DII flow features ──
    _FLOW_KEYS = ["fii_5d_sum", "dii_5d_sum", "fii_5d_z", "dii_5d_z"]
    if flow_features_df is not None and not flow_features_df.empty:
        flow_row = _pick_row_at_or_before(flow_features_df, as_of_date)
        for key in _FLOW_KEYS:
            v = flow_row.get(key) if flow_row is not None else None
            out[key] = float(v) if v is not None and not pd.isna(v) else 0.0
    else:
        for key in _FLOW_KEYS:
            out[key] = 0.0

    # ── 3. Sentiment features ──
    _SENTIMENT_KEYS = ["sentiment_5d_mean", "sentiment_5d_count"]
    if sentiment_features_df is not None and not sentiment_features_df.empty:
        sent_row = _pick_symbol_row_at_or_before(
            sentiment_features_df, symbol, as_of_date,
        )
        for key in _SENTIMENT_KEYS:
            v = sent_row.get(key) if sent_row is not None else None
            out[key] = float(v) if v is not None and not pd.isna(v) else 0.0
    else:
        for key in _SENTIMENT_KEYS:
            out[key] = 0.0

    # ── 4. Fundamentals features ──
    _FUNDS_KEYS = [
        "eps_yoy_growth", "revenue_yoy_growth", "margin_trend_4q",
        "promoter_delta_4q", "fii_delta_4q", "debt_to_equity",
        "book_value_yoy", "fundamentals_age_days",
    ]
    if fundamentals_features is not None:
        for key in _FUNDS_KEYS:
            v = fundamentals_features.get(key)
            out[key] = float(v) if v is not None and not pd.isna(v) else 0.0
    else:
        for key in _FUNDS_KEYS:
            out[key] = 0.0

    # ── Strict schema check ──
    missing = [k for k in FEATURE_ORDER_BASE if k not in out]
    if missing:
        raise RuntimeError(
            f"lgbm_v2: feature row missing keys {missing}. "
            "This is a parity bug — fix compute_inference_features.",
        )

    return out


def _pick_row_at_or_before(
    df: pd.DataFrame, as_of_date: pd.Timestamp,
) -> Optional[pd.Series]:
    """Get the most recent row in df at or before as_of_date.

    Handles both DatetimeIndex and date-only index. Returns None if no
    row qualifies.
    """
    try:
        valid = df.loc[df.index <= as_of_date]
    except TypeError:
        # Index may be date objects; compare via pd.to_datetime.
        valid = df.loc[
            pd.to_datetime(df.index) <= pd.to_datetime(as_of_date)
        ]
    if valid.empty:
        return None
    return valid.iloc[-1]


def _pick_symbol_row_at_or_before(
    df: pd.DataFrame, symbol: str, as_of_date: pd.Timestamp,
) -> Optional[pd.Series]:
    """Get the most recent row for `symbol` in df at or before as_of_date.

    Handles MultiIndex (symbol, date) and a 'symbol' column variant.
    """
    if isinstance(df.index, pd.MultiIndex):
        try:
            sym_df = df.xs(symbol, level=0)
        except KeyError:
            return None
        return _pick_row_at_or_before(sym_df, as_of_date)
    if "symbol" in df.columns:
        sym_df = df[df["symbol"] == symbol].drop(columns=["symbol"])
        if sym_df.empty:
            return None
        return _pick_row_at_or_before(sym_df, as_of_date)
    # Single-symbol frame (no symbol column).
    return _pick_row_at_or_before(df, as_of_date)
