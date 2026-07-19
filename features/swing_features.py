"""Swing engine feature builder (swing plan Task 1).

THE single builder used by BOTH the trainer and the serving SwingEngine
— importing this in both paths guarantees train/serve parity (audit: the
skew class of bugs). Price-first features with a 5-21 day emphasis:
mean-reversion / short-trend families rather than momentum's long lookbacks.

Implementation note: all per-symbol operations produce index-aligned output —
either via groupby(...).transform(...) or via vectorized grouped ops such as
groupby(...).cumsum() (used for OBV) that also preserve the original index.
groupby(...).apply(...) is intentionally avoided: on pandas 2.1.4 it can return
a DataFrame (not a Series) for a single-symbol panel — the serving-time scoring
path — which crashes column assignment. Do NOT "simplify" the OBV cumsum back
to an apply. The unavoidable per-group library calls (RSI and ADX, from `ta`)
are applied via an explicit per-group concat + reindex (``_per_symbol_series``),
which is robust on both single- and multi-symbol panels.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

_RET_WINDOWS = [1, 2, 3, 5, 10, 21, 42, 63]

_EPS = 1e-9

#: Longest warmup window any feature consumes — the 63-bar return lookback and
#: the 63-bar beta/corr-vs-index windows on daily returns. Callers sizing a raw
#: OHLCV pull must budget raw_days >= cv_days_needed + SWING_WARMUP_BARS +
#: label_horizon, since feature warmup AND the label's forward horizon both
#: shrink the panel.
SWING_WARMUP_BARS = 63

SWING_FEATURE_ORDER = [
    # --- A. multi-horizon raw returns ---
    *[f"ret_{w}d" for w in _RET_WINDOWS],
    # --- B. mean-reversion ---
    "rsi_2", "rsi_14",
    "zscore_10", "zscore_20",
    "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_ema_9",
    "boll_pos_20", "pullback_from_high_21", "bounce_from_low_21", "close_pos_21",
    # --- C. gaps / range ---
    "gap_open", "gap_abs_mean_5", "range_pct_1d", "close_pos_1d",
    # --- D. short trend ---
    "sma_5_20_align", "sma20_slope_10", "ema_9_21_spread", "adx_14", "macd_hist_norm",
    # --- E. volatility / risk ---
    "realized_vol_5", "realized_vol_10", "realized_vol_21",
    "vol_ratio_5_21", "parkinson_vol_10", "atr_pct_14",
    # --- F. volume confirmation ---
    "rel_volume_5", "rel_volume_21", "vol_zscore_10",
    "up_vol_ratio_10", "obv_slope_10", "volume_breakout",
    # --- G. bases for cross-sectional ranks ---
    "vol_adj_mom_21", "mom_consistency_10",
    # --- H. relative strength vs index (NIFTY); NaN when no benchmark ---
    "rs_index_5", "rs_index_10", "rs_index_21",
    "rs_index_slope_5", "beta_index_63", "corr_index_63",
    # --- I. cross-sectional ranks (computed last; within-date) ---
    "xs_rank_ret_5", "xs_rank_ret_10", "xs_rank_ret_21",
    "xs_rank_zscore_20", "xs_rank_vol_adj_mom_21", "xs_rank_rs_index_10",
]


def _rolling_mean_positive(s: pd.Series, window: int) -> pd.Series:
    """Fraction of values > 0 over a rolling window."""
    return s.rolling(window).apply(lambda x: np.mean(x > 0), raw=True)


def _per_symbol_series(
    df: pd.DataFrame, sym: pd.Series, fn: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    """Apply ``fn`` to each symbol's sub-frame and stitch the per-group Series
    back onto the original index.

    Robust replacement for ``groupby(sym).apply(fn)`` for the rare feature that
    needs a whole-sub-frame view (ADX needs high/low/close together; RSI is a
    per-group `ta` call). Each call of ``fn`` returns a Series indexed like its
    sub-frame; we concat and reindex to the original row order. Works
    identically for one symbol or many — no DataFrame-vs-Series ambiguity, no
    cross-symbol leakage.
    """
    parts = [fn(g) for _, g in df.groupby(sym, sort=False)]
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(parts).reindex(df.index)


def build_swing_features(
    panel: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute swing features for a long OHLCV panel.

    Args:
        panel: DataFrame with columns
               ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
               The frame may arrive in any row order.
        benchmark: optional DataFrame ['date', 'close'] (e.g. NIFTY/NSEI). When
               provided, relative-strength-vs-index features are added; when
               None they are emitted as NaN columns (fail-soft).

    Returns:
        DataFrame with columns ['date', 'symbol', *SWING_FEATURE_ORDER].
        The first ~SWING_WARMUP_BARS rows per symbol contain NaN (longest
        window undefined). After merging with labels, the trainer should drop
        NaN on the FEATURE columns specifically:

            df = features.merge(labels, on=["date", "symbol"], how="inner")
            df = df.dropna(subset=SWING_FEATURE_ORDER)

        (a bare ``dropna()`` would also drop valid label rows; dropping only
        on a single feature would miss other warmup-controlled NaNs).
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    sym = df["symbol"]
    eps = _EPS

    # ==================================================================
    # A. Multi-horizon raw returns (close[t]/close[t-w] - 1, per symbol)
    # ==================================================================
    for w in _RET_WINDOWS:
        df[f"ret_{w}d"] = df.groupby(sym)["close"].transform(
            lambda s, _w=w: s / s.shift(_w) - 1.0
        )

    # Daily returns — the base for several derived features. fill_method=None:
    # do NOT forward-fill across gaps (silent ffill is deprecated and wrong for
    # returns — a gap should yield NaN, not a fabricated 0%).
    df["__daily_ret"] = df.groupby(sym)["close"].transform(
        lambda s: s.pct_change(fill_method=None)
    )

    # ==================================================================
    # B. Mean-reversion
    # ==================================================================
    # RSI via `ta` — a per-group library call, applied through the index-safe
    # _per_symbol_series helper (NOT groupby.apply).
    from ta.momentum import RSIIndicator  # noqa: PLC0415

    def _rsi2(g: pd.DataFrame) -> pd.Series:
        return RSIIndicator(g["close"], window=2, fillna=False).rsi()

    def _rsi14(g: pd.DataFrame) -> pd.Series:
        return RSIIndicator(g["close"], window=14, fillna=False).rsi()

    df["rsi_2"] = _per_symbol_series(df, sym, _rsi2)
    df["rsi_14"] = _per_symbol_series(df, sym, _rsi14)

    _sma5 = df.groupby(sym)["close"].transform(lambda s: s.rolling(5).mean())
    _sma10 = df.groupby(sym)["close"].transform(lambda s: s.rolling(10).mean())
    _sma20 = df.groupby(sym)["close"].transform(lambda s: s.rolling(20).mean())
    _std10 = df.groupby(sym)["close"].transform(lambda s: s.rolling(10).std())
    _std20 = df.groupby(sym)["close"].transform(lambda s: s.rolling(20).std())
    df["zscore_10"] = (df["close"] - _sma10) / (_std10 + eps)
    df["zscore_20"] = (df["close"] - _sma20) / (_std20 + eps)
    for w in (5, 10, 20):
        df[f"dist_sma_{w}"] = df.groupby(sym)["close"].transform(
            lambda s, _w=w: s / s.rolling(_w).mean() - 1.0
        )
    df["dist_ema_9"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.ewm(span=9, adjust=False).mean() - 1.0
    )
    df["boll_pos_20"] = (df["close"] - _sma20) / (2.0 * _std20 + eps)
    df["pullback_from_high_21"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.rolling(21).max() - 1.0
    )
    df["bounce_from_low_21"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.rolling(21).min() - 1.0
    )
    _hi21 = df.groupby(sym)["high"].transform(lambda s: s.rolling(21).max())
    _lo21 = df.groupby(sym)["low"].transform(lambda s: s.rolling(21).min())
    df["close_pos_21"] = (df["close"] - _lo21) / (_hi21 - _lo21 + eps)

    # ==================================================================
    # C. Gaps / range
    # ==================================================================
    # Previous close MUST be per symbol — a raw shift over the (symbol, date)-
    # sorted frame would leak the prior symbol's tail across the boundary.
    _prev_close = df.groupby(sym)["close"].shift(1)
    df["gap_open"] = df["open"] / _prev_close - 1.0
    df["gap_abs_mean_5"] = df["gap_open"].abs().groupby(sym).transform(
        lambda s: s.rolling(5).mean()
    )
    df["range_pct_1d"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["close_pos_1d"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + eps)

    # ==================================================================
    # D. Short trend
    # ==================================================================
    df["sma_5_20_align"] = (_sma5 > _sma20).astype(float)
    df["sma20_slope_10"] = df.groupby(sym)["close"].transform(
        lambda s: s.rolling(20).mean().pct_change(10, fill_method=None)
    )
    _ema9 = df.groupby(sym)["close"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    _ema21 = df.groupby(sym)["close"].transform(
        lambda s: s.ewm(span=21, adjust=False).mean()
    )
    df["ema_9_21_spread"] = (_ema9 - _ema21) / (df["close"] + eps)

    # ADX(14) via `ta` — needs high/low/close together, so applied per symbol
    # through the index-safe _per_symbol_series helper (NOT groupby.apply).
    from ta.trend import ADXIndicator  # noqa: PLC0415

    def _adx(g: pd.DataFrame) -> pd.Series:
        return ADXIndicator(g["high"], g["low"], g["close"], window=14, fillna=False).adx()

    df["adx_14"] = _per_symbol_series(df, sym, _adx)

    # MACD histogram, normalised by price. The MACD line is index-aligned
    # (built from per-symbol EMA transforms), so its signal EMA is grouped by
    # symbol again — never an ungrouped ewm over the stacked frame.
    _ema12 = df.groupby(sym)["close"].transform(
        lambda s: s.ewm(span=12, adjust=False).mean()
    )
    _ema26 = df.groupby(sym)["close"].transform(
        lambda s: s.ewm(span=26, adjust=False).mean()
    )
    _macd = _ema12 - _ema26
    _macd_signal = _macd.groupby(sym).transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist_norm"] = (_macd - _macd_signal) / (df["close"] + eps)

    # ==================================================================
    # E. Volatility / risk
    # ==================================================================
    df["realized_vol_5"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(5).std()
    )
    df["realized_vol_10"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(10).std()
    )
    df["realized_vol_21"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(21).std()
    )
    df["vol_ratio_5_21"] = df["realized_vol_5"] / (df["realized_vol_21"] + eps)
    _hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    df["parkinson_vol_10"] = _hl.pow(2).groupby(sym).transform(
        lambda s: np.sqrt(s.rolling(10).mean() / (4 * np.log(2)))
    )
    # ATR(14) — fully vectorized true range, then a per-symbol rolling mean.
    _tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - _prev_close).abs(),
            (df["low"] - _prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["__tr"] = _tr
    df["atr_pct_14"] = df.groupby(sym)["__tr"].transform(
        lambda s: s.rolling(14).mean()
    ) / (df["close"] + eps)

    # ==================================================================
    # F. Volume confirmation
    # ==================================================================
    df["rel_volume_5"] = df.groupby(sym)["volume"].transform(
        lambda s: s / (s.rolling(5).mean() + eps)
    )
    df["rel_volume_21"] = df.groupby(sym)["volume"].transform(
        lambda s: s / (s.rolling(21).mean() + eps)
    )
    df["vol_zscore_10"] = df.groupby(sym)["volume"].transform(
        lambda s: (s - s.rolling(10).mean()) / (s.rolling(10).std() + eps)
    )
    _upvol = (df["__daily_ret"] > 0).astype(float) * df["volume"]
    df["up_vol_ratio_10"] = _upvol.groupby(sym).transform(
        lambda s: s.rolling(10).sum()
    ) / (df.groupby(sym)["volume"].transform(lambda s: s.rolling(10).sum()) + eps)

    # OBV slope — index-preserving cumsum (see module docstring), then a
    # transform for the normalised slope.
    direction = np.sign(df.groupby(sym)["close"].diff().fillna(0.0))
    df["__obv"] = (direction * df["volume"]).groupby(sym).cumsum()
    df["obv_slope_10"] = df.groupby(sym)["__obv"].transform(
        lambda s: s.diff(10) / (s.abs().rolling(10).mean() + eps)
    )
    df["volume_breakout"] = (df["rel_volume_5"] > 2.0).astype(float)

    # ==================================================================
    # G. Bases for cross-sectional ranks
    # ==================================================================
    df["vol_adj_mom_21"] = df["ret_21d"] / (df["realized_vol_21"] * np.sqrt(21) + eps)
    df["mom_consistency_10"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: _rolling_mean_positive(s, 10)
    )

    # ==================================================================
    # H. Relative strength vs index (benchmark = ['date','close'], e.g. NIFTY)
    #
    # Benchmark close is identical for every symbol on a given date, so all
    # benchmark-derived returns are computed PER SYMBOL via groupby.transform
    # — NEVER a raw shift over the (symbol, date)-sorted frame, which would
    # leak the prior symbol's tail across the block boundary. None / empty
    # benchmark -> NaN columns (fail-soft; the trainer drops NaN feature rows,
    # and serving without a benchmark scores with NaN RS, which LightGBM
    # tolerates).
    # ==================================================================
    _rs_cols = ["rs_index_5", "rs_index_10", "rs_index_21",
                "rs_index_slope_5", "beta_index_63", "corr_index_63"]
    if benchmark is not None and not benchmark.empty:
        b = (
            benchmark[["date", "close"]]
            .rename(columns={"close": "__bclose"})
            .drop_duplicates(subset="date")
        )
        df = df.merge(b, on="date", how="left")
        sym = df["symbol"]  # re-bind: merge returns a new frame/index
        for w in (5, 10, 21):
            bret_w = df.groupby(sym)["__bclose"].transform(
                lambda s, _w=w: s / s.shift(_w) - 1.0
            )
            df[f"rs_index_{w}"] = df[f"ret_{w}d"] - bret_w
        df["rs_index_slope_5"] = df.groupby(sym)["rs_index_10"].transform(
            lambda s: s.diff(5)
        )
        # rolling beta / correlation of stock vs benchmark daily returns.
        df["__bret_sym"] = df.groupby(sym)["__bclose"].transform(
            lambda s: s.pct_change(fill_method=None)
        )

        def _beta(g: pd.DataFrame) -> pd.Series:
            cov = g["__daily_ret"].rolling(63).cov(g["__bret_sym"])
            var = g["__bret_sym"].rolling(63).var()
            return cov / (var + eps)

        def _corr(g: pd.DataFrame) -> pd.Series:
            return g["__daily_ret"].rolling(63).corr(g["__bret_sym"])

        df["beta_index_63"] = _per_symbol_series(df, sym, _beta)
        df["corr_index_63"] = _per_symbol_series(df, sym, _corr)
        df.drop(columns=["__bclose", "__bret_sym"], inplace=True, errors="ignore")
    else:
        for c in _rs_cols:
            df[c] = np.nan

    # ==================================================================
    # I. Cross-sectional percentile ranks (per date, across symbols) — LAST
    # ==================================================================
    df["xs_rank_ret_5"] = df.groupby("date")["ret_5d"].rank(pct=True)
    df["xs_rank_ret_10"] = df.groupby("date")["ret_10d"].rank(pct=True)
    df["xs_rank_ret_21"] = df.groupby("date")["ret_21d"].rank(pct=True)
    df["xs_rank_zscore_20"] = df.groupby("date")["zscore_20"].rank(pct=True)
    df["xs_rank_vol_adj_mom_21"] = df.groupby("date")["vol_adj_mom_21"].rank(pct=True)
    df["xs_rank_rs_index_10"] = df.groupby("date")["rs_index_10"].rank(pct=True)

    # ------------------------------------------------------------------
    # Output: select + reorder columns; internal helpers are dropped by the
    # explicit SWING_FEATURE_ORDER select (no leakage of __* columns).
    # ------------------------------------------------------------------
    cols = ["date", "symbol"] + SWING_FEATURE_ORDER
    return df[cols].reset_index(drop=True)


__all__ = ["SWING_FEATURE_ORDER", "SWING_WARMUP_BARS", "build_swing_features"]
