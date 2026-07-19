"""Positional engine feature builder (positional plan Task 1).

THE single builder used by BOTH the trainer and the serving PositionalEngine
— importing this in both paths guarantees train/serve parity (audit: the
skew class of bugs). Long-horizon families ONLY (63-252 day lookbacks): the
positional book holds for ~60 trading days, so the 1-10d mean-reversion /
gap / short-trend noise that powers the swing engine is deliberately absent.

Implementation note: all per-symbol operations produce index-aligned output —
either via groupby(...).transform(...) or via vectorized grouped ops that also
preserve the original index. groupby(...).apply(...) is intentionally avoided:
on pandas 2.1.4 it can return a DataFrame (not a Series) for a single-symbol
panel — the serving-time scoring path — which crashes column assignment. The
unavoidable per-group whole-frame calls (rolling beta/corr vs the benchmark)
are applied via an explicit per-group concat + reindex (``_per_symbol_series``),
which is robust on both single- and multi-symbol panels. No `ta` indicators
are used here (no ADX/RSI — those are short-horizon families).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

_RET_WINDOWS = [21, 42, 63, 126, 189, 252]

_EPS = 1e-9

#: Conservative warmup budget — the longest lookback chains are the 200-bar
#: SMA slope (200 + 63 = 263), the 12-1 return (shift-252 based, 252 bars, 273
#: by the momentum counting convention) and the 252-bar beta/corr/return
#: windows; 315 adds margin on top (e.g. benchmark gaps that push the first
#: valid beta_index_252 window out). Callers sizing a raw OHLCV pull must
#: budget raw_days >= cv_days_needed + POSITIONAL_WARMUP_BARS + label_horizon,
#: since feature warmup AND the label's forward horizon both shrink the panel.
POSITIONAL_WARMUP_BARS = 315

POSITIONAL_FEATURE_ORDER = [
    # --- A. long-horizon raw returns ---
    *[f"ret_{w}d" for w in _RET_WINDOWS],
    "ret_252_21",
    # --- B. momentum quality (long windows) ---
    "mom_consistency_126", "mom_consistency_252",
    "sharpe_126", "sharpe_252",
    "vol_adj_mom_126", "vol_adj_mom_252",
    # --- C. long trend / MA structure ---
    "dist_sma_50", "dist_sma_100", "dist_sma_200",
    "sma_50_200_align", "sma_100_slope_63", "sma_200_slope_63",
    "pct_days_above_sma200_126", "dist_high_252", "price_vs_52w_range",
    # --- D. drawdown / risk ---
    "drawdown_252", "max_drawdown_126", "ulcer_index_126",
    "realized_vol_63", "realized_vol_126", "realized_vol_252",
    # --- E. volatility structure ---
    "vol_ratio_63_252", "downside_vol_126",
    # --- F. liquidity ---
    "turnover_63", "amihud_illiq_63", "dollar_vol_zscore_126",
    # --- G. relative strength vs index (NIFTY); NaN when no benchmark ---
    "rs_index_63", "rs_index_126", "rs_index_252",
    "rs_index_slope_21", "beta_index_252", "corr_index_252",
    # --- H. cross-sectional ranks (computed last; within-date) ---
    "xs_rank_ret_126", "xs_rank_ret_252", "xs_rank_ret_252_21",
    "xs_rank_vol_adj_mom_126", "xs_rank_vol_adj_mom_252",
    "xs_rank_rs_index_126", "xs_rank_rs_index_252",
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
    needs a whole-sub-frame view (rolling beta/corr need the stock AND benchmark
    return columns together). Each call of ``fn`` returns a Series indexed like
    its sub-frame; we concat and reindex to the original row order. Works
    identically for one symbol or many — no DataFrame-vs-Series ambiguity, no
    cross-symbol leakage.
    """
    parts = [fn(g) for _, g in df.groupby(sym, sort=False)]
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(parts).reindex(df.index)


def build_positional_features(
    panel: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute positional (long-horizon) features for a long OHLCV panel.

    Args:
        panel: DataFrame with columns
               ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
               The frame may arrive in any row order.
        benchmark: optional DataFrame ['date', 'close'] (e.g. NIFTY/NSEI). When
               provided, relative-strength-vs-index features are added; when
               None they are emitted as NaN columns (fail-soft).

    Returns:
        DataFrame with columns ['date', 'symbol', *POSITIONAL_FEATURE_ORDER].
        The first ~POSITIONAL_WARMUP_BARS rows per symbol contain NaN (longest
        window undefined). After merging with labels, the trainer should drop
        NaN on the FEATURE columns specifically:

            df = features.merge(labels, on=["date", "symbol"], how="inner")
            df = df.dropna(subset=POSITIONAL_FEATURE_ORDER)

        (a bare ``dropna()`` would also drop valid label rows; dropping only
        on a single feature would miss other warmup-controlled NaNs).
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    sym = df["symbol"]
    eps = _EPS

    # ==================================================================
    # A. Long-horizon raw returns (close[t]/close[t-w] - 1, per symbol)
    # ==================================================================
    for w in _RET_WINDOWS:
        df[f"ret_{w}d"] = df.groupby(sym)["close"].transform(
            lambda s, _w=w: s / s.shift(_w) - 1.0
        )
    # 12-1 style: return from t-252 to t-21 (skip the most recent month)
    df["ret_252_21"] = df.groupby(sym)["close"].transform(
        lambda s: s.shift(21) / s.shift(252) - 1.0
    )

    # Daily returns — the base for several derived features. fill_method=None:
    # do NOT forward-fill across gaps (silent ffill is deprecated and wrong for
    # returns — a gap should yield NaN, not a fabricated 0%).
    df["__daily_ret"] = df.groupby(sym)["close"].transform(
        lambda s: s.pct_change(fill_method=None)
    )

    # ==================================================================
    # B. Momentum quality (long windows)
    # ==================================================================
    for w in (126, 252):
        df[f"mom_consistency_{w}"] = df.groupby(sym)["__daily_ret"].transform(
            lambda s, _w=w: _rolling_mean_positive(s, _w)
        )
    df["sharpe_126"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(126).mean() / (s.rolling(126).std() + eps)
    )
    df["sharpe_252"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(252).mean() / (s.rolling(252).std() + eps)
    )

    df["realized_vol_63"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(63).std()
    )
    df["realized_vol_126"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(126).std()
    )
    df["realized_vol_252"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(252).std()
    )
    df["vol_adj_mom_126"] = df["ret_126d"] / (df["realized_vol_63"] * np.sqrt(126) + eps)
    df["vol_adj_mom_252"] = df["ret_252d"] / (df["realized_vol_126"] * np.sqrt(252) + eps)

    # ==================================================================
    # C. Long trend / moving-average structure
    # ==================================================================
    for w in (50, 100, 200):
        df[f"dist_sma_{w}"] = df.groupby(sym)["close"].transform(
            lambda s, _w=w: s / s.rolling(_w).mean() - 1.0
        )
    _sma50 = df.groupby(sym)["close"].transform(lambda s: s.rolling(50).mean())
    _sma200 = df.groupby(sym)["close"].transform(lambda s: s.rolling(200).mean())
    df["sma_50_200_align"] = (_sma50 > _sma200).astype(float)
    df["sma_100_slope_63"] = df.groupby(sym)["close"].transform(
        lambda s: s.rolling(100).mean().pct_change(63, fill_method=None)
    )
    df["sma_200_slope_63"] = df.groupby(sym)["close"].transform(
        lambda s: s.rolling(200).mean().pct_change(63, fill_method=None)
    )
    df["pct_days_above_sma200_126"] = (
        (df["close"] > _sma200).astype(float).groupby(sym).transform(
            lambda s: s.rolling(126).mean()
        )
    )
    _hi252 = df.groupby(sym)["high"].transform(lambda s: s.rolling(252).max())
    _lo252 = df.groupby(sym)["low"].transform(lambda s: s.rolling(252).min())
    df["dist_high_252"] = df["close"] / (_hi252 + eps) - 1.0
    df["price_vs_52w_range"] = (df["close"] - _lo252) / (_hi252 - _lo252 + eps)

    # ==================================================================
    # D. Drawdown / risk
    # ==================================================================
    df["drawdown_252"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.rolling(252).max() - 1.0
    )
    df["max_drawdown_126"] = df.groupby(sym)["close"].transform(
        lambda s: (s / s.rolling(126).max() - 1.0).rolling(126).min()
    )
    _dd = df.groupby(sym)["close"].transform(lambda s: s / s.cummax() - 1.0)
    df["ulcer_index_126"] = _dd.pow(2).groupby(sym).transform(
        lambda s: np.sqrt(s.rolling(126).mean())
    )

    # ==================================================================
    # E. Volatility structure
    # ==================================================================
    df["vol_ratio_63_252"] = df["realized_vol_63"] / (df["realized_vol_252"] + eps)
    df["downside_vol_126"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.clip(upper=0).rolling(126).std()
    )

    # ==================================================================
    # F. Liquidity
    # ==================================================================
    df["__tval"] = df["close"] * df["volume"]
    df["turnover_63"] = df.groupby(sym)["__tval"].transform(lambda s: s.rolling(63).mean())
    df["amihud_illiq_63"] = (df["__daily_ret"].abs() / (df["__tval"] + eps)).groupby(
        sym
    ).transform(lambda s: s.rolling(63).mean())
    df["dollar_vol_zscore_126"] = df.groupby(sym)["__tval"].transform(
        lambda s: (s - s.rolling(126).mean()) / (s.rolling(126).std() + eps)
    )

    # ==================================================================
    # G. Relative strength vs index (benchmark = ['date','close'], e.g. NIFTY)
    #
    # Benchmark close is identical for every symbol on a given date, so all
    # benchmark-derived returns are computed PER SYMBOL via groupby.transform
    # — NEVER a raw shift over the (symbol, date)-sorted frame, which would
    # leak the prior symbol's tail across the block boundary. None / empty
    # benchmark -> NaN columns (fail-soft; the trainer drops NaN feature rows,
    # and serving without a benchmark scores with NaN RS, which LightGBM
    # tolerates).
    # ==================================================================
    _rs_cols = ["rs_index_63", "rs_index_126", "rs_index_252",
                "rs_index_slope_21", "beta_index_252", "corr_index_252"]
    if benchmark is not None and not benchmark.empty:
        b = (
            benchmark[["date", "close"]]
            .rename(columns={"close": "__bclose"})
            .drop_duplicates(subset="date")
        )
        df = df.merge(b, on="date", how="left")
        sym = df["symbol"]  # re-bind: merge returns a new frame/index
        for w in (63, 126, 252):
            bret_w = df.groupby(sym)["__bclose"].transform(
                lambda s, _w=w: s / s.shift(_w) - 1.0
            )
            df[f"rs_index_{w}"] = df[f"ret_{w}d"] - bret_w
        df["rs_index_slope_21"] = df.groupby(sym)["rs_index_126"].transform(
            lambda s: s.diff(21)
        )
        # rolling beta / correlation of stock vs benchmark daily returns.
        df["__bret_sym"] = df.groupby(sym)["__bclose"].transform(
            lambda s: s.pct_change(fill_method=None)
        )

        def _beta(g: pd.DataFrame) -> pd.Series:
            cov = g["__daily_ret"].rolling(252).cov(g["__bret_sym"])
            var = g["__bret_sym"].rolling(252).var()
            return cov / (var + eps)

        def _corr(g: pd.DataFrame) -> pd.Series:
            return g["__daily_ret"].rolling(252).corr(g["__bret_sym"])

        df["beta_index_252"] = _per_symbol_series(df, sym, _beta)
        df["corr_index_252"] = _per_symbol_series(df, sym, _corr)
        df.drop(columns=["__bclose", "__bret_sym"], inplace=True, errors="ignore")
    else:
        for c in _rs_cols:
            df[c] = np.nan

    # ==================================================================
    # H. Cross-sectional percentile ranks (per date, across symbols) — LAST
    # ==================================================================
    df["xs_rank_ret_126"] = df.groupby("date")["ret_126d"].rank(pct=True)
    df["xs_rank_ret_252"] = df.groupby("date")["ret_252d"].rank(pct=True)
    df["xs_rank_ret_252_21"] = df.groupby("date")["ret_252_21"].rank(pct=True)
    df["xs_rank_vol_adj_mom_126"] = df.groupby("date")["vol_adj_mom_126"].rank(pct=True)
    df["xs_rank_vol_adj_mom_252"] = df.groupby("date")["vol_adj_mom_252"].rank(pct=True)
    df["xs_rank_rs_index_126"] = df.groupby("date")["rs_index_126"].rank(pct=True)
    df["xs_rank_rs_index_252"] = df.groupby("date")["rs_index_252"].rank(pct=True)

    # ------------------------------------------------------------------
    # Output: select + reorder columns; internal helpers are dropped by the
    # explicit POSITIONAL_FEATURE_ORDER select (no leakage of __* columns).
    # ------------------------------------------------------------------
    cols = ["date", "symbol"] + POSITIONAL_FEATURE_ORDER
    return df[cols].reset_index(drop=True)


__all__ = [
    "POSITIONAL_FEATURE_ORDER",
    "POSITIONAL_WARMUP_BARS",
    "build_positional_features",
]
