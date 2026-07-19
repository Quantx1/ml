"""Momentum engine feature builder (spec §5.1).

THE single builder used by BOTH the trainer and the serving MomentumEngine
— importing this in both paths guarantees train/serve parity (audit: the
skew class of bugs). Absolute + intra-universe features only.

Implementation note: all per-symbol operations produce index-aligned output —
either via groupby(...).transform(...) or via vectorized grouped ops such as
groupby(...).cumsum() (used for OBV) that also preserve the original index.
groupby(...).apply(...) is intentionally avoided: on pandas 2.1.4 it can return
a DataFrame (not a Series) for a single-symbol panel — the serving-time scoring
path — which crashes column assignment. Do NOT "simplify" the OBV cumsum back
to an apply. The one unavoidable per-group library call (ADX, from `ta`) is
applied via an explicit per-group concat + reindex (``_per_symbol_series``),
which is robust on both single- and multi-symbol panels.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

_RET_WINDOWS = [5, 10, 21, 63, 126, 252]

_EPS = 1e-9

#: Longest warmup window any feature consumes — 252-bar return / range / SMA
#: lookbacks plus the 200-bar SMA slope (200 + 63 = 263) and the 252-bar
#: forward-return ratio (252 + 21 = 273). Callers sizing a raw OHLCV pull must
#: budget raw_days >= cv_days_needed + MOMENTUM_WARMUP_BARS + label_horizon,
#: since feature warmup AND the label's forward horizon both shrink the panel.
MOMENTUM_WARMUP_BARS = 273

MOMENTUM_FEATURE_ORDER = [
    # --- A. multi-horizon raw returns ---
    *[f"ret_{w}d" for w in _RET_WINDOWS],
    "ret_252_21",
    # --- B. momentum quality ---
    "mom_consistency_21", "mom_consistency_63", "mom_consistency_126", "mom_consistency_252",
    "mom_accel", "mom_accel_63_126", "mom_decay",
    "vol_adj_mom_63", "vol_adj_mom_126",
    "sharpe_63", "sharpe_126", "win_loss_ratio_63",
    # --- C. trend / MA alignment ---
    "dist_sma_20", "dist_sma_50", "dist_sma_100", "dist_sma_200", "dist_ema_21",
    "sma_20_50_align", "sma_50_200_align", "sma_50_slope_21", "sma_200_slope_63",
    "pct_days_above_sma50_63", "above_high_63", "dist_high_252", "price_vs_52w_range",
    "adx_14",
    # --- D. volume confirmation ---
    "rel_volume_21", "rel_volume_63", "vol_trend_21", "vol_zscore_21",
    "volume_breakout", "up_vol_ratio_21", "obv_slope_21", "obv_slope_63", "pvt_slope_21",
    # --- E. volatility / risk ---
    "realized_vol_21", "realized_vol_63", "realized_vol_126",
    "vol_ratio", "vol_of_vol_21", "downside_vol_63",
    "drawdown_252", "max_drawdown_63", "ulcer_index_63",
    "parkinson_vol_21", "atr_pct_14",
    # --- H. liquidity ---
    "turnover_21", "amihud_illiq_21", "dollar_vol_zscore_63",
    # --- G. relative strength vs index (NIFTY); NaN when no benchmark ---
    "rs_index_21", "rs_index_63", "rs_index_126", "rs_index_252",
    "rs_index_slope_21", "beta_index_63", "corr_index_63",
    # --- I. cross-sectional ranks (computed last; within-date) ---
    "xs_rank_ret_21", "xs_rank_ret_63", "xs_rank_ret_126", "xs_rank_ret_252",
    "xs_rank_vol_adj_mom_63", "xs_rank_vol_adj_mom_126", "xs_rank_sharpe_63",
    "xs_rank_rs_index_63",
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
    needs a whole-sub-frame view (ADX needs high/low/close together). Each call
    of ``fn`` returns a Series indexed like its sub-frame; we concat and reindex
    to the original row order. Works identically for one symbol or many — no
    DataFrame-vs-Series ambiguity, no cross-symbol leakage.
    """
    parts = [fn(g) for _, g in df.groupby(sym, sort=False)]
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(parts).reindex(df.index)


def build_momentum_features(
    panel: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute momentum features for a long OHLCV panel.

    Args:
        panel: DataFrame with columns
               ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
               The frame may arrive in any row order.
        benchmark: optional DataFrame ['date', 'close'] (e.g. NIFTY/NSEI). When
               provided, relative-strength-vs-index features are added; when
               None they are emitted as NaN columns (fail-soft). (RS lives in a
               sibling module section — see Task 2 in the plan.)

    Returns:
        DataFrame with columns ['date', 'symbol', *MOMENTUM_FEATURE_ORDER].
        The first ~MOMENTUM_WARMUP_BARS rows per symbol contain NaN (longest
        window undefined). After merging with labels, the trainer should drop
        NaN on the FEATURE columns specifically:

            df = features.merge(labels, on=["date", "symbol"], how="inner")
            df = df.dropna(subset=MOMENTUM_FEATURE_ORDER)

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
    # B. Momentum quality
    # ==================================================================
    for w in (21, 63, 126, 252):
        df[f"mom_consistency_{w}"] = df.groupby(sym)["__daily_ret"].transform(
            lambda s, _w=w: _rolling_mean_positive(s, _w)
        )
    df["mom_accel"] = df["ret_21d"] - df["ret_63d"]
    df["mom_accel_63_126"] = df["ret_63d"] - df["ret_126d"]
    df["mom_decay"] = df["ret_21d"] - df["ret_63d"] / 3.0  # short vs long-implied

    df["realized_vol_21"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(21).std()
    )
    df["realized_vol_63"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(63).std()
    )
    df["realized_vol_126"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(126).std()
    )
    df["vol_adj_mom_63"] = df["ret_63d"] / (df["realized_vol_21"] * np.sqrt(63) + eps)
    df["vol_adj_mom_126"] = df["ret_126d"] / (df["realized_vol_63"] * np.sqrt(126) + eps)

    df["sharpe_63"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(63).mean() / (s.rolling(63).std() + eps)
    )
    df["sharpe_126"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.rolling(126).mean() / (s.rolling(126).std() + eps)
    )

    def _win_loss(s: pd.Series) -> pd.Series:
        up = s.clip(lower=0).rolling(63).mean()
        dn = (-s.clip(upper=0)).rolling(63).mean()
        return up / (dn + eps)

    df["win_loss_ratio_63"] = df.groupby(sym)["__daily_ret"].transform(_win_loss)

    # ==================================================================
    # C. Trend / moving-average alignment
    # ==================================================================
    for w in (20, 50, 100, 200):
        df[f"dist_sma_{w}"] = df.groupby(sym)["close"].transform(
            lambda s, _w=w: s / s.rolling(_w).mean() - 1.0
        )
    df["dist_ema_21"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.ewm(span=21, adjust=False).mean() - 1.0
    )
    _sma20 = df.groupby(sym)["close"].transform(lambda s: s.rolling(20).mean())
    _sma50 = df.groupby(sym)["close"].transform(lambda s: s.rolling(50).mean())
    _sma200 = df.groupby(sym)["close"].transform(lambda s: s.rolling(200).mean())
    df["sma_20_50_align"] = (_sma20 > _sma50).astype(float)
    df["sma_50_200_align"] = (_sma50 > _sma200).astype(float)
    df["sma_50_slope_21"] = df.groupby(sym)["close"].transform(
        lambda s: s.rolling(50).mean().pct_change(21)
    )
    df["sma_200_slope_63"] = df.groupby(sym)["close"].transform(
        lambda s: s.rolling(200).mean().pct_change(63)
    )
    df["pct_days_above_sma50_63"] = (
        (df["close"] > _sma50).astype(float).groupby(sym).transform(
            lambda s: s.rolling(63).mean()
        )
    )
    df["above_high_63"] = df.groupby(sym)["close"].transform(
        lambda s: (s >= s.rolling(63).max()).astype(float)
    )
    _hi252 = df.groupby(sym)["high"].transform(lambda s: s.rolling(252).max())
    _lo252 = df.groupby(sym)["low"].transform(lambda s: s.rolling(252).min())
    df["dist_high_252"] = df["close"] / (_hi252 + eps) - 1.0
    df["price_vs_52w_range"] = (df["close"] - _lo252) / (_hi252 - _lo252 + eps)

    # ADX(14) via `ta` — needs high/low/close together, so applied per symbol
    # through the index-safe _per_symbol_series helper (NOT groupby.apply).
    from ta.trend import ADXIndicator  # noqa: PLC0415

    def _adx(g: pd.DataFrame) -> pd.Series:
        return ADXIndicator(g["high"], g["low"], g["close"], window=14, fillna=False).adx()

    df["adx_14"] = _per_symbol_series(df, sym, _adx)

    # ==================================================================
    # D. Volume confirmation
    # ==================================================================
    df["rel_volume_21"] = df.groupby(sym)["volume"].transform(
        lambda s: s / (s.rolling(21).mean() + eps)
    )
    df["rel_volume_63"] = df.groupby(sym)["volume"].transform(
        lambda s: s / (s.rolling(63).mean() + eps)
    )
    df["vol_trend_21"] = df.groupby(sym)["volume"].transform(
        lambda s: s.rolling(21).mean() / (s.rolling(63).mean() + eps)
    )
    df["vol_zscore_21"] = df.groupby(sym)["volume"].transform(
        lambda s: (s - s.rolling(21).mean()) / (s.rolling(21).std() + eps)
    )
    df["volume_breakout"] = (df["rel_volume_21"] > 2.0).astype(float)
    _upvol = (df["__daily_ret"] > 0).astype(float) * df["volume"]
    df["up_vol_ratio_21"] = _upvol.groupby(sym).transform(
        lambda s: s.rolling(21).sum()
    ) / (df.groupby(sym)["volume"].transform(lambda s: s.rolling(21).sum()) + eps)

    # OBV slope — index-preserving cumsum (see module docstring), then a
    # transform for the normalised slope at two horizons.
    direction = np.sign(df.groupby(sym)["close"].diff().fillna(0.0))
    df["__obv"] = (direction * df["volume"]).groupby(sym).cumsum()
    df["obv_slope_21"] = df.groupby(sym)["__obv"].transform(
        lambda s: s.diff(21) / (s.abs().rolling(21).mean() + eps)
    )
    df["obv_slope_63"] = df.groupby(sym)["__obv"].transform(
        lambda s: s.diff(63) / (s.abs().rolling(63).mean() + eps)
    )
    # Price-volume trend (cumulative), normalised 21-day slope.
    df["__pvt"] = (df["__daily_ret"].fillna(0.0) * df["volume"]).groupby(sym).cumsum()
    df["pvt_slope_21"] = df.groupby(sym)["__pvt"].transform(
        lambda s: s.diff(21) / (s.abs().rolling(21).mean() + eps)
    )

    # ==================================================================
    # E. Volatility / risk
    # ==================================================================
    df["vol_ratio"] = df["realized_vol_21"] / (df["realized_vol_63"] + eps)
    df["vol_of_vol_21"] = df.groupby(sym)["realized_vol_21"].transform(
        lambda s: s.rolling(21).std()
    )
    df["downside_vol_63"] = df.groupby(sym)["__daily_ret"].transform(
        lambda s: s.clip(upper=0).rolling(63).std()
    )
    df["drawdown_252"] = df.groupby(sym)["close"].transform(
        lambda s: s / s.rolling(252).max() - 1.0
    )
    df["max_drawdown_63"] = df.groupby(sym)["close"].transform(
        lambda s: (s / s.rolling(63).max() - 1.0).rolling(63).min()
    )
    _dd = df.groupby(sym)["close"].transform(lambda s: s / s.cummax() - 1.0)
    df["ulcer_index_63"] = _dd.pow(2).groupby(sym).transform(
        lambda s: np.sqrt(s.rolling(63).mean())
    )
    _hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    df["parkinson_vol_21"] = _hl.pow(2).groupby(sym).transform(
        lambda s: np.sqrt(s.rolling(21).mean() / (4 * np.log(2)))
    )
    # ATR(14) — fully vectorized true range, then a per-symbol rolling mean.
    _prev_close = df.groupby(sym)["close"].shift(1)
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
    # H. Liquidity
    # ==================================================================
    df["__tval"] = df["close"] * df["volume"]
    df["turnover_21"] = df.groupby(sym)["__tval"].transform(lambda s: s.rolling(21).mean())
    df["amihud_illiq_21"] = (df["__daily_ret"].abs() / (df["__tval"] + eps)).groupby(
        sym
    ).transform(lambda s: s.rolling(21).mean())
    df["dollar_vol_zscore_63"] = df.groupby(sym)["__tval"].transform(
        lambda s: (s - s.rolling(63).mean()) / (s.rolling(63).std() + eps)
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
    _rs_cols = ["rs_index_21", "rs_index_63", "rs_index_126", "rs_index_252",
                "rs_index_slope_21", "beta_index_63", "corr_index_63"]
    if benchmark is not None and not benchmark.empty:
        b = (
            benchmark[["date", "close"]]
            .rename(columns={"close": "__bclose"})
            .drop_duplicates(subset="date")
        )
        df = df.merge(b, on="date", how="left")
        sym = df["symbol"]  # re-bind: merge returns a new frame/index
        for w in (21, 63, 126, 252):
            bret_w = df.groupby(sym)["__bclose"].transform(
                lambda s, _w=w: s / s.shift(_w) - 1.0
            )
            df[f"rs_index_{w}"] = df[f"ret_{w}d"] - bret_w
        df["rs_index_slope_21"] = df.groupby(sym)["rs_index_63"].transform(
            lambda s: s.diff(21)
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
    df["xs_rank_ret_21"] = df.groupby("date")["ret_21d"].rank(pct=True)
    df["xs_rank_ret_63"] = df.groupby("date")["ret_63d"].rank(pct=True)
    df["xs_rank_ret_126"] = df.groupby("date")["ret_126d"].rank(pct=True)
    df["xs_rank_ret_252"] = df.groupby("date")["ret_252d"].rank(pct=True)
    df["xs_rank_vol_adj_mom_63"] = df.groupby("date")["vol_adj_mom_63"].rank(pct=True)
    df["xs_rank_vol_adj_mom_126"] = df.groupby("date")["vol_adj_mom_126"].rank(pct=True)
    df["xs_rank_sharpe_63"] = df.groupby("date")["sharpe_63"].rank(pct=True)
    df["xs_rank_rs_index_63"] = df.groupby("date")["rs_index_63"].rank(pct=True)

    # ------------------------------------------------------------------
    # Output: select + reorder columns; internal helpers are dropped by the
    # explicit MOMENTUM_FEATURE_ORDER select (no leakage of __* columns).
    # ------------------------------------------------------------------
    cols = ["date", "symbol"] + MOMENTUM_FEATURE_ORDER
    return df[cols].reset_index(drop=True)


__all__ = ["MOMENTUM_FEATURE_ORDER", "MOMENTUM_WARMUP_BARS", "build_momentum_features"]
