"""Absolute forward-return quantile labels for cross-sectional rankers.

Per spec §4.7: labels are ABSOLUTE (raw forward return), never relative to
a benchmark. Per rebalance date, symbols are bucketed into `n_quantiles`
by forward return; the bucket index is the LambdaRank relevance grade.

Two optional transforms change the RANKING variable only (the output
``fwd_return`` column always stays raw):

* ``vol_adjust_window`` — rank on the vol-scaled forward return
  (return-per-unit-risk), killing the vol-lottery decile failure.
* ``beta_window`` + ``benchmark`` — rank on the BETA-RESIDUALIZED forward
  return ``fwd_return - beta_t * bench_fwd_return``, removing the market
  component from the label so the ranker is graded on stock SELECTION.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def forward_return_quantile_labels(
    panel: pd.DataFrame,
    horizon: int = 20,
    n_quantiles: int = 10,
    vol_adjust_window: Optional[int] = None,
    benchmark: Optional[pd.DataFrame] = None,
    beta_window: Optional[int] = None,
) -> pd.DataFrame:
    """Return panel rows that have a full forward window, with labels.

    Why ``vol_adjust_window`` exists (the vol-lottery failure mode): within-date
    deciles of RAW forward returns are dominated by per-name volatility — the
    extreme buckets are populated almost entirely by the highest-volatility
    names, because a 5%-daily-vol stock lands in a tail decile regardless of
    its expected return. A ranker trained on those labels therefore learns
    "high volatility ⇒ extreme decile" — lottery-ticket behavior that looks
    like alpha in-sample and inverts out-of-sample (this killed the positional
    engine). Scaling the ranking variable by trailing realized vol turns the
    label into return-per-unit-risk, removing the vol confound, while the
    output ``fwd_return`` stays RAW so evaluation/backtests still measure
    realized money. No leakage: the trailing vol uses data up to and including
    t while the label window is (t, t+H] — scaling by time-t information is a
    deterministic label transform, not lookahead.

    Why ``beta_window`` exists (the beta-domination failure mode): at long
    horizons a raw H-day return is mostly ``beta * market move``, not stock
    selection — at H=60 the market component dwarfs the cross-sectional one,
    so raw-return deciles grade the ranker on market timing it has no features
    for (regime-gated evaluation proved these books' SELECTION excess is
    positive in all regimes while raw returns are beta-dominated). The
    residual label ``fwd_return - beta_t * bench_fwd_return`` removes the
    market component so quantiles grade pure selection. ``beta_t`` is the
    trailing ``beta_window``-day OLS beta of daily returns vs the benchmark
    (rolling cov/var through t — causal, like the vol transform), clipped to
    [0, 3] against outlier estimates. Deliberately UNSHRUNK: any shrinkage
    ``b_hat = w*b + (1-w)`` leaves a systematic ``(1-w)*(beta-1)*bench_fwd``
    term in the residual that a zero-skill beta-ranking model can score IC
    ~0.04-0.05 on in trending windows — above the 0.02 gate (adversarial
    review finding, 2026-07-07). Unshrunk beta is unbiased, so only zero-mean
    estimation noise remains. ``bench_fwd_return`` is future information
    about the BENCHMARK inside the label window (t, t+H] — legitimate label
    material exactly like ``fwd_return`` itself, never a feature. Does NOT
    compose with ``vol_adjust_window`` (raises): the exported
    ``resid_fwd_return`` evaluation target would not match the vol-scaled
    ranking variable, re-creating the gate mismatch ic_target_col closed.

    Args:
        panel: long frame with at least ['date', 'symbol', 'close'].
        horizon: forward window in bars (e.g. 20 trading days).
        n_quantiles: number of relevance buckets (10 = deciles).
        vol_adjust_window: if set (W), rank on the risk-adjusted forward
            return ``fwd_return / (rv * sqrt(horizon) + 1e-9)`` where
            ``rv = daily_ret.rolling(W).std()`` per symbol (trailing, through
            t). Warmup rows with NaN rv are dropped like NaN fwd_return rows.
            None (default) preserves the original raw-return ranking exactly.
        benchmark: ['date', 'close'] index series (e.g. NIFTY). Required when
            ``beta_window`` is set; ignored otherwise.
        beta_window: if set (W), rank on the beta-residualized forward return
            using the trailing W-day rolling beta vs ``benchmark``
            (min_periods = max(20, W // 2)). Warmup rows (NaN beta) and rows
            whose date the benchmark doesn't cover are dropped. None (default)
            preserves the original ranking variable.

    Returns:
        DataFrame ['date', 'symbol', 'fwd_return', 'relevance'] where
        relevance in [0, n_quantiles-1] (higher = better forward return),
        computed per-date cross-sectionally. Rows without a full forward
        window are dropped. ``fwd_return`` is always the RAW forward return,
        even when the ranking variable is transformed. When ``beta_window``
        is set the frame carries an extra ``resid_fwd_return`` column (the
        residual BEFORE any vol scaling) so evaluation can grade the model
        on the target it was trained to rank (EvalSpec.ic_target_col).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if beta_window is not None and benchmark is None:
        raise ValueError("beta_window requires a benchmark frame ['date','close']")
    if beta_window is not None and vol_adjust_window is not None:
        raise ValueError(
            "beta_window + vol_adjust_window combined is unsupported: relevance "
            "would rank the VOL-SCALED residual while the exported "
            "resid_fwd_return evaluation target is the unscaled residual — the "
            "same train/eval mismatch EvalSpec.ic_target_col exists to prevent")
    df = panel.sort_values(["symbol", "date"]).copy()
    df["fwd_close"] = df.groupby("symbol")["close"].shift(-horizon)
    df["fwd_return"] = df["fwd_close"] / df["close"] - 1.0

    rank_col = "fwd_return"
    drop_cols = ["fwd_return"]

    if beta_window is not None:
        bench = benchmark.sort_values("date").copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench["_bench_ret"] = bench["close"].pct_change(fill_method=None)
        bench["_bench_fwd"] = bench["close"].shift(-horizon) / bench["close"] - 1.0
        # many_to_one: duplicate benchmark dates would silently fan out panel
        # rows into duplicate (date, symbol) labels — fail loud instead.
        df = df.merge(bench[["date", "_bench_ret", "_bench_fwd"]],
                      on="date", how="left", validate="many_to_one")
        df["_ret_1d"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
        min_p = max(20, beta_window // 2)
        betas = []
        for _, g in df.groupby("symbol", sort=False):
            cov = g["_ret_1d"].rolling(beta_window, min_periods=min_p).cov(g["_bench_ret"])
            var = g["_bench_ret"].rolling(beta_window, min_periods=min_p).var()
            betas.append(cov / var.replace(0.0, np.nan))
        # UNSHRUNK by design (see docstring): shrinkage leaves a systematic
        # beta-rankable term in the residual; the clip alone tames outliers.
        df["_beta"] = pd.concat(betas).clip(0.0, 3.0)
        df["_resid_fwd"] = df["fwd_return"] - df["_beta"] * df["_bench_fwd"]
        rank_col = "_resid_fwd"
        drop_cols = ["fwd_return", "_resid_fwd"]

    if vol_adjust_window is not None:
        daily_ret = df.groupby("symbol")["close"].pct_change(fill_method=None)
        rv = daily_ret.groupby(df["symbol"]).transform(
            lambda s: s.rolling(vol_adjust_window).std()
        )
        df["_risk_adj_fwd"] = df[rank_col] / (rv * np.sqrt(horizon) + 1e-9)
        rank_col = "_risk_adj_fwd"
        drop_cols = list(dict.fromkeys([*drop_cols, "_risk_adj_fwd"]))
    df = df.dropna(subset=drop_cols).reset_index(drop=True)

    def _bucket(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        q = min(n_quantiles, n)
        if q < 2:
            return pd.Series(0, index=group.index)
        ranks = group[rank_col].rank(method="first")
        labels = pd.qcut(ranks, q, labels=False, duplicates="drop")
        return pd.Series(labels.values, index=group.index)

    df["relevance"] = (
        df.groupby("date", group_keys=False).apply(_bucket).astype(int)
    )
    out_cols = ["date", "symbol", "fwd_return", "relevance"]
    if beta_window is not None:
        df = df.rename(columns={"_resid_fwd": "resid_fwd_return"})
        out_cols.append("resid_fwd_return")
    return df[out_cols].reset_index(drop=True)
