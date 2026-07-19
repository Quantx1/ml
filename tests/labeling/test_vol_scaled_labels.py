"""Vol-scaled ranking labels (``vol_adjust_window``) — the fix for rankers
learning "high volatility => extreme decile" lottery behavior.

Covers:
- default (None) is byte-identical to the original raw-return algorithm,
- with vol scaling, ranking happens on risk-adjusted return while the output
  ``fwd_return`` column stays RAW,
- warmup rows (first W per symbol, NaN trailing vol) are dropped.
"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ml.labeling.ranking_labels import forward_return_quantile_labels


def _make_panel(prices: dict, dates) -> pd.DataFrame:
    rows = []
    for sym, px in prices.items():
        for d, p in zip(dates, px):
            rows.append({"date": d, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def _random_panel(n_symbols: int = 6, n_dates: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    prices = {}
    for i in range(n_symbols):
        rets = rng.normal(loc=0.0005 * (i + 1), scale=0.005 * (i + 1), size=n_dates)
        prices[f"SYM{i}"] = 100.0 * np.cumprod(1.0 + rets)
    return _make_panel(prices, dates)


def _raw_reference(panel: pd.DataFrame, horizon: int, n_quantiles: int) -> pd.DataFrame:
    """Independent re-implementation of the ORIGINAL (raw forward return)
    ranking algorithm, used as the frozen pre-change snapshot."""
    df = panel.sort_values(["symbol", "date"]).copy()
    df["fwd_return"] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1.0
    df = df.dropna(subset=["fwd_return"]).reset_index(drop=True)
    df["relevance"] = 0
    for _, g in df.groupby("date"):
        q = min(n_quantiles, len(g))
        if q < 2:
            continue
        ranks = g["fwd_return"].rank(method="first")
        labels = pd.qcut(ranks, q, labels=False, duplicates="drop")
        df.loc[g.index, "relevance"] = labels.values
    df["relevance"] = df["relevance"].astype(int)
    return df[["date", "symbol", "fwd_return", "relevance"]].reset_index(drop=True)


def test_default_none_matches_original_algorithm_exactly():
    panel = _random_panel()
    out = forward_return_quantile_labels(panel, horizon=5, n_quantiles=4)
    expected = _raw_reference(panel, horizon=5, n_quantiles=4)
    assert list(out.columns) == ["date", "symbol", "fwd_return", "relevance"]
    assert_frame_equal(out, expected)


def _two_symbol_vol_panel():
    """Symbol A: high vol (alternating +5%/-3%). Symbol B: low vol
    (alternating +0.6%/+0.2%). On odd-index dates A's raw 5d forward return
    (~+9%) dwarfs B's (~+2%), but B's risk-adjusted return is far higher."""
    n = 80
    dates = pd.bdate_range("2021-01-04", periods=n)
    ret_a = np.array([0.05 if i % 2 == 0 else -0.03 for i in range(n)])
    ret_b = np.array([0.006 if i % 2 == 0 else 0.002 for i in range(n)])
    close_a = 100.0 * np.cumprod(1.0 + ret_a)
    close_b = 100.0 * np.cumprod(1.0 + ret_b)
    return _make_panel({"A": close_a, "B": close_b}, dates), dates


def test_vol_scaled_ranks_risk_adjusted_but_outputs_raw_return():
    panel, _ = _two_symbol_vol_panel()
    horizon, window = 5, 21
    out = forward_return_quantile_labels(
        panel, horizon=horizon, n_quantiles=10, vol_adjust_window=window
    )
    assert list(out.columns) == ["date", "symbol", "fwd_return", "relevance"]

    wide_fwd = out.pivot(index="date", columns="symbol", values="fwd_return")
    wide_rel = out.pivot(index="date", columns="symbol", values="relevance")
    both = wide_fwd.dropna().index

    # Dates where A's RAW forward return is the largest — the lottery dates.
    lottery_dates = [d for d in both if wide_fwd.loc[d, "A"] > wide_fwd.loc[d, "B"]]
    assert len(lottery_dates) > 0, "panel must contain dates where A's raw return wins"
    for d in lottery_dates:
        # Ranking is on the vol-scaled variable: low-vol B outranks high-vol A...
        assert wide_rel.loc[d, "B"] > wide_rel.loc[d, "A"]
        # ...while the OUTPUT fwd_return stays raw (A's is still larger).
        assert wide_fwd.loc[d, "A"] > wide_fwd.loc[d, "B"]

    # Output fwd_return is byte-identical to the raw (unscaled) computation
    # on the surviving rows — scaling touched only the ranking variable.
    raw = forward_return_quantile_labels(panel, horizon=horizon, n_quantiles=10)
    merged = out.merge(raw, on=["date", "symbol"], suffixes=("_vol", "_raw"))
    assert len(merged) == len(out)
    assert np.allclose(merged["fwd_return_vol"], merged["fwd_return_raw"])


def test_warmup_rows_dropped_per_symbol():
    panel, dates = _two_symbol_vol_panel()
    horizon, window = 5, 21
    raw = forward_return_quantile_labels(panel, horizon=horizon, n_quantiles=10)
    out = forward_return_quantile_labels(
        panel, horizon=horizon, n_quantiles=10, vol_adjust_window=window
    )
    for sym in ("A", "B"):
        raw_dates = raw.loc[raw["symbol"] == sym, "date"]
        vol_dates = out.loc[out["symbol"] == sym, "date"]
        # Unscaled output starts at the very first date...
        assert raw_dates.min() == dates[0]
        # ...vol-scaled output drops exactly the first W rows (rv warmup).
        assert vol_dates.min() == dates[window]
        assert len(vol_dates) == len(raw_dates) - window
