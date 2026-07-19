import numpy as np
import pandas as pd

from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER

_RS_COLS = ["rs_index_21", "rs_index_63", "rs_index_126", "rs_index_252",
            "rs_index_slope_21", "beta_index_63", "corr_index_63", "xs_rank_rs_index_63"]


def _series(n=400, slope=40.0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return idx, 100 + np.linspace(0, slope, n)


def _panel(idx, close, symbol="AAA"):
    return pd.DataFrame({"date": idx, "symbol": symbol, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1_000_000})


def test_rs_features_with_benchmark():
    idx, close = _series(slope=40)            # stock up 40%
    _, bench = _series(slope=10)              # NIFTY up 10% -> stock has positive RS
    panel = _panel(idx, close)
    benchmark = pd.DataFrame({"date": idx, "close": bench})
    out = build_momentum_features(panel, benchmark=benchmark)
    for col in _RS_COLS:
        assert col in out.columns, f"missing column {col}"
        assert col in MOMENTUM_FEATURE_ORDER, f"{col} not in feature order"
    last = out.dropna(subset=["rs_index_63"]).iloc[-1]
    assert last["rs_index_63"] > 0          # outperforming the benchmark


def test_rs_features_failsoft_without_benchmark():
    idx, close = _series()
    panel = _panel(idx, close)
    out = build_momentum_features(panel, benchmark=None)   # no benchmark -> RS cols NaN, not a crash
    for col in ["rs_index_21", "rs_index_63", "rs_index_126", "rs_index_252",
                "rs_index_slope_21", "beta_index_63", "corr_index_63"]:
        assert out[col].isna().all(), f"{col} should be all-NaN without a benchmark"


def test_rs_no_cross_symbol_leak_two_symbols():
    idx, close_a = _series(slope=40)
    _, close_b = _series(slope=20)
    _, bench = _series(slope=10)
    panel = pd.concat([_panel(idx, close_a, "AAA"), _panel(idx, close_b, "BBB")], ignore_index=True)
    benchmark = pd.DataFrame({"date": idx, "close": bench})
    out = build_momentum_features(panel, benchmark=benchmark)

    last_date = idx[-1]
    a = out[(out.symbol == "AAA") & (out.date == last_date)].iloc[0]
    b = out[(out.symbol == "BBB") & (out.date == last_date)].iloc[0]

    # rs_index_63 for BBB must equal stock_63d_ret - bench_63d_ret computed
    # independently (per-symbol benchmark return, no cross-symbol shift).
    stock_ret = close_b[-1] / close_b[-1 - 63] - 1.0
    bench_ret = bench[-1] / bench[-1 - 63] - 1.0
    assert abs(b["rs_index_63"] - (stock_ret - bench_ret)) < 1e-9

    # The implied benchmark return (ret_63d - rs_index_63) depends ONLY on the
    # benchmark, so it must be identical across symbols on the same date. A
    # cross-symbol shift leak would make these differ.
    impl_a = a["ret_63d"] - a["rs_index_63"]
    impl_b = b["ret_63d"] - b["rs_index_63"]
    assert abs(impl_a - impl_b) < 1e-9

    # An UNGROUPED shift would fill each later symbol's warmup rows with the
    # previous symbol's tail (non-NaN garbage). Warmup rows must be NaN for
    # EVERY symbol — this single assertion catches any ungrouped shift.
    assert out.groupby("symbol")["ret_63d"].apply(
        lambda s: s.head(63).isna().all()).all()
    assert out.groupby("symbol")["rs_index_63"].apply(
        lambda s: s.head(63).isna().all()).all()
