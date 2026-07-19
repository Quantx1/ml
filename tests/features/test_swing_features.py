import numpy as np
import pandas as pd

from ml.features.swing_features import build_swing_features, SWING_FEATURE_ORDER

_RS_COLS = ["rs_index_5", "rs_index_10", "rs_index_21",
            "rs_index_slope_5", "beta_index_63", "corr_index_63"]


def _panel(n_days=300, symbols=("A", "B", "C")):
    days = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    rng = np.random.default_rng(0)
    for s_i, sym in enumerate(symbols):
        price = 100 + np.cumsum(rng.normal(0.1 * (s_i + 1), 1.0, n_days))
        for d, p in zip(days, price):
            rows.append({"date": d, "symbol": sym, "open": p, "high": p + 1,
                         "low": p - 1, "close": p, "volume": 1000 + s_i})
    return pd.DataFrame(rows)


def _series(n=400, slope=40.0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return idx, 100 + np.linspace(0, slope, n)


def _trend_panel(idx, close, symbol="AAA"):
    return pd.DataFrame({"date": idx, "symbol": symbol, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1_000_000})


def test_columns_match_feature_order():
    feats = build_swing_features(_panel())
    for col in SWING_FEATURE_ORDER:
        assert col in feats.columns, f"missing {col}"
    assert {"date", "symbol"}.issubset(feats.columns)


def test_no_lookahead_in_returns():
    panel = _panel()
    feats = build_swing_features(panel)
    a = feats[feats.symbol == "A"].sort_values("date").reset_index(drop=True)
    pa = panel[panel.symbol == "A"].sort_values("date").reset_index(drop=True)
    t = 100
    # ret_5d at row t must equal close[t]/close[t-5]-1 within the symbol —
    # uses only past bars (no lookahead).
    expected = pa["close"].iloc[t] / pa["close"].iloc[t - 5] - 1
    assert abs(a["ret_5d"].iloc[t] - expected) < 1e-9


def test_train_serve_parity_same_input_same_output():
    panel = _panel()
    f1 = build_swing_features(panel)
    f2 = build_swing_features(panel.copy())
    pd.testing.assert_frame_equal(f1, f2)


def test_single_symbol_serve_path_does_not_crash():
    # Serving scores one symbol at a time. A single-symbol panel must
    # produce all feature columns without the groupby-returns-DataFrame
    # crash (regression class from the momentum obv_slope bug).
    feats = build_swing_features(_panel(symbols=("A",)))
    for col in SWING_FEATURE_ORDER:
        assert col in feats.columns, f"missing {col}"
    assert set(feats["symbol"]) == {"A"}
    assert feats["obv_slope_10"].notna().any()


def test_expanded_feature_set_present_and_finite():
    # one symbol, 400 trading days of a gentle uptrend, WITH a benchmark so
    # the RS-vs-index cols are finite too.
    idx, close = _series(slope=40)
    panel = _trend_panel(idx, close)
    benchmark = pd.DataFrame({"date": idx, "close": 100 + np.linspace(0, 10, 400)})
    out = build_swing_features(panel, benchmark=benchmark)
    # every declared feature column is produced
    for col in SWING_FEATURE_ORDER:
        assert col in out.columns, f"missing {col}"
    # last row (serving bar) is finite for every feature after warmup
    last = out.dropna(subset=SWING_FEATURE_ORDER).iloc[-1]
    assert np.isfinite(last[SWING_FEATURE_ORDER].astype(float)).all()
    assert len(SWING_FEATURE_ORDER) >= 50


def test_rs_features_failsoft_without_benchmark():
    idx, close = _series()
    panel = _trend_panel(idx, close)
    out = build_swing_features(panel, benchmark=None)  # no benchmark -> RS cols NaN, not a crash
    for col in _RS_COLS:
        assert out[col].isna().all(), f"{col} should be all-NaN without a benchmark"


def test_rs_no_cross_symbol_leak_two_symbols():
    idx, close_a = _series(slope=40)
    _, close_b = _series(slope=20)
    _, bench = _series(slope=10)
    panel = pd.concat(
        [_trend_panel(idx, close_a, "AAA"), _trend_panel(idx, close_b, "BBB")],
        ignore_index=True,
    )
    benchmark = pd.DataFrame({"date": idx, "close": bench})
    out = build_swing_features(panel, benchmark=benchmark)

    last_date = idx[-1]
    a = out[(out.symbol == "AAA") & (out.date == last_date)].iloc[0]
    b = out[(out.symbol == "BBB") & (out.date == last_date)].iloc[0]

    # rs_index_10 for BBB must equal stock_10d_ret - bench_10d_ret computed
    # independently (per-symbol benchmark return, no cross-symbol shift).
    stock_ret = close_b[-1] / close_b[-1 - 10] - 1.0
    bench_ret = bench[-1] / bench[-1 - 10] - 1.0
    assert abs(b["rs_index_10"] - (stock_ret - bench_ret)) < 1e-9

    # The implied benchmark return (ret_10d - rs_index_10) depends ONLY on the
    # benchmark, so it must be identical across symbols on the same date. A
    # cross-symbol shift leak would make these differ.
    impl_a = a["ret_10d"] - a["rs_index_10"]
    impl_b = b["ret_10d"] - b["rs_index_10"]
    assert abs(impl_a - impl_b) < 1e-9

    # An UNGROUPED shift would fill each later symbol's warmup rows with the
    # previous symbol's tail (non-NaN garbage). Warmup rows must be NaN for
    # EVERY symbol — this single assertion catches any ungrouped shift.
    assert out.groupby("symbol")["ret_10d"].apply(
        lambda s: s.head(10).isna().all()).all()
    assert out.groupby("symbol")["rs_index_10"].apply(
        lambda s: s.head(10).isna().all()).all()
