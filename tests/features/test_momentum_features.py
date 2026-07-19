import numpy as np
import pandas as pd
from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER


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


def test_columns_match_feature_order():
    feats = build_momentum_features(_panel())
    for col in MOMENTUM_FEATURE_ORDER:
        assert col in feats.columns, f"missing {col}"
    assert {"date", "symbol"}.issubset(feats.columns)


def test_no_lookahead_in_returns():
    panel = _panel()
    feats = build_momentum_features(panel)
    a = feats[feats.symbol == "A"].sort_values("date").reset_index(drop=True)
    pa = panel[panel.symbol == "A"].sort_values("date").reset_index(drop=True)
    t = 100
    # ret_21d at row t must equal close[t]/close[t-21]-1 within the symbol —
    # uses only past bars (no lookahead).
    expected = pa["close"].iloc[t] / pa["close"].iloc[t - 21] - 1
    assert abs(a["ret_21d"].iloc[t] - expected) < 1e-9


def test_train_serve_parity_same_input_same_output():
    panel = _panel()
    f1 = build_momentum_features(panel)
    f2 = build_momentum_features(panel.copy())
    pd.testing.assert_frame_equal(f1, f2)


def test_single_symbol_serve_path_does_not_crash():
    # Serving scores one symbol at a time. A single-symbol panel must
    # produce all feature columns without the groupby-returns-DataFrame
    # crash (regression test for the obv_slope_21 bug).
    feats = build_momentum_features(_panel(symbols=("A",)))
    for col in MOMENTUM_FEATURE_ORDER:
        assert col in feats.columns, f"missing {col}"
    assert set(feats["symbol"]) == {"A"}
    assert feats["obv_slope_21"].notna().any()


def test_expanded_feature_set_present_and_finite():
    # one symbol, 400 trading days of a gentle uptrend
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    close = 100 + np.linspace(0, 40, 400)
    panel = pd.DataFrame({"date": idx, "symbol": "AAA", "open": close, "high": close * 1.01,
                          "low": close * 0.99, "close": close, "volume": 1_000_000})
    # the full feature vector includes RS-vs-index cols, which need a benchmark
    # to be finite (without one they are NaN — see test_momentum_rs.py).
    benchmark = pd.DataFrame({"date": idx, "close": 100 + np.linspace(0, 10, 400)})
    out = build_momentum_features(panel, benchmark=benchmark)
    # every declared feature column is produced
    for col in MOMENTUM_FEATURE_ORDER:
        assert col in out.columns, f"missing {col}"
    # the new feature families exist
    for col in ["ret_252_21", "mom_decay", "sharpe_63", "dist_ema_21", "sma_50_200_align",
                "vol_zscore_21", "parkinson_vol_21", "ulcer_index_63", "adx_14",
                "turnover_21", "amihud_illiq_21", "xs_rank_ret_252"]:
        assert col in MOMENTUM_FEATURE_ORDER, f"{col} not in feature order"
    # last row (serving bar) is finite for every feature after warmup
    last = out.dropna(subset=MOMENTUM_FEATURE_ORDER).iloc[-1]
    assert np.isfinite(last[MOMENTUM_FEATURE_ORDER].astype(float)).all()
    assert len(MOMENTUM_FEATURE_ORDER) >= 60


def test_single_symbol_path_no_crash():
    idx = pd.date_range("2022-01-01", periods=320, freq="B")
    close = 100 + np.linspace(0, 30, 320)
    panel = pd.DataFrame({"date": idx, "symbol": "ONE", "open": close, "high": close * 1.01,
                          "low": close * 0.99, "close": close, "volume": 500_000})
    out = build_momentum_features(panel)   # must not raise on a single symbol
    assert set(MOMENTUM_FEATURE_ORDER).issubset(out.columns)
