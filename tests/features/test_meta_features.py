"""Meta-labeling conviction features — PIT and cross-sectional correctness."""
import numpy as np
import pandas as pd
import pytest

from ml.features.meta_features import (
    META_ENGINE_MAP,
    build_meta_features,
    meta_feature_cols,
)

N_DATES, N_SYMS = 30, 10


def _world(seed: int = 5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=N_DATES)
    syms = [f"S{i}" for i in range(N_SYMS)]
    preds = pd.DataFrame(
        [{"date": d, "symbol": s, "pred": float(rng.normal()), "fwd_return": 0.01}
         for d in dates for s in syms])
    ef_cols = (META_ENGINE_MAP["momentum"]["name_features"]
               + META_ENGINE_MAP["momentum"]["forecast_features"])
    engine_feats = pd.DataFrame(
        [{"date": d, "symbol": s, **{c: float(rng.normal()) for c in ef_cols}}
         for d in dates for s in syms])
    states = ["bull"] * 10 + ["bear"] * 10 + ["sideways"] * 10
    regime = pd.DataFrame({"date": dates, "state_name": states,
                           "confidence": np.linspace(0.5, 1.0, N_DATES)})
    bench_dates = pd.bdate_range("2024-08-01", periods=180)  # warmup room
    bench = pd.DataFrame({"date": bench_dates,
                          "close": 1000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 180))})
    return preds, engine_feats, regime, bench


def test_feature_frame_complete_and_ordered():
    preds, ef, regime, bench = _world()
    df, cols = build_meta_features(preds, ef, "momentum", regime, bench)
    assert cols == meta_feature_cols("momentum")
    assert len(df) == len(preds)
    assert df[cols].notna().all().all()  # benchmark warmup pre-dates the panel
    # pass-through of preds columns
    assert {"pred", "fwd_return"} <= set(df.columns)


def test_score_pct_is_within_date_percentile():
    preds, ef, regime, bench = _world()
    df, _ = build_meta_features(preds, ef, "momentum", regime, bench)
    one = df[df["date"] == df["date"].iloc[0]].sort_values("pred")
    assert one["score_pct"].iloc[-1] == 1.0            # best name that date
    assert one["score_pct"].is_monotonic_increasing    # percentile == order
    assert one["score_dispersion"].nunique() == 1      # per-date constant


def test_days_since_switch_is_causal_cumcount():
    preds, ef, regime, bench = _world()
    df, _ = build_meta_features(preds, ef, "momentum", regime, bench)
    per_date = df.drop_duplicates("date").set_index("date")
    dates = sorted(per_date.index)
    # bull run: days 0..9; switch to bear at date 10 resets to 0
    assert per_date.loc[dates[9], "days_since_switch"] == 9.0
    assert per_date.loc[dates[10], "days_since_switch"] == 0.0
    assert per_date.loc[dates[10], "regime_bear"] == 1.0


def test_dates_outside_regime_series_stay_nan():
    """Pre-regime-series rows carry NaN (dropped downstream) — never filled."""
    preds, ef, regime, bench = _world()
    regime_late = regime.iloc[5:]  # regime series starts 5 days late
    df, _ = build_meta_features(preds, ef, "momentum", regime_late, bench)
    early = df[df["date"] < regime_late["date"].min()]
    assert len(early) == 5 * N_SYMS
    assert early["regime_confidence"].isna().all()


def test_missing_engine_column_raises():
    preds, ef, regime, bench = _world()
    with pytest.raises(ValueError, match="missing"):
        build_meta_features(preds, ef.drop(columns=["beta_index_63"]),
                            "momentum", regime, bench)


def test_unknown_engine_raises():
    preds, ef, regime, bench = _world()
    with pytest.raises(KeyError):
        build_meta_features(preds, ef, "positional", regime, bench)
