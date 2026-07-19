import numpy as np
import pandas as pd

from ml.features.forecast_features import merge_forecast_features, FORECAST_FEATURES


def test_ens_fwd_ret_is_mean_of_tsfm_and_kronos():
    # 2 backends present -> ens is formed (>= 2 rule); chronos absent -> NaN col
    base = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "ret_21d": [0.1]})
    tsfm = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"],
                         "tsfm_fwd_ret": [0.04], "tsfm_uncert": [0.02]})
    kron = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "kronos_fwd_ret": [0.06]})
    out = merge_forecast_features(base, [tsfm, kron])
    assert "ens_fwd_ret" in out.columns and "ens_fwd_ret" in FORECAST_FEATURES
    assert abs(out["ens_fwd_ret"].iloc[0] - 0.05) < 1e-9   # mean(0.04, 0.06)
    # contract guarantee: absent backend columns exist as NaN
    assert all(c in out.columns for c in FORECAST_FEATURES)
    assert np.isnan(out["chronos_fwd_ret"].iloc[0])
    assert np.isnan(out["chronos_uncert"].iloc[0])


def test_ens_fwd_ret_is_mean_of_all_three_backends():
    base = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "ret_21d": [0.1]})
    tsfm = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"],
                         "tsfm_fwd_ret": [0.04], "tsfm_uncert": [0.02]})
    kron = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "kronos_fwd_ret": [0.06]})
    chro = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"],
                         "chronos_fwd_ret": [0.08], "chronos_uncert": [0.01]})
    out = merge_forecast_features(base, [tsfm, kron, chro])
    assert abs(out["ens_fwd_ret"].iloc[0] - 0.06) < 1e-9   # mean(0.04, 0.06, 0.08)
    assert all(c in out.columns for c in FORECAST_FEATURES)


def test_ens_fwd_ret_nan_when_only_one_forecaster_present():
    # only TimesFM present (< 2 backends) -> ens cannot be formed -> NaN (fail-soft)
    base = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "ret_21d": [0.1]})
    tsfm = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"],
                         "tsfm_fwd_ret": [0.04], "tsfm_uncert": [0.02]})
    out = merge_forecast_features(base, [tsfm])
    assert "ens_fwd_ret" in out.columns
    assert np.isnan(out["ens_fwd_ret"].iloc[0])
    # contract guarantee still holds with a single backend
    assert all(c in out.columns for c in FORECAST_FEATURES)
