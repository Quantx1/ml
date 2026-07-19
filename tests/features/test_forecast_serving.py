"""forecast_serving.latest_forecasts — serving-side cache reader tests.

Synthetic parquets in a tmp cache dir; no GPU, no network, no real cache.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features.forecast_serving import ENGINE_CACHE_FILES, latest_forecasts


def _frame(cols, symbols=("AAA", "BBB"), n=30, start="2026-01-01"):
    """Per-symbol daily rows; the value encodes (symbol index, day index) so
    tests can assert the LAST row per symbol was picked."""
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for i, s in enumerate(symbols):
        for j, d in enumerate(dates):
            r = {"date": d, "symbol": s}
            for c in cols:
                r[c] = (i + 1) * 1.0 + j * 0.001
            rows.append(r)
    return pd.DataFrame(rows)


def _write(cache: Path, name: str, df: pd.DataFrame):
    cache.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache / name, index=False)


def test_latest_row_per_symbol_with_ensemble(tmp_path):
    _write(tmp_path, "momentum_tsfm.parquet",
           _frame(["tsfm_fwd_ret", "tsfm_uncert"]))
    _write(tmp_path, "momentum_kronos.parquet",
           _frame(["kronos_fwd_ret"]))

    out = latest_forecasts("momentum", cache_dir=tmp_path)
    assert out is not None and len(out) == 2
    out = out.set_index("symbol")
    # last row per symbol: j = n-1 = 29 -> value (i+1) + 0.029
    assert out.loc["AAA", "tsfm_fwd_ret"] == pytest.approx(1.029)
    assert out.loc["BBB", "tsfm_fwd_ret"] == pytest.approx(2.029)
    assert out.loc["AAA", "kronos_fwd_ret"] == pytest.approx(1.029)
    # ens = mean of the two available fwd_ret backends (>= 2 present)
    assert out.loc["BBB", "ens_fwd_ret"] == pytest.approx(
        (out.loc["BBB", "tsfm_fwd_ret"] + out.loc["BBB", "kronos_fwd_ret"]) / 2)
    # age = days since the cache's max date (fixed past dates -> positive)
    assert (out["forecast_age_days"] >= 0).all()
    assert out["forecast_age_days"].nunique() == 1


def test_swing_reads_all_three_files(tmp_path):
    _write(tmp_path, "momentum_tsfm.parquet", _frame(["tsfm_fwd_ret", "tsfm_uncert"]))
    _write(tmp_path, "momentum_kronos.parquet", _frame(["kronos_fwd_ret"]))
    _write(tmp_path, "swing_chronos.parquet", _frame(["chronos_fwd_ret", "chronos_uncert"]))

    out = latest_forecasts("swing", cache_dir=tmp_path)
    assert out is not None
    row = out.set_index("symbol").loc["AAA"]
    assert row["ens_fwd_ret"] == pytest.approx(
        (row["tsfm_fwd_ret"] + row["kronos_fwd_ret"] + row["chronos_fwd_ret"]) / 3)


def test_absent_cache_dir_returns_none(tmp_path):
    assert latest_forecasts("momentum", cache_dir=tmp_path / "nope") is None
    # dir exists but no parquets
    assert latest_forecasts("swing", cache_dir=tmp_path) is None


def test_history_starved_cache_is_skipped(tmp_path):
    """Poisoning guard (2026-07-06 incident shape): one symbol spans the whole
    cache but the MEDIAN symbol has only a few tail days -> file skipped."""
    dates_full = pd.bdate_range("2025-06-01", periods=200)
    rows = [{"date": d, "symbol": "AAA", "tsfm_fwd_ret": 0.01, "tsfm_uncert": 0.02}
            for d in dates_full]
    for s in ("BBB", "CCC"):  # history-starved: last 3 days only
        rows += [{"date": d, "symbol": s, "tsfm_fwd_ret": 0.03, "tsfm_uncert": 0.04}
                 for d in dates_full[-3:]]
    _write(tmp_path, "momentum_tsfm.parquet", pd.DataFrame(rows))

    # only the poisoned file present -> everything skipped -> None
    assert latest_forecasts("momentum", cache_dir=tmp_path) is None

    # a healthy kronos file alongside -> tsfm still skipped, kronos served,
    # ens NaN (only 1 backend present — mirror merge_forecast_features)
    _write(tmp_path, "momentum_kronos.parquet",
           _frame(["kronos_fwd_ret"], symbols=("AAA", "BBB", "CCC")))
    out = latest_forecasts("momentum", cache_dir=tmp_path)
    assert out is not None
    assert "tsfm_fwd_ret" not in out.columns
    assert out["kronos_fwd_ret"].notna().all()
    assert out["ens_fwd_ret"].isna().all()


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        latest_forecasts("intraday")


def test_engine_file_map_contract():
    assert ENGINE_CACHE_FILES["momentum"] == (
        "momentum_tsfm.parquet", "momentum_kronos.parquet")
    assert set(ENGINE_CACHE_FILES["swing"]) == set(ENGINE_CACHE_FILES["positional"])
    assert "swing_chronos.parquet" in ENGINE_CACHE_FILES["swing"]


def test_env_var_cache_dir(tmp_path, monkeypatch):
    _write(tmp_path, "momentum_tsfm.parquet", _frame(["tsfm_fwd_ret", "tsfm_uncert"]))
    _write(tmp_path, "momentum_kronos.parquet", _frame(["kronos_fwd_ret"]))
    monkeypatch.setenv("FORECAST_CACHE_DIR", str(tmp_path))
    out = latest_forecasts("momentum")
    assert out is not None and len(out) == 2
