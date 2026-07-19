"""The forecast-cache read path: reuse persisted backfill, top-up only the tail."""
import numpy as np
import pandas as pd
import pytest

from datetime import date

import ml.features.forecast_features as ff
from ml.training.trainers.momentum_lambdarank import MomentumTrainer, MomentumConfig


def _panel(n=320, syms=("A", "B")):
    days = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(syms):
        close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
        for d, c in zip(days, close):
            rows.append({"date": d, "symbol": s, "open": c, "high": c + 1,
                         "low": c - 1, "close": c, "volume": 1000})
    return pd.DataFrame(rows)


def _fake_cache(dates, syms=("A", "B")):
    tsfm = pd.DataFrame([{"date": d, "symbol": s, "tsfm_fwd_ret": 0.01, "tsfm_uncert": 0.0}
                         for d in dates for s in syms])
    kron = pd.DataFrame([{"date": d, "symbol": s, "kronos_fwd_ret": 0.02}
                         for d in dates for s in syms])
    return tsfm, kron


def _trainer(monkeypatch, tmp_path, panel):
    monkeypatch.setenv("FORECAST_CACHE_DIR", str(tmp_path))
    cfg = MomentumConfig(with_forecasts=True, forecast_stride=5,
                         start=date(2024, 1, 1), end=date(2025, 12, 31))
    t = MomentumTrainer(cfg=cfg, symbols=["A", "B"])
    return t


def test_current_cache_skips_all_compute(monkeypatch, tmp_path):
    panel = _panel()
    rdates = ff._rebalance_dates(panel["date"].to_numpy(), 5)
    tsfm, kron = _fake_cache(rdates)                      # cache covers ALL dates
    tsfm.to_parquet(tmp_path / "momentum_tsfm.parquet", index=False)
    kron.to_parquet(tmp_path / "momentum_kronos.parquet", index=False)

    def _boom(*a, **k):
        raise AssertionError("forecaster should NOT run when cache is current")
    monkeypatch.setattr(ff, "timesfm_forecast_features", _boom)
    monkeypatch.setattr(ff, "kronos_forecast_features", _boom)

    t = _trainer(monkeypatch, tmp_path, panel)
    feats, cols = t.build_features(panel)
    assert "tsfm_fwd_ret" in cols and "kronos_fwd_ret" in cols and "ens_fwd_ret" in cols
    assert feats["tsfm_fwd_ret"].notna().any()


def test_stale_cache_tops_up_only_tail(monkeypatch, tmp_path):
    panel = _panel()
    rdates = ff._rebalance_dates(panel["date"].to_numpy(), 5)
    half = rdates[: len(rdates) // 2]                     # cache covers first half only
    tsfm, kron = _fake_cache(half)
    tsfm.to_parquet(tmp_path / "momentum_tsfm.parquet", index=False)
    kron.to_parquet(tmp_path / "momentum_kronos.parquet", index=False)
    cache_max = pd.Timestamp(pd.Series(half).max())

    calls = {}

    def _fake_tsfm(p, horizon, stride, min_date=None):
        calls["tsfm_min"] = min_date
        newd = ff._rebalance_dates(p["date"].to_numpy(), stride, min_date=min_date)
        return _fake_cache(newd)[0]

    def _fake_kronos(p, horizon, stride, min_date=None):
        calls["kron_min"] = min_date
        newd = ff._rebalance_dates(p["date"].to_numpy(), stride, min_date=min_date)
        return _fake_cache(newd)[1]

    monkeypatch.setattr(ff, "timesfm_forecast_features", _fake_tsfm)
    monkeypatch.setattr(ff, "kronos_forecast_features", _fake_kronos)

    t = _trainer(monkeypatch, tmp_path, panel)
    feats, cols = t.build_features(panel)

    # compute was called ONLY for the tail (min_date = cache max + 1 day)
    assert calls["tsfm_min"] == cache_max + pd.Timedelta(days=1)
    assert calls["kron_min"] == cache_max + pd.Timedelta(days=1)
    # combined cache now saved covering the full range
    saved = pd.read_parquet(tmp_path / "momentum_tsfm.parquet")
    assert pd.Timestamp(saved["date"].max()) == pd.Timestamp(rdates[-1])
    assert saved.duplicated(["date", "symbol"]).sum() == 0


def test_sparse_cache_is_treated_as_absent(monkeypatch, tmp_path):
    """POISONING GUARD: a cache covering <90% of panel symbols (e.g. a smoke
    leftover) must trigger a FULL recompute, not a max-date 'current' skip."""
    panel = _panel(syms=("A", "B"))
    rdates = ff._rebalance_dates(panel["date"].to_numpy(), 5)
    tsfm, kron = _fake_cache(rdates, syms=("A",))       # covers only 1 of 2 symbols
    tsfm.to_parquet(tmp_path / "momentum_tsfm.parquet", index=False)
    kron.to_parquet(tmp_path / "momentum_kronos.parquet", index=False)

    calls = {}

    def _fake_tsfm(p, horizon, stride, min_date=None):
        calls["tsfm_min"] = ("SET", min_date)
        return _fake_cache(ff._rebalance_dates(p["date"].to_numpy(), stride))[0]

    def _fake_kronos(p, horizon, stride, min_date=None):
        calls["kron_min"] = ("SET", min_date)
        return _fake_cache(ff._rebalance_dates(p["date"].to_numpy(), stride))[1]

    monkeypatch.setattr(ff, "timesfm_forecast_features", _fake_tsfm)
    monkeypatch.setattr(ff, "kronos_forecast_features", _fake_kronos)

    t = _trainer(monkeypatch, tmp_path, panel)
    t.build_features(panel)
    # full recompute (min_date=None), NOT a top-up and NOT a skip
    assert calls["tsfm_min"] == ("SET", None)
    assert calls["kron_min"] == ("SET", None)
