import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.ai.signals.engines.momentum import MomentumEngine
from backend.ai.signals.style_types import MomentumSignal
from ml.data.benchmark import load_nifty_benchmark
from ml.features.momentum_features import MOMENTUM_FEATURE_ORDER


class _FakeBooster:
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        return arr.mean(axis=1)


class _NanMeanBooster:
    """Mean that tolerates NaN forecast cols (like real LightGBM at predict)."""
    def predict(self, X):
        return np.nanmean(np.asarray(X, dtype=float), axis=1)


def _trading_dates(n: int = 400) -> pd.DatetimeIndex:
    """Last ``n`` real NSE trading dates from the local NIFTY cache.

    MomentumEngine loads its RS benchmark via load_nifty_benchmark (the real
    local cache), NOT the injected OHLCV loader — so the fake panel must sit
    on actual trading dates for the RS merge to align. Synthetic freq="B"
    dates include NSE holidays, which NaN out the beta/corr windows and drop
    every row.
    """
    bench = load_nifty_benchmark(date.today() - timedelta(days=3 * 365), date.today())
    assert bench is not None and len(bench) >= n, \
        "local NSEI cache (data/cache/NSEI_10y.csv) required for this test"
    return pd.DatetimeIndex(bench["date"].tail(n).reset_index(drop=True))


def _ramp_ohlcv(sym_offset: float, idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    base = 100.0 + sym_offset
    close = base + np.linspace(0, 40 + sym_offset, n)
    return pd.DataFrame({
        "date": idx, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1_000_000,
    })


def _fake_loader_for(offsets: dict, idx: pd.DatetimeIndex):
    def fake_loader(symbols, start, end, freq="eod", provider=None):
        frames = []
        for s in symbols:
            df = _ramp_ohlcv(offsets.get(s, 3.0), idx); df["symbol"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    return fake_loader


def test_ranks_and_levels_and_outputs():
    syms = ["AAA", "BBB", "CCC"]
    offsets = {"AAA": 0.0, "BBB": 5.0, "CCC": 10.0}
    idx = _trading_dates(400)

    eng = MomentumEngine(_booster=_FakeBooster(), _feature_order=None,
                         _universe=lambda limit=None: syms,
                         _load_ohlcv=_fake_loader_for(offsets, idx))
    sigs = eng.run(top_n=3)
    assert len(sigs) == 3
    assert all(isinstance(s, MomentumSignal) for s in sigs)
    ranks = [s.rank for s in sigs]
    assert sorted(ranks) == [1, 2, 3]
    assert sigs[0].rank == 1 and sigs[0].percentile == 1.0
    for s in sigs:
        assert s.direction == "BUY"
        assert s.stop_loss < s.entry_price < s.target
        assert s.risk_reward > 0
        assert -1.0 <= s.expected_return <= 1.0
        assert 0.0 <= s.top_decile_prob <= 1.0


def test_xs_rank_features_are_cross_sectional_not_degenerate():
    """Regression: features must be built on the full panel so xs_rank_ret_*
    are real cross-sectional ranks (not a constant 1.0 from per-symbol build)."""
    syms = ["AAA", "BBB", "CCC"]
    offsets = {"AAA": 0.0, "BBB": 5.0, "CCC": 10.0}
    idx = _trading_dates(400)

    class _CapturingBooster:
        def __init__(self):
            self.seen = []
        def predict(self, X):
            self.seen.append(X.copy())
            return np.asarray(X, dtype=float).mean(axis=1)

    cap = _CapturingBooster()
    eng = MomentumEngine(_booster=cap, _feature_order=None,
                         _universe=lambda limit=None: syms,
                         _load_ohlcv=_fake_loader_for(offsets, idx))
    eng.run(top_n=3)
    xs_vals = [df["xs_rank_ret_21"].iloc[0] for df in cap.seen]
    assert len(xs_vals) >= 2
    assert len({round(v, 6) for v in xs_vals}) > 1, \
        "xs_rank_ret_21 is degenerate across symbols — features built per-symbol, not cross-sectionally"


def test_honest_empty_when_model_missing():
    def boom(*a, **k):
        raise LookupError("no prod version")
    eng = MomentumEngine(_model_loader=boom, _universe=lambda limit=None: ["AAA"],
                         _load_ohlcv=lambda *a, **k: pd.DataFrame())
    sigs = eng.run()
    assert sigs == []
    assert eng.status == "model_not_loaded"


# ── Phase 2: forecast-cache merge at serving ────────────────────────────────

_FORECAST_ORDER = list(MOMENTUM_FEATURE_ORDER) + [
    "tsfm_fwd_ret", "tsfm_uncert", "kronos_fwd_ret", "ens_fwd_ret",
]


def _write_forecast_cache(cache_dir, symbols):
    """Synthetic weekly cache parquets — value encodes (symbol, day) so the
    tests can assert the LAST row per symbol got merged."""
    dates = pd.bdate_range("2026-05-01", periods=10)

    def frame(cols):
        rows = []
        for i, s in enumerate(symbols):
            for j, d in enumerate(dates):
                r = {"date": d, "symbol": s}
                for c in cols:
                    r[c] = (i + 1) * 1.0 + j * 0.001
                rows.append(r)
        return pd.DataFrame(rows)

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame(["tsfm_fwd_ret", "tsfm_uncert"]).to_parquet(
        cache_dir / "momentum_tsfm.parquet", index=False)
    frame(["kronos_fwd_ret"]).to_parquet(
        cache_dir / "momentum_kronos.parquet", index=False)


class _CapturingNanBooster:
    def __init__(self):
        self.seen = []
    def predict(self, X):
        self.seen.append(X.copy())
        return np.nanmean(np.asarray(X, dtype=float), axis=1)


def test_forecast_cols_merged_from_cache(tmp_path, monkeypatch):
    """Artifact feature_order includes forecast cols + a fresh cache dir ->
    run() ranks signals with the latest cached forecast values merged."""
    monkeypatch.setenv("FORECAST_CACHE_DIR", str(tmp_path))
    syms = ["AAA", "BBB", "CCC"]
    _write_forecast_cache(tmp_path, syms)
    idx = _trading_dates(400)

    cap = _CapturingNanBooster()
    eng = MomentumEngine(_booster=cap, _feature_order=list(_FORECAST_ORDER),
                         _universe=lambda limit=None: syms,
                         _load_ohlcv=_fake_loader_for(
                             {"AAA": 0.0, "BBB": 5.0, "CCC": 10.0}, idx))
    sigs = eng.run(top_n=3)
    assert len(sigs) == 3
    assert eng.status == "ok"
    assert eng.forecast_degraded is False
    # every scored row carries the LAST cached forecast row per symbol
    # (last day j=9 -> value (i+1) + 0.009)
    tsfm_seen = sorted(round(float(df["tsfm_fwd_ret"].iloc[0]), 6) for df in cap.seen)
    assert tsfm_seen == [1.009, 2.009, 3.009]
    for df in cap.seen:
        row = df.iloc[0]
        assert row["ens_fwd_ret"] == pytest.approx(
            (row["tsfm_fwd_ret"] + row["kronos_fwd_ret"]) / 2)


def test_degraded_but_serving_when_cache_absent(tmp_path, monkeypatch, caplog):
    """No cache -> forecast cols NaN, loud warning, degraded=True — but the
    engine still returns ranked signals (status stays ok)."""
    monkeypatch.setenv("FORECAST_CACHE_DIR", str(tmp_path / "empty"))
    syms = ["AAA", "BBB", "CCC"]
    idx = _trading_dates(400)

    cap = _CapturingNanBooster()
    eng = MomentumEngine(_booster=cap, _feature_order=list(_FORECAST_ORDER),
                         _universe=lambda limit=None: syms,
                         _load_ohlcv=_fake_loader_for(
                             {"AAA": 0.0, "BBB": 5.0, "CCC": 10.0}, idx))
    with caplog.at_level(logging.WARNING):
        sigs = eng.run(top_n=3)
    assert len(sigs) == 3
    assert eng.status == "ok"
    assert eng.forecast_degraded is True
    assert any("degraded" in r.message for r in caplog.records)
    for df in cap.seen:
        assert df[["tsfm_fwd_ret", "tsfm_uncert", "kronos_fwd_ret", "ens_fwd_ret"]] \
            .isna().all().all()
