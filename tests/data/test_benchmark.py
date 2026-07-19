from datetime import date
from pathlib import Path

import pandas as pd

from ml.data.benchmark import load_nifty_benchmark


def test_loads_nifty_from_cache_with_naive_dates():
    out = load_nifty_benchmark(date(2021, 1, 1), date(2021, 12, 31))
    assert out is not None and not out.empty
    assert list(out.columns) == ["date", "close"]
    # date normalized to tz-naive midnight (so it merges with the equity panel)
    assert out["date"].dt.tz is None
    assert (out["date"] == out["date"].dt.normalize()).all()
    assert out["close"].gt(0).all()
    assert out["date"].is_monotonic_increasing


def test_missing_file_returns_none(tmp_path: Path):
    out = load_nifty_benchmark(date(2021, 1, 1), date(2021, 12, 31),
                               cache_file=tmp_path / "nope.csv")
    assert out is None


def test_yfinance_fallback_when_default_cache_missing(monkeypatch, tmp_path: Path):
    import sys
    import types

    import numpy as np

    idx = pd.date_range("2021-01-04", periods=10, freq="B")  # tz-naive, like yfinance daily
    idx.name = "Date"
    fake_df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0,
         "Close": np.arange(100.0, 110.0), "Volume": 0},
        index=idx,
    )
    fake_yf = types.SimpleNamespace(download=lambda *a, **k: fake_df)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    from ml.data import benchmark as bm
    monkeypatch.setattr(bm, "_NSEI_CACHE", tmp_path / "absent.csv")  # force cache miss on default path
    out = bm.load_nifty_benchmark(date(2021, 1, 1), date(2021, 12, 31))
    assert out is not None and list(out.columns) == ["date", "close"]
    assert out["date"].dt.tz is None and out["close"].gt(0).all()
    assert out["date"].is_monotonic_increasing


def test_range_filter_excludes_out_of_window(tmp_path: Path):
    csv = tmp_path / "NSEI_10y.csv"
    csv.write_text(
        "date,open,high,low,close,volume\n"
        "2020-01-01 00:00:00+05:30,100,101,99,100,10\n"
        "2021-06-01 00:00:00+05:30,200,201,199,200,10\n"
        "2026-01-01 00:00:00+05:30,300,301,299,300,10\n"
    )
    out = load_nifty_benchmark(date(2021, 1, 1), date(2021, 12, 31), cache_file=csv)
    assert out is not None
    assert len(out) == 1
    assert abs(out["close"].iloc[0] - 200) < 1e-9
