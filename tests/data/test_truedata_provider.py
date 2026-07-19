from datetime import date

import pandas as pd
import pytest

from backend.data.providers.base import OHLCVRequest
from backend.data.providers.truedata_provider import TrueDataProvider


class _FakeHist:
    """Mimics truedata.TD_hist.get_historic_data output (timestamp + capital
    'Volume' for intraday + an 'oi' column we must drop)."""

    def __init__(self, empty_for=()):
        self.empty_for = set(empty_for)

    def get_historic_data(self, contract, start_time=None, end_time=None,
                          bar_size="1 min", bidask=False, delivery=False):
        if contract in self.empty_for:
            return pd.DataFrame()
        idx = pd.to_datetime(["2026-06-15 09:15:00", "2026-06-15 09:20:00"])
        return pd.DataFrame({
            "timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.5,
            "close": 100.5, "Volume": 1000, "oi": 0,
        })


def test_normalizes_to_tidy_long_frame():
    p = TrueDataProvider(_hist=_FakeHist())
    df = p.get_ohlcv(OHLCVRequest(["RELIANCE", "TCS"], date(2026, 6, 1), date(2026, 6, 16), freq="5min"))
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert set(df["symbol"]) == {"RELIANCE", "TCS"}
    assert "oi" not in df.columns  # extra cols dropped


def test_raises_when_all_empty():
    p = TrueDataProvider(_hist=_FakeHist(empty_for=("RELIANCE",)))
    with pytest.raises(RuntimeError, match="no OHLCV"):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2026, 6, 1), date(2026, 6, 16), freq="eod"))


def test_tick_rejected():
    p = TrueDataProvider(_hist=_FakeHist())
    with pytest.raises(NotImplementedError):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2026, 6, 1), date(2026, 6, 16), freq="tick"))


def test_requires_creds_or_client(monkeypatch):
    monkeypatch.delenv("TRUEDATA_LOGIN", raising=False)
    monkeypatch.delenv("TRUEDATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="TRUEDATA_LOGIN"):
        TrueDataProvider(login=None, password=None)
