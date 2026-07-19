from datetime import date
import pandas as pd
import pytest
from backend.data.providers.base import OHLCVRequest
from backend.data.providers.free_provider import FreeDataProvider


def _fake_cache(symbol, start, end):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100},
        index=idx,
    )


def test_returns_tidy_long_frame():
    p = FreeDataProvider(_loader=_fake_cache)
    df = p.get_ohlcv(OHLCVRequest(["RELIANCE", "TCS"], date(2020, 1, 1), date(2020, 1, 10)))
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert set(df["symbol"]) == {"RELIANCE", "TCS"}
    assert df.sort_values(["symbol", "date"]).equals(df)  # already sorted


def test_partial_symbol_success_skips_empty_and_returns_rest():
    # One symbol has data, one is empty (e.g. delisted) — provider must skip
    # the empty one (warning path) and return only the symbol with data.
    def _loader(symbol, start, end):
        if symbol == "DELISTED":
            return pd.DataFrame()
        return _fake_cache(symbol, start, end)

    p = FreeDataProvider(_loader=_loader)
    df = p.get_ohlcv(OHLCVRequest(["RELIANCE", "DELISTED"], date(2020, 1, 1), date(2020, 1, 10)))
    assert set(df["symbol"]) == {"RELIANCE"}
    assert len(df) == 5


def test_raises_when_all_symbols_empty():
    p = FreeDataProvider(_loader=lambda s, a, b: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no OHLCV"):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 10)))


def test_intraday_freq_rejected():
    p = FreeDataProvider(_loader=_fake_cache)
    with pytest.raises(NotImplementedError, match="TrueData"):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 10), freq="5min"))
