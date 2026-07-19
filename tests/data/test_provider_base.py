from datetime import date

import pytest

from backend.data.providers.base import OHLCVRequest, DataProvider


def test_ohlcv_request_defaults_and_validation():
    req = OHLCVRequest(symbols=["RELIANCE"], start=date(2020, 1, 1), end=date(2021, 1, 1))
    assert req.freq == "eod"
    assert req.symbols == ["RELIANCE"]


def test_ohlcv_request_rejects_empty_symbols():
    with pytest.raises(ValueError):
        OHLCVRequest(symbols=[], start=date(2020, 1, 1), end=date(2021, 1, 1))


def test_ohlcv_request_rejects_end_before_start():
    with pytest.raises(ValueError):
        OHLCVRequest(symbols=["RELIANCE"], start=date(2021, 1, 1), end=date(2020, 1, 1))


def test_ohlcv_request_rejects_suffixed_symbol():
    with pytest.raises(ValueError):
        OHLCVRequest(symbols=["RELIANCE.NS"], start=date(2020, 1, 1), end=date(2021, 1, 1))


def test_dataprovider_is_protocol():
    class Dummy:
        name = "dummy"
        def get_ohlcv(self, req): ...
    assert isinstance(Dummy(), DataProvider)
