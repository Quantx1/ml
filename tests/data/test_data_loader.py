from datetime import date
import pandas as pd
from ml.data.data_loader import load_ohlcv, get_provider


def test_get_provider_defaults_to_free(monkeypatch):
    monkeypatch.delenv("DATA_PROVIDER", raising=False)
    assert get_provider().name == "free"


def test_load_ohlcv_uses_injected_provider():
    class Stub:
        name = "stub"
        def get_ohlcv(self, req):
            return pd.DataFrame({
                "date": pd.to_datetime(["2020-01-01"]), "symbol": ["RELIANCE"],
                "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10],
            })
    df = load_ohlcv(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 2), provider=Stub())
    assert len(df) == 1 and df.iloc[0]["symbol"] == "RELIANCE"
