"""Single OHLCV entry point for trainers + serving engines (spec §3.5).

Provider chosen by settings.DATA_PROVIDER ("free" | "truedata"). Guarantees
the same tidy long frame regardless of backend, so train==serve data shape.
"""
from __future__ import annotations

import os
from datetime import date
from typing import List, Optional

import pandas as pd

from ml._vendor.backend.data.providers.base import DataProvider, OHLCVRequest
from ml._vendor.backend.data.providers.free_provider import FreeDataProvider


def get_provider() -> DataProvider:
    """Return the configured provider. Defaults to free."""
    name = os.environ.get("DATA_PROVIDER", "free").strip().lower()
    if name == "free":
        return FreeDataProvider()
    if name == "truedata":
        # Lazy import — only when explicitly enabled (creds required).
        from backend.data.providers.truedata_provider import TrueDataProvider  # noqa: PLC0415
        return TrueDataProvider()
    raise ValueError(f"unknown DATA_PROVIDER={name!r} (expected 'free' or 'truedata')")


def load_ohlcv(
    symbols: List[str],
    start: date,
    end: date,
    freq: str = "eod",
    provider: Optional[DataProvider] = None,
) -> pd.DataFrame:
    """Load OHLCV for symbols over [start, end]. See base.DataProvider for schema."""
    prov = provider or get_provider()
    return prov.get_ohlcv(OHLCVRequest(symbols=symbols, start=start, end=end, freq=freq))
