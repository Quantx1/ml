"""Pluggable market-data provider interface (spec 2026-06-15 §3.0).

FreeDataProvider ships now; TrueDataProvider drops in later behind the
same Protocol. Engines depend on this interface, never a concrete vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Protocol, runtime_checkable

import pandas as pd

#: Supported bar frequencies. Free provider supports the daily set;
#: TrueData adds the intraday set later.
FREQS = ("eod", "week", "month", "1min", "3min", "5min", "15min", "30min", "60min", "tick")


@dataclass
class OHLCVRequest:
    """A request for OHLCV history.

    symbols: NSE trading symbols (no suffix), e.g. ["RELIANCE", "TCS"].
    start/end: inclusive date bounds.
    freq: one of FREQS. Free provider supports eod/week/month only.
    """

    symbols: List[str]
    start: date
    end: date
    freq: str = "eod"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("OHLCVRequest.symbols must be non-empty")
        for sym in self.symbols:
            if not isinstance(sym, str) or not sym.strip():
                raise ValueError(f"symbol must be a non-empty string, got {sym!r}")
            if "." in sym:
                raise ValueError(
                    f"symbol {sym!r} must be a bare NSE symbol with no suffix "
                    f"(use 'RELIANCE', not 'RELIANCE.NS')"
                )
        if self.freq not in FREQS:
            raise ValueError(f"freq {self.freq!r} not in {FREQS}")
        if self.end < self.start:
            raise ValueError("end must be >= start")


@runtime_checkable
class DataProvider(Protocol):
    """Every provider returns a tidy long OHLCV frame.

    Columns (exact): ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
    One row per (symbol, bar). Sorted by ['symbol', 'date'].
    MUST raise on total failure — never return an empty frame silently
    (audit: production_ohlcv empty-frame masking).
    """

    name: str

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame: ...
