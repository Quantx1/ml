"""Free-data provider: pg candle cache → yfinance.

Daily/weekly/monthly only. Intraday + options require TrueData (later).
Fail-loud: raises if EVERY requested symbol comes back empty (audit fix).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Optional

import pandas as pd

from .base import OHLCVRequest

logger = logging.getLogger(__name__)

_DAILY_FREQS = {"eod": "1d", "week": "1wk", "month": "1mo"}


def _default_loader(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Load one symbol's daily OHLCV. Tries pg candles first, then yfinance.

    Returns a DatetimeIndex frame with columns open/high/low/close/volume,
    or an empty frame if nothing is available for this symbol. If the pg
    source returns data missing any required column, it is treated as a
    miss and we fall through to yfinance (never return a partial frame —
    that would crash the tidy-frame assembly downstream).

    Note: production_ohlcv expects a Sequence[str] of symbols and returns a
    MultiIndex column DataFrame (group_by='ticker'). We pass [symbol] and
    extract the single-symbol slice.
    """
    _REQUIRED = ["open", "high", "low", "close", "volume"]
    # 0) local CSV cache (offline-first: data/cache/{SYMBOL}_NS_10y.csv).
    # Lets trainers run with zero network on the cached universe. Files carry
    # a tz-aware 'date' column + open/high/low/close/volume.
    try:
        from pathlib import Path  # noqa: PLC0415
        cache_dir = Path(__file__).resolve().parents[3] / "data" / "cache"
        cache_file = cache_dir / f"{symbol}_NS_10y.csv"
        if cache_file.exists():
            cdf = pd.read_csv(cache_file)
            if not cdf.empty and set(_REQUIRED).issubset(c.lower() for c in cdf.columns):
                cdf.columns = [c.lower() for c in cdf.columns]
                # Preserve the IST trading date (cache stamps are tz-aware IST);
                # normalize to naive midnight so cross-sectional groupby('date')
                # aligns every symbol on the same trading day.
                cdf["date"] = (
                    pd.to_datetime(cdf["date"], utc=True)
                    .dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
                )
                cdf = cdf.set_index("date").sort_index()
                mask = (cdf.index >= pd.Timestamp(start)) & (cdf.index <= pd.Timestamp(end))
                window = cdf.loc[mask, _REQUIRED]
                if not window.empty:
                    return window
    except Exception as e:  # noqa: BLE001
        logger.debug("cache CSV miss for %s: %s", symbol, e)
    # 1) pg candle cache (authoritative; corp-action adjusted)
    try:
        from ml.data.production_ohlcv import production_ohlcv  # noqa: PLC0415
        raw = production_ohlcv([symbol], start=start, end=end)
        if raw is not None and not raw.empty:
            # MultiIndex columns: (field, symbol_with_suffix) or flat if single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                # Extract single-symbol slice → flat columns
                ticker_keys = raw.columns.get_level_values(1).unique()
                key = ticker_keys[0]  # e.g. "RELIANCE.NS"
                df = raw.xs(key, axis=1, level=1)
            else:
                df = raw
            df = df.rename(columns=str.lower)
            if set(_REQUIRED).issubset(df.columns):
                return df[_REQUIRED]
            logger.debug(
                "pg candles for %s missing columns %s — falling through to yfinance",
                symbol, set(_REQUIRED) - set(df.columns),
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("pg candles miss for %s: %s", symbol, e)
    # 2) yfinance fallback
    try:
        import yfinance as yf  # noqa: PLC0415
        raw = yf.download(f"{symbol}.NS", start=str(start), end=str(end),
                          progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
            return raw[["open", "high", "low", "close", "volume"]]
    except Exception as e:  # noqa: BLE001
        logger.debug("yfinance miss for %s: %s", symbol, e)
    return pd.DataFrame()


class FreeDataProvider:
    """DataProvider over free sources. Satisfies the base.DataProvider Protocol."""

    name = "free"

    def __init__(self, _loader: Optional[Callable[[str, date, date], pd.DataFrame]] = None):
        # _loader injectable for tests.
        self._loader = _loader or _default_loader

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        if req.freq not in _DAILY_FREQS:
            raise NotImplementedError(
                f"FreeDataProvider supports {sorted(_DAILY_FREQS)} only; "
                f"freq={req.freq!r} needs TrueData (enable DATA_PROVIDER=truedata)"
            )
        frames = []
        for sym in req.symbols:
            one = self._loader(sym, req.start, req.end)
            if one is None or one.empty:
                logger.warning("FreeDataProvider: no data for %s", sym)
                continue
            one = one.copy()
            one.index.name = "date"
            one = one.reset_index()
            one["symbol"] = sym
            frames.append(one)
        if not frames:
            raise RuntimeError(
                f"FreeDataProvider returned no OHLCV for any of {len(req.symbols)} "
                f"symbols ({req.start}..{req.end}) — check data source, not masking empty"
            )
        out = pd.concat(frames, ignore_index=True)
        out = out[["date", "symbol", "open", "high", "low", "close", "volume"]]
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)
