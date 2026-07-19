"""NIFTY (NSEI) benchmark loader for relative-strength features.

NSEI is an INDEX, not an equity ticker. Two sources, in order:
  1. the tracked offline cache ``data/cache/NSEI_10y.csv`` (tz-aware IST stamps),
     normalized to naive-midnight IST exactly like ``FreeDataProvider``;
  2. yfinance ``^NSEI`` (the index symbol — NOT ``NSEI.NS``, which is an equity
     ticker and 404s), used when the cache is absent (e.g. a fresh GPU pod where
     ``data/cache`` is gitignored). The index series is reliable from yfinance
     even when per-stock data is degraded, and its tz-naive daily dates line up
     with the equity panel's yfinance dates so the RS merge aligns.

Shared by the momentum trainer and the serving engine. Pure pandas + an optional
lazy yfinance import — no backend imports (import-linter: ml must not import
backend.api/services/platform).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: ml/data/benchmark.py -> parents[2] == repo root
_NSEI_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "NSEI_10y.csv"


def _from_cache(path: Path, start: date, end: date) -> Optional[pd.DataFrame]:
    """['date','close'] from the offline CSV, or None. Date normalization mirrors
    backend.data.providers.free_provider (utc -> Asia/Kolkata -> normalize ->
    tz-naive) so the merge key matches the equity panel exactly."""
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        lowered = {c.lower() for c in df.columns}
        if df.empty or "date" not in lowered or "close" not in lowered:
            return None
        df.columns = [c.lower() for c in df.columns]
        df["date"] = (
            pd.to_datetime(df["date"], utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        out = (
            df.loc[mask, ["date", "close"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return out if not out.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("NIFTY benchmark cache load failed (%s): %s", path, exc)
        return None


def _from_yfinance(start: date, end: date) -> Optional[pd.DataFrame]:
    """['date','close'] for ^NSEI from yfinance, or None. Tz-naive daily dates
    (matching the equity yfinance path) so the RS merge aligns."""
    try:
        import yfinance as yf  # noqa: PLC0415
        raw = yf.download("^NSEI", start=str(start), end=str(end),
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        # flatten any single-ticker MultiIndex columns to flat lowercase names
        raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
        if "close" not in raw.columns:
            return None
        out = raw.reset_index()
        date_col = "Date" if "Date" in out.columns else out.columns[0]
        out = out.rename(columns={date_col: "date"})
        out["date"] = pd.to_datetime(out["date"])
        if out["date"].dt.tz is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        out = out[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        return out if not out.empty else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("NIFTY benchmark yfinance (^NSEI) fallback failed: %s", exc)
        return None


def load_nifty_benchmark(
    start: date,
    end: date,
    cache_file: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Return ``['date', 'close']`` for NIFTY over ``[start, end]``, or ``None``.

    ``None`` (never an empty frame or a raise) signals "no benchmark" so callers
    treat RS features as fail-soft. Tries the offline cache first; for the
    DEFAULT path (no explicit ``cache_file``) it falls back to yfinance ``^NSEI``
    when the cache is absent — so a fresh pod is self-sufficient. An explicit
    ``cache_file`` is honored strictly (no network fallback) for reproducibility.
    """
    path = cache_file or _NSEI_CACHE
    out = _from_cache(path, start, end)
    if out is not None:
        return out
    if cache_file is None:
        return _from_yfinance(start, end)
    return None


__all__ = ["load_nifty_benchmark"]
