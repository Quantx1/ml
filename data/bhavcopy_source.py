"""
PR 179 — jugaad-data NSE bhavcopy as primary daily OHLCV source.

yfinance is convenient but its NSE.NS adjustments are imperfect: bonus
issues + rights issues sometimes show up as price gaps, volume isn't
adjusted in lock-step with price, and Yahoo's series is occasionally
late on corporate-action backfills. NSE bhavcopy (the daily settlement
file published by the exchange) is the authoritative source — used by
every Indian quant fund.

This module wraps `jugaad-data` to provide a fast, cached batch
download that mirrors the yfinance.download() shape, so trainers can
swap with one config flag.

Public surface:
    from ml.data.bhavcopy_source import bhavcopy_download

    df = bhavcopy_download(
        symbols=["RELIANCE", "TCS"],
        start="2018-01-01",
        end="2025-12-31",
    )
    # Returns: MultiIndex columns (symbol, OHLCV_field), date-indexed rows.

Failure modes:
    - jugaad-data unavailable / NSE rate-limit → raise BhavcopyError so
      the caller can fall back to yfinance. NEVER silently falls back —
      the caller must explicitly choose.
    - Missing days (NSE holidays, weekends): naturally absent from the
      output index.

References:
    https://github.com/jugaad-py/jugaad-data
    NSE Bhavcopy: https://www.nseindia.com/all-reports
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BHAVCOPY_PROGRESS_EVERY = int(os.environ.get("BHAVCOPY_PROGRESS_EVERY", "10"))
_BHAVCOPY_SYMBOL_TIMEOUT = float(os.environ.get("BHAVCOPY_SYMBOL_TIMEOUT", "60"))
_BHAVCOPY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bhavcopy")


# PR 190 — local parquet cache so 200 symbols × 8y don't pummel the
# NSE archive on every training run. Cache is keyed by symbol; per-call
# we slice the requested [start, end] window from the symbol parquet.
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "bhavcopy"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _symbol_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def _load_symbol_cache(symbol: str) -> Optional[pd.DataFrame]:
    p = _symbol_cache_path(symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.debug("bhavcopy cache read %s failed: %s", symbol, exc)
        return None


def _save_symbol_cache(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_symbol_cache_path(symbol))
    except Exception as exc:  # noqa: BLE001
        logger.debug("bhavcopy cache write %s failed: %s", symbol, exc)


class BhavcopyError(RuntimeError):
    """Raised when jugaad-data NSE bhavcopy fetch fails. Callers may
    catch and fall back to yfinance, but the failure must surface."""


def _to_date(d: str | Date | datetime | pd.Timestamp) -> Date:
    if isinstance(d, Date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def bhavcopy_download(
    symbols: Sequence[str],
    start: str | Date | datetime,
    end: str | Date | datetime,
    *,
    series: str = "EQ",
) -> pd.DataFrame:
    """Batch-download daily OHLCV for `symbols` from NSE bhavcopy.

    Args:
        symbols: NSE symbol codes WITHOUT the ".NS" suffix (e.g. "RELIANCE",
                 "TCS"). Internally jugaad-data uses the bare NSE symbol.
        start, end: ISO-8601 date strings or date/datetime/Timestamp.
                    Inclusive on both ends.
        series: NSE series filter. "EQ" = equity (default). Use "BE" for
                trade-to-trade settlement names.

    Returns:
        Multi-index column DataFrame matching yfinance.download(group_by=None)
        shape: outer level = OHLCV field {Open, High, Low, Close, Volume},
        inner level = symbol. Index = trading dates.

    Raises:
        BhavcopyError: jugaad-data import or NSE rate-limit failure.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        raise ValueError(f"end {end_d} before start {start_d}")

    try:
        from jugaad_data.nse import stock_df  # noqa: PLC0415
    except ImportError as exc:
        raise BhavcopyError(
            "jugaad-data not installed — pip install jugaad-data",
        ) from exc

    per_sym: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    n_total = len(symbols)
    n_cache_hit = 0
    n_fresh = 0
    t_loop_start = time.time()
    print(
        f"[bhavcopy] starting: {n_total} symbols, "
        f"window {start_d}..{end_d}, "
        f"symbol-timeout={_BHAVCOPY_SYMBOL_TIMEOUT:.0f}s",
        flush=True,
    )
    for i, sym in enumerate(symbols, start=1):
        # PR 190 — try local cache first. If it covers [start_d, end_d]
        # we skip the NSE round trip entirely. Otherwise fetch fresh +
        # upsert into the cache for next time.
        cached = _load_symbol_cache(sym)
        # Phase 1.7 audit fix #4.3 — tightened stale-cache window from
        # 2 days to 1 day. NSE bhavcopy is published T+0 evening, so
        # if today is Tuesday and we ask for through Monday, the cache
        # SHOULD contain Monday. The 2-day window quietly returned
        # cache missing the most recent trading day, biasing every
        # training run toward stale closes.
        cache_covers = (
            cached is not None
            and not cached.empty
            and cached.index.min().date() <= start_d
            and cached.index.max().date() >= end_d - timedelta(days=1)
        )

        def _sliced_cached() -> pd.DataFrame:
            """Cached frame trimmed to the [start_d, end_d] window."""
            if cached is None or cached.empty:
                return cached
            return cached.loc[
                (cached.index >= pd.Timestamp(start_d))
                & (cached.index <= pd.Timestamp(end_d))
            ]

        if cache_covers:
            per_sym[sym] = _sliced_cached()
            n_cache_hit += 1
            if i % _BHAVCOPY_PROGRESS_EVERY == 0 or i == n_total:
                print(
                    f"[bhavcopy] {i}/{n_total} "
                    f"(cache={n_cache_hit} fresh={n_fresh} fail={len(failed)}) "
                    f"elapsed={time.time()-t_loop_start:.1f}s",
                    flush=True,
                )
            continue

        t_sym = time.time()
        try:
            fut = _BHAVCOPY_EXECUTOR.submit(
                stock_df,
                symbol=sym, from_date=start_d, to_date=end_d, series=series,
            )
            df = fut.result(timeout=_BHAVCOPY_SYMBOL_TIMEOUT)
        except _FutureTimeout:
            print(
                f"[bhavcopy] TIMEOUT {sym} after {_BHAVCOPY_SYMBOL_TIMEOUT:.0f}s — skipping",
                flush=True,
            )
            # Phase 1.7 audit fix #4.3 — fall back to the SLICED cache so
            # we don't return rows outside the requested window. The
            # legacy fallback returned the entire cache file, polluting
            # downstream slicing with bars from prior training runs.
            if cached is not None and not cached.empty:
                per_sym[sym] = _sliced_cached()
                continue
            failed.append(sym)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("bhavcopy %s failed: %s", sym, exc)
            if cached is not None and not cached.empty:
                per_sym[sym] = _sliced_cached()
                continue
            failed.append(sym)
            continue
        if df is None or df.empty:
            if cached is not None and not cached.empty:
                per_sym[sym] = _sliced_cached()
                continue
            failed.append(sym)
            continue
        n_fresh += 1
        if i % _BHAVCOPY_PROGRESS_EVERY == 0 or i == n_total:
            print(
                f"[bhavcopy] {i}/{n_total} "
                f"(cache={n_cache_hit} fresh={n_fresh} fail={len(failed)}) "
                f"last={sym} {time.time()-t_sym:.1f}s "
                f"elapsed={time.time()-t_loop_start:.1f}s",
                flush=True,
            )
        # jugaad-data returns columns including DATE, OPEN, HIGH, LOW,
        # CLOSE, LTP, VOLUME, TURNOVER, SERIES. Normalize to yfinance
        # column names so downstream code treats the two interchangeably.
        df = df.rename(columns={
            "DATE": "Date",
            "OPEN": "Open",
            "HIGH": "High",
            "LOW": "Low",
            "CLOSE": "Close",
            "VOLUME": "Volume",
            "PREV. CLOSE": "Adj Close",
        })
        df = df.set_index(pd.to_datetime(df["Date"])).sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        # PR 190 — merge with any prior cache + persist.
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            df = merged
        _save_symbol_cache(sym, df)
        per_sym[sym] = df

    if not per_sym:
        raise BhavcopyError(
            f"bhavcopy returned no data for {len(symbols)} symbols "
            f"({len(failed)} failed)",
        )

    if failed:
        logger.warning(
            "bhavcopy: %d/%d symbols had no data: %s",
            len(failed), len(symbols), failed[:10],
        )

    # Build the multi-index column frame: outer=field, inner=symbol.
    # PR fix 2026-05-11: NSE bhavcopy + yfinance fallback occasionally
    # return the same date twice (cache merge + fresh fetch overlap, or
    # NSE archive duplicate row). pd.DataFrame({sym: series, ...})
    # internally reindexes and fails on duplicate-label axes. Dedupe
    # per-symbol up front so the multi-index assembly is robust.
    fields = ["Open", "High", "Low", "Close", "Volume"]
    for sym in list(per_sym.keys()):
        df = per_sym[sym]
        if df.index.duplicated().any():
            per_sym[sym] = df[~df.index.duplicated(keep="last")].sort_index()

    blocks = {}
    for field in fields:
        blocks[field] = pd.DataFrame(
            {sym: df[field] for sym, df in per_sym.items()},
        )
    out = pd.concat(blocks, axis=1)
    out.columns.names = [None, None]   # match yfinance shape
    return out.sort_index()


def bhavcopy_download_with_fallback(
    symbols: Sequence[str],
    start: str | Date | datetime,
    end: str | Date | datetime,
    *,
    yfinance_kwargs: Optional[dict] = None,
) -> tuple[pd.DataFrame, str]:
    """Try bhavcopy first; fall back to yfinance on failure.

    Returns:
        (frame, source) where source ∈ {"bhavcopy", "yfinance"} so
        trainers can record which source was used in metrics.
    """
    # Catch BROAD Exception (not just BhavcopyError) so any bhavcopy
    # failure — including ValueError from duplicate-index issues or
    # KeyError from malformed jugaad-data responses — falls through to
    # yfinance. The previous narrow catch was the bug that killed the
    # 2026-05-11 smoke run when bhavcopy raised ValueError instead of
    # BhavcopyError.
    try:
        df = bhavcopy_download(symbols, start, end)
        return df, "bhavcopy"
    except Exception as exc:  # noqa: BLE001 — yfinance fallback is the point
        logger.warning("bhavcopy failed (%s); falling back to yfinance", exc)

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise BhavcopyError(
            "neither jugaad-data nor yfinance available",
        ) from exc

    tickers = [f"{s}.NS" for s in symbols]
    yf_kwargs = {"progress": False, "auto_adjust": True, **(yfinance_kwargs or {})}
    df = yf.download(tickers, start=str(start), end=str(end), **yf_kwargs)
    if df is None or df.empty:
        raise BhavcopyError("yfinance fallback returned empty frame")
    # yfinance occasionally returns a date twice when sessions overlap with
    # cached data. Same fix as bhavcopy_download path — dedupe before
    # callers reindex.
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df, "yfinance"


__all__ = [
    "BhavcopyError",
    "bhavcopy_download",
    "bhavcopy_download_with_fallback",
]
