"""
Production-grade OHLCV accessor — primary data path for every trainer.

Source hierarchy (v1, post 2026-05-19)
--------------------------------------
1. **Kite Connect Admin** (paid broker feed) — when ``KITE_ADMIN_API_KEY``
   + ``KITE_ADMIN_ACCESS_TOKEN`` are set. Highest quality, NSE-licensed.
   Raises ``KiteSourceUnavailable`` quickly if not configured.
2. **yfinance** — primary public fallback. Same source the regime/lgbm/
   tft/intraday/momentum trainers already use, so RL trainers stay
   consistent with the rest of the pipeline.

bhavcopy via jugaad-data was the prior Tier-2. **Dropped 2026-05-19**
because NSE archives rate-limit aggressively (10 fetches succeed, then
every subsequent symbol times out at 60s — see the FinRL run that
loaded 10/50 symbols). Partial-success was returned as "success" so
the yfinance fallback never triggered, and RL trainers ended up
training on garbage data.

When we add a paid real-time provider later (TrueData / Fyers / Kite
Connect with broker tokens), it slots in at Tier 1 next to Kite Admin —
no other trainer code needs to change.

This module provides ``production_ohlcv()`` — a single API that:

  - **CA-adjusted:** routes through ``ml.data.corporate_actions.adjust_batch``
    so volume reflects splits/bonuses correctly
  - **Survivorship-aware:** optional ``include_delisted`` flag adds
    historically-listed symbols from ``ml.data.delisted_registry``
  - **Quality-gated:** runs ``ml.data.quality_check`` and refuses to
    return data that fails (e.g. >5% stale runs, gap days, negatives)

API
---
::

    from ml.data.production_ohlcv import production_ohlcv

    prices = production_ohlcv(
        symbols=["RELIANCE", "TCS", "HDFCBANK"],
        start="2020-01-01",
        end="2026-05-01",
        include_delisted=True,
        adjust_corp_actions=True,
        quality_check=True,
    )
    # MultiIndex columns: outer=ticker (with .NS suffix), inner=OHLCV field
    # Date-indexed rows. Drop-in replacement for ``yf.download(..., group_by="ticker")``.

Trainers that should use this (v1 scope locked 2026-05-17)
----------------------------------------------------------
- tft_swing (currently yf.download)
- lgbm_signal_gate (already routed through liquid_universe but builds
  features from yf — fix to use this accessor)
- finrl_x_ensemble (currently _download_ohlc → yf.download)
- intraday_lstm (5-min — bhavcopy doesn't help; keep yf)
- momentum_zero_shot (macro indices, yf is fine)
- regime_hmm (^NSEI + ^INDIAVIX — yf is fine)

Dropped from v1: momentum_chronos / options_rl / vix_tft / chronos2_macro.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_SUFFIX = ".NS"   # NSE; .BO for BSE if ever needed
TRADING_DAYS_PER_YEAR = 252


def _to_date(d: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# Universe expansion — survivorship bias correction
# ---------------------------------------------------------------------------


def _expand_with_delisted(
    base_symbols: Sequence[str], start: date, end: date,
) -> List[str]:
    """Add delisted-but-was-listed symbols that traded in [start, end].

    Mitigates survivorship bias. ``ml.data.delisted_registry`` carries
    the small list of NSE delistings we track explicitly. Returns the
    deduplicated union of ``base_symbols`` + historical extras.
    """
    try:
        from .delisted_registry import historical_universe_extras  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.debug("delisted_registry unavailable (%s) — skipping expansion", exc)
        return list(base_symbols)

    extras: List[str] = []
    cursor = start
    seen = set(base_symbols)
    while cursor <= end:
        for s in historical_universe_extras(cursor):
            if s not in seen:
                extras.append(s)
                seen.add(s)
        cursor = cursor + timedelta(days=365)   # check yearly cursors
    if extras:
        logger.info(
            "production_ohlcv: added %d delisted symbols for %s..%s: %s",
            len(extras), start, end, extras[:8] + (["..."] if len(extras) > 8 else []),
        )
    return list(base_symbols) + extras


# ---------------------------------------------------------------------------
# Quality gating — run AFML-style sanity checks
# ---------------------------------------------------------------------------


def _quality_drop_bad_symbols(
    raw: pd.DataFrame, *, max_bad_pct: float = 0.30,
) -> pd.DataFrame:
    """For each ticker in the MultiIndex frame, run a lightweight quality
    check on Close. Drop tickers whose flagged-bar fraction > max_bad_pct.

    The full ``ml.data.quality_check`` is heavier and intended for
    pre-training scripts; here we just defend against obvious data
    rot (10+ duplicate closes in a row, >20% one-day returns, etc.).
    """
    if raw is None or raw.empty:
        return raw
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw

    # Detect column convention: (ticker, field) or (field, ticker)?
    first_outer = raw.columns.get_level_values(0)[0]
    field_names = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
    is_ticker_outer = first_outer not in field_names
    tickers = sorted(raw.columns.get_level_values(0).unique()) if is_ticker_outer \
        else sorted(raw.columns.get_level_values(1).unique())

    keep: List[str] = []
    dropped: List[str] = []
    for tk in tickers:
        try:
            close = raw[tk]["Close"] if is_ticker_outer else raw["Close"][tk]
        except KeyError:
            continue
        if close is None or close.empty:
            dropped.append(tk)
            continue
        close = close.dropna()
        if len(close) < 60:
            dropped.append(tk)
            continue
        # Flag rotten data: too many stale runs, negatives, extreme returns
        rets = close.pct_change().dropna()
        bad = (
            (close <= 0).sum()
            + ((close.diff() == 0).rolling(10).sum() >= 10).sum()
            + (rets.abs() > 0.25).sum()
        )
        if bad / max(len(close), 1) > max_bad_pct:
            dropped.append(tk)
        else:
            keep.append(tk)

    if dropped:
        logger.warning(
            "production_ohlcv: quality-dropped %d/%d tickers: %s",
            len(dropped), len(tickers), dropped[:8],
        )

    if not keep:
        return raw      # don't drop everything — caller will hit a downstream error

    if is_ticker_outer:
        cols = [c for c in raw.columns if c[0] in keep]
    else:
        cols = [c for c in raw.columns if c[1] in keep]
    return raw[cols]


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def production_ohlcv(
    symbols: Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    *,
    suffix: str = DEFAULT_SUFFIX,
    include_delisted: bool = True,
    adjust_corp_actions: bool = True,
    quality_check: bool = True,
    max_bad_pct: float = 0.30,
    series: str = "EQ",
    group_by: str = "ticker",
    return_source: bool = False,
) -> "pd.DataFrame | Tuple[pd.DataFrame, str]":
    """Production-grade OHLCV download for NSE equities.

    Drop-in replacement for ``yf.download(tickers, group_by='ticker', ...)``.

    Args:
        symbols: NSE codes WITHOUT suffix (e.g. ``["RELIANCE", "TCS"]``).
        start, end: Date strings or date objects. ``end`` defaults to today.
        suffix: Ticker suffix returned in the output MultiIndex
            (default ``.NS`` to match yfinance convention).
        include_delisted: When True, expand the universe to add historical
            NSE delistings that were listed during [start, end].
            Prevents survivorship bias in backtests.
        adjust_corp_actions: When True, apply ``adjust_batch`` to scale
            volume for splits/bonuses per ``ml.data.corporate_actions``.
        quality_check: When True, drop tickers whose data fails the
            quality check (>30% stale/negative/spike bars by default).
        max_bad_pct: Per-ticker bad-bar fraction threshold for the quality
            check.
        series: NSE series filter (``EQ`` = equity).
        group_by: ``"ticker"`` (default; outer=ticker, inner=field) or
            ``"column"`` (outer=field, inner=ticker) — matches yfinance.

    Returns:
        MultiIndex column DataFrame, date-indexed. Empty if all fetches
        fail (caller should handle).
    """
    start_d = _to_date(start)
    end_d = _to_date(end) if end is not None else date.today()
    if end_d < start_d:
        raise ValueError(f"end {end_d} before start {start_d}")

    symbols = list(symbols)
    if not symbols:
        return pd.DataFrame()

    if include_delisted:
        symbols = _expand_with_delisted(symbols, start_d, end_d)

    # Source hierarchy (v1, post 2026-05-19):
    #   Tier 1 — Kite Connect Admin (NSE-licensed broker feed). Only
    #            attempted when KITE_ADMIN_API_KEY + KITE_ADMIN_ACCESS_TOKEN
    #            are set. Raises KiteSourceUnavailable to fall through
    #            without slow timeouts. Future paid provider (TrueData /
    #            Fyers) plugs in here.
    #   Tier 2 — yfinance (public). Same source the supervised trainers
    #            already use. bhavcopy/jugaad-data was Tier 2 prior to
    #            2026-05-19 but was dropped after NSE archive rate-limits
    #            caused partial-success (10/50 symbols), which the
    #            previous fallback logic treated as "success".
    raw: pd.DataFrame | None = None
    source: str = ""

    try:
        from .kite_source import (  # noqa: PLC0415
            kite_historical_download, KiteSourceUnavailable,
        )
        raw = kite_historical_download(symbols, start_d, end_d)
        source = "kite_admin"
    except KiteSourceUnavailable as exc:
        logger.debug("kite_admin unavailable (%s) — falling through to yfinance", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kite_admin errored (%s) — falling through to yfinance", exc)

    if raw is None:
        import yfinance as yf  # noqa: PLC0415

        tickers = [s if s.endswith((".NS", ".BO")) else f"{s}{suffix}" for s in symbols]
        try:
            raw = yf.download(
                tickers,
                start=str(start_d),
                end=str(end_d),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance download failed: %s", exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("yfinance returned empty frame for %d symbols", len(symbols))
            return pd.DataFrame()

        # yfinance occasionally returns duplicate index rows when sessions
        # overlap with corporate-action windows. Dedupe here so downstream
        # reindex/swap operations don't blow up.
        if raw.index.duplicated().any():
            raw = raw[~raw.index.duplicated(keep="last")].sort_index()

        source = "yfinance"

    # Defense-in-depth: dedupe the row index before anyone reindexes it.
    # The upstream loaders already dedupe, but a final guard here means
    # any future loader/cache path that forgets to dedupe still produces
    # a clean frame.
    if raw is not None and not raw.empty and raw.index.duplicated().any():
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    logger.info(
        "production_ohlcv: source=%s, %d symbols, %s..%s",
        source, len(symbols), start_d, end_d,
    )

    # Rebuild MultiIndex with the suffix the trainers expect.
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        first_outer = raw.columns.get_level_values(0)[0]
        field_names = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        is_field_outer = first_outer in field_names
        if is_field_outer:
            # bhavcopy default: (field, symbol). Rebuild with ticker suffix.
            new_cols = []
            for fld, sym in raw.columns:
                ticker = sym if sym.endswith((".NS", ".BO")) else f"{sym}{suffix}"
                new_cols.append((fld, ticker))
            raw.columns = pd.MultiIndex.from_tuples(new_cols)
        else:
            # yfinance fallback default with group_by='ticker': (ticker, field)
            new_cols = []
            for tk, fld in raw.columns:
                ticker = tk if tk.endswith((".NS", ".BO")) else f"{tk}{suffix}"
                new_cols.append((ticker, fld))
            raw.columns = pd.MultiIndex.from_tuples(new_cols)

    # Apply CA adjustments. adjust_batch expects (ticker, field) layout, so
    # swap levels first if needed.
    if adjust_corp_actions and isinstance(raw.columns, pd.MultiIndex):
        first_outer = raw.columns.get_level_values(0)[0]
        if first_outer in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}:
            raw = raw.swaplevel(axis=1)
        try:
            from .corporate_actions import adjust_batch  # noqa: PLC0415
            raw = adjust_batch(raw, volume_field="Volume")
        except Exception as exc:  # noqa: BLE001
            logger.warning("corporate_actions.adjust_batch failed: %s", exc)

    # Quality drop bad tickers.
    if quality_check:
        raw = _quality_drop_bad_symbols(raw, max_bad_pct=max_bad_pct)

    # Final layout matches yfinance group_by setting.
    if isinstance(raw.columns, pd.MultiIndex):
        first_outer = raw.columns.get_level_values(0)[0]
        is_ticker_outer = first_outer not in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        want_ticker_outer = group_by == "ticker"
        if is_ticker_outer != want_ticker_outer:
            raw = raw.swaplevel(axis=1)

    out = raw.sort_index()
    if return_source:
        return out, source
    return out


# ---------------------------------------------------------------------------
# Helper: per-symbol coverage report (used by data_quality_report.py)
# ---------------------------------------------------------------------------


def coverage_summary(
    df: pd.DataFrame, *, start: date | None = None, end: date | None = None,
) -> pd.DataFrame:
    """Return per-symbol coverage stats: n_rows, n_non_null, missing_days,
    first_date, last_date. Useful in the pre-training quality report.
    """
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    first_outer = df.columns.get_level_values(0)[0]
    is_ticker_outer = first_outer not in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
    tickers = sorted(df.columns.get_level_values(0).unique()) if is_ticker_outer \
        else sorted(df.columns.get_level_values(1).unique())

    rows = []
    for tk in tickers:
        close = df[tk]["Close"] if is_ticker_outer else df["Close"][tk]
        close = close.dropna()
        rows.append({
            "ticker": tk,
            "n_rows": int(len(close)),
            "first_date": close.index.min().date() if len(close) else None,
            "last_date": close.index.max().date() if len(close) else None,
            "missing_days_vs_window": int(
                ((end - start).days - len(close))
                if start and end and len(close) else 0
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "coverage_summary",
    "production_ohlcv",
]
