"""
Kite Connect Admin training-data source — Tier 1 highest quality.

When the admin Kite Connect subscription credentials are present in env,
this source delivers official broker-feed OHLCV (NSE/BSE-licensed,
pre-adjusted for corporate actions, same data institutional algo traders
use).

When credentials are absent, ``KiteSourceUnavailable`` is raised
immediately so ``production_ohlcv`` falls through to the bhavcopy
(jugaad-data) source without a slow timeout.

Env contract::

    KITE_ADMIN_API_KEY        — Kite Connect developer API key
    KITE_ADMIN_API_SECRET     — paired secret
    KITE_ADMIN_ACCESS_TOKEN   — session token (rotated daily 6 AM IST
                                via scheduler.refresh_kite_admin_token)

Output shape matches ``bhavcopy_download``: MultiIndex columns where
the outer level is the OHLCV field name (Open/High/Low/Close/Volume)
and the inner level is the unsuffixed NSE symbol. ``production_ohlcv``
re-arranges to ``(ticker, field)`` with the ``.NS`` suffix.
"""

from __future__ import annotations

import logging
import os
from datetime import date as Date, datetime, timedelta
from typing import Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class KiteSourceUnavailable(RuntimeError):
    """Raised when Kite admin credentials are missing or invalid.

    ``production_ohlcv`` catches this to fall through to Tier 2
    (bhavcopy) without a slow timeout.
    """


def _have_credentials() -> bool:
    return bool(
        os.environ.get("KITE_ADMIN_API_KEY")
        and os.environ.get("KITE_ADMIN_ACCESS_TOKEN")
    )


def _build_admin_client():
    """Lazy-construct a KiteAdminClient. Raises KiteSourceUnavailable
    on any setup failure so the caller can fall through quickly."""
    if not _have_credentials():
        raise KiteSourceUnavailable(
            "KITE_ADMIN_API_KEY / KITE_ADMIN_ACCESS_TOKEN not set"
        )
    try:
        from backend.services.kite_data_provider import (  # noqa: PLC0415
            KiteAdminClient, KiteDataProvider,
        )
        admin = KiteAdminClient()
        admin.set_access_token(os.environ["KITE_ADMIN_ACCESS_TOKEN"])
        return KiteDataProvider(admin)
    except Exception as exc:  # noqa: BLE001
        raise KiteSourceUnavailable(f"Kite admin client setup failed: {exc}") from exc


def kite_historical_download(
    symbols: Sequence[str],
    start: str | Date | datetime,
    end: str | Date | datetime,
) -> pd.DataFrame:
    """Pull official Kite Connect historical OHLCV for ``symbols`` and
    return a bhavcopy-shaped MultiIndex frame.

    Args:
        symbols: Unsuffixed NSE codes (``["RELIANCE", "TCS", ...]``).
        start, end: Date strings or date objects.

    Returns:
        DataFrame, index=trade date, columns=MultiIndex (field, symbol).
        Field set: Open, High, Low, Close, Volume.

    Raises:
        KiteSourceUnavailable: credentials missing, token invalid, or
            zero successful per-symbol fetches.
    """
    provider = _build_admin_client()

    start_d = pd.Timestamp(start).date() if not isinstance(start, Date) else start
    end_d = pd.Timestamp(end).date() if not isinstance(end, Date) else end
    days = max((end_d - start_d).days, 1)
    period = f"{days}d"

    per_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = provider.get_historical(sym, period=period, interval="1d")
        except Exception as exc:  # noqa: BLE001
            logger.debug("kite fetch failed for %s: %s", sym, exc)
            continue
        if df is None or df.empty:
            continue
        # KiteDataProvider returns lowercase cols + datetime index.
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        # Trim to requested window (provider may include extra)
        df = df.loc[(df.index.date >= start_d) & (df.index.date <= end_d)]
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")].sort_index()
        per_sym[sym] = df

    if not per_sym:
        raise KiteSourceUnavailable(
            f"Kite returned zero rows for all {len(symbols)} symbols "
            f"in [{start_d}, {end_d}] — token expired or rate-limited"
        )

    # Build MultiIndex (field, symbol) shape matching bhavcopy_download.
    blocks = []
    for sym, df in per_sym.items():
        block = df.copy()
        block.columns = pd.MultiIndex.from_product(
            [block.columns, [sym]], names=[None, None],
        )
        blocks.append(block)
    out = pd.concat(blocks, axis=1)
    if out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def kite_historical_with_fallback_status() -> Tuple[bool, str]:
    """Quick availability probe used by production_ohlcv to decide whether
    to attempt Kite at all. Returns (available, reason)."""
    if not _have_credentials():
        return False, "credentials missing"
    try:
        _build_admin_client()
        return True, "ready"
    except KiteSourceUnavailable as exc:
        return False, str(exc)


__all__ = [
    "KiteSourceUnavailable",
    "kite_historical_download",
    "kite_historical_with_fallback_status",
]
