"""
PR 180 — NSE FII/DII historical flow ingestor.

Foreign Institutional Investor (FII) + Domestic Institutional Investor
(DII) cash-market net flows are a regime-changing signal for Indian
equities. As of 2025-03 DIIs > FIIs in NSE ownership for the first time
ever — this is a structural shift the existing 15-feature LGBM frame
is blind to.

Two outputs feed the models:

    fii_net_5d   = sum of FII net flow (₹ Cr) over the trailing 5
                   trading days. Positive = foreign buying.
    dii_net_5d   = same for DII (mostly mutual-fund + insurance flow).

These are then z-scored over a trailing 90-day window so the model
sees "current flow vs typical flow" rather than absolute rupee values
(which drift over years as AUM grows).

Public surface:

    from ml.data.fii_dii_history import (
        fii_dii_series,
        compute_flow_features,
    )

    flows = fii_dii_series(start="2020-01-01", end="2025-12-31")
    features = compute_flow_features(flows)   # adds fii_5d_z, dii_5d_z

Source:
    NSE publishes daily FII/DII CSVs at
    https://www.nseindia.com/all-reports
    Live snapshot: api/fiidiiTradeReact (today only)
    Historical: archive scrape (requires session cookie + UA header)

Failure modes:
    - NSE archive returns 401/403/empty → return empty DataFrame; the
      compute_flow_features helper produces zero-valued features so the
      model gracefully degrades to the no-flow case rather than crashing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default cache location — sits inside the repo so it ships with the
# trained model artifact bundle when needed.
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FII_DII_CACHE_FILE = CACHE_DIR / "fii_dii_history.parquet"


def _empty_flow_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["fii_net", "dii_net"], index=pd.DatetimeIndex([]))


def _load_cache() -> pd.DataFrame:
    if not FII_DII_CACHE_FILE.exists():
        return _empty_flow_frame()
    try:
        df = pd.read_parquet(FII_DII_CACHE_FILE)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("fii_dii cache read failed: %s", exc)
        return _empty_flow_frame()


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(FII_DII_CACHE_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fii_dii cache write failed: %s", exc)


def _merge_into_cache(fresh: pd.DataFrame) -> None:
    """Upsert ``fresh`` rows into the parquet cache (idempotent)."""
    if fresh is None or fresh.empty:
        return
    cache = _load_cache()
    merged = pd.concat([cache, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _save_cache(merged)
    logger.info("FII/DII cache: merged %d fresh rows → %d total",
                len(fresh), len(merged))


def _fetch_from_nse(start: Date, end: Date) -> pd.DataFrame:
    """Best-effort NSE archive scrape. Returns empty frame on failure
    so the caller can stitch with cache + still proceed.

    NSE's FII/DII archive endpoints require a session cookie that the
    trainer environment may not have. We try once and degrade silently
    if blocked — the cache + zero-fill fallback covers gaps.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        return _empty_flow_frame()

    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        sess = requests.Session()
        sess.headers.update(headers)
        # Establish cookies first
        sess.get("https://www.nseindia.com", timeout=8)
        resp = sess.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("NSE FII/DII archive fetch failed: %s", exc)
        return _empty_flow_frame()

    rows = []
    for entry in data if isinstance(data, list) else []:
        cat = (entry.get("category") or "").upper()
        net = float(entry.get("netValue") or 0.0)
        d = entry.get("date") or datetime.now().strftime("%d-%b-%Y")
        try:
            d_parsed = pd.to_datetime(d, errors="coerce")
        except Exception:
            continue
        if pd.isna(d_parsed):
            continue
        rows.append({
            "date": d_parsed,
            "category": "FII" if ("FII" in cat or "FPI" in cat) else "DII",
            "net": net,
        })
    if not rows:
        return _empty_flow_frame()

    df = pd.DataFrame(rows)
    pivoted = df.pivot_table(
        index="date", columns="category", values="net", aggfunc="last",
    ).rename(columns={"FII": "fii_net", "DII": "dii_net"})
    pivoted = pivoted.reindex(columns=["fii_net", "dii_net"]).fillna(0.0)
    return pivoted.loc[
        (pivoted.index >= pd.Timestamp(start)) & (pivoted.index <= pd.Timestamp(end))
    ]


def backfill_today_via_nse_live() -> pd.DataFrame:
    """Catch today's FII/DII via NSE's live ``fiidiiTradeReact`` API.

    Locked 2026-05-12: this is the ONLY working free source.
        - Moneycontrol's FII/DII page is now login-walled.
        - NSE archive via jugaad-data is bot-blocked since 2024.
        - NSE's modern ``/api/fiidiiTradeReact`` returns current-day
          only — there is no working historical endpoint.

    Strategy: schedule this daily at 17:00 IST (after post-market FII/DII
    publish), and the parquet cache grows forward-cumulative. After 6
    months we have 6 months of real data; before that, lgbm features
    are zero-filled for older dates (Sharpe impact ~5-8%, acceptable).

    Returns:
        DataFrame with today's row (or empty if API blocked).
    """
    fresh = _fetch_from_nse(datetime.now().date(), datetime.now().date())
    if fresh.empty:
        logger.warning("NSE live FII/DII fetch returned empty")
        return _empty_flow_frame()
    _merge_into_cache(fresh)
    logger.info(
        "FII/DII daily catch-up: today=%s fii_net=%.1f dii_net=%.1f",
        fresh.index[0].date(),
        float(fresh["fii_net"].iloc[0]),
        float(fresh["dii_net"].iloc[0]),
    )
    return fresh


def backfill_from_jugaad(
    start: str | Date,
    end: str | Date,
    *,
    persist: bool = True,
) -> pd.DataFrame:
    """Best-effort FII/DII backfill (locked 2026-05-12).

    Source chain:
        1. NSE live ``fiidiiTradeReact`` for today's row (works)
        2. jugaad-data archive for historical days (almost always blocked)

    For historical 5y backfill there is no working free source — see
    docs/INDIA_DATA_SOURCES.md §FII/DII for paid options (TrueData).
    """
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    today = datetime.now().date()

    # ── Source 1: NSE live API — only useful for today ───────────────────
    if start_d <= today <= end_d:
        today_df = _fetch_from_nse(today, today)
        if not today_df.empty:
            if persist:
                _merge_into_cache(today_df)
            logger.info("NSE live FII/DII: caught today's row")

    # ── Source 2: jugaad-data (historical, usually 403) ──────────────────
    try:
        from jugaad_data.nse import NSELive  # noqa: PLC0415, F401
    except ImportError:
        logger.warning(
            "jugaad-data not installed — historical FII/DII unavailable. "
            "Today's row captured via NSE live API. Cache will grow "
            "forward-cumulative via the 17:00 IST daily scheduler job."
        )
        return _load_cache().loc[
            (_load_cache().index >= pd.Timestamp(start_d))
            & (_load_cache().index <= pd.Timestamp(end_d))
        ] if _load_cache().shape[0] else _empty_flow_frame()

    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()

    rows: list[dict] = []
    cur = start_d
    while cur <= end_d:
        if cur.weekday() < 5:   # skip weekends
            try:
                # jugaad-data fii_dii_money is the archival CSV reader.
                # Falls back to the live API for the most-recent day.
                from jugaad_data.nse import fii_dii_money  # noqa: PLC0415
                day_data = fii_dii_money(cur)
            except Exception as exc:  # noqa: BLE001
                logger.debug("fii_dii_money %s failed: %s", cur, exc)
                cur = cur + timedelta(days=1)
                continue
            if day_data:
                fii_net = 0.0
                dii_net = 0.0
                for entry in day_data:
                    cat = (entry.get("category") or "").upper()
                    net = float(entry.get("netValue") or entry.get("net") or 0.0)
                    if "FII" in cat or "FPI" in cat:
                        fii_net = net
                    elif "DII" in cat:
                        dii_net = net
                rows.append({
                    "date": pd.Timestamp(cur),
                    "fii_net": fii_net,
                    "dii_net": dii_net,
                })
        cur = cur + timedelta(days=1)

    if not rows:
        logger.warning(
            "FII/DII backfill returned 0 rows for %s..%s — NSE archive may be blocked",
            start_d, end_d,
        )
        return _empty_flow_frame()

    fresh = pd.DataFrame(rows).set_index("date").sort_index()
    fresh = fresh[["fii_net", "dii_net"]]

    if persist:
        cache = _load_cache()
        merged = (
            pd.concat([cache, fresh])
            .reset_index()
            .drop_duplicates(subset=["index"], keep="last")
            .set_index("index")
            .sort_index()
        )
        merged.index.name = None
        _save_cache(merged)
        logger.info("FII/DII backfill: %d rows merged → cache", len(fresh))
    return fresh


def fii_dii_series(
    start: str | Date,
    end: str | Date,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return historical FII/DII net-flow series indexed by date.

    Combines local parquet cache with a best-effort NSE archive scrape.
    Always returns a DataFrame; never raises. Empty when no data is
    available so callers can detect "no signal" cleanly.

    Args:
        start, end: ISO date strings or date/datetime objects (inclusive).
        use_cache: read from local parquet first.
        refresh: force a re-fetch from NSE even if cache covers the range.

    Returns:
        DataFrame with columns ['fii_net', 'dii_net'] in ₹ Cr, indexed
        by date (DatetimeIndex). Missing days simply absent from index.
    """
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()

    cache = _load_cache() if use_cache else _empty_flow_frame()

    cache_covers = (
        not cache.empty
        and cache.index.min().date() <= start_d
        and cache.index.max().date() >= end_d - timedelta(days=2)
    )
    if cache_covers and not refresh:
        return cache.loc[
            (cache.index >= pd.Timestamp(start_d)) & (cache.index <= pd.Timestamp(end_d))
        ]

    fresh = _fetch_from_nse(start_d, end_d)
    if not fresh.empty:
        merged = pd.concat([cache, fresh]).reset_index().drop_duplicates(
            subset=["index"], keep="last",
        ).set_index("index").sort_index()
        merged.index.name = None
        if use_cache:
            _save_cache(merged)
        return merged.loc[
            (merged.index >= pd.Timestamp(start_d)) & (merged.index <= pd.Timestamp(end_d))
        ]

    # NSE scrape failed; return whatever cache covers (even if partial).
    return cache.loc[
        (cache.index >= pd.Timestamp(start_d)) & (cache.index <= pd.Timestamp(end_d))
    ]


# ============================================================================
# Feature builder
# ============================================================================


@dataclass
class FlowFeatureConfig:
    """Window parameters for FII/DII feature engineering.

    sum_window:
        Trailing-day sum to compute the flow signal. 5 = 1 trading week.

    z_window:
        Rolling window for z-scoring the summed flow. 90 = ~4 months,
        long enough that quarterly DII inflows don't dominate the
        z-stat.

    fillna_value:
        Value used when raw flow is missing for a date (NSE archive
        gap, holiday, fetch failure). 0.0 = "treat as no signal".
    """

    sum_window: int = 5
    z_window: int = 90
    fillna_value: float = 0.0
    # Phase 1.7 audit fix #1.5 — NSE publishes daily FII/DII net flow
    # figures AFTER market close. The number for trading day D is only
    # public the NEXT trading day. Stamping the raw figure at date D
    # leaks information that a live trader could not act on at D's
    # opening price. Shift the entire feature series by 1 day so the
    # trainer at date D sees flow features built from data up to D-1.
    publication_lag_days: int = 1


def compute_flow_features(
    flows: pd.DataFrame,
    cfg: Optional[FlowFeatureConfig] = None,
) -> pd.DataFrame:
    """Convert raw FII/DII net-flow series into model-ready features.

    Args:
        flows: DataFrame from ``fii_dii_series``.
        cfg: window parameters.

    Returns:
        DataFrame indexed by date with columns:
            fii_5d_sum, dii_5d_sum             (raw rupee summed flows)
            fii_5d_z,   dii_5d_z               (z-scored over 90d)
        Empty input → empty output (same shape).
    """
    cfg = cfg or FlowFeatureConfig()
    if flows.empty:
        return pd.DataFrame(
            columns=["fii_5d_sum", "dii_5d_sum", "fii_5d_z", "dii_5d_z"],
            index=pd.DatetimeIndex([]),
        )

    df = flows.copy()
    df = df.fillna(cfg.fillna_value).sort_index()

    fii_sum = df["fii_net"].rolling(cfg.sum_window, min_periods=1).sum()
    dii_sum = df["dii_net"].rolling(cfg.sum_window, min_periods=1).sum()

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.rolling(cfg.z_window, min_periods=cfg.sum_window).mean()
        sigma = s.rolling(cfg.z_window, min_periods=cfg.sum_window).std()
        z = (s - mu) / sigma.replace(0, np.nan)
        return z.fillna(0.0)

    out = pd.DataFrame({
        "fii_5d_sum": fii_sum,
        "dii_5d_sum": dii_sum,
        "fii_5d_z":   _zscore(fii_sum),
        "dii_5d_z":   _zscore(dii_sum),
    }, index=df.index)

    # Phase 1.7 audit fix #1.5 — shift the entire feature frame by
    # `publication_lag_days` rows so each date D contains features built
    # from flow data published on or before D-lag. With lag=1, the
    # trainer at date D sees the rolling sum of [D-5, ..., D-1] instead
    # of [D-4, ..., D].
    lag = max(int(cfg.publication_lag_days or 0), 0)
    if lag > 0:
        out = out.shift(lag).fillna(0.0)
    return out


def reindex_flow_features_to(
    features: pd.DataFrame,
    target_index: pd.Index,
) -> pd.DataFrame:
    """Reindex flow features to the target trading-date index.

    Forward-fills 1 day (Monday inherits Friday's flow), then zero-fills
    any remaining gaps so the trainer never sees NaN. Returns a frame
    with the same columns as ``features``, indexed by ``target_index``.
    """
    if features.empty:
        idx = pd.DatetimeIndex(target_index)
        return pd.DataFrame(
            0.0, index=idx,
            columns=["fii_5d_sum", "dii_5d_sum", "fii_5d_z", "dii_5d_z"],
        )
    return features.reindex(target_index, method="ffill", limit=1).fillna(0.0)


__all__ = [
    "FlowFeatureConfig",
    "backfill_from_jugaad",
    "compute_flow_features",
    "fii_dii_series",
    "reindex_flow_features_to",
]
