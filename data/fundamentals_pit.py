"""
PR 184 — point-in-time fundamentals layer.

Quant funds extract serious alpha from forward EPS, revenue trajectory,
operating margin trend, promoter holding deltas, and FII shareholding
flows. Until this PR our trainers consumed only OHLCV-derived features.
The earnings_xgb model in particular was structurally limited because
its inputs were price patterns + earnings DATES — never the actual
fundamentals.

This module is **point-in-time correct** by design:

  Each fundamentals row has a `published_date` field — the date the
  filing actually became public on NSE/BSE. Trainers join by
  `as_of_date < published_date` so the model NEVER sees data that
  wouldn't have been available to a live trader at that timestamp.
  This is the discipline AFML calls "preventing look-ahead leakage
  via PIT data joins."

Storage shape (parquet cache):

    columns: symbol, period_end, published_date, eps_ttm, revenue_yoy,
             operating_margin, promoter_pct, fii_pct, dii_pct,
             debt_to_equity, book_value, source

  - `period_end`     = quarter end (Q1=Jun-30, Q2=Sep-30, etc.)
  - `published_date` = when the filing dropped (4-8 weeks AFTER
                       period_end on NSE)
  - `source`         = "nse_filing" | "yfinance" | "screener_in"

Public surface:

    from ml.data.fundamentals_pit import (
        FundamentalsRecord,
        ingest_yfinance_fundamentals,
        get_pit_fundamentals,
        compute_fundamentals_features,
    )

    # Backfill (one-time / nightly):
    ingest_yfinance_fundamentals(symbols=["RELIANCE", "TCS"])

    # Trainer-facing:
    fund = get_pit_fundamentals(["RELIANCE"], as_of="2024-06-15")
    feats = compute_fundamentals_features(fund)
    # → DataFrame with eps_yoy_growth, revenue_yoy_growth,
    #   margin_trend_4q, promoter_delta_4q, fii_delta_4q

The cache is bootstrapped from yfinance (free, latest-restated values
acceptable for v1) and screener.in scrape (when accessible). For
production-grade PIT correctness, the next iteration ingests directly
from NSE corporate filings, but yfinance + a 60-day publication lag
assumption is sufficient to land sensible features in v1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FUNDAMENTALS_CACHE_FILE = CACHE_DIR / "fundamentals_pit.parquet"


# Average lag from quarter-end to NSE filing publication. Indian listed
# companies must file results within 45 days (extended to 60 for
# certain segments). 60 days is the conservative PIT assumption when
# the actual publication date is unknown.
DEFAULT_PUBLICATION_LAG_DAYS = 60


# ============================================================================
# Data model
# ============================================================================


FUNDAMENTAL_FIELDS = [
    "eps_ttm",            # trailing-12-month EPS in ₹
    "revenue_yoy",        # YoY revenue growth, e.g. 0.12 = +12%
    "operating_margin",   # OPM as fraction (0.18 = 18%)
    "promoter_pct",       # promoter shareholding (0..1)
    "fii_pct",            # FII shareholding (0..1)
    "dii_pct",            # DII shareholding (0..1)
    "debt_to_equity",     # D/E ratio
    "book_value",         # book value per share in ₹
]

CACHE_COLUMNS = [
    "symbol", "period_end", "published_date", "source",
    *FUNDAMENTAL_FIELDS,
]


@dataclass
class FundamentalsRecord:
    """One quarterly fundamentals snapshot for one symbol.

    period_end:
        Date the financial period ended (e.g. 2024-06-30 for Q1 FY25).

    published_date:
        Date the filing became public on NSE. This is what trainers
        join on for PIT discipline. If unknown, set to
        period_end + DEFAULT_PUBLICATION_LAG_DAYS.

    source:
        Where this row came from. Lower-priority sources are overwritten
        by higher-priority sources during ingestion.
    """

    symbol: str
    period_end: Date
    published_date: Date
    source: str
    eps_ttm: float = float("nan")
    revenue_yoy: float = float("nan")
    operating_margin: float = float("nan")
    promoter_pct: float = float("nan")
    fii_pct: float = float("nan")
    dii_pct: float = float("nan")
    debt_to_equity: float = float("nan")
    book_value: float = float("nan")

    def to_row(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period_end": pd.Timestamp(self.period_end),
            "published_date": pd.Timestamp(self.published_date),
            "source": self.source,
            **{k: float(getattr(self, k)) for k in FUNDAMENTAL_FIELDS},
        }


# ============================================================================
# Cache I/O
# ============================================================================


def _empty_cache_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=CACHE_COLUMNS)
    return df.astype({
        "symbol": str, "source": str,
        **{k: float for k in FUNDAMENTAL_FIELDS},
    })


def _load_cache() -> pd.DataFrame:
    if not FUNDAMENTALS_CACHE_FILE.exists():
        return _empty_cache_frame()
    try:
        df = pd.read_parquet(FUNDAMENTALS_CACHE_FILE)
        for col in ("period_end", "published_date"):
            df[col] = pd.to_datetime(df[col])
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamentals cache read failed: %s", exc)
        return _empty_cache_frame()


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(FUNDAMENTALS_CACHE_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamentals cache write failed: %s", exc)


# Source priority — higher = preferred. When two records exist for the
# same (symbol, period_end), the higher-priority source wins.
_SOURCE_PRIORITY = {"nse_filing": 3, "screener_in": 2, "yfinance": 1, "manual": 4}


def upsert_records(records: Iterable[FundamentalsRecord]) -> int:
    """Insert/upsert records into the cache. Returns count written."""
    rows = [r.to_row() for r in records]
    if not rows:
        return 0
    new_df = pd.DataFrame(rows).astype({"symbol": str, "source": str})
    cache = _load_cache()
    if cache.empty:
        merged = new_df
    else:
        # Concatenate; for duplicate (symbol, period_end), keep the row
        # with highest source priority. Build a priority column for
        # ranking, drop after.
        combined = pd.concat([cache, new_df], ignore_index=True)
        combined["_prio"] = combined["source"].map(_SOURCE_PRIORITY).fillna(0)
        combined = combined.sort_values(["symbol", "period_end", "_prio"])
        merged = combined.drop_duplicates(
            subset=["symbol", "period_end"], keep="last",
        ).drop(columns=["_prio"]).reset_index(drop=True)
    _save_cache(merged)
    return len(rows)


# ============================================================================
# yfinance ingestion (free, latest-restated)
# ============================================================================


def ingest_yfinance_fundamentals(
    symbols: Sequence[str],
    *,
    publication_lag_days: int = DEFAULT_PUBLICATION_LAG_DAYS,
    persist: bool = True,
) -> List[FundamentalsRecord]:
    """Fetch quarterly fundamentals from yfinance + write to cache.

    yfinance returns the latest *restated* values (not original
    filings). For v1 this is acceptable but flagged as a known
    PIT-correctness limitation: any restatement that occurred after
    `period_end + publication_lag_days` is silently absorbed. Real
    PIT-correct ingestion needs NSE filings (next iteration).

    Returns the list of FundamentalsRecord built. Optionally persists.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        logger.warning("yfinance unavailable; skipping fundamentals ingest")
        return []

    out: List[FundamentalsRecord] = []
    for sym in symbols:
        ticker = f"{sym}.NS"
        try:
            t = yf.Ticker(ticker)
            qfin = t.quarterly_financials
            qbs = t.quarterly_balance_sheet
            info = t.info or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance fundamentals %s failed: %s", sym, exc)
            continue
        if qfin is None or qfin.empty:
            continue

        # qfin columns are quarter-end dates; rows are line items.
        for period_end_ts in qfin.columns[:8]:   # last 8 quarters max
            try:
                period_end = pd.Timestamp(period_end_ts).date()
            except Exception:
                continue
            published = period_end + timedelta(days=publication_lag_days)
            rec = FundamentalsRecord(
                symbol=sym,
                period_end=period_end,
                published_date=published,
                source="yfinance",
            )
            # Operating margin = Operating Income / Total Revenue
            try:
                rev = float(qfin.loc["Total Revenue", period_end_ts])
                op_inc = float(qfin.loc["Operating Income", period_end_ts])
                if rev and not np.isnan(rev):
                    rec.operating_margin = op_inc / rev
            except (KeyError, ValueError, TypeError):
                pass
            # Revenue YoY: this period vs same period last year
            try:
                same_period_ly = period_end_ts - pd.DateOffset(years=1)
                # Find closest column
                if len(qfin.columns) > 4:
                    ly_idx = qfin.columns[4] if len(qfin.columns) >= 5 else None
                    if ly_idx is not None:
                        ly_rev = float(qfin.loc["Total Revenue", ly_idx])
                        if ly_rev and not np.isnan(ly_rev):
                            rec.revenue_yoy = (rev - ly_rev) / abs(ly_rev)
            except Exception:
                pass
            # Debt-to-equity from balance sheet — already PIT-correct
            # because qbs columns are quarter-end dates.
            try:
                if qbs is not None and not qbs.empty and period_end_ts in qbs.columns:
                    debt = float(qbs.loc["Total Debt", period_end_ts])
                    equity = float(qbs.loc["Stockholders Equity", period_end_ts])
                    if equity and not np.isnan(equity) and equity != 0:
                        rec.debt_to_equity = debt / equity
            except (KeyError, ValueError, TypeError):
                pass

            # Phase 1.7 audit fix #1.4 — eps_ttm, book_value, and fii_pct
            # come from `yfinance.Ticker.info`, which returns CURRENT
            # values (latest snapshot). The legacy code stamped the same
            # current values on every historical quarter, which is a
            # textbook look-ahead leak: a 2024-Q1 training row got fed
            # 2026 EPS and 2026 institutional holdings. The promote-gate
            # backtest then "succeeded" by predicting the present.
            #
            # Repair strategy: compute these PIT-correctly from per-quarter
            # statements when the data is present in qfin/qbs. Otherwise
            # leave the field NaN and DO NOT backfill with the live info
            # snapshot.
            #
            # Book value/share PIT: Stockholders Equity / Diluted Shares
            # Outstanding at the same period_end column.
            try:
                if qbs is not None and not qbs.empty and period_end_ts in qbs.columns:
                    equity = float(qbs.loc["Stockholders Equity", period_end_ts])
                    shares = None
                    # qbs row names differ across vendors; try the common ones.
                    for row_name in (
                        "Share Issued", "Ordinary Shares Number",
                        "Common Stock Equity", "Shares Outstanding",
                    ):
                        if row_name in qbs.index:
                            try:
                                shares = float(qbs.loc[row_name, period_end_ts])
                                if shares and not np.isnan(shares) and shares > 0:
                                    break
                                shares = None
                            except (KeyError, ValueError, TypeError):
                                shares = None
                    if (
                        shares
                        and equity
                        and not np.isnan(equity)
                        and not np.isnan(shares)
                        and shares > 0
                    ):
                        rec.book_value = equity / shares
            except (KeyError, ValueError, TypeError):
                pass

            # EPS_ttm PIT: Sum of last 4 quarters of Diluted EPS as of
            # period_end. Skip the symbol when fewer than 4 prior columns
            # exist (early-history rows would otherwise be biased toward
            # whichever quarter happened to be available).
            try:
                diluted_eps_row = None
                for row_name in ("Diluted EPS", "Basic EPS"):
                    if row_name in qfin.index:
                        diluted_eps_row = row_name
                        break
                if diluted_eps_row is not None:
                    pe_pos = list(qfin.columns).index(period_end_ts)
                    if pe_pos >= 3:
                        cols_4q = qfin.columns[pe_pos - 3: pe_pos + 1]
                        vals = []
                        for c in cols_4q:
                            v = qfin.loc[diluted_eps_row, c]
                            try:
                                fv = float(v)
                                if not np.isnan(fv):
                                    vals.append(fv)
                            except (TypeError, ValueError):
                                continue
                        if len(vals) == 4:
                            rec.eps_ttm = float(sum(vals))
            except (KeyError, ValueError, IndexError, TypeError):
                pass

            # fii_pct PIT: yfinance does NOT expose historical
            # institutional shareholding — only the CURRENT snapshot in
            # `info["heldPercentInstitutions"]`. Stamping that on
            # historical rows leaks the present into the past.
            #
            # The trainer must source FII shareholding from NSE corporate
            # filings (shareholding pattern, quarterly) — that lives in a
            # separate pipeline. Until that backfill lands, fii_pct stays
            # NaN on historical quarters and the consumer (reindex_*)
            # zero-fills. Better to feed the model an honest zero than a
            # peek at the future.

            out.append(rec)

    if persist and out:
        n = upsert_records(out)
        logger.info(
            "fundamentals_pit: ingested %d rows from yfinance for %d symbols",
            n, len(symbols),
        )
    return out


# ============================================================================
# Point-in-time lookup
# ============================================================================


def get_pit_fundamentals(
    symbols: Sequence[str],
    as_of: str | Date,
    *,
    n_quarters: int = 4,
) -> pd.DataFrame:
    """Return the most-recent N fundamentals quarters PIT-published as of `as_of`.

    A row is included only if its `published_date <= as_of`. This is
    the discipline that prevents look-ahead leakage.

    Returns:
        DataFrame indexed by (symbol, period_end). Empty when no
        records are available — caller zero-fills features.
    """
    cache = _load_cache()
    if cache.empty:
        return pd.DataFrame(
            columns=FUNDAMENTAL_FIELDS,
        ).set_index(pd.MultiIndex.from_tuples([], names=["symbol", "period_end"]))

    as_of_ts = pd.Timestamp(as_of)
    out_rows: List[pd.DataFrame] = []
    for sym in symbols:
        sym_rows = cache.loc[
            (cache["symbol"] == sym) & (cache["published_date"] <= as_of_ts)
        ].sort_values("period_end", ascending=False).head(n_quarters)
        if sym_rows.empty:
            continue
        out_rows.append(sym_rows)

    if not out_rows:
        return pd.DataFrame(
            columns=FUNDAMENTAL_FIELDS,
        ).set_index(pd.MultiIndex.from_tuples([], names=["symbol", "period_end"]))

    df = pd.concat(out_rows, ignore_index=True)
    return df.set_index(["symbol", "period_end"])[FUNDAMENTAL_FIELDS]


# ============================================================================
# Feature builder
# ============================================================================


@dataclass
class FundamentalsFeatureConfig:
    """Aggregation parameters for the fundamentals feature builder.

    n_quarters_for_growth:
        How many quarters back to look for YoY growth comparison.
        4 = same quarter previous year (standard YoY).

    fillna_value:
        Value for missing features (no fundamentals data, IPO too
        recent, etc.). 0.0 = "no signal".
    """

    n_quarters_for_growth: int = 4
    fillna_value: float = 0.0


# Feature names emitted by compute_fundamentals_features
FUNDAMENTALS_FEATURE_NAMES = [
    "eps_yoy_growth",        # latest EPS / 4-quarters-ago EPS - 1
    "revenue_yoy_growth",    # latest revenue_yoy
    "margin_trend_4q",       # latest opmargin minus 4-quarter mean
    "promoter_delta_4q",     # latest promoter_pct minus 4q-ago promoter_pct
    "fii_delta_4q",          # latest fii_pct minus 4q-ago fii_pct
    "debt_to_equity",        # latest D/E (lower = healthier)
    "book_value_yoy",        # latest book_value / 4-quarters-ago - 1
    "fundamentals_age_days", # how stale is the latest snapshot (vs as_of)
]


def compute_fundamentals_features(
    fundamentals: pd.DataFrame,
    as_of: str | Date,
    cfg: Optional[FundamentalsFeatureConfig] = None,
) -> pd.DataFrame:
    """Compute per-symbol features from a PIT fundamentals frame.

    Args:
        fundamentals: output of get_pit_fundamentals() —
                      MultiIndex(symbol, period_end), columns =
                      FUNDAMENTAL_FIELDS.
        as_of: anchor date for the freshness feature.
        cfg: feature parameters.

    Returns:
        DataFrame indexed by symbol, columns = FUNDAMENTALS_FEATURE_NAMES.
        Symbols with no data → row of fillna_value (caller may drop or
        keep depending on downstream merge semantics).
    """
    cfg = cfg or FundamentalsFeatureConfig()
    as_of_ts = pd.Timestamp(as_of)
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(
            columns=FUNDAMENTALS_FEATURE_NAMES,
        )

    rows = []
    for sym, sym_df in fundamentals.groupby(level="symbol"):
        sym_df = sym_df.reset_index().sort_values("period_end")
        latest = sym_df.iloc[-1]
        prior = sym_df.iloc[-cfg.n_quarters_for_growth - 1] if len(sym_df) > cfg.n_quarters_for_growth else None

        feat = {n: cfg.fillna_value for n in FUNDAMENTALS_FEATURE_NAMES}

        # YoY growth metrics
        if prior is not None:
            if not np.isnan(latest["eps_ttm"]) and not np.isnan(prior["eps_ttm"]) and abs(prior["eps_ttm"]) > 1e-9:
                feat["eps_yoy_growth"] = float((latest["eps_ttm"] - prior["eps_ttm"]) / abs(prior["eps_ttm"]))
            if not np.isnan(latest["book_value"]) and not np.isnan(prior["book_value"]) and abs(prior["book_value"]) > 1e-9:
                feat["book_value_yoy"] = float((latest["book_value"] - prior["book_value"]) / abs(prior["book_value"]))

        if not np.isnan(latest["revenue_yoy"]):
            feat["revenue_yoy_growth"] = float(latest["revenue_yoy"])

        # Operating margin trend: latest minus mean of last 4 quarters
        if "operating_margin" in sym_df:
            recent = sym_df["operating_margin"].dropna().tail(cfg.n_quarters_for_growth)
            if len(recent) >= 2 and not np.isnan(latest["operating_margin"]):
                feat["margin_trend_4q"] = float(latest["operating_margin"] - recent.mean())

        # Promoter / FII deltas
        if prior is not None:
            if not np.isnan(latest["promoter_pct"]) and not np.isnan(prior["promoter_pct"]):
                feat["promoter_delta_4q"] = float(latest["promoter_pct"] - prior["promoter_pct"])
            if not np.isnan(latest["fii_pct"]) and not np.isnan(prior["fii_pct"]):
                feat["fii_delta_4q"] = float(latest["fii_pct"] - prior["fii_pct"])

        if not np.isnan(latest["debt_to_equity"]):
            feat["debt_to_equity"] = float(latest["debt_to_equity"])

        # Freshness: how many days old is the latest published snapshot?
        latest_pub = pd.Timestamp(latest["period_end"])
        # Note: we only have period_end here; published_date is in cache.
        # Approximate freshness as (as_of - period_end) which is biased
        # but trainable.
        feat["fundamentals_age_days"] = float((as_of_ts - latest_pub).days)

        rows.append({"symbol": sym, **feat})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=FUNDAMENTALS_FEATURE_NAMES)
    return out.set_index("symbol")[FUNDAMENTALS_FEATURE_NAMES]


def reindex_fundamentals_to(
    feature_row: Optional[pd.Series],
    target_index: pd.Index,
    cfg: Optional[FundamentalsFeatureConfig] = None,
) -> pd.DataFrame:
    """Broadcast a per-symbol fundamentals feature row across a date axis.

    Fundamentals only update quarterly; for a daily date axis we
    broadcast the latest value to every day. When the symbol has no
    record, return a fillna_value-filled frame.
    """
    cfg = cfg or FundamentalsFeatureConfig()
    idx = pd.DatetimeIndex(target_index)
    if feature_row is None or feature_row.empty:
        return pd.DataFrame(
            cfg.fillna_value, index=idx, columns=FUNDAMENTALS_FEATURE_NAMES,
        )
    return pd.DataFrame(
        {col: float(feature_row.get(col, cfg.fillna_value)) for col in FUNDAMENTALS_FEATURE_NAMES},
        index=idx,
    )


__all__ = [
    "DEFAULT_PUBLICATION_LAG_DAYS",
    "FUNDAMENTAL_FIELDS",
    "FUNDAMENTALS_FEATURE_NAMES",
    "FundamentalsFeatureConfig",
    "FundamentalsRecord",
    "compute_fundamentals_features",
    "get_pit_fundamentals",
    "ingest_yfinance_fundamentals",
    "reindex_fundamentals_to",
    "upsert_records",
]
