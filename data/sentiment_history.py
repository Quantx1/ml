"""
PR 183 — historical FinBERT-India sentiment cache + feature builder.

The backend already has a live FinBERT-India scorer (``Vansh180/FinBERT-India-v1``
at backend/ai/sentiment/finbert_india.py) used to enrich live signals.
This module brings sentiment INTO the training feature set.

Pipeline:

    raw headlines (RSS / NSE announcements / scraped MoneyControl)
        ↓
    score_headlines_to_daily(symbol, date, headlines) → daily aggregate
        ↓
    parquet cache at ml/data/cache/sentiment_history.parquet
        ↓
    sentiment_features_for(symbols, start, end) → feature DataFrame
        with columns: sentiment_5d_mean, sentiment_5d_count

The trainer reads sentiment features just like FII/DII flow features.
When the cache is empty (early launch, no headline corpus yet) the
features zero-fill — equivalent to "no news signal" — so training
proceeds without artificial divergence.

Public surface:

    from ml.data.sentiment_history import (
        SentimentFeatureConfig,
        score_headlines_to_daily,
        sentiment_features_for,
    )

    # Backfill (one-time / nightly batch):
    score_headlines_to_daily(
        symbol="RELIANCE",
        date="2025-04-15",
        headlines=["Reliance Q4 beats estimates", "..."],
    )

    # Trainer-facing:
    feats = sentiment_features_for(["RELIANCE", "TCS"], "2024-01-01", "2025-04-30")

References:
    Vansh180/FinBERT-India-v1 (HuggingFace, Apache 2.0)
    AFML doesn't cover sentiment directly but the integration pattern
    mirrors FII/DII flow integration (PR 180): aggregate per-day,
    z-score over rolling window, reindex onto trainer's date axis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Local cache. Stays inside the repo so the sentiment history travels
# with the codebase (small enough — ~1 row per symbol per day).
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_CACHE_FILE = CACHE_DIR / "sentiment_history.parquet"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SentimentFeatureConfig:
    """Sentiment feature engineering parameters.

    rolling_window:
        Trailing-day window for mean + count features. 5 = trailing
        trading week.

    fillna_value:
        Value used when sentiment is missing (no news that day, no
        cache, FinBERT load failure). 0.0 = "no news signal".

    min_observations:
        Minimum headlines per day for the score to be considered
        meaningful. Days with fewer headlines have their score
        zeroed before aggregation.
    """

    rolling_window: int = 5
    fillna_value: float = 0.0
    min_observations: int = 1


# ============================================================================
# Cache I/O
# ============================================================================


def _empty_cache_frame() -> pd.DataFrame:
    """Schema: (symbol, date) → daily_score, headline_count.

    Stored as parquet with multi-column layout for fast per-symbol slice."""
    return pd.DataFrame(
        columns=["symbol", "date", "daily_score", "headline_count"],
    ).astype({
        "symbol": str, "daily_score": float, "headline_count": int,
    })


def _load_cache() -> pd.DataFrame:
    if not SENTIMENT_CACHE_FILE.exists():
        return _empty_cache_frame()
    try:
        df = pd.read_parquet(SENTIMENT_CACHE_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentiment cache read failed: %s", exc)
        return _empty_cache_frame()


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(SENTIMENT_CACHE_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentiment cache write failed: %s", exc)


# ============================================================================
# FinBERT scoring + cache write
# ============================================================================


def score_headlines_to_daily(
    symbol: str,
    date: str | Date,
    headlines: Sequence[str],
    *,
    persist: bool = True,
    finbert=None,
) -> dict:
    """Score a batch of headlines for one symbol-day with FinBERT-India.

    Args:
        symbol: NSE symbol (no .NS suffix).
        date: ISO date or date object.
        headlines: list of headline strings.
        persist: write the aggregated result to the parquet cache.
        finbert: optional pre-loaded FinBERTIndia instance (mainly for tests).

    Returns:
        dict with keys: symbol, date, daily_score, headline_count.
        daily_score = mean of (P(pos) - P(neg)) across headlines, in
        [-1, +1]. headline_count is the number of headlines actually
        scored (after FinBERT's neutral-fallback for failures).
    """
    if not headlines:
        result = {
            "symbol": symbol,
            "date": pd.to_datetime(date),
            "daily_score": 0.0,
            "headline_count": 0,
        }
        return result

    if finbert is None:
        try:
            from backend.ai.sentiment.finbert_india import get_finbert  # noqa: PLC0415
            finbert = get_finbert()
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT-India unavailable (%s); zero-filling score", exc)
            return {
                "symbol": symbol, "date": pd.to_datetime(date),
                "daily_score": 0.0, "headline_count": 0,
            }

    if not finbert.ready and not finbert.load():
        return {
            "symbol": symbol, "date": pd.to_datetime(date),
            "daily_score": 0.0, "headline_count": 0,
        }

    scored = finbert.classify_batch(list(headlines))
    if not scored:
        return {
            "symbol": symbol, "date": pd.to_datetime(date),
            "daily_score": 0.0, "headline_count": 0,
        }

    scores = [float(s.get("score", 0.0)) for s in scored]
    daily_score = float(np.mean(scores))
    result = {
        "symbol": symbol,
        "date": pd.to_datetime(date),
        "daily_score": daily_score,
        "headline_count": len(scored),
    }
    if persist:
        _persist_one(result)
    return result


def _persist_one(row: dict) -> None:
    """Append/upsert one (symbol, date) row into the cache."""
    cache = _load_cache()
    new_df = pd.DataFrame([row])
    if cache.empty:
        merged = new_df
    else:
        # Drop the same (symbol, date) if it exists, then append
        mask = ~((cache["symbol"] == row["symbol"]) & (cache["date"] == row["date"]))
        merged = pd.concat([cache.loc[mask], new_df], ignore_index=True)
    _save_cache(merged)


# ============================================================================
# Trainer-facing feature builder
# ============================================================================


def sentiment_features_for(
    symbols: Iterable[str],
    start: str | Date,
    end: str | Date,
    cfg: Optional[SentimentFeatureConfig] = None,
) -> pd.DataFrame:
    """Per-symbol per-date sentiment features for trainer feature joins.

    Args:
        symbols: NSE symbols (no .NS suffix).
        start, end: ISO date strings or date objects (inclusive).
        cfg: feature parameters.

    Returns:
        DataFrame with MultiIndex (symbol, date) and columns:
            sentiment_5d_mean — rolling-window mean of daily_score
            sentiment_5d_count — rolling-window total headline count

        Empty cache → all-zero feature frame with the right shape so
        the trainer can still merge by date.
    """
    cfg = cfg or SentimentFeatureConfig()
    start_d = pd.to_datetime(start)
    end_d = pd.to_datetime(end)
    cache = _load_cache()

    syms = list(symbols)
    date_range = pd.date_range(start_d, end_d, freq="B")
    out_records = []

    for sym in syms:
        sym_cache = cache.loc[cache["symbol"] == sym] if not cache.empty else cache
        # Daily series (zero-filled where no entry)
        if sym_cache.empty:
            scores = pd.Series(cfg.fillna_value, index=date_range, dtype=float)
            counts = pd.Series(0, index=date_range, dtype=int)
        else:
            sym_cache = sym_cache.set_index("date").sort_index()
            # Mask days with too-few headlines (treat as no signal)
            valid = sym_cache["headline_count"] >= cfg.min_observations
            scores = sym_cache["daily_score"].where(valid, cfg.fillna_value)
            scores = scores.reindex(date_range, fill_value=cfg.fillna_value)
            counts = sym_cache["headline_count"].reindex(
                date_range, fill_value=0,
            ).astype(int)

        roll_mean = scores.rolling(cfg.rolling_window, min_periods=1).mean()
        roll_count = counts.rolling(cfg.rolling_window, min_periods=1).sum()

        for d in date_range:
            out_records.append({
                "symbol": sym, "date": d,
                "sentiment_5d_mean": float(roll_mean.loc[d]),
                "sentiment_5d_count": int(roll_count.loc[d]),
            })

    if not out_records:
        return pd.DataFrame(
            columns=["sentiment_5d_mean", "sentiment_5d_count"],
        ).set_index(pd.MultiIndex.from_tuples([], names=["symbol", "date"]))

    out = pd.DataFrame(out_records).set_index(["symbol", "date"])
    return out


def reindex_sentiment_to(
    sentiment: pd.DataFrame,
    symbol: str,
    target_index: pd.Index,
    cfg: Optional[SentimentFeatureConfig] = None,
) -> pd.DataFrame:
    """Slice a per-symbol sentiment frame onto a trainer's date axis.

    Returns DataFrame with same length as target_index, zero-filled
    where the source had no entry. Drops the symbol level so the
    output can concatenate column-wise into a per-symbol feature
    frame.
    """
    cfg = cfg or SentimentFeatureConfig()
    cols = ["sentiment_5d_mean", "sentiment_5d_count"]
    if sentiment is None or sentiment.empty:
        return pd.DataFrame(
            {c: cfg.fillna_value if "mean" in c else 0 for c in cols},
            index=pd.DatetimeIndex(target_index),
        )
    try:
        slice_ = sentiment.loc[symbol]
    except KeyError:
        return pd.DataFrame(
            {c: cfg.fillna_value if "mean" in c else 0 for c in cols},
            index=pd.DatetimeIndex(target_index),
        )
    out = slice_.reindex(target_index, method="ffill", limit=1)
    out["sentiment_5d_mean"] = out["sentiment_5d_mean"].fillna(cfg.fillna_value)
    out["sentiment_5d_count"] = out["sentiment_5d_count"].fillna(0).astype(int)
    return out


__all__ = [
    "SentimentFeatureConfig",
    "reindex_sentiment_to",
    "score_headlines_to_daily",
    "sentiment_features_for",
]
