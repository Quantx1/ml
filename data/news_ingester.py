"""
PR 194 — Google News RSS headline ingester for the sentiment cache.

The FinBERT-India sentiment cache (PR 183) is empty until something
populates it. The audit flagged that no headline ingestion pipeline
existed, so sentiment_5d_mean / sentiment_5d_count are zero across
the entire training window.

This module fills that gap with a Google News RSS scraper. It's
deliberately minimal:
  - One feed per symbol, query = "{symbol} stock NSE"
  - Returns last 100 results (Google News RSS default cap)
  - Today-only — Google News RSS doesn't support historical date queries

For real backfill we'd need MoneyControl scrape / NSE corporate
filings / paid news APIs (Bloomberg, Refinitiv). v1 ships the
real-time pipeline; ops can run this nightly via the scheduler so the
cache accumulates 30-90 days before the first trainer pass uses it.

Public surface:

    from ml.data.news_ingester import (
        fetch_headlines_for_symbol,
        backfill_sentiment_cache,
    )

    # One symbol:
    headlines = fetch_headlines_for_symbol("RELIANCE")
    score = score_headlines_to_daily("RELIANCE", today, headlines)

    # Universe-wide (typical nightly cron):
    backfill_sentiment_cache(symbols=liquid_universe())
"""

from __future__ import annotations

import logging
import time
from datetime import date as Date, datetime
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def fetch_headlines_for_symbol(
    symbol: str,
    *,
    query_template: str = "{symbol} stock NSE",
    limit: int = 100,
    timeout: float = 8.0,
) -> List[str]:
    """Fetch up to ``limit`` recent headlines for one NSE symbol.

    Returns plain title strings. Empty list on any failure (network,
    parse error, rate-limit) so the caller can skip and try again later.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.warning("requests missing — news ingester disabled")
        return []

    try:
        from xml.etree import ElementTree as ET  # noqa: PLC0415
    except ImportError:
        return []

    query = query_template.format(symbol=symbol)
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; QuantX-NewsBot/1.0; "
            "+https://github.com/Ri2506/quantx)"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml",
    }
    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.debug("news fetch %s failed: %s", symbol, exc)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    titles: List[str] = []
    for item in root.iter("item"):
        title_el = item.find("title")
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()
        # Google News appends the publication name after a final " - "
        # separator. Strip it so FinBERT sees the headline only.
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def backfill_sentiment_cache(
    symbols: Sequence[str],
    *,
    target_date: Optional[str | Date] = None,
    rate_limit_seconds: float = 1.5,
    finbert=None,
) -> dict:
    """Run news fetch + FinBERT scoring for every symbol; persist to
    the sentiment parquet cache.

    Args:
        symbols: NSE symbol codes (no .NS suffix).
        target_date: date to attach to scored aggregates. Default today.
        rate_limit_seconds: sleep between Google News calls. 1.5s is
                            polite — Google rate-limits aggressive
                            scrapers.
        finbert: pre-loaded FinBERTIndia (mainly for tests). Default
                 lazy-loads via get_finbert().

    Returns:
        {"n_symbols": ..., "n_with_headlines": ..., "n_scored": ...,
         "errors": [{"symbol": ..., "reason": ...}, ...]}
    """
    from .sentiment_history import score_headlines_to_daily  # noqa: PLC0415

    if target_date is None:
        target_date = datetime.now().date()

    n_symbols = 0
    n_with_headlines = 0
    n_scored = 0
    errors = []

    for sym in symbols:
        n_symbols += 1
        try:
            headlines = fetch_headlines_for_symbol(sym)
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": sym, "reason": f"fetch: {exc}"})
            time.sleep(rate_limit_seconds)
            continue

        if not headlines:
            time.sleep(rate_limit_seconds)
            continue
        n_with_headlines += 1
        try:
            result = score_headlines_to_daily(
                symbol=sym, date=target_date, headlines=headlines,
                persist=True, finbert=finbert,
            )
            if result.get("headline_count", 0) > 0:
                n_scored += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": sym, "reason": f"score: {exc}"})

        time.sleep(rate_limit_seconds)

    logger.info(
        "sentiment backfill: %d/%d scored, %d had headlines, %d errors",
        n_scored, n_symbols, n_with_headlines, len(errors),
    )
    return {
        "n_symbols": n_symbols,
        "n_with_headlines": n_with_headlines,
        "n_scored": n_scored,
        "errors": errors[:50],   # cap for response size
    }


__all__ = [
    "GOOGLE_NEWS_RSS",
    "backfill_sentiment_cache",
    "fetch_headlines_for_symbol",
]
