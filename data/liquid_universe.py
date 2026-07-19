"""
PR 164 — liquid NSE universe builder.

Replaces hardcoded 18-30 large-cap symbol lists in the trainers with a
deterministic, daily-recomputed list of the top-N most-liquid NSE
stocks by 30-day median ADV (average dollar volume = close * volume).

Why this matters for real-money:

  - Hardcoded 18-30 names overfits the model to today's bluechips and
    misses mid-cap alpha
  - When a name's liquidity drops below tradeable, training continues
    on it and inference produces signals nobody can fill
  - "Top-N by liquidity" is the universe definition every serious
    quant fund uses; matches Qlib's CSI300 convention adapted to NSE

Public surface:

    from ml.data import liquid_universe, LiquidUniverseConfig

    cfg = LiquidUniverseConfig(top_n=200, lookback_days=30)
    symbols = liquid_universe(cfg)
    # ['RELIANCE', 'HDFCBANK', 'TCS', ...] sorted by liquidity desc

Caching:
    The universe is cached in memory per (top_n, lookback_days) for the
    duration of a Python process. Trainers should call once at the top
    of train() and reuse.

Failure modes:
    - yfinance batch download fails → fall back to NIFTY_200_FALLBACK
      (a static list of the historical Nifty 200 constituents). Never
      raises so trainers don't blow up on transient yfinance issues.
    - Insufficient symbols downloaded → trim top_n to what's available
      and log warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .delisted_registry import historical_universe_extras, was_listed_at

logger = logging.getLogger(__name__)


DateLike = Union[str, Date, pd.Timestamp, datetime]


def _to_date(d: Optional[DateLike]) -> Optional[Date]:
    if d is None:
        return None
    if isinstance(d, Date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# ============================================================================
# Static fallbacks (Nifty 50 + Nifty 200 historical constituents)
# ============================================================================
#
# Used when yfinance batch download fails. These are the historical
# constituents and may drift from current. The dynamic builder is the
# correct path; these are safety nets.


NIFTY_50_FALLBACK: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "WIPRO", "HCLTECH", "ULTRACEMCO",
    "TITAN", "SUNPHARMA", "POWERGRID", "NTPC", "ADANIENT", "ONGC",
    "TATAMOTORS", "JSWSTEEL", "GRASIM", "DIVISLAB", "TECHM", "DRREDDY",
    "NESTLEIND", "EICHERMOT", "BAJAJFINSV", "TATASTEEL", "COALINDIA",
    "BPCL", "BAJAJ-AUTO", "INDUSINDBK", "HDFCLIFE", "SBILIFE", "M&M",
    "CIPLA", "BRITANNIA", "HEROMOTOCO", "TATACONSUM", "APOLLOHOSP",
    "ADANIPORTS", "HINDALCO", "UPL", "SHRIRAMFIN",
]


# Nifty 200 = Nifty 100 + Nifty Next 100. Approximate as of 2024-2025.
# Mid-caps with consistently high ADV.
NIFTY_200_EXTRAS: List[str] = [
    "GODREJCP", "PIDILITIND", "DABUR", "AMBUJACEM", "ACC",
    "ICICIPRULI", "MUTHOOTFIN", "LTIM", "DMART", "VEDL",
    "GAIL", "MARICO", "BERGEPAINT", "SIEMENS", "BOSCHLTD",
    "MOTHERSON", "PIIND", "LICI", "TRENT", "PERSISTENT",
    "CONCOR", "HAL", "BEL", "BHEL", "BANKBARODA",
    "PNB", "CANBK", "UNIONBANK", "FEDERALBNK", "IDFCFIRSTB",
    "RBLBANK", "AUBANK", "BANDHANBNK", "ZOMATO", "PAYTM",
    "POLICYBZR", "NYKAA", "POLYCAB", "HAVELLS", "VOLTAS",
    "CROMPTON", "DIXON", "TATAELXSI", "MPHASIS", "COFORGE",
    "OFSS", "INDIGO", "INDIAMART", "JUBLFOOD", "NAUKRI",
    "ASTRAL", "DEEPAKNTR", "ALKEM", "TORNTPHARM", "BIOCON",
    "AUROPHARMA", "LUPIN", "GLAND", "GLENMARK", "ZYDUSLIFE",
    "ABBOTINDIA", "PFIZER", "TVSMOTOR", "ASHOKLEY", "BHARATFORG",
    "MRF", "BALKRISIND", "APOLLOTYRE", "CEAT", "ESCORTS",
    "EXIDEIND", "AMARAJABAT", "RECLTD", "PFC", "POWERINDIA",
    "ABCAPITAL", "ADANIPOWER", "ADANITRANS", "ADANIGREEN", "ADANITOTAL",
    "TORNTPOWER", "JSWENERGY", "TATAPOWER", "NHPC", "NLCINDIA",
    "JINDALSTEL", "SAIL", "NMDC", "NATIONALUM", "HINDCOPPER",
    "MOIL", "RATNAMANI", "APLAPOLLO", "JSWINFRA", "ADANITRANS",
    "MCDOWELL-N", "UBL", "VBL", "RADICO", "GODREJIND",
    "GODREJPROP", "DLF", "SOBHA", "OBEROIRLTY", "PRESTIGE",
    "BRIGADE", "PHOENIXLTD", "BAJAJHLDNG", "INDHOTEL", "CHOLAFIN",
    "CHOLAHLDNG", "SRTRANSFIN", "POONAWALLA", "MANAPPURAM", "L&TFH",
    "LICHSGFIN", "PNBHOUSING", "REPCOHOME", "INDIACEM", "RAMCOCEM",
    "SHREECEM", "PRSMJOHNSN", "HEIDELBERG", "JKCEMENT", "JKLAKSHMI",
    "DALBHARAT", "STARCEMENT", "BIRLASOFT", "INTELLECT", "SONATSOFTW",
    "RPOWER", "TATACOMM", "RAILTEL", "TEJASNET", "SHYAMMETL",
    "RAINBOW", "MAXHEALTH", "FORTIS", "MEDPLUS", "GLOBALHLTH",
    "METROPOLIS", "DRLAL", "VIJAYA", "AAVAS", "HOMEFIRST",
    "LATENTVIEW", "MAPMYINDIA", "EASEMYTRIP", "ROUTE", "LICHSGFIN",
]


NIFTY_200_FALLBACK: List[str] = sorted(set(NIFTY_50_FALLBACK + NIFTY_200_EXTRAS))


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LiquidUniverseConfig:
    """Universe selection parameters.

    top_n:
        Number of symbols to keep. 200 covers >90% of NSE trading
        volume and matches the Nifty 200 index design.

    lookback_days:
        Window for ADV (average daily traded value) calculation.
        30 = roughly 1 month, the convention for liquidity scoring.

    min_price:
        Filter out penny stocks below this price (₹). Below ₹10 the
        circuit-breaker rules + tick size (₹0.05) make the model's
        signals unreliable.

    min_avg_volume:
        Filter symbols with median daily volume < this. 100,000 shares
        is the floor for "tradeable without market impact" at retail
        size. AutoPilot needs more (1M+); set per use-case.

    candidate_pool:
        Where to draw the universe from. By default we start from the
        Nifty 200 fallback list and rank by ADV. To search wider,
        pass a custom pool (e.g. all NSE F&O stocks).

    as_of_date:
        Point-in-time anchor (PR 177). When set, the lookback window
        ends at this date instead of today. Symbols delisted on or
        before this date are excluded; symbols that were tradeable at
        this date but have since delisted are reincluded — neutralizing
        survivorship bias in historical backtests. Pass None for "use
        today" (live trading / current-universe selection).
    """

    top_n: int = 500
    lookback_days: int = 30
    min_price: float = 10.0
    min_avg_volume: int = 100_000
    candidate_pool: Optional[List[str]] = None
    as_of_date: Optional[DateLike] = None
    strict: bool = False
    """When True, raise RuntimeError instead of silently returning
    NIFTY_200_FALLBACK if the data source yields nothing (audit fix).
    Trainers should pass strict=True; dev/notebook callers may use the
    lenient default."""


# Module-level cache. Key = (top_n, lookback_days, frozenset(candidate_pool)).
_UNIVERSE_CACHE: dict = {}


def _load_tier_pool() -> Optional[list[str]]:
    """PR 207 — load from data/nse_tiers/nifty500.txt (broadest tier).

    Returns None if the tier file is missing so callers can fall back
    to NIFTY_200_FALLBACK. Returns a deduplicated, uppercased list
    when found.
    """
    from pathlib import Path  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "nse_tiers" / "nse_all.txt",
        repo_root / "data" / "nse_tiers" / "nifty500.txt",
        repo_root / "data" / "nse_tiers" / "nifty250.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            symbols: list[str] = []
            seen = set()
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip().upper()
                if line and line not in seen:
                    symbols.append(line)
                    seen.add(line)
            if symbols:
                logger.info(
                    "liquid_universe: loaded %d symbols from %s",
                    len(symbols), path.name,
                )
                return symbols
        except Exception as exc:  # noqa: BLE001
            logger.debug("tier file %s read failed: %s", path, exc)
    return None


def liquid_universe(cfg: Optional[LiquidUniverseConfig] = None) -> List[str]:
    """Return the top-N most-liquid NSE symbols by 30-day median ADV.

    Sorted by liquidity descending. Cached per (top_n, lookback_days,
    candidate_pool) for the lifetime of the Python process.

    Returns the static NIFTY_200_FALLBACK[:top_n] if yfinance batch
    download fails or returns insufficient data.
    """
    cfg = cfg or LiquidUniverseConfig()
    as_of = _to_date(cfg.as_of_date)

    # PR 207 — broader candidate pool. Try the tier files first
    # (Nifty 500 = ~500 names across all sectors), fall back to
    # NIFTY_200_FALLBACK only if tier files are missing.
    if cfg.candidate_pool is not None:
        base_pool = list(cfg.candidate_pool)
    else:
        base_pool = _load_tier_pool() or list(NIFTY_200_FALLBACK)
    # PR 177 — survivorship-tolerant pool. When as_of is set, drop
    # symbols delisted by then; add back symbols that were tradeable
    # then but have since delisted.
    if as_of is not None:
        base_pool = [s for s in base_pool if was_listed_at(s, as_of)]
        for extra in historical_universe_extras(as_of):
            if extra not in base_pool:
                base_pool.append(extra)

    pool = base_pool
    cache_key = (cfg.top_n, cfg.lookback_days, frozenset(pool), as_of)
    if cache_key in _UNIVERSE_CACHE:
        return list(_UNIVERSE_CACHE[cache_key])

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        logger.warning("yfinance unavailable; returning static fallback")
        result = pool[: cfg.top_n]
        _UNIVERSE_CACHE[cache_key] = result
        return list(result)

    tickers = [f"{s}.NS" for s in pool]
    lookback_window = max(cfg.lookback_days + 5, 35)
    # PR 177 — when as_of is set, fetch a fixed end-anchored window so
    # we measure liquidity AS OF that date (not as of today). Live use
    # passes as_of=None and we use the period= API.
    yf_kwargs: dict = dict(
        progress=False, auto_adjust=True, group_by="ticker", threads=True,
        interval="1d",
    )
    if as_of is not None:
        yf_kwargs["end"] = (as_of + timedelta(days=1)).isoformat()
        yf_kwargs["start"] = (as_of - timedelta(days=lookback_window + 7)).isoformat()
    else:
        yf_kwargs["period"] = f"{lookback_window}d"

    try:
        data = yf.download(tickers, **yf_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance batch download failed: %s — using fallback", exc)
        if cfg.strict:
            raise RuntimeError(
                "liquid universe build failed: yfinance batch download raised an "
                "exception and strict=True (refusing silent NIFTY-200 fallback — "
                "audit fix)"
            ) from exc
        result = pool[: cfg.top_n]
        _UNIVERSE_CACHE[cache_key] = result
        return list(result)

    if data is None or len(data) == 0:
        logger.warning("yfinance returned empty frame — using fallback")
        if cfg.strict:
            raise RuntimeError(
                "liquid universe build failed: data source returned nothing "
                "and strict=True (refusing silent NIFTY-200 fallback — audit fix)"
            )
        result = pool[: cfg.top_n]
        _UNIVERSE_CACHE[cache_key] = result
        return list(result)

    # Compute median ADV per symbol over the lookback window. When as_of
    # is set, we already fetched only data up to as_of_date.
    rows = []
    delisted_with_data = 0
    for sym in pool:
        ticker = f"{sym}.NS"
        try:
            sym_df = data[ticker].dropna(subset=["Close", "Volume"])
        except (KeyError, AttributeError):
            continue
        if len(sym_df) < cfg.lookback_days // 2:
            # Not enough data — common for delisted symbols. Log and skip.
            if as_of is not None and not was_listed_at(sym, as_of - timedelta(days=cfg.lookback_days)):
                delisted_with_data += 1
            continue
        recent = sym_df.tail(cfg.lookback_days)
        median_close = float(recent["Close"].median())
        median_volume = float(recent["Volume"].median())
        if median_close < cfg.min_price:
            continue
        if median_volume < cfg.min_avg_volume:
            continue
        median_adv = median_close * median_volume
        rows.append({"symbol": sym, "adv": median_adv})

    if not rows:
        logger.warning("no symbols passed liquidity filter — using fallback")
        if cfg.strict:
            raise RuntimeError(
                "liquid universe build failed: no symbols survived the liquidity "
                "filter and strict=True (refusing silent NIFTY-200 fallback — "
                "audit fix)"
            )
        result = pool[: cfg.top_n]
        _UNIVERSE_CACHE[cache_key] = result
        return list(result)

    df = pd.DataFrame(rows).sort_values("adv", ascending=False)
    result = df["symbol"].head(cfg.top_n).tolist()
    if as_of is not None and delisted_with_data:
        logger.info(
            "liquid_universe(as_of=%s): %d delisted symbols in pool had no usable data",
            as_of, delisted_with_data,
        )

    if len(result) < cfg.top_n:
        logger.info(
            "liquid_universe: %d symbols passed filter (requested top_n=%d)",
            len(result), cfg.top_n,
        )

    _UNIVERSE_CACHE[cache_key] = result
    return list(result)


def clear_cache() -> None:
    """Clear the universe cache. Useful in tests + long-running services."""
    _UNIVERSE_CACHE.clear()
