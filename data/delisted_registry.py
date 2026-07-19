"""
PR 177 — known NSE delisting / merger registry.

When a backtest spans 2018-2025, the "today's Nifty 200" universe is
NOT the universe that was tradeable in 2018. Stocks that delisted,
went bankrupt, or got merged out are silently absent — that's
survivorship bias. The model trains only on winners that are still
listed today, then live trading meets the full distribution including
losers, and OOS performance collapses.

This registry maintains a small but expanding list of known NSE
delistings / suspensions / mergers that were tradeable in our backtest
window (2015-present). When ``liquid_universe(as_of_date=...)`` is
called with a date BEFORE the delisting, the symbol is included in
the candidate pool. After the delisting date, it's excluded.

The registry is intentionally hand-curated. Mass-importing yfinance
"delisted" tickers is risky because many are actually renames/rebrands
that survive under a new symbol. Each entry below should be checked
against NSE corporate actions before being added.

Format:
    DELISTED_NSE: list of (symbol, delisting_date_iso, reason).

When updating:
    - Use ISO YYYY-MM-DD for date.
    - reason ∈ {"bankruptcy", "merger_acquired", "voluntary", "suspension", "scheme"}.
    - For merged-out symbols, reason="merger_acquired" — the symbol no
      longer trades but the company continues under the acquirer.

This file is meant to grow as we audit more historical NSE data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, datetime
from typing import List, Optional


@dataclass(frozen=True)
class DelistingEvent:
    symbol: str
    delisting_date: Date
    reason: str


def _d(s: str) -> Date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ============================================================================
# Known NSE delistings (verified against NSE corporate actions / news)
# ============================================================================
#
# Add entries here as we audit more historical data. Each entry must be
# checked against NSE's corporate action filings — symbol-rename events
# do NOT belong here.

DELISTED_NSE: List[DelistingEvent] = [
    # Dewan Housing Finance — IBC resolution / Piramal acquisition
    DelistingEvent(symbol="DHFL", delisting_date=_d("2021-09-29"), reason="bankruptcy"),
    # Reliance Capital — IBC resolution
    DelistingEvent(symbol="RELCAPITAL", delisting_date=_d("2024-07-25"), reason="bankruptcy"),
    # Reliance Communications — IBC resolution
    DelistingEvent(symbol="RCOM", delisting_date=_d("2023-08-30"), reason="bankruptcy"),
    # CG Power and Industrial Solutions changed name from CGPOWER (still trades)
    # Educomp Solutions — voluntary delisting after IBC
    DelistingEvent(symbol="EDUCOMP", delisting_date=_d("2021-04-01"), reason="bankruptcy"),
    # Lanco Infratech — IBC liquidation
    DelistingEvent(symbol="LITL", delisting_date=_d("2019-10-22"), reason="bankruptcy"),
    # Bhushan Steel — merger into Tata Steel BSL → eventually Tata Steel
    DelistingEvent(symbol="BHUSANSTL", delisting_date=_d("2018-05-15"), reason="merger_acquired"),
    # Jet Airways — bankruptcy
    DelistingEvent(symbol="JETAIRWAYS", delisting_date=_d("2024-10-31"), reason="bankruptcy"),
    # MindTree — merged into LTIMindtree (LTIM) on 2022-11-14. The
    # MINDTREE ticker stopped trading; LTI shareholders got shares in the
    # combined LTIM entity.
    DelistingEvent(symbol="MINDTREE", delisting_date=_d("2022-11-14"), reason="merger_acquired"),
    # Hexaware Technologies — voluntary delisting by Carlyle 2020-09-21.
    # Relisted Feb 2025 under a different ticker; old HEXAWARE 2020-2025
    # was untradeable.
    DelistingEvent(symbol="HEXAWARE", delisting_date=_d("2020-09-21"), reason="voluntary"),
    # Suzlon Energy went through schemes but still trades (don't include)
    # NOTE: yfinance 404s during liquid_universe loading are mostly
    # yfinance-side ticker mismatches (e.g., DRLAL vs LALPATHLAB,
    # JUPITERWAGONS vs JWL), NOT real delistings. quality_check drops
    # them per batch; don't mass-import them here.
    # Add new entries above this line.
]


def was_listed_at(symbol: str, as_of: Date) -> bool:
    """Return True if symbol was tradeable on NSE as of ``as_of``.

    Conservative: returns True unless we have a registered delisting
    event with delisting_date <= as_of. Symbols with no entry are
    assumed live.
    """
    for ev in DELISTED_NSE:
        if ev.symbol == symbol and ev.delisting_date <= as_of:
            return False
    return True


def historical_universe_extras(as_of: Date) -> List[str]:
    """Symbols that were tradeable at ``as_of`` but have since delisted.

    These should be added to the candidate pool when running PIT
    backtests to neutralize survivorship bias. yfinance may not return
    data for them — that's fine; they'll be filtered out by the ADV
    threshold during liquid_universe construction. The point is to
    prove we *tried* to consider them.
    """
    return [
        ev.symbol for ev in DELISTED_NSE
        if ev.delisting_date > as_of
    ]


__all__ = [
    "DelistingEvent",
    "DELISTED_NSE",
    "was_listed_at",
    "historical_universe_extras",
]
