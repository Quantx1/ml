"""
================================================================================
Options Strategy Base Classes
================================================================================
Abstract base + data contracts for all options strategies.
Parallels ml.strategies.base (equity) but handles multi-leg, Greeks, chain data.
================================================================================
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


# ============================================================================
# DATA CONTRACTS
# ============================================================================

@dataclass
class OptionSnapshot:
    """Single option contract snapshot from chain data."""
    strike: float
    option_type: str  # 'CE' or 'PE'
    expiry: date
    ltp: float
    bid: float = 0.0
    ask: float = 0.0
    iv: float = 0.0
    oi: int = 0
    oi_change: int = 0
    volume: int = 0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class OptionsChainSnapshot:
    """Complete options chain snapshot for a symbol at a point in time."""
    symbol: str
    spot_price: float
    atm_strike: float
    strike_gap: float         # e.g. 50 for NIFTY, 100 for BANKNIFTY
    lot_size: int
    expiry: date
    chain: List[OptionSnapshot]
    iv_index: float = 0.0     # VIX or ATM IV
    pcr: float = 1.0          # Put-Call Ratio (OI based)
    timestamp: Optional[datetime] = None

    def get_contract(self, strike: float, option_type: str) -> Optional[OptionSnapshot]:
        """Find a specific contract in the chain."""
        for c in self.chain:
            if c.strike == strike and c.option_type == option_type:
                return c
        return None

    def calls(self) -> List[OptionSnapshot]:
        return [c for c in self.chain if c.option_type == 'CE']

    def puts(self) -> List[OptionSnapshot]:
        return [c for c in self.chain if c.option_type == 'PE']


@dataclass
class FOLeg:
    """Single leg of an options trade."""
    symbol: str
    strike: float
    option_type: str       # 'CE' or 'PE'
    direction: str         # 'BUY' or 'SELL'
    lots: int = 1
    entry_price: float = 0.0


@dataclass
class OptionsTradeSignal:
    """Multi-leg options trade signal output."""
    strategy: str
    symbol: str
    legs: List[FOLeg]
    net_premium: float         # +ve = credit, -ve = debit
    max_profit: float
    max_loss: float
    confidence: float = 0.0   # 0-100
    reasons: List[str] = field(default_factory=list)
    margin_required: float = 0.0
    hold_type: str = 'intraday'  # intraday, overnight, expiry, carry


@dataclass
class ExitSignal:
    """Signal to close an options position."""
    reason: str     # sl_hit, trailing_sl, target_hit, eod_exit, overnight_exit, etc.
    exit_price: float = 0.0


# ============================================================================
# HELPERS
# ============================================================================

def normalize_percentile(value: float, history: Optional[List[float]] = None,
                         default: float = 0.5) -> float:
    """Normalize a value to 0-1 range using sigmoid if no history."""
    if history and len(history) > 5:
        below = sum(1 for h in history if h <= value)
        return below / len(history)
    # Sigmoid fallback: maps any real number to (0, 1)
    return 1.0 / (1.0 + math.exp(-value))


def avg(values: List[float]) -> float:
    """Safe average."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseOptionsStrategy(ABC):
    """
    Abstract base for all options strategies.

    Each strategy must implement:
    - scan(): Analyze chain data → OptionsTradeSignal or None
    - should_exit(): Check if position should close → ExitSignal or None
    """

    name: str = "BaseOptionsStrategy"
    category: str = "options"
    template_slug: str = "base"

    @abstractmethod
    def scan(self, chain: OptionsChainSnapshot, params: Dict) -> Optional[OptionsTradeSignal]:
        """
        Scan options chain for a trading setup.

        Args:
            chain: Current options chain snapshot with all strikes, OI, IV, Greeks.
            params: Strategy parameters from user deployment (sl_pct, target_type, etc.)

        Returns:
            OptionsTradeSignal if setup found, None otherwise.
        """
        ...

    @abstractmethod
    def should_exit(self, chain: OptionsChainSnapshot, position: Dict,
                    params: Dict) -> Optional[ExitSignal]:
        """
        Check if an open options position should be closed.

        Args:
            chain: Current chain snapshot.
            position: Dict with entry details (legs, entry_price, entry_time, etc.)
            params: Strategy parameters.

        Returns:
            ExitSignal if position should close, None to keep holding.
        """
        ...

    @staticmethod
    def is_market_hours(start: time = time(10, 15), end: time = time(14, 15)) -> bool:
        """Check if current time is within trading window."""
        now = datetime.now().time()
        return start <= now <= end

    @staticmethod
    def get_option_ltp(chain: OptionsChainSnapshot, leg: Dict) -> float:
        """Get current LTP for a position leg from chain data."""
        contract = chain.get_contract(leg.get('strike', 0), leg.get('option_type', 'CE'))
        return contract.ltp if contract else 0.0
