"""
Quant X Strategy Base Classes
=============================
Base interface for all 15 trading strategies.
Each strategy is a standalone trading system: scan → signal → manage → exit.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import pandas as pd


class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradeSignal:
    """Output when a strategy finds a trading setup.

    entry_price is the signal candle's close (the price at signal generation).
    Actual execution happens at NEXT BAR OPEN — the backtest engine and
    position manager handle gap validation, R:R re-check, and slippage.
    """
    strategy: str
    symbol: str
    direction: Direction
    entry_price: float  # Signal candle close (not execution price)
    stop_loss: float    # Technical stop level (absolute)
    target: float       # Technical target level (absolute)
    risk_reward: float = 0.0
    confidence: float = 0.0  # 0-100
    reasons: List[str] = field(default_factory=list)

    def __post_init__(self):
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        if risk > 0:
            self.risk_reward = round(reward / risk, 2)


@dataclass
class ExitSignal:
    """Output when a strategy says to close a position."""
    reason: str  # "stop_loss", "target_hit", "trailing_stop", "time_exit", "signal_exit"
    exit_price: float


@dataclass
class Position:
    """Tracks an open position for position management."""
    symbol: str
    strategy: str
    direction: Direction
    entry_price: float
    entry_date: pd.Timestamp
    stop_loss: float
    target: float
    quantity: int = 0
    trailing_stop: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float('inf')
    hold_days: int = 0


class BaseStrategy(ABC):
    """
    Abstract base for all 15 strategies.

    Each strategy must implement:
    - scan(): Look at current data and return TradeSignal or None
    - should_exit(): Check if an open position should be closed
    """

    name: str = "BaseStrategy"
    category: str = "unknown"
    min_bars: int = 200  # minimum historical bars needed
    max_hold_bars: int = 25  # max bars before time exit (if losing)

    @abstractmethod
    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        """
        Scan stock data for a trading setup.

        Args:
            df: DataFrame with OHLCV + indicator columns (lowercase).
                Must have columns: open, high, low, close, volume + computed indicators.
            symbol: Stock symbol for the signal.

        Returns:
            TradeSignal if setup found, None otherwise.
        """
        ...

    @abstractmethod
    def should_exit(self, df: pd.DataFrame, position: Position) -> Optional[ExitSignal]:
        """
        Check if an open position should be closed.

        Args:
            df: Current DataFrame with OHLCV + indicators.
            position: The open position to evaluate.

        Returns:
            ExitSignal if position should close, None to keep holding.
        """
        ...

    def _recent_swing_low(self, df: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """Find the lowest low in the last `lookback` bars."""
        start = max(0, idx - lookback)
        return df['low'].iloc[start:idx + 1].min()

    def _recent_swing_high(self, df: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """Find the highest high in the last `lookback` bars."""
        start = max(0, idx - lookback)
        return df['high'].iloc[start:idx + 1].max()

    def _check_time_exit(self, df: pd.DataFrame, position: 'Position') -> Optional[ExitSignal]:
        """Force exit losing positions held beyond max_hold_bars."""
        if position.hold_days >= self.max_hold_bars:
            close = float(df.iloc[-1]['close'])
            if close <= position.entry_price:
                return ExitSignal(reason="time_exit", exit_price=close)
        return None

    @staticmethod
    def confluence_bonus(curr) -> tuple:
        """Compute universal confluence bonus from Golden Cross, ADX, MACD.

        Returns (bonus_score, reason_list). Adds +2 to +8 confidence points.
        No mandatory gates — purely optional bonus.
        """
        import pandas as _pd
        score = 0.0
        reasons = []

        # Golden Cross active (50-SMA > 200-SMA) → trend confirmed
        gc = curr.get('golden_cross', None)
        if gc is not None and not _pd.isna(gc) and gc:
            score += 3.0
            reasons.append("Golden Cross")

        # ADX > 25 → strong trend
        adx = curr.get('adx', None)
        if adx is not None and not _pd.isna(adx) and adx > 25:
            score += 3.0
            reasons.append(f"ADX {adx:.0f}")

        # MACD histogram positive → momentum bullish
        macd_h = curr.get('macd_hist', None)
        if macd_h is not None and not _pd.isna(macd_h) and macd_h > 0:
            score += 2.0
            reasons.append("MACD bullish")

        return score, reasons
