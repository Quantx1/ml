"""
Quant X Strategies Module
==========================
6 rule-based swing trading strategies, each standalone.

Categories:
- Pattern/Breakout: Consolidation Breakout (triangle, flag, wedge, channel)
- Trend: Trend Pullback (MA pullback in weekly uptrend)
- Price Action: Candle Reversal (candlestick reversals at key support)
- SMC: BOS Structure (Break of Structure / Change of Character)
- Reversal: Reversal Patterns (Double Bottom, IH&S, Cup & Handle, Triple Bottom)
- Volume: Volume Reversal (volume climax + reversal candle, Wyckoff/VPA)
"""

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from .options_base import (
    BaseOptionsStrategy, OptionsChainSnapshot, OptionsTradeSignal,
    FOLeg, OptionSnapshot,
)

__all__ = [
    'BaseStrategy',
    'TradeSignal',
    'ExitSignal',
    'Position',
    'Direction',
    'BaseOptionsStrategy',
    'OptionsChainSnapshot',
    'OptionsTradeSignal',
    'FOLeg',
    'OptionSnapshot',
]
