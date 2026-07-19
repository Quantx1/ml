"""
================================================================================
QUANT X ML MODULE
================================================================================
Production algo swing trading bot with 6 rule-based strategies.

STRUCTURE:
├── features/
│   ├── indicators.py           - All technical indicators (40+ columns)
│   ├── patterns.py             - Chart pattern detection (zigzag, trendlines)
│   └── volume_analysis.py      - Wyckoff / VPA volume signals
├── strategies/
│   ├── base.py                 - BaseStrategy ABC + TradeSignal dataclass
│   ├── consolidation_breakout.py - Chart pattern breakouts (triangle, flag, wedge)
│   ├── trend_pullback.py       - MA pullback in weekly uptrend
│   ├── candle_reversal.py      - Candlestick reversals at key support
│   ├── bos_structure.py        - Break of Structure / CHOCH (SMC)
│   ├── reversal_patterns.py    - Double Bottom, IH&S, Cup & Handle
│   └── volume_reversal.py      - Volume climax + reversal candle (Wyckoff/VPA)
├── scanner.py                  - Stock universe scanner
├── risk_manager.py             - Position sizing and risk limits
└── backtest/
    └── engine.py               - Production backtesting

USAGE:
    from ml.scanner import scan_stock, get_all_strategies
    from ml.backtest.engine import BacktestEngine
================================================================================
"""

__version__ = '2.0.0'
