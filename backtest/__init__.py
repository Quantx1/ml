"""
Quant X Backtest Module
=======================
Production-grade backtesting with:
- Bar-by-bar simulation (no look-ahead bias)
- Entry at NEXT BAR OPEN
- Slippage, brokerage, STT modeling
- Per-strategy breakdown
"""

from .engine import BacktestEngine, BacktestConfig

__all__ = ['BacktestEngine', 'BacktestConfig']
