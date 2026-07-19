"""
Quant X Stock Scanner
======================
Scans the stock universe for trading setups using all 15 strategies.

Workflow:
1. Fetch OHLCV for each stock in universe
2. Compute all indicators
3. Run each strategy's scan() method
4. Collect and rank all signals
5. Return top signals respecting portfolio limits
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd

from .features.indicators import compute_all_indicators, classify_trend_tier
from .strategies.base import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning the full stock universe."""
    signals: List[TradeSignal]
    stocks_scanned: int
    errors: List[str]


def get_all_strategies(ml_labeler=None):
    """Phase 8: 6 strategies — chart-pattern + candlestick first.

    Pure indicator strategies removed (lagging signals).
    Indicators kept only as confirmation filters in indicators.py.

    Pattern strategies:
    - Consolidation_Breakout: Chart pattern breakouts (triangle, flag, wedge, channel)
    - Trend_Pullback: MA pullback in weekly uptrend
    - Reversal_Patterns: Double Bottom, IH&S, Cup & Handle, Triple Bottom (Bulkowski data)

    Price action strategies:
    - Candle_Reversal: Candlestick reversals at key support

    SMC strategies:
    - BOS_Structure: Break of market structure with candle + weekly filter

    Volume strategies:
    - Volume_Reversal: Volume climax + reversal candle at support (Wyckoff/VPA)
    """
    from .strategies.consolidation_breakout import ConsolidationBreakout
    from .strategies.trend_pullback import TrendPullback
    from .strategies.candle_reversal import CandleReversal
    from .strategies.bos_structure import BOSStructure
    from .strategies.reversal_patterns import ReversalPatterns
    from .strategies.volume_reversal import VolumeReversal

    return [
        ConsolidationBreakout(ml_labeler=ml_labeler),
        TrendPullback(),
        CandleReversal(),
        BOSStructure(),
        ReversalPatterns(),
        VolumeReversal(),
    ]


# Cap-specific market regime indices (TrueData-style bare names)
REGIME_INDICES = {
    'large': 'NIFTY 50',
    'mid': 'NIFTY MIDCAP 150',
    'small': 'NIFTY SMLCAP 100',
}


def get_market_regime() -> dict:
    """Check market regime for each cap category using 200-day SMA.

    Returns:
        Dict like {'large': True, 'mid': False, 'small': False}
        True = bullish (index above 200-SMA), False = bearish.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
    from data_provider import get_provider

    provider = get_provider()
    regimes = {}
    for cap, index_name in REGIME_INDICES.items():
        try:
            idx = provider.get_historical(index_name, period="1y", interval="1d")
            if idx is not None and not idx.empty:
                # Normalize columns
                idx.columns = [c.lower() if isinstance(c, str) else c for c in idx.columns]
                if isinstance(idx.columns, pd.MultiIndex):
                    idx.columns = [c[0] if isinstance(c, tuple) else c for c in idx.columns]
                if len(idx) >= 200:
                    sma200 = float(idx['close'].rolling(200).mean().iloc[-1])
                    current = float(idx['close'].iloc[-1])
                    regimes[cap] = current > sma200
                    status = "BULLISH" if regimes[cap] else "BEARISH"
                    logger.info(f"Market regime {cap}: {status} (price={current:.0f}, SMA200={sma200:.0f})")
                else:
                    regimes[cap] = True
            else:
                regimes[cap] = True
        except Exception as e:
            logger.warning(f"Failed to check regime for {cap} ({index_name}): {e}")
            regimes[cap] = True

    return regimes


# Strategy tier classification for trend gate routing
TIER1_ONLY = {'Consolidation_Breakout', 'Trend_Pullback', 'Candle_Reversal', 'BOS_Structure'}
TIER2_STRATEGIES = {'Reversal_Patterns', 'Volume_Reversal'}

MIN_CONFIDENCE = 65.0  # Only accept high-confidence signals


def scan_stock(
    df: pd.DataFrame,
    symbol: str,
    strategies: Optional[list] = None,
    min_confidence: float = MIN_CONFIDENCE,
    interval: str = '1d',
) -> List[TradeSignal]:
    """
    Scan a single stock with all strategies.

    Args:
        df: Raw OHLCV DataFrame (open, high, low, close, volume)
        symbol: Stock symbol
        strategies: List of strategy instances (auto-created if None)
        min_confidence: Minimum confidence to accept a signal
        interval: Candlestick timeframe ('1d', '1wk', '4h', '1h', '15m')

    Returns:
        List of TradeSignal from any strategy that found a setup.
    """
    if strategies is None:
        strategies = get_all_strategies()

    # Propagate timeframe to all strategies
    for s in strategies:
        if hasattr(s, 'interval'):
            s.interval = interval

    if len(df) < 200:
        return []

    # Compute all indicators once
    try:
        df = compute_all_indicators(df)
    except Exception as e:
        logger.error(f"Indicator computation failed for {symbol}: {e}")
        return []

    # Two-tier trend gate: classify before running strategies
    tier = classify_trend_tier(df)

    signals = []
    for strategy in strategies:
        # Tier gate: route strategies by trend state
        if tier == 'skip' and strategy.name not in TIER2_STRATEGIES:
            continue  # Downtrend: only reversal + volume strategies
        if tier == 'tier2' and strategy.name in TIER1_ONLY:
            continue  # Base formation: skip continuation strategies

        try:
            signal = strategy.scan(df, symbol=symbol)
            if signal is not None and signal.confidence >= min_confidence:
                # Tag signal with trend tier for transparency
                signal.reasons.append(f"[Trend: {tier.upper()}]")
                signals.append(signal)
        except Exception as e:
            logger.warning(f"Strategy {strategy.name} failed on {symbol}: {e}")

    return signals


def scan_universe(
    stock_data: Dict[str, pd.DataFrame],
    strategies: Optional[list] = None,
    max_signals: int = 20,
) -> ScanResult:
    """
    Scan entire stock universe for trading setups.

    Args:
        stock_data: Dict mapping symbol -> OHLCV DataFrame
        strategies: Strategy instances (auto-created if None)
        max_signals: Maximum signals to return

    Returns:
        ScanResult with sorted signals (best first)
    """
    if strategies is None:
        strategies = get_all_strategies()

    all_signals = []
    errors = []
    scanned = 0

    for symbol, df in stock_data.items():
        try:
            signals = scan_stock(df, symbol, strategies)
            all_signals.extend(signals)
            scanned += 1
        except Exception as e:
            errors.append(f"{symbol}: {str(e)}")
            logger.error(f"Failed to scan {symbol}: {e}")

    # Sort by confidence (highest first), then risk:reward
    all_signals.sort(key=lambda s: (s.confidence, s.risk_reward), reverse=True)

    # Limit to max signals
    all_signals = all_signals[:max_signals]

    return ScanResult(
        signals=all_signals,
        stocks_scanned=scanned,
        errors=errors,
    )
