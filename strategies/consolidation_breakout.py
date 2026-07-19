"""
Strategy: Consolidation Breakout
=================================
Simple approach:
1. Stock must be in uptrend (close > SMA50)
2. Pattern engine detects consolidation + breakout (neurotrader888 code)
3. ML meta-labeler filters weak breakouts (optional, ~+8% win rate)
4. Entry/stop/target from pattern engine's compute_trade_levels()

Patterns: ascending triangle, horizontal channel, symmetrical triangle,
          falling wedge, bull flag, bull pennant

Max Hold: 40 bars
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from ..features.patterns import (
    scan_for_patterns, detect_breakout,
    BreakoutMetaLabeler, walkforward_train,
)


class ConsolidationBreakout(BaseStrategy):
    name = "Consolidation_Breakout"
    category = "pattern"
    min_bars = 200
    max_hold_bars = 40

    interval = '1d'

    _CACHE_INTERVAL = 10

    def __init__(self, ml_labeler: BreakoutMetaLabeler = None):
        super().__init__()
        self._cache_key = None
        self._cache_patterns = None
        self._active_pattern_type = None
        self.ml_labeler = ml_labeler

    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        bar_idx = len(df) - 1
        curr = df.iloc[bar_idx]
        close = float(curr['close'])

        # --- Gate 1: Uptrend context ---
        # Stock must be above SMA50 (intermediate uptrend)
        sma50 = curr.get('sma_50', None)
        if sma50 is not None and not pd.isna(sma50):
            if close < sma50:
                return None

        # RSI hard cap — don't buy extremely overbought
        rsi = curr.get('rsi_14', None)
        if rsi is not None and not pd.isna(rsi) and rsi > 78:
            return None

        # --- Gate 2: Pattern detection (neurotrader888 engine) ---
        cache_key = (symbol, len(df) // self._CACHE_INTERVAL)
        if cache_key != self._cache_key:
            self._cache_patterns = scan_for_patterns(
                df, lookback=250, min_duration=15, min_touches=3,
                interval=self.interval,
            )
            self._cache_key = cache_key
        patterns = self._cache_patterns

        if not patterns:
            return None

        # Try each pattern (best quality first)
        for pattern in patterns:
            # Only consolidation patterns (have trendlines)
            if pattern.support_line is None or pattern.resistance_line is None:
                continue

            # Minimum quality gate
            if pattern.quality_score < 55:
                continue

            # Get the breakout signal (already confirmed by pattern engine)
            breakout = detect_breakout(df, bar_idx, pattern)
            if breakout is None:
                continue

            # Basic R:R check
            risk = breakout.entry_price - breakout.stop_loss
            if risk <= 0:
                continue
            reward = breakout.target - breakout.entry_price
            rr = reward / risk
            if rr < 1.0:
                continue

            # --- Gate 3: ML meta-labeler filter (optional) ---
            ml_score = -1.0
            if self.ml_labeler and self.ml_labeler.is_trained:
                ml_score = self.ml_labeler.score_signal(df, breakout)
                if 0 <= ml_score < 0.35:
                    continue  # ML says this breakout will likely fail

            # Build reasons
            reasons = [
                f"{pattern.pattern_type.replace('_', ' ').title()} "
                f"({pattern.duration_bars} bars, Q={pattern.quality_score:.0f})",
                f"Breakout above {pattern.breakout_level:.2f}",
                f"Volume {breakout.volume_ratio:.1f}x avg",
                f"R:R {rr:.1f}",
            ]

            # Confidence: start from pattern quality, add bonuses
            confidence = 40.0 + pattern.quality_score * 0.3

            # Volume bonus
            if breakout.volume_ratio > 1.5:
                confidence += min(8.0, (breakout.volume_ratio - 1.5) * 4.0)
                reasons.append(f"Strong volume (+{min(8.0, (breakout.volume_ratio - 1.5) * 4.0):.0f})")

            # ML score bonus
            if ml_score >= 0.5:
                confidence += 10
                reasons.append(f"ML score {ml_score:.2f} (+10)")
            elif ml_score >= 0.35:
                confidence += 5
                reasons.append(f"ML score {ml_score:.2f} (+5)")

            # Confluence bonus (Golden Cross, ADX, MACD)
            conf_bonus, conf_reasons = self.confluence_bonus(curr)
            confidence += conf_bonus
            reasons.extend(conf_reasons)

            confidence = min(95.0, confidence)

            self._active_pattern_type = pattern.pattern_type

            return TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.BUY,
                entry_price=breakout.entry_price,
                stop_loss=breakout.stop_loss,
                target=breakout.target,
                confidence=round(confidence, 1),
                reasons=reasons,
            )

        return None

    def should_exit(
        self, df: pd.DataFrame, position: Position
    ) -> Optional[ExitSignal]:
        if len(df) < 2:
            return None

        curr = df.iloc[-1]
        close = float(curr['close'])
        low = float(curr['low'])
        high = float(curr['high'])

        position.highest_since_entry = max(position.highest_since_entry, high)

        risk = position.entry_price - position.stop_loss
        if risk <= 0:
            risk = position.entry_price * 0.03

        # 1. Hard stop
        if low <= position.stop_loss:
            return ExitSignal(reason="stop_loss", exit_price=position.stop_loss)

        # 2. Target hit
        if high >= position.target:
            return ExitSignal(reason="target_hit", exit_price=position.target)

        # 3. Move to breakeven after 2R profit
        if (high >= position.entry_price + 2.0 * risk
                and position.stop_loss < position.entry_price):
            position.stop_loss = position.entry_price

        # 4. Trailing stop using 21-EMA after 3R profit
        if position.highest_since_entry >= position.entry_price + 3.0 * risk:
            ema_val = curr.get('ema_21', 0)
            if not pd.isna(ema_val) and ema_val > 0:
                trail = ema_val - close * 0.005
                if trail > position.stop_loss and trail < close:
                    position.stop_loss = trail

        # 5. Time exit
        time_exit = self._check_time_exit(df, position)
        if time_exit:
            return time_exit

        return None
