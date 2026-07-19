"""
Strategy: Reversal Patterns (Double Bottom, IH&S, Cup & Handle, Triple Bottom)
================================================================================
Trades high-probability chart reversal patterns detected by swing-point analysis.

Based on Bulkowski's research:
- Double Bottom: 68-78% success rate, avg +40% move
- Inverse Head & Shoulders: 74-89% success rate, avg +38% move
- Cup and Handle: 61-95% success rate, avg +54% move
- Triple Bottom: 74-79% success rate, avg +45% move

Entry Rules:
- Reversal pattern detected with quality >= 70
- Breakout above neckline/resistance with volume > 1.5x avg
- Bullish candle = confidence bonus (not required)
- Prior downtrend validated (10% decline or below 200-EMA at pattern start)
- ADX > 15 (sufficient trending movement for reversal)
- Minimum R:R = 1.0 (loose — breakout + volume are the core signals)

Stop Loss: Below pattern's deepest point - 0.5x ATR (max 6%)
Target: Measured move (pattern depth projected from breakout)
Max Hold: 25 days
"""

from typing import Optional
import pandas as pd

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from ..features.patterns import scan_for_reversal_patterns, detect_reversal_breakout


class ReversalPatterns(BaseStrategy):
    name = "Reversal_Patterns"
    category = "pattern"
    min_bars = 200
    max_hold_bars = 35  # Reversal patterns need longer to play out

    min_quality = 70
    min_volume_mult = 1.5
    interval = '1d'  # Candlestick timeframe — set by scanner

    # Pattern cache: reuse expensive reversal pattern results within 10-bar window
    _CACHE_INTERVAL = 10

    def __init__(self):
        super().__init__()
        self._cache_key = None
        self._cache_patterns = None

    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        bar_idx = len(df) - 1
        curr = df.iloc[bar_idx]
        close = float(curr['close'])

        # Trend context checks (may be bypassed for long accumulation patterns)
        ema200 = curr.get('ema_200', None)
        above_ema200 = not pd.isna(ema200) and close >= ema200

        weekly_aligned = curr.get('weekly_trend_aligned', None)
        has_weekly = not pd.isna(weekly_aligned) and weekly_aligned

        # Detect reversal patterns — cached to avoid recomputation every bar
        cache_key = (symbol, len(df) // self._CACHE_INTERVAL)
        if cache_key != self._cache_key:
            self._cache_patterns = scan_for_reversal_patterns(df, lookback=250, interval=self.interval)
            self._cache_key = cache_key
        patterns = self._cache_patterns

        if not patterns:
            return None

        # Try breakout on each pattern (best quality first)
        patterns.sort(key=lambda p: p.quality_score, reverse=True)

        for pattern in patterns:
            if pattern.quality_score < self.min_quality:
                continue

            # Reversal patterns form in DOWNTRENDS — prior downtrend is
            # already validated by _validate_prior_downtrend() inside
            # detect_double_bottom / detect_triple_bottom / detect_ihs.
            # No uptrend check needed here.
            # Only sanity gate: reject if ADX < 15 (no trend = no reversal)
            adx_val = curr.get('adx', None)
            if adx_val is not None and not pd.isna(adx_val) and adx_val < 15:
                continue  # No trending movement at all — skip

            breakout = detect_reversal_breakout(
                df, bar_idx, pattern, volume_mult=self.min_volume_mult
            )

            if breakout is None:
                continue

            # Verify R:R — loose: only reject terrible setups
            risk = breakout.entry_price - breakout.stop_loss
            if risk <= 0:
                continue
            reward = breakout.target - breakout.entry_price
            if reward / risk < 1.0:
                continue

            # Build signal reasons
            ptype = pattern.pattern_type.replace('_', ' ').title()
            reasons = [
                f"{ptype} ({pattern.duration_bars} bars, quality={pattern.quality_score})",
                f"Neckline breakout above {pattern.neckline_level:.2f}",
                f"Volume {breakout.volume_ratio:.1f}x avg",
                f"Pattern depth {pattern.pattern_height:.2f}",
                f"Target {breakout.target:.2f} (measured move)",
            ]

            # Base confidence from quality score
            confidence = min(80.0, pattern.quality_score * 0.8 + 15)

            # Pivot candle bonus (from candlestick-confirmed pivots)
            if pattern.candle_confirmed_touches > 0:
                pivot_bonus = min(10.0, pattern.candle_confirmed_touches * 5.0)
                confidence = min(90.0, confidence + pivot_bonus)
                reasons.append(
                    f"{pattern.candle_confirmed_touches} candle-confirmed "
                    f"pivot(s) (+{pivot_bonus:.0f})"
                )

            # Volume boost
            if breakout.volume_ratio > 1.8:
                confidence = min(90.0, confidence + 5)
                reasons.append(f"Strong volume ({breakout.volume_ratio:.1f}x)")

            # Neckline breakout candle quality
            has_bullish_breakout = (
                bool(curr.get('candle_engulfing_bull', False))
                or bool(curr.get('candle_three_white_soldiers', False))
                or bool(curr.get('candle_morning_star', False))
                or float(curr.get('body_pct', 0)) > 0.55
            )
            if not has_bullish_breakout:
                confidence -= 5
                reasons.append("Weak breakout candle (-5)")
            elif curr.get('candle_three_white_soldiers', False):
                reasons.append("Three White Soldiers confirmation")
            elif curr.get('candle_engulfing_bull', False):
                reasons.append("Bullish engulfing on breakout")

            # Breakout alpha score (candle type, volume trend, fake detection)
            confidence = min(90.0, confidence + breakout.alpha_score)
            reasons.extend(breakout.alpha_reasons)

            # Pattern-specific bonus (higher WR patterns get slight boost)
            if pattern.pattern_type in ('inverse_head_shoulders', 'cup_and_handle'):
                confidence = min(90.0, confidence + 3)

            # Long accumulation: pattern formed over many bars with high quality
            is_long_accumulation = (pattern.duration_bars > 60 and pattern.quality_score >= 70)

            if has_weekly:
                reasons.append("Weekly trend aligned")
            if is_long_accumulation and not above_ema200:
                reasons.append(f"Long accumulation ({pattern.duration_bars} bars)")

            # Accumulation phase bonus
            if curr.get('accumulation_phase', False):
                confidence = min(90.0, confidence + 3)
                reasons.append("Accumulation phase detected")

            # Universal confluence (Golden Cross, ADX, MACD)
            conf_bonus, conf_reasons = self.confluence_bonus(curr)
            confidence = min(90.0, confidence + conf_bonus)
            reasons.extend(conf_reasons)

            # Floor — should never be <40 given quality>=55, but guard edge cases
            confidence = max(40.0, confidence)

            return TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.BUY,
                entry_price=round(breakout.entry_price, 2),
                stop_loss=round(breakout.stop_loss, 2),
                target=round(breakout.target, 2),
                confidence=round(confidence, 1),
                reasons=reasons,
            )

        return None

    def should_exit(
        self, df: pd.DataFrame, position: Position
    ) -> Optional[ExitSignal]:
        if len(df) < 2:
            return None

        i = len(df) - 1
        curr = df.iloc[i]
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

        # 3. Breakeven at 1.5R (was 1R — too early for reversal patterns)
        if (high >= position.entry_price + 1.5 * risk
                and position.stop_loss < position.entry_price):
            position.stop_loss = position.entry_price

        # 4. Trail with 21-EMA after 2.5R profit (swing-appropriate)
        if position.highest_since_entry >= position.entry_price + 2.5 * risk:
            ema21 = curr.get('ema_21', 0)
            if not pd.isna(ema21) and ema21 > 0:
                trail = ema21 - position.entry_price * 0.005  # 0.5% buffer
                if trail > position.stop_loss and trail < close:
                    position.stop_loss = trail

        # 5. Time exit (force close losing positions past max hold)
        time_exit = self._check_time_exit(df, position)
        if time_exit:
            return time_exit

        return None
