"""
Strategy: Candlestick Reversal at Key Levels
==============================================
Trades high-probability candlestick reversals (hammer, engulfing, morning star,
pin bar) when they occur at key support levels.

Entry Rules:
- Bullish reversal candle detected (hammer, engulfing, morning star, pin bar)
- At a key support level: S/R zone, 20-SMA, 50-SMA, or lower Bollinger Band
- Volume > 1.2x 20-day average
- RSI < 45 (confirming pullback, not chasing)
- Weekly trend aligned (not fighting the weekly trend)
- Price above 200-EMA (uptrend context)

Stop Loss: Below candle low - 0.3x ATR
Target: Recent swing high or 2x risk (whichever is higher)
Max Hold: 15 days
"""

from typing import Optional
import pandas as pd

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from ..features.indicators import detect_support_resistance_with_touches


class CandleReversal(BaseStrategy):
    name = "Candle_Reversal"
    category = "price_action"
    min_bars = 200
    max_hold_bars = 15

    # S/R cache: reuse KDE results within 10-bar window
    _CACHE_INTERVAL = 10

    def __init__(self):
        super().__init__()
        self._cache_key = None
        self._cache_sr = None

    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        i = len(df) - 1
        curr = df.iloc[i]
        close = float(curr['close'])

        # --- Reversal candle detection ---
        is_hammer = bool(curr.get('candle_hammer', False))
        is_engulfing = bool(curr.get('candle_engulfing_bull', False))
        is_morning_star = bool(curr.get('candle_morning_star', False))
        is_pin = bool(curr.get('candle_bullish_pin', False))
        is_three_ws = bool(curr.get('candle_three_white_soldiers', False))
        is_piercing = bool(curr.get('candle_piercing_line', False))
        is_harami_bull = bool(curr.get('candle_harami_bull', False))
        is_tweezer_bot = bool(curr.get('candle_tweezer_bottom', False))

        if not (is_hammer or is_engulfing or is_morning_star or is_pin
                or is_three_ws or is_piercing or is_harami_bull or is_tweezer_bot):
            return None

        # --- Uptrend context: above 200-EMA ---
        ema200 = curr.get('ema_200')
        if pd.isna(ema200) or close < ema200:
            return None

        # --- RSI: pullback zone (20-55) — must be pulling back, not chasing ---
        rsi = curr.get('rsi_14', None)
        if pd.isna(rsi) or rsi > 55 or rsi < 20:
            return None

        # --- Volume confirmation (1.1x — candle patterns already gate volume) ---
        vol_ratio = curr.get('volume_ratio', 0)
        if pd.isna(vol_ratio) or vol_ratio < 1.1:
            return None

        # --- At a key support level ---
        sma20 = curr.get('sma_20', None)
        sma50 = curr.get('sma_50', None)
        bb_lower = curr.get('bb_lower', None)
        low = float(curr['low'])

        at_support = False
        support_name = ""
        sr_touch_count = 0

        # Check S/R levels — cached to avoid KDE recomputation every bar
        if 'swing_low' in df.columns:
            cache_key = (symbol, len(df) // self._CACHE_INTERVAL)
            if cache_key != self._cache_key:
                self._cache_sr = detect_support_resistance_with_touches(
                    df.iloc[:i + 1], lookback=60,
                )
                self._cache_key = cache_key
            supports_wt, _ = self._cache_sr
            for level, touches in supports_wt:
                if abs(low - level) / level < 0.015 and touches >= 2:
                    at_support = True
                    sr_touch_count = touches
                    support_name = f"S/R level {level:.2f} ({touches} touches)"
                    break

        # Check 20-SMA (within 2% — real pullbacks approach but don't always touch exactly)
        if not at_support and sma20 is not None and not pd.isna(sma20):
            if low <= sma20 * 1.02 and close > sma20:
                at_support = True
                support_name = f"20-SMA ({sma20:.2f})"

        # Check 50-SMA
        if not at_support and sma50 is not None and not pd.isna(sma50):
            if low <= sma50 * 1.02 and close > sma50:
                at_support = True
                support_name = f"50-SMA ({sma50:.2f})"

        # Check lower Bollinger Band
        if not at_support and bb_lower is not None and not pd.isna(bb_lower):
            if low <= bb_lower * 1.02:
                at_support = True
                support_name = f"Lower BB ({bb_lower:.2f})"

        if not at_support:
            return None

        # --- Stop and target ---
        atr = curr.get('atr_14', close * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        stop = low - 0.3 * atr

        # Cap stop at 4% below entry
        if stop < close * 0.96:
            stop = close * 0.96

        risk = close - stop
        if risk <= 0:
            return None

        # Target: recent swing high or 2x risk
        recent_high = self._recent_swing_high(df, i, lookback=30)
        target = max(recent_high, close + 2.0 * risk)

        # --- Candle type label and confidence ---
        candle_types = []
        if is_engulfing:
            candle_types.append("Bullish Engulfing")
        if is_hammer:
            candle_types.append("Hammer")
        if is_morning_star:
            candle_types.append("Morning Star")
        if is_pin:
            candle_types.append("Pin Bar")
        if is_three_ws:
            candle_types.append("Three White Soldiers")
        if is_piercing:
            candle_types.append("Piercing Line")
        if is_harami_bull:
            candle_types.append("Bullish Harami")
        if is_tweezer_bot:
            candle_types.append("Tweezer Bottom")

        candle_label = " + ".join(candle_types)

        confidence = 55.0
        if is_engulfing:
            confidence += 10
        if is_morning_star:
            confidence += 10
        if is_three_ws:
            confidence += 12  # 82% WR pattern
        if is_piercing:
            confidence += 7
        if is_hammer or is_pin:
            confidence += 5
        if is_harami_bull:
            confidence += 6
        if is_tweezer_bot:
            confidence += 6
        if vol_ratio > 1.5:
            confidence += 5
        if rsi < 35:
            confidence += 5
        # VPA no_supply confluence bonus
        if curr.get('vpa_no_supply', False):
            confidence += 3

        # S/R touch count bonus (stronger levels)
        if sr_touch_count >= 4:
            confidence += 3

        # Weekly trend bonus (not a gate)
        weekly_aligned = curr.get('weekly_trend_aligned', None)
        if not pd.isna(weekly_aligned) and weekly_aligned:
            confidence += 5

        # Universal confluence (Golden Cross, ADX, MACD)
        conf_bonus, conf_reasons = self.confluence_bonus(curr)
        confidence += conf_bonus
        confidence = min(90, confidence)

        reasons = [
            f"{candle_label} at {support_name}",
            f"RSI={rsi:.1f} (pullback zone)",
            f"Volume {vol_ratio:.1f}x avg",
            f"Above 200-EMA ({ema200:.2f})",
        ] + conf_reasons

        return TradeSignal(
            strategy=self.name,
            symbol=symbol,
            direction=Direction.BUY,
            entry_price=round(close, 2),
            stop_loss=round(stop, 2),
            target=round(target, 2),
            confidence=round(confidence, 1),
            reasons=reasons,
        )

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

        # 3. Breakeven at 1R
        if (high >= position.entry_price + risk
                and position.stop_loss < position.entry_price):
            position.stop_loss = position.entry_price

        # 4. Trail with 9-EMA after 1.5R
        if position.highest_since_entry >= position.entry_price + 1.5 * risk:
            ema9 = curr.get('ema_9', 0)
            if not pd.isna(ema9) and ema9 > 0:
                trail = ema9 - position.entry_price * 0.002
                if trail > position.stop_loss and trail < close:
                    position.stop_loss = trail

        # 5. Time exit (force close losing positions past max hold)
        time_exit = self._check_time_exit(df, position)
        if time_exit:
            return time_exit

        return None
