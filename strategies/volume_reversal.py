"""
Strategy: Volume Reversal (Volume Climax + Candlestick Reversal)
================================================================
Trades volume-based reversal signals using Wyckoff / VPA concepts.

Concept: When selling pressure exhausts (volume climax sell / stopping volume),
combined with a bullish reversal candle at a support level, a reversal is likely.

Entry Rules:
- Volume climax sell OR stopping volume bull OR bullish absorption detected
- Bullish reversal candle (hammer, engulfing, morning star, pin bar, three white soldiers)
- At or near support level (S/R zone, SMA, or lower Bollinger Band)
- RSI < 45 (confirming oversold/pullback)
- Price above 200-EMA (uptrend context — buying the dip, not catching knife)
- Volume bull score >= 2

Stop Loss: Below climax/reversal candle low - 0.5x ATR (max 5%)
Target: Recent swing high or 2x risk (whichever is higher)
Max Hold: 15 days
"""

from typing import Optional
import pandas as pd

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from ..features.indicators import detect_support_resistance


class VolumeReversal(BaseStrategy):
    name = "Volume_Reversal"
    category = "volume"
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
        low = float(curr['low'])

        # --- Uptrend context ---
        ema200 = curr.get('ema_200', None)
        if pd.isna(ema200) or close < ema200:
            return None

        # --- Volume signal detection ---
        # Check current bar AND previous bar (volume climax often precedes reversal)
        vol_climax_sell = bool(curr.get('volume_climax_sell', False))
        stopping_bull = bool(curr.get('stopping_volume_bull', False))
        absorption_bull = bool(curr.get('vpa_absorption_bull', False))
        no_supply = bool(curr.get('vpa_no_supply', False))

        # Also check previous bar for climax (reversal candle comes after)
        # NaN-safe: bool(NaN) is True in Python, so guard with pd.isna check
        if i >= 1:
            prev = df.iloc[i - 1]
            pv = prev.get('volume_climax_sell', False)
            vol_climax_sell = vol_climax_sell or (not pd.isna(pv) and bool(pv))
            pv = prev.get('stopping_volume_bull', False)
            stopping_bull = stopping_bull or (not pd.isna(pv) and bool(pv))
            pv = prev.get('vpa_absorption_bull', False)
            absorption_bull = absorption_bull or (not pd.isna(pv) and bool(pv))
            pv = prev.get('vpa_no_supply', False)
            no_supply = no_supply or (not pd.isna(pv) and bool(pv))

        # Need at least one volume signal
        has_volume_signal = vol_climax_sell or stopping_bull or absorption_bull or no_supply
        if not has_volume_signal:
            return None

        # --- Bullish reversal candle on current bar ---
        is_hammer = bool(curr.get('candle_hammer', False))
        is_engulfing = bool(curr.get('candle_engulfing_bull', False))
        is_morning_star = bool(curr.get('candle_morning_star', False))
        is_pin = bool(curr.get('candle_bullish_pin', False))
        is_three_ws = bool(curr.get('candle_three_white_soldiers', False))
        is_piercing = bool(curr.get('candle_piercing_line', False))
        is_harami_bull = bool(curr.get('candle_harami_bull', False))
        is_tweezer_bot = bool(curr.get('candle_tweezer_bottom', False))

        # Also accept a strong bullish close as reversal confirmation
        # (volume climax + strong bullish close is a valid reversal signal even without named pattern)
        close_position = (close - low) / max(float(curr['high']) - low, 0.01)
        is_strong_bullish = (close > float(curr['open'])) and close_position > 0.60 and float(curr.get('body_pct', 0)) > 0.45

        has_reversal_candle = (
            is_hammer or is_engulfing or is_morning_star
            or is_pin or is_three_ws or is_piercing
            or is_harami_bull or is_tweezer_bot
            or is_strong_bullish
        )

        if not has_reversal_candle:
            return None

        # --- RSI confirmation: pullback zone, not overbought ---
        rsi = curr.get('rsi_14', None)
        if pd.isna(rsi) or rsi > 55:
            return None

        # --- At/near support level ---
        sma20 = curr.get('sma_20', None)
        sma50 = curr.get('sma_50', None)
        bb_lower = curr.get('bb_lower', None)

        at_support = False
        support_name = ""

        # Check S/R levels — cached to avoid KDE recomputation every bar
        if 'swing_low' in df.columns:
            cache_key = (symbol, len(df) // self._CACHE_INTERVAL)
            if cache_key != self._cache_key:
                self._cache_sr = detect_support_resistance(df.iloc[:i + 1], lookback=60)
                self._cache_key = cache_key
            supports, _ = self._cache_sr
            for level in supports:
                if abs(low - level) / level < 0.02:
                    at_support = True
                    support_name = f"S/R {level:.2f}"
                    break

        # Check moving averages as support (2% tolerance — real pullbacks approach but don't always touch)
        if not at_support and sma20 is not None and not pd.isna(sma20):
            if low <= sma20 * 1.02 and close > sma20:
                at_support = True
                support_name = f"20-SMA ({sma20:.2f})"

        if not at_support and sma50 is not None and not pd.isna(sma50):
            if low <= sma50 * 1.02 and close > sma50:
                at_support = True
                support_name = f"50-SMA ({sma50:.2f})"

        if not at_support and bb_lower is not None and not pd.isna(bb_lower):
            if low <= bb_lower * 1.025:
                at_support = True
                support_name = f"Lower BB ({bb_lower:.2f})"

        if not at_support:
            return None

        # --- Stop and target ---
        atr = curr.get('atr_14', close * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        stop = low - 0.5 * atr

        # Cap stop at 5% below entry
        if stop < close * 0.95:
            stop = close * 0.95

        risk = close - stop
        if risk <= 0:
            return None

        # Target: recent swing high or 2x risk
        recent_high = self._recent_swing_high(df, i, lookback=30)
        target = max(recent_high, close + 2.0 * risk)

        # Minimum R:R check
        reward = target - close
        if risk > 0 and reward / risk < 1.5:
            target = close + 1.5 * risk

        # --- Build signal ---
        candle_types = []
        if is_engulfing:
            candle_types.append("Engulfing")
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
        if is_strong_bullish and not candle_types:
            candle_types.append("Strong Bullish Close")
        candle_label = " + ".join(candle_types)

        volume_signals = []
        if vol_climax_sell:
            volume_signals.append("Volume Climax (sell exhaustion)")
        if stopping_bull:
            volume_signals.append("Stopping Volume")
        if absorption_bull:
            volume_signals.append("Bullish Absorption")
        if no_supply:
            volume_signals.append("No Supply")
        vol_label = " + ".join(volume_signals)

        reasons = [
            f"{candle_label} reversal at {support_name}",
            f"Volume: {vol_label}",
            f"RSI={rsi:.1f}",
            f"Above 200-EMA ({ema200:.2f})",
        ]

        # Confidence scoring
        confidence = 55.0

        # Volume signal strength
        if vol_climax_sell:
            confidence += 12
        if stopping_bull:
            confidence += 8
        if absorption_bull:
            confidence += 6

        # Candle strength
        if is_engulfing:
            confidence += 8
        if is_three_ws:
            confidence += 10
        if is_morning_star:
            confidence += 8
        if is_hammer or is_pin:
            confidence += 5
        if is_piercing:
            confidence += 5
        if is_harami_bull:
            confidence += 5
        if is_strong_bullish and not (is_engulfing or is_three_ws or is_morning_star or is_hammer or is_pin):
            confidence += 3  # Weaker than named patterns but still valid with volume confirmation
        if is_tweezer_bot:
            confidence += 5

        # RSI oversold bonus
        if rsi < 35:
            confidence += 5

        # Accumulation phase bonus
        if curr.get('accumulation_phase', False):
            confidence += 4
            reasons.append("Accumulation phase")

        # Weekly trend
        weekly_aligned = curr.get('weekly_trend_aligned', None)
        if not pd.isna(weekly_aligned) and weekly_aligned:
            confidence += 3
            reasons.append("Weekly trend aligned")

        # Universal confluence (Golden Cross, ADX, MACD)
        conf_bonus, conf_reasons = self.confluence_bonus(curr)
        confidence += conf_bonus
        reasons.extend(conf_reasons)

        confidence = min(90.0, confidence)

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

        # 4. Trail after 1.5R using 9-EMA
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
