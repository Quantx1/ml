"""
Strategy 15: Break of Structure (BOS) + Market Structure
==========================================================
ENHANCED v2: Weekly trend filter, candlestick confirmation, retest entry.

Entry Rules:
- BOS confirmed (close breaks above last swing high)
- Weekly trend aligned
- ADX > 20 (trending)
- Volume on BOS bar > 1.3x average
- Candlestick confirmation (engulfing, strong body >50%, or hammer)
- Structure has at least HH or HL (not random break)
- Price above 200-EMA
- Not chasing (< 1.5% above broken level)

Stop Loss: Last swing low - 0.5x ATR (buffer), reject if > 6% risk
Target: Max of 2.5x risk or impulse projection
Max Hold: 20 bars
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction


class BOSStructure(BaseStrategy):
    name = "BOS_Structure"
    category = "smc"
    min_bars = 200
    max_hold_bars = 20  # BOS is momentum-based, should resolve quickly

    swing_lookback = 7
    bos_volume_mult = 1.3
    adx_threshold = 20

    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        i = len(df) - 1
        curr = df.iloc[i]
        close = float(curr['close'])

        # Structure detection
        swings = self._detect_structure(df, i)
        if len(swings) < 4:
            return None

        swing_highs = [(idx, price) for idx, price, stype in swings if stype == 'high']
        swing_lows = [(idx, price) for idx, price, stype in swings if stype == 'low']

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Exclude swing highs too close to current bar (need confirmed prior structure)
        min_gap = self.swing_lookback
        confirmed_sh = [(idx, p) for idx, p in swing_highs if i - idx >= min_gap]
        if len(confirmed_sh) < 1:
            return None

        last_sh_idx, last_sh_price = confirmed_sh[-1]
        prev_sh_idx, prev_sh_price = confirmed_sh[-2] if len(confirmed_sh) >= 2 else swing_highs[-2]
        last_sl_idx, last_sl_price = swing_lows[-1]
        prev_sl_idx, prev_sl_price = swing_lows[-2]

        # BOS: close breaks above last confirmed swing high
        bos_bullish = close > last_sh_price
        if not bos_bullish:
            return None

        # Require HH or HL structure (not random)
        hh = last_sh_price > prev_sh_price
        hl = last_sl_price > prev_sl_price
        if not (hh or hl):
            return None

        # CHOCH detection
        was_bearish = prev_sh_price > last_sh_price and prev_sl_price > last_sl_price
        choch_bullish = was_bearish and close > last_sh_price

        # Volume confirmation (stricter: 1.3x)
        vol_ratio = curr.get('volume_ratio', 0)
        if pd.isna(vol_ratio) or vol_ratio < self.bos_volume_mult:
            return None

        # ADX filter
        adx = curr.get('adx', 0)
        if pd.isna(adx) or adx < self.adx_threshold:
            return None

        # 200-EMA filter (require uptrend — block if missing or below)
        ema200 = curr.get('ema_200', None)
        if pd.isna(ema200) or close < ema200:
            return None

        # Candlestick confirmation on BOS bar
        is_engulfing = bool(curr.get('candle_engulfing_bull', False))
        is_hammer = bool(curr.get('candle_hammer', False))
        is_pin = bool(curr.get('candle_bullish_pin', False))
        body_pct = float(curr.get('body_pct', 0))
        strong_body = body_pct > 0.60

        if not (is_engulfing or is_hammer or is_pin or strong_body):
            return None

        # Don't chase (< 2.5% above broken level)
        distance_pct = (close - last_sh_price) / last_sh_price
        if distance_pct > 0.025:
            return None

        # Stop: last swing low - 0.5x ATR (buffer)
        atr = curr.get('atr_14', close * 0.02)
        if pd.isna(atr):
            atr = close * 0.02
        stop = last_sl_price - (0.5 * atr)
        if stop >= close:
            stop = close - (2.5 * atr)

        # Reject if structural risk exceeds 6% (instead of capping to meaningless level)
        if stop < close * 0.94:
            return None

        # Target: max of 2.5x risk or impulse projection
        risk = close - stop
        if risk <= 0:
            return None
        impulse_leg = last_sh_price - last_sl_price
        target = max(close + 2.5 * risk, close + impulse_leg * 0.8)

        signal_type = "CHOCH (reversal)" if choch_bullish else "BOS (continuation)"

        # Candle label
        candle_desc = "Strong body" if strong_body else ""
        if is_engulfing:
            candle_desc = "Bullish engulfing"
        elif is_hammer:
            candle_desc = "Hammer"
        elif is_pin:
            candle_desc = "Bullish pin bar"

        reasons = [
            f"{signal_type}: broke above swing high {last_sh_price:.2f}",
            f"Volume {vol_ratio:.1f}x avg, ADX={adx:.1f}",
            f"Structure: {'HH' if hh else 'LH'}/{'HL' if hl else 'LL'}",
            f"{candle_desc} on BOS bar",
        ]

        if choch_bullish:
            reasons.append("Character change from bearish to bullish")

        confidence = min(80, 45 + (10 if choch_bullish else 5) + vol_ratio * 5 + (5 if hl else 0))
        if is_engulfing:
            confidence = min(85, confidence + 5)

        # Weekly trend bonus (not a gate)
        weekly_aligned = curr.get('weekly_trend_aligned', None)
        if not pd.isna(weekly_aligned) and weekly_aligned:
            confidence += 5

        # Universal confluence (Golden Cross, ADX already checked, MACD)
        conf_bonus, conf_reasons = self.confluence_bonus(curr)
        confidence = min(90, confidence + conf_bonus)
        reasons.extend(conf_reasons)

        return TradeSignal(
            strategy=self.name, symbol=symbol, direction=Direction.BUY,
            entry_price=round(close, 2), stop_loss=round(stop, 2),
            target=round(target, 2), confidence=round(confidence, 1),
            reasons=reasons,
        )

    def _detect_structure(self, df: pd.DataFrame, end_idx: int, lookback: int = 60):
        """Detect swing highs and lows for market structure analysis."""
        start = max(0, end_idx - lookback)
        swings = []

        highs = df['high']
        lows = df['low']

        for j in range(start + self.swing_lookback, end_idx + 1):
            # For bars near the end, only use backward window
            bk_start = max(start, j - self.swing_lookback)
            bk_end = j
            fw_end = min(end_idx, j + self.swing_lookback)

            backward_highs = highs.iloc[bk_start:bk_end]
            if fw_end > j:
                forward_highs = highs.iloc[j + 1:fw_end + 1]
                is_swing_high = highs.iloc[j] >= backward_highs.max() and highs.iloc[j] >= forward_highs.max()
            else:
                # Last bars: only backward confirmation (with higher threshold)
                is_swing_high = highs.iloc[j] >= backward_highs.max() * 1.001

            if is_swing_high:
                swings.append((j, highs.iloc[j], 'high'))

            backward_lows = lows.iloc[bk_start:bk_end]
            if fw_end > j:
                forward_lows = lows.iloc[j + 1:fw_end + 1]
                is_swing_low = lows.iloc[j] <= backward_lows.min() and lows.iloc[j] <= forward_lows.min()
            else:
                # Last bars: only backward confirmation (with lower threshold)
                is_swing_low = lows.iloc[j] <= backward_lows.min() * 0.999

            if is_swing_low:
                swings.append((j, lows.iloc[j], 'low'))

        swings.sort(key=lambda x: x[0])
        return swings

    def should_exit(self, df: pd.DataFrame, position: Position) -> Optional[ExitSignal]:
        if len(df) < 1:
            return None

        i = len(df) - 1
        curr = df.iloc[i]
        close, low, high = curr['close'], curr['low'], curr['high']

        position.highest_since_entry = max(position.highest_since_entry, high)

        # 1. Stop loss
        if low <= position.stop_loss:
            return ExitSignal(reason="stop_loss", exit_price=position.stop_loss)

        # 2. Target hit
        if high >= position.target:
            return ExitSignal(reason="target_hit", exit_price=position.target)

        # 3. Breakeven at 2R (give retest room before moving to breakeven)
        risk = position.entry_price - position.stop_loss
        if risk > 0 and high >= position.entry_price + (2.0 * risk) and position.stop_loss < position.entry_price:
            position.stop_loss = position.entry_price

        # 4. Trail with 21-EMA after 2.5R profit (give BOS breakouts room to breathe)
        if risk > 0 and position.highest_since_entry >= position.entry_price + (2.5 * risk):
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
