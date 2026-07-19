"""
Strategy: Trend Pullback
=========================
Buy pullbacks to moving averages in strong weekly uptrends.

Weekly Trend Check (pre-computed in indicators):
- Weekly 10-EMA > 20-EMA > 50-EMA (strong uptrend)
- Price above weekly 20-EMA

Daily Entry Trigger:
- Price has pulled back to daily 20-SMA or 50-SMA zone
- RSI between 38 and 55 (pullback, not oversold crash)
- Today's candle is bullish (bounce signal)
- Volume rising vs previous 3 bars

Stop Loss: Below 50-SMA or recent swing low (whichever is closer to price)
Target: Recent swing high, minimum 2:1 R:R
Max Hold: 25 days
"""

from typing import Optional
import pandas as pd

from .base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction


class TrendPullback(BaseStrategy):
    name = "Trend_Pullback"
    category = "trend"
    min_bars = 200

    def scan(self, df: pd.DataFrame, symbol: str = "") -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        i = len(df) - 1
        curr = df.iloc[i]
        close = float(curr['close'])

        # --- Weekly trend alignment ---
        weekly_aligned = curr.get('weekly_trend_aligned', None)
        if pd.isna(weekly_aligned) or not weekly_aligned:
            return None

        weekly_ema20 = curr.get('weekly_ema_20', None)
        if pd.isna(weekly_ema20) or close <= weekly_ema20:
            return None

        # --- Daily MA values ---
        sma20 = curr.get('sma_20', None)
        sma50 = curr.get('sma_50', None)
        if pd.isna(sma20) or pd.isna(sma50):
            return None

        # Daily trend must be aligned (20-SMA > 50-SMA)
        if sma20 <= sma50:
            return None

        # Uptrend context: price above 200-EMA
        ema200 = curr.get('ema_200', None)
        if pd.isna(ema200) or close < ema200:
            return None

        # Pullback to MA: low must dip to/below the MA (actual test of support)
        # OR close within 1% above (very tight)
        low = float(curr['low'])

        near_sma20 = low <= sma20 * 1.02 and close > sma20
        near_sma50 = low <= sma50 * 1.02 and close > sma50

        if not (near_sma20 or near_sma50):
            return None

        # Price must close ABOVE 50-SMA (bounce, not breakdown)
        if close < sma50:
            return None

        # --- RSI filter: pullback zone (38-55) ---
        rsi = curr.get('rsi_14', None)
        if pd.isna(rsi) or rsi < 38 or rsi > 55:
            return None

        # --- Bullish candle or reversal pattern ---
        is_bullish = bool(curr.get('is_bullish', False))
        has_reversal = bool(
            curr.get('candle_hammer', False)
            or curr.get('candle_bullish_pin', False)
            or curr.get('candle_engulfing_bull', False)
        )
        if not (is_bullish or has_reversal):
            return None

        # --- Volume rising vs previous 3 bars ---
        if i < 3:
            return None
        curr_vol = float(curr['volume'])
        avg_prev_vol = float(df['volume'].iloc[i - 3:i].mean())
        if avg_prev_vol <= 0 or curr_vol < avg_prev_vol * 1.1:
            return None

        # --- Stop loss ---
        atr = curr.get('atr_14', close * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        # Below 50-SMA or recent swing low, whichever is closer to price
        stop_sma = sma50 - 0.5 * atr
        stop_swing = self._recent_swing_low(df, i, lookback=20) - 0.3 * atr
        stop = max(stop_sma, stop_swing)  # closer to price = tighter risk

        if stop >= close:
            stop = close - 2.5 * atr

        # Cap stop at 5% below entry
        if stop < close * 0.95:
            stop = close * 0.95

        risk = close - stop
        if risk <= 0:
            return None

        # --- Target: recent swing high, minimum 2:1 R:R ---
        recent_high = self._recent_swing_high(df, i, lookback=40)
        target = max(recent_high, close + 2.0 * risk)

        # --- Confidence ---
        confidence = 50.0
        vol_ratio = curr.get('volume_ratio', 1.0)
        if not pd.isna(vol_ratio):
            confidence += min(10, vol_ratio * 3)
        adx = curr.get('adx', 0)
        if not pd.isna(adx) and adx > 25:
            confidence += min(10, (adx - 20) * 0.5)
        if near_sma20:
            confidence += 5  # Bouncing off 20-SMA is higher quality
        if has_reversal:
            confidence += 5  # Reversal candle at MA support

        # Universal confluence (Golden Cross, ADX, MACD)
        conf_bonus, conf_reasons = self.confluence_bonus(curr)
        confidence += conf_bonus
        confidence = min(90, confidence)

        candle_desc = "Bullish candle"
        if curr.get('candle_engulfing_bull', False):
            candle_desc = "Bullish engulfing"
        elif curr.get('candle_hammer', False):
            candle_desc = "Hammer"
        elif curr.get('candle_bullish_pin', False):
            candle_desc = "Bullish pin bar"

        reasons = [
            "Weekly trend aligned (10 > 20 > 50 EMA)",
            f"Pullback to {'20-SMA' if near_sma20 else '50-SMA'}",
            f"RSI={rsi:.1f} (pullback zone)",
            candle_desc,
            f"Volume rising ({curr_vol / avg_prev_vol:.1f}x vs prev 3 bars)",
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
