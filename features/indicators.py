"""
Quant X Indicator Module
========================
Computes technical indicators needed by the 6 active strategies.
Uses the `ta` library (already in requirements.txt).

Indicator Groups:
- Trend: EMA(9,21,200), SMA(20,50,200), ADX, Golden Cross
- Momentum: RSI(14), MACD
- Volatility: Bollinger Bands (upper/lower), ATR(14)
- Volume: OBV, Volume SMA, Volume Ratio
- Structure: Swing Highs/Lows
- Candlestick: 20 candle patterns
- Weekly: EMA(10,20,50), trend alignment
- Volume Analysis: Wyckoff/VPA (accumulation, climax, absorption)
"""

import numpy as np
import pandas as pd
import ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators needed by the 15 strategies.
    Mutates the DataFrame in place and returns it.

    Args:
        df: DataFrame with lowercase columns: open, high, low, close, volume

    Returns:
        Same DataFrame with indicator columns added.
    """
    df = df.copy()

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]

    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume'].astype(float)

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    # EMAs
    df['ema_9'] = ta.trend.ema_indicator(close, window=9)
    df['ema_21'] = ta.trend.ema_indicator(close, window=21)
    df['ema_200'] = ta.trend.ema_indicator(close, window=200)

    # SMAs
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)
    df['sma_200'] = ta.trend.sma_indicator(close, window=200)

    # Golden Cross / Death Cross (trend filter — not entry signal)
    df['golden_cross'] = df['sma_50'] > df['sma_200']

    # ADX + Directional Indicators (DI+/DI- used for breakout direction confirmation)
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df['adx'] = adx_indicator.adx()
    df['di_plus'] = adx_indicator.adx_pos()
    df['di_minus'] = adx_indicator.adx_neg()

    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================

    # RSI
    df['rsi_14'] = ta.momentum.rsi(close, window=14)

    # MACD
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================

    # Bollinger Bands (upper/lower only — middle and %B unused)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    # ATR
    df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)

    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================

    # OBV
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    # Volume SMA and ratio
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20'].replace(0, np.nan)

    # =========================================================================
    # STRUCTURE DETECTION
    # =========================================================================

    # Swing Highs and Lows (5-bar lookback)
    df['swing_high'] = _detect_swing_highs(df, lookback=5)
    df['swing_low'] = _detect_swing_lows(df, lookback=5)

    # =========================================================================
    # CANDLE ANATOMY (for pattern strategies)
    # =========================================================================
    body = abs(close - df['open'])
    total_range = high - low
    df['body_pct'] = (body / total_range.replace(0, np.nan)).fillna(np.nan)
    df['is_bullish'] = close > df['open']

    # =========================================================================
    # CANDLESTICK PATTERNS
    # =========================================================================
    df = _compute_candlestick_patterns(df)

    # =========================================================================
    # ADVANCED VOLUME ANALYSIS (Wyckoff / VPA)
    # =========================================================================
    from .volume_analysis import compute_volume_analysis
    df = compute_volume_analysis(df)

    # =========================================================================
    # WEEKLY TIMEFRAME INDICATORS (for multi-timeframe strategies)
    # =========================================================================
    df = _compute_weekly_indicators(df)

    # =========================================================================
    # SCREENER INDICATORS (used by LiveScreenerEngine)
    # =========================================================================
    df = _compute_screener_indicators(df)

    return df


def _compute_screener_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extra indicators used by the LiveScreenerEngine for scanner filters.
    Added as a separate function to keep compute_all_indicators clean.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 52-week and 10-day high/low
    df['high_52w'] = high.rolling(window=min(252, len(df))).max()
    df['low_52w'] = low.rolling(window=min(252, len(df))).min()
    df['high_10d'] = high.rolling(window=min(10, len(df))).max()
    df['low_10d'] = low.rolling(window=min(10, len(df))).min()

    # Daily range (for NR4, NR7, Inside Bar)
    df['daily_range'] = high - low

    # NR4: today's range is narrowest of last 4 days
    range_min_4 = df['daily_range'].rolling(window=4).min()
    df['nr4'] = (df['daily_range'] <= range_min_4) & (df['daily_range'] > 0)

    # NR7: today's range is narrowest of last 7 days
    range_min_7 = df['daily_range'].rolling(window=7).min()
    df['nr7'] = (df['daily_range'] <= range_min_7) & (df['daily_range'] > 0)

    # Inside Bar: today's range fully within yesterday's range
    df['inside_bar'] = (high <= high.shift(1)) & (low >= low.shift(1))

    # Pivot points (classic floor pivots)
    df['pivot'] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    df['pivot_r1'] = 2 * df['pivot'] - low.shift(1)
    df['pivot_s1'] = 2 * df['pivot'] - high.shift(1)

    # SuperTrend (period=10, multiplier=2)
    atr = df['atr_14'] if 'atr_14' in df.columns else (high - low).rolling(14).mean()
    hl2 = (high + low) / 2
    upper_band = hl2 + 2.0 * atr
    lower_band = hl2 - 2.0 * atr

    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i],
                                     supertrend.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i],
                                     supertrend.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]

    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction  # 1=bullish, -1=bearish

    # Parabolic SAR
    try:
        psar = ta.trend.PSARIndicator(high, low, close)
        df['psar'] = psar.psar()
        df['psar_up'] = psar.psar_up()
        df['psar_down'] = psar.psar_down()
        # Bullish when price is above PSAR
        df['psar_bullish'] = close > df['psar']
    except Exception:
        df['psar'] = np.nan
        df['psar_bullish'] = False

    # TTM Squeeze: Bollinger Bands inside Keltner Channels
    kc_mid = df['sma_20'] if 'sma_20' in df.columns else close.rolling(20).mean()
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    bb_upper = df.get('bb_upper', close.rolling(20).mean() + 2 * close.rolling(20).std())
    bb_lower = df.get('bb_lower', close.rolling(20).mean() - 2 * close.rolling(20).std())
    df['ttm_squeeze'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # ATR-based trailing stop (for Scanner 18)
    df['atr_trailing_stop'] = close - 2.0 * atr

    # Change percent (today vs yesterday)
    df['change_pct'] = close.pct_change() * 100

    # Previous close
    df['prev_close'] = close.shift(1)

    # =========================================================================
    # PR-S18 — INSTITUTIONAL SETUP COLUMNS
    # Backs the IDs 72-86 setup scanners. Each column verified against a
    # cited source (see screener/filters.py for per-scanner sources).
    # =========================================================================

    # EMA(10) — needed by Pocket Pivot (Kacher/Morales)
    df['ema_10'] = ta.trend.ema_indicator(close, window=10)

    # SMA(150) ≈ 30-week MA — needed by Stage 2 (Weinstein)
    df['sma_150'] = ta.trend.sma_indicator(close, window=150)
    # Rising 30-week MA flag: today's SMA150 > SMA150 20 bars ago
    df['sma_150_rising'] = df['sma_150'] > df['sma_150'].shift(20)

    # Pocket Pivot — max volume on a down-close day over last 10 bars
    volume = df['volume'] if 'volume' in df.columns else close * 0
    down_day_vol = volume.where(close < close.shift(1), 0)
    df['max_down_day_vol_10'] = down_day_vol.rolling(window=10).max()
    df['pocket_pivot_volume'] = volume > df['max_down_day_vol_10']

    # Gap pct vs prior close (Episodic Pivot, Gap-Fill Reversal)
    df['gap_pct'] = (df['open'] - close.shift(1)) / close.shift(1) * 100

    # Bollinger band width %-of-price + percentile rank over 60 bars
    # (Coiled Spring / volatility compression detection)
    bb_width = (df['bb_upper'] - df['bb_lower']) / close
    df['bb_width_pct'] = bb_width
    df['bb_width_pct_rank_60'] = bb_width.rolling(60).rank(pct=True)

    # Inside Bar (already computed) + 2-bar inside detection
    # Trend Day flag: prior bar's range >= 1.5× ATR(20) AND closed at extreme
    atr_20 = (high - low).rolling(20).mean()
    prior_range = (high - low).shift(1)
    prior_body = (close - df['open']).abs().shift(1)
    df['prior_trend_day_up'] = (
        (prior_range >= 1.5 * atr_20.shift(1)) &
        (prior_body >= 0.7 * prior_range) &
        (close.shift(1) > df['open'].shift(1))
    )
    df['prior_trend_day_down'] = (
        (prior_range >= 1.5 * atr_20.shift(1)) &
        (prior_body >= 0.7 * prior_range) &
        (close.shift(1) < df['open'].shift(1))
    )

    # Base height % over last 60 bars (CAN SLIM base detection proxy)
    high_60 = high.rolling(60).max()
    low_60 = low.rolling(60).min()
    df['base_height_pct_60'] = (high_60 - low_60) / high_60

    # Channel slope (60-bar linear regression slope normalized by price)
    # Used by Channel Mid-Reversion + Stage 2 trend qualification
    try:
        import numpy as _np
        idx = _np.arange(60)
        rolling_slope = close.rolling(60).apply(
            lambda y: _np.polyfit(idx, y.values, 1)[0] if y.notna().sum() >= 60 else _np.nan,
            raw=False,
        )
        df['channel_slope_60'] = rolling_slope / close
    except Exception:
        df['channel_slope_60'] = np.nan

    # Three consecutive green closes (Three White Soldiers prerequisite)
    is_green = close > df['open']
    df['three_green'] = is_green & is_green.shift(1) & is_green.shift(2)

    # Dragonfly doji detection on current bar (body ≤10% of range, lower wick ≥66% of range)
    body_abs = (close - df['open']).abs()
    rng = (high - low).replace(0, np.nan)
    upper_shadow = high - close.where(close > df['open'], df['open'])
    lower_shadow = close.where(close < df['open'], df['open']) - low
    df['dragonfly_doji'] = (
        (body_abs <= 0.10 * rng) &
        (upper_shadow <= 0.10 * rng) &
        (lower_shadow >= 0.66 * rng)
    )

    # Bullish engulfing on current bar (confirmation candle for many setups)
    df['bull_engulf'] = (
        (close.shift(1) < df['open'].shift(1)) &
        (close > df['open']) &
        (close >= df['open'].shift(1)) &
        (df['open'] <= close.shift(1))
    )

    # Weekly CPR (TC, BC, R1, S1) — Ochoa / Zerodha Varsity formula
    # Uses prior 5-bar (calendar approx) H/L/C as the weekly inputs.
    w_high = high.shift(1).rolling(5).max()
    w_low = low.shift(1).rolling(5).min()
    w_close = close.shift(1)
    w_pivot = (w_high + w_low + w_close) / 3
    df['cpr_bc'] = (w_high + w_low) / 2
    df['cpr_tc'] = 2 * w_pivot - df['cpr_bc']
    df['cpr_p'] = w_pivot
    df['cpr_width_pct'] = (df['cpr_tc'] - df['cpr_bc']).abs() / w_pivot
    df['cpr_narrow'] = df['cpr_width_pct'] <= 0.005   # narrow-CPR = explosive

    return df


def classify_trend_tier(df: pd.DataFrame) -> str:
    """
    Classify stock's trend state for strategy routing.

    Reads already-computed indicator columns from the last row — negligible cost.

    Returns:
        'tier1' - Confirmed uptrend → run ALL 6 strategies
        'tier2' - Base formation   → run only reversal strategies
        'skip'  - Downtrend        → skip entirely
    """
    if len(df) < 200:
        return 'skip'

    curr = df.iloc[-1]
    close = float(curr['close'])

    # Read indicators (all already computed)
    ema_200 = curr.get('ema_200', None)
    sma_50 = curr.get('sma_50', None)
    sma_200 = curr.get('sma_200', None)
    weekly_aligned = curr.get('weekly_trend_aligned', None)
    rsi = curr.get('rsi_14', None)
    accum = curr.get('accumulation_phase', False)

    # Handle NaN safely
    has_ema200 = ema_200 is not None and not pd.isna(ema_200) and ema_200 > 0
    has_sma50 = sma_50 is not None and not pd.isna(sma_50) and sma_50 > 0
    has_sma200 = sma_200 is not None and not pd.isna(sma_200) and sma_200 > 0
    has_weekly = weekly_aligned is not None and not pd.isna(weekly_aligned)
    has_rsi = rsi is not None and not pd.isna(rsi)

    # --- TIER 1: Confirmed uptrend (all continuation strategies allowed) ---
    # close > ema_200 AND sma_50 > sma_200 AND (weekly_aligned OR close > sma_50)
    if has_ema200 and close > ema_200:
        no_death_cross = has_sma50 and has_sma200 and sma_50 > sma_200
        medium_term_ok = (
            (has_weekly and bool(weekly_aligned))
            or (has_sma50 and close > sma_50)
        )
        if no_death_cross and medium_term_ok:
            return 'tier1'

    # --- TIER 2: Base formation (only reversal strategies) ---
    # close > lowest_low_20 AND within 25% of ema_200
    # AND (accumulation OR 2+ of: close > sma_50, weekly_aligned, rsi < 45)
    lowest_20 = float(df['low'].iloc[-20:].min()) if len(df) >= 20 else 0
    not_new_lows = close > lowest_20

    within_ema200 = True  # Default if no ema_200
    if has_ema200:
        within_ema200 = close >= ema_200 * 0.75  # Within 25% of 200-EMA

    if not_new_lows and within_ema200:
        # Check accumulation OR multi-signal confluence
        if accum and not pd.isna(accum) and bool(accum):
            return 'tier2'

        # Count bullish signals
        bullish_signals = 0
        if has_sma50 and close > sma_50:
            bullish_signals += 1
        if has_weekly and bool(weekly_aligned):
            bullish_signals += 1
        if has_rsi and rsi < 45:  # Oversold = reversal opportunity for Tier 2
            bullish_signals += 1

        if bullish_signals >= 2:
            return 'tier2'

    # --- SKIP: Downtrend ---
    return 'skip'


def _compute_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect candlestick reversal patterns.

    Adds boolean columns:
    - candle_hammer: Small body at top, long lower wick (>2x body)
    - candle_inverted_hammer: Small body at bottom, long upper wick (>2x body)
    - candle_engulfing_bull: Current bullish bar engulfs prior bearish bar
    - candle_engulfing_bear: Current bearish bar engulfs prior bullish bar
    - candle_doji: Body < 10% of total range
    - candle_morning_star: 3-bar bullish reversal (bearish, small body, bullish)
    - candle_pin_bar: Small body with one dominant wick (>2.5x body)
    - candle_dark_cloud: Bearish reversal (open above prev high, close below prev midpoint)
    - candle_evening_star: 3-bar bearish reversal
    - candle_harami_bull/bear: Small body inside previous large body
    - candle_tweezer_bottom/top: Matching lows/highs on consecutive bars
    """
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    vol = df['volume'].astype(float)
    body = abs(c - o)
    total_range = (h - l).replace(0, np.nan)
    body_pct = body / total_range
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    # Volume confirmation gates (strict)
    vol_sma = vol.rolling(20).mean().replace(0, np.nan)
    vol_ratio = vol / vol_sma
    vol_ok = (vol_ratio >= 1.0)        # At least average volume
    vol_strong = (vol_ratio >= 1.5)    # 50% above average for key patterns

    # ATR-relative minimum candle size — ignore tiny noise candles
    atr = df['atr_14'] if 'atr_14' in df.columns else total_range.rolling(14).mean()
    min_candle = total_range >= (atr * 0.5)  # Candle range >= half ATR

    # Common conditions
    prev_bearish = c.shift(1) < o.shift(1)
    prev_bullish = c.shift(1) > o.shift(1)
    curr_bullish = c > o
    curr_bearish = c < o
    big_body = body_pct > 0.50          # Relaxed from 0.55 — real engulfing candles exist at 50%+
    small_body = body_pct < 0.20        # Relaxed from 0.15 — captures more harami/star patterns

    # Trend context: local SMA crossover is sufficient for candle pattern context
    # The strategy-level filters (RSI, EMA200, weekly trend) provide the higher-level confirmation.
    sma_5 = c.rolling(5).mean()
    sma_20_ctx = c.rolling(20).mean()
    local_downtrend = (sma_5 < sma_20_ctx) & (c < sma_5)
    local_uptrend = (sma_5 > sma_20_ctx) & (c > sma_5)
    # Use local_downtrend/uptrend directly as context — the double-AND with recent_decline
    # was killing 90%+ of valid candle patterns in Indian markets.
    bearish_ctx = local_downtrend
    bullish_ctx = local_uptrend

    # --- Hammer: small body in upper portion, long lower wick ---
    df['candle_hammer'] = (
        (body_pct < 0.35)                    # Relaxed from 0.20 — body up to 35% of range
        & (lower_wick > body * 2.0)          # Relaxed from 2.5x — standard 2x ratio
        & (upper_wick < body * 0.5)          # Relaxed from 0.15x — small upper wick OK
        & (lower_wick > total_range * 0.55)  # Relaxed from 0.60 — lower shadow dominates
        & min_candle                         # Not a tiny candle
        & bearish_ctx
        & vol_ok
    ).fillna(False)

    # --- Inverted hammer: small body in lower portion, long upper wick ---
    df['candle_inverted_hammer'] = (
        (body_pct < 0.35)                    # Relaxed from 0.20
        & (upper_wick > body * 2.0)          # Relaxed from 2.5x
        & (lower_wick < body * 0.5)          # Relaxed from 0.15x
        & (upper_wick > total_range * 0.55)  # Relaxed from 0.60
        & min_candle
        & bearish_ctx
        & vol_ok
    ).fillna(False)

    # --- Bullish engulfing: current body engulfs prior body ---
    prev_body = abs(c.shift(1) - o.shift(1))
    df['candle_engulfing_bull'] = (
        prev_bearish & curr_bullish
        & (o <= c.shift(1))               # Opens at/below prev close
        & (c >= o.shift(1))               # Closes at/above prev open
        & (body >= prev_body * 1.2)       # Relaxed from 1.5x — 1.2x is standard engulfing
        & big_body                         # Current body is dominant
        & min_candle
        & bearish_ctx                      # Only in downtrend context
        & vol_strong                        # Engulfing needs conviction volume (1.5x avg)
    ).fillna(False)

    # --- Bearish engulfing: current body engulfs prior body ---
    df['candle_engulfing_bear'] = (
        prev_bullish & curr_bearish
        & (o >= c.shift(1))               # Opens at/above prev close
        & (c <= o.shift(1))               # Closes at/below prev open
        & (body >= prev_body * 1.2)       # Relaxed from 1.5x
        & big_body
        & min_candle
        & bullish_ctx                      # Only in uptrend context
        & vol_strong                       # Engulfing needs conviction volume (1.5x avg)
    ).fillna(False)

    # --- Doji: body < 2% of range, must be meaningful candle ---
    df['candle_doji'] = ((body_pct < 0.02) & min_candle & vol_ok).fillna(False)

    # --- Dragonfly Doji: bullish reversal (long lower shadow, no upper) ---
    df['candle_dragonfly_doji'] = (
        (body_pct < 0.03)
        & (lower_wick > total_range * 0.70)   # 70% lower shadow
        & (upper_wick < total_range * 0.05)   # Almost no upper shadow
        & min_candle
        & bearish_ctx
        & vol_ok
    ).fillna(False)

    # --- Gravestone Doji: bearish reversal (long upper shadow, no lower) ---
    df['candle_gravestone_doji'] = (
        (body_pct < 0.03)
        & (upper_wick > total_range * 0.70)   # 70% upper shadow
        & (lower_wick < total_range * 0.05)   # Almost no lower shadow
        & min_candle
        & bullish_ctx
        & vol_ok
    ).fillna(False)

    # --- Morning star: bar[-2] big bearish, bar[-1] small body, bar[0] big bullish ---
    prev2_bearish = (c.shift(2) < o.shift(2)) & (body_pct.shift(2) > 0.50)  # Relaxed from 0.60
    prev1_tiny = body_pct.shift(1) < 0.20     # Relaxed from 0.10 — small body, not just doji
    curr_big_bull = curr_bullish & (body_pct > 0.50)  # Relaxed from 0.60
    prev2_mid = (o.shift(2) + c.shift(2)) / 2
    # Middle bar's body center below first bar's close (gap or continuation down)
    mid_body_center = (o.shift(1) + c.shift(1)) / 2
    mid_gapped = mid_body_center < c.shift(2)
    df['candle_morning_star'] = (
        prev2_bearish & prev1_tiny & curr_big_bull
        & mid_gapped                      # Middle bar gapped/continued lower
        & (c > prev2_mid)                 # 3rd bar closes above 1st bar midpoint
        & min_candle                      # Current bar is meaningful
        & bearish_ctx.shift(2)            # Pattern starts in downtrend
        & vol_ok                          # Relaxed from vol_strong — avg volume sufficient
    ).fillna(False)

    # --- Pin bar: one wick dominates (>2.5x body), other wick small ---
    bullish_pin = (
        (lower_wick > body * 2.5) & (upper_wick < body * 0.4)  # Relaxed from 3x/0.15
        & (lower_wick > total_range * 0.60)  # Relaxed from 0.65
    )
    bearish_pin = (
        (upper_wick > body * 2.5) & (lower_wick < body * 0.4)  # Relaxed from 3x/0.15
        & (upper_wick > total_range * 0.60)
    )
    df['candle_pin_bar'] = ((bullish_pin | bearish_pin) & min_candle & vol_ok).fillna(False)
    df['candle_bullish_pin'] = (bullish_pin & min_candle & bearish_ctx & vol_ok).fillna(False)

    # --- Three White Soldiers: 3 bullish candles, each closing higher, ---
    # each opening within previous body, small upper wicks, decent bodies
    three_bull = curr_bullish & curr_bullish.shift(1) & curr_bullish.shift(2)
    three_higher = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    opens_in_body = (
        (o >= pd.concat([c.shift(1), o.shift(1)], axis=1).min(axis=1))
        & (o <= pd.concat([c.shift(1), o.shift(1)], axis=1).max(axis=1))
    )
    opens_in_body_prev = (
        (o.shift(1) >= pd.concat([c.shift(2), o.shift(2)], axis=1).min(axis=1))
        & (o.shift(1) <= pd.concat([c.shift(2), o.shift(2)], axis=1).max(axis=1))
    )
    small_upper_wicks = (
        (upper_wick < total_range * 0.30)          # Relaxed from 0.20 — 30% max upper wick
        & (upper_wick.shift(1) < total_range.shift(1) * 0.30)
        & (upper_wick.shift(2) < total_range.shift(2) * 0.30)
    )
    decent_bodies = (
        (body_pct > 0.50) & (body_pct.shift(1) > 0.50) & (body_pct.shift(2) > 0.50)  # Relaxed from 0.60
    )
    # All 3 candles must be meaningful size
    three_min_candle = min_candle & min_candle.shift(1) & min_candle.shift(2)
    df['candle_three_white_soldiers'] = (
        three_bull & three_higher & opens_in_body & opens_in_body_prev
        & small_upper_wicks & decent_bodies & three_min_candle
        & bearish_ctx.shift(2)  # Pattern starts from a downtrend
        & vol_ok                # Relaxed from vol_strong
    ).fillna(False)

    # --- Three Line Strike (Bullish, 78% WR Bulkowski): ---
    # 3 bearish candles progressively lower, 4th bullish candle engulfs all 3
    three_bear = curr_bearish.shift(1) & curr_bearish.shift(2) & curr_bearish.shift(3)
    progressive_lower = (
        (c.shift(1) < c.shift(2))
        & (c.shift(2) < c.shift(3))
    )
    # All 3 bearish bars must have decent bodies
    three_bear_bodies = (
        (body_pct.shift(1) > 0.40)
        & (body_pct.shift(2) > 0.40)
        & (body_pct.shift(3) > 0.40)
    )
    strike_close = c >= o.shift(3)    # Closes above 1st bearish candle's open
    strike_big = body_pct > 0.65      # Strong bullish body
    df['candle_three_line_strike'] = (
        three_bear & progressive_lower & three_bear_bodies
        & curr_bullish & strike_close
        & strike_big & min_candle
        & vol_strong
    ).fillna(False)

    # --- Piercing Line: bar[-1] big bearish, bar[0] opens below prev low, ---
    # closes 50%+ into prev body (standard definition)
    prev_body_range = abs(o.shift(1) - c.shift(1)).replace(0, np.nan)
    penetration = (c - c.shift(1)) / prev_body_range
    df['candle_piercing_line'] = (
        prev_bearish & curr_bullish
        & big_body.shift(1)               # Prev bar must have big body
        & (o < l.shift(1))                # Opens below prev low
        & (penetration >= 0.50)           # Relaxed from 0.65 — standard 50% penetration
        & (c < o.shift(1))               # Doesn't fully engulf
        & min_candle & min_candle.shift(1)
        & bearish_ctx                     # In downtrend context
        & vol_ok                          # Relaxed from vol_strong
    ).fillna(False)

    # --- Bullish Abandoned Baby: bar[-2] big bearish, bar[-1] doji with TRUE gap, ---
    # bar[0] big bullish with gap up — extremely rare but very reliable
    doji = body_pct < 0.03
    gap_down_prev = h.shift(1) < l.shift(2)   # Complete gap: mid bar high < prev low
    gap_up_curr = l > h.shift(1)               # Complete gap: current low > mid bar high
    df['candle_abandoned_baby_bull'] = (
        (c.shift(2) < o.shift(2)) & big_body.shift(2)  # Bar[-2] big bearish
        & doji.shift(1)                       # Bar[-1] true doji
        & gap_down_prev                       # Gap down to doji
        & curr_bullish & big_body             # Bar[0] big bullish
        & gap_up_curr                         # Gap up from doji
        & min_candle
        & vol_strong
    ).fillna(False)

    # =====================================================================
    # NEW PATTERNS (Phase 7)
    # =====================================================================

    # --- Dark Cloud Cover (bearish): ---
    # Bar[-1] big bullish, bar[0] opens above prev high, closes 60%+ into prev body
    prev1_mid = (o.shift(1) + c.shift(1)) / 2
    prev1_40pct = o.shift(1) + (c.shift(1) - o.shift(1)) * 0.40  # 40% from open
    df['candle_dark_cloud'] = (
        prev_bullish & (body_pct.shift(1) > 0.60)
        & curr_bearish & big_body
        & (o > h.shift(1))               # Opens above prev high (gap up)
        & (c < prev1_40pct)              # Closes below 60% of prev body
        & (c > o.shift(1))              # Doesn't close below prev open (not full engulf)
        & min_candle & min_candle.shift(1)
        & bullish_ctx
        & vol_ok                          # Relaxed from vol_strong
    ).fillna(False)

    # --- Evening Star (bearish): ---
    # Bar[-2] big bullish, bar[-1] small body, bar[0] big bearish closing deep into bar[-2]
    prev2_bullish = (c.shift(2) > o.shift(2)) & (body_pct.shift(2) > 0.50)  # Relaxed from 0.60
    prev1_tiny = body_pct.shift(1) < 0.20  # Relaxed from 0.10
    prev2_mid_bull = (o.shift(2) + c.shift(2)) / 2
    curr_big_bear = curr_bearish & (body_pct > 0.50)  # Relaxed from 0.60
    mid_body_center_ev = (o.shift(1) + c.shift(1)) / 2
    mid_gapped_ev = mid_body_center_ev > c.shift(2)   # Middle bar above 1st bar close
    df['candle_evening_star'] = (
        prev2_bullish & prev1_tiny & curr_big_bear
        & mid_gapped_ev                  # Middle bar gapped/stayed above
        & (c < prev2_mid_bull)           # 3rd bar closes below 1st midpoint
        & min_candle
        & bullish_ctx.shift(2)           # Pattern starts in uptrend
        & vol_ok                          # Relaxed from vol_strong
    ).fillna(False)

    # --- Bullish Harami: ---
    # Previous big bearish body (>60%), current tiny body entirely within previous
    prev_body_top = pd.concat([c.shift(1), o.shift(1)], axis=1).max(axis=1)
    prev_body_bot = pd.concat([c.shift(1), o.shift(1)], axis=1).min(axis=1)
    curr_body_top = pd.concat([c, o], axis=1).max(axis=1)
    curr_body_bot = pd.concat([c, o], axis=1).min(axis=1)
    # Current body must be significantly smaller (< 30% of prev body)
    body_ratio = body / prev_body.replace(0, np.nan)
    df['candle_harami_bull'] = (
        prev_bearish & (body_pct.shift(1) > 0.50)  # Relaxed from 0.60
        & (body_pct < 0.25)              # Relaxed from 0.15 — small body, not just tiny
        & (curr_body_top <= prev_body_top)
        & (curr_body_bot >= prev_body_bot)
        & (body_ratio < 0.40)            # Relaxed from 0.30 — body < 40% of prior
        & min_candle.shift(1)            # Previous bar meaningful
        & bearish_ctx                    # In downtrend
        & vol_ok
    ).fillna(False)

    # --- Bearish Harami: ---
    df['candle_harami_bear'] = (
        prev_bullish & (body_pct.shift(1) > 0.60)
        & (body_pct < 0.15)
        & (curr_body_top <= prev_body_top)
        & (curr_body_bot >= prev_body_bot)
        & (body_ratio < 0.30)
        & min_candle.shift(1)
        & bullish_ctx
        & vol_ok
    ).fillna(False)

    # --- Tweezer Bottom: matching lows within 0.3%, both bars meaningful ---
    low_match = ((l - l.shift(1)).abs() / l.shift(1).replace(0, np.nan)) < 0.0015  # Normalized: 0.15%
    df['candle_tweezer_bottom'] = (
        low_match
        & prev_bearish & curr_bullish
        & min_candle & min_candle.shift(1)
        & (body_pct > 0.25)              # Relaxed from 0.30
        & (body_pct.shift(1) > 0.25)     # Relaxed from 0.30
        & bearish_ctx
        & vol_ok
    ).fillna(False)

    # --- Tweezer Top: matching highs within 0.05% ---
    high_match = ((h - h.shift(1)).abs() / h.shift(1).replace(0, np.nan)) < 0.0015  # Normalized: 0.15%
    df['candle_tweezer_top'] = (
        high_match
        & prev_bullish & curr_bearish
        & min_candle & min_candle.shift(1)
        & (body_pct > 0.30)
        & (body_pct.shift(1) > 0.30)
        & bullish_ctx
        & vol_ok
    ).fillna(False)

    return df


def _compute_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly and compute trend EMAs.

    Adds columns (forward-filled to daily):
    - weekly_ema_10, weekly_ema_20, weekly_ema_50
    - weekly_trend_aligned: True when EMA10 > EMA50 and close > EMA20
    - weekly_close_above_20ema: True when close > weekly 20-EMA

    Uses end-of-week values forward-filled to the next week's daily bars,
    so there is no look-ahead bias.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Cannot resample without datetime index — skip
        cols = ['weekly_ema_10', 'weekly_ema_20', 'weekly_ema_50',
                'weekly_trend_aligned']
        for c in cols:
            df[c] = np.nan
        return df

    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])

    if len(weekly) < 50:
        cols = ['weekly_ema_10', 'weekly_ema_20', 'weekly_ema_50',
                'weekly_trend_aligned']
        for c in cols:
            df[c] = np.nan
        return df

    weekly['weekly_ema_10'] = weekly['close'].ewm(span=10, adjust=False).mean()
    weekly['weekly_ema_20'] = weekly['close'].ewm(span=20, adjust=False).mean()
    weekly['weekly_ema_50'] = weekly['close'].ewm(span=50, adjust=False).mean()

    weekly['weekly_trend_aligned'] = (
        (weekly['weekly_ema_10'] > weekly['weekly_ema_50'])
        & (weekly['close'] > weekly['weekly_ema_20'])
    )

    for col in ['weekly_ema_10', 'weekly_ema_20', 'weekly_ema_50',
                'weekly_trend_aligned']:
        df[col] = weekly[col].reindex(df.index, method='ffill')

    return df


def _detect_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect swing highs: bar where high is higher than surrounding bars."""
    high = df['high']
    result = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window_before = high.iloc[i - lookback:i]
        window_after = high.iloc[i + 1:i + lookback + 1]
        if high.iloc[i] > window_before.max() and high.iloc[i] > window_after.max():
            result.iloc[i] = True
    return result


def _detect_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect swing lows: bar where low is lower than surrounding bars."""
    low = df['low']
    result = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window_before = low.iloc[i - lookback:i]
        window_after = low.iloc[i + 1:i + lookback + 1]
        if low.iloc[i] < window_before.min() and low.iloc[i] < window_after.min():
            result.iloc[i] = True
    return result


def _collect_pivot_data(df: pd.DataFrame, lookback: int = 60, confirmation_lag: int = 5):
    """Collect swing pivot data from recent bars for S/R detection.

    A swing pivot at bar i is only confirmed after `confirmation_lag` bars
    have passed (the forward window needed to verify the pivot). To prevent
    look-ahead bias, the last `confirmation_lag` bars are excluded when
    reading swing_high / swing_low columns.

    Args:
        df: DataFrame with swing_high, swing_low, atr_14, volume_ratio columns.
        lookback: Number of bars to look back for pivots.
        confirmation_lag: Number of bars to exclude from the end (default 5,
            matching the lookback used by _detect_swing_highs/_detect_swing_lows).

    Returns:
        (support_data, resistance_data, atr_median) where each data list
        contains (price, volume_weight) tuples.
    """
    recent = df.iloc[-lookback:]
    has_vol_ratio = 'volume_ratio' in recent.columns

    # Get median ATR for bandwidth
    atr_median = None
    if 'atr_14' in recent.columns:
        atr_vals = recent['atr_14'].dropna()
        if len(atr_vals) > 0:
            atr_median = float(atr_vals.median())

    if atr_median is None or atr_median <= 0:
        atr_median = float(recent['close'].median()) * 0.015

    # Exclude the last `confirmation_lag` bars from swing detection to prevent
    # look-ahead bias. A pivot at bar i needs `confirmation_lag` future bars
    # to confirm it, so only pivots in confirmed_recent are usable.
    if confirmation_lag > 0 and len(recent) > confirmation_lag:
        confirmed_recent = recent.iloc[:-confirmation_lag]
    else:
        confirmed_recent = recent

    support_data = []
    if 'swing_low' in confirmed_recent.columns:
        for idx in confirmed_recent.index[confirmed_recent['swing_low']]:
            price = float(recent.loc[idx, 'low'])
            vol_w = float(recent.loc[idx, 'volume_ratio']) if has_vol_ratio else 1.0
            if pd.isna(vol_w) or vol_w <= 0:
                vol_w = 1.0
            support_data.append((price, vol_w))

    resistance_data = []
    if 'swing_high' in confirmed_recent.columns:
        for idx in confirmed_recent.index[confirmed_recent['swing_high']]:
            price = float(recent.loc[idx, 'high'])
            vol_w = float(recent.loc[idx, 'volume_ratio']) if has_vol_ratio else 1.0
            if pd.isna(vol_w) or vol_w <= 0:
                vol_w = 1.0
            resistance_data.append((price, vol_w))

    return support_data, resistance_data, atr_median


def _kde_levels(pivot_data, bandwidth, min_density_ratio=0.25, max_levels=8):
    """KDE-based price level detection from pivot points.

    Uses gaussian_kde with ATR-scaled bandwidth to find statistically
    significant price clusters (peaks in the density curve).

    Args:
        pivot_data: list of (price, volume_weight) tuples
        bandwidth: bandwidth in price units (typically 0.5 * ATR)
        min_density_ratio: min density as fraction of peak to count as level
        max_levels: maximum levels to return

    Returns:
        list of (price, touch_count, strength) sorted by strength descending.
    """
    if len(pivot_data) < 2:
        if not pivot_data:
            return []
        price = pivot_data[0][0]
        vol = pivot_data[0][1]
        return [(round(price, 2), 1, vol)]

    prices = np.array([p[0] for p in pivot_data])
    volumes = np.array([p[1] for p in pivot_data])

    price_range = prices.max() - prices.min()
    if price_range <= 0:
        avg_price = float(np.mean(prices))
        return [(round(avg_price, 2), len(prices),
                 len(prices) * float(np.mean(volumes)))]

    # Convert bandwidth from price units to relative factor for KDE
    bw_factor = bandwidth / price_range
    bw_factor = max(0.03, min(0.5, bw_factor))

    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks as sp_find_peaks

        kde = gaussian_kde(prices, bw_method=bw_factor)
    except Exception:
        return []

    # Evaluate on fine grid
    grid = np.linspace(prices.min() - bandwidth, prices.max() + bandwidth, 500)
    density = kde(grid)

    # Find peaks in density curve
    peak_indices, _ = sp_find_peaks(density, height=density.max() * min_density_ratio)

    if len(peak_indices) == 0:
        # No peaks — use global maximum
        peak_idx = int(np.argmax(density))
        peak_price = float(grid[peak_idx])
        touches = len(prices)
        strength = touches * float(np.mean(volumes))
        return [(round(peak_price, 2), touches, strength)]

    results = []
    used_mask = np.zeros(len(prices), dtype=bool)

    for pi in peak_indices:
        peak_price = float(grid[pi])
        peak_density = float(density[pi])

        # Count touches: pivots within bandwidth of this peak
        within = np.abs(prices - peak_price) <= bandwidth
        # Avoid double-counting: only count pivots not yet assigned
        assignable = within & ~used_mask
        touch_count = int(np.sum(assignable))

        if touch_count == 0:
            continue

        used_mask |= assignable

        touch_prices = prices[assignable]
        touch_vols = volumes[assignable]
        total_vol = touch_vols.sum()

        if total_vol > 0:
            weighted_price = float(np.sum(touch_prices * touch_vols) / total_vol)
        else:
            weighted_price = float(np.mean(touch_prices))

        avg_vol = float(np.mean(touch_vols))
        # Strength = touches * avg_volume * density (higher density = more concentrated)
        strength = touch_count * avg_vol * (1.0 + peak_density)
        results.append((round(weighted_price, 2), touch_count, strength))

    # Sort by strength descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:max_levels]


def detect_support_resistance_kde(
    df: pd.DataFrame, lookback: int = 60, bandwidth_atr_mult: float = 0.5,
):
    """KDE-based support/resistance detection.

    Uses gaussian_kde on swing pivot prices with ATR-scaled bandwidth for
    statistically principled level detection. Peaks in the density curve
    represent the most frequently tested price levels.

    Args:
        df: DataFrame with swing_high, swing_low, atr_14, volume_ratio columns.
        lookback: Number of bars to look back.
        bandwidth_atr_mult: Bandwidth as multiple of median ATR (default 0.5).

    Returns:
        (support_levels, resistance_levels): Each is a list of
        (price, touch_count, strength) sorted by strength descending.
    """
    support_data, resistance_data, atr_median = _collect_pivot_data(df, lookback)
    bandwidth = bandwidth_atr_mult * atr_median

    support_levels = _kde_levels(support_data, bandwidth)
    resistance_levels = _kde_levels(resistance_data, bandwidth)

    return support_levels, resistance_levels


def detect_support_resistance(df: pd.DataFrame, lookback: int = 60, cluster_pct: float = 0.008):
    """
    Detect support and resistance levels.

    Primary: KDE with ATR-scaled bandwidth (statistically principled).
    Fallback: Volume-weighted nearest-neighbor clustering.

    Args:
        df: DataFrame with swing_high, swing_low, and volume_ratio columns.
        lookback: Number of bars to look back.
        cluster_pct: Price percentage for fallback clustering (0.008 = 0.8%).

    Returns:
        (support_levels, resistance_levels): Sorted by strength (strongest first).
    """
    # Try KDE first
    try:
        kde_support, kde_resistance = detect_support_resistance_kde(df, lookback)
        if kde_support or kde_resistance:
            support_levels = [r[0] for r in kde_support]
            resistance_levels = [r[0] for r in kde_resistance]
            return support_levels, resistance_levels
    except Exception:
        pass

    # Legacy fallback: nearest-neighbor clustering
    support_data, resistance_data, _ = _collect_pivot_data(df, lookback)

    support_levels = _cluster_levels_weighted(
        [(p, v) for p, v in support_data], cluster_pct
    )
    resistance_levels = _cluster_levels_weighted(
        [(p, v) for p, v in resistance_data], cluster_pct
    )

    return support_levels, resistance_levels


def _cluster_levels_weighted(price_vol_pairs: list, cluster_pct: float) -> list:
    """Cluster nearby price levels using volume-weighted average.

    Returns levels sorted by strength (most touches first, then highest volume).
    """
    if not price_vol_pairs:
        return []

    sorted_pairs = sorted(price_vol_pairs, key=lambda x: x[0])
    clusters: list = [[sorted_pairs[0]]]

    for pair in sorted_pairs[1:]:
        last_price = clusters[-1][-1][0]
        if last_price > 0 and abs(pair[0] - last_price) / last_price < cluster_pct:
            clusters[-1].append(pair)
        else:
            clusters.append([pair])

    # Build results: volume-weighted average price, sorted by strength
    results = []
    for cluster in clusters:
        prices = [p[0] for p in cluster]
        volumes = [p[1] for p in cluster]
        touches = len(cluster)
        total_vol = sum(volumes)

        if total_vol > 0:
            weighted_price = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        else:
            weighted_price = float(np.mean(prices))

        # Strength = touches * avg volume weight
        strength = touches * (total_vol / max(touches, 1))
        results.append((round(weighted_price, 2), strength))

    # Sort by strength descending (strongest levels first)
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]


def detect_support_resistance_with_touches(
    df: pd.DataFrame, lookback: int = 60, cluster_pct: float = 0.008,
):
    """Like detect_support_resistance but returns (price, touch_count) tuples.

    Primary: KDE with ATR-scaled bandwidth.
    Fallback: Volume-weighted nearest-neighbor clustering.

    Returns:
        (support_levels, resistance_levels): Each is a list of
        (price, num_touches) sorted by strength (strongest first).
    """
    # Try KDE first
    try:
        kde_support, kde_resistance = detect_support_resistance_kde(df, lookback)
        if kde_support or kde_resistance:
            support_levels = [(r[0], r[1]) for r in kde_support]
            resistance_levels = [(r[0], r[1]) for r in kde_resistance]
            return support_levels, resistance_levels
    except Exception:
        pass

    # Legacy fallback
    support_data, resistance_data, _ = _collect_pivot_data(df, lookback)

    support_levels = _cluster_with_touches(
        [(p, v) for p, v in support_data], cluster_pct
    )
    resistance_levels = _cluster_with_touches(
        [(p, v) for p, v in resistance_data], cluster_pct
    )
    return support_levels, resistance_levels


def _cluster_with_touches(price_vol_pairs: list, cluster_pct: float) -> list:
    """Cluster nearby levels, return (price, touch_count) sorted by strength."""
    if not price_vol_pairs:
        return []

    sorted_pairs = sorted(price_vol_pairs, key=lambda x: x[0])
    clusters = [[sorted_pairs[0]]]

    for pair in sorted_pairs[1:]:
        last_price = clusters[-1][-1][0]
        if last_price > 0 and abs(pair[0] - last_price) / last_price < cluster_pct:
            clusters[-1].append(pair)
        else:
            clusters.append([pair])

    results = []
    for cluster in clusters:
        prices = [p[0] for p in cluster]
        volumes = [p[1] for p in cluster]
        touches = len(cluster)
        total_vol = sum(volumes)

        if total_vol > 0:
            weighted_price = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        else:
            weighted_price = float(np.mean(prices))

        strength = touches * (total_vol / max(touches, 1))
        results.append((round(weighted_price, 2), touches, strength))

    results.sort(key=lambda x: x[2], reverse=True)
    return [(r[0], r[1]) for r in results]


def detect_fibonacci_levels(df: pd.DataFrame, lookback: int = 60):
    """
    Auto-detect swing high/low and compute Fibonacci retracement levels.

    Returns:
        dict with keys: swing_high, swing_low, levels (dict of fib ratios to prices)
        or None if no valid swing found.
    """
    recent = df.iloc[-lookback:]
    swing_high_idx = recent['high'].idxmax()
    swing_low_idx = recent['low'].idxmin()

    swing_high = recent.loc[swing_high_idx, 'high']
    swing_low = recent.loc[swing_low_idx, 'low']

    if swing_high <= swing_low:
        return None

    diff = swing_high - swing_low

    # For uptrend pullback (high came after low): retracement from high
    # For downtrend pullback (low came after high): retracement from low
    if swing_high_idx > swing_low_idx:
        # Uptrend - measure retracement from high
        levels = {
            0.0: swing_high,
            0.236: swing_high - 0.236 * diff,
            0.382: swing_high - 0.382 * diff,
            0.500: swing_high - 0.500 * diff,
            0.618: swing_high - 0.618 * diff,
            0.786: swing_high - 0.786 * diff,
            1.0: swing_low,
        }
        trend = "up"
    else:
        # Downtrend - measure retracement from low
        levels = {
            0.0: swing_low,
            0.236: swing_low + 0.236 * diff,
            0.382: swing_low + 0.382 * diff,
            0.500: swing_low + 0.500 * diff,
            0.618: swing_low + 0.618 * diff,
            0.786: swing_low + 0.786 * diff,
            1.0: swing_high,
        }
        trend = "down"

    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'trend': trend,
        'levels': levels,
    }
