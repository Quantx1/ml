"""
Quant X Advanced Volume Analysis
==================================
Wyckoff-inspired volume analysis and Volume Price Analysis (VPA) signals.

Adds columns to DataFrame (bullish signals only — used by active strategies):
- volume_climax_sell: Sell exhaustion climax (potential bottom)
- stopping_volume_bull: High volume reversal at support
- vpa_no_supply: Sellers exhausted (low vol narrow bar closing above mid)
- vpa_absorption_bull: Buyers absorbing selling pressure
- accumulation_phase: OBV rising while price flat/declining

References:
- Wyckoff Method (Richard Wyckoff)
- Volume Price Analysis (Anna Coulling)
- Master the Markets (Tom Williams)
"""

import numpy as np
import pandas as pd


def compute_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all advanced volume analysis columns.

    Expects df to already have basic indicators computed (volume_sma_20,
    volume_ratio, obv, atr, is_bullish, is_bearish, body_pct, etc.).

    Returns:
        Same DataFrame with new volume analysis columns added.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume'].astype(float)

    vol_sma = df.get('volume_sma_20')
    if vol_sma is None:
        vol_sma = volume.rolling(20).mean()

    atr = df.get('atr_14')
    if atr is None:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

    bar_range = (high - low).replace(0, np.nan)
    avg_range = bar_range.rolling(20).mean()

    # =========================================================================
    # 1. VOLUME CLIMAX DETECTION
    # =========================================================================
    # Sell climax: very high volume + wide range + close near low + after decline
    # Buy climax:  very high volume + wide range + close near high + after rally

    vol_ratio = volume / vol_sma.replace(0, np.nan)
    range_ratio = bar_range / avg_range.replace(0, np.nan)
    close_position = (close - low) / bar_range  # 0=low, 1=high

    # 5-bar price trend (simple)
    trend_5 = close - close.shift(5)

    # Sell climax: extreme volume, wide bar, close in bottom 20%, after decline
    df['volume_climax_sell'] = (
        (vol_ratio > 2.0) &       # was 3.0 — too strict for NSE
        (range_ratio > 1.2) &     # was 1.5
        (close_position < 0.20) & # was 0.15 — slightly relaxed
        (trend_5 < 0)
    ).fillna(False).astype(bool)

    # =========================================================================
    # 2. STOPPING VOLUME (high vol at support with reversal)
    # =========================================================================

    prev_close = close.shift(1)
    # Stopping volume at support: downtrend bar followed by close near high
    df['stopping_volume_bull'] = (
        (vol_ratio > 1.5) &       # was 2.0
        (close_position > 0.55) & # was 0.60 — slightly relaxed
        (close > prev_close) &
        (trend_5 < 0)  # Was declining
    ).fillna(False).astype(bool)

    # =========================================================================
    # 3. VPA NO DEMAND / NO SUPPLY
    # =========================================================================
    # No Demand: narrow range up bar on below-average volume (weakness)
    # No Supply: narrow range down bar on below-average volume (strength at bottom)

    # No Supply: down bar on low vol, narrow range, close NOT at bottom — sellers exhausted
    df['vpa_no_supply'] = (
        (close < open_) &           # Down bar
        (vol_ratio < 0.7) &         # Below avg volume
        (range_ratio < 0.7) &       # Narrow range
        (close_position > 0.60)     # Close well above low (sellers couldn't push down)
    ).fillna(False).astype(bool)

    # =========================================================================
    # 4. VPA EFFORT vs RESULT
    # =========================================================================
    # Effort vs Result (EVR): high volume with proportionally low range = absorption
    # EVR = vol_ratio / range_ratio — values > 3.0 indicate effort exceeding result
    evr = vol_ratio / range_ratio.replace(0, np.nan)

    # Bullish absorption: high EVR on down bar, close above midpoint (buyers absorbing)
    df['vpa_absorption_bull'] = (
        (evr > 2.0) &              # was 3.0 — too strict for NSE
        (vol_ratio > 1.5) &
        (close < open_) &           # Down bar
        (close_position > 0.50)     # But close above midpoint
    ).fillna(False).astype(bool)

    # =========================================================================
    # 5. ACCUMULATION PHASE
    # =========================================================================
    # Accumulation: OBV rising while price is flat or declining
    # Distribution: OBV falling while price is flat or rising
    # Uses 20-bar slopes

    obv = df.get('obv')
    if obv is not None:
        # Calculate slopes over 20 bars using linear regression
        obv_slope_20 = _rolling_slope(obv, 20)
        price_slope_20 = _rolling_slope(close, 20)

        # Percentile-based normalization (60-bar rolling rank)
        # Avoids arbitrary threshold problems — adapts to each stock's own dynamics
        obv_slope_pctile = obv_slope_20.rolling(60, min_periods=30).rank(pct=True)
        price_slope_pctile = price_slope_20.rolling(60, min_periods=30).rank(pct=True)

        # Accumulation: OBV slope in top 30% while price slope in bottom 50%
        df['accumulation_phase'] = (
            (obv_slope_pctile > 0.70) &
            (price_slope_pctile < 0.50)
        ).fillna(False).astype(bool)

    else:
        df['accumulation_phase'] = False

    return df


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling linear regression slope over a window."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / x_var

    return series.rolling(window).apply(_slope, raw=True)
