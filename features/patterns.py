"""
Quant X Chart Pattern Detection
================================
Unified detection using neurotrader888 approach (detection=confirmation).
Patterns fire at the exact bar where price confirms breakout.

Ported from: github.com/neurotrader888/TechnicalAnalysisAutomation
Adapted to output PatternResult / BreakoutSignal types for backward compatibility.

This file provides:
  - Dataclasses: ZigzagPivot, TrendLine, PatternResult, BreakoutSignal
  - Rolling Window Pivots: rw_top(), rw_bottom(), rw_extremes()
  - Perceptually Important Points: find_pips()
  - R² Quality Metric: compute_pattern_r2()
  - Unified Detectors: find_ihs_patterns(), find_double_bottoms(),
    find_triple_bottoms(), find_cup_and_handle(), find_bull_flags(),
    find_consolidation_breakouts()
  - Trade Levels: compute_trade_levels()
  - Master Scan: scan_all_patterns()
  - Public API: scan_for_patterns(), scan_for_reversal_patterns(),
                detect_breakout(), detect_reversal_breakout()

All indices are iloc-based (integer position in the DataFrame).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque


# ===================================================================
# Timeframe Scaling
# ===================================================================

_TIMEFRAME_BARS_PER_DAY = {
    '1d': 1.0,
    '1wk': 0.2,
    '4h': 1.625,
    '1h': 6.5,
    '15m': 26.0,
}


def _tf_scale(interval: str = '1d') -> float:
    """Return bars-per-day multiplier for the given candlestick timeframe."""
    return _TIMEFRAME_BARS_PER_DAY.get(interval, 1.0)


# ===================================================================
# Dataclasses
# ===================================================================

@dataclass
class ZigzagPivot:
    """A structurally significant pivot point detected by the zigzag algorithm."""
    index: int
    price: float
    pivot_type: str       # "high" or "low"
    strength: float = 0.0
    atr_multiple: float = 0.0
    volume_ratio: float = 1.0


@dataclass
class TrendLine:
    """A fitted trendline through price points."""
    slope: float
    intercept: float
    r_squared: float
    points: List[Tuple[int, float]]
    start_idx: int
    end_idx: int
    respect_ratio: float = 1.0

    @property
    def num_touches(self) -> int:
        return len(self.points)

    def value_at(self, bar_idx: int) -> float:
        return self.slope * bar_idx + self.intercept

    @property
    def pct_slope_per_bar(self) -> float:
        mid_price = self.value_at((self.start_idx + self.end_idx) // 2)
        return (self.slope / mid_price * 100) if mid_price > 0 else 0.0


@dataclass
class PatternResult:
    """A detected chart pattern."""
    pattern_type: str
    support_line: Optional['TrendLine']
    resistance_line: Optional['TrendLine']
    duration_bars: int
    breakout_level: float
    support_level: float
    pattern_height: float
    quality_score: float
    volume_declining: bool = False
    neckline_level: float = 0.0
    candle_confirmed_touches: int = 0
    respect_ratio: float = 1.0
    pivot_indices: List[Tuple[int, float, str]] = field(default_factory=list)
    _v2_confirmed: bool = False
    _v2_signal: object = None


@dataclass
class BreakoutSignal:
    """A breakout signal from a pattern."""
    pattern: PatternResult
    entry_price: float
    stop_loss: float
    target: float
    volume_ratio: float
    alpha_score: float = 0.0
    alpha_reasons: list = None
    sweep_detected: bool = False
    sweep_depth: float = 0.0
    ml_score: float = -1.0  # ML meta-label probability (-1 = not scored)
    confidence: str = 'high'  # 'high' or 'low' — set by scan_all_patterns()

    def __post_init__(self):
        if self.alpha_reasons is None:
            self.alpha_reasons = []


@dataclass
class LocalExtreme:
    """A confirmed local extreme point (from ATR directional change)."""
    ext_type: int      # 1=high, -1=low
    index: int         # bar index of the extreme
    price: float
    conf_index: int    # bar index where confirmed
    conf_price: float


# ===================================================================
# ATR-Based Directional Change (from neurotrader888/market-structure)
# ===================================================================

class ATRDirectionalChange:
    """Volatility-adaptive pivot detection. Confirms a top when price drops
    by atr_mult * ATR from pending max, bottom when price rises by atr_mult * ATR."""

    def __init__(self, atr_lookback: int = 14, atr_mult: float = 1.0):
        self._up_move = True
        self._pend_max = np.nan
        self._pend_min = np.nan
        self._pend_max_i = 0
        self._pend_min_i = 0
        self._atr_lb = atr_lookback
        self._atr_mult = atr_mult
        self._atr_sum = np.nan
        self.extremes: List[LocalExtreme] = []

    def update(self, i: int, high: np.ndarray, low: np.ndarray, close: np.ndarray):
        if i < self._atr_lb:
            return
        elif i == self._atr_lb:
            h_window = high[i - self._atr_lb + 1: i + 1]
            l_window = low[i - self._atr_lb + 1: i + 1]
            c_window = close[i - self._atr_lb: i]
            tr1 = h_window - l_window
            tr2 = np.abs(h_window - c_window)
            tr3 = np.abs(l_window - c_window)
            self._atr_sum = np.sum(np.max(np.stack([tr1, tr2, tr3]), axis=0))
        else:
            tr_val_curr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            rm_i = i - self._atr_lb
            tr_val_remove = max(
                high[rm_i] - low[rm_i],
                abs(high[rm_i] - close[rm_i - 1]),
                abs(low[rm_i] - close[rm_i - 1])
            )
            self._atr_sum += tr_val_curr
            self._atr_sum -= tr_val_remove

        atr = self._atr_sum / self._atr_lb * self._atr_mult

        if np.isnan(self._pend_max):
            self._pend_max = high[i]
            self._pend_min = low[i]
            self._pend_max_i = self._pend_min_i = i

        if self._up_move:
            if high[i] > self._pend_max:
                self._pend_max = high[i]
                self._pend_max_i = i
            elif low[i] < self._pend_max - atr:
                self.extremes.append(LocalExtreme(
                    ext_type=1, index=self._pend_max_i,
                    price=high[self._pend_max_i],
                    conf_index=i, conf_price=close[i],
                ))
                self._up_move = False
                self._pend_min = low[i]
                self._pend_min_i = i
        else:
            if low[i] < self._pend_min:
                self._pend_min = low[i]
                self._pend_min_i = i
            elif high[i] > self._pend_min + atr:
                self.extremes.append(LocalExtreme(
                    ext_type=-1, index=self._pend_min_i,
                    price=low[self._pend_min_i],
                    conf_index=i, conf_price=close[i],
                ))
                self._up_move = True
                self._pend_max = high[i]
                self._pend_max_i = i


def atr_extremes(df: pd.DataFrame, atr_lookback: int = 14, atr_mult: float = 1.0) -> Tuple[List, List]:
    """ATR-based directional change pivot detection.
    atr_mult: multiplier for ATR threshold (higher = fewer, larger pivots).
    Returns (tops, bottoms) in same format as rw_extremes: [conf_i, ext_i, ext_p]."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    dc = ATRDirectionalChange(atr_lookback, atr_mult)
    for i in range(len(close)):
        dc.update(i, high, low, close)

    tops = [[e.conf_index, e.index, e.price] for e in dc.extremes if e.ext_type == 1]
    bottoms = [[e.conf_index, e.index, e.price] for e in dc.extremes if e.ext_type == -1]
    return tops, bottoms


# ===================================================================
# Pattern type sets
# ===================================================================

_CONSOLIDATION_TYPES = {
    'ascending_triangle', 'symmetrical_triangle', 'horizontal_channel',
    'falling_wedge', 'bull_flag', 'bull_pennant',
}

_REVERSAL_TYPES = {
    'inverse_head_shoulders', 'double_bottom', 'triple_bottom',
    'cup_and_handle',
}

# Per-pattern ML thresholds (default = ml_threshold param, typically 0.35)
_ML_THRESHOLDS = {}

# Patterns with lower win rates — flagged as 'low' confidence
_LOW_CONFIDENCE_TYPES = {
    'ascending_triangle', 'horizontal_channel', 'bull_pennant',
}


# ===================================================================
# Module 1: Rolling Window Pivots (from neurotrader888)
# ===================================================================

def rw_top(data: np.ndarray, curr_index: int, order: int) -> bool:
    """Check if there is a local top confirmed at curr_index.
    The actual top is at curr_index - order."""
    if curr_index < order * 2 + 1:
        return False
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            return False
    return True


def rw_bottom(data: np.ndarray, curr_index: int, order: int) -> bool:
    """Check if there is a local bottom confirmed at curr_index.
    The actual bottom is at curr_index - order."""
    if curr_index < order * 2 + 1:
        return False
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            return False
    return True


def rw_extremes(data: np.ndarray, order: int) -> Tuple[List, List]:
    """Find all rolling-window tops and bottoms.
    Returns (tops, bottoms) where each entry = [conf_i, ext_i, ext_p]."""
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            tops.append([i, i - order, data[i - order]])
        if rw_bottom(data, i, order):
            bottoms.append([i, i - order, data[i - order]])
    return tops, bottoms


# ===================================================================
# Module 2: Perceptually Important Points (from neurotrader888)
# ===================================================================

def find_pips(data: np.ndarray, n_pips: int, dist_measure: int = 3) -> Tuple[List[int], List[float]]:
    """Find N perceptually important points using iterative max-distance insertion.
    dist_measure: 1=Euclidean, 2=Perpendicular, 3=Vertical (default)."""
    pips_x = [0, len(data) - 1]
    pips_y = [data[0], data[-1]]

    for curr_point in range(2, n_pips):
        md = 0.0
        md_i = -1
        insert_index = -1

        for k in range(0, curr_point - 1):
            left_adj = k
            right_adj = k + 1
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            if time_diff == 0:
                continue
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                if dist_measure == 1:
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                else:
                    d = abs((slope * i + intercept) - data[i])

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        if md_i == -1:
            break
        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y


# ===================================================================
# Module 3: R² Quality Metric (from neurotrader888)
# ===================================================================

def compute_pattern_r2(data: np.ndarray, pivot_indices: List[int], pivot_prices: List[float]) -> float:
    """Compute R² of piecewise-linear model through pattern pivots vs actual prices."""
    if len(pivot_indices) < 2:
        return 0.0

    start_i = pivot_indices[0]
    end_i = pivot_indices[-1]
    if end_i <= start_i:
        return 0.0

    model = np.zeros(end_i - start_i)
    for seg in range(len(pivot_indices) - 1):
        seg_start = pivot_indices[seg] - start_i
        seg_end = pivot_indices[seg + 1] - start_i
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue
        slope = (pivot_prices[seg + 1] - pivot_prices[seg]) / seg_len
        model[seg_start:seg_end] = pivot_prices[seg] + np.arange(seg_len) * slope

    raw_data = data[start_i:end_i]
    if len(raw_data) != len(model):
        return 0.0

    mean = np.mean(raw_data)
    ss_tot = np.sum((raw_data - mean) ** 2.0)
    if ss_tot == 0:
        return 0.0

    ss_res = np.sum((raw_data - model) ** 2.0)
    return 1.0 - ss_res / ss_tot


# ===================================================================
# Module 3b: Trendline Automation (from neurotrader888/TrendLineAutomation)
# ===================================================================

def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    """Validate trendline through pivot with given slope.
    Returns squared error sum, or -1.0 if line violates support/resistance constraint."""
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    return (diffs ** 2.0).sum()


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> Tuple[float, float]:
    """Optimize trendline slope using gradient descent.
    Returns (slope, intercept)."""
    slope_unit = (y.max() - y.min()) / len(y)
    min_step = 0.0001
    curr_step = 1.0

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    if best_err < 0:
        return (init_slope, -init_slope * pivot + y[pivot])

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                break

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.ndarray) -> Tuple[Tuple, Tuple]:
    """Fit support and resistance trendlines to a single data series.
    Returns ((sup_slope, sup_intercept), (res_slope, res_intercept))."""
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = int((data - line_points).argmax())
    lower_pivot = int((data - line_points).argmin())

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[Tuple, Tuple]:
    """Fit support through lows, resistance through highs.
    Returns ((sup_slope, sup_intercept), (res_slope, res_intercept))."""
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = int((high - line_points).argmax())
    lower_pivot = int((low - line_points).argmin())

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)
    return (support_coefs, resist_coefs)


# ===================================================================
# Module 4a: Inverse Head & Shoulders (from neurotrader888)
# ===================================================================

def find_ihs_patterns(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
    atr_mult: float = 2.0,
) -> List[BreakoutSignal]:
    """Find Inverse Head & Shoulders patterns using ATR-based pivots.
    Detection = confirmation: fires when close breaks above neckline."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tops_all, bottoms_all = atr_extremes(df, atr_lookback, atr_mult=atr_mult)
    # Merge and sort all extrema by confirmation index
    all_ext = []
    for t in tops_all:
        if t[1] >= start:
            all_ext.append((t[0], t[1], t[2], 1))
    for b in bottoms_all:
        if b[1] >= start:
            all_ext.append((b[0], b[1], b[2], -1))
    all_ext.sort(key=lambda x: x[0])

    if len(all_ext) < 4:
        return signals

    lock_bar = -1
    for ei in range(3, len(all_ext)):
        # Need 4 alternating extrema: bottom, top, bottom(head), top
        # or equivalently: top, bottom, top, bottom — we need the 4 ending here
        group = all_ext[ei - 3:ei + 1]

        # Check alternating types
        alternating = True
        for j in range(1, len(group)):
            if group[j][3] == group[j - 1][3]:
                alternating = False
                break
        if not alternating:
            continue

        ihs_extrema = [g[1] for g in group]  # ext indices

        # Scan forward from last extrema confirmation to find neckline break
        scan_start = group[-1][0]
        # Determine scan end: next extrema confirmation or end of data
        scan_end = all_ext[ei + 1][0] if ei + 1 < len(all_ext) else n
        scan_end = min(scan_end, n)

        for i in range(max(scan_start, lock_bar + 1), scan_end):
            sig = _check_ihs(ihs_extrema, close, high, low, i, df, 0)
            if sig is not None:
                lock_bar = i
                signals.append(sig)
                break

    return signals


def _check_ihs(
    extrema_indices: List[int],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    i: int,
    df: pd.DataFrame,
    order: int,
) -> Optional[BreakoutSignal]:
    """Check if 4 extrema + current bar form an Inverse H&S with neckline break."""
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return None

    segment = close[r_armpit + 1: i]
    if len(segment) == 0:
        return None
    r_shoulder = r_armpit + segment.argmin() + 1

    if close[head] >= min(close[l_shoulder], close[r_shoulder]):
        return None

    r_midpoint = 0.5 * (close[r_shoulder] + close[r_armpit])
    l_midpoint = 0.5 * (close[l_shoulder] + close[l_armpit])
    if close[l_shoulder] > r_midpoint or close[r_shoulder] > l_midpoint:
        return None

    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time <= 0 or l_to_h_time <= 0:
        return None
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None

    neck_run = r_armpit - l_armpit
    if neck_run == 0:
        return None
    neck_rise = close[r_armpit] - close[l_armpit]
    neck_slope = neck_rise / neck_run
    neck_val = close[l_armpit] + (i - l_armpit) * neck_slope

    if close[i] < neck_val:
        return None

    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = close[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return None
        if close[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    head_height = (close[l_armpit] + (head - l_armpit) * neck_slope) - close[head]
    # Quality: shoulder symmetry + depth + duration
    shoulder_sym = max(0, 1.0 - abs(close[l_shoulder] - close[r_shoulder]) / max(close[l_shoulder], close[r_shoulder]) / 0.05)
    depth_pct = head_height / neck_val if neck_val > 0 else 0
    depth_score = min(1.0, depth_pct / 0.08)
    dur = i - pat_start
    dur_score = min(1.0, dur / 40)
    ihs_quality = max(35, min(100, shoulder_sym * 40 + depth_score * 30 + dur_score * 30))

    neckline = TrendLine(
        slope=neck_slope,
        intercept=close[l_armpit] - neck_slope * l_armpit,
        r_squared=max(shoulder_sym, 0.0),
        points=[(l_armpit, close[l_armpit]), (r_armpit, close[r_armpit])],
        start_idx=l_armpit,
        end_idx=i,
    )

    pattern = PatternResult(
        pattern_type='inverse_head_shoulders',
        support_line=None,
        resistance_line=neckline,
        duration_bars=dur,
        breakout_level=neck_val,
        support_level=close[head],
        pattern_height=head_height,
        quality_score=ihs_quality,
        neckline_level=neck_val,
        pivot_indices=[
            (pat_start, neck_start, 'start'),
            (l_shoulder, close[l_shoulder], 'left_shoulder'),
            (l_armpit, close[l_armpit], 'left_armpit'),
            (head, close[head], 'head'),
            (r_armpit, close[r_armpit], 'right_armpit'),
            (r_shoulder, close[r_shoulder], 'right_shoulder'),
            (i, close[i], 'break'),
        ],
    )

    levels = compute_trade_levels(df, i, pattern, 'reversal')
    if levels is None:
        return None

    entry, stop, target = levels
    vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[max(0, i - 20):i].mean() if i > 0 else 1.0

    return BreakoutSignal(
        pattern=pattern,
        entry_price=entry,
        stop_loss=stop,
        target=target,
        volume_ratio=vol_ratio,
    )


# ===================================================================
# Module 4b: Double Bottom
# ===================================================================

def find_double_bottoms(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
    atr_mult: float = 2.0,
) -> List[BreakoutSignal]:
    """Find double bottom patterns: two troughs at similar levels with
    confirmation when close breaks above the peak between them."""
    close = df['close'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tops_all, bottoms_all = atr_extremes(df, atr_lookback, atr_mult=atr_mult)
    tops = [t for t in tops_all if t[1] >= start]
    bottoms = [b for b in bottoms_all if b[1] >= start]

    if len(bottoms) < 2 or len(tops) < 1:
        return signals

    lock_bar = -1
    for bi in range(1, len(bottoms)):
        b1 = bottoms[bi - 1]
        b2 = bottoms[bi]

        # Confirmation bar is when b2 is confirmed
        conf_bar = b2[0]
        if conf_bar >= n or conf_bar <= lock_bar:
            continue

        mid_top = None
        for t in reversed(tops):
            if b1[1] < t[1] < b2[1]:
                mid_top = t
                break
        if mid_top is None:
            continue

        b1_p, b2_p = b1[2], b2[2]
        neckline_p = mid_top[2]

        trough_diff = abs(b1_p - b2_p) / max(b1_p, b2_p)
        if trough_diff > 0.03:
            continue
        if neckline_p <= max(b1_p, b2_p):
            continue

        height = neckline_p - min(b1_p, b2_p)
        if height / neckline_p < 0.02:
            continue

        # Check for neckline break at or after confirmation
        break_bar = None
        for i in range(conf_bar, min(conf_bar + 20, n)):
            if close[i] >= neckline_p:
                break_bar = i
                break
        if break_bar is None:
            continue

        duration = break_bar - b1[1]
        if duration < 10:
            continue

        lock_bar = break_bar

        # Quality for reversal patterns: symmetry + depth + duration (R² doesn't work for W-shapes)
        symmetry = max(0, 1.0 - trough_diff / 0.03)  # 1.0 = perfect, 0 at 3% diff
        depth_pct = height / neckline_p
        depth_score = min(1.0, depth_pct / 0.08)  # 1.0 at 8%+ depth
        dur_score = min(1.0, duration / 40)  # 1.0 at 40+ bars
        quality = max(30, min(100, (symmetry * 40 + depth_score * 30 + dur_score * 30)))

        pattern = PatternResult(
            pattern_type='double_bottom',
            support_line=None,
            resistance_line=None,
            duration_bars=duration,
            breakout_level=neckline_p,
            support_level=min(b1_p, b2_p),
            pattern_height=height,
            quality_score=quality,
            neckline_level=neckline_p,
            pivot_indices=[
                (b1[1], b1_p, 'bottom1'),
                (mid_top[1], neckline_p, 'neckline'),
                (b2[1], b2_p, 'bottom2'),
                (break_bar, close[break_bar], 'break'),
            ],
        )

        levels = compute_trade_levels(df, break_bar, pattern, 'reversal')
        if levels is None:
            continue

        entry, stop, target = levels
        vol_ratio = df['volume'].iloc[break_bar] / df['volume'].iloc[max(0, break_bar - 20):break_bar].mean() if break_bar > 0 else 1.0

        signals.append(BreakoutSignal(
            pattern=pattern,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            volume_ratio=vol_ratio,
        ))

    return signals


# ===================================================================
# Module 4b2: Triple Bottom
# ===================================================================

def find_triple_bottoms(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
    atr_mult: float = 2.0,
) -> List[BreakoutSignal]:
    """Find triple bottom patterns: three troughs at similar levels."""
    close = df['close'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tops_all, bottoms_all = atr_extremes(df, atr_lookback, atr_mult=atr_mult)
    tops = [t for t in tops_all if t[1] >= start]
    bottoms = [b for b in bottoms_all if b[1] >= start]

    if len(bottoms) < 3 or len(tops) < 2:
        return signals

    lock_bar = -1
    for bi in range(2, len(bottoms)):
        b1, b2, b3 = bottoms[bi - 2], bottoms[bi - 1], bottoms[bi]

        conf_bar = b3[0]
        if conf_bar >= n or conf_bar <= lock_bar:
            continue

        t1, t2 = None, None
        for t in tops:
            if b1[1] < t[1] < b2[1]:
                t1 = t
            if b2[1] < t[1] < b3[1]:
                t2 = t
        if t1 is None or t2 is None:
            continue

        prices = [b1[2], b2[2], b3[2]]
        avg_trough = np.mean(prices)
        if any(abs(p - avg_trough) / avg_trough > 0.03 for p in prices):
            continue

        neckline_p = max(t1[2], t2[2])
        height = neckline_p - min(prices)
        if height / neckline_p < 0.02:
            continue

        # Check for neckline break at or after confirmation
        break_bar = None
        for i in range(conf_bar, min(conf_bar + 20, n)):
            if close[i] >= neckline_p:
                break_bar = i
                break
        if break_bar is None:
            continue

        duration = break_bar - b1[1]
        if duration < 15:
            continue

        lock_bar = break_bar

        # Quality: trough symmetry + depth + duration (R² is negative for W/M shapes)
        trough_spread = max(prices) - min(prices)
        symmetry = max(0, 1.0 - (trough_spread / avg_trough) / 0.03) if avg_trough > 0 else 0
        depth_ratio = height / neckline_p if neckline_p > 0 else 0
        depth_score = min(1.0, depth_ratio / 0.10)
        dur_score = min(1.0, duration / 60)
        quality_score = max(0, min(100, symmetry * 40 + depth_score * 30 + dur_score * 30))

        pattern = PatternResult(
            pattern_type='triple_bottom',
            support_line=None,
            resistance_line=None,
            duration_bars=duration,
            breakout_level=neckline_p,
            support_level=min(prices),
            pattern_height=height,
            quality_score=quality_score,
            neckline_level=neckline_p,
            pivot_indices=[
                (b1[1], b1[2], 'bottom1'),
                (t1[1], t1[2], 'neckline1'),
                (b2[1], b2[2], 'bottom2'),
                (t2[1], t2[2], 'neckline2'),
                (b3[1], b3[2], 'bottom3'),
                (break_bar, close[break_bar], 'break'),
            ],
        )

        levels = compute_trade_levels(df, break_bar, pattern, 'reversal')
        if levels is None:
            continue

        entry, stop, target = levels
        vol_ratio = df['volume'].iloc[break_bar] / df['volume'].iloc[max(0, break_bar - 20):break_bar].mean() if break_bar > 0 else 1.0

        signals.append(BreakoutSignal(
            pattern=pattern,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            volume_ratio=vol_ratio,
        ))

    return signals


# ===================================================================
# Module 4c: Cup and Handle
# ===================================================================

def find_cup_and_handle(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
    atr_mult: float = 2.0,
) -> List[BreakoutSignal]:
    """Find Cup & Handle: U-shape cup with shallow handle, breakout above rim."""
    close = df['close'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tops_all, bottoms_all = atr_extremes(df, atr_lookback, atr_mult=atr_mult)
    tops = [t for t in tops_all if t[1] >= start]
    bottoms = [b for b in bottoms_all if b[1] >= start]

    if len(tops) < 2 or len(bottoms) < 2:
        return signals

    lock_bar = -1
    # Check each pair of bottoms as potential cup_bottom + handle_bottom
    for bi in range(1, len(bottoms)):
        b_handle = bottoms[bi]
        if b_handle[0] <= lock_bar:
            continue

        # Find right rim: latest top before the handle bottom
        t_right = None
        for t in reversed(tops):
            if t[1] < b_handle[1]:
                t_right = t
                break
        if t_right is None:
            continue

        # Find cup bottom: a bottom before the right rim
        b_cup = None
        for b in reversed(bottoms):
            if b[1] < t_right[1]:
                b_cup = b
                break
        if b_cup is None:
            continue

        # Find left rim: a top before the cup bottom
        t_left = None
        for t in reversed(tops):
            if t[1] < b_cup[1]:
                t_left = t
                break
        if t_left is None:
            continue

        left_rim = t_left[2]
        right_rim = t_right[2]
        cup_bottom = b_cup[2]

        rim_diff = abs(left_rim - right_rim) / max(left_rim, right_rim)
        if rim_diff > 0.05:
            continue

        rim_level = max(left_rim, right_rim)
        cup_depth = rim_level - cup_bottom
        if cup_depth / rim_level < 0.05:
            continue

        handle_depth = right_rim - b_handle[2]
        if handle_depth > cup_depth * 0.5 or handle_depth < 0:
            continue

        cup_width = t_right[1] - t_left[1]
        handle_width = b_handle[1] - t_right[1]
        if handle_width > cup_width * 0.5:
            continue

        # Check for rim break after handle is confirmed
        break_bar = None
        for i in range(b_handle[0], min(b_handle[0] + 20, n)):
            if close[i] >= rim_level:
                break_bar = i
                break
        if break_bar is None:
            continue

        duration = break_bar - t_left[1]
        if duration < 20:
            continue

        lock_bar = break_bar

        # Quality: rim symmetry + cup depth + handle shallowness + duration
        rim_sym = max(0, 1.0 - rim_diff / 0.05)
        depth_ratio = cup_depth / rim_level if rim_level > 0 else 0
        depth_score = min(1.0, depth_ratio / 0.15)
        handle_ratio = handle_depth / cup_depth if cup_depth > 0 else 1.0
        handle_score = max(0, 1.0 - handle_ratio / 0.5)  # shallower handle = better
        dur_score = min(1.0, duration / 80)
        quality_score = max(0, min(100, rim_sym * 30 + depth_score * 25 + handle_score * 25 + dur_score * 20))

        pattern = PatternResult(
            pattern_type='cup_and_handle',
            support_line=None,
            resistance_line=None,
            duration_bars=duration,
            breakout_level=rim_level,
            support_level=b_handle[2],
            pattern_height=cup_depth,
            quality_score=quality_score,
            neckline_level=rim_level,
            pivot_indices=[
                (t_left[1], left_rim, 'left_rim'),
                (b_cup[1], cup_bottom, 'cup_bottom'),
                (t_right[1], right_rim, 'right_rim'),
                (b_handle[1], b_handle[2], 'handle_bottom'),
                (break_bar, close[break_bar], 'break'),
            ],
        )

        levels = compute_trade_levels(df, break_bar, pattern, 'reversal')
        if levels is None:
            continue

        entry, stop, target = levels
        vol_ratio = df['volume'].iloc[break_bar] / df['volume'].iloc[max(0, break_bar - 20):break_bar].mean() if break_bar > 0 else 1.0

        signals.append(BreakoutSignal(
            pattern=pattern,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            volume_ratio=vol_ratio,
        ))

    return signals


# ===================================================================
# Module 4d: Bull Flag / Pennant (PIP-based, from neurotrader888)
# ===================================================================

def find_bull_flags(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
) -> List[BreakoutSignal]:
    """Find bull flags and pennants using PIP-based detection with ATR pivots.
    Fires when price breaks above flag resistance."""
    close = df['close'].values
    volume = df['volume'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    _, bottoms_all = atr_extremes(df, atr_lookback)
    bottoms = [b for b in bottoms_all if b[1] >= start]

    for b in bottoms:
        base_x, base_y = b[1], b[2]
        conf_bar = b[0]

        # Scan from confirmation bar forward looking for flag breakout
        for i in range(conf_bar, min(conf_bar + 60, n)):
            data_slice = close[base_x: i + 1]
            if len(data_slice) < 5:
                continue
            max_i_rel = data_slice.argmax()
            max_i = max_i_rel + base_x

            pole_width = max_i - base_x
            if pole_width < 3:
                continue

            if i - max_i < 5:
                continue

            flag_width = i - max_i
            if flag_width > pole_width * 0.5:
                break  # Too wide, abandon this bottom

            pole_height = close[max_i] - base_y
            if pole_height <= 0:
                continue

            flag_height = close[max_i] - close[max_i:i + 1].min()
            if flag_height > pole_height * 0.5:
                continue

            if pole_height / base_y < 0.03:
                continue

            flag_data = close[max_i:i + 1]
            if len(flag_data) < 5:
                continue

            pips_x, pips_y = find_pips(flag_data, 5, 3)

            if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
                continue

            resist_run = pips_x[2] - pips_x[0]
            if resist_run == 0:
                continue
            resist_slope = (pips_y[2] - pips_y[0]) / resist_run
            resist_endpoint = pips_y[0] + resist_slope * pips_x[4]

            if pips_y[4] < resist_endpoint:
                continue

            support_run = pips_x[3] - pips_x[1]
            if support_run == 0:
                continue
            support_slope = (pips_y[3] - pips_y[1]) / support_run

            is_pennant = support_slope > 0

            # For pennants: require converging trendlines (resist slopes down, support slopes up)
            if is_pennant:
                if resist_slope >= 0:
                    continue  # Pennant needs declining resistance
                # Convergence rate: lines should narrow by at least 30% over flag width
                start_gap = abs(pips_y[0] - pips_y[1])
                end_gap = abs((pips_y[0] + resist_slope * pips_x[4]) -
                              (pips_y[1] + support_slope * pips_x[4]))
                if start_gap > 0 and end_gap / start_gap > 0.70:
                    continue  # Not converging enough
                # Pennant pole must be meaningful (>5% move)
                if pole_height / base_y < 0.05:
                    continue

            support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope
            resist_intercept = pips_y[0]
            if resist_slope != support_slope:
                intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
                if 0 <= intersection <= pips_x[4]:
                    continue
                if intersection < 0 and intersection > -flag_width:
                    continue

            vol_declining = False
            if max_i + 2 < i:
                pole_vol = volume[base_x:max_i + 1].mean() if max_i > base_x else volume[max_i]
                flag_vol = volume[max_i + 1:i + 1].mean()
                vol_declining = flag_vol < pole_vol

            resist_tl = TrendLine(
                slope=resist_slope,
                intercept=close[max_i] - resist_slope * 0,
                r_squared=0.0,
                points=[(max_i + pips_x[0], pips_y[0]), (max_i + pips_x[2], pips_y[2])],
                start_idx=max_i,
                end_idx=i,
            )

            pat_type = 'bull_pennant' if is_pennant else 'bull_flag'

            pattern = PatternResult(
                pattern_type=pat_type,
                support_line=None,
                resistance_line=resist_tl,
                duration_bars=i - base_x,
                breakout_level=resist_endpoint,
                support_level=close[max_i:i + 1].min(),
                pattern_height=pole_height,
                quality_score=60.0 + (20.0 if vol_declining else 0.0),
                volume_declining=vol_declining,
                pivot_indices=[
                    (base_x, base_y, 'pole_base'),
                    (max_i, close[max_i], 'pole_tip'),
                    (max_i + pips_x[1], pips_y[1], 'flag_low1'),
                    (max_i + pips_x[2], pips_y[2], 'flag_high'),
                    (max_i + pips_x[3], pips_y[3], 'flag_low2'),
                    (i, close[i], 'break'),
                ],
            )

            levels = compute_trade_levels(df, i, pattern, 'continuation')
            if levels is None:
                continue

            entry, stop, target = levels
            vol_ratio = volume[i] / volume[max(0, i - 20):i].mean() if i > 0 else 1.0

            signals.append(BreakoutSignal(
                pattern=pattern,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                volume_ratio=vol_ratio,
            ))
            break  # Found flag for this bottom, move to next

    return signals


# ===================================================================
# Module 4e: Consolidation Breakouts (Triangle/Wedge/Channel)
# ===================================================================

def find_bull_flags_trendline(
    df: pd.DataFrame,
    lookback: int = 250,
    atr_lookback: int = 14,
) -> List[BreakoutSignal]:
    """Find bull flags/pennants using gradient-descent trendline fitting.
    Ported from neurotrader888/flags_pennants.py trendline method."""
    close = df['close'].values
    volume = df['volume'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tops_all, bottoms_all = atr_extremes(df, atr_lookback)
    tops = [t for t in tops_all if t[1] >= start]
    bottoms = [b for b in bottoms_all if b[1] >= start]

    # For each bottom→top pair: bottom is pole base, top is pole tip
    for b in bottoms:
        # Find next top after this bottom
        pole_tip = None
        for t in tops:
            if t[1] > b[1]:
                pole_tip = t
                break
        if pole_tip is None:
            continue

        base_x, base_y = b[1], b[2]
        tip_x, tip_y = pole_tip[1], pole_tip[2]

        pole_height = tip_y - base_y
        pole_width = tip_x - base_x
        if pole_height <= 0 or pole_width < 3:
            continue
        if pole_height / base_y < 0.03:
            continue

        # Scan for flag breakout after pole tip is confirmed
        conf_start = pole_tip[0]
        for i in range(conf_start, min(conf_start + 40, n)):
            # Price shouldn't exceed pole tip in the flag area
            if i > tip_x + 1 and close[tip_x + 1:i].max() > tip_y:
                break

            flag_width = i - tip_x
            if flag_width < 5:
                continue
            if flag_width > pole_width * 0.5:
                break

            flag_min = close[tip_x:i].min()
            flag_height = tip_y - flag_min
            if flag_height > pole_height * 0.75:
                continue

            # Fit trendlines on the flag area (tip to bar before current)
            flag_data = close[tip_x:i]
            if len(flag_data) < 5:
                continue

            try:
                support_coefs, resist_coefs = fit_trendlines_single(flag_data)
            except Exception:
                continue

            sup_slope, sup_intercept = support_coefs
            res_slope, res_intercept = resist_coefs

            # Breakout: current bar above resistance trendline
            current_resist = res_intercept + res_slope * (flag_width + 1)
            if close[i] <= current_resist:
                continue

            is_pennant = sup_slope > 0

            # For pennants: require converging trendlines and meaningful pole
            if is_pennant:
                if res_slope >= 0:
                    continue  # Pennant needs declining resistance
                # Check convergence (lines narrowing)
                start_gap = abs(res_intercept - (sup_intercept + sup_slope * 0))
                end_gap = abs((res_intercept + res_slope * flag_width) -
                              (sup_intercept + sup_slope * flag_width))
                if start_gap > 0 and end_gap / start_gap > 0.70:
                    continue  # Not converging enough
                if pole_height / base_y < 0.05:
                    continue  # Pole too small for pennant

            pat_type = 'bull_pennant' if is_pennant else 'bull_flag'

            vol_declining = False
            if tip_x + 2 < i:
                pole_vol = volume[base_x:tip_x + 1].mean() if tip_x > base_x else volume[tip_x]
                flag_vol = volume[tip_x + 1:i + 1].mean()
                vol_declining = flag_vol < pole_vol

            resist_tl = TrendLine(
                slope=res_slope, intercept=res_intercept,
                r_squared=0.0,
                points=[(tip_x, res_intercept), (i, current_resist)],
                start_idx=tip_x, end_idx=i,
            )

            pattern = PatternResult(
                pattern_type=pat_type,
                support_line=None,
                resistance_line=resist_tl,
                duration_bars=i - base_x,
                breakout_level=current_resist,
                support_level=flag_min,
                pattern_height=pole_height,
                quality_score=60.0 + (20.0 if vol_declining else 0.0),
                volume_declining=vol_declining,
                pivot_indices=[
                    (base_x, base_y, 'pole_base'),
                    (tip_x, tip_y, 'pole_tip'),
                    (i, close[i], 'break'),
                ],
            )

            levels = compute_trade_levels(df, i, pattern, 'continuation')
            if levels is None:
                continue

            entry, stop, target = levels
            vol_ratio = volume[i] / volume[max(0, i - 20):i].mean() if i > 0 else 1.0

            signals.append(BreakoutSignal(
                pattern=pattern,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                volume_ratio=vol_ratio,
            ))
            break  # Found flag for this bottom, move to next

    return signals


def find_consolidation_breakouts(
    df: pd.DataFrame,
    lookback: int = 250,
    interval: str = '1d',
) -> List[BreakoutSignal]:
    """Find consolidation patterns (ascending triangle, symmetrical triangle,
    falling wedge, horizontal channel) using gradient-descent trendline fitting
    with breakout embedded in the scan loop."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(df)
    start = max(0, n - lookback)
    signals = []

    tf = _tf_scale(interval)
    scan_windows = [int(w * tf) for w in [30, 50, 80]]
    scan_windows = [w for w in scan_windows if w >= 15]

    for window in scan_windows:
        last_signal_bar = -999

        for i in range(start + window, n):
            if i - last_signal_bar < window // 2:
                continue

            seg_start = i - window
            # Fit trendlines on segment EXCLUDING current bar so breakout is possible
            seg_close = close[seg_start:i]
            seg_high = high[seg_start:i]
            seg_low = low[seg_start:i]

            if len(seg_close) < 10:
                continue

            try:
                support_coefs, resist_coefs = fit_trendlines_high_low(seg_high, seg_low, seg_close)
                sup_slope, sup_intercept = support_coefs
                res_slope, res_intercept = resist_coefs
            except Exception:
                continue

            pat_type = _classify_consolidation(sup_slope, res_slope, seg_close)
            if pat_type is None:
                continue

            # Horizontal channels need uptrend context (SMA20 > SMA50)
            if pat_type == 'horizontal_channel' and i >= 50:
                sma20 = np.mean(close[i - 20:i])
                sma50 = np.mean(close[i - 50:i])
                if sma20 <= sma50:
                    continue

            # Project trendlines to current bar (one step beyond fitted segment)
            resist_at_bar = res_intercept + res_slope * len(seg_close)
            support_at_bar = sup_intercept + sup_slope * len(seg_close)

            height = resist_at_bar - support_at_bar
            if height <= 0:
                continue

            if close[i] <= resist_at_bar:
                continue

            if i > 0 and close[i - 1] > resist_at_bar:
                continue

            avg_vol = volume[max(0, i - 20):i].mean()
            if avg_vol > 0:
                vol_ratio = volume[i] / avg_vol
                if vol_ratio < 1.0:
                    continue
            else:
                vol_ratio = 1.0

            body = abs(close[i] - df['open'].iloc[i])
            candle_range = high[i] - low[i]
            if candle_range > 0:
                body_pct = body / candle_range
                if body_pct < 0.35:
                    continue
            else:
                continue

            last_signal_bar = i

            # Convert local intercepts to absolute index space
            abs_sup_intercept = sup_intercept - sup_slope * seg_start
            abs_res_intercept = res_intercept - res_slope * seg_start

            support_tl = TrendLine(
                slope=sup_slope, intercept=abs_sup_intercept,
                r_squared=0.0, points=[],
                start_idx=seg_start, end_idx=i,
            )
            resist_tl = TrendLine(
                slope=res_slope, intercept=abs_res_intercept,
                r_squared=0.0, points=[],
                start_idx=seg_start, end_idx=i,
            )

            pattern = PatternResult(
                pattern_type=pat_type,
                support_line=support_tl,
                resistance_line=resist_tl,
                duration_bars=window,
                breakout_level=resist_at_bar,
                support_level=support_at_bar,
                pattern_height=height,
                quality_score=50.0 + min(20.0, vol_ratio * 10),
                volume_declining=False,
                pivot_indices=[
                    (seg_start, close[seg_start], 'start'),
                    (i, close[i], 'break'),
                ],
            )

            levels = compute_trade_levels(df, i, pattern, 'consolidation')
            if levels is None:
                continue

            entry, stop, target = levels

            signals.append(BreakoutSignal(
                pattern=pattern,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                volume_ratio=vol_ratio,
            ))

    return signals


def _classify_consolidation(sup_slope: float, res_slope: float, close: np.ndarray) -> Optional[str]:
    """Classify consolidation pattern type from trendline slopes."""
    price_level = np.mean(close)
    if price_level <= 0:
        return None

    sup_pct = sup_slope / price_level * 100
    res_pct = res_slope / price_level * 100

    converging = res_pct < sup_pct

    if abs(res_pct) < 0.05 and sup_pct > 0.02:
        return 'ascending_triangle'

    if converging and res_pct < -0.01 and sup_pct > 0.01:
        return 'symmetrical_triangle'

    if res_pct < -0.01 and sup_pct < -0.005 and converging:
        return 'falling_wedge'

    if abs(res_pct) < 0.05 and abs(sup_pct) < 0.05:
        return 'horizontal_channel'

    return None


# ===================================================================
# Module 5: Trade Levels (ATR-adaptive)
# ===================================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    h = df['high']
    l = df['low']
    c = df['close']
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_trade_levels(
    df: pd.DataFrame,
    bar_idx: int,
    pattern: PatternResult,
    category: str,
) -> Optional[Tuple[float, float, float]]:
    """Compute entry, stop, target using ATR-adaptive math.
    Returns (entry, stop_loss, target) or None if R:R < 1.0."""
    atr_series = _compute_atr(df)
    if bar_idx >= len(atr_series) or pd.isna(atr_series.iloc[bar_idx]):
        return None

    atr = atr_series.iloc[bar_idx]
    entry = df['close'].iloc[bar_idx]

    if atr <= 0 or entry <= 0:
        return None

    if category == 'reversal':
        # Stop below the pattern support (lowest bottom) with ATR buffer
        stop = pattern.support_level - 0.5 * atr
        # Cap at 6% below entry to avoid excessively wide stops
        if stop < entry * 0.94:
            stop = entry * 0.94
        if stop <= 0:
            stop = entry * 0.95
    elif category == 'continuation':
        stop = pattern.support_level - 0.5 * atr
        if stop <= 0:
            stop = entry * 0.95
    else:  # consolidation
        # Stop below pattern support trendline (true invalidation level)
        # breakout_level is resistance-turned-support but retests are common
        stop = pattern.support_level - 0.3 * atr
        # Cap at 5% below entry to avoid excessively wide stops
        if stop < entry * 0.95:
            stop = entry * 0.95
        if stop <= 0:
            stop = entry * 0.95

    risk = entry - stop
    if risk <= 0:
        return None

    if category == 'reversal':
        raw_target = entry + min(
            pattern.pattern_height * 0.80,  # Classic measured move (80% of height)
            4.0 * atr,
            entry * 0.12,
        )
    elif category == 'continuation':
        raw_target = entry + min(
            pattern.pattern_height * 0.5,
            2.5 * atr,
            entry * 0.08,
        )
    else:  # consolidation
        raw_target = entry + min(
            pattern.pattern_height * 0.75,
            3.0 * atr,
            entry * 0.10,
        )

    reward = raw_target - entry
    rr = reward / risk if risk > 0 else 0

    if rr < 1.0:
        return None

    return (entry, stop, raw_target)


# ===================================================================
# Module 6: Master Scan
# ===================================================================

def scan_all_patterns(
    df: pd.DataFrame,
    lookback: int = 250,
    interval: str = '1d',
    orders: List[int] = None,  # deprecated, kept for backward compat
    ml_labeler: 'BreakoutMetaLabeler' = None,
    ml_threshold: float = 0.35,
) -> List[BreakoutSignal]:
    """Master scan: run all pattern detectors and return deduplicated signals.
    Uses ATR-based pivots for volatility-adaptive detection.

    If ml_labeler is provided and trained, scores each signal with ML probability
    and filters out signals below ml_threshold. Each signal's ml_score field
    is set to the model's predicted probability of a profitable breakout.
    """
    if len(df) < 50:
        return []

    signals = []

    # Reversal patterns at two ATR scales:
    #   atr_mult=1.5 catches 1-2 month formations (medium pivots)
    #   atr_mult=2.0 catches 2-4 month formations (large pivots)
    for am in (1.5, 2.0):
        signals.extend(find_ihs_patterns(df, lookback=lookback, atr_mult=am))
        signals.extend(find_double_bottoms(df, lookback=lookback, atr_mult=am))
        signals.extend(find_triple_bottoms(df, lookback=lookback, atr_mult=am))
        signals.extend(find_cup_and_handle(df, lookback=lookback, atr_mult=am))

    # Continuation patterns (ATR-based pivots)
    signals.extend(find_bull_flags(df, lookback=lookback))
    signals.extend(find_bull_flags_trendline(df, lookback=lookback))

    # Consolidation patterns (gradient-descent trendlines)
    signals.extend(find_consolidation_breakouts(df, lookback, interval))

    deduped = _deduplicate_signals(signals)

    # Filter low-quality reversal patterns (poor R² fit)
    deduped = [s for s in deduped
               if s.pattern.quality_score >= 25
               or s.pattern.pattern_type not in _REVERSAL_TYPES]

    # ML meta-label filter with per-pattern thresholds
    if ml_labeler is not None and ml_labeler.is_trained:
        scored = []
        for sig in deduped:
            prob = ml_labeler.score_signal(df, sig, lookback=30)
            sig.ml_score = prob
            thresh = _ML_THRESHOLDS.get(sig.pattern.pattern_type, ml_threshold)
            if prob >= thresh or prob < 0:
                scored.append(sig)
        deduped = scored

    # Label confidence level
    for sig in deduped:
        if sig.pattern.pattern_type in _LOW_CONFIDENCE_TYPES:
            sig.confidence = 'low'

    return deduped


def _deduplicate_signals(signals: List[BreakoutSignal]) -> List[BreakoutSignal]:
    """Keep best signal per pattern type (highest quality score).
    Also removes signals that are too close in time (within 5 bars)."""
    if not signals:
        return []

    by_type: Dict[str, List[BreakoutSignal]] = {}
    for sig in signals:
        ptype = sig.pattern.pattern_type
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(sig)

    result = []
    for ptype, sigs in by_type.items():
        sigs.sort(key=lambda s: s.pattern.quality_score, reverse=True)

        selected = []
        for sig in sigs:
            sig_bar = sig.pattern.pivot_indices[-1][0] if sig.pattern.pivot_indices else 0
            too_close = False
            for existing in selected:
                exist_bar = existing.pattern.pivot_indices[-1][0] if existing.pattern.pivot_indices else 0
                if abs(sig_bar - exist_bar) < 5:
                    too_close = True
                    break
            if not too_close:
                selected.append(sig)
        result.extend(selected)

    return result


# ===================================================================
# Public API — backward-compatible wrappers
# ===================================================================

def scan_for_patterns(
    df: pd.DataFrame,
    lookback: int = 250,
    min_duration: int = 15,
    min_touches: int = 3,
    max_deviation_pct: float = 0.012,
    interval: str = '1d',
) -> List[PatternResult]:
    """Scan for all chart patterns (consolidation + reversal).
    Returns PatternResult list with cached BreakoutSignal in _v2_signal."""
    signals = scan_all_patterns(df, lookback=lookback, interval=interval)
    patterns = []
    for sig in signals:
        pat = sig.pattern
        pat._v2_confirmed = True
        pat._v2_signal = sig
        patterns.append(pat)
    # Deduplicate by type (keep highest quality)
    patterns.sort(key=lambda p: p.quality_score, reverse=True)
    seen = set()
    unique = []
    for p in patterns:
        if p.pattern_type not in seen:
            seen.add(p.pattern_type)
            unique.append(p)
    return unique


def scan_for_reversal_patterns(
    df: pd.DataFrame,
    lookback: int = 250,
    interval: str = '1d',
) -> List[PatternResult]:
    """Scan for reversal patterns only."""
    signals = scan_all_patterns(df, lookback=lookback, interval=interval)
    patterns = []
    for sig in signals:
        if sig.pattern.pattern_type in _REVERSAL_TYPES:
            pat = sig.pattern
            pat._v2_confirmed = True
            pat._v2_signal = sig
            patterns.append(pat)
    return patterns


def detect_breakout(
    df: pd.DataFrame,
    bar_idx: int,
    pattern: PatternResult,
    volume_mult: float = 1.15,
) -> Optional[BreakoutSignal]:
    """Check whether bar_idx is a valid breakout from pattern.
    Patterns are already confirmed — returns cached signal."""
    if pattern._v2_confirmed and pattern._v2_signal is not None:
        return pattern._v2_signal
    return None


def detect_reversal_breakout(
    df: pd.DataFrame,
    bar_idx: int,
    pattern: PatternResult,
    volume_mult: float = 1.15,
) -> Optional[BreakoutSignal]:
    """Check whether bar_idx breaks out of a reversal pattern.
    Patterns are already confirmed — returns cached signal."""
    if pattern._v2_confirmed and pattern._v2_signal is not None:
        return pattern._v2_signal
    return None


# ===================================================================
# Module 8: ML Meta-Label Breakout Filter
# ===================================================================
# Ported from: github.com/neurotrader888/TrendlineBreakoutMetaLabel
# Concept: Not all breakouts are profitable. A RandomForest classifier
# learns which breakout features predict success, filtering out weak ones.
#
# Features per breakout (all ATR-normalized):
#   resist_slope  — resistance trendline slope / ATR
#   tl_err        — avg distance from resistance line / ATR
#   max_dist      — max distance from resistance line / ATR
#   volume        — volume ratio vs 20-bar avg
#   adx           — ADX trend strength
#   quality       — pattern quality score (0-100)
#   pattern_height — pattern height / ATR
#   rr_ratio      — reward-to-risk ratio
# ===================================================================


def _compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ADX (Average Directional Index)."""
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    n = len(df)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Smoothed with Wilder's method
    atr_s = np.zeros(n)
    plus_di_s = np.zeros(n)
    minus_di_s = np.zeros(n)

    atr_s[period] = np.sum(tr[1:period + 1])
    plus_di_s[period] = np.sum(plus_dm[1:period + 1])
    minus_di_s[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        plus_di_s[i] = plus_di_s[i - 1] - plus_di_s[i - 1] / period + plus_dm[i]
        minus_di_s[i] = minus_di_s[i - 1] - minus_di_s[i - 1] / period + minus_dm[i]

    with np.errstate(invalid='ignore', divide='ignore'):
        plus_di = np.where(atr_s > 0, 100 * plus_di_s / atr_s, 0.0)
        minus_di = np.where(atr_s > 0, 100 * minus_di_s / atr_s, 0.0)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    dx = np.nan_to_num(dx, nan=0.0)

    adx = np.zeros(n)
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def compute_breakout_features(
    df: pd.DataFrame,
    signal: BreakoutSignal,
    lookback: int = 30,
) -> Optional[Dict[str, float]]:
    """Extract ML features for a single breakout signal.

    Returns a dict of ATR-normalized features, or None if data is insufficient.
    """
    # Find the breakout bar index
    pivots = signal.pattern.pivot_indices
    if not pivots:
        return None
    bar_idx = pivots[-1][0]

    if bar_idx < lookback or bar_idx >= len(df):
        return None

    atr_series = _compute_atr(df)
    if bar_idx >= len(atr_series) or pd.isna(atr_series.iloc[bar_idx]):
        return None
    atr = float(atr_series.iloc[bar_idx])
    if atr <= 0:
        return None

    close = df['close'].values
    volume = df['volume'].values

    # Fit trendlines on the lookback window before the breakout bar
    window = close[bar_idx - lookback: bar_idx]
    if len(window) < 10:
        return None

    try:
        _, r_coefs = fit_trendlines_single(window)
    except Exception:
        return None

    res_slope, res_intercept = r_coefs

    # Resistance line values over the window
    line_vals = res_intercept + np.arange(lookback) * res_slope

    # Feature 1: Resistance slope / ATR
    resist_s = res_slope / atr

    # Feature 2: Average trendline error / ATR (how tight price hugs the line)
    err = np.sum(line_vals - window) / lookback / atr

    # Feature 3: Max distance from resistance / ATR
    diff = line_vals - window
    max_dist = float(diff.max()) / atr

    # Feature 4: Volume ratio (already on the signal, but recompute for consistency)
    avg_vol = volume[max(0, bar_idx - 20):bar_idx].mean()
    vol_ratio = volume[bar_idx] / avg_vol if avg_vol > 0 else 1.0

    # Feature 5: ADX
    adx_arr = _compute_adx(df, min(14, lookback))
    adx_val = float(adx_arr[bar_idx]) if bar_idx < len(adx_arr) else 0.0

    # Feature 6: Pattern quality score
    quality = signal.pattern.quality_score

    # Feature 7: Pattern height / ATR
    height_atr = signal.pattern.pattern_height / atr if atr > 0 else 0.0

    # Feature 8: Reward-to-risk ratio
    risk = signal.entry_price - signal.stop_loss
    reward = signal.target - signal.entry_price
    rr_ratio = reward / risk if risk > 0 else 0.0

    return {
        'resist_s': resist_s,
        'tl_err': err,
        'max_dist': max_dist,
        'vol': vol_ratio,
        'adx': adx_val,
        'quality': quality,
        'height_atr': height_atr,
        'rr_ratio': rr_ratio,
    }


# Feature names in fixed order for model input
_ML_FEATURE_NAMES = ['resist_s', 'tl_err', 'max_dist', 'vol', 'adx',
                     'quality', 'height_atr', 'rr_ratio']


def _features_to_array(feat_dict: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    return np.array([feat_dict[k] for k in _ML_FEATURE_NAMES])


class BreakoutMetaLabeler:
    """ML meta-label filter for breakout signals.

    Uses a RandomForest classifier trained on historical breakout outcomes
    to predict which new breakouts will be profitable.

    Walk-forward usage:
        labeler = BreakoutMetaLabeler()
        labeler.train(X_train, y_train)
        filtered = labeler.filter_signals(df, signals, threshold=0.5)
    """

    def __init__(self, n_estimators: int = 500, max_depth: int = 3,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the RandomForest on feature matrix X and binary labels y.
        y[i] = 1 if trade was profitable, 0 otherwise."""
        from sklearn.ensemble import RandomForestClassifier

        if len(X) < 20:
            return  # not enough data to train

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict probability of profitable breakout.
        features: 1D array of shape (n_features,)."""
        if not self._is_trained:
            return -1.0
        proba = self.model.predict_proba(features.reshape(1, -1))
        # Return probability of class 1 (profitable)
        if proba.shape[1] < 2:
            return float(proba[0][0])
        return float(proba[0][1])

    def score_signal(self, df: pd.DataFrame, signal: BreakoutSignal,
                     lookback: int = 30) -> float:
        """Compute ML probability for a single signal."""
        features = compute_breakout_features(df, signal, lookback)
        if features is None:
            return -1.0
        return self.predict_proba(_features_to_array(features))

    def filter_signals(
        self, df: pd.DataFrame, signals: List[BreakoutSignal],
        threshold: float = 0.5, lookback: int = 30,
    ) -> List[BreakoutSignal]:
        """Score all signals and keep those above threshold.
        Mutates ml_score on each signal. Returns filtered list."""
        if not self._is_trained:
            return signals  # pass-through if no model

        filtered = []
        for sig in signals:
            prob = self.score_signal(df, sig, lookback)
            sig.ml_score = prob
            if prob >= threshold or prob < 0:
                # Keep signals that pass OR couldn't be scored
                filtered.append(sig)
        return filtered

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        import pickle
        if not self._is_trained:
            return
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'feature_names': _ML_FEATURE_NAMES,
            }, f)

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        import pickle
        import os
        if not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self._is_trained = True


def build_training_dataset(
    df: pd.DataFrame,
    lookback: int = 250,
    hold_period: int = 10,
    interval: str = '1d',
) -> Tuple[np.ndarray, np.ndarray, List[BreakoutSignal]]:
    """Build ML training data by scanning historical patterns and labeling outcomes.

    For each detected breakout, computes features and checks if price hit
    target before stop within hold_period bars.

    Returns (X, y, signals) where X is feature matrix, y is binary labels,
    signals is the list of BreakoutSignal objects.
    """
    signals = scan_all_patterns(df, lookback=lookback, interval=interval)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)

    X_list = []
    y_list = []
    valid_signals = []

    for sig in signals:
        features = compute_breakout_features(df, sig, lookback=30)
        if features is None:
            continue

        # Determine outcome: did price hit target before stop within hold_period?
        pivots = sig.pattern.pivot_indices
        if not pivots:
            continue
        bar_idx = pivots[-1][0]

        end_bar = min(bar_idx + hold_period, n)
        if bar_idx + 1 >= n:
            continue

        outcome = 0  # default: loss/neutral
        for j in range(bar_idx + 1, end_bar):
            if high[j] >= sig.target:
                outcome = 1  # hit target
                break
            if low[j] <= sig.stop_loss:
                outcome = 0  # hit stop
                break

        X_list.append(_features_to_array(features))
        y_list.append(outcome)
        valid_signals.append(sig)

    if not X_list:
        return np.array([]).reshape(0, len(_ML_FEATURE_NAMES)), np.array([]), []

    return np.array(X_list), np.array(y_list), valid_signals


def walkforward_train(
    stock_dfs: Dict[str, pd.DataFrame],
    lookback: int = 250,
    hold_period: int = 10,
    interval: str = '1d',
) -> BreakoutMetaLabeler:
    """Train a meta-labeler from multiple stocks' data.

    Args:
        stock_dfs: dict mapping symbol -> DataFrame with OHLCV data.
        lookback: pattern lookback window.
        hold_period: bars to hold each trade for outcome labeling.
        interval: timeframe interval.

    Returns a trained BreakoutMetaLabeler.
    """
    all_X = []
    all_y = []

    for sym, df in stock_dfs.items():
        if len(df) < 100:
            continue
        try:
            X, y, _ = build_training_dataset(df, lookback, hold_period, interval)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception:
            continue

    if not all_X:
        return BreakoutMetaLabeler()

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    labeler = BreakoutMetaLabeler()
    labeler.train(X_combined, y_combined)
    return labeler


# ===================================================================
# Legacy stubs — kept for backward compatibility of imports
# ===================================================================

def detect_peaks_troughs(df: pd.DataFrame, min_distance: int = 5,
                         prominence_pct: float = 0.008):
    """Legacy stub — returns pivots using rolling window."""
    close = df['close'].values
    order = max(3, min_distance)
    tops, bottoms = rw_extremes(close, order)
    peaks = [(t[1], t[2]) for t in tops]
    troughs = [(b[1], b[2]) for b in bottoms]
    return peaks, troughs


def _zigzag_pivots(df: pd.DataFrame, atr_multiplier: float = 1.0) -> List[ZigzagPivot]:
    """Legacy stub — returns ZigzagPivot list using rolling window."""
    close = df['close'].values
    order = max(3, int(6 * atr_multiplier))
    tops, bottoms = rw_extremes(close, order)
    pivots = []
    for t in tops:
        pivots.append(ZigzagPivot(index=t[1], price=t[2], pivot_type='high'))
    for b in bottoms:
        pivots.append(ZigzagPivot(index=b[1], price=b[2], pivot_type='low'))
    pivots.sort(key=lambda p: p.index)
    return pivots


def _get_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Legacy stub — compute ATR."""
    return _compute_atr(df, period)


def _check_trend_alignment(df: pd.DataFrame, bar_idx: int) -> bool:
    """Legacy stub — check if SMA50 > SMA200 (uptrend)."""
    if bar_idx >= len(df):
        return False
    row = df.iloc[bar_idx]
    sma50 = row.get('sma_50', None)
    sma200 = row.get('sma_200', None)
    if sma50 is None or sma200 is None:
        return True
    if pd.isna(sma50) or pd.isna(sma200):
        return True
    return float(sma50) > float(sma200)
