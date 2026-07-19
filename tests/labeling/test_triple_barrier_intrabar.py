import numpy as np
import pytest
from ml.labeling.triple_barrier import triple_barrier_events, TripleBarrierConfig


def test_intrabar_high_triggers_upper_even_if_close_does_not():
    cfg = TripleBarrierConfig(profit_target_atr=2.0, stop_loss_atr=2.0,
                              vertical_barrier_days=3, min_atr_pct=0.0, asymmetric=False)
    close = np.array([100.0, 100.5, 101.0, 101.0, 101.0])
    high = np.array([100.0, 100.5, 103.0, 101.0, 101.0])   # bar 2 spikes to 103 (>= 100+2*1)
    low = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg, high=high, low=low)
    assert labels[0] == 1
    assert t1[0] == 2


def test_intrabar_low_triggers_stop():
    cfg = TripleBarrierConfig(profit_target_atr=2.0, stop_loss_atr=2.0,
                              vertical_barrier_days=3, min_atr_pct=0.0, asymmetric=False)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 97.0, 100.0, 100.0])     # bar 2 drops to 97 (<= 100-2*1)
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg, high=high, low=low)
    assert labels[0] == -1 and t1[0] == 2


def test_same_bar_double_touch_is_not_optimistically_a_win():
    # A bar that gaps through BOTH barriers in the same session must NOT be
    # auto-labeled +1 (optimistic same-bar bias). Resolve conservatively to
    # the stop (-1) since intra-bar order is unknowable from OHLC.
    cfg = TripleBarrierConfig(profit_target_atr=2.0, stop_loss_atr=2.0,
                              vertical_barrier_days=3, min_atr_pct=0.0, asymmetric=False)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 103.0, 100.0, 100.0, 100.0])   # bar 1 high >= 102 (upper)
    low = np.array([100.0, 97.0, 100.0, 100.0, 100.0])     # bar 1 low  <= 98  (lower)  -> BOTH
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg, high=high, low=low)
    assert labels[0] == -1   # conservative, NOT +1
    assert t1[0] == 1


def test_partial_high_low_raises():
    # Supplying only one of high/low must raise — a one-sided correction
    # would silently mislabel the other barrier.
    cfg = TripleBarrierConfig(min_atr_pct=0.0, vertical_barrier_days=3)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="both"):
        triple_barrier_events(close, atr, cfg, high=close)
    with pytest.raises(ValueError, match="both"):
        triple_barrier_events(close, atr, cfg, low=close)


def test_backward_compatible_without_high_low():
    cfg = TripleBarrierConfig(min_atr_pct=0.0, vertical_barrier_days=3)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg)
    assert labels.shape == close.shape
    assert (labels == 0).all()  # flat close never touches a barrier
