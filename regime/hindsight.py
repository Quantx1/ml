"""Hindsight regime labels + online-vs-hindsight agreement report.

EVALUATION ONLY — NEVER A FEATURE. ``hindsight_labels`` looks FORWARD (and
backward) around each day, i.e. it is deliberately non-causal: it encodes
what a human with full hindsight would call the regime. It exists to score
the online detectors (agreement, turning-point lag, whipsaw); feeding it —
or anything derived from it — into a model or gate is lookahead and will
fabricate backtest performance. Keep it on the eval side of the wall.

Design principles (see ``ml.regime.features``): regimes are latent, so even
these labels are a smoothed convention, not ground truth. The labeler is
simple and deterministic on purpose — an auditable yardstick, not a model.

State labels: 0 = bear, 1 = sideways, 2 = bull.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def hindsight_labels(
    nifty: pd.DataFrame,
    fwd_window: int = 42,
    bear_thresh: float = -0.07,
    bull_thresh: float = 0.07,
) -> pd.DataFrame:
    """Centered (lookahead) regime labels from +/-``fwd_window``-day returns.

    Per day t, with fwd = close[t+w]/close[t] - 1 and bwd = close[t]/close[t-w] - 1:
      * bull if fwd > ``bull_thresh``, OR the day sits inside a sustained
        uptrend segment (bwd > ``bull_thresh`` while fwd is not yet negative);
      * bear if fwd < ``bear_thresh``, OR sustained downtrend symmetrically;
      * conflicts (both fire) and everything else -> sideways.
    Window edges (fwd undefined in the last w rows, bwd in the first w) fall
    back to the side that exists. A 10-day centered median filter then removes
    single-day speckle. Deterministic; EVALUATION ONLY (see module docstring).

    Args:
        nifty: DataFrame ``['date', 'close']`` daily closes.
    Returns:
        DataFrame ``['date', 'state']`` (0 = bear, 1 = sideways, 2 = bull).
    """
    df = (
        nifty[["date", "close"]]
        .dropna()
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    c = df["close"].astype(float)
    w = int(fwd_window)
    fwd = c.shift(-w) / c - 1.0  # LOOKAHEAD by construction — eval only
    bwd = c / c.shift(w) - 1.0

    bull = (
        (fwd > bull_thresh)
        | ((bwd > bull_thresh) & (fwd > 0))
        | (fwd.isna() & (bwd > bull_thresh))
    ).to_numpy()
    bear = (
        (fwd < bear_thresh)
        | ((bwd < bear_thresh) & (fwd < 0))
        | (fwd.isna() & (bwd < bear_thresh))
    ).to_numpy()
    state = np.where(bull & ~bear, 2, np.where(bear & ~bull, 0, 1))

    smoothed = (
        pd.Series(state, dtype=float)
        .rolling(10, center=True, min_periods=1)
        .median()
        .round()
        .astype(np.int64)
    )
    return pd.DataFrame({"date": df["date"].to_numpy(), "state": smoothed.to_numpy()})


def _segments(states: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as (start_idx, end_idx_exclusive, state)."""
    segs: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            segs.append((start, i, int(states[start])))
            start = i
    return segs


def agreement_report(online_states: pd.DataFrame, hindsight: pd.DataFrame) -> Dict[str, object]:
    """Score an online (causal) regime path against hindsight labels.

    Args:
        online_states: DataFrame with ``['date', 'state']`` (e.g. the output
            of ``RegimeEnsemble.run_online``; extra columns ignored).
        hindsight: DataFrame ``['date', 'state']`` from ``hindsight_labels``.

    Returns dict with:
        * ``agreement_pct``: % of overlapping days with matching state.
        * ``per_state_recall``: {state_name: recall on hindsight days of that
          state} (NaN-free; states absent from hindsight are omitted).
        * ``avg_detection_lag_days``: mean days from each hindsight regime
          start until the online path first shows that state (a segment never
          matched counts as its full length — the worst case).
        * ``n_switches_online`` / ``n_switches_hindsight``: state-change
          counts (whipsaw diagnostics).
        * ``median_regime_duration_days``: median online segment length.
    """
    merged = online_states[["date", "state"]].merge(
        hindsight[["date", "state"]], on="date", how="inner", suffixes=("_online", "_hind")
    )
    if merged.empty:
        raise ValueError("no overlapping dates between online states and hindsight labels")
    on = merged["state_online"].to_numpy(dtype=np.int64)
    hi = merged["state_hind"].to_numpy(dtype=np.int64)

    from ml.regime.ensemble import STATE_NAMES  # noqa: PLC0415 — avoid cycle at import

    per_state_recall = {
        STATE_NAMES.get(s, str(s)): float((on[hi == s] == s).mean())
        for s in np.unique(hi)
    }

    hind_segs = _segments(hi)
    lags: list[int] = []
    for start, end, state in hind_segs[1:]:  # skip the initial segment (no "turn")
        hits = np.nonzero(on[start:end] == state)[0]
        lags.append(int(hits[0]) if len(hits) else end - start)

    online_segs = _segments(on)
    return {
        "agreement_pct": float((on == hi).mean() * 100.0),
        "per_state_recall": per_state_recall,
        "avg_detection_lag_days": float(np.mean(lags)) if lags else 0.0,
        "n_switches_online": len(online_segs) - 1,
        "n_switches_hindsight": len(hind_segs) - 1,
        "median_regime_duration_days": float(np.median([e - s for s, e, _ in online_segs])),
    }


__all__ = ["agreement_report", "hindsight_labels"]
