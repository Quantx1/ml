"""Tests for ml.regime — synthetic, fast, no network / no full-universe loads.

Synthetic 3-regime series per the module spec: concatenated segments of
~150 trading days — bull (+0.10%/day, low vol), sideways (0%/day, mid vol),
bear (-0.15%/day, high vol) — seeded, so every assertion is deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.regime.ensemble import RegimeEnsemble, apply_hysteresis, rules_states
from ml.regime.features import REGIME_FEATURES, build_regime_features
from ml.regime.hindsight import agreement_report, hindsight_labels
from ml.regime.hmm_model import RegimeHMM
from ml.regime.jump_model import JumpModel

# Segment spec: (drift/day, vol/day, construction label 0=bear/1=side/2=bull)
_SEGMENTS = [(0.001, 0.001, 2), (0.0, 0.002, 1), (-0.0015, 0.004, 0)]


def _synthetic_nifty(seg_days: int = 150, seed: int = 7, reps: int = 1):
    """Geometric price path over bull -> sideways -> bear segments (x reps)."""
    rng = np.random.default_rng(seed)
    rets, labels = [], []
    for drift, vol, lab in _SEGMENTS * reps:
        rets.append(rng.normal(drift, vol, seg_days))
        labels.extend([lab] * seg_days)
    close = 100.0 * np.exp(np.cumsum(np.concatenate(rets)))
    dates = pd.bdate_range("2018-01-02", periods=len(close))
    return pd.DataFrame({"date": dates, "close": close}), np.asarray(labels)


def _jump_X(nifty: pd.DataFrame):
    """Minimal (mean, vol) feature matrix — 21d trailing windows only, so the
    construction labels stay usable as truth (short warmup, small smear)."""
    r = nifty["close"].pct_change(fill_method=None)
    X = pd.DataFrame({"ret_21": r.rolling(21).mean(), "vol_21": r.rolling(21).std()})
    valid = X.notna().all(axis=1).to_numpy()
    return X.to_numpy()[valid], valid


# ------------------------------------------------------------- 1. jump model
def test_jump_model_recovers_synthetic_regimes():
    nifty, truth = _synthetic_nifty()
    X, valid = _jump_X(nifty)
    jm = JumpModel(n_states=3, lambda_jump=50.0).fit(X)
    agree = float((jm.states_ == truth[valid]).mean())
    assert agree >= 0.85, f"in-sample agreement {agree:.3f} < 0.85"
    # deterministic relabeling: centroid mean of feature 0 ascends bear->bull
    assert np.all(np.diff(jm.centroids_[:, 0]) > 0)


def test_jump_online_is_causal_truncation_invariant():
    nifty, _ = _synthetic_nifty()
    X, _ = _jump_X(nifty)
    jm = JumpModel(n_states=3, lambda_jump=50.0).fit(X)
    full = jm.predict_online(X)
    trunc = jm.predict_online(X[:250])
    # appending future rows must not change states for days <= t
    assert np.array_equal(full[:250], trunc)
    # and the online path actually moves between regimes (no frozen collapse)
    assert len(np.unique(full)) == 3


def test_hmm_filtered_probs_are_causal():
    nifty, _ = _synthetic_nifty()
    X, _ = _jump_X(nifty)
    hm = RegimeHMM(n_states=3, n_iter=50).fit(X)
    full = hm.filtered_probs(X)
    trunc = hm.filtered_probs(X[:200])
    # forward-only filtering: past probabilities unchanged by future rows
    np.testing.assert_allclose(full[:200], trunc, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-9)


# ------------------------------------------------------------- 2. hysteresis
def test_hysteresis_ignores_short_flip_but_accepts_sustained():
    base = [2] * 20
    two_day = np.array(base + [0] * 2 + base)
    official = apply_hysteresis(two_day, hysteresis_days=3)
    assert np.all(official == 2), "2-day raw flip must NOT switch the official state"

    five_day = np.array(base + [0] * 5 + base)
    official = apply_hysteresis(five_day, hysteresis_days=3)
    # switches on the 3rd consecutive bear day...
    assert official[22] == 0 and np.all(official[:22] == 2)
    # ...and back to bull on the 3rd consecutive bull day after the flip
    assert official[24] == 0 and official[27] == 2


# ------------------------------------------------------- 3. hindsight labels
def test_hindsight_labels_match_construction():
    nifty, truth = _synthetic_nifty()
    # thresholds scaled to the synthetic drifts (+-7% defaults suit real NIFTY;
    # +0.1%/d over 42d is ~+4.3%, so the test passes proportionate thresholds)
    labels = hindsight_labels(nifty, fwd_window=42, bear_thresh=-0.04, bull_thresh=0.025)
    assert list(labels.columns) == ["date", "state"]
    agree = float((labels["state"].to_numpy() == truth).mean())
    assert agree >= 0.90, f"hindsight agreement {agree:.3f} < 0.90"


def test_agreement_report_keys_complete():
    nifty, _ = _synthetic_nifty()
    hind = hindsight_labels(nifty, fwd_window=42, bear_thresh=-0.04, bull_thresh=0.025)
    online = hind.copy()  # a 5-day-lagged copy of hindsight as the online path
    online["state"] = online["state"].shift(5).bfill().astype(int)
    rep = agreement_report(online, hind)
    assert set(rep) == {
        "agreement_pct",
        "per_state_recall",
        "avg_detection_lag_days",
        "n_switches_online",
        "n_switches_hindsight",
        "median_regime_duration_days",
    }
    assert 0.0 <= rep["agreement_pct"] <= 100.0
    assert rep["agreement_pct"] > 90.0  # only ~5d lag per turn
    assert rep["avg_detection_lag_days"] >= 5.0  # lag is detected
    assert set(rep["per_state_recall"]) <= {"bear", "sideways", "bull"}


# --------------------------------------------------------------- 4. features
def test_build_regime_features_no_lookahead_truncation():
    nifty, _ = _synthetic_nifty(seg_days=110)  # 330d — light
    rng = np.random.default_rng(3)
    days = nifty["date"]
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": days,
                    "symbol": s,
                    "close": 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days)))),
                }
            )
            for s in ("AAA", "BBB", "CCC")
        ],
        ignore_index=True,
    )
    cut = 260
    cutoff = days.iloc[cut - 1]
    full = build_regime_features(nifty, panel)
    trunc = build_regime_features(
        nifty.iloc[:cut], panel[panel["date"] <= cutoff]
    )
    assert list(full.columns) == ["date", *REGIME_FEATURES]
    # rows <= t identical whether or not the future exists (NaN == NaN ok)
    pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), trunc)


def test_build_regime_features_breadth_fail_soft():
    nifty, _ = _synthetic_nifty(seg_days=80)
    feats = build_regime_features(nifty)  # no universe panel
    assert feats["breadth_200"].isna().all()
    assert feats["breadth_change_21"].isna().all()
    assert feats["ret_21d"].notna().sum() > 0


# --------------------------------------------------------------- 5. ensemble
def test_ensemble_end_to_end_online():
    # 6 segments x 150d = 900d; ~200d warmup leaves all 3 regimes represented
    nifty, _ = _synthetic_nifty(reps=2)
    feats = build_regime_features(nifty)
    ens = RegimeEnsemble(models=("jump", "hmm", "rules"), hysteresis_days=3)
    out = ens.fit(feats).run_online(feats)

    assert list(out.columns) == ["date", "state", "state_name", "confidence", "raw_state"]
    assert set(out["state"].unique()) <= {0, 1, 2}
    assert set(out["state_name"].unique()) <= {"bear", "sideways", "bull"}
    assert out["confidence"].between(1 / 3, 1.0).all()
    assert len(out) == len(feats.dropna(subset=ens.feature_cols_))

    # hysteresis can only remove switches, never add them
    n_official = int((out["state"].diff().fillna(0) != 0).sum())
    n_raw = int((out["raw_state"].diff().fillna(0) != 0).sum())
    assert n_official <= n_raw

    # the report wiring runs end-to-end against hindsight labels
    hind = hindsight_labels(nifty, fwd_window=42, bear_thresh=-0.04, bull_thresh=0.025)
    rep = agreement_report(out, hind)
    assert rep["n_switches_online"] == n_official


def test_rules_states_nan_warmup_is_sideways():
    nifty, _ = _synthetic_nifty(seg_days=80)
    feats = build_regime_features(nifty)
    states = rules_states(feats)
    # trend_200 undefined for the first 199 rows -> neutral sideways, no crash
    assert np.all(states[:100] == 1)
    assert set(np.unique(states)) <= {0, 1, 2}
