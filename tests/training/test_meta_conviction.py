"""MetaConvictionTrainer end-to-end (mocked engine hooks, real spine).

Two full pipeline runs:
- informative world -> gates pass -> booster + calibration.json +
  conviction_bands.json written, calibration knots monotone in [0,1];
- shuffled world -> gates fail -> quality_pass False and NO calibration
  artifact (no-fallbacks: a failed gate ships nothing).
"""
import json

import numpy as np
import pandas as pd
import pytest

from ml.features.meta_features import META_ENGINE_MAP
from ml.training.purged_cv import PurgedCVConfig
from ml.training.trainers.meta_conviction import (
    MetaConvictionConfig,
    MetaConvictionTrainer,
)

N_DATES, N_SYMS = 400, 25


def _world(tmp_path, informative: bool, seed: int = 17):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=N_DATES)
    syms = [f"S{i}" for i in range(N_SYMS)]

    # regime: 60-day blocks bull/bear/sideways
    states = [["bull", "bear", "sideways"][(i // 60) % 3] for i in range(N_DATES)]
    regime = pd.DataFrame({"date": dates, "state_name": states,
                           "confidence": rng.uniform(0.5, 1.0, N_DATES)})
    regime_path = tmp_path / "regime.parquet"
    regime.to_parquet(regime_path, index=False)

    bull = dict(zip(dates, [s == "bull" for s in states]))
    rows = []
    for d in dates:
        preds = rng.normal(size=N_SYMS)
        pct = pd.Series(preds).rank(pct=True).to_numpy()
        for s, p, q in zip(syms, preds, pct):
            edge = 0.05 * (q - 0.45) + 0.02 * bull[d] if informative else 0.0
            rows.append({"date": d, "symbol": s, "pred": float(p),
                         "fwd_return": float(edge + rng.normal(0.0, 0.03)),
                         "fold": 0})
    preds_df = pd.DataFrame(rows)
    preds_path = tmp_path / "preds.parquet"
    preds_df.to_parquet(preds_path, index=False)

    ef_cols = (META_ENGINE_MAP["momentum"]["name_features"]
               + META_ENGINE_MAP["momentum"]["forecast_features"])
    engine_feats = pd.DataFrame(
        [{"date": d, "symbol": s, **{c: float(rng.normal()) for c in ef_cols}}
         for d in dates for s in syms])

    bench_dates = pd.bdate_range("2023-03-01", periods=N_DATES + 220)
    bench = pd.DataFrame({"date": bench_dates,
                          "close": 1000 * np.cumprod(1 + rng.normal(5e-4, 0.01,
                                                                    N_DATES + 220))})
    return preds_path, regime_path, engine_feats, bench


class _EngineStub:
    def __init__(self, feats):
        self._feats = feats

    def load_panel(self):
        return pd.DataFrame()

    def build_features(self, panel):
        return self._feats, [c for c in self._feats.columns
                             if c not in ("date", "symbol")]


def _trainer(preds_path, regime_path, engine_feats, bench, monkeypatch):
    cfg = MetaConvictionConfig(
        engine="momentum", preds_path=preds_path, regime_path=regime_path,
        cv=PurgedCVConfig(n_folds=3, test_days=40, embargo_days=20, train_days=150),
    )
    t = MetaConvictionTrainer(cfg=cfg)
    monkeypatch.setattr(t, "_engine_trainer", lambda: _EngineStub(engine_feats))
    import ml.data.benchmark as bm
    monkeypatch.setattr(bm, "load_nifty_benchmark", lambda *a, **k: bench)
    return t


def test_informative_world_passes_and_calibrates(tmp_path, monkeypatch):
    t = _trainer(*_world(tmp_path, informative=True), monkeypatch)
    out = tmp_path / "model"
    result = t.train(out)
    m = result.metrics
    assert m["meta_conviction_momentum_quality_pass"] is True
    assert m["auc_mean"] >= 0.55 and m["tercile_lift"] >= 0.05
    assert (out / "meta_conviction_momentum.txt").exists()
    calib = json.loads((out / "calibration.json").read_text())
    xs, ys = calib["x"], calib["y"]
    assert xs == sorted(xs) and ys == sorted(ys)          # isotonic: monotone
    assert 0.0 <= min(ys) and max(ys) <= 1.0
    bands = json.loads((out / "conviction_bands.json").read_text())
    assert 0.0 < bands["low_max"] <= bands["medium_max"] < 1.0


def test_shuffled_world_fails_gate_and_ships_nothing(tmp_path, monkeypatch):
    t = _trainer(*_world(tmp_path, informative=False, seed=23), monkeypatch)
    out = tmp_path / "model"
    result = t.train(out)
    m = result.metrics
    assert m["meta_conviction_momentum_quality_pass"] is False
    assert "below gate" in m["meta_conviction_momentum_quality_reason"]
    assert not (out / "calibration.json").exists()
    assert not (out / "conviction_bands.json").exists()


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="unknown engine"):
        MetaConvictionConfig(engine="positional")


def test_bad_label_mode_rejected():
    with pytest.raises(ValueError, match="label_mode"):
        MetaConvictionConfig(label_mode="absolute")


def test_excess_label_demeans_per_date(tmp_path, monkeypatch):
    """On a date where the whole market rallies (+10% everywhere), raw win
    would be 100%; the excess label must still split the cross-section."""
    preds_path, regime_path, engine_feats, bench = _world(tmp_path, True)
    t = _trainer(preds_path, regime_path, engine_feats, bench, monkeypatch)
    panel = t.load_panel()
    panel["fwd_return"] = panel["fwd_return"] + 0.10  # everyone "wins" raw
    labels = t.build_labels(panel)
    per_date_rate = labels.groupby(labels["date"])["meta_win"].mean()
    assert per_date_rate.max() < 0.75 and per_date_rate.min() > 0.25
    # raw mode on the same panel: nearly everything "wins" (3-sigma tail only)
    t.cfg.label_mode = "raw"
    raw = t.build_labels(panel)
    assert raw["meta_win"].mean() > 0.99


def test_missing_preds_gives_instructions(tmp_path):
    cfg = MetaConvictionConfig(engine="swing", preds_path=tmp_path / "nope.parquet")
    t = MetaConvictionTrainer(cfg=cfg)
    with pytest.raises(Exception, match="--dump-preds"):
        t.load_panel()


def test_embargo_floor_is_engine_horizon():
    cfg = MetaConvictionConfig(
        engine="momentum",
        cv=PurgedCVConfig(n_folds=3, test_days=40, embargo_days=5, train_days=150))
    assert cfg.cv.embargo_days == 20  # raised to the momentum label horizon
