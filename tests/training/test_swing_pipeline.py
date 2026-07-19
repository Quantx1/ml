import json
from datetime import date

from ml.training.trainers.swing_lambdarank import SwingTrainer, SwingConfig, cached_universe
from ml.features.swing_features import SWING_FEATURE_ORDER


def test_swing_trainer_runs_via_spine(tmp_path):
    cfg = SwingConfig(with_forecasts=False, start=date(2021, 1, 1), end=date(2026, 2, 1))
    t = SwingTrainer(cfg=cfg, symbols=cached_universe(limit=8))
    res = t.train(tmp_path)
    m = res.metrics
    assert m["model"] == "swing_lambdarank"
    # RS included (benchmark wired); price-only run => exactly the feature order
    assert m["n_features"] == len(SWING_FEATURE_ORDER)
    # the spine ran its shared stages, not just training
    assert "rank_ic_mean" in m and "eda" in m and "feature_audit" in m and "hpo" in m
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == list(SWING_FEATURE_ORDER)
    # Stage-9 results + drift baseline shipped with the artifact
    assert (tmp_path / "report.md").exists() and (tmp_path / "drift_baseline.json").exists()
    # train/serve contract still holds (booster names == feature_order.json)
    ok, why = t.serve_smoke(tmp_path)
    assert ok, why


def test_swing_trainer_discoverable():
    # `python -m ml.training.runner --list` must show swing_lambdarank
    from ml.training.discovery import discover_sorted
    names = [t.name for t in discover_sorted()]
    assert "swing_lambdarank" in names
