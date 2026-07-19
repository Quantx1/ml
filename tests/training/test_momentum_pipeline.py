import json
from datetime import date

from ml.training.trainers.momentum_lambdarank import MomentumTrainer, MomentumConfig, cached_universe
from ml.features.momentum_features import MOMENTUM_FEATURE_ORDER


def test_momentum_trainer_runs_via_spine(tmp_path):
    cfg = MomentumConfig(with_forecasts=False, start=date(2021, 1, 1), end=date(2026, 2, 1))
    t = MomentumTrainer(cfg=cfg, symbols=cached_universe(limit=8))
    res = t.train(tmp_path)
    m = res.metrics
    assert m["model"] == "momentum_lambdarank"
    # RS included (benchmark wired); price-only run => exactly the feature order
    assert m["n_features"] == len(MOMENTUM_FEATURE_ORDER)
    # the spine ran its shared stages, not just training
    assert "rank_ic_mean" in m and "eda" in m and "feature_audit" in m and "hpo" in m
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == list(MOMENTUM_FEATURE_ORDER)
    # Stage-9 results + drift baseline shipped with the artifact
    assert (tmp_path / "report.md").exists() and (tmp_path / "drift_baseline.json").exists()
    # train/serve contract still holds (booster names == feature_order.json)
    ok, why = t.serve_smoke(tmp_path)
    assert ok, why
