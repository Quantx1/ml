import json
from pathlib import Path

from ml.training.trainers.momentum_lambdarank import (
    MomentumConfig,
    cached_universe,
    train_momentum,
)
from ml.training.purged_cv import PurgedCVConfig


def test_momentum_trains_end_to_end_and_saves_artifact(tmp_path):
    syms = cached_universe(limit=15)
    assert len(syms) >= 10, "needs the local data/cache CSVs present"
    cfg = MomentumConfig(cv=PurgedCVConfig(n_folds=2, test_days=63, embargo_days=20, train_days=300))
    out = tmp_path / "mom"
    metrics = train_momentum(cfg=cfg, symbols=syms, out_dir=out)

    # produced real metrics + a finite OOS rank-IC
    for key in ("rank_ic_mean", "rank_icir", "decile_spread_mean", "n_folds", "n_rows"):
        assert key in metrics
    assert metrics["n_folds"] == 2
    assert metrics["n_rows"] > 0

    # artifact + feature-order sidecar written (serving contract)
    assert (out / "momentum_lambdarank.txt").exists()
    order = json.loads((out / "feature_order.json").read_text())
    assert isinstance(order, list) and len(order) == metrics["n_features"]
    assert (out / "metrics.json").exists()
