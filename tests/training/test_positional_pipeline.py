import json
from datetime import date

from ml.training.purged_cv import PurgedCVConfig
from ml.training.trainers.positional_lambdarank import (
    PositionalTrainer, PositionalConfig, cached_universe,
)
from ml.features.positional_features import POSITIONAL_FEATURE_ORDER


def test_positional_trainer_runs_via_spine(tmp_path):
    # The production-default CV (train 504 + embargo 60 + 4x126 test = 1068
    # unique days) needs more post-warmup history than this offline-cache
    # smoke window holds: 2021-01-01..2026-02-01 is ~1257 trading days, and
    # the ~262-bar feature warmup + 60d label horizon leave ~935 usable days.
    # Shrink ONLY test_days for the smoke run (need = 504+60+4x63 = 816).
    cfg = PositionalConfig(
        with_forecasts=False, start=date(2021, 1, 1), end=date(2026, 2, 1),
        cv=PurgedCVConfig(n_folds=4, test_days=63, embargo_days=60, train_days=504),
    )
    t = PositionalTrainer(cfg=cfg, symbols=cached_universe(limit=8))
    res = t.train(tmp_path)
    m = res.metrics
    assert m["model"] == "positional_lambdarank"
    # RS included (benchmark wired); price-only run => exactly the feature order
    assert m["n_features"] == len(POSITIONAL_FEATURE_ORDER)
    # the spine ran its shared stages, not just training
    assert "rank_ic_mean" in m and "eda" in m and "feature_audit" in m and "hpo" in m
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == list(POSITIONAL_FEATURE_ORDER)
    # Stage-9 results + drift baseline shipped with the artifact
    assert (tmp_path / "report.md").exists() and (tmp_path / "drift_baseline.json").exists()
    # train/serve contract still holds (booster names == feature_order.json)
    ok, why = t.serve_smoke(tmp_path)
    assert ok, why


def test_positional_trainer_discoverable():
    # `python -m ml.training.runner --list` must show positional_lambdarank
    from ml.training.discovery import discover_sorted
    names = [t.name for t in discover_sorted()]
    assert "positional_lambdarank" in names
