"""Classification evaluation branch of the training spine (meta-labeling).

Covers:
- separable synthetic data passes the AUC/tercile/Brier gates,
- label-shuffled data fails the gate (and says why),
- fold frames carry P(class 1) probabilities, not hard classes,
- the ranking path is untouched when task != classification.
"""
import numpy as np
import pandas as pd
import pytest

from ml.training.pipeline import (
    PipelineContext,
    _classification_fold_stats,
    _evaluate_classification,
    _stage_evaluation,
)
from ml.training.specs import CVSpec, EngineSpec, EvalSpec


def _spec(**eval_kwargs) -> EngineSpec:
    return EngineSpec(
        name="meta_test", horizon=20, label_col="label",
        fwd_return_col="net_fwd_return",
        cv=CVSpec(n_folds=3, test_days=30, embargo_days=5, train_days=120),
        eval=EvalSpec(task="classification", min_auc=0.55, min_fold_auc=0.5,
                      min_tercile_lift=0.05, require_brier_beats_climatology=True,
                      **eval_kwargs),
    )


def _fold_frame(n: int, informative: bool, seed: int) -> pd.DataFrame:
    """Pseudo fold-preds frame. informative=True: pred is a noisy but real
    probability of the label; False: pred is independent noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.55).astype(float)
    if informative:
        pred = np.clip(0.55 + 0.35 * (y - 0.5) * 2 + rng.normal(0, 0.12, n), 0.01, 0.99)
    else:
        pred = rng.uniform(0.01, 0.99, n)
    dates = pd.bdate_range("2025-01-01", periods=max(1, n // 20))
    return pd.DataFrame({
        "date": np.resize(dates.to_numpy(), n),
        "symbol": [f"S{i % 20}" for i in range(n)],
        "pred": pred, "fwd_return": rng.normal(0.01, 0.05, n), "label": y,
    })


def _ctx(spec: EngineSpec, frames) -> PipelineContext:
    ctx = PipelineContext(trainer=None, spec=spec, out_dir=None)
    ctx.fold_preds = list(frames)
    return ctx


def test_informative_predictions_pass_gates():
    ctx = _ctx(_spec(), [_fold_frame(600, True, s) for s in (1, 2, 3)])
    _evaluate_classification(ctx)
    m = ctx.metrics
    assert m["primary_metric"] == "auc_mean"
    assert m["auc_mean"] > 0.75
    assert m["tercile_lift"] > 0.05
    assert m["brier_mean"] <= m["brier_climatology"]
    assert m["meta_test_quality_pass"] is True


def test_noise_predictions_fail_gates_with_reason():
    ctx = _ctx(_spec(), [_fold_frame(600, False, s) for s in (4, 5, 6)])
    _evaluate_classification(ctx)
    m = ctx.metrics
    assert abs(m["auc_mean"] - 0.5) < 0.08
    assert m["meta_test_quality_pass"] is False
    assert "below gate" in m["meta_test_quality_reason"]


def test_stage_evaluation_routes_by_task():
    """task='classification' must route to the classification branch (no
    rank_ic keys); the ranking branch must never emit auc keys."""
    ctx = _ctx(_spec(), [_fold_frame(600, True, s) for s in (7, 8, 9)])
    _stage_evaluation(ctx)
    assert "auc_mean" in ctx.metrics and "rank_ic_mean" not in ctx.metrics


def test_fold_stats_single_class_and_small_folds():
    """Single-class folds give NaN AUC (excluded from means, never a crash);
    tiny folds skip terciles."""
    df = _fold_frame(40, True, 10)
    df["label"] = 1.0
    s = _classification_fold_stats(df)
    assert s["auc"] != s["auc"]  # NaN
    tiny = _classification_fold_stats(_fold_frame(10, True, 11))
    assert tiny["tercile_win_rates"] is None


def test_crossfit_calibration_rescues_miscalibrated_ranker():
    """Well-ranked but badly calibrated raw scores (crammed near 1.0): the
    RAW Brier is terrible, but the gate judges cross-fitted isotonic output —
    the shipped pipeline — which restores calibration. Gate must pass."""
    frames = []
    for s in (12, 13, 14):
        f = _fold_frame(600, True, s)
        f["pred"] = 0.90 + 0.09 * f["pred"]  # order kept, calibration destroyed
        frames.append(f)
    ctx = _ctx(_spec(), frames)
    _evaluate_classification(ctx)
    m = ctx.metrics
    assert m["auc_mean"] > 0.75
    assert m["brier_mean"] > m["brier_climatology"]                 # raw: awful
    assert m["brier_calibrated_crossfit"] <= m["brier_climatology"]  # fixed OOS
    assert m["meta_test_quality_pass"] is True


def test_crossfit_brier_needs_two_folds():
    from ml.training.pipeline import _crossfit_calibrated_brier
    only = _crossfit_calibrated_brier([_fold_frame(600, True, 15)])
    assert only != only  # NaN -> gate fails when required (honest, not a crash)


@pytest.mark.parametrize("informative", [True])
def test_ranking_metrics_unchanged_for_ranking_task(informative):
    """Regression guard: a ranking-task context still produces rank-IC keys
    exactly as before (the classification branch must not leak)."""
    spec = EngineSpec(name="rank_test", horizon=5, label_col="relevance",
                      fwd_return_col="fwd_return",
                      eval=EvalSpec(task="ranking", min_ic=0.0, min_icir=0.0))
    rng = np.random.default_rng(21)
    frames = []
    for _ in range(3):
        n = 400
        dates = pd.bdate_range("2025-02-01", periods=20)
        fwd = rng.normal(0.01, 0.05, n)
        frames.append(pd.DataFrame({
            "date": np.resize(dates.to_numpy(), n),
            "symbol": [f"S{i % 20}" for i in range(n)],
            "pred": fwd + rng.normal(0, 0.05, n),
            "fwd_return": fwd,
        }))
    ctx = _ctx(spec, frames)
    ctx.n_hpo_trials = 1
    _stage_evaluation(ctx)
    assert "rank_ic_mean" in ctx.metrics and "auc_mean" not in ctx.metrics
    assert ctx.metrics["rank_ic_mean"] > 0.3
