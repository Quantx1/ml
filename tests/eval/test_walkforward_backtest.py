"""Synthetic-engine tests for walkforward_portfolio_backtest (no disk, no net).

A tiny duck-typed trainer over 5 symbols x 500 business days where symbol S0
is engineered to persistently outperform and one feature identifies it. The
harness must (a) find the winner OOS (net Sharpe > 0), (b) charge costs
(gross >= net), (c) score a zero-signal engine (labels shuffled within each
date) below the winner.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.eval.walkforward_backtest import walkforward_portfolio_backtest
from ml.training.specs import CVSpec, EngineSpec

N_SYMBOLS = 5
N_DAYS = 500
HORIZON = 5
WINNER = "S0"

EXPECTED_KEYS = {
    "engine", "horizon", "n_folds", "n_test_dates", "top_n", "cost_bps_side",
    "gross", "net", "excess", "deflated_sharpe_net", "per_fold_net_sharpe",
    "avg_daily_turnover", "cost_bps_per_day", "notes",
}


def _panel(seed: int = 7) -> pd.DataFrame:
    """5 symbols x 500 days; S0 drifts +0.4%/day, the rest are driftless."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    frames = []
    for j in range(N_SYMBOLS):
        drift = 0.004 if f"S{j}" == WINNER else 0.0
        rets = drift + rng.normal(0.0, 0.01, N_DAYS)
        frames.append(pd.DataFrame({
            "date": dates, "symbol": f"S{j}",
            "close": 100.0 * np.cumprod(1.0 + rets),
        }))
    return pd.concat(frames, ignore_index=True)


def _labels(panel: pd.DataFrame, shuffle: bool = False, seed: int = 11) -> pd.DataFrame:
    """H-day fwd_return + per-date rank relevance; shuffle=True breaks the
    feature->label link (labels permuted jointly within each date)."""
    df = panel.sort_values(["symbol", "date"]).copy()
    df["fwd_return"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(-HORIZON) / s - 1.0)
    df = df.dropna(subset=["fwd_return"])
    df["relevance"] = (df.groupby("date")["fwd_return"]
                         .rank(method="first").astype(int) - 1)
    if shuffle:
        rng = np.random.default_rng(seed)
        parts = []
        for _, g in df.groupby("date", sort=True):
            g = g.copy()
            perm = rng.permutation(len(g))
            g[["relevance", "fwd_return"]] = g[["relevance", "fwd_return"]].to_numpy()[perm]
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
    return df[["date", "symbol", "relevance", "fwd_return"]]


class SyntheticTrainer:
    """Duck-typed PipelineTrainer: just the hooks the harness uses."""

    name = "synthetic_engine"

    def __init__(self, panel: pd.DataFrame, labels: pd.DataFrame):
        self._panel = panel
        self._labels = labels

    def engine_spec(self) -> EngineSpec:
        return EngineSpec(
            name=self.name, horizon=HORIZON,
            label_col="relevance", fwd_return_col="fwd_return",
            cv=CVSpec(n_folds=4, test_days=30, embargo_days=5, train_days=200),
        )

    def load_panel(self) -> pd.DataFrame:
        return self._panel

    def build_features(self, panel: pd.DataFrame):
        rng = np.random.default_rng(3)
        feats = panel[["date", "symbol"]].copy()
        feats["sig_alpha"] = ((panel["symbol"] == WINNER).astype(float)
                              + rng.normal(0.0, 0.1, len(panel)))
        feats["sig_noise"] = rng.normal(0.0, 1.0, len(panel))
        return feats, ["sig_alpha", "sig_noise"]

    def build_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        return self._labels

    def make_model(self, params):
        import lightgbm as lgb  # noqa: PLC0415
        p = dict(objective="lambdarank", n_estimators=30, num_leaves=7,
                 min_child_samples=5, learning_rate=0.1, random_state=0,
                 n_jobs=1, verbose=-1)
        p.update(params or {})
        return lgb.LGBMRanker(**p)

    def fit_args(self, df_tr: pd.DataFrame):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False)
                                      .size().to_numpy()}


@pytest.fixture(scope="module")
def winner_result() -> dict:
    panel = _panel()
    trainer = SyntheticTrainer(panel, _labels(panel))
    return walkforward_portfolio_backtest(trainer, top_n=1, cost_bps_side=30.0)


def test_result_keys_complete_and_jsonb_safe(winner_result):
    assert EXPECTED_KEYS <= set(winner_result)
    json.dumps(winner_result)  # JSONB-safe: no NaN/inf/ndarray/Timestamp
    assert winner_result["engine"] == "synthetic_engine"
    assert winner_result["horizon"] == HORIZON
    assert winner_result["top_n"] == 1
    assert winner_result["avg_daily_turnover"] == pytest.approx(1.0 / HORIZON)
    # 4 folds x 30 test days each
    assert winner_result["n_folds"] == 4
    assert winner_result["n_test_dates"] == 4 * 30
    assert 0.0 <= winner_result["deflated_sharpe_net"] <= 1.0
    for leg in ("gross", "net", "excess"):
        assert {"sharpe", "cagr", "max_drawdown_pct", "calmar",
                "daily_win_rate"} <= set(winner_result[leg])


def test_persistent_winner_is_found_net_of_costs(winner_result):
    # S0 outperforms by ~0.4%/day; net of 12 bps/day cost the top-1 book
    # must still show clearly positive OOS Sharpe.
    assert winner_result["net"]["sharpe"] > 0
    assert winner_result["excess"]["sharpe"] > 0


def test_costs_bite_gross_over_net(winner_result):
    assert winner_result["gross"]["sharpe"] >= winner_result["net"]["sharpe"]
    assert (winner_result["gross"]["total_return_pct"]
            > winner_result["net"]["total_return_pct"])
    assert winner_result["cost_bps_per_day"] == pytest.approx(
        2 * 30.0 / HORIZON, rel=1e-6)


def test_per_fold_series_matches_fold_count(winner_result):
    assert len(winner_result["per_fold_net_sharpe"]) == winner_result["n_folds"]
    assert all(isinstance(x, float) for x in winner_result["per_fold_net_sharpe"])


def test_zero_signal_engine_scores_below_winner(winner_result):
    panel = _panel()
    shuffled = SyntheticTrainer(panel, _labels(panel, shuffle=True))
    rand = walkforward_portfolio_backtest(shuffled, top_n=1, cost_bps_side=30.0)
    # A label-shuffled engine has no alpha: its |net Sharpe| must be sensible
    # (well below the engineered winner's), not another strong positive.
    assert abs(rand["net"]["sharpe"]) < winner_result["net"]["sharpe"]
    assert rand["net"]["sharpe"] < winner_result["net"]["sharpe"] / 2


def test_iid_stats_present_and_conservative(winner_result):
    """Non-overlap evaluation: independent samples, present + conservative."""
    r = winner_result
    assert "net_iid" in r and "excess_iid" in r and "deflated_sharpe_net_iid" in r
    ni = r["net_iid"]
    assert ni["n_periods"] > 0
    # every H-th date => roughly n_test_dates / H periods (fold boundaries allow slack)
    assert ni["n_periods"] <= r["n_test_dates"] // r["horizon"] + r["n_folds"]
    # the engineered winner should still show positive iid sharpe
    assert ni["sharpe"] > 0
    # iid sharpe must not exceed the overlap-smoothed sharpe materially
    assert ni["sharpe"] <= r["net"]["sharpe"] * 1.10


def test_dump_preds_writes_meta_training_parquet(tmp_path):
    """--dump-preds contract: per-name OOS predictions with realized returns,
    one row per (fold, date, symbol) — the meta-labeling training set."""
    panel = _panel()
    trainer = SyntheticTrainer(panel, _labels(panel))
    path = tmp_path / "preds.parquet"
    res = walkforward_portfolio_backtest(
        trainer, top_n=1, cost_bps_side=30.0, dump_preds_path=path)
    assert res["fold_preds_path"] == str(path)
    out = pd.read_parquet(path)
    assert list(out.columns) == ["date", "symbol", "pred", "fwd_return", "fold"]
    assert out["fold"].nunique() == res["n_folds"]
    assert int(out.groupby("fold")["date"].nunique().sum()) == res["n_test_dates"]
    # full cross-section every test date, nothing NaN/NaT
    assert (out.groupby(["fold", "date"]).size() == panel["symbol"].nunique()).all()
    assert out["date"].notna().all() and out["pred"].notna().all()
    assert out["fwd_return"].notna().all()
