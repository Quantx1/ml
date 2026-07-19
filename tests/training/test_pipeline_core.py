import json
import numpy as np, pandas as pd
from ml.training.base import PipelineTrainer, TrainResult
from ml.training.specs import EngineSpec, CVSpec, EvalSpec, EDASpec


def _toy_panel(n_days=500, syms=("A", "B", "C", "D", "E")):
    days = pd.bdate_range("2021-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(syms):
        close = 100 + np.cumsum(rng.normal(0.05 * (i + 1), 1.0, n_days))
        for d, c in zip(days, close):
            rows.append({"date": d, "symbol": s, "close": c})
    return pd.DataFrame(rows)


class _ToyRanker(PipelineTrainer):
    name = "toy_ranker"
    skip_promote_gate = True

    def engine_spec(self):
        return EngineSpec(
            name="toy_ranker", horizon=5,
            cv=CVSpec(n_folds=2, test_days=40, embargo_days=5, train_days=200),
            eval=EvalSpec(task="ranking", min_ic=-1.0, min_icir=-1.0),
            eda=EDASpec(min_abs_ic=0.0, run_ic_leakage=False),
        )

    def load_panel(self):
        return _toy_panel()

    def build_features(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["mom_5"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5) - 1.0)
        df["xs_rank"] = df.groupby("date")["mom_5"].rank(pct=True)
        cols = ["mom_5", "xs_rank"]
        return df[["date", "symbol", *cols]], cols

    def build_labels(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
        df["relevance"] = df.groupby("date")["fwd_return"].rank(pct=True).mul(9).round()
        return df[["date", "symbol", "relevance", "fwd_return"]]

    def make_model(self, params):
        import lightgbm as lgb
        base = dict(objective="lambdarank", metric="ndcg", n_estimators=60,
                    num_leaves=15, learning_rate=0.05, verbose=-1, random_state=0)
        base.update(params or {})
        return lgb.LGBMRanker(**base)

    def fit_args(self, df_tr):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False).size().to_numpy()}


def test_run_pipeline_ranking_end_to_end(tmp_path):
    from ml.training.pipeline import run_pipeline
    res = run_pipeline(_ToyRanker(), tmp_path)
    assert isinstance(res, TrainResult)
    m = res.metrics
    assert "rank_ic_mean" in m and "rank_ic_per_fold" in m and "n_folds" in m
    assert m["n_features"] == 2 and m["n_folds"] == 2
    names = {p.name for p in res.artifacts}
    assert "feature_order.json" in names and "metrics.json" in names
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == ["mom_5", "xs_rank"]
