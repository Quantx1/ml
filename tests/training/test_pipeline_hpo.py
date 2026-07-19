import numpy as np, pandas as pd
from ml.training.pipeline import _stage_hpo, PipelineContext
from ml.training.specs import EngineSpec
from ml.training.optuna_search import SearchSpace


class _Eng:
    name = "t"
    def make_model(self, params):
        import lightgbm as lgb
        base = dict(objective="lambdarank", n_estimators=40, verbose=-1, random_state=0)
        base.update(params or {}); return lgb.LGBMRanker(**base)
    def fit_args(self, df_tr):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False).size().to_numpy()}
    def search_space(self):
        return SearchSpace(suggest=lambda tr: {"num_leaves": tr.suggest_int("num_leaves", 7, 31)})


def _toy_df(n=400):
    days = pd.bdate_range("2021-01-01", periods=n); rng = np.random.default_rng(0); rows = []
    for i, s in enumerate("ABCDE"):
        close = 100 + np.cumsum(rng.normal(0.04 * (i + 1), 1, n))
        for d, c in zip(days, close):
            rows.append({"date": d, "symbol": s, "close": c})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"])
    df["f"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5) - 1)
    df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1)
    df["relevance"] = df.groupby("date")["fwd_return"].rank(pct=True).mul(9).round()
    return df.dropna().sort_values("date").reset_index(drop=True)


def test_hpo_sets_best_params_and_trial_count():
    eng = _Eng()
    spec = EngineSpec(name="t", horizon=5, hpo_trials=4)
    ctx = PipelineContext(trainer=eng, spec=spec, out_dir=None)
    ctx.df = _toy_df(); ctx.feature_cols = ["f"]
    _stage_hpo(ctx, n_trials=4)
    assert "num_leaves" in ctx.best_params
    assert ctx.n_hpo_trials >= 1
    assert ctx.metrics["hpo"]["optimized"] in (True, False)


def test_hpo_skipped_without_search_space():
    class NoSpace(_Eng):
        def search_space(self): return None
    spec = EngineSpec(name="t", hpo_trials=4)
    ctx = PipelineContext(trainer=NoSpace(), spec=spec, out_dir=None)
    ctx.df = _toy_df(); ctx.feature_cols = ["f"]
    _stage_hpo(ctx, n_trials=4)
    assert ctx.best_params == {} and ctx.metrics["hpo"]["optimized"] is False
