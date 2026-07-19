"""Dataset-collapse guard: a poisoned feature column must fail LOUD."""
import numpy as np
import pandas as pd
import pytest

from ml.training.base import PipelineTrainer
from ml.training.pipeline import PipelineError, run_pipeline
from ml.training.specs import CVSpec, EDASpec, EngineSpec, EvalSpec


class _PoisonedRanker(PipelineTrainer):
    """One feature is NaN for ~92% of rows -> dropna collapses the dataset."""
    name = "poisoned"
    skip_promote_gate = True

    def engine_spec(self):
        return EngineSpec(name="poisoned", horizon=5,
                          cv=CVSpec(n_folds=2, test_days=40, embargo_days=5, train_days=200),
                          eval=EvalSpec(min_ic=-1.0, min_icir=-1.0),
                          eda=EDASpec(min_abs_ic=0.0, run_ic_leakage=False, max_nan_pct=0.99))

    def load_panel(self):
        days = pd.bdate_range("2021-01-01", periods=500)
        rng = np.random.default_rng(0)
        rows = []
        for i, s in enumerate("ABCDE"):
            close = 100 + np.cumsum(rng.normal(0.05, 1.0, 500))
            rows.extend({"date": d, "symbol": s, "close": c} for d, c in zip(days, close))
        return pd.DataFrame(rows)

    def build_features(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["mom_5"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5) - 1.0)
        # poisoned column: only the last ~8% of dates have values
        cutoff = df["date"].quantile(0.92)
        df["poisoned_col"] = np.where(df["date"] >= cutoff, 1.0, np.nan)
        return df[["date", "symbol", "mom_5", "poisoned_col"]], ["mom_5", "poisoned_col"]

    def build_labels(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
        df["relevance"] = df.groupby("date")["fwd_return"].rank(pct=True).mul(9).round()
        return df[["date", "symbol", "relevance", "fwd_return"]]

    def make_model(self, params):
        import lightgbm as lgb
        return lgb.LGBMRanker(objective="lambdarank", n_estimators=20, verbose=-1)

    def fit_args(self, df_tr):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False).size().to_numpy()}


def test_collapse_guard_fails_loud_and_names_culprit(tmp_path):
    with pytest.raises(PipelineError) as e:
        run_pipeline(_PoisonedRanker(), tmp_path)
    msg = str(e.value)
    assert "dataset collapse" in msg
    assert "poisoned_col" in msg   # the culprit is named
