import numpy as np, pandas as pd, pytest
from ml.training.pipeline import _stage_quality, PipelineContext, PipelineError
from ml.training.specs import EngineSpec, EDASpec


def _ctx(df, cols, max_constant=5):
    spec = EngineSpec(name="t", eda=EDASpec(max_constant_features=max_constant))
    c = PipelineContext(trainer=None, spec=spec, out_dir=None)
    c.df = df; c.feature_cols = cols
    return c


def test_quality_blocks_on_too_many_dead_features():
    n = 200
    df = pd.DataFrame({"date": pd.bdate_range("2022-01-01", periods=n), "symbol": ["A"] * n})
    cols = []
    for i in range(6):
        df[f"dead{i}"] = 1.0; cols.append(f"dead{i}")
    df["live"] = np.linspace(0, 1, n); cols.append("live")
    ctx = _ctx(df, cols, max_constant=5)
    with pytest.raises(PipelineError) as e:
        _stage_quality(ctx)
    assert "dead" in str(e.value).lower()
    assert ctx.metrics["feature_audit"]["n_constant"] == 6


def test_quality_passes_live_features():
    rng = np.random.default_rng(0); n = 200
    df = pd.DataFrame({"date": pd.bdate_range("2022-01-01", periods=n), "symbol": ["A"] * n,
                       "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    ctx = _ctx(df, ["a", "b"])
    _stage_quality(ctx)
    assert ctx.metrics["feature_audit"]["n_constant"] == 0
