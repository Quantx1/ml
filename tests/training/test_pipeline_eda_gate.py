import numpy as np, pandas as pd, pytest
from ml.training.pipeline import _stage_eda, PipelineContext, PipelineError
from ml.training.specs import EngineSpec, EDASpec


def _ctx(df, feature_cols, eda):
    spec = EngineSpec(name="t", eda=eda)
    c = PipelineContext(trainer=None, spec=spec, out_dir=None)
    c.df = df; c.feature_cols = feature_cols
    return c


def test_eda_blocks_on_high_nan_feature():
    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-01", periods=100),
        "symbol": ["A"] * 100,
        "good": np.linspace(0, 1, 100),
        "mostly_nan": [np.nan] * 80 + list(np.linspace(0, 1, 20)),
        "relevance": np.arange(100) % 10, "fwd_return": np.linspace(-0.1, 0.1, 100),
    })
    ctx = _ctx(df, ["good", "mostly_nan"], EDASpec(max_nan_pct=0.50, run_ic_leakage=False))
    with pytest.raises(PipelineError) as e:
        _stage_eda(ctx)
    assert "high_nan" in str(e.value)


def test_eda_passes_clean_data_and_records_summary():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-01", periods=200),
        "symbol": ["A"] * 200,
        "f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200),
        "relevance": rng.integers(0, 10, 200), "fwd_return": rng.normal(0, 0.02, 200),
    })
    ctx = _ctx(df, ["f1", "f2"], EDASpec(min_abs_ic=0.0, run_ic_leakage=False))
    _stage_eda(ctx)
    assert "eda" in ctx.metrics and ctx.metrics["eda"]["n_features"] == 2
