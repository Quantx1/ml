from ml.training.specs import CVSpec, EvalSpec, EDASpec, EngineSpec


def test_engine_spec_defaults_for_a_ranking_engine():
    spec = EngineSpec(name="momentum_lambdarank")
    assert spec.name == "momentum_lambdarank"
    assert spec.eval.task == "ranking"
    assert spec.eval.primary_metric == "rank_ic_mean"
    assert spec.eval.min_ic == 0.02 and spec.eval.min_icir == 0.5
    assert spec.cv.n_folds == 5 and spec.cv.embargo_days == 20
    assert spec.eda.max_nan_pct == 0.50 and spec.eda.min_abs_ic == 0.005
    assert spec.label_col == "relevance" and spec.fwd_return_col == "fwd_return"
    assert spec.horizon == 20 and spec.hpo_trials == 0


def test_specs_are_overridable():
    spec = EngineSpec(
        name="x", horizon=10, hpo_trials=15,
        cv=CVSpec(n_folds=3, test_days=21),
        eval=EvalSpec(task="classification", primary_metric="f1", min_ic=0.0),
        eda=EDASpec(run_ic_leakage=False, min_class_pct=0.1),
    )
    assert spec.cv.n_folds == 3 and spec.eval.task == "classification"
    assert spec.eda.run_ic_leakage is False and spec.eda.min_class_pct == 0.1
    assert spec.hpo_trials == 15
