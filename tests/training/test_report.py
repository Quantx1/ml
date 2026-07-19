import json
from ml.training.report import write_report


def test_write_report_emits_json_and_md(tmp_path):
    metrics = {
        "model": "toy", "rank_ic_mean": 0.08, "rank_icir": 2.0,
        "decile_spread_mean": 0.03, "rank_ic_per_fold": [0.07, 0.09],
        "decile_spread_per_fold": [0.02, 0.04], "n_folds": 2, "n_features": 3,
        "deflated_sharpe": 0.6, "probability_backtest_overfitting": 0.3,
        "feature_importance": {"a": 10.0, "b": 5.0, "c": 0.0},
        "toy_quality_pass": True,
    }
    paths = write_report(metrics, tmp_path, model_name="toy")
    names = {p.name for p in paths}
    assert "report.json" in names and "report.md" in names
    rj = json.loads((tmp_path / "report.json").read_text())
    assert rj["rank_ic_mean"] == 0.08
    md = (tmp_path / "report.md").read_text()
    assert "toy" in md and "rank_ic_mean" in md and "shippable" in md.lower()


def test_write_report_survives_without_matplotlib(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def no_mpl(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mpl)
    paths = write_report({"model": "toy", "rank_ic_mean": 0.0, "feature_importance": {}},
                         tmp_path, model_name="toy")
    assert (tmp_path / "report.json").exists()
