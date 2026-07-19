import numpy as np
import pandas as pd

from ml.training.serve_smoke import check_lgbm_feature_contract, round_trip


def _tiny_booster(tmp_path, cols):
    import lightgbm as lgb

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(80, len(cols)), columns=cols)
    y = rng.randint(0, 2, 80)
    booster = lgb.train({"objective": "binary", "verbose": -1},
                        lgb.Dataset(X, y), num_boost_round=5)
    path = tmp_path / "m.txt"
    booster.save_model(str(path))
    return path


def test_contract_passes_when_feature_order_matches(tmp_path):
    path = _tiny_booster(tmp_path, ["a", "b", "c"])
    ok, reason = check_lgbm_feature_contract(path, ["a", "b", "c"])
    assert ok, reason


def test_contract_fails_on_feature_skew(tmp_path):
    # The exact audit scenario: serving feature_order has a column the booster
    # was never trained on (15-vs-30). This MUST block promotion.
    path = _tiny_booster(tmp_path, ["a", "b", "c"])
    ok, reason = check_lgbm_feature_contract(path, ["a", "b", "c", "d"])
    assert not ok
    assert "mismatch" in reason and "d" in reason


def test_round_trip_scores_serve_features(tmp_path):
    path = _tiny_booster(tmp_path, ["a", "b", "c"])
    serve = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]})
    ok, reason = round_trip(path, ["a", "b", "c"], serve)
    assert ok, reason


def test_round_trip_fails_when_serve_builder_missing_column(tmp_path):
    path = _tiny_booster(tmp_path, ["a", "b", "c"])
    serve = pd.DataFrame({"a": [0.1], "b": [0.2]})  # missing 'c'
    ok, reason = round_trip(path, ["a", "b", "c"], serve)
    assert not ok and "missing" in reason
