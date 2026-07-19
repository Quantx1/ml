import json
import numpy as np, pandas as pd
from ml.training.baseline_drift import write_baseline


def test_baseline_records_per_feature_stats(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
    p = write_baseline(df, ["a", "b"], tmp_path)
    assert p.name == "drift_baseline.json" and p.exists()
    base = json.loads(p.read_text())
    assert set(base["features"]) == {"a", "b"}
    assert abs(base["stats"]["b"]["mean"] - 5) < 0.5
    assert "p10" in base["stats"]["a"] and "p90" in base["stats"]["a"]
    assert base["n_rows"] == 500
