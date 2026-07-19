import numpy as np
import pandas as pd
from ml.labeling.ranking_labels import forward_return_quantile_labels


def _panel():
    rows = []
    prices = {"A": [10, 11, 13, 14], "B": [10, 10.5, 11, 11.2],
              "C": [10, 10.2, 10.3, 10.3], "D": [10, 9.8, 9.5, 9.4]}
    full_dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    for sym, px in prices.items():
        for d, p in zip(full_dates, px):
            rows.append({"date": d, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def test_top_symbol_gets_highest_grade():
    out = forward_return_quantile_labels(_panel(), horizon=1, n_quantiles=4)
    first = out[out["date"] == pd.Timestamp("2020-01-01")].set_index("symbol")
    assert first.loc["A", "relevance"] == 3      # top quartile
    assert first.loc["D", "relevance"] == 0      # bottom quartile
    assert "fwd_return" in out.columns


def test_absolute_not_benchmark_relative():
    out = forward_return_quantile_labels(_panel(), horizon=1, n_quantiles=4)
    a0 = out[(out.symbol == "A") & (out.date == pd.Timestamp("2020-01-01"))].iloc[0]
    assert abs(a0["fwd_return"] - (11 / 10 - 1)) < 1e-9
