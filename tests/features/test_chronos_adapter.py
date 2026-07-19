"""Chronos-2 forecast adapter: fake-pipeline values, shapes, and early-exit.

The real model never loads here — ``BaseChronosPipeline.from_pretrained`` is
monkeypatched in the ``chronos`` module namespace (the adapter's lazy
``from chronos import BaseChronosPipeline`` resolves the same class object).
"""
import numpy as np
import pandas as pd
import pytest
import torch

import chronos

from ml.features.forecast_features import (
    CHRONOS_FEATURES,
    _rebalance_dates,
    chronos_forecast_features,
)


def _panel(n=30, syms=("AAA", "BBB"), close=100.0):
    """Constant-close panel so q50/last_close and (q90-q10)/last_close are exact."""
    days = pd.bdate_range("2025-01-01", periods=n)
    rows = [{"date": d, "symbol": s, "open": close, "high": close + 1,
             "low": close - 1, "close": close, "volume": 1000}
            for s in syms for d in days]
    return pd.DataFrame(rows)


class _FakePipeline:
    """predict_quantiles returning known quantiles: q10=99, q50=105, q90=101
    against last_close=100 -> chronos_fwd_ret=0.05, chronos_uncert=0.02."""

    def __init__(self, as_list: bool):
        self.as_list = as_list

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        assert quantile_levels == [0.1, 0.5, 0.9]
        one = torch.tensor([[99.0, 105.0, 101.0]] * prediction_length)  # (h, 3)
        mean = torch.full((len(inputs), prediction_length), 105.0)
        if self.as_list:
            # Chronos2Pipeline REAL shape (live-probed on 2.2.2): a list of
            # per-series tensors each with a LEADING BATCH DIM — (1, h, 3).
            # The smoke run caught this after the mock originally emitted
            # (h, 3); keep the mock faithful to the real API.
            return [one.clone().unsqueeze(0) for _ in inputs], mean
        return torch.stack([one] * len(inputs)), mean  # classic stacked tensor


@pytest.mark.parametrize("as_list", [True, False], ids=["list-of-tensors", "stacked-tensor"])
def test_chronos_values_columns_and_row_count(monkeypatch, as_list):
    called = {}

    def _fake_from_pretrained(model_id, device_map=None, **kw):
        called["model_id"] = model_id
        return _FakePipeline(as_list=as_list)

    monkeypatch.setattr(chronos.BaseChronosPipeline, "from_pretrained", _fake_from_pretrained)

    panel = _panel(n=30)
    out = chronos_forecast_features(panel, horizon=10, stride=5, min_history=10)

    assert called["model_id"] == "amazon/chronos-2"
    assert list(out.columns) == ["date", "symbol", *CHRONOS_FEATURES]

    # expected rows = rebalance dates with >= min_history bars, x symbols
    rdates = _rebalance_dates(panel["date"].to_numpy(), 5)
    uniq = sorted(pd.unique(panel["date"].to_numpy()))
    eligible = [d for d in rdates if uniq.index(d) + 1 >= 10]
    assert len(out) == len(eligible) * 2
    assert set(out["symbol"]) == {"AAA", "BBB"}

    # q50/last_close - 1 = 105/100 - 1 ; (q90-q10)/last_close = (101-99)/100
    assert np.allclose(out["chronos_fwd_ret"], 0.05)
    assert np.allclose(out["chronos_uncert"], 0.02)


def test_chronos_min_date_early_exit_skips_model_load(monkeypatch):
    loaded = {"called": False}

    def _boom(*a, **k):
        loaded["called"] = True
        raise AssertionError("from_pretrained must NOT be called on early exit")

    monkeypatch.setattr(chronos.BaseChronosPipeline, "from_pretrained", _boom)

    panel = _panel(n=30)
    beyond = panel["date"].max() + pd.Timedelta(days=30)
    out = chronos_forecast_features(panel, horizon=10, stride=5,
                                    min_history=10, min_date=beyond)

    assert loaded["called"] is False
    assert out.empty
    assert list(out.columns) == ["date", "symbol", *CHRONOS_FEATURES]
