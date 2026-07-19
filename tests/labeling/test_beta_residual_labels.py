"""Beta-residualized ranking labels (``beta_window``) — the fix for long-horizon
labels being dominated by ``beta * market move`` instead of stock selection.

Covers:
- with residualization, a pure-alpha name outranks a high-beta zero-alpha name
  even when the high-beta name's RAW forward return is larger (up market),
- the output ``fwd_return`` column stays RAW,
- ``beta_window`` without a benchmark raises,
- passing a benchmark with ``beta_window=None`` changes nothing,
- warmup rows (NaN rolling beta) are dropped.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml.labeling.ranking_labels import forward_return_quantile_labels

HORIZON = 10
BETA_WINDOW = 40  # min_periods = 20


def _alpha_beta_world(n: int = 160, seed: int = 11):
    """Benchmark drifts up (+40bps/day mean); five deterministic-linear names
    with DISTINCT daily alphas so the true selection ordering is well-defined
    (unshrunk rolling betas estimate exactly in this world, so each name's
    residual collapses to its compounded alpha):

        name       beta  alpha/day   true selection rank
        ALPHA      1.0   30bps       1 (best)
        HI_BETA    2.0   20bps       2
        MID_BETA   1.5   10bps       3
        MKT_BETA   1.0    5bps       4
        LO_BETA    0.5    0bps       5

    In an up market the RAW forward return ranks HI_BETA on top (2x market
    swamps 1% of alpha spread); the residual label must rank ALPHA on top."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    bench_ret = rng.normal(loc=0.004, scale=0.008, size=n)
    bench_close = 1000.0 * np.cumprod(1.0 + bench_ret)
    bench = pd.DataFrame({"date": dates, "close": bench_close})

    def _px(mult: float, alpha: float) -> np.ndarray:
        return 100.0 * np.cumprod(1.0 + mult * bench_ret + alpha)

    rows = []
    for sym, px in {
        "HI_BETA": _px(2.0, 0.002),
        "MID_BETA": _px(1.5, 0.001),
        "MKT_BETA": _px(1.0, 0.0005),
        "LO_BETA": _px(0.5, 0.0),
        "ALPHA": _px(1.0, 0.003),
    }.items():
        for d, p in zip(dates, px):
            rows.append({"date": d, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows), bench, dates


def test_residual_label_ranks_alpha_over_beta():
    panel, bench, _ = _alpha_beta_world()
    # 5 quantiles for 5 names => one name per grade, no top-bucket ties.
    raw = forward_return_quantile_labels(panel, horizon=HORIZON, n_quantiles=5)
    resid = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=5,
        benchmark=bench, beta_window=BETA_WINDOW,
    )

    def _top_share(out: pd.DataFrame, sym: str) -> float:
        wide = out.pivot(index="date", columns="symbol", values="relevance").dropna()
        return float((wide[sym] == wide.max(axis=1)).mean())

    # Raw labels: the up-market makes 2x-beta the usual winner (the failure mode).
    assert _top_share(raw, "HI_BETA") > 0.5
    # Residual labels: pure selection alpha wins nearly always.
    assert _top_share(resid, "ALPHA") > 0.9
    assert _top_share(resid, "HI_BETA") < 0.1


def test_resid_column_emitted_and_correctly_signed():
    """The extra resid_fwd_return column exists only on the beta path, and each
    name's mean residual recovers its compounded alpha (betas estimate exactly
    in this world, so the market component subtracts out entirely)."""
    panel, bench, _ = _alpha_beta_world()
    raw = forward_return_quantile_labels(panel, horizon=HORIZON, n_quantiles=3)
    assert "resid_fwd_return" not in raw.columns
    resid = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=3,
        benchmark=bench, beta_window=BETA_WINDOW,
    )
    assert "resid_fwd_return" in resid.columns
    means = resid.groupby("symbol")["resid_fwd_return"].mean()
    # ALPHA compounds ~30bps/day of selection over H=10 → ~3% residual.
    assert means["ALPHA"] > 0.02
    # Residual means recover the alpha ordering with real margins.
    assert means["ALPHA"] > means["HI_BETA"] + 0.005
    assert means["HI_BETA"] > means["MID_BETA"] + 0.005
    assert means["ALPHA"] > means["LO_BETA"] + 0.02
    # LO_BETA has zero alpha: its residual is ~0, not its beta-scaled return.
    assert abs(means["LO_BETA"]) < 0.005


def test_ic_grades_residual_target_via_ic_target_col():
    """The spine's rank-IC must grade against ic_target_col when set: a scorer
    that perfectly ranks SELECTION scores IC=1.0 on the residual target while
    its raw-return IC is degraded by the beta ordering."""
    from ml.training.pipeline import _rank_ic

    panel, bench, _ = _alpha_beta_world()
    resid = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=3,
        benchmark=bench, beta_window=BETA_WINDOW,
    )
    # A "perfect selection" scorer: the true alpha ordering (residuals
    # collapse to compounded alphas in this world, so this ranks the
    # residual target exactly).
    score = {"ALPHA": 5.0, "HI_BETA": 4.0, "MID_BETA": 3.0,
             "MKT_BETA": 2.0, "LO_BETA": 1.0}
    fp = resid.copy()
    fp["pred"] = fp["symbol"].map(score)
    ic_resid, n = _rank_ic(fp, target="resid_fwd_return")
    ic_raw, _ = _rank_ic(fp)
    assert n > 0
    assert ic_resid > ic_raw  # the raw target is beta-contaminated
    assert ic_resid > 0.5


def test_output_fwd_return_stays_raw():
    panel, bench, _ = _alpha_beta_world()
    raw = forward_return_quantile_labels(panel, horizon=HORIZON, n_quantiles=3)
    resid = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=3,
        benchmark=bench, beta_window=BETA_WINDOW,
    )
    merged = resid.merge(raw, on=["date", "symbol"], suffixes=("_resid", "_raw"))
    assert len(merged) == len(resid)
    assert np.allclose(merged["fwd_return_resid"], merged["fwd_return_raw"])


def test_beta_window_without_benchmark_raises():
    panel, _, _ = _alpha_beta_world()
    with pytest.raises(ValueError, match="benchmark"):
        forward_return_quantile_labels(panel, horizon=HORIZON, beta_window=BETA_WINDOW)


def test_beta_window_plus_vol_adjust_raises():
    """Combined config would rank the vol-scaled residual while exporting the
    unscaled residual as the eval target — the gate mismatch ic_target_col
    exists to prevent. Unsupported until the export matches the rank variable."""
    panel, bench, _ = _alpha_beta_world()
    with pytest.raises(ValueError, match="unsupported"):
        forward_return_quantile_labels(
            panel, horizon=HORIZON, benchmark=bench,
            beta_window=BETA_WINDOW, vol_adjust_window=21)


def test_benchmark_ignored_when_beta_window_none():
    panel, bench, _ = _alpha_beta_world()
    plain = forward_return_quantile_labels(panel, horizon=HORIZON, n_quantiles=3)
    with_bench = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=3, benchmark=bench)
    assert_frame_equal(plain, with_bench)


def test_warmup_rows_dropped_per_symbol():
    panel, bench, dates = _alpha_beta_world()
    raw = forward_return_quantile_labels(panel, horizon=HORIZON, n_quantiles=3)
    resid = forward_return_quantile_labels(
        panel, horizon=HORIZON, n_quantiles=3,
        benchmark=bench, beta_window=BETA_WINDOW,
    )
    min_p = max(20, BETA_WINDOW // 2)
    for sym in ("HI_BETA", "MID_BETA", "MKT_BETA", "LO_BETA", "ALPHA"):
        raw_dates = raw.loc[raw["symbol"] == sym, "date"]
        resid_dates = resid.loc[resid["symbol"] == sym, "date"]
        assert raw_dates.min() == dates[0]
        # first daily return is NaN, so the beta window needs min_p returns
        # starting from bar 1 => first labeled bar is index min_p.
        assert resid_dates.min() == dates[min_p]
        assert len(resid_dates) == len(raw_dates) - min_p
