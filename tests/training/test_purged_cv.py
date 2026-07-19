import numpy as np
import pandas as pd
from ml.training.purged_cv import purged_walk_forward_by_date, PurgedCVConfig


def _dates():
    days = pd.bdate_range("2020-01-01", periods=100)
    rows = [d for d in days for _ in range(3)]   # 3 symbols/day → 300 rows
    return pd.Series(pd.to_datetime(rows), name="date")


def test_no_symbol_date_splits_across_boundary():
    dates = _dates()
    cfg = PurgedCVConfig(n_folds=3, test_days=10, embargo_days=5, train_days=40)
    for train_idx, test_idx in purged_walk_forward_by_date(dates, cfg):
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        assert train_dates.isdisjoint(test_dates)
        # Embargo magnitude (not just disjointness) in EVERY fold: the
        # business-day gap strictly between last-train and first-test must
        # be >= embargo_days (the purged window).
        n_business_between = (
            len(pd.bdate_range(max(train_dates), min(test_dates))) - 2
        )
        assert n_business_between >= 5


def test_embargo_measured_in_days_not_rows():
    dates = _dates()
    cfg = PurgedCVConfig(n_folds=2, test_days=10, embargo_days=5, train_days=40)
    folds = list(purged_walk_forward_by_date(dates, cfg))
    train_idx, test_idx = folds[0]
    last_train_day = dates.iloc[train_idx].max()
    first_test_day = dates.iloc[test_idx].min()
    n_business_between = len(pd.bdate_range(last_train_day, first_test_day)) - 2
    assert n_business_between >= 5
