# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for data_splitter module."""

import pandas as pd
import pytest

from src.ml.data_splitter import (
    DataSplitter,
    _validate_ratios,
    time_based_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n_dates: int = 20,
    bars_per_date: int = 10,
    with_date_col: bool = True,
    with_target: bool = True,
) -> pd.DataFrame:
    """Synthetic multi-date feature DataFrame with ascending timestamps."""
    rows = []
    base_ts = 1_740_992_400_000  # 2025-03-03 09:30 ET in ms

    for d in range(n_dates):
        date_str = f"2025-03-{d + 1:02d}"
        for b in range(bars_per_date):
            ts = base_ts + d * 86_400_000 + b * 60_000
            row = {"timestamp": ts, "feature": float(d * bars_per_date + b)}
            if with_date_col:
                row["date"] = date_str
            if with_target:
                row["target"] = 1 if b == 5 else 0
            rows.append(row)

    return pd.DataFrame(rows)


def _make_config(train_ratio: float = 0.70, val_ratio: float = 0.15) -> dict:
    return {
        "data_preparation": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        }
    }


# ---------------------------------------------------------------------------
# TestValidateRatios
# ---------------------------------------------------------------------------


class TestValidateRatios:
    def test_valid_ratios_pass(self):
        _validate_ratios(0.70, 0.15)  # should not raise

    def test_train_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="train_ratio"):
            _validate_ratios(0.0, 0.15)

    def test_val_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="val_ratio"):
            _validate_ratios(0.70, 0.0)

    def test_ratios_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="< 1.0"):
            _validate_ratios(0.70, 0.30)

    def test_ratios_exceed_one_raises(self):
        with pytest.raises(ValueError, match="< 1.0"):
            _validate_ratios(0.80, 0.30)


# ---------------------------------------------------------------------------
# TestTimeBasedSplit — date-level splitting
# ---------------------------------------------------------------------------


class TestTimeBasedSplitDateLevel:
    def test_returns_three_dataframes(self):
        df = _make_df(n_dates=20)
        result = time_based_split(df)
        assert len(result) == 3
        assert all(isinstance(r, pd.DataFrame) for r in result)

    def test_row_counts_sum_to_total(self):
        df = _make_df(n_dates=20, bars_per_date=10)
        train, val, test = time_based_split(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_data_leakage_by_date(self):
        """Dates in train must not appear in val or test."""
        df = _make_df(n_dates=20)
        train, val, test = time_based_split(df)

        train_dates = set(train["date"])
        val_dates = set(val["date"])
        test_dates = set(test["date"])

        assert train_dates.isdisjoint(val_dates)
        assert train_dates.isdisjoint(test_dates)
        assert val_dates.isdisjoint(test_dates)

    def test_chronological_order(self):
        """All training dates must be earlier than all validation dates,
        which must be earlier than all test dates."""
        df = _make_df(n_dates=20)
        train, val, test = time_based_split(df)

        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_train_is_approximately_70_pct(self):
        df = _make_df(n_dates=100, bars_per_date=5)
        train, val, test = time_based_split(df, train_ratio=0.70, val_ratio=0.15)
        train_dates = train["date"].nunique()
        assert abs(train_dates - 70) <= 2  # allow ±2 days rounding

    def test_whole_days_not_split(self):
        """Every trading day must be entirely in one set — never straddling two."""
        df = _make_df(n_dates=30, bars_per_date=8)
        train, val, test = time_based_split(df)

        for date in df["date"].unique():
            date_sets = sum([
                date in train["date"].values,
                date in val["date"].values,
                date in test["date"].values,
            ])
            assert date_sets == 1, f"Date {date} appears in more than one set"

    def test_empty_df_returns_three_empty(self):
        df = _make_df(n_dates=0)
        train, val, test = time_based_split(df)
        assert train.empty
        assert val.empty
        assert test.empty

    def test_index_reset_on_all_splits(self):
        df = _make_df(n_dates=20)
        train, val, test = time_based_split(df)
        for split in (train, val, test):
            if not split.empty:
                assert list(split.index) == list(range(len(split)))

    def test_requires_timestamp_column(self):
        df = pd.DataFrame({"date": ["2025-03-03"], "close": [4.0]})
        with pytest.raises(ValueError, match="timestamp"):
            time_based_split(df)


# ---------------------------------------------------------------------------
# TestTimeBasedSplit — row-level fallback (no 'date' column)
# ---------------------------------------------------------------------------


class TestTimeBasedSplitRowLevel:
    def test_row_counts_sum_to_total(self):
        df = _make_df(n_dates=20, with_date_col=False)
        train, val, test = time_based_split(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_train_timestamps_before_val(self):
        df = _make_df(n_dates=20, with_date_col=False)
        train, val, _ = time_based_split(df)
        assert train["timestamp"].max() < val["timestamp"].min()

    def test_val_timestamps_before_test(self):
        df = _make_df(n_dates=20, with_date_col=False)
        _, val, test = time_based_split(df)
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_custom_ratios(self):
        df = _make_df(n_dates=10, bars_per_date=10, with_date_col=False)
        # 100 rows total: 80/10/10
        train, val, test = time_based_split(df, train_ratio=0.80, val_ratio=0.10)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


# ---------------------------------------------------------------------------
# TestDataSplitter class
# ---------------------------------------------------------------------------


class TestDataSplitter:
    def test_default_ratios(self):
        splitter = DataSplitter({})
        assert splitter.train_ratio == 0.70
        assert splitter.val_ratio == 0.15
        assert abs(splitter.test_ratio - 0.15) < 1e-9

    def test_config_overrides_applied(self):
        splitter = DataSplitter(_make_config(train_ratio=0.80, val_ratio=0.10))
        assert splitter.train_ratio == 0.80
        assert splitter.val_ratio == 0.10

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            DataSplitter(_make_config(train_ratio=0.80, val_ratio=0.30))

    def test_split_delegates_to_time_based_split(self):
        df = _make_df(n_dates=20)
        splitter = DataSplitter(_make_config())
        train, val, test = splitter.split(df)
        assert not train.empty and not val.empty and not test.empty

    def test_test_ratio_property(self):
        splitter = DataSplitter(_make_config(0.70, 0.15))
        assert abs(splitter.test_ratio - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# TestSplitDates
# ---------------------------------------------------------------------------


class TestSplitDates:
    def _splitter(self):
        return DataSplitter(_make_config(0.70, 0.15))

    def test_returns_three_lists(self):
        splitter = self._splitter()
        dates = [f"2025-03-{d:02d}" for d in range(1, 21)]
        result = splitter.split_dates(dates)
        assert len(result) == 3
        assert all(isinstance(r, list) for r in result)

    def test_dates_sum_to_total(self):
        splitter = self._splitter()
        dates = [f"2025-03-{d:02d}" for d in range(1, 21)]
        train_d, val_d, test_d = splitter.split_dates(dates)
        assert len(train_d) + len(val_d) + len(test_d) == 20

    def test_chronological_order_of_date_lists(self):
        splitter = self._splitter()
        dates = [f"2025-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 4)]
        train_d, val_d, test_d = splitter.split_dates(dates)
        if train_d and val_d:
            assert max(train_d) < min(val_d)
        if val_d and test_d:
            assert max(val_d) < min(test_d)

    def test_no_overlap_between_splits(self):
        splitter = self._splitter()
        dates = [f"2025-03-{d:02d}" for d in range(1, 21)]
        train_d, val_d, test_d = splitter.split_dates(dates)
        assert set(train_d).isdisjoint(val_d)
        assert set(train_d).isdisjoint(test_d)
        assert set(val_d).isdisjoint(test_d)

    def test_empty_list_returns_three_empty_lists(self):
        splitter = self._splitter()
        train_d, val_d, test_d = splitter.split_dates([])
        assert train_d == [] and val_d == [] and test_d == []

    def test_three_dates_each_set_gets_one(self):
        splitter = self._splitter()
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        train_d, val_d, test_d = splitter.split_dates(dates)
        assert len(train_d) >= 1
        assert len(val_d) >= 1
        assert len(test_d) >= 1


# ---------------------------------------------------------------------------
# TestGetSummary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_summary_has_required_keys(self):
        df = _make_df(n_dates=20)
        splitter = DataSplitter(_make_config())
        train, val, test = splitter.split(df)
        summary = splitter.get_summary(train, val, test)

        assert "train" in summary
        assert "val" in summary
        assert "test" in summary
        assert "total_rows" in summary
        assert "configured_ratios" in summary

    def test_total_rows_correct(self):
        df = _make_df(n_dates=20)
        splitter = DataSplitter(_make_config())
        train, val, test = splitter.split(df)
        summary = splitter.get_summary(train, val, test)
        assert summary["total_rows"] == len(df)

    def test_summary_includes_positive_rate(self):
        df = _make_df(n_dates=20, with_target=True)
        splitter = DataSplitter(_make_config())
        train, val, test = splitter.split(df)
        summary = splitter.get_summary(train, val, test)
        assert "positive_rate" in summary["train"]
        assert "positive_rate" in summary["test"]

    def test_summary_includes_date_range(self):
        df = _make_df(n_dates=20)
        splitter = DataSplitter(_make_config())
        train, val, test = splitter.split(df)
        summary = splitter.get_summary(train, val, test)
        assert "date_start" in summary["train"]
        assert "date_end" in summary["test"]

    def test_configured_ratios_reported(self):
        splitter = DataSplitter(_make_config(0.70, 0.15))
        empty = pd.DataFrame(columns=["timestamp", "date"])
        summary = splitter.get_summary(empty, empty, empty)
        assert summary["configured_ratios"]["train"] == 0.70
        assert summary["configured_ratios"]["val"] == 0.15
        assert abs(summary["configured_ratios"]["test"] - 0.15) < 1e-9
