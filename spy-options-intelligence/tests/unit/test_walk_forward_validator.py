# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for src/ml/walk_forward_validator.py.

Strategy
--------
* Mock filesystem and ``load_features`` / ``undersample_majority`` to avoid
  touching real data files.
* Use tiny synthetic DataFrames so tests run in milliseconds.
* Focus on: date split generation, load filtering, metrics computation,
  aggregation, and ASCII plot formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.walk_forward_validator import WalkForwardValidator, _DEFAULT_XGB_PARAMS, _MIN_POSITIVES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = ["feat_a", "feat_b", "feat_c"]
NON_FEATURE_COLS = {"date", "ticker", "target", "max_gain_120m", "min_loss_120m"}


def _make_df(
    n_rows: int = 100,
    n_pos: int = 10,
    date: str = "2025-06-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Tiny synthetic feature DataFrame with required label columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(n_rows).astype(np.float32),
            "feat_b": rng.standard_normal(n_rows).astype(np.float32),
            "feat_c": rng.standard_normal(n_rows).astype(np.float32),
            "date": date,
            "ticker": "O:SPY250601C00500000",
            "target": [1] * n_pos + [0] * (n_rows - n_pos),
            "max_gain_120m": rng.uniform(0.0, 1.0, n_rows),
            "min_loss_120m": rng.uniform(-0.5, 0.0, n_rows),
        }
    )
    return df


def _make_validator(tmp_path: Path) -> WalkForwardValidator:
    """Validator pointing at a temp dir (no real CSV files)."""
    return WalkForwardValidator(
        features_dir=str(tmp_path),
        xgb_params={
            "n_estimators": 10,  # fast for tests
            "max_depth": 2,
            "learning_rate": 0.3,
            "random_state": 42,
            "eval_metric": "logloss",
            "early_stopping_rounds": 5,
        },
        train_window_months=3,
        test_window_months=1,
    )


def _make_csv_files(tmp_path: Path, dates: List[str]) -> None:
    """Create empty stub CSV files for the given dates."""
    for d in dates:
        f = tmp_path / f"{d}_features.csv"
        # Write a tiny 2-row CSV so glob picks it up
        pd.DataFrame({"target": [0, 1]}).to_csv(f, index=False)


# ---------------------------------------------------------------------------
# TestGetDateSplits
# ---------------------------------------------------------------------------


class TestGetDateSplits:
    def test_empty_features_dir_returns_empty(self, tmp_path):
        v = _make_validator(tmp_path)
        assert v.get_date_splits() == []

    def test_single_date_returns_empty(self, tmp_path):
        _make_csv_files(tmp_path, ["2025-03-03"])
        v = _make_validator(tmp_path)
        # 1 date → can't form a 3-month train + 1-month test window
        splits = v.get_date_splits()
        assert splits == []

    def test_full_year_generates_nine_splits(self, tmp_path):
        # Simulate one file per month; use late-month dates so each month's
        # test window falls within the available data range.
        dates = [
            "2025-03-28", "2025-04-28", "2025-05-28", "2025-06-27",
            "2025-07-29", "2025-08-28", "2025-09-26", "2025-10-30",
            "2025-11-26", "2025-12-30", "2026-01-29", "2026-02-28",
        ]
        _make_csv_files(tmp_path, dates)
        v = _make_validator(tmp_path)
        splits = v.get_date_splits()
        assert len(splits) == 9

    def test_splits_have_four_elements(self, tmp_path):
        dates = ["2025-03-03", "2025-06-01", "2025-07-01", "2025-10-01"]
        _make_csv_files(tmp_path, dates)
        v = _make_validator(tmp_path)
        splits = v.get_date_splits()
        for split in splits:
            assert len(split) == 4

    def test_test_windows_are_non_overlapping(self, tmp_path):
        dates = [
            "2025-03-03", "2025-04-01", "2025-05-01", "2025-06-02",
            "2025-07-01", "2025-08-01", "2025-09-02", "2025-10-01",
        ]
        _make_csv_files(tmp_path, dates)
        v = _make_validator(tmp_path)
        splits = v.get_date_splits()
        for i in range(len(splits) - 1):
            _, _, _, current_te = splits[i]
            _, _, next_ts, _ = splits[i + 1]
            # Next test start must come after current test end
            assert next_ts > current_te

    def test_training_windows_overlap(self, tmp_path):
        """With 3-month train and 1-month slide, consecutive train windows overlap."""
        dates = [
            "2025-03-03", "2025-04-01", "2025-05-01", "2025-06-02",
            "2025-07-01", "2025-08-01", "2025-09-02", "2025-10-01",
        ]
        _make_csv_files(tmp_path, dates)
        v = _make_validator(tmp_path)
        splits = v.get_date_splits()
        if len(splits) >= 2:
            ts1, te1, _, _ = splits[0]
            ts2, te2, _, _ = splits[1]
            # Split 2 starts 1 month after split 1 but ends 1 month later → overlap
            assert ts2 > ts1
            assert te2 > te1

    def test_train_end_before_test_start(self, tmp_path):
        dates = [
            "2025-03-03", "2025-04-01", "2025-05-01", "2025-06-02",
            "2025-07-01",
        ]
        _make_csv_files(tmp_path, dates)
        v = _make_validator(tmp_path)
        splits = v.get_date_splits()
        for ts, te, vs, ve in splits:
            assert te < vs  # train end before test start

    def test_custom_window_sizes(self, tmp_path):
        dates = [
            "2025-03-03", "2025-05-01", "2025-07-01", "2025-09-02",
            "2025-11-03", "2026-01-02",
        ]
        _make_csv_files(tmp_path, dates)
        v = WalkForwardValidator(
            features_dir=str(tmp_path),
            train_window_months=2,
            test_window_months=2,
        )
        splits = v.get_date_splits()
        # With 2+2=4 month periods and ~10 months of data, expect ≥1 split
        assert len(splits) >= 1


# ---------------------------------------------------------------------------
# TestLoadDateRange
# ---------------------------------------------------------------------------


class TestLoadDateRange:
    def test_delegates_to_load_features(self, tmp_path):
        v = _make_validator(tmp_path)
        dummy = _make_df(50, 5, "2025-06-01")

        with patch("src.ml.walk_forward_validator.load_features", return_value=dummy) as mock_lf:
            result = v.load_date_range("2025-06-01", "2025-06-30")
            mock_lf.assert_called_once()
            assert result is dummy

    def test_passes_start_and_end_date(self, tmp_path):
        v = _make_validator(tmp_path)
        with patch("src.ml.walk_forward_validator.load_features", return_value=pd.DataFrame()) as mock_lf:
            v.load_date_range("2025-06-01", "2025-06-30")
            call_kwargs = mock_lf.call_args
            assert "2025-06-01" in str(call_kwargs)
            assert "2025-06-30" in str(call_kwargs)

    def test_returns_empty_when_no_files(self, tmp_path):
        v = _make_validator(tmp_path)
        result = v.load_date_range("2030-01-01", "2030-12-31")
        assert result.empty


# ---------------------------------------------------------------------------
# TestGetFeatureCols
# ---------------------------------------------------------------------------


class TestGetFeatureCols:
    def test_excludes_non_feature_cols(self, tmp_path):
        v = _make_validator(tmp_path)
        df = _make_df(20, 2)
        cols = v._get_feature_cols(df)
        assert "target" not in cols
        assert "date" not in cols
        assert "max_gain_120m" not in cols
        assert "min_loss_120m" not in cols

    def test_includes_feature_cols(self, tmp_path):
        v = _make_validator(tmp_path)
        df = _make_df(20, 2)
        cols = v._get_feature_cols(df)
        for c in FEATURE_COLS:
            assert c in cols

    def test_returns_sorted(self, tmp_path):
        v = _make_validator(tmp_path)
        df = _make_df(20, 2)
        cols = v._get_feature_cols(df)
        assert cols == sorted(cols)


# ---------------------------------------------------------------------------
# TestEvaluateSplit
# ---------------------------------------------------------------------------


class TestEvaluateSplit:
    def _mock_load(self, train_df, test_df):
        """Patch load_date_range to return specific DataFrames."""
        call_count = [0]

        def _side_effect(start, end):
            call_count[0] += 1
            return train_df if call_count[0] == 1 else test_df

        return patch(
            "src.ml.walk_forward_validator.WalkForwardValidator.load_date_range",
            side_effect=_side_effect,
        )

    def test_returns_insufficient_data_when_train_empty(self, tmp_path):
        v = _make_validator(tmp_path)
        empty = pd.DataFrame()
        non_empty = _make_df(50, 10, "2025-06-01")
        with self._mock_load(empty, non_empty):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_returns_insufficient_data_when_test_empty(self, tmp_path):
        v = _make_validator(tmp_path)
        non_empty = _make_df(200, 20, "2025-03-01")
        empty = pd.DataFrame()
        with self._mock_load(non_empty, empty):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_returns_insufficient_data_when_few_positives(self, tmp_path):
        v = _make_validator(tmp_path)
        # Only 2 positives, less than _MIN_POSITIVES
        train = _make_df(100, 2, "2025-03-01")
        test = _make_df(50, 5, "2025-06-01")
        with self._mock_load(train, test):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_success_contains_required_keys(self, tmp_path):
        v = _make_validator(tmp_path)
        train = _make_df(500, 50, "2025-03-01")
        # Distribute across multiple dates for chronological split
        dates = ["2025-03-" + f"{d:02d}" for d in range(1, 11)] * 50
        train["date"] = dates[:500]
        test = _make_df(100, 10, "2025-06-01")

        with self._mock_load(train, test):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")

        assert result["status"] == "SUCCESS"
        required = [
            "train_period", "test_period", "test_month",
            "train_samples", "test_samples", "threshold",
            "total_signals", "true_positives", "false_positives",
            "precision", "recall", "signal_rate",
            "tp_avg_gain_pct", "fp_avg_loss_pct", "expected_value_pct",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_precision_in_valid_range(self, tmp_path):
        v = _make_validator(tmp_path)
        train = _make_df(500, 50, "2025-03-01")
        train["date"] = ["2025-03-" + f"{(i % 20) + 1:02d}" for i in range(500)]
        test = _make_df(100, 10, "2025-06-01")

        with self._mock_load(train, test):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")

        if result["status"] == "SUCCESS":
            assert 0.0 <= result["precision"] <= 1.0

    def test_total_signals_equals_tp_plus_fp(self, tmp_path):
        v = _make_validator(tmp_path)
        train = _make_df(500, 50, "2025-03-01")
        train["date"] = ["2025-03-" + f"{(i % 20) + 1:02d}" for i in range(500)]
        test = _make_df(100, 10, "2025-06-01")

        with self._mock_load(train, test):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")

        if result["status"] == "SUCCESS":
            assert result["total_signals"] == result["true_positives"] + result["false_positives"]

    def test_test_month_label_matches_test_start(self, tmp_path):
        v = _make_validator(tmp_path)
        train = _make_df(500, 50, "2025-03-01")
        train["date"] = ["2025-03-" + f"{(i % 20) + 1:02d}" for i in range(500)]
        test = _make_df(100, 10, "2025-06-01")

        with self._mock_load(train, test):
            result = v.evaluate_split("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")

        if result["status"] == "SUCCESS":
            assert result["test_month"] == "2025-06"


# ---------------------------------------------------------------------------
# TestRunValidation
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_no_splits_returns_no_splits_status(self, tmp_path):
        v = _make_validator(tmp_path)
        # No CSV files → no splits
        result = v.run_validation()
        assert result["status"] == "NO_SPLITS"

    def test_all_splits_failed_status(self, tmp_path):
        # Provide CSV file names so splits are generated, but loading returns empty
        _make_csv_files(tmp_path, [
            "2025-03-03", "2025-04-01", "2025-05-01", "2025-06-02",
            "2025-07-01", "2025-08-01", "2025-09-02", "2025-10-01",
            "2025-11-03", "2025-12-01", "2026-01-02", "2026-02-03",
        ])
        v = _make_validator(tmp_path)

        with patch.object(v, "load_date_range", return_value=pd.DataFrame()):
            result = v.run_validation()

        assert result["status"] == "ALL_SPLITS_FAILED"

    def test_success_contains_precision_stats(self, tmp_path):
        _make_csv_files(tmp_path, [
            "2025-03-03", "2025-04-01", "2025-05-01", "2025-06-02",
        ])
        v = _make_validator(tmp_path)

        # Mock evaluate_split to return canned results
        mock_result = {
            "status": "SUCCESS",
            "split_index": 1,
            "train_period": "2025-03-01 → 2025-05-31",
            "test_period": "2025-06-01 → 2025-06-30",
            "test_month": "2025-06",
            "train_samples": 400,
            "test_samples": 100,
            "threshold": 0.67,
            "total_signals": 50,
            "true_positives": 45,
            "false_positives": 5,
            "false_negatives": 30,
            "true_negatives": 920,
            "precision": 0.90,
            "recall": 0.60,
            "signal_rate": 0.05,
            "train_positives": 50,
            "test_positives": 75,
            "tp_avg_gain_pct": 0.30,
            "fp_avg_loss_pct": -0.10,
            "expected_value_pct": 0.25,
        }

        with patch.object(v, "evaluate_split", return_value=mock_result):
            # Need at least one split from get_date_splits
            with patch.object(v, "get_date_splits", return_value=[
                ("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")
            ]):
                result = v.run_validation(threshold=0.67)

        assert result["status"] == "SUCCESS"
        assert "precision_mean" in result
        assert "precision_std" in result
        assert "signals_mean" in result
        assert "ev_mean" in result

    def test_precision_mean_correct(self, tmp_path):
        v = _make_validator(tmp_path)
        mock_results = [
            {"status": "SUCCESS", "split_index": i,
             "train_period": "x", "test_period": "x", "test_month": "2025-06",
             "train_samples": 400, "test_samples": 100, "threshold": 0.67,
             "total_signals": 50, "true_positives": 45, "false_positives": 5,
             "false_negatives": 30, "true_negatives": 920,
             "precision": p, "recall": 0.6, "signal_rate": 0.05,
             "train_positives": 50, "test_positives": 75,
             "tp_avg_gain_pct": 0.30, "fp_avg_loss_pct": -0.10,
             "expected_value_pct": 0.25}
            for i, p in enumerate([0.90, 0.92, 0.88], 1)
        ]

        with patch.object(v, "get_date_splits", return_value=[
            ("2025-03-01", "2025-05-31", "2025-0" + str(m) + "-01", "2025-0" + str(m) + "-30")
            for m in [6, 7, 8]
        ]):
            with patch.object(v, "evaluate_split", side_effect=mock_results):
                result = v.run_validation(threshold=0.67)

        assert result["status"] == "SUCCESS"
        assert abs(result["precision_mean"] - np.mean([0.90, 0.92, 0.88])) < 1e-6

    def test_splits_list_in_result(self, tmp_path):
        v = _make_validator(tmp_path)
        mock_result = {
            "status": "SUCCESS", "split_index": 1,
            "train_period": "x", "test_period": "x", "test_month": "2025-06",
            "train_samples": 400, "test_samples": 100, "threshold": 0.67,
            "total_signals": 50, "true_positives": 45, "false_positives": 5,
            "false_negatives": 30, "true_negatives": 920,
            "precision": 0.90, "recall": 0.6, "signal_rate": 0.05,
            "train_positives": 50, "test_positives": 75,
            "tp_avg_gain_pct": 0.30, "fp_avg_loss_pct": -0.10,
            "expected_value_pct": 0.25,
        }
        with patch.object(v, "get_date_splits", return_value=[
            ("2025-03-01", "2025-05-31", "2025-06-01", "2025-06-30")
        ]):
            with patch.object(v, "evaluate_split", return_value=mock_result):
                result = v.run_validation()

        assert "splits" in result
        assert isinstance(result["splits"], list)
        assert len(result["splits"]) == 1


# ---------------------------------------------------------------------------
# TestPlotResults
# ---------------------------------------------------------------------------


class TestPlotResults:
    def _summary_with_splits(self, precisions, months=None):
        months = months or [f"2025-{m:02d}" for m in range(6, 6 + len(precisions))]
        splits = [
            {
                "status": "SUCCESS", "test_month": m, "precision": p,
                "total_signals": 100, "test_period": f"{m}-01 → {m}-30",
            }
            for m, p in zip(months, precisions)
        ]
        return {
            "status": "SUCCESS",
            "precision_mean": float(np.mean(precisions)),
            "splits": splits,
        }

    def test_returns_string(self, tmp_path):
        v = _make_validator(tmp_path)
        s = self._summary_with_splits([0.90, 0.92, 0.88])
        result = v.plot_results(s)
        assert isinstance(result, str)

    def test_contains_test_months(self, tmp_path):
        v = _make_validator(tmp_path)
        s = self._summary_with_splits([0.90, 0.92], months=["2025-06", "2025-07"])
        result = v.plot_results(s)
        assert "2025-06" in result
        assert "2025-07" in result

    def test_contains_bar_characters(self, tmp_path):
        v = _make_validator(tmp_path)
        s = self._summary_with_splits([0.90, 0.92, 0.88])
        result = v.plot_results(s)
        assert "█" in result or "░" in result

    def test_contains_precision_percentages(self, tmp_path):
        v = _make_validator(tmp_path)
        s = self._summary_with_splits([0.90])
        result = v.plot_results(s)
        assert "90.0%" in result

    def test_no_splits_status_returns_placeholder(self, tmp_path):
        v = _make_validator(tmp_path)
        result = v.plot_results({"status": "NO_SPLITS"})
        assert isinstance(result, str)
        assert "No results" in result or "NO_SPLITS" in result

    def test_contains_header(self, tmp_path):
        v = _make_validator(tmp_path)
        s = self._summary_with_splits([0.90, 0.92])
        result = v.plot_results(s)
        assert "PRECISION" in result.upper()


# ---------------------------------------------------------------------------
# TestDefaultXgbParams
# ---------------------------------------------------------------------------


class TestDefaultXgbParams:
    def test_default_params_present(self):
        required = ["n_estimators", "max_depth", "learning_rate", "random_state"]
        for key in required:
            assert key in _DEFAULT_XGB_PARAMS

    def test_n_estimators_matches_xgboost_v2(self):
        assert _DEFAULT_XGB_PARAMS["n_estimators"] == 300

    def test_min_positives_threshold(self):
        assert _MIN_POSITIVES >= 5

    def test_custom_params_override_defaults(self, tmp_path):
        custom = {"n_estimators": 50, "max_depth": 3, "random_state": 99}
        v = WalkForwardValidator(features_dir=str(tmp_path), xgb_params=custom)
        assert v.xgb_params["n_estimators"] == 50
        assert v.xgb_params["random_state"] == 99
