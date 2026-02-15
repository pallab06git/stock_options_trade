# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for TrainingDataPrep — offline ML training data generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processing.training_data_prep import TrainingDataPrep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = (1707480000000 // 60000) * 60000  # minute-aligned
DATE = "2024-02-09"


def _config(tmp_path, window=5, coverage=50.0):
    """Build a minimal config dict for TrainingDataPrep."""
    return {
        "signal_validation": {"prediction_window_minutes": window},
        "sinks": {
            "parquet": {
                "consolidated_path": str(tmp_path / "consolidated"),
                "training_path": str(tmp_path / "training"),
                "compression": "snappy",
            }
        },
        "training": {"min_target_coverage_pct": coverage},
    }


def _consolidated_df(n_minutes=10, tickers=None):
    """Create a synthetic consolidated DataFrame (per-option-per-minute).

    If tickers is provided, creates rows for each ticker at each minute.
    Otherwise creates SPY-only rows (ticker=None).
    """
    rows = []
    for m in range(n_minutes):
        ts = BASE_TS + m * 60000
        if tickers:
            for ticker in tickers:
                rows.append({
                    "timestamp": ts,
                    "ticker": ticker,
                    "option_avg_price": 5.0 + m * 0.1 + (0.5 if "P" in ticker else 0.0),
                    "spy_close": 450.0 + m * 0.05,
                    "source": "consolidated",
                })
        else:
            rows.append({
                "timestamp": ts,
                "ticker": None,
                "option_avg_price": np.nan,
                "spy_close": 450.0 + m * 0.05,
                "source": "consolidated",
            })
    return pd.DataFrame(rows)


def _write_consolidated(df, tmp_path, date=DATE):
    """Write a consolidated Parquet file."""
    out_dir = tmp_path / "consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:

    def test_init_loads_config(self, tmp_path):
        cfg = _config(tmp_path, window=120, coverage=60.0)
        prep = TrainingDataPrep(cfg)
        assert prep.prediction_window_minutes == 120
        assert prep.min_target_coverage_pct == 60.0
        assert prep.consolidated_path == tmp_path / "consolidated"
        assert prep.training_path == tmp_path / "training"

    def test_init_defaults(self, tmp_path):
        prep = TrainingDataPrep({})
        assert prep.prediction_window_minutes == 120
        assert prep.min_target_coverage_pct == 50.0
        assert prep.compression == "snappy"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:

    def test_load_missing_file(self, tmp_path):
        prep = TrainingDataPrep(_config(tmp_path))
        result = prep._load_consolidated("2024-01-01")
        assert result is None

    def test_load_existing_file(self, tmp_path):
        df = _consolidated_df(5, tickers=["OPT_A"])
        _write_consolidated(df, tmp_path)
        prep = TrainingDataPrep(_config(tmp_path))
        result = prep._load_consolidated(DATE)
        assert len(result) == 5
        assert "timestamp" in result.columns


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------


class TestTargetFuturePrices:

    def test_full_window_lookup(self, tmp_path):
        """With enough future data, all lookahead slots should be filled."""
        # 10 minutes of data, window=5 → first 5 rows can look ahead fully
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=5))
        result = prep._compute_target_future_prices(df)

        assert "target_future_prices" in result.columns
        # Row 0 has minutes 1..5 ahead available → all 5 should be non-NaN
        target_0 = result["target_future_prices"].iloc[0]
        assert len(target_0) == 5
        non_nan = sum(1 for v in target_0 if not np.isnan(v))
        assert non_nan == 5

    def test_truncated_end_of_day(self, tmp_path):
        """Last row has no future data → all NaN in target."""
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=5))
        result = prep._compute_target_future_prices(df)

        # Last row (minute 9) can't look ahead at all → 5 NaNs
        target_last = result["target_future_prices"].iloc[-1]
        assert len(target_last) == 5
        assert all(np.isnan(v) for v in target_last)

    def test_partial_window(self, tmp_path):
        """Row near end of data gets partial NaN in target."""
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=5))
        result = prep._compute_target_future_prices(df)

        # Row 7 (minute 7): can look ahead to minutes 8, 9 (2 valid), then 3 NaN
        target_7 = result["target_future_prices"].iloc[7]
        non_nan = sum(1 for v in target_7 if not np.isnan(v))
        assert non_nan == 2

    def test_spy_only_rows_get_none(self, tmp_path):
        """Rows without a ticker should get None target."""
        df = _consolidated_df(5, tickers=None)  # SPY-only
        prep = TrainingDataPrep(_config(tmp_path, window=3))
        result = prep._compute_target_future_prices(df)

        for val in result["target_future_prices"]:
            assert val is None

    def test_per_ticker_isolation(self, tmp_path):
        """Each ticker's lookahead uses only its own prices."""
        # Build custom data with distinct prices per ticker
        rows = []
        for m in range(10):
            ts = BASE_TS + m * 60000
            rows.append({"timestamp": ts, "ticker": "CALL_A", "option_avg_price": 5.0 + m * 0.1, "spy_close": 450.0})
            rows.append({"timestamp": ts, "ticker": "PUT_B", "option_avg_price": 8.0 + m * 0.2, "spy_close": 450.0})
        df = pd.DataFrame(rows)

        prep = TrainingDataPrep(_config(tmp_path, window=3))
        result = prep._compute_target_future_prices(df)

        # Row 0 is CALL_A at minute 0 → should look up CALL_A at minutes 1,2,3
        target_a = result["target_future_prices"].iloc[0]
        # Row 1 is PUT_B at minute 0 → should look up PUT_B at minutes 1,2,3
        target_b = result["target_future_prices"].iloc[1]

        # CALL_A prices: 5.0 + m*0.1 → T+1: 5.1, T+2: 5.2, T+3: 5.3
        assert abs(target_a[0] - 5.1) < 0.01
        assert abs(target_a[1] - 5.2) < 0.01
        assert abs(target_a[2] - 5.3) < 0.01

        # PUT_B prices: 8.0 + m*0.2 → T+1: 8.2, T+2: 8.4, T+3: 8.6
        assert abs(target_b[0] - 8.2) < 0.01
        assert abs(target_b[1] - 8.4) < 0.01
        assert abs(target_b[2] - 8.6) < 0.01

    def test_correct_price_values(self, tmp_path):
        """Verify target contains actual future option_avg_price values."""
        df = _consolidated_df(5, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=2))
        result = prep._compute_target_future_prices(df)

        # Row 0 target should be [price at minute 1, price at minute 2]
        expected = [
            df[df["timestamp"] == BASE_TS + 60000]["option_avg_price"].iloc[0],
            df[df["timestamp"] == BASE_TS + 120000]["option_avg_price"].iloc[0],
        ]
        actual = result["target_future_prices"].iloc[0]
        assert abs(actual[0] - expected[0]) < 0.001
        assert abs(actual[1] - expected[1]) < 0.001


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:

    def test_spy_only_rows_removed(self, tmp_path):
        """SPY-only rows (no ticker) are always filtered out."""
        df = _consolidated_df(5, tickers=None)
        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=0.0))
        df = prep._compute_target_future_prices(df)
        result = prep._filter_by_target_coverage(df)
        assert len(result) == 0

    def test_high_coverage_kept(self, tmp_path):
        """Rows with 100% coverage pass any threshold."""
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=50.0))
        df = prep._compute_target_future_prices(df)
        result = prep._filter_by_target_coverage(df)
        # First 7 rows have full 3-minute lookahead → 100% coverage
        # Row 7: 2/3, Row 8: 1/3, Row 9: 0/3
        # At 50% threshold: rows 0-7 pass (coverage >= 50%)
        assert len(result) == 8

    def test_strict_coverage_filters_more(self, tmp_path):
        """Higher coverage threshold filters more rows."""
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=100.0))
        df = prep._compute_target_future_prices(df)
        result = prep._filter_by_target_coverage(df)
        # Only rows 0-6 have 100% coverage (3/3 non-NaN)
        assert len(result) == 7

    def test_zero_coverage_keeps_all_with_ticker(self, tmp_path):
        """0% coverage keeps all rows that have a ticker (even all-NaN)."""
        df = _consolidated_df(10, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=0.0))
        df = prep._compute_target_future_prices(df)
        result = prep._filter_by_target_coverage(df)
        # All 10 rows have a ticker, even row 9 with 0/3 → coverage=0 >= 0%
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Full pipeline (prepare)
# ---------------------------------------------------------------------------


class TestPrepare:

    def test_prepare_single_date(self, tmp_path):
        """Full pipeline for one date: load → target → filter → write."""
        df = _consolidated_df(10, tickers=["OPT_A", "OPT_B"])
        _write_consolidated(df, tmp_path)

        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=50.0))
        stats = prep.prepare([DATE])

        assert stats["status"] == "success"
        assert stats["dates_processed"] == 1
        assert stats["dates_skipped"] == 0
        assert stats["total_rows_in"] == 20  # 10 minutes × 2 tickers
        assert stats["total_rows_out"] > 0
        assert stats["total_rows_out"] <= 20
        assert stats["unique_options"] == 2
        assert stats["prediction_window_minutes"] == 3
        assert stats["min_target_coverage_pct"] == 50.0

        # Output file should exist
        out = tmp_path / "training" / f"{DATE}.parquet"
        assert out.exists()

        result = pd.read_parquet(out)
        assert "target_future_prices" in result.columns
        assert len(result) == stats["total_rows_out"]

    def test_prepare_missing_date_skipped(self, tmp_path):
        """Missing consolidated file → date skipped."""
        prep = TrainingDataPrep(_config(tmp_path, window=3))
        stats = prep.prepare(["2024-01-01"])
        assert stats["dates_processed"] == 0
        assert stats["dates_skipped"] == 1
        assert stats["total_rows_in"] == 0
        assert stats["total_rows_out"] == 0

    def test_prepare_multiple_dates(self, tmp_path):
        """Multiple dates each processed independently."""
        for date in ["2024-02-09", "2024-02-10"]:
            df = _consolidated_df(10, tickers=["OPT_A"])
            _write_consolidated(df, tmp_path, date=date)

        prep = TrainingDataPrep(_config(tmp_path, window=3, coverage=50.0))
        stats = prep.prepare(["2024-02-09", "2024-02-10"])

        assert stats["dates_processed"] == 2
        assert stats["dates_skipped"] == 0
        assert stats["total_rows_in"] == 20  # 10 × 2 dates

        for date in ["2024-02-09", "2024-02-10"]:
            assert (tmp_path / "training" / f"{date}.parquet").exists()

    def test_prepare_all_filtered_out(self, tmp_path):
        """If all rows are filtered, date still counts as processed."""
        # Only 2 minutes of data, window=5, coverage=100% → nothing passes
        df = _consolidated_df(2, tickers=["OPT_A"])
        _write_consolidated(df, tmp_path)

        prep = TrainingDataPrep(_config(tmp_path, window=5, coverage=100.0))
        stats = prep.prepare([DATE])

        assert stats["dates_processed"] == 1
        assert stats["total_rows_in"] == 2
        assert stats["total_rows_out"] == 0
        # No output file written when 0 rows survive
        assert not (tmp_path / "training" / f"{DATE}.parquet").exists()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TestOutput:

    def test_output_parquet_written(self, tmp_path):
        """Verify output Parquet is written to training_path."""
        df = _consolidated_df(5, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=2, coverage=0.0))
        df = prep._compute_target_future_prices(df)
        path = prep._write_output(df, DATE)
        assert path.exists()
        result = pd.read_parquet(path)
        assert len(result) == 5

    def test_output_preserves_columns(self, tmp_path):
        """Output includes both original columns and target_future_prices."""
        df = _consolidated_df(5, tickers=["OPT_A"])
        prep = TrainingDataPrep(_config(tmp_path, window=2, coverage=0.0))
        df = prep._compute_target_future_prices(df)
        path = prep._write_output(df, DATE)
        result = pd.read_parquet(path)
        for col in ["timestamp", "ticker", "option_avg_price", "spy_close", "target_future_prices"]:
            assert col in result.columns
