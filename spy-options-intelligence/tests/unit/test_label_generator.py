# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for label_generator module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from src.processing.label_generator import (
    LabelGenerator,
    _generate_single_group,
    _validate_input,
    generate_labels,
)

ET_TZ = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _et_ts(date_str: str, hour: int, minute: int) -> int:
    """Convert ET date + time to Unix milliseconds."""
    from datetime import datetime

    base = ET_TZ.localize(
        datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=hour, minute=minute, second=0
        )
    )
    return int(base.timestamp() * 1000)


def _make_opt_df(
    date: str,
    n_bars: int = 10,
    base_price: float = 4.0,
    spike_at: int | None = None,
    spike_multiplier: float = 1.30,
    ticker: str = "O:SPY250304C00594000",
) -> pd.DataFrame:
    """Synthetic single-contract options DataFrame.

    If spike_at is given, bar at that index is set to
    base_price * spike_multiplier (simulates a >20% intraday spike).
    """
    base_ts = _et_ts(date, 9, 30)
    rows = []
    for i in range(n_bars):
        ts = base_ts + i * 60_000
        price = base_price * spike_multiplier if (spike_at is not None and i == spike_at) else base_price
        rows.append(
            {
                "timestamp": ts,
                "open": price * 0.98,
                "high": price * 1.02,
                "low": price * 0.97,
                "close": price,
                "volume": 100 + i,
                "ticker": ticker,
            }
        )
    return pd.DataFrame(rows)


def _make_config() -> dict:
    return {
        "feature_engineering": {
            "target_threshold_pct": 20.0,
            "target_lookforward_minutes": 5,
        }
    }


# ---------------------------------------------------------------------------
# TestValidateInput
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_passes_when_columns_present(self):
        df = pd.DataFrame({"timestamp": [1], "close": [4.0]})
        _validate_input(df)  # should not raise

    def test_raises_when_timestamp_missing(self):
        df = pd.DataFrame({"close": [4.0]})
        with pytest.raises(ValueError, match="timestamp"):
            _validate_input(df)

    def test_raises_when_close_missing(self):
        df = pd.DataFrame({"timestamp": [1]})
        with pytest.raises(ValueError, match="close"):
            _validate_input(df)


# ---------------------------------------------------------------------------
# TestGenerateLabelsSingleTicker
# ---------------------------------------------------------------------------


class TestGenerateLabelsSingleTicker:
    DATE = "2025-03-03"

    def test_columns_added(self):
        df = _make_opt_df(self.DATE, n_bars=10)
        result = generate_labels(df)
        assert "target" in result.columns
        assert "max_gain_pct" in result.columns
        assert "time_to_max_min" in result.columns

    def test_target_is_binary(self):
        df = _make_opt_df(self.DATE, n_bars=10)
        result = generate_labels(df)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_target_1_when_spike_within_window(self):
        # spike_at=3 means bar 3 has 30% gain; bars 0-2 look forward and see it
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.30)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)
        # bar 0 should see the spike at bar 3 (within 5 minutes)
        assert result["target"].iloc[0] == 1

    def test_target_0_when_no_spike(self):
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.05)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)
        assert result["target"].sum() == 0

    def test_target_0_when_spike_outside_window(self):
        # Spike at bar 8; forward window is only 3 minutes; bar 0 can't see bar 8
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=8, spike_multiplier=1.30)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=3)
        assert result["target"].iloc[0] == 0

    def test_last_bar_has_no_future(self):
        df = _make_opt_df(self.DATE, n_bars=10)
        result = generate_labels(df)
        assert result["target"].iloc[-1] == 0
        assert pd.isna(result["max_gain_pct"].iloc[-1])
        assert pd.isna(result["time_to_max_min"].iloc[-1])

    def test_max_gain_pct_positive_for_spiked_bar(self):
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.30)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)
        assert result["max_gain_pct"].iloc[0] > 0

    def test_time_to_max_equals_minutes_to_spike_bar(self):
        # bar 0 at 09:30, spike at bar 3 = 09:33 → time_to_max = 3 min
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.30)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)
        assert abs(result["time_to_max_min"].iloc[0] - 3.0) < 0.01

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "close"])
        result = generate_labels(df)
        assert result.empty
        assert "target" in result.columns

    def test_raises_on_missing_timestamp(self):
        df = pd.DataFrame({"close": [4.0]})
        with pytest.raises(ValueError):
            generate_labels(df)

    def test_raises_on_missing_close(self):
        df = pd.DataFrame({"timestamp": [1000000]})
        with pytest.raises(ValueError):
            generate_labels(df)

    def test_zero_price_bars_get_target_zero(self):
        df = _make_opt_df(self.DATE, n_bars=5)
        df.at[0, "close"] = 0.0
        result = generate_labels(df)
        assert result["target"].iloc[0] == 0

    def test_nan_price_bars_get_target_zero(self):
        df = _make_opt_df(self.DATE, n_bars=5)
        df.at[0, "close"] = np.nan
        result = generate_labels(df)
        assert result["target"].iloc[0] == 0

    def test_original_df_not_mutated(self):
        df = _make_opt_df(self.DATE, n_bars=10)
        original_cols = set(df.columns)
        _ = generate_labels(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# TestGenerateLabelsMultiTicker
# ---------------------------------------------------------------------------


class TestGenerateLabelsMultiTicker:
    DATE = "2025-03-03"

    def _make_two_tickers(self) -> pd.DataFrame:
        df_a = _make_opt_df(
            self.DATE, n_bars=8, spike_at=5, spike_multiplier=1.30,
            ticker="O:SPY250304C00594000"
        )
        df_b = _make_opt_df(
            self.DATE, n_bars=8, spike_at=None,  # no spike in B
            ticker="O:SPY250304P00593000"
        )
        return pd.concat([df_a, df_b], ignore_index=True)

    def test_labels_computed_per_ticker(self):
        df = self._make_two_tickers()
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)

        call_rows = result[result["ticker"] == "O:SPY250304C00594000"]
        put_rows = result[result["ticker"] == "O:SPY250304P00593000"]

        # Call should have at least one positive label (spike at bar 5)
        assert call_rows["target"].sum() > 0
        # Put has no spike — all targets should be 0
        assert put_rows["target"].sum() == 0

    def test_output_has_same_row_count(self):
        df = self._make_two_tickers()
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=5)
        assert len(result) == len(df)

    def test_cross_ticker_isolation(self):
        """A spike in ticker B must NOT cause target=1 in ticker A."""
        # Put has a huge spike at bar 7; Call has NO spike
        df_a = _make_opt_df(self.DATE, n_bars=8, spike_at=None,
                            ticker="O:SPY250304C00594000")
        df_b = _make_opt_df(self.DATE, n_bars=8, spike_at=7,
                            spike_multiplier=3.0, ticker="O:SPY250304P00593000")
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = generate_labels(df, threshold_pct=20.0, lookforward_minutes=10)

        call_rows = result[result["ticker"] == "O:SPY250304C00594000"]
        assert call_rows["target"].sum() == 0


# ---------------------------------------------------------------------------
# TestLabelGeneratorClass
# ---------------------------------------------------------------------------


class TestLabelGeneratorClass:
    DATE = "2025-03-03"

    def test_generate_uses_configured_threshold(self):
        cfg = {"feature_engineering": {"target_threshold_pct": 50.0,
                                        "target_lookforward_minutes": 5}}
        gen = LabelGenerator(cfg)
        # spike of 30% should NOT trigger if threshold is 50%
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.30)
        result = gen.generate(df)
        assert result["target"].sum() == 0

    def test_generate_uses_configured_window(self):
        cfg = {"feature_engineering": {"target_threshold_pct": 20.0,
                                        "target_lookforward_minutes": 2}}
        gen = LabelGenerator(cfg)
        # spike at bar 5 is outside a 2-minute window for bar 0
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=5, spike_multiplier=1.30)
        result = gen.generate(df)
        assert result["target"].iloc[0] == 0

    def test_defaults_applied_when_config_empty(self):
        gen = LabelGenerator({})
        assert gen.threshold_pct == 20.0
        assert gen.lookforward_minutes == 120

    def test_validate_returns_stats(self):
        gen = LabelGenerator(_make_config())
        df = _make_opt_df(self.DATE, n_bars=10, spike_at=3, spike_multiplier=1.30)
        labeled = gen.generate(df)
        stats = gen.validate(labeled)

        assert "n_total" in stats
        assert "n_positive" in stats
        assert "positive_rate" in stats
        assert "coverage_pct" in stats
        assert stats["n_total"] == 10

    def test_validate_detects_missing_columns(self):
        gen = LabelGenerator(_make_config())
        df = _make_opt_df(self.DATE, n_bars=10)
        stats = gen.validate(df)
        assert len(stats["missing_columns"]) > 0

    def test_generate_for_csv_file(self):
        gen = LabelGenerator(_make_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2025-03-03_features.csv"
            df = _make_opt_df(self.DATE, n_bars=10, spike_at=3,
                               spike_multiplier=1.30)
            df.to_csv(path, index=False)

            result = gen.generate_for_file(path)
            assert "target" in result.columns
            assert len(result) == 10

    def test_generate_for_parquet_file(self):
        gen = LabelGenerator(_make_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2025-03-03_features.parquet"
            df = _make_opt_df(self.DATE, n_bars=10)
            df.to_parquet(path, index=False)

            result = gen.generate_for_file(path)
            assert "target" in result.columns

    def test_generate_for_file_overwrites_existing_target(self):
        """If a 'target' column already exists it should be replaced."""
        gen = LabelGenerator(_make_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            df = _make_opt_df(self.DATE, n_bars=10)
            df["target"] = 99  # dummy stale value
            df.to_csv(path, index=False)

            result = gen.generate_for_file(path)
            assert set(result["target"].unique()).issubset({0, 1})

    def test_generate_for_file_raises_on_missing_file(self):
        gen = LabelGenerator(_make_config())
        with pytest.raises(FileNotFoundError):
            gen.generate_for_file("/nonexistent/path/file.csv")
