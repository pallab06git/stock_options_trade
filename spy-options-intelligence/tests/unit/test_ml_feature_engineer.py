# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for MLFeatureEngineer."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytz

from src.processing.ml_feature_engineer import MLFeatureEngineer

ET_TZ = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _et_ts(date_str: str, hour: int, minute: int) -> int:
    """Convert an ET date + time to a Unix millisecond timestamp."""
    from datetime import datetime

    dt_et = ET_TZ.localize(
        datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=hour, minute=minute, second=0
        )
    )
    return int(dt_et.timestamp() * 1000)


def _make_config(tmp_path: Path) -> dict:
    return {
        "feature_engineering": {
            "lookback_windows_minutes": [1, 5],
            "volatility_windows_minutes": [5],
            "rsi_period": 14,
            "ema_periods": [9, 21],
            "target_threshold_pct": 20.0,
            "target_lookforward_minutes": 5,  # short window for tests
        },
        "pipeline_v2": {
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
            "feature_engineering": {"market_open_et": "09:30"},
        },
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "raw"),
                "compression": "snappy",
            }
        },
    }


def _make_spy_df(date_str: str, n_bars: int = 30) -> pd.DataFrame:
    """Build a synthetic SPY DataFrame with n_bars starting at 09:30 ET."""
    from datetime import datetime, timedelta

    base_ts = _et_ts(date_str, 9, 30)
    rows = []
    for i in range(n_bars):
        ts = base_ts + i * 60_000  # add i minutes in ms
        rows.append(
            {
                "timestamp": ts,
                "open": 590.0 + i * 0.1,
                "high": 591.0 + i * 0.1,
                "low": 589.5 + i * 0.1,
                "close": 590.5 + i * 0.1,
                "volume": 1000.0 + i * 10,
                "vwap": 590.3 + i * 0.1,
                "transactions": 50 + i,
                "source": "spy",
            }
        )
    return pd.DataFrame(rows)


def _make_option_df(
    date_str: str,
    ticker: str,
    n_bars: int = 10,
    start_hour: int = 9,
    start_minute: int = 30,
    base_price: float = 4.0,
    spike_at: int = 3,        # bar index that jumps to trigger a 20%+ spike
    spike_multiplier: float = 1.30,
) -> pd.DataFrame:
    """Build a synthetic options DataFrame.

    The bar at index `spike_at` is set to base_price * spike_multiplier so
    that earlier bars (looking forward) should detect a ≥20% spike.
    """
    rows = []
    for i in range(n_bars):
        ts = _et_ts(date_str, start_hour, start_minute + i)
        price = base_price * spike_multiplier if i == spike_at else base_price
        rows.append(
            {
                "timestamp": ts,
                "open": price * 0.98,
                "high": price * 1.02,
                "low": price * 0.97,
                "close": price,
                "volume": 100 + i * 5,
                "vwap": price * 1.01,
                "transactions": 10 + i,
                "ticker": ticker,
                "source": "options_minute_massive",
            }
        )
    return pd.DataFrame(rows)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestMLFeatureEngineerInit:
    def test_defaults_applied_when_config_empty(self, tmp_path):
        eng = MLFeatureEngineer({"sinks": {"parquet": {"base_path": str(tmp_path)}}})
        assert eng.lookback_windows == [1, 5, 15, 30, 60]
        assert eng.target_threshold_pct == 20.0
        assert eng.target_lookforward_minutes == 120

    def test_config_overrides_respected(self, tmp_path):
        config = _make_config(tmp_path)
        eng = MLFeatureEngineer(config)
        assert eng.lookback_windows == [1, 5]
        assert eng.target_lookforward_minutes == 5
        assert eng.rsi_period == 14

    def test_output_path_is_set(self, tmp_path):
        eng = MLFeatureEngineer(_make_config(tmp_path))
        assert eng._output_path == Path("data/processed/features")


# ---------------------------------------------------------------------------
# TestComputeSpyFeatures
# ---------------------------------------------------------------------------


class TestComputeSpyFeatures:
    def setup_method(self):
        from pathlib import Path
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.config = _make_config(self.tmp)
        self.eng = MLFeatureEngineer(self.config)
        self.date = "2025-03-03"
        self.spy_df = _make_spy_df(self.date, n_bars=30)

    def test_returns_dataframe(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30

    def test_timestamp_column_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        assert "timestamp" in result.columns

    def test_time_features_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        for col in ["hour_et", "minute_et", "minute_of_day", "minutes_since_open",
                    "is_morning", "pct_day_elapsed", "is_last_hour", "spy_bar_count"]:
            assert col in result.columns, f"Missing: {col}"

    def test_spy_return_columns_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        assert "spy_return_1m" in result.columns
        assert "spy_return_5m" in result.columns

    def test_volume_features_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        for col in ["spy_volume", "spy_vol_ma5", "spy_vol_ma30",
                    "spy_vol_ratio_5m", "spy_vol_ratio_30m", "spy_vol_zscore"]:
            assert col in result.columns, f"Missing: {col}"

    def test_volatility_features_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        assert "spy_vol_std_5m" in result.columns
        assert "spy_hl_range_5m" in result.columns

    def test_technical_columns_present(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        for col in ["spy_rsi_14", "spy_ema_9", "spy_ema_21", "spy_ema_diff",
                    "spy_macd", "spy_macd_signal", "spy_macd_hist",
                    "spy_bb_upper", "spy_bb_lower", "spy_bb_pct_b"]:
            assert col in result.columns, f"Missing: {col}"

    def test_no_raw_ohlcv_columns(self):
        """Raw columns must not appear — only spy_* prefixed versions."""
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        for raw in ["open", "high", "low", "close", "volume", "source"]:
            assert raw not in result.columns, f"Raw column leaked: {raw}"

    def test_is_morning_correct(self):
        """First bar (09:30) should be is_morning=1; bars 120+ min in should be 0."""
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        # All 30 bars are within the first 30 minutes → all is_morning=1
        assert result["is_morning"].iloc[0] == 1
        # Bar at minute 0 → minutes_since_open=0 → is_morning=1
        assert result["is_morning"].iloc[-1] == 1  # still 29 min in

    def test_spy_bar_count_sequential(self):
        result = self.eng._compute_spy_features(self.spy_df, self.date)
        assert result["spy_bar_count"].tolist() == list(range(1, 31))

    def test_no_nan_in_technical_cols_after_fill(self):
        """After ffill/bfill, technical indicator columns should have no NaN.

        MACD signal requires fast(12) + slow(26) + signal(9) - 2 = 35 bars.
        Use 40 bars so every indicator can produce at least one valid value,
        allowing bfill to propagate the first valid value to earlier rows.
        """
        spy_df = _make_spy_df(self.date, n_bars=40)
        eng = MLFeatureEngineer(_make_config(self.tmp))
        result = eng._compute_spy_features(spy_df, self.date)
        ta_cols = [c for c in result.columns
                   if c.startswith("spy_rsi") or c.startswith("spy_ema")
                   or c.startswith("spy_macd") or c.startswith("spy_bb")]
        for col in ta_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"


# ---------------------------------------------------------------------------
# TestComputeTargets
# ---------------------------------------------------------------------------


class TestComputeTargets:
    def setup_method(self):
        from pathlib import Path
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        config = _make_config(self.tmp)
        self.eng = MLFeatureEngineer(config)
        self.date = "2025-03-03"

    def test_target_column_added(self):
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        assert "target" in result.columns

    def test_target_binary(self):
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_target_1_when_spike_within_window(self):
        """Bar 0 should see the spike at bar 3 (within the 5-min forward window)."""
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        assert result["target"].iloc[0] == 1

    def test_target_0_when_no_spike(self):
        """No spike → all targets should be 0."""
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.05)  # only 5% gain
        result = self.eng._compute_targets(df)
        assert result["target"].sum() == 0

    def test_max_gain_column_added(self):
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        assert "max_gain_120m" in result.columns

    def test_time_to_max_min_column_added(self):
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=10,
                             spike_at=3, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        assert "time_to_max_min" in result.columns

    def test_last_bars_have_no_target(self):
        """Bars near end of window have no future bars → target stays 0."""
        df = _make_option_df(self.date, "O:SPY250304C00594000", n_bars=6,
                             spike_at=5, spike_multiplier=1.30)
        result = self.eng._compute_targets(df)
        # The last bar has no forward bars at all
        assert result["target"].iloc[-1] == 0


# ---------------------------------------------------------------------------
# TestParseContractMeta
# ---------------------------------------------------------------------------


class TestParseContractMeta:
    def setup_method(self):
        from pathlib import Path
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.eng = MLFeatureEngineer(_make_config(self.tmp))

    def test_call_ticker(self):
        strike, tte, flag = self.eng._parse_contract_meta(
            "O:SPY250307C00625000", "2025-03-03"
        )
        assert abs(strike - 625.0) < 0.001
        assert flag == "c"
        assert tte > 0

    def test_put_ticker(self):
        strike, tte, flag = self.eng._parse_contract_meta(
            "O:SPY250307P00593000", "2025-03-03"
        )
        assert flag == "p"
        assert abs(strike - 593.0) < 0.001

    def test_tte_calculation(self):
        # Expiry 4 days from trade date
        strike, tte, flag = self.eng._parse_contract_meta(
            "O:SPY250307C00620000", "2025-03-03"
        )
        assert abs(tte - 4.0) < 0.1

    def test_invalid_ticker_returns_fallback(self):
        strike, tte, flag = self.eng._parse_contract_meta("INVALID_TICKER", "2025-03-03")
        assert strike == 400.0
        assert tte == 1.0
        assert flag == "c"

    def test_underscore_prefix_parsed(self):
        """safe_ticker format O_SPY... should also parse correctly."""
        strike, tte, flag = self.eng._parse_contract_meta(
            "O_SPY250307C00625000", "2025-03-03"
        )
        assert abs(strike - 625.0) < 0.001


# ---------------------------------------------------------------------------
# TestEngineerDate (end-to-end with file I/O)
# ---------------------------------------------------------------------------


class TestEngineerDate:
    def setup_method(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.config = _make_config(self.tmp)
        self.eng = MLFeatureEngineer(self.config)
        self.date = "2025-03-03"

    def _write_test_data(self):
        raw = self.tmp / "raw"
        spy_df = _make_spy_df(self.date, n_bars=30)
        _write_parquet(spy_df, raw / "spy" / f"{self.date}.parquet")

        ticker = "O:SPY250304C00594000"
        safe = "O_SPY250304C00594000"
        opt_df = _make_option_df(
            self.date, ticker, n_bars=10, spike_at=3, spike_multiplier=1.30
        )
        _write_parquet(opt_df, raw / "options" / "minute" / safe / f"{self.date}.parquet")

    def test_empty_when_no_spy_data(self):
        result = self.eng.engineer_date(self.date)
        assert result.empty

    def test_empty_when_no_options_data(self):
        raw = self.tmp / "raw"
        spy_df = _make_spy_df(self.date)
        _write_parquet(spy_df, raw / "spy" / f"{self.date}.parquet")
        result = self.eng.engineer_date(self.date)
        assert result.empty

    def test_returns_dataframe_with_features(self):
        self._write_test_data()
        result = self.eng.engineer_date(self.date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # 10 option bars

    def test_feature_columns_present(self):
        self._write_test_data()
        result = self.eng.engineer_date(self.date)
        for col in ["hour_et", "minutes_since_open", "spy_return_1m", "spy_rsi_14",
                    "opt_return_1m", "moneyness", "implied_volatility",
                    "target", "max_gain_120m"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_target_column_binary(self):
        self._write_test_data()
        result = self.eng.engineer_date(self.date)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_csv_file_created(self):
        self._write_test_data()
        # Patch output path to tmp dir
        self.eng._output_path = self.tmp / "features"
        self.eng.engineer_date(self.date)
        expected = self.tmp / "features" / f"{self.date}_features.csv"
        assert expected.exists()

    def test_date_column_populated(self):
        self._write_test_data()
        result = self.eng.engineer_date(self.date)
        assert "date" in result.columns
        assert (result["date"] == self.date).all()


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    def setup_method(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.config = _make_config(self.tmp)
        self.eng = MLFeatureEngineer(self.config)

    def test_skips_missing_dates(self):
        stats = self.eng.run("2025-03-03", "2025-03-05")
        assert stats["dates_skipped"] == 3
        assert stats["dates_processed"] == 0

    def test_counts_rows(self):
        date = "2025-03-03"
        raw = self.tmp / "raw"
        spy_df = _make_spy_df(date, n_bars=30)
        _write_parquet(spy_df, raw / "spy" / f"{date}.parquet")

        ticker = "O:SPY250304C00594000"
        safe = "O_SPY250304C00594000"
        opt_df = _make_option_df(date, ticker, n_bars=8)
        _write_parquet(opt_df, raw / "options" / "minute" / safe / f"{date}.parquet")

        self.eng._output_path = self.tmp / "features"
        stats = self.eng.run(date, date)
        assert stats["dates_processed"] == 1
        assert stats["total_rows"] == 8

    def test_positive_rate_in_stats(self):
        date = "2025-03-03"
        raw = self.tmp / "raw"
        spy_df = _make_spy_df(date, n_bars=30)
        _write_parquet(spy_df, raw / "spy" / f"{date}.parquet")

        safe = "O_SPY250304C00594000"
        ticker = "O:SPY250304C00594000"
        opt_df = _make_option_df(date, ticker, n_bars=8,
                                 spike_at=3, spike_multiplier=1.30)
        _write_parquet(opt_df, raw / "options" / "minute" / safe / f"{date}.parquet")

        self.eng._output_path = self.tmp / "features"
        stats = self.eng.run(date, date)
        assert 0.0 <= stats["positive_rate"] <= 1.0
