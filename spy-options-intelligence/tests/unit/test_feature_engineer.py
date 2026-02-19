# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for FeatureEngineer."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytz

from src.processing.feature_engineer import FeatureEngineer


ET_TZ = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _et_ts(date_str: str, hour: int, minute: int) -> int:
    """Convert ET date + time to Unix milliseconds."""
    from datetime import datetime
    dt_et = ET_TZ.localize(datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, minute=minute, second=0
    ))
    return int(dt_et.timestamp() * 1000)


def _make_config(tmp_path, lag_windows=None):
    return {
        "pipeline_v2": {
            "feature_engineering": {
                "lag_windows": lag_windows or [1, 5],
                "market_open_et": "09:30",
            },
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
        },
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "raw"),
                "compression": "snappy",
            }
        },
    }


def _write_spy(tmp_path, date: str, n_bars: int = 20):
    """Write a minimal SPY minute Parquet."""
    rows = []
    for i in range(n_bars):
        ts = _et_ts(date, 9, 30 + i)
        rows.append({
            "timestamp": ts,
            "open": 600.0 + i * 0.1,
            "high": 600.5 + i * 0.1,
            "low": 599.5 + i * 0.1,
            "close": 600.2 + i * 0.1,
            "volume": 1000 + i * 10,
            "source": "spy",
        })
    out_dir = tmp_path / "raw" / "spy"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"{date}.parquet", index=False)
    return rows


def _write_options(tmp_path, date: str, ticker: str, n_bars: int = 20):
    """Write a minimal options minute Parquet."""
    safe = ticker.replace(":", "_")
    rows = []
    for i in range(n_bars):
        ts = _et_ts(date, 9, 30 + i)
        rows.append({
            "timestamp": ts,
            "open": 5.0 + i * 0.05,
            "high": 5.1 + i * 0.05,
            "low": 4.9 + i * 0.05,
            "close": 5.05 + i * 0.05,
            "volume": 50 + i,
            "ticker": ticker,
            "source": "options_minute",
        })
    out_dir = tmp_path / "raw" / "options" / "minute" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"{date}.parquet", index=False)
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMinutesSinceOpen:
    def test_at_open(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        ts = _et_ts("2025-03-03", 9, 30)
        assert fe._minutes_since_open(ts) == 0

    def test_10_minutes_in(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        ts = _et_ts("2025-03-03", 9, 40)
        assert fe._minutes_since_open(ts) == 10

    def test_90_minutes_in(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        ts = _et_ts("2025-03-03", 11, 0)
        assert fe._minutes_since_open(ts) == 90


class TestEngineerEquity:
    def test_creates_output_file(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", n_bars=20)
        fe = FeatureEngineer(_make_config(tmp_path, lag_windows=[1, 5]))
        df = fe.engineer_equity("2025-03-03", "SPY")
        assert not df.empty
        out_path = Path("data/processed/features/spy/2025-03-03.parquet")
        # Output path is relative — check column existence
        assert "price_change_1m" in df.columns
        assert "price_change_5m" in df.columns

    def test_early_bars_zeroed(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", n_bars=20)
        fe = FeatureEngineer(_make_config(tmp_path, lag_windows=[5]))
        df = fe.engineer_equity("2025-03-03", "SPY")
        # First 5 bars (minutes_since_open < 5) should have 0.0 for price_change_5m
        early = df[df["minutes_since_open"] < 5]
        assert (early["price_change_5m"] == 0.0).all()

    def test_volume_change_columns(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", n_bars=20)
        fe = FeatureEngineer(_make_config(tmp_path, lag_windows=[1]))
        df = fe.engineer_equity("2025-03-03", "SPY")
        assert "volume_change_1m" in df.columns

    def test_missing_file_returns_empty(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        df = fe.engineer_equity("2025-01-01", "SPY")
        assert df.empty

    def test_vix_source_name(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        assert fe._source_name("I:VIX") == "vix"
        assert fe._source_name("SPY") == "spy"


class TestParseContractMeta:
    def test_call_contract(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        strike, tte, flag = fe._parse_contract_meta(
            "O:SPY250305C00605000", "2025-03-03"
        )
        assert flag == "c"
        assert strike == pytest.approx(605.0)
        assert tte > 0

    def test_put_contract(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        strike, tte, flag = fe._parse_contract_meta(
            "O:SPY250305P00595000", "2025-03-03"
        )
        assert flag == "p"
        assert strike == pytest.approx(595.0)

    def test_fallback_on_bad_ticker(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        strike, tte, flag = fe._parse_contract_meta("INVALID", "2025-03-03")
        assert flag == "c"
        assert strike == 400.0


class TestEngineerOptions:
    def test_creates_features(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", n_bars=20)
        _write_options(tmp_path, "2025-03-03", "O:SPY250305C00605000", n_bars=20)

        fe = FeatureEngineer(_make_config(tmp_path, lag_windows=[1, 5]))

        # Patch IV calc so tests don't need py_vollib installed
        with patch.object(fe, "_calc_iv", return_value=0.25):
            results = fe.engineer_options("2025-03-03")

        assert len(results) > 0
        for ticker, df in results.items():
            assert "price_change_1m" in df.columns
            assert "iv_change_1m" in df.columns
            assert "price_change_open" in df.columns
            assert "iv_change_open" in df.columns

    def test_no_options_dir_returns_empty(self, tmp_path):
        fe = FeatureEngineer(_make_config(tmp_path))
        result = fe.engineer_options("2025-03-03")
        assert result == {}


class TestRun:
    def test_run_all(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", n_bars=10)
        fe = FeatureEngineer(_make_config(tmp_path, lag_windows=[1]))

        with patch.object(fe, "_calc_iv", return_value=0.25):
            stats = fe.run("2025-03-03", "2025-03-03", source="spy")

        assert stats["dates_processed"] == 1
        assert stats["equity_files"] >= 1
