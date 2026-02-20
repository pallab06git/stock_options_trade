# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for OptionsScanner."""

from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import pytz

from src.processing.options_scanner import OptionsScanner


ET_TZ = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _et_ts(date_str: str, hour: int, minute: int) -> int:
    from datetime import datetime
    dt = ET_TZ.localize(
        datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute)
    )
    return int(dt.timestamp() * 1000)


def _make_config(tmp_path):
    features_path = str(tmp_path / "features")
    return {
        "pipeline_v2": {
            "scanner": {
                "reference_window_minutes": 10,
                "trigger_threshold_pct": 20.0,
                "sustained_threshold_pct": 10.0,
            },
            "reporting": {
                "reports_dir": str(tmp_path / "reports"),
                "features_path": features_path,
            },
        }
    }


def _write_feature_file(tmp_path, date: str, ticker: str, closes: list):
    """Write a minimal options feature Parquet with given close prices."""
    safe = ticker.replace(":", "_")
    feat_dir = tmp_path / "features" / "options" / safe
    feat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "timestamp": _et_ts(date, 9, 30 + i),
            "close": c,
            "minutes_since_open": i,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(feat_dir / f"{date}.parquet", index=False)
    return feat_dir / f"{date}.parquet"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOptionsScanner:
    def test_no_event_flat_price(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        prices = [10.0] * 30
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert events == []

    def test_detects_20pct_move(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        # First 10 bars at 10.0, then a 25% spike
        prices = [10.0] * 10 + [12.5] + [12.5] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        assert events[0]["gain_pct"] >= 20.0

    def test_event_schema(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        prices = [10.0] * 10 + [13.0] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        e = events[0]
        required_keys = {
            "date", "ticker", "trigger_time_et", "reference_time_et",
            "gain_pct", "above_20pct_duration_min", "above_10pct_duration_min",
            "reference_price", "trigger_price",
        }
        assert required_keys.issubset(e.keys())

    def test_below_trigger_no_event(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        # Only 15% move, below 20% threshold
        prices = [10.0] * 10 + [11.5]
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert events == []

    def test_consumed_bars_no_double_count(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        # One large spike then stays elevated
        prices = [10.0] * 10 + [13.0] * 20
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        # Should only find one event (not one per bar in the spike)
        assert len(events) == 1

    def test_no_refire_from_same_reference_low(self, tmp_path):
        """After a move drops below 10%, the scanner must NOT re-trigger from the
        same reference low.  This was the live-data bug: one intraday trough
        generating 5–10 events on the same contract-day."""
        scanner = OptionsScanner(_make_config(tmp_path))
        # Low: 10.0 (bars 0–9)
        # Spike A: 12.5 (+25%) bars 10–14  → event fires
        # Drop:    10.5 (< 10% above 10.0=11.0) bars 15–19  → below sustained
        # Spike B: 12.5 (+19% above new ref 10.5) bars 20–24 → should NOT fire
        #          (19% < 20% threshold from the fresh reference window)
        prices = [10.0] * 10 + [12.5] * 5 + [10.5] * 5 + [12.5] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) == 1, (
            f"Expected 1 event (same-low re-fire suppressed), got {len(events)}: {events}"
        )

    def test_independent_second_event_allowed(self, tmp_path):
        """A genuine new 20%+ move from a fresh low established AFTER the first
        event ends must still be detected as a separate event."""
        scanner = OptionsScanner(_make_config(tmp_path))
        # Event A: low 10.0 → spike 12.5 → drops to 10.5 (event ends)
        # New low:  9.0 established after event A
        # Event B: 9.0 → spike 11.0 (+22%) — genuinely independent
        prices = (
            [10.0] * 10   # reference low for event A
            + [12.5] * 5  # event A spike
            + [10.5] * 3  # drops below 10% (10.0 * 1.10 = 11.0 > 10.5)
            + [9.0]  * 3  # new lower low after event A ends
            + [11.0] * 5  # event B: 11.0 / 9.0 = +22.2%
        )
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) == 2, (
            f"Expected 2 independent events, got {len(events)}: {events}"
        )


class TestScan:
    def test_scan_empty_dir(self, tmp_path):
        """Scanner on a tmp dir with no feature files should find nothing."""
        scanner = OptionsScanner(_make_config(tmp_path))
        # features_path points to tmp_path/features — does not exist yet
        events = scanner.scan("2025-03-01", "2025-03-31")
        assert events == []

    def test_scan_finds_events(self, tmp_path):
        """Scanner traverses options feature dir and finds events."""
        prices = [10.0] * 10 + [13.0] * 5
        _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        scanner = OptionsScanner(_make_config(tmp_path))
        events = scanner.scan("2025-03-01", "2025-03-31")
        assert len(events) >= 1


class TestGenerateReport:
    def test_creates_csv(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        events = [
            {
                "date": "2025-03-03",
                "ticker": "O:SPY250305C00605000",
                "trigger_time_et": "10:30:00",
                "reference_time_et": "09:30:00",
                "gain_pct": 25.0,
                "above_20pct_duration_min": 5,
                "above_10pct_duration_min": 10,
                "reference_price": 10.0,
                "trigger_price": 12.5,
            }
        ]
        path = scanner.generate_report(events, "2025-03-01", "2025-03-31")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["gain_pct"] == 25.0

    def test_empty_events_creates_empty_csv(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        path = scanner.generate_report([], "2025-03-01", "2025-03-31")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 0


_SAMPLE_EVENT = {
    "date": "2025-03-03",
    "ticker": "O:SPY250305C00605000",
    "trigger_time_et": "10:30:00",
    "reference_time_et": "09:30:00",
    "gain_pct": 25.0,
    "above_20pct_duration_min": 10,
    "above_10pct_duration_min": 15,
    "reference_price": 10.0,
    "trigger_price": 12.5,
}


class TestScanStats:
    def test_scan_updates_contract_days_and_bars(self, tmp_path):
        """scan() tracks contract_days and total_bars in _last_scan_stats."""
        prices = [10.0] * 20
        _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner.scan("2025-03-01", "2025-03-31")
        assert scanner._last_scan_stats["contract_days"] == 1
        assert scanner._last_scan_stats["total_bars"] == 20

    def test_scan_stats_empty_dir(self, tmp_path):
        """No feature files → stats are zeros."""
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner.scan("2025-03-01", "2025-03-31")
        assert scanner._last_scan_stats["contract_days"] == 0
        assert scanner._last_scan_stats["total_bars"] == 0

    def test_scan_multiple_contract_days(self, tmp_path):
        """Two ticker-date parquets → contract_days=2, bars summed."""
        _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", [10.0] * 15)
        _write_feature_file(tmp_path, "2025-03-04", "O:SPY250305C00606000", [8.0] * 25)
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner.scan("2025-03-01", "2025-03-31")
        assert scanner._last_scan_stats["contract_days"] == 2
        assert scanner._last_scan_stats["total_bars"] == 40


class TestGenerateReportMetrics:
    def test_prints_contract_days_and_bars(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 5, "total_bars": 300}
        scanner.generate_report([_SAMPLE_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Contract-days scanned:" in out
        assert "5" in out
        assert "Total minute bars:" in out
        assert "300" in out

    def test_prints_total_events(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 2, "total_bars": 100}
        scanner.generate_report([_SAMPLE_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Total events:" in out
        assert "1" in out

    def test_prints_events_per_contract_day(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 3, "total_bars": 150}
        # 1 event across 3 contract-days → two zero-event days
        scanner.generate_report([_SAMPLE_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Events/contract-day:" in out
        assert "min=0" in out
        assert "max=1" in out

    def test_prints_positive_minute_stats(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Total >20% minutes:" in out
        assert "10" in out  # above_20pct_duration_min = 10
        assert "Positive-minute rate:" in out
        assert "10.00%" in out  # 10/100 * 100
        assert "Duration >20% (med/mean):" in out

    def test_prints_hour_distribution(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Event distribution by trigger hour" in out
        assert "10:xx" in out  # trigger_time_et = "10:30:00"

    def test_no_events_prints_zeros(self, tmp_path, capsys):
        scanner = OptionsScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 5, "total_bars": 300}
        scanner.generate_report([], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Total events:             0" in out
        assert "0.00%" in out
        assert "min=0 median=0.0 max=0" in out


class TestLoadReports:
    def test_load_returns_empty_when_no_dir(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        df = scanner.load_reports()
        assert df.empty

    def test_load_and_filter(self, tmp_path):
        scanner = OptionsScanner(_make_config(tmp_path))
        events = [
            {
                "date": "2025-03-10",
                "ticker": "O:SPY250312C00610000",
                "trigger_time_et": "11:00:00",
                "reference_time_et": "09:30:00",
                "gain_pct": 30.0,
                "above_20pct_duration_min": 3,
                "above_10pct_duration_min": 7,
                "reference_price": 8.0,
                "trigger_price": 10.4,
            }
        ]
        scanner.generate_report(events, "2025-03-01", "2025-03-31")

        # Load without filter
        df = scanner.load_reports()
        assert len(df) >= 1

        # Load with date filter that excludes the event
        df_filtered = scanner.load_reports(start_date="2025-04-01")
        assert len(df_filtered) == 0
