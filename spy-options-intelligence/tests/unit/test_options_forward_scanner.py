# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for OptionsForwardScanner."""

import pandas as pd
import numpy as np
import pytest
import pytz

from src.processing.options_forward_scanner import OptionsForwardScanner


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


def _make_config(tmp_path, window: int = 10):
    return {
        "pipeline_v2": {
            "scanner": {
                "reference_window_minutes": window,
                "trigger_threshold_pct": 20.0,
                "sustained_threshold_pct": 10.0,
            },
            "reporting": {
                "reports_dir": str(tmp_path / "reports"),
                "features_path": str(tmp_path / "features"),
            },
        }
    }


def _write_feature_file(tmp_path, date: str, ticker: str, closes: list):
    safe = ticker.replace(":", "_")
    feat_dir = tmp_path / "features" / "options" / safe
    feat_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"timestamp": _et_ts(date, 9, 30 + i), "close": c, "minutes_since_open": i}
        for i, c in enumerate(closes)
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(feat_dir / f"{date}.parquet", index=False)
    return feat_dir / f"{date}.parquet"


# ---------------------------------------------------------------------------
# _scan_single unit tests
# ---------------------------------------------------------------------------


class TestOptionsForwardScanner:
    def test_no_event_flat_price(self, tmp_path):
        """Flat prices — no forward 20% gain, no events."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 30
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert events == []

    def test_detects_forward_20pct_move(self, tmp_path):
        """Entry at bar 0 (price 10.0); bars 5–14 at 12.5 (+25%). Event fires."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 5 + [12.5] * 10
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        assert events[0]["gain_pct"] >= 20.0

    def test_event_schema(self, tmp_path):
        """Event dict contains all required keys."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 3 + [13.0] * 10
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        required_keys = {
            "date", "ticker", "entry_time_et", "trigger_time_et",
            "minutes_to_trigger", "gain_pct",
            "above_20pct_duration_min", "above_10pct_duration_min",
            "entry_price", "trigger_price",
        }
        assert required_keys.issubset(events[0].keys())

    def test_minutes_to_trigger_correct(self, tmp_path):
        """Entry at bar 0, first 20%-hit at bar 3 → minutes_to_trigger == 3."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        # bars 0-2: 10.0 (entry at bar 0); bar 3 onward: 12.5 (+25%)
        prices = [10.0] * 3 + [12.5] * 8
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        assert events[0]["minutes_to_trigger"] == 3

    def test_below_trigger_no_event(self, tmp_path):
        """Only 15% forward gain — below 20% threshold, no event."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 5 + [11.5] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert events == []

    def test_forward_window_limits_earliest_entry(self, tmp_path):
        """Bar 0 cannot see a spike more than window bars away; a closer bar can.
        window=5, spike at bar 6: bar 0 looks at bars 1–5 (all 10.0, no trigger).
        Bar 1 looks at bars 2–6 (bar 6 = 12.5, +25%, trigger).
        Exactly 1 event fires (bar 1), minutes_to_trigger == 5."""
        scanner = OptionsForwardScanner(_make_config(tmp_path, window=5))
        prices = [10.0] * 6 + [12.5] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) == 1
        assert events[0]["minutes_to_trigger"] == 5

    def test_deduplication_one_run_one_event(self, tmp_path):
        """Single run (bars 5–14 at 13.0) discovered from entry bar 0.
        Bars 5–14 are consumed; no second event is produced."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 5 + [13.0] * 10
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) == 1

    def test_above_20_duration_counts_consecutive_bars(self, tmp_path):
        """Entry at bar 0 (10.0); trigger at bar 2 (12.5); bars 2–5 at 12.5 (+25%),
        bar 6 drops to 11.5 (still ≥10% but < 20%): above_20_dur == 4."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 2 + [12.5] * 4 + [11.5] * 5
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        assert events[0]["above_20pct_duration_min"] == 4

    def test_above_10_duration_stops_on_drop(self, tmp_path):
        """Sustained (≥10%) duration ends when price first drops below 10%.
        Entry 10.0; trigger at bar 2 (12.5); bars 2–5 at 12.5, bar 6 at 10.5
        (< 10% → 11.0): above_10_dur == 4, stops at bar 6."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [10.0] * 2 + [12.5] * 4 + [10.5] + [12.0] * 4
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        # above_10_dur counts bars from trigger until drop below 10%
        # 10% above entry (10.0) → 11.0; bars 2–5 are 12.5 (≥11.0), bar 6 is 10.5 (<11.0)
        assert events[0]["above_10pct_duration_min"] == 4

    def test_two_independent_runs(self, tmp_path):
        """Two separate runs separated by a low price → two events."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        # Run A: entry bar 0 (10.0), trigger bar 1 (13.0), drops bar 4
        # Run B: new entry bar 5 (9.0), trigger bar 6 (11.0 = +22%)
        prices = (
            [10.0]       # entry bar 0 → sees run A
            + [13.0] * 3 # run A: bars 1-3 (+30%)
            + [9.5]      # drops below 10% from 10.0 (11.0 > 9.5) → run A ends
            + [9.0]      # new entry point for run B
            + [11.0] * 4 # run B: 11.0/9.0 = +22.2%
        )
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) == 2

    def test_entry_price_matches_entry_bar_close(self, tmp_path):
        """entry_price must equal close[t] for the recorded entry bar."""
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        prices = [8.5] * 2 + [11.0] * 8   # 11.0/8.5 = +29.4%
        path = _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        events = scanner._scan_single(path, "O_SPY250305C00605000", "2025-03-03")
        assert len(events) >= 1
        assert abs(events[0]["entry_price"] - 8.5) < 1e-6


# ---------------------------------------------------------------------------
# scan() integration tests
# ---------------------------------------------------------------------------


class TestScanForward:
    def test_scan_empty_dir(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        events = scanner.scan("2025-03-01", "2025-03-31")
        assert events == []

    def test_scan_finds_events(self, tmp_path):
        prices = [10.0] * 3 + [13.0] * 8
        _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        events = scanner.scan("2025-03-01", "2025-03-31")
        assert len(events) >= 1

    def test_scan_updates_stats(self, tmp_path):
        prices = [10.0] * 20
        _write_feature_file(tmp_path, "2025-03-03", "O:SPY250305C00605000", prices)
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner.scan("2025-03-01", "2025-03-31")
        assert scanner._last_scan_stats["contract_days"] == 1
        assert scanner._last_scan_stats["total_bars"] == 20


# ---------------------------------------------------------------------------
# generate_report() tests
# ---------------------------------------------------------------------------

_SAMPLE_FWD_EVENT = {
    "date": "2025-03-03",
    "ticker": "O:SPY250305C00605000",
    "entry_time_et": "09:30:00",
    "trigger_time_et": "10:00:00",
    "minutes_to_trigger": 30,
    "gain_pct": 25.0,
    "above_20pct_duration_min": 8,
    "above_10pct_duration_min": 15,
    "entry_price": 10.0,
    "trigger_price": 12.5,
}


class TestGenerateReportForward:
    def test_creates_csv(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        path = scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["gain_pct"] == 25.0

    def test_empty_events_creates_empty_csv(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        path = scanner.generate_report([], "2025-03-01", "2025-03-31")
        assert path.exists()
        assert pd.read_csv(path).empty

    def test_csv_filename_contains_forward(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        path = scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        assert "forward" in path.name

    def test_prints_minutes_to_trigger(self, tmp_path, capsys):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Mins to trigger" in out

    def test_prints_duration_stats(self, tmp_path, capsys):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Duration" in out
        assert "20%" in out
        assert "10%" in out

    def test_prints_entry_hour_distribution(self, tmp_path, capsys):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Entry distribution by hour" in out
        assert "09:xx" in out   # entry_time_et = "09:30:00"

    def test_prints_trigger_hour_distribution(self, tmp_path, capsys):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Event distribution by trigger hour" in out
        assert "10:xx" in out   # trigger_time_et = "10:00:00"

    def test_no_events_prints_na(self, tmp_path, capsys):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 3, "total_bars": 150}
        scanner.generate_report([], "2025-03-01", "2025-03-31")
        out = capsys.readouterr().out
        assert "Total qualifying entries: 0" in out
        assert "N/A" in out

    def test_load_reports_returns_empty_when_no_dir(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        assert scanner.load_reports().empty

    def test_load_reports_and_filter(self, tmp_path):
        scanner = OptionsForwardScanner(_make_config(tmp_path))
        scanner._last_scan_stats = {"contract_days": 1, "total_bars": 100}
        scanner.generate_report([_SAMPLE_FWD_EVENT], "2025-03-01", "2025-03-31")
        df = scanner.load_reports()
        assert len(df) >= 1
        df_filtered = scanner.load_reports(start_date="2025-04-01")
        assert len(df_filtered) == 0
