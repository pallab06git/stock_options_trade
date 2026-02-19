# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for HardwareMonitor."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.hardware_monitor import HardwareMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, enabled=True):
    return {
        "pipeline_v2": {
            "hardware": {
                "enabled": enabled,
                "output_dir": str(tmp_path / "reports" / "hardware"),
            }
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHardwareMonitorStartStop:
    def test_start_sets_state(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("test-command")
        assert monitor._command_name == "test-command"
        assert monitor._start_time is not None
        assert monitor._proc is not None

    def test_stop_returns_metrics(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("test-cmd")
        time.sleep(0.05)
        result = monitor.stop()

        assert result["command"] == "test-cmd"
        assert result["elapsed_sec"] > 0
        assert "cpu_pct_avg" in result
        assert "mem_rss_start_mb" in result
        assert "mem_rss_end_mb" in result
        assert "mem_delta_mb" in result
        assert "disk_read_mb" in result
        assert "disk_write_mb" in result

    def test_stop_writes_json(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("write-test")
        monitor.stop()

        out_dir = tmp_path / "reports" / "hardware"
        jsons = list(out_dir.glob("*.json"))
        assert len(jsons) == 1

        with open(jsons[0]) as f:
            data = json.load(f)
        assert data["command"] == "write-test"

    def test_stop_without_start_returns_empty(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        result = monitor.stop()
        assert result == {}

    def test_disabled_start_stop_no_op(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path, enabled=False))
        monitor.start("disabled-cmd")
        result = monitor.stop()
        assert result == {}
        # No JSON should be written
        out_dir = tmp_path / "reports" / "hardware"
        assert not out_dir.exists() or len(list(out_dir.glob("*.json"))) == 0

    def test_json_filename_format(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("my-command")
        monitor.stop()

        out_dir = tmp_path / "reports" / "hardware"
        jsons = list(out_dir.glob("*.json"))
        assert len(jsons) == 1
        # Filename should be YYYY-MM-DD_{command}.json
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}_.*\.json$", jsons[0].name)

    def test_iso_timestamps(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("ts-test")
        result = monitor.stop()
        assert "T" in result["start_time"]
        assert "T" in result["end_time"]


class TestDailySummary:
    def test_returns_empty_when_no_dir(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        df = monitor.daily_summary("2025-03-03")
        assert df.empty

    def test_loads_json_files(self, tmp_path):
        monitor = HardwareMonitor(_make_config(tmp_path))
        monitor.start("cmd-1")
        monitor.stop()

        monitor2 = HardwareMonitor(_make_config(tmp_path))
        monitor2.start("cmd-2")
        monitor2.stop()

        # Use the same date format as _write_json (local datetime)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        df = monitor.daily_summary(today)
        assert len(df) >= 1
        assert "command" in df.columns


class TestTrackHardwareDecorator:
    def test_decorator_runs_function(self, tmp_path):
        results = []

        @HardwareMonitor.track_hardware("decorated-fn")
        def my_fn():
            results.append(True)

        my_fn()
        assert results == [True]

    def test_decorator_does_not_suppress_exceptions(self, tmp_path):
        @HardwareMonitor.track_hardware("error-fn")
        def failing_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_fn()
