# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for SchemaMonitor."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.monitoring.schema_monitor import SchemaMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal config dict with monitoring.schema settings."""
    defaults = {
        "alert_on_new_columns": True,
        "alert_on_missing_columns": True,
        "alert_on_type_changes": True,
        "auto_update_baseline": False,
    }
    defaults.update(overrides)
    return {"monitoring": {"schema": defaults}}


def _make_monitor(**overrides):
    """Create a SchemaMonitor with default config (overridable)."""
    return SchemaMonitor(_make_config(**overrides))


def _write_parquet(path, columns):
    """Write a simple Parquet file from a column dict.

    Args:
        path: File path to write.
        columns: Dict of column_name → list of values.
    """
    table = pa.table(columns)
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for __init__ configuration parsing."""

    def test_defaults_from_empty_config(self):
        """Empty config falls back to hardcoded defaults."""
        monitor = SchemaMonitor({})

        assert monitor.alert_on_new_columns is True
        assert monitor.alert_on_missing_columns is True
        assert monitor.alert_on_type_changes is True
        assert monitor.auto_update_baseline is False

    def test_custom_config(self):
        """Custom config values are applied."""
        config = _make_config(
            alert_on_new_columns=False,
            auto_update_baseline=True,
        )
        monitor = SchemaMonitor(config)

        assert monitor.alert_on_new_columns is False
        assert monitor.alert_on_missing_columns is True
        assert monitor.auto_update_baseline is True


# ---------------------------------------------------------------------------
# Test: capture_baseline
# ---------------------------------------------------------------------------

class TestCaptureBaseline:
    """Tests for capture_baseline()."""

    def test_extract_from_parquet(self, tmp_path):
        """Extracts column names and dtypes from Parquet metadata."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {
            "timestamp": [1, 2, 3],
            "open": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300],
        })

        monitor = _make_monitor()
        baseline = monitor.capture_baseline("spy", str(pq_file))

        assert baseline["source"] == "spy"
        assert "captured_at" in baseline
        assert baseline["sample_file"] == str(pq_file)
        assert "timestamp" in baseline["schema"]
        assert "open" in baseline["schema"]
        assert "volume" in baseline["schema"]

    def test_column_count(self, tmp_path):
        """column_count matches number of columns."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {
            "a": [1],
            "b": [2.0],
            "c": ["x"],
            "d": [True],
        })

        monitor = _make_monitor()
        baseline = monitor.capture_baseline("test", str(pq_file))

        assert baseline["column_count"] == 4


# ---------------------------------------------------------------------------
# Test: detect_schema_changes
# ---------------------------------------------------------------------------

class TestDetectSchemaChanges:
    """Tests for detect_schema_changes() — pure diff logic."""

    def test_no_changes(self):
        """Identical schemas produce empty changes."""
        monitor = _make_monitor()
        baseline = {"a": "int64", "b": "double"}
        current = {"a": "int64", "b": "double"}

        changes = monitor.detect_schema_changes(baseline, current)

        assert changes["new_columns"] == []
        assert changes["missing_columns"] == []
        assert changes["type_changes"] == []

    def test_new_column(self):
        """Column in current but not baseline is detected."""
        monitor = _make_monitor()
        baseline = {"a": "int64"}
        current = {"a": "int64", "b": "double"}

        changes = monitor.detect_schema_changes(baseline, current)

        assert changes["new_columns"] == ["b"]
        assert changes["missing_columns"] == []
        assert changes["type_changes"] == []

    def test_missing_column(self):
        """Column in baseline but not current is detected."""
        monitor = _make_monitor()
        baseline = {"a": "int64", "b": "double"}
        current = {"a": "int64"}

        changes = monitor.detect_schema_changes(baseline, current)

        assert changes["new_columns"] == []
        assert changes["missing_columns"] == ["b"]
        assert changes["type_changes"] == []

    def test_type_change(self):
        """Same column with different type is detected."""
        monitor = _make_monitor()
        baseline = {"a": "int64", "b": "double"}
        current = {"a": "int64", "b": "string"}

        changes = monitor.detect_schema_changes(baseline, current)

        assert changes["new_columns"] == []
        assert changes["missing_columns"] == []
        assert len(changes["type_changes"]) == 1
        assert changes["type_changes"][0]["column"] == "b"
        assert changes["type_changes"][0]["baseline_type"] == "double"
        assert changes["type_changes"][0]["current_type"] == "string"

    def test_multiple_changes(self):
        """Detects new, missing, and type changes simultaneously."""
        monitor = _make_monitor()
        baseline = {"a": "int64", "b": "double", "c": "string"}
        current = {"a": "float32", "c": "string", "d": "bool"}

        changes = monitor.detect_schema_changes(baseline, current)

        assert changes["new_columns"] == ["d"]
        assert changes["missing_columns"] == ["b"]
        assert len(changes["type_changes"]) == 1
        assert changes["type_changes"][0]["column"] == "a"


# ---------------------------------------------------------------------------
# Test: format_alerts
# ---------------------------------------------------------------------------

class TestFormatAlerts:
    """Tests for format_alerts() — alert message generation."""

    def test_new_column_alert(self):
        """New columns produce an alert."""
        monitor = _make_monitor()
        changes = {
            "new_columns": ["extra_col"],
            "missing_columns": [],
            "type_changes": [],
        }

        alerts = monitor.format_alerts("spy", changes)

        assert len(alerts) == 1
        assert "New columns detected" in alerts[0]
        assert "extra_col" in alerts[0]
        assert "[spy]" in alerts[0]

    def test_missing_column_alert(self):
        """Missing columns produce an alert."""
        monitor = _make_monitor()
        changes = {
            "new_columns": [],
            "missing_columns": ["gone_col"],
            "type_changes": [],
        }

        alerts = monitor.format_alerts("vix", changes)

        assert len(alerts) == 1
        assert "Missing columns" in alerts[0]
        assert "gone_col" in alerts[0]

    def test_type_change_alert(self):
        """Type changes produce one alert per column."""
        monitor = _make_monitor()
        changes = {
            "new_columns": [],
            "missing_columns": [],
            "type_changes": [
                {"column": "price", "baseline_type": "int64", "current_type": "double"},
            ],
        }

        alerts = monitor.format_alerts("options", changes)

        assert len(alerts) == 1
        assert "Type change" in alerts[0]
        assert "'price'" in alerts[0]
        assert "int64" in alerts[0]
        assert "double" in alerts[0]

    def test_config_suppression(self):
        """Disabled alert toggles suppress corresponding alerts."""
        monitor = _make_monitor(
            alert_on_new_columns=False,
            alert_on_missing_columns=False,
            alert_on_type_changes=False,
        )
        changes = {
            "new_columns": ["a"],
            "missing_columns": ["b"],
            "type_changes": [
                {"column": "c", "baseline_type": "int64", "current_type": "string"},
            ],
        }

        alerts = monitor.format_alerts("spy", changes)

        assert alerts == []


# ---------------------------------------------------------------------------
# Test: check_drift
# ---------------------------------------------------------------------------

class TestCheckDrift:
    """Tests for check_drift() — end-to-end drift checking."""

    def test_no_baseline_auto_captures(self, tmp_path):
        """First call auto-captures baseline and returns empty alerts."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {"a": [1], "b": [2.0]})

        monitor = _make_monitor()
        monitor._baseline_dir = tmp_path / "baselines"

        alerts = monitor.check_drift("spy", str(pq_file))

        assert alerts == []
        # Baseline file should now exist
        assert (tmp_path / "baselines" / "spy_baseline.json").exists()

    def test_drift_detected(self, tmp_path):
        """Returns alerts when schema differs from baseline."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {"a": [1], "b": [2.0], "c": ["x"]})

        monitor = _make_monitor()
        monitor._baseline_dir = tmp_path / "baselines"
        monitor._drift_dir = tmp_path / "drift"

        # Set up baseline with different schema (missing "c")
        baseline = {
            "source": "spy",
            "captured_at": "2026-01-01T00:00:00+00:00",
            "sample_file": str(pq_file),
            "schema": {"a": "int64", "b": "double"},
            "column_count": 2,
        }
        monitor.save_baseline("spy", baseline)

        alerts = monitor.check_drift("spy", str(pq_file))

        assert len(alerts) >= 1
        assert any("New columns" in a for a in alerts)

    def test_no_drift(self, tmp_path):
        """Returns empty list when schema matches baseline."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {"a": [1], "b": [2.0]})

        monitor = _make_monitor()
        monitor._baseline_dir = tmp_path / "baselines"

        # Capture baseline from the same file
        baseline = monitor.capture_baseline("spy", str(pq_file))
        monitor.save_baseline("spy", baseline)

        alerts = monitor.check_drift("spy", str(pq_file))

        assert alerts == []


# ---------------------------------------------------------------------------
# Test: save_baseline / load_baseline
# ---------------------------------------------------------------------------

class TestSaveLoadBaseline:
    """Tests for baseline persistence."""

    def test_roundtrip(self, tmp_path):
        """Save then load returns identical data."""
        monitor = _make_monitor()
        monitor._baseline_dir = tmp_path / "baselines"

        baseline = {
            "source": "spy",
            "captured_at": "2026-02-14T15:30:00+00:00",
            "sample_file": "data/raw/spy/2026-02-14.parquet",
            "schema": {"timestamp": "int64", "open": "double"},
            "column_count": 2,
        }

        monitor.save_baseline("spy", baseline)
        loaded = monitor.load_baseline("spy")

        assert loaded == baseline

    def test_nonexistent_returns_none(self, tmp_path):
        """Loading a non-existent baseline returns None."""
        monitor = _make_monitor()
        monitor._baseline_dir = tmp_path / "baselines"

        result = monitor.load_baseline("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# Test: log_drift
# ---------------------------------------------------------------------------

class TestLogDrift:
    """Tests for drift event logging."""

    def test_writes_drift_json(self, tmp_path):
        """log_drift writes a JSON file with drift details."""
        monitor = _make_monitor()
        monitor._drift_dir = tmp_path / "drift"

        changes = {
            "new_columns": ["extra"],
            "missing_columns": [],
            "type_changes": [],
        }

        path = monitor.log_drift("spy", "2026-02-14", changes)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["source"] == "spy"
        assert data["date"] == "2026-02-14"
        assert data["new_columns"] == ["extra"]
        assert "detected_at" in data


# ---------------------------------------------------------------------------
# Test: auto_update_baseline
# ---------------------------------------------------------------------------

class TestAutoUpdateBaseline:
    """Tests for auto_update_baseline behavior."""

    def test_auto_update_on_drift(self, tmp_path):
        """When auto_update_baseline=True, baseline is updated on drift."""
        pq_file = tmp_path / "test.parquet"
        _write_parquet(pq_file, {"a": [1], "b": [2.0], "c": ["x"]})

        monitor = _make_monitor(auto_update_baseline=True)
        monitor._baseline_dir = tmp_path / "baselines"
        monitor._drift_dir = tmp_path / "drift"

        # Save old baseline missing "c"
        old_baseline = {
            "source": "spy",
            "captured_at": "2026-01-01T00:00:00+00:00",
            "sample_file": str(pq_file),
            "schema": {"a": "int64", "b": "double"},
            "column_count": 2,
        }
        monitor.save_baseline("spy", old_baseline)

        # Check drift — should detect "c" as new, then auto-update
        alerts = monitor.check_drift("spy", str(pq_file))
        assert len(alerts) >= 1

        # Baseline should now include "c"
        updated = monitor.load_baseline("spy")
        assert "c" in updated["schema"]
        assert updated["column_count"] == 3
