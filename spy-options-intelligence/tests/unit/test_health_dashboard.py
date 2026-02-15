# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for HealthDashboard."""

import json
import pytest
from unittest.mock import patch

from src.monitoring.health_dashboard import HealthDashboard
from src.orchestrator.parallel_runner import ParallelRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics_file(metrics_dir, label, timestamp_str, data):
    """Create a metrics JSON file in the expected naming format."""
    filename = f"metrics_{label}_{timestamp_str}.json"
    path = metrics_dir / filename
    path.write_text(json.dumps(data))
    return path


def _make_registry(tmp_path, data):
    """Write a registry file and point ParallelRunner at it."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    ParallelRunner.REGISTRY_PATH = path
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetAllSessions:
    """Tests for get_all_sessions()."""

    def test_empty_dir(self, tmp_path):
        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        assert dashboard.get_all_sessions() == {}

    def test_nonexistent_dir(self, tmp_path):
        dashboard = HealthDashboard(metrics_dir=str(tmp_path / "nonexistent"))
        assert dashboard.get_all_sessions() == {}

    def test_single_session(self, tmp_path):
        _make_metrics_file(tmp_path, "spy", "2025-01-27_143000", {
            "total_records_processed": 1000,
            "total_operations": 50,
            "memory_usage_mb": 200.5,
        })

        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        sessions = dashboard.get_all_sessions()

        assert "spy" in sessions
        assert sessions["spy"]["total_records_processed"] == 1000

    def test_multiple_sessions(self, tmp_path):
        _make_metrics_file(tmp_path, "spy", "2025-01-27_143000", {
            "total_records_processed": 1000,
        })
        _make_metrics_file(tmp_path, "tsla", "2025-01-27_143001", {
            "total_records_processed": 500,
        })

        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        sessions = dashboard.get_all_sessions()

        assert "spy" in sessions
        assert "tsla" in sessions

    def test_latest_file_wins(self, tmp_path):
        """When multiple files exist for same session, most recent is used."""
        _make_metrics_file(tmp_path, "spy", "2025-01-27_100000", {
            "total_records_processed": 500,
        })
        _make_metrics_file(tmp_path, "spy", "2025-01-27_143000", {
            "total_records_processed": 1000,
        })

        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        sessions = dashboard.get_all_sessions()

        # Sorted reverse by name, so later timestamp file is read first
        assert sessions["spy"]["total_records_processed"] == 1000


class TestGetHealthSummary:
    """Tests for get_health_summary()."""

    @patch("src.monitoring.health_dashboard.psutil.pid_exists", return_value=True)
    def test_merge_registry_and_metrics(self, mock_pid, tmp_path):
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()

        _make_registry(tmp_path, {
            "SPY": {"pid": 1001, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
        })
        _make_metrics_file(metrics_dir, "spy", "2025-01-27_143000", {
            "total_records_processed": 45000,
            "total_operations": 150,
            "memory_usage_mb": 234.5,
            "stale_operations": [],
            "operations": {"write": {"latency_p50": 0.05}},
        })

        dashboard = HealthDashboard(metrics_dir=str(metrics_dir))
        summary = dashboard.get_health_summary()

        assert "SPY" in summary
        assert summary["SPY"]["pid"] == 1001
        assert summary["SPY"]["alive"] is True
        assert summary["SPY"]["total_records"] == 45000
        assert summary["SPY"]["memory_mb"] == 234.5

    @patch("src.monitoring.health_dashboard.psutil.pid_exists", return_value=False)
    def test_dead_process_in_summary(self, mock_pid, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 9999, "started_at": "2025-01-27T10:00:00Z", "status": "completed"},
        })

        dashboard = HealthDashboard(metrics_dir=str(tmp_path / "empty"))
        summary = dashboard.get_health_summary()

        assert summary["SPY"]["alive"] is False

    def test_empty_summary(self, tmp_path):
        _make_registry(tmp_path, {})

        dashboard = HealthDashboard(metrics_dir=str(tmp_path / "empty"))
        summary = dashboard.get_health_summary()

        assert summary == {}


class TestFormatTable:
    """Tests for format_table()."""

    def test_empty_summary(self):
        dashboard = HealthDashboard()
        assert dashboard.format_table({}) == "No sessions found."

    def test_formatted_output(self):
        dashboard = HealthDashboard()
        summary = {
            "SPY": {
                "pid": 1001,
                "alive": True,
                "total_records": 45000,
                "total_operations": 150,
                "memory_mb": 234.5,
                "stale_operations": [],
                "status": "running",
            },
        }

        table = dashboard.format_table(summary)

        assert "SPY" in table
        assert "1001" in table
        assert "running" in table
        assert "45000" in table


class TestGetSessionDetail:
    """Tests for get_session_detail()."""

    def test_existing_session(self, tmp_path):
        _make_metrics_file(tmp_path, "spy", "2025-01-27_143000", {
            "total_records_processed": 1000,
            "operations": {"write": {"latency_p50": 0.05, "latency_p95": 0.2}},
        })

        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        detail = dashboard.get_session_detail("SPY")

        assert detail is not None
        assert detail["total_records_processed"] == 1000
        assert "write" in detail["operations"]

    def test_nonexistent_session(self, tmp_path):
        dashboard = HealthDashboard(metrics_dir=str(tmp_path))
        assert dashboard.get_session_detail("AAPL") is None
