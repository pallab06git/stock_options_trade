# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for ProcessManager."""

import json
import os
import signal
import pytest
from unittest.mock import patch, MagicMock

from src.orchestrator.process_manager import ProcessManager
from src.orchestrator.parallel_runner import ParallelRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path, data):
    """Write a registry file and point ParallelRunner at it."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    ParallelRunner.REGISTRY_PATH = path
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListWorkers:
    """Tests for list_workers()."""

    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=True)
    def test_list_running_workers(self, mock_pid, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 1001, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
            "TSLA": {"pid": 1002, "started_at": "2025-01-27T10:00:01Z", "status": "running"},
        })

        manager = ProcessManager({})
        workers = manager.list_workers()

        assert len(workers) == 2
        spy = next(w for w in workers if w["ticker"] == "SPY")
        assert spy["pid"] == 1001
        assert spy["alive"] is True

    def test_list_empty_registry(self, tmp_path):
        _make_registry(tmp_path, {})

        manager = ProcessManager({})
        workers = manager.list_workers()

        assert workers == []

    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=False)
    def test_list_dead_workers(self, mock_pid, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 9999, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
        })

        manager = ProcessManager({})
        workers = manager.list_workers()

        assert len(workers) == 1
        assert workers[0]["alive"] is False


class TestStopWorker:
    """Tests for stop_worker()."""

    @patch("src.orchestrator.process_manager.os.kill")
    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=True)
    def test_stop_running_worker(self, mock_pid, mock_kill, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 1001, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
        })

        manager = ProcessManager({})
        result = manager.stop_worker("SPY")

        assert result is True
        mock_kill.assert_called_once_with(1001, signal.SIGTERM)

        # Registry should be updated
        registry = json.loads(ParallelRunner.REGISTRY_PATH.read_text())
        assert registry["SPY"]["status"] == "stopped"

    def test_stop_nonexistent_ticker(self, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 1001, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
        })

        manager = ProcessManager({})
        result = manager.stop_worker("AAPL")

        assert result is False

    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=False)
    def test_stop_dead_process(self, mock_pid, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 9999, "started_at": "2025-01-27T10:00:00Z", "status": "completed"},
        })

        manager = ProcessManager({})
        result = manager.stop_worker("SPY")

        assert result is False


class TestStopAll:
    """Tests for stop_all()."""

    @patch("src.orchestrator.process_manager.os.kill")
    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=True)
    def test_stop_all_workers(self, mock_pid, mock_kill, tmp_path):
        _make_registry(tmp_path, {
            "SPY": {"pid": 1001, "started_at": "2025-01-27T10:00:00Z", "status": "running"},
            "TSLA": {"pid": 1002, "started_at": "2025-01-27T10:00:01Z", "status": "running"},
        })

        manager = ProcessManager({})
        results = manager.stop_all()

        assert results["SPY"] is True
        assert results["TSLA"] is True
        assert mock_kill.call_count == 2
