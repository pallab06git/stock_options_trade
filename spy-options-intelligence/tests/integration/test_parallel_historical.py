# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for multi-ticker parallel backfill.

Tests the ParallelRunner, ProcessManager, and HealthDashboard together
using mocked subprocesses to verify the full orchestration flow.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.orchestrator.parallel_runner import ParallelRunner
from src.orchestrator.process_manager import ProcessManager
from src.monitoring.health_dashboard import HealthDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tickers=None, rate_limit=10):
    return {
        "orchestrator": {
            "tickers": tickers or ["SPY", "TSLA"],
            "max_workers": len(tickers or ["SPY", "TSLA"]),
        },
        "polygon": {
            "api_key": "pk_test_12345678",
            "rate_limiting": {
                "total_requests_per_minute": rate_limit,
            },
        },
    }


def _mock_popen(pid=12345, exit_code=0, stdout=b"ok", stderr=b""):
    proc = MagicMock()
    proc.pid = pid
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = exit_code
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelBackfillIntegration:
    """End-to-end test: spawn, registry, workers list, health."""

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_multi_ticker_spawn_and_registry(self, mock_popen_cls, tmp_path):
        """backfill-all spawns N processes and writes registry."""
        pid_counter = {"value": 1000}

        def make_proc(*args, **kwargs):
            pid_counter["value"] += 1
            return _mock_popen(pid=pid_counter["value"])

        mock_popen_cls.side_effect = make_proc

        config = _make_config(tickers=["SPY", "TSLA"], rate_limit=10)
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        results = runner.run("2025-01-27", "2025-01-27")

        # Both tickers spawned
        assert "SPY" in results
        assert "TSLA" in results
        assert results["SPY"]["exit_code"] == 0
        assert results["TSLA"]["exit_code"] == 0

        # Registry has both tickers
        registry = json.loads(runner.REGISTRY_PATH.read_text())
        assert "SPY" in registry
        assert "TSLA" in registry

        # Rate limit divided equally
        assert runner.per_worker_rate == 5.0

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_workers_list_after_spawn(self, mock_popen_cls, tmp_path):
        """workers list shows processes from registry after backfill-all."""
        mock_popen_cls.return_value = _mock_popen(pid=42)

        config = _make_config(tickers=["SPY"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"
        ParallelRunner.REGISTRY_PATH = runner.REGISTRY_PATH

        runner.run("2025-01-27", "2025-01-27")

        manager = ProcessManager({})
        workers = manager.list_workers()

        assert len(workers) == 1
        assert workers[0]["ticker"] == "SPY"
        assert workers[0]["pid"] == 42

    @patch("src.orchestrator.process_manager.os.kill")
    @patch("src.orchestrator.process_manager.psutil.pid_exists", return_value=True)
    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_stop_single_worker(self, mock_popen_cls, mock_pid, mock_kill, tmp_path):
        """workers stop --ticker SPY stops only SPY."""
        pid_counter = {"value": 100}

        def make_proc(*args, **kwargs):
            pid_counter["value"] += 1
            return _mock_popen(pid=pid_counter["value"])

        mock_popen_cls.side_effect = make_proc

        config = _make_config(tickers=["SPY", "TSLA"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"
        ParallelRunner.REGISTRY_PATH = runner.REGISTRY_PATH

        runner.run("2025-01-27", "2025-01-27")

        manager = ProcessManager({})
        result = manager.stop_worker("SPY")

        assert result is True
        # Only one kill call (for SPY)
        assert mock_kill.call_count == 1

        # Registry should show SPY as stopped
        registry = json.loads(runner.REGISTRY_PATH.read_text())
        assert registry["SPY"]["status"] == "stopped"
        assert registry["TSLA"]["status"] == "completed"

    @patch("src.monitoring.health_dashboard.psutil.pid_exists", return_value=False)
    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_health_dashboard_after_backfill(self, mock_popen_cls, mock_pid, tmp_path):
        """health shows session data after backfill completes."""
        mock_popen_cls.return_value = _mock_popen(pid=42)

        config = _make_config(tickers=["SPY"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"
        ParallelRunner.REGISTRY_PATH = runner.REGISTRY_PATH

        runner.run("2025-01-27", "2025-01-27")

        # Create a mock metrics file
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        metrics_file = metrics_dir / "metrics_spy_2025-01-27_143000.json"
        metrics_file.write_text(json.dumps({
            "total_records_processed": 930,
            "total_operations": 5,
            "memory_usage_mb": 150.0,
            "stale_operations": [],
            "operations": {},
        }))

        dashboard = HealthDashboard(metrics_dir=str(metrics_dir))
        summary = dashboard.get_health_summary()

        assert "SPY" in summary
        assert summary["SPY"]["total_records"] == 930
        assert summary["SPY"]["pid"] == 42

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_separate_checkpoint_files(self, mock_popen_cls, tmp_path):
        """Each ticker spawns with its own --ticker flag for separate checkpoints."""
        cmds = []

        def capture_cmd(cmd, **kwargs):
            cmds.append(cmd)
            return _mock_popen()

        mock_popen_cls.side_effect = capture_cmd

        config = _make_config(tickers=["SPY", "TSLA"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        runner.run("2025-01-27", "2025-01-27")

        # Verify each command has its own --ticker
        tickers_in_cmds = []
        for cmd in cmds:
            ticker_idx = cmd.index("--ticker") + 1
            tickers_in_cmds.append(cmd[ticker_idx])

        assert "SPY" in tickers_in_cmds
        assert "TSLA" in tickers_in_cmds
