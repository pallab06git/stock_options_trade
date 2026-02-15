# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for ParallelRunner."""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from src.orchestrator.parallel_runner import ParallelRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tickers=None, rate_limit=5):
    """Build a minimal config dict for ParallelRunner."""
    return {
        "orchestrator": {
            "tickers": tickers or ["SPY"],
            "max_workers": len(tickers or ["SPY"]),
        },
        "polygon": {
            "api_key": "pk_test_12345678",
            "rate_limiting": {
                "total_requests_per_minute": rate_limit,
            },
        },
    }


def _mock_popen(exit_code=0, stdout=b"ok", stderr=b""):
    """Create a mock Popen object."""
    proc = MagicMock()
    proc.pid = 12345
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = exit_code
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for __init__ configuration parsing."""

    def test_single_ticker(self):
        config = _make_config(tickers=["SPY"], rate_limit=5)
        runner = ParallelRunner(config)

        assert runner.tickers == ["SPY"]
        assert runner.per_worker_rate == 5.0

    def test_multiple_tickers_rate_division(self):
        """Rate limit is divided equally among tickers."""
        config = _make_config(tickers=["SPY", "TSLA"], rate_limit=10)
        runner = ParallelRunner(config)

        assert runner.tickers == ["SPY", "TSLA"]
        assert runner.per_worker_rate == 5.0

    def test_rate_floor_at_one(self):
        """Per-worker rate never goes below 1."""
        config = _make_config(tickers=["SPY", "TSLA", "AAPL"], rate_limit=2)
        runner = ParallelRunner(config)

        assert runner.per_worker_rate == 1

    def test_default_tickers(self):
        """Default is ['SPY'] when orchestrator config is missing."""
        runner = ParallelRunner({})
        assert runner.tickers == ["SPY"]


class TestRun:
    """Tests for run() — subprocess spawning and registry."""

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_spawns_one_process_per_ticker(self, mock_popen_cls, tmp_path):
        """One subprocess is spawned per ticker."""
        config = _make_config(tickers=["SPY", "TSLA"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        mock_popen_cls.return_value = _mock_popen()

        results = runner.run("2025-01-27", "2025-01-27")

        assert mock_popen_cls.call_count == 2
        assert "SPY" in results
        assert "TSLA" in results

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_spawn_command_includes_ticker_and_rate(self, mock_popen_cls, tmp_path):
        """Subprocess command includes --ticker and --rate-limit flags."""
        config = _make_config(tickers=["TSLA"], rate_limit=10)
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        mock_popen_cls.return_value = _mock_popen()

        runner.run("2025-01-27", "2025-01-27")

        cmd = mock_popen_cls.call_args[0][0]
        assert "--ticker" in cmd
        assert "TSLA" in cmd
        assert "--rate-limit" in cmd
        assert "10.0" in cmd

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_resume_flag_passed(self, mock_popen_cls, tmp_path):
        """When resume=True, --resume is passed to subprocesses."""
        config = _make_config(tickers=["SPY"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        mock_popen_cls.return_value = _mock_popen()

        runner.run("2025-01-27", "2025-01-27", resume=True)

        cmd = mock_popen_cls.call_args[0][0]
        assert "--resume" in cmd

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_registry_persisted(self, mock_popen_cls, tmp_path):
        """Registry JSON is written to disk with process info."""
        config = _make_config(tickers=["SPY", "TSLA"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        mock_popen_cls.return_value = _mock_popen()

        runner.run("2025-01-27", "2025-01-27")

        assert runner.REGISTRY_PATH.exists()
        registry = json.loads(runner.REGISTRY_PATH.read_text())
        assert "SPY" in registry
        assert "TSLA" in registry
        assert registry["SPY"]["pid"] == 12345
        assert registry["SPY"]["status"] in ("completed", "failed")

    @patch("src.orchestrator.parallel_runner.subprocess.Popen")
    def test_failed_worker_recorded(self, mock_popen_cls, tmp_path):
        """Failed subprocess is recorded with 'failed' status."""
        config = _make_config(tickers=["SPY"])
        runner = ParallelRunner(config)
        runner.REGISTRY_PATH = tmp_path / "registry.json"

        mock_popen_cls.return_value = _mock_popen(exit_code=1, stderr=b"error")

        results = runner.run("2025-01-27", "2025-01-27")

        assert results["SPY"]["exit_code"] == 1
        registry = json.loads(runner.REGISTRY_PATH.read_text())
        assert registry["SPY"]["status"] == "failed"


class TestLoadRegistry:
    """Tests for load_registry class method."""

    def test_load_empty(self, tmp_path):
        """Returns empty dict when no registry file exists."""
        ParallelRunner.REGISTRY_PATH = tmp_path / "nonexistent.json"
        assert ParallelRunner.load_registry() == {}

    def test_load_existing(self, tmp_path):
        """Returns registry contents from existing file."""
        registry_path = tmp_path / "registry.json"
        data = {"SPY": {"pid": 123, "status": "running"}}
        registry_path.write_text(json.dumps(data))

        ParallelRunner.REGISTRY_PATH = registry_path
        result = ParallelRunner.load_registry()

        assert result == data

    def test_load_corrupt_file(self, tmp_path):
        """Returns empty dict on corrupt JSON."""
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("not json")

        ParallelRunner.REGISTRY_PATH = registry_path
        assert ParallelRunner.load_registry() == {}
