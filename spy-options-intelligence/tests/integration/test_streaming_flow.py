# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for the streaming pipeline.

Mocks the WebSocket connection to simulate message bursts and verifies
the full pipeline: messages → validator → deduplicator → Parquet sink.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_config(tmp_path):
    """Config wired to tmp_path for isolated test output."""
    heartbeat_dir = tmp_path / "heartbeat"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "feed": "delayed",
            "rate_limiting": {"total_requests_per_minute": 60},
            "spy": {
                "ticker": "SPY",
                "multiplier": 1,
                "timespan": "second",
                "limit_per_request": 50000,
            },
        },
        "retry": {
            "polygon": {
                "rate_limit_wait_seconds": 12,
                "max_attempts": 3,
                "initial_wait_seconds": 0.01,
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
        },
        "streaming": {
            "batch_size": 5,
            "market_hours": {
                "timezone": "America/New_York",
                "active_days": [0, 1, 2, 3, 4],
                "start_time": "09:30",
                "end_time": "16:00",
            },
            "monitoring": {
                "heartbeat_interval_seconds": 1,  # Low for testing
                "alert_threshold_seconds": 900,
            },
        },
        "monitoring": {
            "performance": {
                "commit_latency_seconds": 300,
                "throughput_min_records_per_sec": 0,  # Disable throughput alert
                "memory_usage_mb_threshold": 10000,
                "error_rate_percent": 50.0,  # High to avoid alerts in tests
                "error_window_minutes": 15,
                "latency_window_size": 100,
                "metrics_dump_interval_minutes": 15,
            },
        },
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "data"),
                "compression": "snappy",
                "row_group_size": 10000,
            },
        },
    }


def _make_record(timestamp=1704067200000):
    """Create a valid equity record."""
    return {
        "timestamp": timestamp,
        "open": 450.10,
        "high": 450.50,
        "low": 449.90,
        "close": 450.30,
        "volume": 1500,
        "vwap": 450.20,
        "transactions": 25,
        "source": "spy",
    }


class TestStreamingPipeline:
    """Full pipeline integration: messages → validator → deduplicator → Parquet."""

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_full_pipeline_writes_parquet(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Streamed messages are written to Parquet files."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 15

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner

        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        # Generate 10 unique records
        records = [_make_record(1704067200000 + i * 1000) for i in range(10)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream

        stats = runner.run()

        assert stats["messages_received"] == 10
        assert stats["messages_written"] == 10
        assert stats["messages_invalid"] == 0

        # Verify Parquet files exist
        data_dir = tmp_path / "data" / "spy"
        parquet_files = list(data_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_heartbeat_file_created(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Heartbeat file is created during streaming."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 10

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner

        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(5)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream

        stats = runner.run()

        # Heartbeat file should exist (heartbeat interval set to 1s)
        heartbeat_path = Path("data/logs/heartbeat") / "streaming_spy_status.json"
        assert heartbeat_path.exists()

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_stats_accuracy(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Stats accurately reflect processed record counts."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 15

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner

        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        records = [
            _make_record(1704067200000),
            _make_record(1704067200000),      # duplicate
            _make_record(1704067201000),
            {                                 # invalid: missing close
                "timestamp": 1704067202000,
                "open": 450.10,
                "high": 450.50,
                "low": 449.90,
                "volume": 1500,
                "vwap": 450.20,
                "source": "spy",
            },
            _make_record(1704067203000),
        ]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream

        stats = runner.run()

        assert stats["messages_received"] == 5
        assert stats["messages_invalid"] == 1    # Missing close
        assert stats["messages_duplicates"] == 1  # Same timestamp
        assert stats["messages_written"] == 3     # 5 - 1 invalid - 1 dup

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_market_close_triggers_shutdown(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Market close during streaming triggers graceful shutdown."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            # Open for initial check + first 2 records, then closed
            return call_count["n"] <= 3

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner

        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100  # Large batch to avoid mid-stream flush
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(100)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream

        stats = runner.run()

        # Should have stopped well before processing all 100 records
        assert stats["messages_received"] < 100
        assert stats["status"] == "completed"
        # Buffer should have been flushed
        assert stats["batches_flushed"] >= 1
