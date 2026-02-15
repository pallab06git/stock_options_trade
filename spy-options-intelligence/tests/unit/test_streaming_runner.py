# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for StreamingRunner."""

import threading

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path=None):
    """Create a minimal config for StreamingRunner."""
    base_path = str(tmp_path) if tmp_path else "/tmp/test_streaming"
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
            "batch_size": 3,
            "market_hours": {
                "timezone": "America/New_York",
                "active_days": [0, 1, 2, 3, 4],
                "start_time": "09:30",
                "end_time": "16:00",
            },
            "monitoring": {
                "heartbeat_interval_seconds": 300,
                "alert_threshold_seconds": 900,
            },
        },
        "monitoring": {
            "performance": {
                "commit_latency_seconds": 300,
                "throughput_min_records_per_sec": 100,
                "memory_usage_mb_threshold": 1000,
                "error_rate_percent": 1.0,
                "error_window_minutes": 15,
                "latency_window_size": 100,
                "metrics_dump_interval_minutes": 15,
            },
        },
        "sinks": {
            "parquet": {
                "base_path": base_path,
                "compression": "snappy",
                "row_group_size": 10000,
            },
        },
    }


def _make_record(timestamp=1704067200000, valid=True):
    """Create a test record."""
    if valid:
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
    return {
        "timestamp": timestamp,
        "open": -1.0,  # Invalid: negative price
        "high": 450.50,
        "low": 449.90,
        "close": 450.30,
        "volume": 1500,
        "vwap": 450.20,
        "source": "spy",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingRunnerMarketClosed:

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_market_closed_returns_early(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """When market is closed, run() returns immediately with market_closed status."""
        mock_mh_cls.return_value.is_market_open.return_value = False

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        stats = runner.run()

        assert stats["status"] == "market_closed"
        assert stats["messages_received"] == 0


class TestStreamingRunnerProcessing:

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_stream_processes_records(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Records flow through validate → deduplicate → write."""
        # Market open on first call, then closed to stop
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 4  # Open for first 4 checks, then closed

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        # Mock the client to yield 3 records then stop
        records = [_make_record(1704067200000 + i * 1000) for i in range(3)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream

        # Mock sink
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        assert stats["messages_received"] == 3
        assert stats["messages_written"] == 3
        assert stats["messages_invalid"] == 0

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_buffer_flush_at_threshold(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Buffer is flushed when batch_size records accumulate."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 10

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 2  # Small batch for testing
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(5)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        assert stats["messages_received"] == 5
        # 2 flushes at threshold (2+2) + 1 final flush (1) = 3
        assert stats["batches_flushed"] == 3

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_final_buffer_flush_on_exit(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Remaining buffer is flushed when stream ends."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 5

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100  # Larger than record count
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(2)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        # Only one flush at end (buffer never hit threshold)
        assert stats["batches_flushed"] == 1
        assert stats["messages_written"] == 2

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_market_close_stops_stream(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Stream stops when market closes mid-session."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            # Open for first 2 calls (init + first record), closed on 3rd
            return call_count["n"] <= 2

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(10)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        # Should have stopped after first record (market closed check)
        assert stats["messages_received"] < 10

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_heartbeat_sent_on_interval(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Heartbeat is sent when interval elapses."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 5

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(3)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()

        mock_heartbeat = MagicMock()
        mock_heartbeat.should_send_heartbeat.return_value = True
        mock_heartbeat.check_stalled_stream.return_value = False
        runner.heartbeat = mock_heartbeat

        stats = runner.run()

        assert mock_heartbeat.send_heartbeat.call_count >= 1
        assert mock_heartbeat.record_message.call_count == 3

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_invalid_records_tracked(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Invalid records are counted in stats."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 8

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = StreamingRunner(config)

        records = [
            _make_record(1704067200000, valid=True),
            _make_record(1704067201000, valid=False),  # invalid
            _make_record(1704067202000, valid=True),
        ]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        assert stats["messages_received"] == 3
        assert stats["messages_invalid"] == 1
        assert stats["messages_written"] == 2

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_deduplication_in_flush(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Duplicate records are removed in flush."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 8

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = StreamingRunner(config)

        # Two records with same timestamp = duplicate
        records = [
            _make_record(1704067200000),
            _make_record(1704067200000),  # duplicate
            _make_record(1704067201000),
        ]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        assert stats["messages_received"] == 3
        assert stats["messages_duplicates"] == 1
        assert stats["messages_written"] == 2

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_cleanup_on_error(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Sink and connection manager are cleaned up on error."""
        mock_mh = mock_mh_cls.return_value
        mock_mh.is_market_open.return_value = True

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        runner = StreamingRunner(config)

        def error_stream(**kwargs):
            yield _make_record()
            raise RuntimeError("test error")

        runner.client.stream_realtime = error_stream
        mock_sink = MagicMock()
        runner.sink = mock_sink
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        mock_sink.disconnect.assert_called_once()
        runner.connection_manager.close.assert_called_once()

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_stats_returned(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Run returns complete stats dict."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 5

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.streaming_runner import StreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = StreamingRunner(config)

        records = [_make_record(1704067200000 + i * 1000) for i in range(2)]

        def fake_stream(**kwargs):
            for r in records:
                yield r

        runner.client.stream_realtime = fake_stream
        runner.sink = MagicMock()
        runner.heartbeat = MagicMock()
        runner.heartbeat.should_send_heartbeat.return_value = False
        runner.heartbeat.check_stalled_stream.return_value = False

        stats = runner.run()

        assert "status" in stats
        assert "messages_received" in stats
        assert "messages_written" in stats
        assert "messages_invalid" in stats
        assert "messages_duplicates" in stats
        assert "batches_flushed" in stats
        assert "start_time" in stats
        assert "end_time" in stats
        assert stats["status"] == "completed"
