# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for OptionsStreamingRunner."""

import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.sinks.parquet_sink import ParquetSink


# ---------------------------------------------------------------------------
# Config / record helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path=None):
    """Create a minimal config for OptionsStreamingRunner."""
    base_path = str(tmp_path) if tmp_path else "/tmp/test_options_streaming"
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "feed": "delayed",
            "rate_limiting": {"total_requests_per_minute": 60},
            "options": {
                "underlying_ticker": "SPY",
                "strike_range_pct": 0.01,
                "max_contracts": 100,
                "expiration_lookahead_days": 1,
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


def _make_record(
    timestamp=1704067200000,
    ticker="O:SPY260210C00430000",
    valid=True,
):
    """Create a test options record."""
    if valid:
        return {
            "timestamp": timestamp,
            "open": 5.10,
            "high": 5.20,
            "low": 5.00,
            "close": 5.15,
            "volume": 100,
            "vwap": 5.12,
            "transactions": 10,
            "ticker": ticker,
            "source": "options",
        }
    return {
        "timestamp": timestamp,
        "open": -1.0,  # Invalid: negative price
        "high": 5.20,
        "low": 5.00,
        "close": 5.15,
        "volume": 100,
        "ticker": ticker,
        "source": "options",
    }


# ---------------------------------------------------------------------------
# Tests: OptionsStreamingRunner
# ---------------------------------------------------------------------------


class TestOptionsStreamingRunnerMarketClosed:

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_market_closed_returns_early(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """When market is closed, run() returns immediately."""
        mock_mh_cls.return_value.is_market_open.return_value = False

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        runner = OptionsStreamingRunner(config, date="2026-02-09")

        stats = runner.run()

        assert stats["status"] == "market_closed"
        assert stats["messages_received"] == 0


class TestOptionsStreamingRunnerProcessing:

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_stream_processes_records(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Records flow through validate → deduplicate → write."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 6

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        runner = OptionsStreamingRunner(config, date="2026-02-09")

        records = [
            _make_record(1704067200000 + i * 1000, f"O:SPY260210C0043{i}000")
            for i in range(3)
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
        assert stats["messages_written"] == 3
        assert stats["messages_invalid"] == 0

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_buffer_flush_at_threshold(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Buffer is flushed when batch_size records accumulate."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 10

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 2
        runner = OptionsStreamingRunner(config, date="2026-02-09")

        records = [
            _make_record(1704067200000 + i * 1000, f"O:SPY260210C0043{i}000")
            for i in range(5)
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

        assert stats["messages_received"] == 5
        # 2 flushes at threshold (2+2) + 1 final flush (1) = 3
        assert stats["batches_flushed"] == 3

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_compound_dedup(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Two contracts with same timestamp are both kept; same contract+timestamp deduped."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 8

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = OptionsStreamingRunner(config, date="2026-02-09")

        ts = 1704067200000
        records = [
            _make_record(ts, "O:SPY260210C00430000"),  # Contract A, ts1
            _make_record(ts, "O:SPY260210P00429000"),  # Contract B, same ts — NOT a dup
            _make_record(ts, "O:SPY260210C00430000"),  # Contract A, same ts — IS a dup
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
        assert stats["messages_duplicates"] == 1  # Only contract A repeated
        assert stats["messages_written"] == 2     # Contract A + Contract B

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_invalid_records_tracked(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Invalid records are counted in stats."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 8

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = OptionsStreamingRunner(config, date="2026-02-09")

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

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_cleanup_on_error(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Sink and connection manager are cleaned up on error."""
        mock_mh = mock_mh_cls.return_value
        mock_mh.is_market_open.return_value = True

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        runner = OptionsStreamingRunner(config, date="2026-02-09")

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

    @patch("src.orchestrator.options_streaming_runner.MarketHours")
    @patch("src.orchestrator.options_streaming_runner.ConnectionManager")
    def test_stats_returned(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Run returns complete stats dict."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 5

        mock_mh.is_market_open.side_effect = market_open_side_effect

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner
        config = _make_config(tmp_path)
        config["streaming"]["batch_size"] = 100
        runner = OptionsStreamingRunner(config, date="2026-02-09")

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


# ---------------------------------------------------------------------------
# Tests: ParquetSink compound dedup
# ---------------------------------------------------------------------------


class TestParquetCompoundDedup:

    def test_parquet_compound_dedup(self, tmp_path):
        """ParquetSink with dedup_subset=["ticker", "timestamp"] keeps both contracts."""
        config = {
            "sinks": {
                "parquet": {
                    "base_path": str(tmp_path),
                    "compression": "snappy",
                    "row_group_size": 10000,
                },
            },
        }
        sink = ParquetSink(config, dedup_subset=["ticker", "timestamp"])
        sink.connect()

        ts = 1704067200000

        # First batch: contract A
        batch1 = [
            {
                "timestamp": ts,
                "open": 5.10,
                "high": 5.20,
                "low": 5.00,
                "close": 5.15,
                "volume": 100,
                "ticker": "O:SPY260210C00430000",
                "source": "options",
            },
        ]
        sink.write_batch(batch1, "2024-01-01")

        # Second batch: contract B with same timestamp
        batch2 = [
            {
                "timestamp": ts,
                "open": 3.10,
                "high": 3.20,
                "low": 3.00,
                "close": 3.15,
                "volume": 50,
                "ticker": "O:SPY260210P00429000",
                "source": "options",
            },
        ]
        sink.write_batch(batch2, "2024-01-01")

        # Both should be kept (different ticker)
        path = tmp_path / "options" / "2024-01-01.parquet"
        df = pd.read_parquet(path)
        assert len(df) == 2

        # Third batch: duplicate of contract A (same ticker + timestamp)
        batch3 = [
            {
                "timestamp": ts,
                "open": 5.50,
                "high": 5.60,
                "low": 5.40,
                "close": 5.55,
                "volume": 200,
                "ticker": "O:SPY260210C00430000",
                "source": "options",
            },
        ]
        sink.write_batch(batch3, "2024-01-01")

        # Should still be 2 (contract A replaced, contract B kept)
        df = pd.read_parquet(path)
        assert len(df) == 2
        # Verify contract A was updated (last write wins)
        contract_a = df[df["ticker"] == "O:SPY260210C00430000"].iloc[0]
        assert contract_a["open"] == 5.50

    def test_parquet_default_dedup_unchanged(self, tmp_path):
        """Default ParquetSink still deduplicates by timestamp only."""
        config = {
            "sinks": {
                "parquet": {
                    "base_path": str(tmp_path),
                    "compression": "snappy",
                    "row_group_size": 10000,
                },
            },
        }
        sink = ParquetSink(config)
        sink.connect()

        assert sink.dedup_subset == ["timestamp"]
