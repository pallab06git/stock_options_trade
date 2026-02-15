# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for the real-time streaming pipeline.

Uses FeedSimulator as a drop-in data source (no live API dependency),
wired into StreamingRunner to verify the full streaming pipeline:
simulator → buffer → validator → deduplicator → Parquet sink.

Also tests the simulator standalone for multi-source replay.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.orchestrator.simulator import FeedSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_spy_parquet(path, n=50):
    """Write synthetic SPY Parquet data for simulation."""
    records = []
    base_ts = 1707480000000  # ~2024-02-09 12:00 UTC
    for i in range(n):
        records.append({
            "timestamp": base_ts + i * 1000,
            "open": 450.0 + i * 0.01,
            "high": 450.5 + i * 0.01,
            "low": 449.5 + i * 0.01,
            "close": 450.2 + i * 0.01,
            "volume": 1000 + i,
            "vwap": 450.1 + i * 0.01,
            "transactions": 50,
            "source": "spy",
        })
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), engine="pyarrow", index=False)
    return records


def _write_vix_parquet(path, n=30):
    """Write synthetic VIX Parquet data for simulation."""
    records = []
    base_ts = 1707480000000
    for i in range(n):
        records.append({
            "timestamp": base_ts + i * 2000,
            "open": 18.0 + i * 0.01,
            "high": 18.5 + i * 0.01,
            "low": 17.5 + i * 0.01,
            "close": 18.2 + i * 0.01,
            "volume": 0,
            "vwap": 18.1,
            "transactions": 0,
            "source": "vix",
        })
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), engine="pyarrow", index=False)
    return records


def _write_news_parquet(path, n=10):
    """Write synthetic news Parquet data for simulation."""
    records = []
    base_ts = 1707480000000
    for i in range(n):
        records.append({
            "timestamp": base_ts + i * 60000,
            "article_id": f"art_{i}",
            "title": f"Market Update {i}",
            "description": f"Description {i}",
            "author": "Reporter",
            "article_url": "https://example.com",
            "tickers": ["SPY"],
            "keywords": ["market"],
            "sentiment": 0.65 + i * 0.01,
            "sentiment_reasoning": "Positive outlook",
            "publisher_name": "News Corp",
            "source": "news",
        })
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), engine="pyarrow", index=False)
    return records


def _make_streaming_config(tmp_path):
    """Config for streaming runner with simulator."""
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "feed": "delayed",
            "rate_limiting": {"total_requests_per_minute": 60},
            "spy": {
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
            "batch_size": 10,
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
                "throughput_min_records_per_sec": 0,
                "memory_usage_mb_threshold": 10000,
                "error_rate_percent": 50.0,
                "error_window_minutes": 15,
                "latency_window_size": 100,
                "metrics_dump_interval_minutes": 15,
            },
        },
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "output"),
                "compression": "snappy",
                "row_group_size": 10000,
            },
        },
        "simulator": {
            "speed_multiplier": 0,  # No delay for tests
        },
    }


# ---------------------------------------------------------------------------
# Tests: Simulator standalone
# ---------------------------------------------------------------------------

class TestSimulatorStandalone:
    """FeedSimulator replay without StreamingRunner."""

    def test_spy_replay_all_records(self, tmp_path):
        """Simulator replays all SPY records."""
        spy_dir = tmp_path / "spy"
        _write_spy_parquet(spy_dir / "2026-02-10.parquet", n=50)

        config = {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}}
        sim = FeedSimulator(config, source="spy", date="2026-02-10")

        emitted = list(sim.stream_realtime())

        assert len(emitted) == 50
        stats = sim.get_stats()
        assert stats["records_loaded"] == 50
        assert stats["records_emitted"] == 50

    def test_vix_replay(self, tmp_path):
        """Simulator replays VIX records."""
        vix_dir = tmp_path / "vix"
        _write_vix_parquet(vix_dir / "2026-02-10.parquet", n=30)

        config = {"simulator": {"data_dir": str(vix_dir), "speed_multiplier": 0}}
        sim = FeedSimulator(config, source="vix", date="2026-02-10")

        emitted = list(sim.stream_realtime())
        assert len(emitted) == 30
        assert all(r["source"] == "vix" for r in emitted)

    def test_news_replay(self, tmp_path):
        """Simulator replays news records."""
        news_dir = tmp_path / "news"
        _write_news_parquet(news_dir / "2026-02-10.parquet", n=10)

        config = {"simulator": {"data_dir": str(news_dir), "speed_multiplier": 0}}
        sim = FeedSimulator(config, source="news", date="2026-02-10")

        emitted = list(sim.stream_realtime())
        assert len(emitted) == 10
        assert all("article_id" in r for r in emitted)

    def test_stop_event_halts_replay(self, tmp_path):
        """stop_event interrupts replay mid-stream."""
        spy_dir = tmp_path / "spy"
        _write_spy_parquet(spy_dir / "2026-02-10.parquet", n=100)

        config = {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}}
        sim = FeedSimulator(config, source="spy", date="2026-02-10")

        stop = threading.Event()
        emitted = []
        for record in sim.stream_realtime(stop_event=stop):
            emitted.append(record)
            if len(emitted) >= 20:
                stop.set()

        assert 20 <= len(emitted) <= 21
        assert sim.get_stats()["records_emitted"] < 100

    def test_records_in_timestamp_order(self, tmp_path):
        """Emitted records are in ascending timestamp order."""
        spy_dir = tmp_path / "spy"
        _write_spy_parquet(spy_dir / "2026-02-10.parquet", n=50)

        config = {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}}
        sim = FeedSimulator(config, source="spy", date="2026-02-10")

        emitted = list(sim.stream_realtime())
        timestamps = [r["timestamp"] for r in emitted]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Tests: Simulator → StreamingRunner pipeline
# ---------------------------------------------------------------------------

class TestSimulatorWithStreamingRunner:
    """FeedSimulator injected into StreamingRunner as the data source."""

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_full_pipeline_writes_parquet(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Simulator → StreamingRunner → Parquet output."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 60

        mock_mh.is_market_open.side_effect = market_open_side_effect

        # Write source data
        spy_dir = tmp_path / "spy"
        _write_spy_parquet(spy_dir / "2026-02-10.parquet", n=25)

        config = _make_streaming_config(tmp_path)
        sim = FeedSimulator(
            {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}},
            source="spy",
            date="2026-02-10",
        )

        from src.orchestrator.streaming_runner import StreamingRunner

        runner = StreamingRunner(config, ticker="SPY")
        runner.client = sim  # Inject simulator

        stats = runner.run()

        assert stats["messages_received"] == 25
        assert stats["messages_written"] == 25
        assert stats["messages_invalid"] == 0
        assert stats["status"] == "completed"

        # Verify output files
        output_dir = tmp_path / "output" / "spy"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 25

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_pipeline_handles_duplicates(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Duplicate timestamps in simulated feed are deduplicated."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 30

        mock_mh.is_market_open.side_effect = market_open_side_effect

        # Write data with duplicate timestamps
        records = []
        base_ts = 1707480000000
        for i in range(10):
            rec = {
                "timestamp": base_ts + (i // 2) * 1000,  # 5 unique timestamps
                "open": 450.0, "high": 450.5, "low": 449.5,
                "close": 450.2, "volume": 1000, "vwap": 450.1,
                "transactions": 50, "source": "spy",
            }
            records.append(rec)
        df = pd.DataFrame(records)
        spy_dir = tmp_path / "spy"
        spy_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(spy_dir / "2026-02-10.parquet"), engine="pyarrow", index=False)

        config = _make_streaming_config(tmp_path)
        sim = FeedSimulator(
            {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}},
            source="spy",
            date="2026-02-10",
        )

        from src.orchestrator.streaming_runner import StreamingRunner

        runner = StreamingRunner(config, ticker="SPY")
        runner.client = sim

        stats = runner.run()

        assert stats["messages_received"] == 10
        assert stats["messages_duplicates"] == 5
        assert stats["messages_written"] == 5

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_pipeline_filters_invalid(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Invalid records in simulated feed are filtered by validator."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 30

        mock_mh.is_market_open.side_effect = market_open_side_effect

        # Write mix of valid and invalid records
        records = []
        base_ts = 1707480000000
        for i in range(5):
            records.append({
                "timestamp": base_ts + i * 1000,
                "open": 450.0, "high": 450.5, "low": 449.5,
                "close": 450.2, "volume": 1000, "vwap": 450.1,
                "transactions": 50, "source": "spy",
            })
        for i in range(3):
            records.append({
                "timestamp": base_ts + (5 + i) * 1000,
                "open": -1.0,  # Invalid
                "high": 450.5, "low": 449.5,
                "close": 450.2, "volume": 1000, "vwap": 450.1,
                "transactions": 50, "source": "spy",
            })
        df = pd.DataFrame(records)
        spy_dir = tmp_path / "spy"
        spy_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(spy_dir / "2026-02-10.parquet"), engine="pyarrow", index=False)

        config = _make_streaming_config(tmp_path)
        sim = FeedSimulator(
            {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}},
            source="spy",
            date="2026-02-10",
        )

        from src.orchestrator.streaming_runner import StreamingRunner

        runner = StreamingRunner(config, ticker="SPY")
        runner.client = sim

        stats = runner.run()

        assert stats["messages_received"] == 8
        assert stats["messages_invalid"] == 3
        assert stats["messages_written"] == 5

    @patch("src.orchestrator.streaming_runner.MarketHours")
    @patch("src.orchestrator.streaming_runner.ConnectionManager")
    def test_batch_flushing(self, mock_cm_cls, mock_mh_cls, tmp_path):
        """Records are flushed in batches of configured batch_size."""
        mock_mh = mock_mh_cls.return_value
        call_count = {"n": 0}

        def market_open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            return call_count["n"] <= 60

        mock_mh.is_market_open.side_effect = market_open_side_effect

        spy_dir = tmp_path / "spy"
        _write_spy_parquet(spy_dir / "2026-02-10.parquet", n=25)

        config = _make_streaming_config(tmp_path)
        config["streaming"]["batch_size"] = 10  # Flush every 10

        sim = FeedSimulator(
            {"simulator": {"data_dir": str(spy_dir), "speed_multiplier": 0}},
            source="spy",
            date="2026-02-10",
        )

        from src.orchestrator.streaming_runner import StreamingRunner

        runner = StreamingRunner(config, ticker="SPY")
        runner.client = sim

        stats = runner.run()

        # 25 records / 10 batch_size = 2 full batches + 1 remainder
        assert stats["batches_flushed"] == 3
        assert stats["messages_written"] == 25
