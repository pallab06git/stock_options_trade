# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for FeedSimulator."""

import threading
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.orchestrator.simulator import FeedSimulator, _SOURCE_DIRS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**sim_overrides):
    """Build a minimal config dict with optional simulator overrides."""
    config = {}
    if sim_overrides:
        config["simulator"] = sim_overrides
    return config


def _write_parquet(path, records):
    """Write a list of record dicts to a Parquet file."""
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), engine="pyarrow", index=False)


def _make_records(n=5, start_ts=1000000, gap_ms=1000):
    """Generate n records with evenly spaced timestamps."""
    return [
        {
            "timestamp": start_ts + i * gap_ms,
            "open": 100.0 + i,
            "close": 100.5 + i,
            "high": 101.0 + i,
            "low": 99.5 + i,
            "volume": 1000 + i * 10,
            "source": "spy",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for __init__ configuration."""

    def test_defaults(self, tmp_path):
        """Default speed is 1.0."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10")
        assert sim.speed == 1.0
        assert sim.source == "spy"
        assert sim.date == "2026-02-10"

    def test_custom_speed(self, tmp_path):
        """CLI speed parameter is used."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10", speed=10.0)
        assert sim.speed == 10.0

    def test_config_speed_override(self, tmp_path):
        """Config speed_multiplier is used when CLI speed is default."""
        config = _make_config(speed_multiplier=5.0)
        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        assert sim.speed == 5.0

    def test_cli_speed_overrides_config(self, tmp_path):
        """Explicit CLI speed takes precedence over config."""
        config = _make_config(speed_multiplier=5.0)
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=20.0)
        assert sim.speed == 20.0

    def test_negative_speed_clamped(self):
        """Negative speed is clamped to 0."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10", speed=-5.0)
        assert sim.speed == 0.0

    def test_source_lowercased(self):
        """Source name is lowercased."""
        sim = FeedSimulator({}, source="SPY", date="2026-02-10")
        assert sim.source == "spy"

    def test_custom_data_dir(self, tmp_path):
        """Custom data_dir from config is used."""
        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        assert sim._parquet_path == tmp_path / "2026-02-10.parquet"


# ---------------------------------------------------------------------------
# Test: load_records
# ---------------------------------------------------------------------------

class TestLoadRecords:
    """Tests for load_records()."""

    def test_load_sorted_by_timestamp(self, tmp_path):
        """Records are sorted by timestamp ascending."""
        records = [
            {"timestamp": 3000, "value": "c"},
            {"timestamp": 1000, "value": "a"},
            {"timestamp": 2000, "value": "b"},
        ]
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        loaded = sim.load_records()

        assert len(loaded) == 3
        assert loaded[0]["timestamp"] == 1000
        assert loaded[1]["timestamp"] == 2000
        assert loaded[2]["timestamp"] == 3000

    def test_load_updates_stats(self, tmp_path):
        """records_loaded stat is updated."""
        records = _make_records(10)
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        sim.load_records()

        assert sim.get_stats()["records_loaded"] == 10

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing Parquet file."""
        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="9999-01-01")

        with pytest.raises(FileNotFoundError, match="No Parquet file"):
            sim.load_records()

    def test_load_without_timestamp_column(self, tmp_path):
        """Records without timestamp column are still loaded."""
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="news", date="2026-02-10")
        loaded = sim.load_records()

        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Test: stream_realtime
# ---------------------------------------------------------------------------

class TestStreamRealtime:
    """Tests for stream_realtime() — the streaming generator."""

    def test_yields_all_records_no_delay(self, tmp_path):
        """With speed=0, all records are yielded without delay."""
        records = _make_records(5)
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=0)

        emitted = list(sim.stream_realtime())

        assert len(emitted) == 5
        assert sim.get_stats()["records_emitted"] == 5
        assert sim.get_stats()["total_delay_seconds"] == 0.0

    def test_delays_proportional_to_speed(self, tmp_path):
        """Records are delayed based on timestamp gaps and speed."""
        # 3 records, 1000ms apart = 1s gaps
        records = _make_records(3, gap_ms=1000)
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        # speed=1000 means 1s gap / 1000 = 0.001s actual delay
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=1000)

        start = time.monotonic()
        emitted = list(sim.stream_realtime())
        elapsed = time.monotonic() - start

        assert len(emitted) == 3
        # 2 gaps of ~0.001s each — should be very fast
        assert elapsed < 1.0

    def test_stop_event_interrupts(self, tmp_path):
        """Setting stop_event stops the generator mid-stream."""
        records = _make_records(100, gap_ms=100)
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=0)

        stop = threading.Event()
        emitted = []
        for record in sim.stream_realtime(stop_event=stop):
            emitted.append(record)
            if len(emitted) >= 10:
                stop.set()

        # Should have stopped after ~10 records (not all 100)
        assert len(emitted) <= 11

    def test_empty_file(self, tmp_path):
        """Empty Parquet file yields no records."""
        records = []
        df = pd.DataFrame({"timestamp": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64")})
        pq_file = tmp_path / "2026-02-10.parquet"
        pq_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(pq_file), engine="pyarrow", index=False)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=0)

        emitted = list(sim.stream_realtime())
        assert emitted == []

    def test_records_without_timestamp_no_delay(self, tmp_path):
        """Records without timestamp field yield without delay."""
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="news", date="2026-02-10", speed=1.0)

        emitted = list(sim.stream_realtime())
        assert len(emitted) == 3
        # No timestamp means no delay even at speed=1.0
        assert sim.get_stats()["total_delay_seconds"] == 0.0

    def test_delay_capped_at_5_seconds(self, tmp_path):
        """Large timestamp gaps are capped at 5s per record."""
        # 2 records with 1-hour gap
        records = [
            {"timestamp": 1000000, "value": 1},
            {"timestamp": 1000000 + 3600000, "value": 2},  # +1 hour
        ]
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        # speed=1.0 means 3600s gap, but capped to 5s
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=1.0)

        start = time.monotonic()
        emitted = list(sim.stream_realtime())
        elapsed = time.monotonic() - start

        assert len(emitted) == 2
        # Should not take 3600s — capped at 5s
        assert elapsed < 6.0
        assert sim.get_stats()["total_delay_seconds"] == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: connect / disconnect stubs
# ---------------------------------------------------------------------------

class TestBaseSourceStubs:
    """Tests for BaseSource compatibility stubs."""

    def test_connect_noop(self):
        """connect() is a no-op."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10")
        sim.connect()  # Should not raise

    def test_disconnect_noop(self):
        """disconnect() is a no-op."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10")
        sim.disconnect()  # Should not raise


# ---------------------------------------------------------------------------
# Test: get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for get_stats()."""

    def test_initial_stats(self):
        """Stats are zeroed on init."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10", speed=5.0)
        stats = sim.get_stats()

        assert stats["records_loaded"] == 0
        assert stats["records_emitted"] == 0
        assert stats["total_delay_seconds"] == 0.0
        assert stats["source"] == "spy"
        assert stats["date"] == "2026-02-10"
        assert stats["speed"] == 5.0

    def test_stats_after_replay(self, tmp_path):
        """Stats reflect loaded and emitted counts after replay."""
        records = _make_records(7)
        pq_file = tmp_path / "2026-02-10.parquet"
        _write_parquet(pq_file, records)

        config = _make_config(data_dir=str(tmp_path))
        sim = FeedSimulator(config, source="spy", date="2026-02-10", speed=0)

        list(sim.stream_realtime())
        stats = sim.get_stats()

        assert stats["records_loaded"] == 7
        assert stats["records_emitted"] == 7

    def test_stats_returns_copy(self):
        """get_stats returns a copy, not the internal dict."""
        sim = FeedSimulator({}, source="spy", date="2026-02-10")
        stats = sim.get_stats()
        stats["records_loaded"] = 999

        assert sim.get_stats()["records_loaded"] == 0


# ---------------------------------------------------------------------------
# Test: Source directory mapping
# ---------------------------------------------------------------------------

class TestSourceDirs:
    """Tests for source → directory mapping."""

    def test_known_sources_mapped(self):
        """All known sources have directory mappings."""
        assert "spy" in _SOURCE_DIRS
        assert "vix" in _SOURCE_DIRS
        assert "options" in _SOURCE_DIRS
        assert "news" in _SOURCE_DIRS
        assert "consolidated" in _SOURCE_DIRS

    def test_unknown_source_fallback(self):
        """Unknown source falls back to data/raw/{source}."""
        sim = FeedSimulator({}, source="custom_feed", date="2026-02-10")
        assert "data/raw/custom_feed" in str(sim._parquet_path)
