# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for the historical ingestion pipeline.

Tests the full batch pipeline wiring: mock data source → Validator →
Deduplicator → ParquetSink, including multi-day execution, checkpoint/
resume, and multi-source (SPY, VIX, news) pipelines.

No live API calls — uses mock clients that yield synthetic records.
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd
import pytest

from src.orchestrator.historical_runner import HistoricalRunner
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator
from src.sinks.parquet_sink import ParquetSink


# ---------------------------------------------------------------------------
# Fake data source clients
# ---------------------------------------------------------------------------

class FakeEquityClient:
    """Mock equity client that yields synthetic per-second bars."""

    def __init__(self, records_per_day=100):
        self.records_per_day = records_per_day
        self._connected = False

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator[Dict[str, Any], None, None]:
        from datetime import datetime, timedelta

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            base_ts = int(current.timestamp()) * 1000
            for i in range(self.records_per_day):
                yield {
                    "timestamp": base_ts + i * 1000,
                    "open": 450.0 + i * 0.01,
                    "high": 450.5 + i * 0.01,
                    "low": 449.5 + i * 0.01,
                    "close": 450.2 + i * 0.01,
                    "volume": 1000 + i,
                    "vwap": 450.1 + i * 0.01,
                    "transactions": 50,
                    "source": "spy",
                }
            current += timedelta(days=1)


class FakeVIXClient:
    """Mock VIX client that yields synthetic VIX bars."""

    def __init__(self, records_per_day=50):
        self.records_per_day = records_per_day

    def connect(self):
        pass

    def disconnect(self):
        pass

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator[Dict[str, Any], None, None]:
        from datetime import datetime, timedelta

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            base_ts = int(current.timestamp()) * 1000
            for i in range(self.records_per_day):
                yield {
                    "timestamp": base_ts + i * 2000,
                    "open": 18.0 + i * 0.01,
                    "high": 18.5 + i * 0.01,
                    "low": 17.5 + i * 0.01,
                    "close": 18.2 + i * 0.01,
                    "volume": 0,
                    "vwap": 18.1,
                    "transactions": 0,
                    "source": "vix",
                }
            current += timedelta(days=1)


class FakeNewsClient:
    """Mock news client that yields synthetic news records."""

    def __init__(self, records_per_day=5):
        self.records_per_day = records_per_day

    def connect(self):
        pass

    def disconnect(self):
        pass

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator[Dict[str, Any], None, None]:
        from datetime import datetime, timedelta

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        day_num = 0
        while current <= end:
            base_ts = int(current.timestamp()) * 1000
            for i in range(self.records_per_day):
                yield {
                    "timestamp": base_ts + i * 60000,
                    "article_id": f"art_{day_num}_{i}",
                    "title": f"Market Update {day_num}-{i}",
                    "description": f"Description for day {day_num}",
                    "author": "Reporter",
                    "article_url": "https://example.com",
                    "tickers": ["SPY"],
                    "keywords": ["market"],
                    "sentiment": 0.65,
                    "sentiment_reasoning": "Positive",
                    "publisher_name": "News Corp",
                    "source": "news",
                }
            current += timedelta(days=1)
            day_num += 1


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

def _make_config(tmp_path, batch_size=500):
    """Config wired to tmp_path for isolated output."""
    return {
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "data"),
                "compression": "snappy",
                "row_group_size": 10000,
            }
        },
        "historical": {
            "backfill": {
                "start_date": "2026-02-10",
                "end_date": "2026-02-12",
                "batch_size": batch_size,
                "trading_days": 30,
            }
        },
        "logging": {
            "execution_log_path": str(tmp_path / "logs"),
        },
        "monitoring": {
            "performance": {
                "commit_latency_seconds": 300,
                "throughput_min_records_per_sec": 0,
                "memory_usage_mb_threshold": 10000,
            }
        },
    }


# ---------------------------------------------------------------------------
# Tests: SPY historical pipeline
# ---------------------------------------------------------------------------

class TestSPYHistoricalFlow:
    """Multi-day SPY backfill with mock client."""

    def test_multi_day_backfill(self, tmp_path):
        """3-day SPY backfill writes Parquet files for each date."""
        config = _make_config(tmp_path)
        client = FakeEquityClient(records_per_day=100)
        validator = RecordValidator.for_equity("SPY")

        runner = HistoricalRunner(
            config,
            ticker="SPY",
            client=client,
            validator=validator,
        )
        stats = runner.run()

        assert stats["dates_processed"] == 3
        assert stats["total_fetched"] == 300
        assert stats["total_written"] == 300
        assert stats["total_invalid"] == 0

        # Verify Parquet files exist for each date
        data_dir = tmp_path / "data" / "spy"
        parquet_files = sorted(data_dir.glob("*.parquet"))
        assert len(parquet_files) == 3

    def test_checkpoint_created(self, tmp_path):
        """Checkpoint file is created after each date."""
        config = _make_config(tmp_path)
        client = FakeEquityClient(records_per_day=10)
        validator = RecordValidator.for_equity("SPY")

        runner = HistoricalRunner(
            config,
            ticker="SPY",
            client=client,
            validator=validator,
        )
        runner.run()

        # Checkpoint file should exist
        log_dir = tmp_path / "logs"
        checkpoint_files = list(log_dir.glob("checkpoint_SPY_*.json"))
        assert len(checkpoint_files) == 1

        # Should contain all 3 dates
        data = json.loads(checkpoint_files[0].read_text())
        assert len(data["completed_dates"]) == 3

    def test_resume_skips_completed(self, tmp_path):
        """Resume run skips already-completed dates."""
        config = _make_config(tmp_path)
        client = FakeEquityClient(records_per_day=10)
        validator = RecordValidator.for_equity("SPY")

        # First run
        runner1 = HistoricalRunner(
            config,
            ticker="SPY",
            client=client,
            validator=validator,
        )
        stats1 = runner1.run()
        assert stats1["dates_processed"] == 3

        # Second run with resume
        runner2 = HistoricalRunner(
            config,
            ticker="SPY",
            client=client,
            validator=validator,
        )
        stats2 = runner2.run(resume=True)

        assert stats2["dates_skipped"] == 3
        assert stats2["dates_processed"] == 0

    def test_deduplication_within_batch(self, tmp_path):
        """Duplicate records within a batch are removed."""
        config = _make_config(tmp_path, batch_size=500)

        # Client that produces duplicates
        class DuplicatingClient:
            def connect(self): pass
            def disconnect(self): pass
            def fetch_historical(self, start_date, end_date, **kwargs):
                from datetime import datetime
                base_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
                for i in range(10):
                    record = {
                        "timestamp": base_ts + i * 1000,
                        "open": 450.0, "high": 450.5, "low": 449.5,
                        "close": 450.2, "volume": 1000, "vwap": 450.1,
                        "transactions": 50, "source": "spy",
                    }
                    yield record
                    yield record  # Duplicate

        client = DuplicatingClient()
        validator = RecordValidator.for_equity("SPY")

        # Only process 1 day for simplicity
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-10"

        runner = HistoricalRunner(
            config, ticker="SPY", client=client, validator=validator,
        )
        stats = runner.run()

        assert stats["total_fetched"] == 20
        assert stats["total_duplicates"] == 10
        assert stats["total_written"] == 10

    def test_invalid_records_filtered(self, tmp_path):
        """Invalid records are filtered out by the validator."""
        config = _make_config(tmp_path)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-10"

        class MixedClient:
            def connect(self): pass
            def disconnect(self): pass
            def fetch_historical(self, start_date, end_date, **kwargs):
                from datetime import datetime
                base_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
                # 5 valid records
                for i in range(5):
                    yield {
                        "timestamp": base_ts + i * 1000,
                        "open": 450.0, "high": 450.5, "low": 449.5,
                        "close": 450.2, "volume": 1000, "vwap": 450.1,
                        "transactions": 50, "source": "spy",
                    }
                # 3 invalid records (negative price)
                for i in range(3):
                    yield {
                        "timestamp": base_ts + (5 + i) * 1000,
                        "open": -1.0, "high": 450.5, "low": 449.5,
                        "close": 450.2, "volume": 1000, "vwap": 450.1,
                        "transactions": 50, "source": "spy",
                    }

        runner = HistoricalRunner(
            config, ticker="SPY", client=MixedClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        stats = runner.run()

        assert stats["total_fetched"] == 8
        assert stats["total_invalid"] == 3
        assert stats["total_written"] == 5


# ---------------------------------------------------------------------------
# Tests: VIX historical pipeline
# ---------------------------------------------------------------------------

class TestVIXHistoricalFlow:
    """VIX backfill with dependency-injected client and validator."""

    def test_vix_backfill(self, tmp_path):
        """VIX historical data is validated and written correctly."""
        config = _make_config(tmp_path)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-11"

        client = FakeVIXClient(records_per_day=50)
        validator = RecordValidator("vix")

        runner = HistoricalRunner(
            config,
            ticker="I:VIX",
            client=client,
            validator=validator,
        )
        stats = runner.run()

        assert stats["dates_processed"] == 2
        assert stats["total_fetched"] == 100
        assert stats["total_written"] == 100

        # Verify Parquet file content
        data_dir = tmp_path / "data" / "vix"
        parquet_files = list(data_dir.glob("*.parquet"))
        assert len(parquet_files) == 2

        df = pd.read_parquet(parquet_files[0])
        assert "close" in df.columns
        assert "source" in df.columns
        assert (df["source"] == "vix").all()


# ---------------------------------------------------------------------------
# Tests: News historical pipeline
# ---------------------------------------------------------------------------

class TestNewsHistoricalFlow:
    """News backfill with article_id deduplication."""

    def test_news_backfill(self, tmp_path):
        """News historical data uses article_id for deduplication."""
        config = _make_config(tmp_path)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-11"

        client = FakeNewsClient(records_per_day=5)
        validator = RecordValidator("news")
        deduplicator = Deduplicator(key_field="article_id")

        runner = HistoricalRunner(
            config,
            ticker="news",
            client=client,
            validator=validator,
            deduplicator=deduplicator,
        )
        stats = runner.run()

        assert stats["dates_processed"] == 2
        assert stats["total_fetched"] == 10
        assert stats["total_written"] == 10

        data_dir = tmp_path / "data" / "news"
        parquet_files = list(data_dir.glob("*.parquet"))
        assert len(parquet_files) == 2

        df = pd.read_parquet(parquet_files[0])
        assert "article_id" in df.columns
        assert "title" in df.columns
        assert "sentiment" in df.columns

    def test_news_dedup_within_batch(self, tmp_path):
        """Duplicate article_ids within a single batch are deduplicated."""
        config = _make_config(tmp_path, batch_size=100)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-10"

        class DuplicatingNewsClient:
            def connect(self): pass
            def disconnect(self): pass
            def fetch_historical(self, start_date, end_date, **kwargs):
                from datetime import datetime
                base_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
                for i in range(6):
                    yield {
                        "timestamp": base_ts + i * 1000,
                        "article_id": f"art_{i % 3}",  # Only 3 unique
                        "title": f"Title {i}",
                        "source": "news",
                    }

        runner = HistoricalRunner(
            config,
            ticker="news",
            client=DuplicatingNewsClient(),
            validator=RecordValidator("news"),
            deduplicator=Deduplicator(key_field="article_id"),
        )
        stats = runner.run()

        assert stats["total_fetched"] == 6
        # deduplicate_batch deduplicates within the batch (all 6 in one batch)
        assert stats["total_duplicates"] == 3
        assert stats["total_written"] == 3


# ---------------------------------------------------------------------------
# Tests: Parquet output verification
# ---------------------------------------------------------------------------

class TestParquetOutput:
    """Verify Parquet file structure and content."""

    def test_parquet_sorted_by_timestamp(self, tmp_path):
        """Written Parquet files have records sorted by timestamp."""
        config = _make_config(tmp_path)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-10"

        client = FakeEquityClient(records_per_day=50)
        runner = HistoricalRunner(
            config, ticker="SPY", client=client,
            validator=RecordValidator.for_equity("SPY"),
        )
        runner.run()

        data_dir = tmp_path / "data" / "spy"
        df = pd.read_parquet(list(data_dir.glob("*.parquet"))[0])

        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_parquet_no_duplicate_timestamps(self, tmp_path):
        """Written Parquet files have unique timestamps."""
        config = _make_config(tmp_path)
        config["historical"]["backfill"]["start_date"] = "2026-02-10"
        config["historical"]["backfill"]["end_date"] = "2026-02-10"

        client = FakeEquityClient(records_per_day=50)
        runner = HistoricalRunner(
            config, ticker="SPY", client=client,
            validator=RecordValidator.for_equity("SPY"),
        )
        runner.run()

        data_dir = tmp_path / "data" / "spy"
        df = pd.read_parquet(list(data_dir.glob("*.parquet"))[0])

        assert df["timestamp"].is_unique
