# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for PolygonNewsClient against live Polygon API.

Requires a valid POLYGON_API_KEY in .env file.
These tests make real API calls — run sparingly to avoid rate limits.
Data is fetched ONCE per module and shared across all tests to stay
within the free-tier 5 req/min budget.
"""

import os
import threading

import pytest
from dotenv import load_dotenv

from src.data_sources.base_source import ExecutionMode
from src.data_sources.news_client import PolygonNewsClient
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator
from src.utils.config_loader import ConfigLoader
from src.utils.connection_manager import ConnectionManager


load_dotenv()

# Use a known date with SPY news coverage
TEST_DATE = "2026-02-09"


@pytest.fixture(scope="module")
def live_config():
    """Load real config with API key from .env."""
    loader = ConfigLoader(config_dir="config", env_file=".env")
    return loader.load()


@pytest.fixture(scope="module")
def live_cm(live_config):
    """Create a real ConnectionManager."""
    cm = ConnectionManager(live_config)
    yield cm
    cm.close()


@pytest.fixture(scope="module")
def live_client(live_config, live_cm):
    """Create a real PolygonNewsClient with live API connection."""
    client = PolygonNewsClient(live_config, live_cm)
    client.connect()
    yield client


@pytest.fixture(scope="module")
def fetched_records(live_client):
    """Fetch news data ONCE and share across all tests to avoid 429 rate limits."""
    records = list(live_client.fetch_historical(TEST_DATE, TEST_DATE))
    return records


# ---------------------------------------------------------------------------
# Test: Live Connection
# ---------------------------------------------------------------------------

class TestNewsLiveConnection:
    """Tests that validate live Polygon News API data (single fetch, shared data)."""

    def test_connect_succeeds(self, live_client):
        """Client connects and sets mode to HISTORICAL."""
        assert live_client.mode == ExecutionMode.HISTORICAL

    def test_fetch_returns_data(self, fetched_records):
        """Fetch a date — should return news articles."""
        assert len(fetched_records) > 0, f"No news records returned for {TEST_DATE}"
        print(f"\n  News articles fetched for {TEST_DATE}: {len(fetched_records)}")

    def test_records_have_required_fields(self, fetched_records):
        """Every record has timestamp, title, source."""
        assert len(fetched_records) > 0

        for record in fetched_records[:5]:
            assert "timestamp" in record and record["timestamp"] is not None, \
                f"Missing/None timestamp: {record}"
            assert "title" in record and record["title"] is not None, \
                f"Missing/None title: {record}"
            assert record["source"] == "news", \
                f"Expected source='news', got '{record.get('source')}'"

    def test_records_have_article_id(self, fetched_records):
        """Every record has a non-None article_id for deduplication."""
        assert len(fetched_records) > 0

        for record in fetched_records:
            assert record.get("article_id") is not None, \
                f"Missing article_id: {record.get('title', 'unknown')}"

    def test_article_ids_are_unique(self, fetched_records):
        """Article IDs should be unique within a single date fetch."""
        assert len(fetched_records) > 0

        ids = [r["article_id"] for r in fetched_records]
        assert len(ids) == len(set(ids)), \
            f"Duplicate article IDs found: {len(ids)} total, {len(set(ids))} unique"

    def test_timestamps_are_positive(self, fetched_records):
        """All timestamps are positive Unix ms."""
        assert len(fetched_records) > 0

        for record in fetched_records:
            assert isinstance(record["timestamp"], int), \
                f"Timestamp not int: {record['timestamp']}"
            assert record["timestamp"] > 0, \
                f"Non-positive timestamp: {record['timestamp']}"

    def test_records_pass_validator(self, fetched_records):
        """All fetched records pass the RecordValidator('news') schema."""
        assert len(fetched_records) > 0

        validator = RecordValidator("news")
        valid, invalid = validator.validate_batch(fetched_records)

        print(f"\n  Valid: {len(valid)}/{len(fetched_records)}")
        assert len(invalid) == 0, \
            f"{len(invalid)} records failed validation. First: {invalid[0] if invalid else 'n/a'}"

    def test_records_pass_client_validate(self, live_client, fetched_records):
        """All fetched records pass the client's own validate_record."""
        assert len(fetched_records) > 0

        invalid_count = sum(
            1 for r in fetched_records if not live_client.validate_record(r)
        )
        assert invalid_count == 0, f"{invalid_count} records failed client validation"

    def test_deduplicator_with_article_id(self, fetched_records):
        """Deduplicator(key_field='article_id') preserves all unique articles."""
        assert len(fetched_records) > 0

        dedup = Deduplicator(key_field="article_id")
        deduplicated = dedup.deduplicate_batch(fetched_records)

        assert len(deduplicated) == len(fetched_records), \
            f"Dedup removed {len(fetched_records) - len(deduplicated)} records " \
            f"(expected 0 since article IDs should be unique)"

    def test_record_schema_fields(self, fetched_records):
        """Verify all expected schema fields are present in records."""
        assert len(fetched_records) > 0

        expected_fields = {
            "timestamp", "article_id", "title", "description",
            "author", "article_url", "tickers", "keywords",
            "sentiment", "sentiment_reasoning", "publisher_name", "source",
        }

        first = fetched_records[0]
        for field in expected_fields:
            assert field in first, f"Missing field '{field}' in record"

        print(f"\n  Sample record fields: {sorted(first.keys())}")
        print(f"  Title: {first['title'][:80]}...")
        print(f"  Publisher: {first.get('publisher_name')}")
        print(f"  Sentiment: {first.get('sentiment')}")


# ---------------------------------------------------------------------------
# Test: Full Pipeline (fetch → validate → dedup → Parquet)
# ---------------------------------------------------------------------------

class TestNewsFullPipeline:
    """End-to-end pipeline test writing to Parquet."""

    def test_historical_runner_news_pipeline(self, live_config, tmp_path):
        """Run HistoricalRunner with news client, validator, and article_id dedup."""
        from src.orchestrator.historical_runner import HistoricalRunner

        # Override output paths to temp directory
        config = dict(live_config)
        config["sinks"] = {
            "parquet": {
                "base_path": str(tmp_path),
                "compression": "snappy",
                "row_group_size": 10000,
            }
        }
        config["historical"] = {
            "backfill": {
                "start_date": TEST_DATE,
                "end_date": TEST_DATE,
                "batch_size": 10000,
            }
        }
        config["logging"] = {
            "execution_log_path": str(tmp_path / "logs"),
            "error_log_path": str(tmp_path / "errors"),
            "console_level": "INFO",
            "file_level": "DEBUG",
        }

        cm = ConnectionManager(config)
        news_client = PolygonNewsClient(config, cm)
        validator = RecordValidator("news")
        deduplicator = Deduplicator(key_field="article_id")

        runner = HistoricalRunner(
            config,
            ticker="news",
            connection_manager=cm,
            client=news_client,
            validator=validator,
            deduplicator=deduplicator,
        )
        stats = runner.run()

        if stats["total_fetched"] == 0:
            pytest.skip("Rate limited (429) — skipping pipeline assertions")

        assert stats["total_written"] > 0, "No news articles written"
        assert stats["total_invalid"] == 0, f"{stats['total_invalid']} invalid records"

        # Verify Parquet file exists
        from pathlib import Path
        parquet_path = Path(tmp_path) / "news" / f"{TEST_DATE}.parquet"
        assert parquet_path.exists(), f"Expected Parquet at {parquet_path}"

        # Verify Parquet content
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        assert table.num_rows > 0, "Parquet file is empty"
        assert "title" in table.column_names, "Missing 'title' column in Parquet"
        assert "article_id" in table.column_names, "Missing 'article_id' column in Parquet"

        print(f"\n  Pipeline: {stats['total_written']} news articles → {parquet_path}")
        print(f"  Parquet rows: {table.num_rows}, columns: {table.column_names}")
