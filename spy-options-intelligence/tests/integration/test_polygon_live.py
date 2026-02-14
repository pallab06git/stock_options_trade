# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for PolygonSPYClient against live Polygon API.

Requires a valid POLYGON_API_KEY in .env file.
These tests make real API calls — run sparingly to avoid rate limits.
Data is fetched ONCE per module and shared across all tests to stay
within the free-tier 5 req/min budget.
"""

import os
import pytest
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.utils.connection_manager import ConnectionManager
from src.data_sources.polygon_client import PolygonSPYClient
from src.data_sources.base_source import ExecutionMode


# Load .env before tests
load_dotenv()


# Use a known recent trading day (Monday Feb 9 2026)
TEST_DATE = "2026-02-09"


@pytest.fixture(scope="module")
def live_config():
    """Load real config with API key from .env."""
    loader = ConfigLoader(config_dir="config", env_file=".env")
    return loader.load()


@pytest.fixture(scope="module")
def live_client(live_config):
    """Create a real PolygonSPYClient with live API connection."""
    cm = ConnectionManager(live_config)
    client = PolygonSPYClient(live_config, cm)
    client.connect()
    yield client
    cm.close()


@pytest.fixture(scope="module")
def fetched_records(live_client):
    """Fetch data ONCE and share across all tests to avoid 429 rate limits."""
    records = list(live_client.fetch_historical(TEST_DATE, TEST_DATE))
    return records


class TestPolygonLiveConnection:
    """Tests that validate live Polygon API data (single fetch, shared data)."""

    def test_connect_succeeds(self, live_client):
        """Client connects and sets mode to HISTORICAL."""
        assert live_client.mode == ExecutionMode.HISTORICAL

    def test_fetch_single_date_returns_data(self, fetched_records):
        """Fetch a real trading day — should return records."""
        assert len(fetched_records) > 0, f"No records returned for {TEST_DATE}"
        print(f"\n  Records fetched for {TEST_DATE}: {len(fetched_records)}")

    def test_records_have_required_fields(self, fetched_records):
        """Every record has all expected fields."""
        assert len(fetched_records) > 0

        required = {"timestamp", "open", "high", "low", "close", "volume", "vwap", "source"}
        for record in fetched_records[:5]:
            for field in required:
                assert field in record, f"Missing field: {field}"
                assert record[field] is not None, f"None value for: {field}"

    def test_records_validate(self, live_client, fetched_records):
        """All fetched records pass validation."""
        assert len(fetched_records) > 0

        invalid_count = 0
        for record in fetched_records:
            if not live_client.validate_record(record):
                invalid_count += 1

        print(f"\n  Valid: {len(fetched_records) - invalid_count}/{len(fetched_records)}")
        assert invalid_count == 0, f"{invalid_count} records failed validation"

    def test_records_source_is_spy(self, fetched_records):
        """All records have source='spy'."""
        assert all(r["source"] == "spy" for r in fetched_records)

    def test_timestamps_are_ascending(self, fetched_records):
        """Records are sorted by timestamp ascending."""
        timestamps = [r["timestamp"] for r in fetched_records]
        assert timestamps == sorted(timestamps), "Timestamps not in ascending order"

    def test_prices_are_reasonable(self, fetched_records):
        """SPY prices should be in a reasonable range (100-1000)."""
        assert len(fetched_records) > 0

        for record in fetched_records[:10]:
            assert 100 < record["open"] < 1000, f"Unreasonable open: {record['open']}"
            assert 100 < record["close"] < 1000, f"Unreasonable close: {record['close']}"
            assert record["high"] >= record["low"], "High < Low"
