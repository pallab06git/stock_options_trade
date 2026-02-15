# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Live integration tests for WebSocket streaming against Polygon.

Requires a valid POLYGON_API_KEY in .env file. Connects to the Polygon
delayed WebSocket feed and streams real SPY aggregate data.

These tests only produce data during/near US market hours (9:30 AM–4:15 PM ET,
accounting for the 15-minute delay on the free-tier feed). On weekends or
outside these hours the tests are skipped automatically.

Run explicitly:
    pytest tests/integration/test_streaming_live.py -v -s
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pytz
import pytest
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.utils.connection_manager import ConnectionManager
from src.data_sources.polygon_client import PolygonEquityClient
from src.data_sources.base_source import ExecutionMode
from src.processing.validator import RecordValidator
from src.processing.deduplicator import Deduplicator
from src.sinks.parquet_sink import ParquetSink


load_dotenv()

# Maximum seconds to wait for the first WebSocket message
STREAM_TIMEOUT = 60
# Maximum records to collect before stopping
MAX_RECORDS = 50


def _is_market_active():
    """Check if the delayed feed is likely to have data.

    The delayed feed lags ~15 minutes behind real-time, so data flows
    roughly from 9:45 AM to 4:15 PM ET on weekdays.
    """
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    # Weekdays only (Mon=0 .. Fri=4)
    if now.weekday() > 4:
        return False
    hour_min = now.hour * 60 + now.minute
    # 9:45 AM = 585, 4:15 PM = 975
    return 585 <= hour_min <= 975


# Skip all tests in this module if market is not active
pytestmark = pytest.mark.skipif(
    not _is_market_active(),
    reason="Polygon delayed feed only delivers data near US market hours (Mon-Fri ~9:45AM-4:15PM ET)",
)


@pytest.fixture(scope="module")
def live_config():
    """Load real config with API key from .env."""
    loader = ConfigLoader(config_dir="config", env_file=".env")
    return loader.load()


@pytest.fixture(scope="module")
def live_records(live_config):
    """Connect to Polygon WebSocket, collect records, and share across tests.

    Streams for up to STREAM_TIMEOUT seconds or until MAX_RECORDS are
    collected, whichever comes first.
    """
    cm = ConnectionManager(live_config)
    client = PolygonEquityClient(live_config, cm, ticker="SPY")

    stop_event = threading.Event()
    collected = []

    start = time.monotonic()
    try:
        for record in client.stream_realtime(stop_event=stop_event):
            collected.append(record)
            if len(collected) >= MAX_RECORDS:
                break
            if time.monotonic() - start > STREAM_TIMEOUT:
                break
    finally:
        stop_event.set()
        cm.close()

    elapsed = time.monotonic() - start
    print(f"\n  WebSocket: collected {len(collected)} records in {elapsed:.1f}s")
    return collected


class TestWebSocketLiveConnection:
    """Tests that validate live Polygon WebSocket streaming."""

    def test_receives_records(self, live_records):
        """WebSocket connection receives at least one record."""
        assert len(live_records) > 0, (
            f"No records received within {STREAM_TIMEOUT}s. "
            "Polygon delayed feed may be inactive."
        )

    def test_records_have_required_fields(self, live_records):
        """Every streamed record has all expected fields."""
        if not live_records:
            pytest.skip("No records received")

        required = {"timestamp", "open", "high", "low", "close", "volume", "vwap", "source"}
        for record in live_records[:10]:
            for field in required:
                assert field in record, f"Missing field: {field}"

    def test_records_have_spy_source(self, live_records):
        """All records have source='spy'."""
        if not live_records:
            pytest.skip("No records received")
        assert all(r["source"] == "spy" for r in live_records)

    def test_records_pass_validation(self, live_records):
        """All streamed records pass the equity validator."""
        if not live_records:
            pytest.skip("No records received")

        validator = RecordValidator.for_equity("SPY")
        valid, invalid = validator.validate_batch(live_records)

        print(f"\n  Validation: {len(valid)} valid, {len(invalid)} invalid out of {len(live_records)}")
        assert len(invalid) == 0, f"{len(invalid)} records failed validation"

    def test_prices_are_reasonable(self, live_records):
        """SPY prices should be in a reasonable range."""
        if not live_records:
            pytest.skip("No records received")

        for record in live_records[:10]:
            assert 100 < record["open"] < 1000, f"Unreasonable open: {record['open']}"
            assert 100 < record["close"] < 1000, f"Unreasonable close: {record['close']}"
            assert record["high"] >= record["low"], "High < Low"

    def test_timestamps_are_positive(self, live_records):
        """All timestamps are positive Unix milliseconds."""
        if not live_records:
            pytest.skip("No records received")

        for record in live_records:
            assert record["timestamp"] > 0, f"Non-positive timestamp: {record['timestamp']}"
            # Should be a recent epoch ms (after 2020-01-01)
            assert record["timestamp"] > 1577836800000, f"Timestamp too old: {record['timestamp']}"


class TestWebSocketLivePipeline:
    """End-to-end: WebSocket → validate → deduplicate → Parquet."""

    def test_full_pipeline_to_parquet(self, live_config, live_records, tmp_path):
        """Streamed records flow through the full pipeline to Parquet files."""
        if not live_records:
            pytest.skip("No records received")

        # Pipeline components
        validator = RecordValidator.for_equity("SPY")
        deduplicator = Deduplicator(key_field="timestamp")

        config_with_tmp = dict(live_config)
        config_with_tmp["sinks"] = {
            "parquet": {
                "base_path": str(tmp_path),
                "compression": "snappy",
                "row_group_size": 10000,
            },
        }
        sink = ParquetSink(config_with_tmp)
        sink.connect()

        # Process
        valid, invalid = validator.validate_batch(live_records)
        deduplicated = deduplicator.deduplicate_batch(valid)

        # Partition by date
        partitions = {}
        for record in deduplicated:
            ts = record.get("timestamp", 0)
            partition_key = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            partitions.setdefault(partition_key, []).append(record)

        for partition_key, records in partitions.items():
            sink.write_batch(records, partition_key)

        sink.disconnect()

        # Verify Parquet files
        parquet_files = list(tmp_path.glob("spy/*.parquet"))
        assert len(parquet_files) >= 1, "No Parquet files written"

        total_written = sum(len(recs) for recs in partitions.values())
        print(
            f"\n  Pipeline: {len(live_records)} received → "
            f"{len(valid)} valid → {len(deduplicated)} deduped → "
            f"{total_written} written to {len(parquet_files)} file(s)"
        )
