# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for ParquetSink with live Polygon data.

Fetches real SPY data from Polygon API, writes to Parquet, reads back
and validates. Proves the full fetch → store → read pipeline works.
"""

import pytest
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.utils.connection_manager import ConnectionManager
from src.data_sources.polygon_client import PolygonSPYClient
from src.sinks.parquet_sink import ParquetSink


load_dotenv()

TEST_DATE = "2026-02-09"


@pytest.fixture(scope="module")
def live_config():
    loader = ConfigLoader(config_dir="config", env_file=".env")
    return loader.load()


@pytest.fixture(scope="module")
def live_records(live_config):
    """Fetch real SPY data once from Polygon."""
    cm = ConnectionManager(live_config)
    client = PolygonSPYClient(live_config, cm)
    client.connect()
    records = list(client.fetch_historical(TEST_DATE, TEST_DATE))
    cm.close()
    return records


@pytest.fixture(scope="module")
def parquet_output(live_config, live_records, tmp_path_factory):
    """Write fetched records to Parquet and return (sink, output_dir)."""
    output_dir = tmp_path_factory.mktemp("parquet_integration")
    config = dict(live_config)
    config["sinks"] = {
        "parquet": {
            "base_path": str(output_dir),
            "compression": "snappy",
            "row_group_size": 10000,
        }
    }
    sink = ParquetSink(config)
    sink.connect()
    sink.write_batch(live_records, partition_key=TEST_DATE)
    return sink, output_dir


class TestParquetLiveIntegration:
    """End-to-end: Polygon API → ParquetSink → read back and validate."""

    def test_fetched_records_not_empty(self, live_records):
        """Polygon returned real data."""
        assert len(live_records) > 0
        print(f"\n  Fetched {len(live_records)} records from Polygon")

    def test_parquet_file_created(self, parquet_output):
        """Parquet file exists at expected path."""
        _, output_dir = parquet_output
        path = output_dir / "spy" / f"{TEST_DATE}.parquet"
        assert path.exists()

    def test_parquet_row_count_matches(self, live_records, parquet_output):
        """Parquet file has same number of rows as fetched records."""
        _, output_dir = parquet_output
        df = pd.read_parquet(output_dir / "spy" / f"{TEST_DATE}.parquet")
        assert len(df) == len(live_records)
        print(f"\n  Parquet rows: {len(df)}")

    def test_parquet_fields_present(self, parquet_output):
        """All expected columns exist in the Parquet file."""
        _, output_dir = parquet_output
        df = pd.read_parquet(output_dir / "spy" / f"{TEST_DATE}.parquet")
        required = {"timestamp", "open", "high", "low", "close", "volume", "vwap", "source"}
        assert required.issubset(set(df.columns))

    def test_parquet_prices_match_source(self, live_records, parquet_output):
        """First 5 records in Parquet match original fetched data."""
        _, output_dir = parquet_output
        df = pd.read_parquet(output_dir / "spy" / f"{TEST_DATE}.parquet")
        for i in range(min(5, len(live_records))):
            assert df.iloc[i]["open"] == live_records[i]["open"]
            assert df.iloc[i]["close"] == live_records[i]["close"]
            assert df.iloc[i]["timestamp"] == live_records[i]["timestamp"]

    def test_parquet_compression_is_snappy(self, parquet_output):
        """File uses Snappy compression."""
        _, output_dir = parquet_output
        meta = pq.read_metadata(output_dir / "spy" / f"{TEST_DATE}.parquet")
        compression = meta.row_group(0).column(0).compression
        assert compression == "SNAPPY"

    def test_duplicate_detection_works(self, live_records, parquet_output):
        """check_duplicate returns True for an existing record."""
        sink, _ = parquet_output
        assert sink.check_duplicate(live_records[0]) is True
