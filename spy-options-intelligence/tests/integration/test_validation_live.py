"""Integration tests for validator and deduplicator with live Polygon data.

Fetches real SPY data from Polygon, validates with RecordValidator,
deduplicates with Deduplicator, and writes to Parquet. Proves the
full fetch → validate → dedup → store pipeline.
"""

import pytest
import pandas as pd
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.utils.connection_manager import ConnectionManager
from src.data_sources.polygon_client import PolygonSPYClient
from src.processing.validator import RecordValidator
from src.processing.deduplicator import Deduplicator
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


class TestValidatorWithLiveData:
    """Validate real Polygon SPY records."""

    def test_all_records_pass_validation(self, live_records):
        """Every record from Polygon passes SPY schema validation."""
        v = RecordValidator("spy")
        valid, invalid = v.validate_batch(live_records)

        print(f"\n  Total: {len(live_records)}, Valid: {len(valid)}, Invalid: {len(invalid)}")
        assert len(invalid) == 0
        assert len(valid) == len(live_records)

    def test_no_validation_errors_on_live_records(self, live_records):
        """get_validation_errors returns empty for every live record."""
        v = RecordValidator("spy")
        for record in live_records[:20]:
            errors = v.get_validation_errors(record)
            assert errors == [], f"Errors on record ts={record['timestamp']}: {errors}"

    def test_vix_validator_rejects_spy_records(self, live_records):
        """SPY records should fail VIX validation (wrong source)."""
        v = RecordValidator("vix")
        valid, invalid = v.validate_batch(live_records)
        assert len(invalid) == len(live_records)
        assert len(valid) == 0


class TestDeduplicatorWithLiveData:
    """Deduplicate real Polygon SPY records."""

    def test_live_records_have_unique_timestamps(self, live_records):
        """Polygon should return records with unique timestamps."""
        d = Deduplicator()
        result = d.deduplicate_batch(live_records)

        print(f"\n  Before: {len(live_records)}, After dedup: {len(result)}")
        assert len(result) == len(live_records), "Unexpected duplicates in live data"

    def test_simulated_duplicates_removed(self, live_records):
        """Doubling the records and deduping should return original count."""
        d = Deduplicator()
        doubled = live_records + live_records
        result = d.deduplicate_batch(doubled)
        assert len(result) == len(live_records)


class TestFullPipelineLive:
    """End-to-end: Polygon → validate → dedup → Parquet."""

    def test_fetch_validate_dedup_store(self, live_config, live_records, tmp_path_factory):
        """Full pipeline produces a valid Parquet file."""
        # Validate
        v = RecordValidator("spy")
        valid, invalid = v.validate_batch(live_records)
        assert len(invalid) == 0

        # Deduplicate
        d = Deduplicator()
        deduped = d.deduplicate_batch(valid)
        assert len(deduped) == len(valid)

        # Store
        output_dir = tmp_path_factory.mktemp("pipeline_live")
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
        sink.write_batch(deduped, partition_key=TEST_DATE)

        # Verify
        path = output_dir / "spy" / f"{TEST_DATE}.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == len(live_records)
        print(f"\n  Pipeline complete: {len(df)} records validated, deduped, stored")
