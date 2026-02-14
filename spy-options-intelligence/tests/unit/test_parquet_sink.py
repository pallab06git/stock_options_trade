"""Unit tests for ParquetSink."""

import pytest
import pandas as pd
import pyarrow.parquet as pq

from src.sinks.parquet_sink import ParquetSink
from src.sinks.base_sink import SinkType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path):
    """Create a minimal config pointing base_path to tmp_path."""
    return {
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "data"),
                "compression": "snappy",
                "row_group_size": 10000,
            }
        }
    }


def _make_record(timestamp=1704067200000, open_=450.10, close=450.30,
                 high=450.50, low=449.90, volume=1500, vwap=450.20,
                 transactions=25, source="spy"):
    return {
        "timestamp": timestamp,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        "transactions": transactions,
        "source": source,
    }


# Timestamps for 2024-01-01 (UTC)
TS_JAN1_A = 1704067200000  # 2024-01-01 00:00:00 UTC
TS_JAN1_B = 1704067260000  # 2024-01-01 00:01:00 UTC
# Timestamps for 2024-01-02 (UTC)
TS_JAN2_A = 1704153600000  # 2024-01-02 00:00:00 UTC


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnect:

    def test_connect_creates_base_dir(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        base = tmp_path / "data"

        assert not base.exists()
        sink.connect()
        assert base.exists()

    def test_sink_type_is_parquet(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        assert sink.sink_type == SinkType.PARQUET


class TestDisconnect:

    def test_disconnect_noop(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()
        sink.disconnect()  # Should not raise


class TestWriteBatch:

    def test_creates_file(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        records = [_make_record(timestamp=TS_JAN1_A)]
        sink.write_batch(records, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        assert path.exists()

    def test_correct_records(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        records = [
            _make_record(timestamp=TS_JAN1_A, open_=450.10),
            _make_record(timestamp=TS_JAN1_B, open_=450.30),
        ]
        sink.write_batch(records, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        df = pd.read_parquet(path)
        assert len(df) == 2
        assert df.iloc[0]["open"] == 450.10
        assert df.iloc[1]["open"] == 450.30

    def test_snappy_compression(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        records = [_make_record(timestamp=TS_JAN1_A)]
        sink.write_batch(records, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        meta = pq.read_metadata(path)
        compression = meta.row_group(0).column(0).compression
        assert compression == "SNAPPY"

    def test_date_partitioning(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        sink.write_batch([_make_record(timestamp=TS_JAN1_A)], partition_key="2024-01-01")
        sink.write_batch([_make_record(timestamp=TS_JAN2_A)], partition_key="2024-01-02")

        assert (tmp_path / "data" / "spy" / "2024-01-01.parquet").exists()
        assert (tmp_path / "data" / "spy" / "2024-01-02.parquet").exists()

    def test_appends_to_existing_no_duplicates(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        # First batch: 2 records
        batch1 = [
            _make_record(timestamp=TS_JAN1_A),
            _make_record(timestamp=TS_JAN1_B),
        ]
        sink.write_batch(batch1, partition_key="2024-01-01")

        # Second batch: 1 new + 1 duplicate
        batch2 = [
            _make_record(timestamp=TS_JAN1_B),  # duplicate
            _make_record(timestamp=TS_JAN1_B + 60000, open_=451.00),  # new
        ]
        sink.write_batch(batch2, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        df = pd.read_parquet(path)
        assert len(df) == 3  # 2 original + 1 new, duplicate removed

    def test_empty_records_noop(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        sink.write_batch([], partition_key="2024-01-01")
        assert not (tmp_path / "data" / "spy" / "2024-01-01.parquet").exists()

    def test_derives_partition_key_from_timestamp(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        records = [_make_record(timestamp=TS_JAN1_A)]
        sink.write_batch(records)  # No explicit partition_key

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        assert path.exists()


class TestWriteSingle:

    def test_delegates_to_write_batch(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        record = _make_record(timestamp=TS_JAN1_A)
        sink.write_single(record, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["close"] == 450.30


class TestCheckDuplicate:

    def test_duplicate_true(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        record = _make_record(timestamp=TS_JAN1_A)
        sink.write_batch([record], partition_key="2024-01-01")

        assert sink.check_duplicate(record) is True

    def test_duplicate_false(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        existing = _make_record(timestamp=TS_JAN1_A)
        sink.write_batch([existing], partition_key="2024-01-01")

        new_record = _make_record(timestamp=TS_JAN1_B)
        assert sink.check_duplicate(new_record) is False

    def test_duplicate_no_file(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        record = _make_record(timestamp=TS_JAN1_A)
        assert sink.check_duplicate(record) is False


class TestOverwrite:

    def test_replaces_data(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)
        sink.connect()

        # Write initial data
        original = [
            _make_record(timestamp=TS_JAN1_A, open_=450.10),
            _make_record(timestamp=TS_JAN1_B, open_=450.30),
        ]
        sink.write_batch(original, partition_key="2024-01-01")

        # Overwrite with different data
        replacement = [_make_record(timestamp=TS_JAN1_A, open_=999.99)]
        sink.overwrite(replacement, partition_key="2024-01-01")

        path = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["open"] == 999.99


class TestHelpers:

    def test_partition_path_structure(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)

        path = sink._partition_path("spy", "2024-01-01")
        expected = tmp_path / "data" / "spy" / "2024-01-01.parquet"
        assert path == expected

    def test_derive_partition_key(self, tmp_path):
        config = _make_config(tmp_path)
        sink = ParquetSink(config)

        record = {"timestamp": 1704067200000}  # 2024-01-01 00:00:00 UTC
        assert sink._derive_partition_key(record) == "2024-01-01"
