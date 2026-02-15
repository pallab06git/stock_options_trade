# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for HistoricalRunner."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Timestamps in Unix ms for two different dates
_TS_DAY1_A = 1737936000000  # 2025-01-27 00:00:00 UTC
_TS_DAY1_B = 1737936001000  # 2025-01-27 00:00:01 UTC
_TS_DAY1_C = 1737936002000  # 2025-01-27 00:00:02 UTC
_TS_DAY2_A = 1738022400000  # 2025-01-28 00:00:00 UTC
_TS_DAY2_B = 1738022401000  # 2025-01-28 00:00:01 UTC


def _make_record(timestamp=_TS_DAY1_A, close=450.0, source="spy"):
    """Create a valid equity aggregate record."""
    return {
        "timestamp": timestamp,
        "open": 449.50,
        "high": 450.50,
        "low": 449.00,
        "close": close,
        "volume": 1000,
        "vwap": 449.80,
        "transactions": 10,
        "source": source,
    }


def _make_invalid_record(timestamp=_TS_DAY1_A):
    """Create a record that will fail validation (negative price)."""
    return {
        "timestamp": timestamp,
        "open": -1.0,
        "high": 450.50,
        "low": 449.00,
        "close": 450.0,
        "volume": 1000,
        "vwap": 449.80,
        "transactions": 10,
        "source": "spy",
    }


def _make_config(start_date=None, end_date=None, batch_size=10000,
                 trading_days=30):
    """Build a minimal config dict for HistoricalRunner."""
    cfg = {
        "polygon": {
            "api_key": "pk_test_12345678",
            "rate_limiting": {"total_requests_per_minute": 5},
            "spy": {
                "ticker": "SPY",
                "multiplier": 1,
                "timespan": "second",
                "limit_per_request": 50000,
            },
        },
        "sinks": {
            "parquet": {
                "base_dir": "/tmp/test_data",
                "source": "spy",
                "compression": "snappy",
            },
        },
        "historical": {
            "backfill": {
                "batch_size": batch_size,
                "trading_days": trading_days,
            },
        },
        "retry": {
            "polygon": {
                "max_retries": 1,
                "base_delay_seconds": 0,
                "max_delay_seconds": 0,
                "retryable_status_codes": [429, 500, 502, 503],
            },
        },
    }
    if start_date:
        cfg["historical"]["backfill"]["start_date"] = start_date
    if end_date:
        cfg["historical"]["backfill"]["end_date"] = end_date
    return cfg


def _build_runner(config, ticker="SPY"):
    """Instantiate HistoricalRunner with all components mocked."""
    with patch("src.orchestrator.historical_runner.ConnectionManager") as MockCM, \
         patch("src.orchestrator.historical_runner.PolygonEquityClient") as MockClient, \
         patch("src.orchestrator.historical_runner.RecordValidator") as MockValidator, \
         patch("src.orchestrator.historical_runner.Deduplicator") as MockDedup, \
         patch("src.orchestrator.historical_runner.ParquetSink") as MockSink:

        runner = __import__(
            "src.orchestrator.historical_runner", fromlist=["HistoricalRunner"]
        ).HistoricalRunner(config, ticker=ticker)

    # Disable checkpoint writes by default (no filesystem in unit tests)
    runner._save_checkpoint = MagicMock()
    runner._load_checkpoint = MagicMock(return_value=set())

    return runner


# ---------------------------------------------------------------------------
# Test: Date Range Resolution
# ---------------------------------------------------------------------------

class TestResolveDateRange:
    """Tests for _resolve_date_range()."""

    def test_explicit_dates_from_config(self):
        """When start_date and end_date are in config, use them directly."""
        config = _make_config(start_date="2025-01-15", end_date="2025-02-14")
        runner = _build_runner(config)

        start, end = runner._resolve_date_range()

        assert start == "2025-01-15"
        assert end == "2025-02-14"

    def test_trading_days_fallback(self):
        """When no explicit dates, compute from trading_days relative to today."""
        config = _make_config(trading_days=10)
        runner = _build_runner(config)

        now = datetime.now()
        expected_end = now.strftime("%Y-%m-%d")
        expected_start = (now - timedelta(days=10)).strftime("%Y-%m-%d")

        start, end = runner._resolve_date_range()

        assert start == expected_start
        assert end == expected_end

    def test_default_trading_days(self):
        """Default is 30 trading_days when not specified in config."""
        config = _make_config()
        del config["historical"]["backfill"]["trading_days"]
        runner = _build_runner(config)

        assert runner.trading_days == 30


# ---------------------------------------------------------------------------
# Test: Date Range List
# ---------------------------------------------------------------------------

class TestDateRangeList:
    """Tests for _date_range_list()."""

    def test_single_date(self):
        from src.orchestrator.historical_runner import HistoricalRunner
        result = HistoricalRunner._date_range_list("2025-01-27", "2025-01-27")
        assert result == ["2025-01-27"]

    def test_multi_date(self):
        from src.orchestrator.historical_runner import HistoricalRunner
        result = HistoricalRunner._date_range_list("2025-01-27", "2025-01-29")
        assert result == ["2025-01-27", "2025-01-28", "2025-01-29"]


# ---------------------------------------------------------------------------
# Test: Partition Key Derivation
# ---------------------------------------------------------------------------

class TestPartitionFromTimestamp:
    """Tests for _partition_from_timestamp()."""

    def test_derives_correct_date(self):
        from src.orchestrator.historical_runner import HistoricalRunner

        assert HistoricalRunner._partition_from_timestamp(_TS_DAY1_A) == "2025-01-27"
        assert HistoricalRunner._partition_from_timestamp(_TS_DAY2_A) == "2025-01-28"


# ---------------------------------------------------------------------------
# Test: Full Pipeline Run
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for run() — full pipeline integration with mocks."""

    def test_full_pipeline_single_date(self):
        """Records from one date flow through validator → dedup → sink."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        records = [_make_record(_TS_DAY1_A), _make_record(_TS_DAY1_B)]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.return_value = (records, [])
        runner.deduplicator.deduplicate_batch.return_value = records

        stats = runner.run()

        assert stats["total_fetched"] == 2
        assert stats["total_valid"] == 2
        assert stats["total_invalid"] == 0
        assert stats["total_duplicates"] == 0
        assert stats["total_written"] == 2
        assert stats["dates_processed"] == 1
        runner.client.connect.assert_called_once()
        runner.sink.connect.assert_called_once()
        runner.sink.write_batch.assert_called_once_with(records, "2025-01-27")

    def test_full_pipeline_two_dates(self):
        """Records spanning two dates produce two partition flushes."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")
        runner = _build_runner(config)

        day1 = [_make_record(_TS_DAY1_A), _make_record(_TS_DAY1_B)]
        day2 = [_make_record(_TS_DAY2_A), _make_record(_TS_DAY2_B)]

        # fetch_historical is called per-date; set side_effect for each call
        runner.client.fetch_historical.side_effect = [iter(day1), iter(day2)]
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        stats = runner.run()

        assert stats["total_fetched"] == 4
        assert stats["dates_processed"] == 2
        assert stats["total_written"] == 4
        assert runner.sink.write_batch.call_count == 2

    def test_empty_fetch(self):
        """No records from client → run completes with zero stats."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        runner.client.fetch_historical.return_value = iter([])

        stats = runner.run()

        assert stats["total_fetched"] == 0
        assert stats["total_written"] == 0
        assert stats["dates_processed"] == 0
        runner.sink.write_batch.assert_not_called()

    def test_connect_and_disconnect_called(self):
        """Source and sink are always connected at start and disconnected at end."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        runner.client.fetch_historical.return_value = iter([])

        runner.run()

        runner.client.connect.assert_called_once()
        runner.sink.connect.assert_called_once()
        runner.client.disconnect.assert_called_once()
        runner.sink.disconnect.assert_called_once()
        runner.connection_manager.close.assert_called_once()

    def test_dates_skipped_zero_when_no_resume(self):
        """Without resume, dates_skipped should be 0."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)
        runner.client.fetch_historical.return_value = iter([])

        stats = runner.run()
        assert stats["dates_skipped"] == 0


# ---------------------------------------------------------------------------
# Test: Batch Flushing at batch_size Threshold
# ---------------------------------------------------------------------------

class TestBatchFlushing:
    """Tests for batch_size threshold flushing."""

    def test_flush_at_batch_size(self):
        """When buffer reaches batch_size, flush without waiting for date end."""
        config = _make_config(
            start_date="2025-01-27", end_date="2025-01-27", batch_size=2
        )
        runner = _build_runner(config)

        records = [
            _make_record(_TS_DAY1_A),
            _make_record(_TS_DAY1_B),
            _make_record(_TS_DAY1_C),
        ]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        stats = runner.run()

        # 2 records hit batch_size → flush, then 1 remaining → final flush
        assert runner.sink.write_batch.call_count == 2
        assert stats["total_fetched"] == 3
        assert stats["total_written"] == 3

    def test_exact_batch_size(self):
        """Exactly batch_size records → one mid-run flush + no leftover flush."""
        config = _make_config(
            start_date="2025-01-27", end_date="2025-01-27", batch_size=2
        )
        runner = _build_runner(config)

        records = [_make_record(_TS_DAY1_A), _make_record(_TS_DAY1_B)]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        stats = runner.run()

        # batch_size hit → flush (buffer cleared), then loop ends,
        # buffer is empty so no final flush → 1 write_batch call
        assert runner.sink.write_batch.call_count == 1
        assert stats["total_fetched"] == 2
        assert stats["total_written"] == 2


# ---------------------------------------------------------------------------
# Test: Date Boundary — Deduplicator Reset
# ---------------------------------------------------------------------------

class TestDateBoundary:
    """Tests for deduplicator reset on each date."""

    def test_deduplicator_resets_per_date(self):
        """Deduplicator.reset() is called once per date in the range."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")
        runner = _build_runner(config)

        day1 = [_make_record(_TS_DAY1_A)]
        day2 = [_make_record(_TS_DAY2_A)]
        runner.client.fetch_historical.side_effect = [iter(day1), iter(day2)]
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        runner.run()

        # reset() called at the start of each date iteration
        assert runner.deduplicator.reset.call_count == 2

    def test_deduplicator_resets_for_single_date(self):
        """Deduplicator is reset even for a single date."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        records = [_make_record(_TS_DAY1_A), _make_record(_TS_DAY1_B)]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        runner.run()

        runner.deduplicator.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Partial Failure — Some Invalid Records
# ---------------------------------------------------------------------------

class TestPartialFailure:
    """Tests for mixed valid/invalid records."""

    def test_invalid_records_tracked_valid_still_written(self):
        """Invalid records are counted; valid ones still go through to sink."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        good = _make_record(_TS_DAY1_A)
        bad = _make_invalid_record(_TS_DAY1_B)
        runner.client.fetch_historical.return_value = iter([good, bad])

        # Validator splits: 1 valid, 1 invalid
        runner.validator.validate_batch.return_value = ([good], [bad])
        runner.deduplicator.deduplicate_batch.return_value = [good]

        stats = runner.run()

        assert stats["total_fetched"] == 2
        assert stats["total_valid"] == 1
        assert stats["total_invalid"] == 1
        assert stats["total_written"] == 1
        runner.sink.write_batch.assert_called_once_with([good], "2025-01-27")

    def test_all_invalid_no_write(self):
        """If all records fail validation, sink.write_batch is not called."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        bad = _make_invalid_record(_TS_DAY1_A)
        runner.client.fetch_historical.return_value = iter([bad])
        runner.validator.validate_batch.return_value = ([], [bad])
        runner.deduplicator.deduplicate_batch.return_value = []

        stats = runner.run()

        assert stats["total_fetched"] == 1
        assert stats["total_valid"] == 0
        assert stats["total_invalid"] == 1
        assert stats["total_written"] == 0
        runner.sink.write_batch.assert_not_called()

    def test_duplicates_counted_correctly(self):
        """Duplicate records are removed by deduplicator and counted in stats."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        rec1 = _make_record(_TS_DAY1_A)
        rec2 = _make_record(_TS_DAY1_A, close=451.0)  # same timestamp = dupe
        runner.client.fetch_historical.return_value = iter([rec1, rec2])

        runner.validator.validate_batch.return_value = ([rec1, rec2], [])
        # Deduplicator keeps only last occurrence
        runner.deduplicator.deduplicate_batch.return_value = [rec2]

        stats = runner.run()

        assert stats["total_valid"] == 2
        assert stats["total_duplicates"] == 1
        assert stats["total_written"] == 1


# ---------------------------------------------------------------------------
# Test: Cleanup on Error
# ---------------------------------------------------------------------------

class TestCleanupOnError:
    """Tests for try/finally cleanup when exceptions occur."""

    def test_disconnect_called_on_fetch_error(self):
        """If fetch_historical raises, disconnect + close still execute."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        runner.client.fetch_historical.side_effect = ConnectionError("API down")

        with pytest.raises(ConnectionError):
            runner.run()

        runner.client.disconnect.assert_called_once()
        runner.sink.disconnect.assert_called_once()
        runner.connection_manager.close.assert_called_once()

    def test_disconnect_called_on_write_error(self):
        """If sink.write_batch raises, disconnect + close still execute."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        runner.client.fetch_historical.return_value = iter([_make_record(_TS_DAY1_A)])
        runner.validator.validate_batch.return_value = (
            [_make_record(_TS_DAY1_A)], []
        )
        runner.deduplicator.deduplicate_batch.return_value = [
            _make_record(_TS_DAY1_A)
        ]
        runner.sink.write_batch.side_effect = IOError("Disk full")

        with pytest.raises(IOError):
            runner.run()

        runner.client.disconnect.assert_called_once()
        runner.sink.disconnect.assert_called_once()
        runner.connection_manager.close.assert_called_once()

    def test_disconnect_called_on_connect_error(self):
        """If client.connect() raises, disconnect + close still execute."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        runner.client.connect.side_effect = RuntimeError("Auth failed")

        with pytest.raises(RuntimeError):
            runner.run()

        runner.client.disconnect.assert_called_once()
        runner.sink.disconnect.assert_called_once()
        runner.connection_manager.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _process_batch
# ---------------------------------------------------------------------------

class TestProcessBatch:
    """Tests for _process_batch() in isolation."""

    def test_returns_correct_stats(self):
        """Stats dict reflects validator, dedup, and write counts."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        records = [_make_record(_TS_DAY1_A), _make_record(_TS_DAY1_B)]
        runner.validator.validate_batch.return_value = (records, [])
        runner.deduplicator.deduplicate_batch.return_value = records

        result = runner._process_batch(records, "2025-01-27")

        assert result == {"valid": 2, "invalid": 0, "duplicates": 0, "written": 2}

    def test_skips_write_when_empty_after_dedup(self):
        """If deduplication removes all records, sink.write_batch is not called."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        records = [_make_record(_TS_DAY1_A)]
        runner.validator.validate_batch.return_value = (records, [])
        runner.deduplicator.deduplicate_batch.return_value = []

        result = runner._process_batch(records, "2025-01-27")

        assert result["written"] == 0
        runner.sink.write_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Checkpoint/Resume
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for checkpoint save/load and resume logic."""

    def test_checkpoint_saved_after_each_date(self):
        """_save_checkpoint is called once per date processed."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")
        runner = _build_runner(config)

        day1 = [_make_record(_TS_DAY1_A)]
        day2 = [_make_record(_TS_DAY2_A)]
        runner.client.fetch_historical.side_effect = [iter(day1), iter(day2)]
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        runner.run()

        assert runner._save_checkpoint.call_count == 2
        runner._save_checkpoint.assert_any_call("2025-01-27", "2025-01-27", "2025-01-28")
        runner._save_checkpoint.assert_any_call("2025-01-28", "2025-01-27", "2025-01-28")

    def test_resume_skips_completed_dates(self):
        """With resume=True and checkpoint present, completed dates are skipped."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")
        runner = _build_runner(config)

        # Checkpoint says day1 is already done
        runner._load_checkpoint.return_value = {"2025-01-27"}

        day2 = [_make_record(_TS_DAY2_A)]
        runner.client.fetch_historical.return_value = iter(day2)
        runner.validator.validate_batch.side_effect = lambda recs: (recs, [])
        runner.deduplicator.deduplicate_batch.side_effect = lambda recs: recs

        stats = runner.run(resume=True)

        # Only day2 should have been fetched
        assert stats["total_fetched"] == 1
        assert stats["dates_processed"] == 1
        assert stats["dates_skipped"] == 1
        # fetch_historical should only be called once (for day2)
        runner.client.fetch_historical.assert_called_once_with("2025-01-28", "2025-01-28")

    def test_resume_no_checkpoint_file(self):
        """Resume with no checkpoint file processes all dates (same as fresh run)."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)

        # _load_checkpoint returns empty set (default from _build_runner)
        records = [_make_record(_TS_DAY1_A)]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.return_value = (records, [])
        runner.deduplicator.deduplicate_batch.return_value = records

        stats = runner.run(resume=True)

        assert stats["total_fetched"] == 1
        assert stats["dates_processed"] == 1
        assert stats["dates_skipped"] == 0

    def test_checkpoint_file_format(self, tmp_path):
        """Checkpoint JSON has expected structure with completed_dates and last_updated."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")

        with patch("src.orchestrator.historical_runner.ConnectionManager"), \
             patch("src.orchestrator.historical_runner.PolygonEquityClient"), \
             patch("src.orchestrator.historical_runner.RecordValidator") as MockVal, \
             patch("src.orchestrator.historical_runner.Deduplicator"), \
             patch("src.orchestrator.historical_runner.ParquetSink"):

            MockVal.for_equity.return_value = MagicMock()
            runner = __import__(
                "src.orchestrator.historical_runner", fromlist=["HistoricalRunner"]
            ).HistoricalRunner(config)

        # Point checkpoint dir to tmp_path
        runner._checkpoint_dir = tmp_path

        # Save a checkpoint using the real method
        runner._save_checkpoint("2025-01-27", "2025-01-27", "2025-01-28")

        # Verify file exists and format — now includes ticker in filename
        path = tmp_path / "checkpoint_SPY_2025-01-27_2025-01-28.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert "completed_dates" in data
        assert "last_updated" in data
        assert "2025-01-27" in data["completed_dates"]

        # Save another date
        runner._save_checkpoint("2025-01-28", "2025-01-27", "2025-01-28")
        data = json.loads(path.read_text())
        assert data["completed_dates"] == ["2025-01-27", "2025-01-28"]

    def test_load_checkpoint_returns_saved_dates(self, tmp_path):
        """_load_checkpoint returns the set of dates saved by _save_checkpoint."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")

        with patch("src.orchestrator.historical_runner.ConnectionManager"), \
             patch("src.orchestrator.historical_runner.PolygonEquityClient"), \
             patch("src.orchestrator.historical_runner.RecordValidator") as MockVal, \
             patch("src.orchestrator.historical_runner.Deduplicator"), \
             patch("src.orchestrator.historical_runner.ParquetSink"):

            MockVal.for_equity.return_value = MagicMock()
            runner = __import__(
                "src.orchestrator.historical_runner", fromlist=["HistoricalRunner"]
            ).HistoricalRunner(config)

        runner._checkpoint_dir = tmp_path

        # No checkpoint yet
        assert runner._load_checkpoint("2025-01-27", "2025-01-28") == set()

        # Save and reload
        runner._save_checkpoint("2025-01-27", "2025-01-27", "2025-01-28")
        loaded = runner._load_checkpoint("2025-01-27", "2025-01-28")
        assert loaded == {"2025-01-27"}


# ---------------------------------------------------------------------------
# Test: Ticker Parameterization
# ---------------------------------------------------------------------------

class TestTickerParam:
    """Tests for multi-ticker support in HistoricalRunner."""

    def test_runner_stores_ticker(self):
        """Runner stores the ticker param."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config, ticker="TSLA")
        assert runner.ticker == "TSLA"

    def test_default_ticker_is_spy(self):
        """Default ticker is SPY when not provided."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config)
        assert runner.ticker == "SPY"

    def test_checkpoint_includes_ticker(self, tmp_path):
        """Checkpoint filename includes the ticker symbol."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-28")

        with patch("src.orchestrator.historical_runner.ConnectionManager"), \
             patch("src.orchestrator.historical_runner.PolygonEquityClient"), \
             patch("src.orchestrator.historical_runner.RecordValidator") as MockVal, \
             patch("src.orchestrator.historical_runner.Deduplicator"), \
             patch("src.orchestrator.historical_runner.ParquetSink"):

            MockVal.for_equity.return_value = MagicMock()
            runner = __import__(
                "src.orchestrator.historical_runner", fromlist=["HistoricalRunner"]
            ).HistoricalRunner(config, ticker="TSLA")

        runner._checkpoint_dir = tmp_path
        runner._save_checkpoint("2025-01-27", "2025-01-27", "2025-01-28")

        path = tmp_path / "checkpoint_TSLA_2025-01-27_2025-01-28.json"
        assert path.exists()

    def test_runner_with_tsla(self):
        """TSLA runner processes data correctly."""
        config = _make_config(start_date="2025-01-27", end_date="2025-01-27")
        runner = _build_runner(config, ticker="TSLA")

        records = [_make_record(_TS_DAY1_A, source="tsla")]
        runner.client.fetch_historical.return_value = iter(records)
        runner.validator.validate_batch.return_value = (records, [])
        runner.deduplicator.deduplicate_batch.return_value = records

        stats = runner.run()

        assert stats["total_fetched"] == 1
        assert stats["total_written"] == 1
