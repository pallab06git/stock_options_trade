"""Unit tests for Deduplicator."""

import pytest

from src.processing.deduplicator import Deduplicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(timestamp=1704067200000, open_=450.10):
    return {
        "timestamp": timestamp,
        "open": open_,
        "high": 450.50,
        "low": 449.90,
        "close": 450.30,
        "source": "spy",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsDuplicate:

    def test_first_record_not_duplicate(self):
        d = Deduplicator()
        assert d.is_duplicate(_record()) is False

    def test_second_same_timestamp_is_duplicate(self):
        d = Deduplicator()
        d.is_duplicate(_record(timestamp=100))
        assert d.is_duplicate(_record(timestamp=100)) is True

    def test_different_timestamps_not_duplicate(self):
        d = Deduplicator()
        d.is_duplicate(_record(timestamp=100))
        assert d.is_duplicate(_record(timestamp=200)) is False


class TestDeduplicateBatch:

    def test_removes_duplicates(self):
        d = Deduplicator()
        records = [
            _record(timestamp=100),
            _record(timestamp=200),
            _record(timestamp=100),  # duplicate
        ]
        result = d.deduplicate_batch(records)
        assert len(result) == 2

    def test_keeps_last_occurrence(self):
        d = Deduplicator()
        records = [
            _record(timestamp=100, open_=1.0),
            _record(timestamp=100, open_=2.0),  # should win
        ]
        result = d.deduplicate_batch(records)
        assert len(result) == 1
        assert result[0]["open"] == 2.0

    def test_preserves_order(self):
        d = Deduplicator()
        records = [
            _record(timestamp=300),
            _record(timestamp=100),
            _record(timestamp=200),
        ]
        result = d.deduplicate_batch(records)
        assert [r["timestamp"] for r in result] == [300, 100, 200]

    def test_empty_batch(self):
        d = Deduplicator()
        result = d.deduplicate_batch([])
        assert result == []


class TestReset:

    def test_clears_state(self):
        d = Deduplicator()
        d.is_duplicate(_record(timestamp=100))
        assert d.is_duplicate(_record(timestamp=100)) is True

        d.reset()
        assert d.is_duplicate(_record(timestamp=100)) is False


class TestSeenCount:

    def test_tracks_unique_keys(self):
        d = Deduplicator()
        d.is_duplicate(_record(timestamp=100))
        d.is_duplicate(_record(timestamp=200))
        d.is_duplicate(_record(timestamp=100))  # duplicate
        assert d.seen_count == 2


class TestCustomKeyField:

    def test_dedup_by_custom_field(self):
        d = Deduplicator(key_field="source")
        records = [
            {"source": "spy", "timestamp": 100},
            {"source": "spy", "timestamp": 200},  # same source = duplicate
            {"source": "vix", "timestamp": 300},
        ]
        result = d.deduplicate_batch(records)
        assert len(result) == 2
