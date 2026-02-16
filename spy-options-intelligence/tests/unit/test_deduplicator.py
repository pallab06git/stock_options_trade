# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

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


class TestLRUEviction:

    def test_evicts_oldest_when_max_size_exceeded(self):
        """When max_size is reached, oldest entries are evicted."""
        d = Deduplicator(max_size=3)
        d.is_duplicate(_record(timestamp=100))
        d.is_duplicate(_record(timestamp=200))
        d.is_duplicate(_record(timestamp=300))
        d.is_duplicate(_record(timestamp=400))  # evicts 100

        assert d.seen_count == 3
        assert d.eviction_count == 1
        # 100 was evicted, so it's no longer a duplicate
        assert d.is_duplicate(_record(timestamp=100)) is False

    def test_max_size_none_allows_unlimited(self):
        """max_size=None allows unlimited entries (backward compatible)."""
        d = Deduplicator(max_size=None)
        for i in range(500):
            d.is_duplicate(_record(timestamp=i))
        assert d.seen_count == 500
        assert d.eviction_count == 0

    def test_lru_order_preserved(self):
        """Recently accessed keys survive eviction; oldest are evicted first."""
        d = Deduplicator(max_size=3)
        d.is_duplicate(_record(timestamp=100))
        d.is_duplicate(_record(timestamp=200))
        d.is_duplicate(_record(timestamp=300))

        # Access 100 again (moves it to end as most recent)
        d.is_duplicate(_record(timestamp=100))

        # Insert 400 — should evict 200 (now the oldest)
        d.is_duplicate(_record(timestamp=400))

        assert d.seen_count == 3
        # 200 should be evicted
        assert d.is_duplicate(_record(timestamp=200)) is False
        # 100 should still be there (was refreshed)
        assert d.is_duplicate(_record(timestamp=100)) is True

    def test_deduplicate_batch_respects_max_size(self):
        """Batch deduplication also respects max_size limit."""
        d = Deduplicator(max_size=3)
        records = [_record(timestamp=i) for i in range(5)]
        d.deduplicate_batch(records)

        assert d.seen_count == 3
        assert d.eviction_count == 2

    def test_reset_clears_eviction_counter(self):
        """reset() clears both seen keys and eviction counter."""
        d = Deduplicator(max_size=2)
        for i in range(5):
            d.is_duplicate(_record(timestamp=i))

        assert d.eviction_count == 3

        d.reset()
        assert d.seen_count == 0
        assert d.eviction_count == 0


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
