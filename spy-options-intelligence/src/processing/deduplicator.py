# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""In-memory deduplication for streaming and batch data records.

Tracks seen keys to filter duplicate records before they reach storage.
Supports configurable key fields, batch processing, and optional LRU
eviction to bound memory usage in long-running streaming sessions.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()


class Deduplicator:
    """
    Track and filter duplicate records using an in-memory ordered dict.

    Default dedup key is 'timestamp'. Call reset() between
    partitions or date boundaries to free memory.

    When ``max_size`` is set, the oldest entries are evicted once the
    capacity is reached (LRU eviction).  This prevents unbounded memory
    growth during long-running streaming sessions.
    """

    def __init__(self, key_field: str = "timestamp", max_size: Optional[int] = None):
        """
        Args:
            key_field: Record field used as the deduplication key.
            max_size: Maximum number of keys to track.  None means unlimited.
                      When the limit is reached, the oldest key is evicted.
        """
        self.key_field = key_field
        self.max_size = max_size
        self._seen: OrderedDict = OrderedDict()
        self._eviction_count = 0

    def is_duplicate(self, record: Dict[str, Any]) -> bool:
        """
        Check if a record is a duplicate and track it.

        Args:
            record: Data record to check.

        Returns:
            True if the key was already seen, False otherwise.
        """
        key = record.get(self.key_field)
        if key in self._seen:
            # Move to end (most recent) for LRU ordering
            self._seen.move_to_end(key)
            return True
        self._seen[key] = None
        self._evict_if_needed()
        return False

    def deduplicate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates from a batch, keeping the last occurrence.

        Preserves insertion order of unique records. Consistent with
        ParquetSink's keep='last' dedup behavior.

        Args:
            records: List of data records (may contain duplicates).

        Returns:
            Deduplicated list with last occurrences preserved.
        """
        # Build a dict keyed by dedup field — last write wins
        seen: Dict[Any, Dict[str, Any]] = {}
        for record in records:
            key = record.get(self.key_field)
            seen[key] = record

        result = list(seen.values())
        duplicates_removed = len(records) - len(result)

        if duplicates_removed > 0:
            logger.info(f"Deduplicator removed {duplicates_removed} duplicates (key={self.key_field})")

        # Track all keys in the instance dict
        for key in seen:
            self._seen[key] = None
            self._seen.move_to_end(key)
        self._evict_if_needed()

        return result

    def reset(self) -> None:
        """Clear all tracked keys and eviction counter."""
        self._seen.clear()
        self._eviction_count = 0

    @property
    def seen_count(self) -> int:
        """Number of unique keys tracked."""
        return len(self._seen)

    @property
    def eviction_count(self) -> int:
        """Number of keys evicted due to max_size."""
        return self._eviction_count

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if over max_size."""
        if self.max_size is not None:
            while len(self._seen) > self.max_size:
                self._seen.popitem(last=False)
                self._eviction_count += 1
