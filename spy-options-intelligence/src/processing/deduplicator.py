# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""In-memory deduplication for streaming and batch data records.

Tracks seen keys to filter duplicate records before they reach storage.
Supports configurable key fields and batch processing.
"""

from typing import Dict, Any, List

from src.utils.logger import get_logger

logger = get_logger()


class Deduplicator:
    """
    Track and filter duplicate records using an in-memory set.

    Default dedup key is 'timestamp'. Call reset() between
    partitions or date boundaries to free memory.
    """

    def __init__(self, key_field: str = "timestamp"):
        """
        Args:
            key_field: Record field used as the deduplication key.
        """
        self.key_field = key_field
        self._seen: set = set()

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
            return True
        self._seen.add(key)
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

        # Track all keys in the instance set
        self._seen.update(seen.keys())

        return result

    def reset(self) -> None:
        """Clear all tracked keys."""
        self._seen.clear()

    @property
    def seen_count(self) -> int:
        """Number of unique keys tracked."""
        return len(self._seen)
