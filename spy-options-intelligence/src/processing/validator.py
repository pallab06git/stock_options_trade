"""Centralized schema validation for all data source records.

Validates records against per-source schemas before they reach storage.
Supports SPY, options, VIX, and news record types.
"""

from typing import Dict, Any, List, Tuple

from src.utils.logger import get_logger

logger = get_logger()

# Per-source validation schemas
_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "spy": {
        "required_fields": {"timestamp", "open", "high", "low", "close", "volume", "vwap", "source"},
        "positive_fields": {"open", "high", "low", "close", "volume"},
        "positive_timestamp": True,
        "expected_source": "spy",
    },
    "options": {
        "required_fields": {"timestamp", "open", "high", "low", "close", "volume", "source"},
        "positive_fields": {"open", "high", "low", "close"},
        "positive_timestamp": True,
        "expected_source": "options",
    },
    "vix": {
        "required_fields": {"timestamp", "open", "high", "low", "close", "source"},
        "positive_fields": {"open", "high", "low", "close"},
        "positive_timestamp": True,
        "expected_source": "vix",
    },
    "news": {
        "required_fields": {"timestamp", "title", "source"},
        "positive_fields": set(),
        "positive_timestamp": True,
        "expected_source": "news",
    },
}


class RecordValidator:
    """
    Validate data records against per-source schemas.

    Each source type (spy, options, vix, news) has its own schema
    defining required fields, positive-value constraints, and
    expected source labels.
    """

    def __init__(self, source: str):
        """
        Args:
            source: Source type key (spy, options, vix, news).

        Raises:
            ValueError: If source is not a known schema.
        """
        if source not in _SCHEMAS:
            raise ValueError(f"Unknown source '{source}'. Valid: {sorted(_SCHEMAS.keys())}")
        self.source = source
        self._schema = _SCHEMAS[source]

    def validate(self, record: Dict[str, Any]) -> bool:
        """
        Validate a single record against the source schema.

        Args:
            record: Data record to validate.

        Returns:
            True if valid, False otherwise.
        """
        return len(self.get_validation_errors(record)) == 0

    def validate_batch(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split records into valid and invalid lists.

        Args:
            records: List of data records.

        Returns:
            Tuple of (valid_records, invalid_records).
        """
        valid = []
        invalid = []
        for record in records:
            if self.validate(record):
                valid.append(record)
            else:
                invalid.append(record)

        if invalid:
            logger.warning(
                f"Validation: {len(invalid)}/{len(records)} {self.source} records invalid"
            )
        return valid, invalid

    def get_validation_errors(self, record: Dict[str, Any]) -> List[str]:
        """
        Return human-readable error strings for a record.

        Args:
            record: Data record to check.

        Returns:
            List of error description strings. Empty if valid.
        """
        errors = []

        # Required fields present and not None
        for field in sorted(self._schema["required_fields"]):
            if field not in record or record[field] is None:
                errors.append(f"Missing or None field: {field}")

        # Positive timestamp
        if self._schema.get("positive_timestamp"):
            ts = record.get("timestamp")
            if ts is not None and not isinstance(ts, str) and ts <= 0:
                errors.append(f"Timestamp must be positive, got: {ts}")

        # Positive numeric fields
        for field in sorted(self._schema.get("positive_fields", set())):
            val = record.get(field)
            if val is not None and val <= 0:
                errors.append(f"Field '{field}' must be positive, got: {val}")

        # Expected source label
        expected = self._schema.get("expected_source")
        if expected and record.get("source") is not None:
            if record["source"] != expected:
                errors.append(f"Expected source='{expected}', got: '{record['source']}'")

        return errors
