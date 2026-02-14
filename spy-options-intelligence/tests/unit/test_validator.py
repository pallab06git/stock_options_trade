"""Unit tests for RecordValidator."""

import pytest

from src.processing.validator import RecordValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spy_record(**overrides):
    """Create a valid SPY record with optional overrides."""
    record = {
        "timestamp": 1704067200000,
        "open": 450.10,
        "high": 450.50,
        "low": 449.90,
        "close": 450.30,
        "volume": 1500,
        "vwap": 450.20,
        "transactions": 25,
        "source": "spy",
    }
    record.update(overrides)
    return record


def _vix_record(**overrides):
    record = {
        "timestamp": 1704067200000,
        "open": 15.20,
        "high": 15.50,
        "low": 15.00,
        "close": 15.30,
        "source": "vix",
    }
    record.update(overrides)
    return record


def _news_record(**overrides):
    record = {
        "timestamp": 1704067200000,
        "title": "SPY hits new high",
        "source": "news",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidate:

    def test_valid_spy_record(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record()) is True

    def test_missing_required_field(self):
        v = RecordValidator("spy")
        record = _spy_record()
        del record["close"]
        assert v.validate(record) is False

    def test_none_field_value(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record(high=None)) is False

    def test_negative_price(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record(open=-1.0)) is False

    def test_zero_price(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record(close=0)) is False

    def test_zero_timestamp(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record(timestamp=0)) is False

    def test_wrong_source(self):
        v = RecordValidator("spy")
        assert v.validate(_spy_record(source="vix")) is False

    def test_valid_vix_record(self):
        v = RecordValidator("vix")
        assert v.validate(_vix_record()) is True

    def test_valid_news_record(self):
        v = RecordValidator("news")
        assert v.validate(_news_record()) is True


class TestValidateBatch:

    def test_splits_valid_and_invalid(self):
        v = RecordValidator("spy")
        records = [
            _spy_record(timestamp=1704067200000),
            _spy_record(timestamp=1704067260000),
            _spy_record(timestamp=0),  # invalid
        ]
        valid, invalid = v.validate_batch(records)
        assert len(valid) == 2
        assert len(invalid) == 1

    def test_all_valid(self):
        v = RecordValidator("spy")
        records = [
            _spy_record(timestamp=1704067200000),
            _spy_record(timestamp=1704067260000),
        ]
        valid, invalid = v.validate_batch(records)
        assert len(valid) == 2
        assert len(invalid) == 0


class TestGetValidationErrors:

    def test_returns_error_strings(self):
        v = RecordValidator("spy")
        record = _spy_record(close=None, open=-1.0)
        errors = v.get_validation_errors(record)
        assert len(errors) >= 2
        assert any("close" in e for e in errors)
        assert any("open" in e for e in errors)

    def test_valid_record_no_errors(self):
        v = RecordValidator("spy")
        errors = v.get_validation_errors(_spy_record())
        assert errors == []


class TestUnknownSource:

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            RecordValidator("crypto")
