"""Unit tests for PolygonSPYClient."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from src.data_sources.base_source import ExecutionMode
from src.data_sources.polygon_client import PolygonSPYClient
from src.utils.retry_handler import RetryableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agg(timestamp=1704067200000, open_=450.10, high=450.50,
              low=449.90, close=450.30, volume=1500, vwap=450.20,
              transactions=25):
    """Create a mock Polygon Agg object."""
    agg = SimpleNamespace(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        transactions=transactions,
        otc=None,
    )
    return agg


def _make_config():
    """Create a minimal config for PolygonSPYClient."""
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "spy": {
                "ticker": "SPY",
                "multiplier": 1,
                "timespan": "second",
                "limit_per_request": 50000,
            },
        },
        "retry": {
            "default": {
                "max_attempts": 3,
                "initial_wait_seconds": 0.01,
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
            "polygon": {
                "max_attempts": 3,
                "initial_wait_seconds": 0.01,
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
        },
    }


def _make_connection_manager():
    """Create a mock ConnectionManager."""
    cm = MagicMock()
    cm.acquire_rate_limit.return_value = True
    cm.get_rest_client.return_value = MagicMock()
    return cm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnect:

    def test_connect_sets_mode(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        client.connect()

        assert client.mode == ExecutionMode.HISTORICAL
        cm.get_rest_client.assert_called_once()


class TestFetchHistorical:

    def test_single_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        aggs = [_make_agg(timestamp=1704067200000 + i * 1000) for i in range(3)]
        cm.get_rest_client.return_value.get_aggs.return_value = aggs

        client = PolygonSPYClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-01"))

        assert len(records) == 3
        assert records[0]["source"] == "spy"
        assert records[0]["open"] == 450.10
        assert records[0]["timestamp"] == 1704067200000

    def test_date_range(self):
        config = _make_config()
        cm = _make_connection_manager()
        # Return 2 aggs per date
        aggs = [_make_agg(), _make_agg(timestamp=1704067201000)]
        cm.get_rest_client.return_value.get_aggs.return_value = aggs

        client = PolygonSPYClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-03"))

        # 3 dates × 2 records each
        assert len(records) == 6

    def test_empty_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = []

        client = PolygonSPYClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-01"))

        assert len(records) == 0

    def test_skips_date_on_persistent_error(self):
        """If a date fails after retries, it's skipped and next date proceeds."""
        config = _make_config()
        cm = _make_connection_manager()

        call_count = {"count": 0}

        def side_effect(*args, **kwargs):
            call_count["count"] += 1
            # First 3 calls (retries for date 1) fail, then succeed for date 2
            if call_count["count"] <= 3:
                exc = Exception("Server error")
                exc.status_code = 500
                raise exc
            return [_make_agg()]

        cm.get_rest_client.return_value.get_aggs.side_effect = side_effect

        client = PolygonSPYClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-02"))

        # Date 1 failed (3 retries exhausted), date 2 succeeded with 1 record
        assert len(records) == 1

    def test_rate_limit_acquired_per_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = [_make_agg()]

        client = PolygonSPYClient(config, cm)
        list(client.fetch_historical("2024-01-01", "2024-01-03"))

        # Rate limit should be acquired once per date (3 dates)
        assert cm.acquire_rate_limit.call_count == 3

    def test_retries_on_server_error(self):
        """500 error triggers retry, succeeds on subsequent attempt."""
        config = _make_config()
        cm = _make_connection_manager()

        call_count = {"count": 0}

        def side_effect(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                exc = Exception("Internal Server Error")
                exc.status_code = 500
                raise exc
            return [_make_agg()]

        cm.get_rest_client.return_value.get_aggs.side_effect = side_effect

        client = PolygonSPYClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-01"))

        assert len(records) == 1
        assert call_count["count"] == 2  # First call failed, second succeeded


class TestTransformAgg:

    def test_transform_agg(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        agg = _make_agg(
            timestamp=1704067200000,
            open_=450.10,
            high=450.50,
            low=449.90,
            close=450.30,
            volume=1500,
            vwap=450.20,
            transactions=25,
        )

        result = client._transform_agg(agg)

        assert result == {
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


class TestValidateRecord:

    def test_valid_record(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        record = {
            "timestamp": 1704067200000,
            "open": 450.10,
            "high": 450.50,
            "low": 449.90,
            "close": 450.30,
            "volume": 1500,
            "vwap": 450.20,
            "source": "spy",
        }
        assert client.validate_record(record) is True

    def test_missing_field(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        record = {
            "timestamp": 1704067200000,
            "open": 450.10,
            "high": 450.50,
            "low": 449.90,
            # "close" missing
            "volume": 1500,
        }
        assert client.validate_record(record) is False

    def test_negative_price(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        record = {
            "timestamp": 1704067200000,
            "open": -1.0,
            "high": 450.50,
            "low": 449.90,
            "close": 450.30,
        }
        assert client.validate_record(record) is False

    def test_zero_timestamp(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        record = {
            "timestamp": 0,
            "open": 450.10,
            "high": 450.50,
            "low": 449.90,
            "close": 450.30,
        }
        assert client.validate_record(record) is False

    def test_none_field(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        record = {
            "timestamp": 1704067200000,
            "open": 450.10,
            "high": None,
            "low": 449.90,
            "close": 450.30,
        }
        assert client.validate_record(record) is False


class TestStreamRealtime:

    def test_not_implemented(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonSPYClient(config, cm)

        with pytest.raises(NotImplementedError, match="Step 10"):
            next(client.stream_realtime())
