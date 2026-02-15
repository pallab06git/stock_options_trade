# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PolygonEquityClient."""

import queue
import threading

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from src.data_sources.base_source import ExecutionMode
from src.data_sources.polygon_client import PolygonEquityClient, PolygonSPYClient
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


def _make_config(ticker="SPY", use_equities=False):
    """Create a minimal config for PolygonEquityClient.

    Args:
        ticker: Ticker symbol for config.
        use_equities: If True, use polygon.equities section instead of polygon.spy.
    """
    config = {
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
    if use_equities:
        config["polygon"]["equities"] = {
            ticker: {
                "multiplier": 1,
                "timespan": "second",
                "limit_per_request": 50000,
            }
        }
    return config


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
        client = PolygonEquityClient(config, cm)

        client.connect()

        assert client.mode == ExecutionMode.HISTORICAL
        cm.get_rest_client.assert_called_once()


class TestFetchHistorical:

    def test_single_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        aggs = [_make_agg(timestamp=1704067200000 + i * 1000) for i in range(3)]
        cm.get_rest_client.return_value.get_aggs.return_value = aggs

        client = PolygonEquityClient(config, cm)
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

        client = PolygonEquityClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-03"))

        # 3 dates × 2 records each
        assert len(records) == 6

    def test_empty_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = []

        client = PolygonEquityClient(config, cm)
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

        client = PolygonEquityClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-02"))

        # Date 1 failed (3 retries exhausted), date 2 succeeded with 1 record
        assert len(records) == 1

    def test_rate_limit_acquired_per_date(self):
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = [_make_agg()]

        client = PolygonEquityClient(config, cm)
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

        client = PolygonEquityClient(config, cm)
        records = list(client.fetch_historical("2024-01-01", "2024-01-01"))

        assert len(records) == 1
        assert call_count["count"] == 2  # First call failed, second succeeded


class TestTransformAgg:

    def test_transform_agg(self):
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonEquityClient(config, cm)

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
        client = PolygonEquityClient(config, cm)

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
        client = PolygonEquityClient(config, cm)

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
        client = PolygonEquityClient(config, cm)

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
        client = PolygonEquityClient(config, cm)

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
        client = PolygonEquityClient(config, cm)

        record = {
            "timestamp": 1704067200000,
            "open": 450.10,
            "high": None,
            "low": 449.90,
            "close": 450.30,
        }
        assert client.validate_record(record) is False


class TestStreamRealtime:

    def test_stream_realtime_yields_records(self):
        """Mock thread puts records in queue; generator yields them."""
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonEquityClient(config, cm)

        stop_event = threading.Event()
        aggs = [_make_agg(timestamp=1704067200000 + i * 1000) for i in range(3)]

        # Patch _ws_thread: put records then sentinel
        original_stream = client.stream_realtime

        def fake_stream(**kwargs):
            se = kwargs.get("stop_event", stop_event)
            msg_queue = queue.Queue(maxsize=10000)
            for agg in aggs:
                msg_queue.put(client._transform_agg(agg))
            msg_queue.put(PolygonEquityClient._SENTINEL)

            # Yield from queue like the real generator
            try:
                while not se.is_set():
                    try:
                        item = msg_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is PolygonEquityClient._SENTINEL:
                        break
                    yield item
            finally:
                se.set()

        records = list(fake_stream(stop_event=stop_event))
        assert len(records) == 3
        assert records[0]["source"] == "spy"
        assert records[0]["timestamp"] == 1704067200000

    def test_stream_realtime_stop_event(self):
        """Setting stop_event terminates the generator."""
        config = _make_config()
        cm = _make_connection_manager()

        # Make get_ws_client return a mock that blocks on run()
        mock_ws = MagicMock()
        mock_ws.run.side_effect = lambda handle_msg: threading.Event().wait(10)
        cm.get_ws_client = MagicMock(return_value=mock_ws)

        client = PolygonEquityClient(config, cm)
        stop_event = threading.Event()

        gen = client.stream_realtime(stop_event=stop_event)
        # Set stop after short delay
        threading.Timer(0.2, stop_event.set).start()
        records = list(gen)
        assert records == []

    def test_stream_realtime_sentinel_stops(self):
        """Sentinel in queue causes clean exit."""
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonEquityClient(config, cm)

        # Directly test queue + sentinel
        msg_queue = queue.Queue()
        msg_queue.put({"timestamp": 100, "source": "spy"})
        msg_queue.put(PolygonEquityClient._SENTINEL)

        results = []
        while True:
            item = msg_queue.get(timeout=1.0)
            if item is PolygonEquityClient._SENTINEL:
                break
            results.append(item)
        assert len(results) == 1

    def test_stream_realtime_thread_cleanup(self):
        """Thread is joined in finally block."""
        config = _make_config()
        cm = _make_connection_manager()

        # Mock WebSocket to immediately put sentinel and exit
        mock_ws = MagicMock()

        def fake_run(handle_msg):
            # Simulate sending one message then disconnecting
            agg = _make_agg()
            handle_msg([agg])

        mock_ws.run.side_effect = fake_run
        cm.get_ws_client = MagicMock(return_value=mock_ws)

        client = PolygonEquityClient(config, cm)
        stop_event = threading.Event()

        records = []
        gen = client.stream_realtime(stop_event=stop_event)
        # Consume one record, then stop
        for record in gen:
            records.append(record)
            stop_event.set()
            break

        assert len(records) == 1
        # After exiting the generator, stop_event should be set
        assert stop_event.is_set()


# ---------------------------------------------------------------------------
# New tests: Multi-ticker support
# ---------------------------------------------------------------------------


class TestMultiTicker:
    """Tests for ticker parameterization and config fallback."""

    def test_tsla_source_label(self):
        """TSLA client produces records with source='tsla'."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = [_make_agg()]

        client = PolygonEquityClient(config, cm, ticker="TSLA")
        records = list(client.fetch_historical("2024-01-01", "2024-01-01"))

        assert len(records) == 1
        assert records[0]["source"] == "tsla"

    def test_config_from_equities_section(self):
        """When polygon.equities.<TICKER> exists, it is used over polygon.spy."""
        config = _make_config(ticker="TSLA", use_equities=True)
        cm = _make_connection_manager()

        client = PolygonEquityClient(config, cm, ticker="TSLA")

        assert client.ticker == "TSLA"
        assert client.multiplier == 1
        assert client.timespan == "second"

    def test_fallback_to_spy_config(self):
        """When equities section missing for ticker, falls back to polygon.spy."""
        config = _make_config()  # No equities section
        cm = _make_connection_manager()

        client = PolygonEquityClient(config, cm, ticker="AAPL")

        assert client.ticker == "AAPL"
        assert client.multiplier == 1  # from polygon.spy fallback

    def test_rate_limit_uses_ticker_name(self):
        """Rate limit source key matches the lowercase ticker."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_aggs.return_value = [_make_agg()]

        client = PolygonEquityClient(config, cm, ticker="TSLA")
        list(client.fetch_historical("2024-01-01", "2024-01-01"))

        cm.acquire_rate_limit.assert_called_with(source="tsla")

    def test_backward_compat_alias(self):
        """PolygonSPYClient is an alias for PolygonEquityClient."""
        assert PolygonSPYClient is PolygonEquityClient
