# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PolygonVIXClient."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Create a minimal config for PolygonVIXClient."""
    vix_defaults = {
        "ticker": "I:VIX",
        "market": "indices",
        "multiplier": 1,
        "timespan": "minute",
        "limit_per_request": 50000,
    }
    vix_defaults.update(overrides)
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "feed": "delayed",
            "rate_limiting": {"total_requests_per_minute": 5},
            "vix": vix_defaults,
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


def _make_agg(timestamp=1707400000000, open_=18.5, high=19.0, low=18.0, close=18.8):
    """Create a mock Polygon Agg object for VIX."""
    agg = SimpleNamespace()
    agg.timestamp = timestamp
    agg.open = open_
    agg.high = high
    agg.low = low
    agg.close = close
    agg.volume = None
    agg.vwap = None
    agg.transactions = None
    return agg


def _build_client(config=None):
    """Build a PolygonVIXClient with a mocked ConnectionManager."""
    if config is None:
        config = _make_config()

    mock_cm = MagicMock()
    mock_cm.get_rest_client.return_value = MagicMock()
    mock_cm.acquire_rate_limit.return_value = True

    from src.data_sources.polygon_vix_client import PolygonVIXClient
    client = PolygonVIXClient(config, mock_cm)
    return client, mock_cm


# ---------------------------------------------------------------------------
# Test: Config & Defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_config_defaults(self):
        """Default ticker/multiplier/timespan from config."""
        client, _ = _build_client()
        assert client.ticker == "I:VIX"
        assert client.multiplier == 1
        assert client.timespan == "minute"
        assert client.limit_per_request == 50000

    def test_config_override(self):
        """Custom config values are used."""
        config = _make_config(
            ticker="I:VIX3M",
            multiplier=5,
            timespan="second",
            limit_per_request=10000,
        )
        client, _ = _build_client(config)
        assert client.ticker == "I:VIX3M"
        assert client.multiplier == 5
        assert client.timespan == "second"
        assert client.limit_per_request == 10000


# ---------------------------------------------------------------------------
# Test: Connect
# ---------------------------------------------------------------------------

class TestConnect:

    def test_connect(self):
        """Connect acquires REST client and sets mode to HISTORICAL."""
        client, mock_cm = _build_client()
        client.connect()
        mock_cm.get_rest_client.assert_called_once()
        from src.data_sources.base_source import ExecutionMode
        assert client.mode == ExecutionMode.HISTORICAL


# ---------------------------------------------------------------------------
# Test: Transform
# ---------------------------------------------------------------------------

class TestTransformAgg:

    def test_transform_agg(self):
        """Transformed record has source='vix' and all fields mapped."""
        client, _ = _build_client()
        agg = _make_agg(timestamp=1707400000000, open_=18.5, high=19.0, low=18.0, close=18.8)
        record = client._transform_agg(agg)

        assert record["timestamp"] == 1707400000000
        assert record["open"] == 18.5
        assert record["high"] == 19.0
        assert record["low"] == 18.0
        assert record["close"] == 18.8
        assert record["source"] == "vix"
        assert record["volume"] is None
        assert record["vwap"] is None
        assert record["transactions"] is None


# ---------------------------------------------------------------------------
# Test: Validate
# ---------------------------------------------------------------------------

class TestValidateRecord:

    def test_validate_record_valid(self):
        """Valid VIX record passes validation."""
        client, _ = _build_client()
        record = {
            "timestamp": 1707400000000,
            "open": 18.5,
            "high": 19.0,
            "low": 18.0,
            "close": 18.8,
            "source": "vix",
        }
        assert client.validate_record(record) is True

    def test_validate_record_missing_field(self):
        """Record missing a required field fails validation."""
        client, _ = _build_client()
        record = {
            "timestamp": 1707400000000,
            "open": 18.5,
            "high": 19.0,
            # missing "low"
            "close": 18.8,
            "source": "vix",
        }
        assert client.validate_record(record) is False

    def test_validate_record_negative_price(self):
        """Record with negative OHLC fails validation."""
        client, _ = _build_client()
        record = {
            "timestamp": 1707400000000,
            "open": -1.0,
            "high": 19.0,
            "low": 18.0,
            "close": 18.8,
            "source": "vix",
        }
        assert client.validate_record(record) is False

    def test_validate_record_zero_timestamp(self):
        """Record with zero timestamp fails validation."""
        client, _ = _build_client()
        record = {
            "timestamp": 0,
            "open": 18.5,
            "high": 19.0,
            "low": 18.0,
            "close": 18.8,
            "source": "vix",
        }
        assert client.validate_record(record) is False


# ---------------------------------------------------------------------------
# Test: Fetch Historical
# ---------------------------------------------------------------------------

class TestFetchHistorical:

    def test_fetch_historical_single_date(self):
        """Fetching a single date yields transformed records."""
        client, mock_cm = _build_client()
        aggs = [_make_agg(1707400000000), _make_agg(1707400060000)]
        mock_rest = MagicMock()
        mock_rest.get_aggs.return_value = aggs
        mock_cm.get_rest_client.return_value = mock_rest

        records = list(client.fetch_historical("2026-02-08", "2026-02-08"))

        assert len(records) == 2
        assert all(r["source"] == "vix" for r in records)
        mock_rest.get_aggs.assert_called_once_with(
            ticker="I:VIX",
            multiplier=1,
            timespan="minute",
            from_="2026-02-08",
            to="2026-02-08",
            limit=50000,
            sort="asc",
        )

    def test_fetch_historical_empty(self):
        """Empty API response yields no records."""
        client, mock_cm = _build_client()
        mock_rest = MagicMock()
        mock_rest.get_aggs.return_value = []
        mock_cm.get_rest_client.return_value = mock_rest

        records = list(client.fetch_historical("2026-02-08", "2026-02-08"))
        assert records == []

    def test_fetch_historical_skips_error_dates(self):
        """Error on one date skips it and continues to the next."""
        client, mock_cm = _build_client()
        mock_rest = MagicMock()

        # First date errors, second date succeeds
        mock_rest.get_aggs.side_effect = [
            RuntimeError("API error"),
            [_make_agg(1707486400000)],
        ]
        mock_cm.get_rest_client.return_value = mock_rest

        records = list(client.fetch_historical("2026-02-08", "2026-02-09"))

        assert len(records) == 1
        assert records[0]["source"] == "vix"


# ---------------------------------------------------------------------------
# Test: Stream Realtime
# ---------------------------------------------------------------------------

class TestStreamRealtime:

    def test_stream_realtime_subscribes(self):
        """WebSocket subscribes to A.I:VIX with Market.Indices."""
        client, mock_cm = _build_client()

        mock_ws = MagicMock()
        # Make run() a no-op that sets stop_event
        stop = threading.Event()

        def fake_run(handle_msg=None):
            # Yield nothing, just exit
            stop.set()

        mock_ws.run.side_effect = fake_run
        mock_cm.get_ws_client.return_value = mock_ws

        gen = client.stream_realtime(stop_event=stop)
        # Consume what's available
        records = []
        for r in gen:
            records.append(r)

        from polygon.websocket.models import Market
        mock_cm.get_ws_client.assert_called_with(market=Market.Indices)
        mock_ws.subscribe.assert_called_with("A.I:VIX")

    def test_stream_realtime_yields_records(self):
        """WebSocket messages are transformed and yielded."""
        import time as _t

        client, mock_cm = _build_client()

        mock_ws = MagicMock()
        stop = threading.Event()

        agg1 = _make_agg(1707400000000, open_=18.5, high=19.0, low=18.0, close=18.8)
        agg2 = _make_agg(1707400060000, open_=18.8, high=19.2, low=18.5, close=19.1)

        def fake_run(handle_msg=None):
            handle_msg([agg1, agg2])
            # Allow main thread to read from queue before signalling stop
            _t.sleep(0.2)
            stop.set()

        mock_ws.run.side_effect = fake_run
        mock_cm.get_ws_client.return_value = mock_ws

        gen = client.stream_realtime(stop_event=stop)
        records = list(gen)

        assert len(records) == 2
        assert records[0]["source"] == "vix"
        assert records[0]["timestamp"] == 1707400000000
        assert records[1]["timestamp"] == 1707400060000
