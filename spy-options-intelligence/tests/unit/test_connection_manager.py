"""Unit tests for connection manager and token bucket."""

import time
import threading

import pytest
from unittest.mock import patch, MagicMock

from src.utils.connection_manager import ConnectionManager, TokenBucket


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Minimal config for ConnectionManager."""
    return {
        "polygon": {
            "api_key": "pk_test_key_12345678",
            "rate_limiting": {
                "total_requests_per_minute": 60,  # 1/sec for fast tests
            },
        },
        "retry": {
            "polygon": {
                "rate_limit_wait_seconds": 12,
            },
        },
    }


@pytest.fixture
def slow_config():
    """Config with very low rate for testing blocking behavior."""
    return {
        "polygon": {
            "api_key": "pk_test_key_12345678",
            "rate_limiting": {
                "total_requests_per_minute": 6,  # 0.1/sec
            },
        },
        "retry": {
            "polygon": {
                "rate_limit_wait_seconds": 5,
            },
        },
    }


# ---------------------------------------------------------------------------
# TokenBucket tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    """Tests for the token bucket rate limiter."""

    def test_acquire_when_tokens_available(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.acquire(timeout=0.1) is True

    def test_acquire_multiple(self):
        bucket = TokenBucket(rate=10.0, capacity=3)
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.1) is True

    def test_blocks_when_exhausted(self):
        bucket = TokenBucket(rate=100.0, capacity=1)
        # Consume the only token
        assert bucket.acquire(timeout=0.1) is True
        # Next one should need to wait for refill
        start = time.monotonic()
        assert bucket.acquire(timeout=1.0) is True
        elapsed = time.monotonic() - start
        assert elapsed > 0.005  # Had to wait for refill

    def test_timeout_returns_false(self):
        bucket = TokenBucket(rate=0.1, capacity=1)
        bucket.acquire(timeout=0.1)  # Consume the token
        # Very short timeout — should fail
        assert bucket.acquire(timeout=0.01) is False

    def test_pause_delays_acquisition(self):
        bucket = TokenBucket(rate=100.0, capacity=5)
        bucket.pause(0.2)
        start = time.monotonic()
        bucket.acquire(timeout=1.0)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # Paused for ~0.2s

    def test_available_tokens_property(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.available_tokens == pytest.approx(5.0, abs=0.5)
        bucket.acquire(timeout=0.1)
        assert bucket.available_tokens == pytest.approx(4.0, abs=0.5)

    def test_thread_safety(self):
        """Multiple threads can acquire tokens without errors."""
        bucket = TokenBucket(rate=1000.0, capacity=100)
        results = []

        def worker():
            for _ in range(10):
                results.append(bucket.acquire(timeout=1.0))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        assert all(results)  # All acquisitions succeeded


# ---------------------------------------------------------------------------
# ConnectionManager tests
# ---------------------------------------------------------------------------


class TestConnectionManager:
    """Tests for Polygon connection management."""

    @patch("src.utils.connection_manager.RESTClient")
    def test_get_rest_client_returns_client(self, mock_rest_cls, config):
        manager = ConnectionManager(config)
        client = manager.get_rest_client()
        mock_rest_cls.assert_called_once_with(api_key="pk_test_key_12345678")
        assert client is mock_rest_cls.return_value

    @patch("src.utils.connection_manager.RESTClient")
    def test_get_rest_client_singleton(self, mock_rest_cls, config):
        manager = ConnectionManager(config)
        client1 = manager.get_rest_client()
        client2 = manager.get_rest_client()
        assert client1 is client2
        mock_rest_cls.assert_called_once()  # Only one instantiation

    def test_acquire_rate_limit(self, config):
        manager = ConnectionManager(config)
        # With 60 req/min (1/sec) and capacity of 60, should acquire easily
        assert manager.acquire_rate_limit(source="spy") is True

    def test_rate_limit_blocks_when_exhausted(self, slow_config):
        manager = ConnectionManager(slow_config)
        # Capacity is 6, exhaust them all
        for _ in range(6):
            manager.acquire_rate_limit()
        # Next acquire should need to wait
        start = time.monotonic()
        result = manager.acquire_rate_limit()
        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed > 0.05  # Had to wait for token refill

    def test_handle_rate_limit_response(self, config):
        manager = ConnectionManager(config)
        # Pause bucket with explicit retry_after
        manager.handle_rate_limit_response(retry_after=1)
        start = time.monotonic()
        manager.acquire_rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.8  # Bucket was paused for ~1s

    def test_handle_rate_limit_response_default(self, config):
        """Without retry_after, uses configured rate_limit_wait_seconds."""
        manager = ConnectionManager(config)
        # Should use rate_limit_wait_seconds=12 from config
        # We just verify it doesn't error — actual 12s wait too slow for tests
        with patch.object(manager._rate_limiter, "pause") as mock_pause:
            manager.handle_rate_limit_response()
            mock_pause.assert_called_once_with(12)

    @patch("src.utils.connection_manager.RESTClient")
    def test_health_check_success(self, mock_rest_cls, config):
        mock_client = mock_rest_cls.return_value
        mock_client.get_market_status.return_value = {"status": "open"}
        manager = ConnectionManager(config)
        assert manager.health_check() is True

    @patch("src.utils.connection_manager.RESTClient")
    def test_health_check_failure(self, mock_rest_cls, config):
        mock_client = mock_rest_cls.return_value
        mock_client.get_market_status.side_effect = Exception("Connection refused")
        manager = ConnectionManager(config)
        assert manager.health_check() is False

    @patch("src.utils.connection_manager.RESTClient")
    def test_close(self, mock_rest_cls, config):
        manager = ConnectionManager(config)
        manager.get_rest_client()  # Initialize client
        manager.close()
        mock_rest_cls.return_value.close.assert_called_once()
        # After close, client should be None
        assert manager._rest_client is None

    @patch("src.utils.connection_manager.RESTClient")
    def test_close_without_client(self, mock_rest_cls, config):
        """Close without initializing client should not error."""
        manager = ConnectionManager(config)
        manager.close()  # No client to close — should be a no-op
