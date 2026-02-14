# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for retry handler."""

import pytest
from unittest.mock import MagicMock, patch
from tenacity import RetryError

from src.utils.retry_handler import with_retry, RetryableError, _get_retry_config


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retry_config():
    """Minimal retry config matching retry_policy.yaml structure."""
    return {
        "retry": {
            "default": {
                "max_attempts": 3,
                "initial_wait_seconds": 0.01,  # Fast for tests
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
            "polygon": {
                "max_attempts": 5,
                "initial_wait_seconds": 0.01,
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWithRetry:
    """Tests for the with_retry decorator factory."""

    def test_succeeds_first_try(self, retry_config):
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_500(self, retry_config):
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Server error", status_code=500)
            return "recovered"

        result = fail_then_succeed()
        assert result == "recovered"
        assert call_count == 3

    def test_retries_on_429(self, retry_config):
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Rate limited", status_code=429)
            return "ok"

        result = rate_limited()
        assert result == "ok"
        assert call_count == 2

    def test_no_retry_on_400(self, retry_config):
        """4xx errors (except 429) should not be retried."""
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def bad_request():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Bad request", status_code=400)

        with pytest.raises(RetryableError, match="Bad request"):
            bad_request()

        assert call_count == 1  # No retry

    def test_no_retry_on_403(self, retry_config):
        """403 Forbidden should not be retried."""
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def forbidden():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Forbidden", status_code=403)

        with pytest.raises(RetryableError, match="Forbidden"):
            forbidden()

        assert call_count == 1

    def test_max_attempts_exceeded(self, retry_config):
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Server error", status_code=500)

        with pytest.raises(RetryableError):
            always_fail()

        assert call_count == 3  # default max_attempts

    def test_uses_source_config(self, retry_config):
        """Polygon config has max_attempts=5."""
        call_count = 0

        @with_retry(source="polygon", config=retry_config)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Server error", status_code=500)

        with pytest.raises(RetryableError):
            always_fail()

        assert call_count == 5  # polygon max_attempts

    def test_falls_back_to_default(self, retry_config):
        """Unknown source falls back to default config."""
        call_count = 0

        @with_retry(source="unknown_source", config=retry_config)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Server error", status_code=500)

        with pytest.raises(RetryableError):
            always_fail()

        assert call_count == 3  # default max_attempts

    def test_logs_attempts(self, retry_config):
        """Logger is called on retry with attempt info."""
        call_count = 0

        @with_retry(source="default", config=retry_config)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Server error", status_code=500)
            return "ok"

        with patch("src.utils.retry_handler.logger") as mock_logger:
            fail_then_succeed()

        # before_sleep should have logged at least once
        mock_logger.warning.assert_called()
        log_msg = mock_logger.warning.call_args[0][0]
        assert "Retry attempt" in log_msg


class TestGetRetryConfig:
    """Tests for config resolution logic."""

    def test_none_config_returns_defaults(self):
        result = _get_retry_config("anything", None)
        assert result["max_attempts"] == 3

    def test_source_config_preferred(self, retry_config):
        result = _get_retry_config("polygon", retry_config)
        assert result["max_attempts"] == 5

    def test_default_fallback(self, retry_config):
        result = _get_retry_config("nonexistent", retry_config)
        assert result["max_attempts"] == 3


class TestRetryableError:
    """Tests for the RetryableError exception."""

    def test_stores_status_code(self):
        err = RetryableError("test", status_code=502)
        assert err.status_code == 502
        assert str(err) == "test"

    def test_default_status_code(self):
        err = RetryableError("test")
        assert err.status_code == 0
