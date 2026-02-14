"""Polygon SDK session management with unified rate limiting.

Provides a single shared RESTClient instance and a token bucket rate
limiter that enforces the Polygon API rate limit budget across all
data sources.
"""

import threading
import time
from typing import Optional

from polygon import RESTClient

from src.utils.logger import get_logger

logger = get_logger()


class TokenBucket:
    """
    Thread-safe token bucket for unified rate limiting.

    Tokens are replenished at a constant rate. Callers block on acquire()
    until a token is available or the timeout expires.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second (e.g., 5/60 for 5 req/min).
            capacity: Maximum tokens (burst size).
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._paused_until = 0.0
        self._lock = threading.Condition(threading.Lock())

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Block until a token is available.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if a token was acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        with self._lock:
            while True:
                self._refill()

                # Respect pause (429 backoff)
                now = time.monotonic()
                if now < self._paused_until:
                    remaining = min(self._paused_until - now, deadline - now)
                    if remaining <= 0:
                        return False
                    self._lock.wait(timeout=remaining)
                    continue

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

                # Wait for roughly the time needed for one token
                wait_time = min((1.0 - self._tokens) / self._rate, remaining)
                self._lock.wait(timeout=wait_time)

    def pause(self, seconds: float) -> None:
        """
        Temporarily halt token consumption (e.g., on 429 response).

        Args:
            seconds: Duration to pause.
        """
        with self._lock:
            self._paused_until = time.monotonic() + seconds
            logger.info(f"Rate limiter paused for {seconds:.0f}s")

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (approximate, for monitoring)."""
        with self._lock:
            self._refill()
            return self._tokens


class ConnectionManager:
    """
    Manage Polygon SDK sessions and unified rate limiting.

    Provides a single shared RESTClient and a token bucket that
    enforces the API rate limit budget across all data sources.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full merged config dict. Expected keys:
                - polygon.api_key: Polygon API key
                - polygon.rate_limiting.total_requests_per_minute: rate limit budget
        """
        polygon_config = config.get("polygon", {})
        self._api_key = polygon_config.get("api_key", "")

        # Rate limiting
        rate_config = polygon_config.get("rate_limiting", {})
        requests_per_minute = rate_config.get("total_requests_per_minute", 5)
        self._rate_limit_wait = (
            config.get("retry", {})
            .get("polygon", {})
            .get("rate_limit_wait_seconds", 12)
        )

        self._rate_limiter = TokenBucket(
            rate=requests_per_minute / 60.0,
            capacity=requests_per_minute,
        )

        self._rest_client: Optional[RESTClient] = None
        self._lock = threading.Lock()

        logger.debug(
            f"ConnectionManager initialized: {requests_per_minute} req/min budget"
        )

    def get_rest_client(self) -> RESTClient:
        """
        Get the shared Polygon RESTClient instance (lazy initialized).

        Returns:
            Configured polygon.RESTClient.
        """
        if self._rest_client is None:
            with self._lock:
                if self._rest_client is None:
                    self._rest_client = RESTClient(api_key=self._api_key)
                    logger.info("Polygon RESTClient initialized")
        return self._rest_client

    def acquire_rate_limit(self, source: str = None) -> bool:
        """
        Block until a rate limit token is available.

        Args:
            source: Optional source name for logging context.

        Returns:
            True if acquired, False if timed out.
        """
        start = time.monotonic()
        acquired = self._rate_limiter.acquire(timeout=120.0)
        elapsed = time.monotonic() - start

        if elapsed > 1.0:
            logger.debug(
                f"Rate limit wait: {elapsed:.1f}s"
                + (f" (source={source})" if source else "")
            )

        if not acquired:
            logger.warning(
                f"Rate limit acquire timed out"
                + (f" (source={source})" if source else "")
            )

        return acquired

    def handle_rate_limit_response(self, retry_after: Optional[int] = None) -> None:
        """
        Handle a 429 rate limit response by pausing the token bucket.

        Args:
            retry_after: Seconds from Retry-After header. Falls back to
                configured rate_limit_wait_seconds.
        """
        pause_seconds = retry_after if retry_after is not None else self._rate_limit_wait
        logger.warning(f"429 rate limited — pausing for {pause_seconds}s")
        self._rate_limiter.pause(pause_seconds)

    def health_check(self) -> bool:
        """
        Check Polygon API connectivity.

        Returns:
            True if the API is reachable, False otherwise.
        """
        try:
            client = self.get_rest_client()
            # Use a lightweight endpoint to verify connectivity
            client.get_market_status()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Clean up the REST client session."""
        with self._lock:
            if self._rest_client is not None:
                try:
                    self._rest_client.close()
                except Exception:
                    pass
                self._rest_client = None
                logger.info("Polygon RESTClient closed")
