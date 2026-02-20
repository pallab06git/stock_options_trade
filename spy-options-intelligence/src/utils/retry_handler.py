# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Configurable retry logic with exponential backoff using tenacity.

Three error categories are handled:

1. Retryable (5xx, 429) — network/response issues worth retrying:
   - Exponential backoff: initial_wait * base^(attempt-1), capped at max_wait.
   - Jitter (uniform [0, 1)) is added when enabled.

2. Auth failures (401, 403) — log and skip, never retry:
   - Retrying wrong credentials risks locking the account.
   - Logged at WARNING with HTTP status for visibility, then returns None.

3. Data quality / schema drift (SkippableError) — log and skip, never retry:
   - Bad or malformed records won't improve on retry.
   - Schema drift requires human review, not automatic retries.
   - Logged at WARNING, then returns None so the caller continues normally.
"""

import functools
import random
from typing import Any, Dict, Optional

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
)

from src.utils.logger import get_logger

logger = get_logger()

# Default retry settings (match config/retry_policy.yaml defaults)
_DEFAULTS = {
    "max_attempts": 3,
    "initial_wait_seconds": 1.0,
    "max_wait_seconds": 30.0,
    "exponential_base": 2,
    "jitter": True,
    "retry_on_status_codes": [500, 502, 503, 504, 429],
}

# Auth status codes — log and skip immediately, never retry (prevents account lockout)
_AUTH_CODES = {401, 403}


class RetryableError(Exception):
    """Exception that carries an HTTP status code for retry decisions."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class SkippableError(Exception):
    """Raised for errors that should be logged and skipped — no retry, no propagation.

    Use for:
    - Data quality issues (malformed records, unexpected nulls, out-of-range values)
    - Schema drift (unexpected fields or type changes that require human review)

    The with_retry decorator catches this, logs a WARNING, and returns None so the
    calling loop can continue to the next record without crashing.
    """
    pass


def _get_retry_config(source: str, config: Optional[Dict[str, Any]]) -> dict:
    """
    Resolve retry config for a given source.

    Looks up config["retry"][source], falls back to config["retry"]["default"],
    then to module-level _DEFAULTS.
    """
    if config is None:
        return _DEFAULTS.copy()

    retry_section = config.get("retry", {})
    source_config = retry_section.get(source)
    if source_config:
        return source_config

    default_config = retry_section.get("default")
    if default_config:
        return default_config

    return _DEFAULTS.copy()


def _should_retry(retryable_codes: list):
    """
    Return a predicate for tenacity that decides whether to retry.

    Retries on RetryableError with status code in retryable_codes.
    Never retries SkippableError (data quality / schema drift).
    Never retries 4xx errors except 429.
    """
    codes = set(retryable_codes)

    def predicate(exception: BaseException) -> bool:
        if not isinstance(exception, RetryableError):
            return False
        sc = exception.status_code
        # Never retry 4xx except 429
        if 400 <= sc < 500 and sc != 429:
            return False
        return sc in codes

    return predicate


def _make_wait_strategy(
    initial_wait: float, exp_base: float, max_wait: float, jitter: bool
):
    """
    Build a tenacity wait callable with exponential backoff.

    All retried errors (5xx, 429) use: initial_wait * exp_base^(attempt-1), capped at max_wait.
    When jitter=True, adds uniform random noise in [0, 1) to the result.
    """
    def _wait(retry_state) -> float:
        attempt = retry_state.attempt_number
        delay = initial_wait * (exp_base ** (attempt - 1))
        delay = min(delay, max_wait)
        if jitter:
            delay += random.uniform(0, 1)
        return delay

    return _wait


def _log_before_sleep(retry_state):
    """Tenacity before_sleep callback that logs retry attempts."""
    attempt = retry_state.attempt_number
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    exc_msg = str(exc) if exc else "unknown"
    logger.warning(
        f"Retry attempt {attempt} in {wait:.1f}s — {exc_msg}"
    )


def with_retry(source: str = "default", config: Optional[Dict[str, Any]] = None):
    """
    Decorator factory that wraps a function with configurable retry logic.

    Behaviour by error type:
    - RetryableError (5xx / 429): retried with tenacity using exponential backoff.
      Raises after max_attempts exceeded.
    - RetryableError (401 / 403): logged and skipped — returns None immediately,
      no retry, to prevent account lockout.
    - SkippableError: logged and skipped — returns None immediately, no retry.
      Use for data quality issues and schema drift.

    Configuration is read from config["retry"][source], falling back to
    config["retry"]["default"], then to built-in defaults.

    Args:
        source: Name of the retry config profile (e.g., "polygon", "default").
        config: Full merged config dict containing a "retry" key.

    Returns:
        A decorator that adds retry/skip behaviour to the wrapped function.

    Example:
        @with_retry(source="polygon", config=app_config)
        def fetch_data():
            ...
    """
    rc = _get_retry_config(source, config)

    max_attempts = rc.get("max_attempts", _DEFAULTS["max_attempts"])
    initial_wait = rc.get("initial_wait_seconds", _DEFAULTS["initial_wait_seconds"])
    max_wait = rc.get("max_wait_seconds", _DEFAULTS["max_wait_seconds"])
    exp_base = rc.get("exponential_base", _DEFAULTS["exponential_base"])
    jitter = rc.get("jitter", _DEFAULTS["jitter"])
    retryable_codes = rc.get(
        "retry_on_status_codes", _DEFAULTS["retry_on_status_codes"]
    )

    wait_strategy = _make_wait_strategy(initial_wait, exp_base, max_wait, jitter)

    tenacity_decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_strategy,
        retry=retry_if_exception(_should_retry(retryable_codes)),
        before_sleep=_log_before_sleep,
        reraise=True,
    )

    def decorator(fn):
        retried_fn = tenacity_decorator(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return retried_fn(*args, **kwargs)
            except SkippableError as e:
                logger.warning(f"Skipping record (data quality/schema drift): {e}")
                return None
            except RetryableError as e:
                if e.status_code in _AUTH_CODES:
                    logger.warning(
                        f"Auth failure (HTTP {e.status_code}), skipping without retry "
                        f"to prevent account lockout: {e}"
                    )
                    return None
                raise

        return wrapper

    return decorator
