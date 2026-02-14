"""Configurable retry logic with exponential backoff using tenacity.

Provides a decorator factory that reads retry configuration from YAML
and wraps functions with tenacity retry behavior. Integrates with the
project logger for retry attempt logging.
"""

from typing import Any, Dict, Optional

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
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


class RetryableError(Exception):
    """Exception that carries an HTTP status code for retry decisions."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


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

    Uses tenacity with exponential backoff. Configuration is read from
    config["retry"][source], falling back to config["retry"]["default"],
    then to built-in defaults.

    Args:
        source: Name of the retry config profile (e.g., "polygon", "default").
        config: Full merged config dict containing a "retry" key.

    Returns:
        A decorator that adds retry behavior to the wrapped function.

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

    # Build wait strategy
    wait_strategy = wait_exponential(
        multiplier=initial_wait, exp_base=exp_base, max=max_wait
    )
    if jitter:
        wait_strategy = wait_strategy + wait_random(0, 1)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_strategy,
        retry=retry_if_exception(_should_retry(retryable_codes)),
        before_sleep=_log_before_sleep,
        reraise=True,
    )
