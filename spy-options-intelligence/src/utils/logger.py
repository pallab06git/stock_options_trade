"""Centralized logging with automatic credential redaction.

Uses loguru for structured logging with:
- Console sink (INFO+, colorized)
- Execution log file (DEBUG+, daily rotation, 30-day retention)
- Error log file (ERROR+, daily rotation, 30-day retention)
- Automatic redaction of API keys, passwords, tokens, connection strings
"""

import re
import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Sensitive pattern definitions
# ---------------------------------------------------------------------------

# Each tuple: (compiled regex, replacement function)
# Order matters — more specific patterns first.
_SENSITIVE_PATTERNS = [
    # Polygon-style keys: pk_... or sk_... (standalone tokens)
    (
        re.compile(r"\b(pk_|sk_)([A-Za-z0-9_]{4,})"),
        lambda m: f"****{m.group(2)[-4:]}",
    ),
    # Key-value patterns: api_key=VALUE, api-key: VALUE, apikey="VALUE", etc.
    (
        re.compile(
            r"(api[_\-]?key|password|passwd|secret|token|bearer)"
            r"([\"'\s:=]+)"
            r"([^\s\"',;}{]+)",
            re.IGNORECASE,
        ),
        lambda m: (
            f"{m.group(1)}{m.group(2)}"
            f"****{m.group(3)[-4:]}"
            if len(m.group(3)) > 4
            else f"{m.group(1)}{m.group(2)}****"
        ),
    ),
    # Postgres/MySQL connection strings: proto://user:password@host
    (
        re.compile(r"((?:postgres|mysql|postgresql)(?:ql)?://[^:]+:)([^@]+)(@)"),
        lambda m: f"{m.group(1)}****{m.group(3)}",
    ),
]

# Log format template
_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
    "{module}:{function}:{line} | {message}"
)

# Module-level state
_configured = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def redact_sensitive(message: str) -> str:
    """
    Replace sensitive patterns in a string with redacted versions.

    Handles: API keys, passwords, tokens, secrets, bearer tokens,
    database connection strings, and Polygon pk_/sk_ key patterns.

    Args:
        message: Raw string that may contain sensitive data.

    Returns:
        String with sensitive values masked.
    """
    for pattern, replacer in _SENSITIVE_PATTERNS:
        message = pattern.sub(replacer, message)
    return message


def _redacting_format(record):
    """
    Loguru format function that applies redaction before formatting.

    Modifies record["message"] in place, then returns the format string
    with a newline (loguru convention for custom format functions).
    """
    record["message"] = redact_sensitive(record["message"])
    return _LOG_FORMAT + "\n"


def setup_logger(config: dict = None) -> "logger.__class__":
    """
    Configure the loguru logger singleton.

    Removes the default stderr handler and adds:
    - Console sink (stderr) at console_level (default INFO)
    - Execution log file at file_level (default DEBUG)
    - Error log file at ERROR level

    All sinks apply automatic credential redaction via custom format function.

    Args:
        config: Optional dict with a "logging" key containing overrides:
            console_level, file_level, retention_days,
            execution_log_path, error_log_path.

    Returns:
        The configured loguru logger instance.
    """
    global _configured

    log_config = (config or {}).get("logging", {})

    console_level = log_config.get("console_level", "INFO")
    file_level = log_config.get("file_level", "DEBUG")
    retention_days = log_config.get("retention_days", 30)
    execution_log_path = log_config.get("execution_log_path", "data/logs/execution")
    error_log_path = log_config.get("error_log_path", "data/logs/errors")

    # Remove all existing handlers (including default stderr)
    logger.remove()

    # 1. Console sink — colorized, INFO+ by default
    logger.add(
        sys.stderr,
        level=console_level,
        format=_redacting_format,
        colorize=True,
    )

    # Ensure log directories exist
    Path(execution_log_path).mkdir(parents=True, exist_ok=True)
    Path(error_log_path).mkdir(parents=True, exist_ok=True)

    # 2. Execution log — DEBUG+ by default, daily rotation
    logger.add(
        str(Path(execution_log_path) / "{time:YYYY-MM-DD}.log"),
        level=file_level,
        format=_redacting_format,
        rotation="00:00",
        retention=f"{retention_days} days",
        encoding="utf-8",
    )

    # 3. Error log — ERROR+ only, daily rotation
    logger.add(
        str(Path(error_log_path) / "{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format=_redacting_format,
        rotation="00:00",
        retention=f"{retention_days} days",
        encoding="utf-8",
    )

    _configured = True
    return logger


def get_logger() -> "logger.__class__":
    """
    Get the configured logger instance.

    Auto-configures with defaults on first call if setup_logger()
    has not been called yet.

    Returns:
        The loguru logger instance.
    """
    if not _configured:
        setup_logger()
    return logger
