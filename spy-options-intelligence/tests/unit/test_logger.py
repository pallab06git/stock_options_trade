# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for logger module — redaction, sinks, and configuration."""

import time as _time
from pathlib import Path

import pytest
from loguru import logger

from src.utils.logger import redact_sensitive, setup_logger, _LOG_FORMAT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_logger():
    """Remove all loguru handlers so each test starts clean."""
    logger.remove()

    # Reset module-level _configured flag
    import src.utils.logger as _mod
    _mod._configured = False


@pytest.fixture(autouse=True)
def clean_logger():
    """Ensure logger is reset before and after each test."""
    _reset_logger()
    yield
    _reset_logger()


# ---------------------------------------------------------------------------
# setup_logger tests
# ---------------------------------------------------------------------------


class TestSetupLogger:
    """Tests for logger initialization and sink configuration."""

    def test_setup_logger_returns_logger(self):
        result = setup_logger()
        # loguru's logger is a module-level object; setup returns it
        assert result is logger

    def test_file_sinks_created(self, tmp_path):
        exec_path = tmp_path / "execution"
        err_path = tmp_path / "errors"

        setup_logger({
            "logging": {
                "execution_log_path": str(exec_path),
                "error_log_path": str(err_path),
            }
        })

        logger.info("test info message")
        logger.error("test error message")

        # Allow loguru to flush
        logger.complete()

        exec_files = list(exec_path.glob("*.log"))
        err_files = list(err_path.glob("*.log"))

        assert len(exec_files) >= 1, "Execution log file not created"
        assert len(err_files) >= 1, "Error log file not created"

    def test_execution_log_captures_debug(self, tmp_path):
        exec_path = tmp_path / "execution"

        setup_logger({
            "logging": {
                "execution_log_path": str(exec_path),
                "error_log_path": str(tmp_path / "errors"),
                "file_level": "DEBUG",
            }
        })

        logger.debug("debug message for test")
        logger.complete()

        log_content = _read_first_log(exec_path)
        assert "debug message for test" in log_content

    def test_error_log_captures_errors_only(self, tmp_path):
        exec_path = tmp_path / "execution"
        err_path = tmp_path / "errors"

        setup_logger({
            "logging": {
                "execution_log_path": str(exec_path),
                "error_log_path": str(err_path),
            }
        })

        logger.info("info message should not appear in error log")
        logger.error("error message should appear")
        logger.complete()

        err_content = _read_first_log(err_path)
        assert "error message should appear" in err_content
        assert "info message should not appear" not in err_content

    def test_config_overrides(self, tmp_path):
        exec_path = tmp_path / "execution"
        err_path = tmp_path / "errors"

        setup_logger({
            "logging": {
                "console_level": "WARNING",
                "file_level": "WARNING",
                "execution_log_path": str(exec_path),
                "error_log_path": str(err_path),
            }
        })

        logger.info("this info should be filtered out")
        logger.warning("this warning should pass")
        logger.complete()

        log_content = _read_first_log(exec_path)
        assert "this info should be filtered out" not in log_content
        assert "this warning should pass" in log_content

    def test_log_format(self, tmp_path):
        exec_path = tmp_path / "execution"

        setup_logger({
            "logging": {
                "execution_log_path": str(exec_path),
                "error_log_path": str(tmp_path / "errors"),
            }
        })

        logger.info("format check message")
        logger.complete()

        log_content = _read_first_log(exec_path)
        # Format: YYYY-MM-DD HH:mm:ss | LEVEL    | module:function:line | message
        assert " | INFO     | " in log_content
        assert "format check message" in log_content


# ---------------------------------------------------------------------------
# redact_sensitive tests
# ---------------------------------------------------------------------------


class TestRedactSensitive:
    """Tests for the redact_sensitive function."""

    def test_redact_api_key(self):
        result = redact_sensitive("api_key=pk_abc123def456")
        assert "pk_abc123def456" not in result
        assert "****" in result

    def test_redact_api_key_colon(self):
        result = redact_sensitive('api_key: pk_abc123def456')
        assert "pk_abc123def456" not in result

    def test_redact_password(self):
        result = redact_sensitive("password=secret123")
        assert "secret123" not in result
        assert "****" in result

    def test_redact_postgres_connection(self):
        result = redact_sensitive("postgres://user:mysecretpass@host/db")
        assert "mysecretpass" not in result
        assert "****" in result
        assert "host/db" in result

    def test_redact_polygon_key_pattern(self):
        result = redact_sensitive("Using key pk_1234567890abcdef")
        assert "pk_1234567890abcdef" not in result
        # Should show last 4 chars
        assert "cdef" in result
        assert "****" in result

    def test_redact_preserves_normal_text(self):
        normal = "Fetched 23400 SPY bars for 2025-10-28"
        assert redact_sensitive(normal) == normal

    def test_redact_token(self):
        result = redact_sensitive("token=abc123xyz789")
        assert "abc123xyz789" not in result
        assert "****" in result

    def test_redact_secret(self):
        result = redact_sensitive("secret=my_top_secret_value")
        assert "my_top_secret_value" not in result

    def test_redact_bearer(self):
        result = redact_sensitive("bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        assert "eyJhbGciOiJIUzI1NiJ9" not in result


class TestRedactionAppliedToFileSink:
    """Verify that redaction is applied end-to-end in file sinks."""

    def test_redaction_applied_to_file_sink(self, tmp_path):
        exec_path = tmp_path / "execution"

        setup_logger({
            "logging": {
                "execution_log_path": str(exec_path),
                "error_log_path": str(tmp_path / "errors"),
            }
        })

        secret_key = "pk_superSecretKey12345678"
        logger.info(f"Connecting with {secret_key}")
        logger.complete()

        log_content = _read_first_log(exec_path)
        assert "pk_superSecretKey12345678" not in log_content
        assert "****" in log_content

    def test_password_redacted_in_error_log(self, tmp_path):
        err_path = tmp_path / "errors"

        setup_logger({
            "logging": {
                "execution_log_path": str(tmp_path / "execution"),
                "error_log_path": str(err_path),
            }
        })

        logger.error("DB connection failed: postgres://admin:hunter2@db.local/prod")
        logger.complete()

        log_content = _read_first_log(err_path)
        assert "hunter2" not in log_content
        assert "****" in log_content


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _read_first_log(log_dir: Path) -> str:
    """Read the first .log file found in the given directory."""
    log_files = sorted(log_dir.glob("*.log"))
    assert log_files, f"No log files found in {log_dir}"
    return log_files[0].read_text(encoding="utf-8")
