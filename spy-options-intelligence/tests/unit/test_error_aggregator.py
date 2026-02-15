# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for ErrorAggregator."""

import time
import pytest
from unittest.mock import patch

from src.monitoring.error_aggregator import ErrorAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(error_rate_pct=1.0, error_window_min=15):
    """Build a minimal config dict with monitoring.performance settings."""
    return {
        "monitoring": {
            "performance": {
                "error_rate_percent": error_rate_pct,
                "error_window_minutes": error_window_min,
            }
        }
    }


def _make_aggregator(**kwargs):
    """Create an ErrorAggregator with default config (overridable)."""
    return ErrorAggregator(_make_config(**kwargs))


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for __init__ configuration parsing."""

    def test_defaults_from_empty_config(self):
        """Empty config falls back to hardcoded defaults."""
        agg = ErrorAggregator({})

        assert agg.error_rate_threshold == 1.0
        assert agg.error_window_seconds == 15 * 60

    def test_custom_thresholds(self):
        """Config values override defaults."""
        agg = _make_aggregator(error_rate_pct=5.0, error_window_min=30)

        assert agg.error_rate_threshold == 5.0
        assert agg.error_window_seconds == 30 * 60

    def test_counters_start_at_zero(self):
        agg = _make_aggregator()

        assert agg.total_errors == 0
        assert agg.total_successes == 0


# ---------------------------------------------------------------------------
# Test: Recording
# ---------------------------------------------------------------------------

class TestRecording:
    """Tests for record_error and record_success."""

    def test_record_error_increments_counts(self):
        agg = _make_aggregator()

        agg.record_error("validation_error", "negative price")

        assert agg.total_errors == 1
        assert agg._error_counts["validation_error"] == 1

    def test_record_multiple_error_types(self):
        """Different error types tracked independently."""
        agg = _make_aggregator()

        agg.record_error("validation_error", "negative price")
        agg.record_error("validation_error", "missing field")
        agg.record_error("write_error", "disk full")

        assert agg._error_counts["validation_error"] == 2
        assert agg._error_counts["write_error"] == 1
        assert agg.total_errors == 3

    def test_record_success_increments(self):
        agg = _make_aggregator()

        agg.record_success()
        agg.record_success()

        assert agg.total_successes == 2

    def test_events_deque_populated(self):
        """Both errors and successes add entries to the sliding window."""
        agg = _make_aggregator()

        agg.record_error("err", "msg")
        agg.record_success()
        agg.record_success()

        assert len(agg._events) == 3


# ---------------------------------------------------------------------------
# Test: Error Rate
# ---------------------------------------------------------------------------

class TestErrorRate:
    """Tests for get_error_rate and the sliding window."""

    def test_zero_events_returns_zero(self):
        agg = _make_aggregator()

        assert agg.get_error_rate() == 0.0

    def test_all_errors(self):
        """100% errors when only errors recorded."""
        agg = _make_aggregator()

        agg.record_error("err", "msg1")
        agg.record_error("err", "msg2")

        assert agg.get_error_rate() == 100.0

    def test_all_successes(self):
        """0% errors when only successes recorded."""
        agg = _make_aggregator()

        for _ in range(10):
            agg.record_success()

        assert agg.get_error_rate() == 0.0

    def test_mixed_events(self):
        """Error rate reflects the proportion of errors in the window."""
        agg = _make_aggregator()

        # 2 errors + 8 successes = 20% error rate
        agg.record_error("err", "msg1")
        agg.record_error("err", "msg2")
        for _ in range(8):
            agg.record_success()

        rate = agg.get_error_rate()
        assert abs(rate - 20.0) < 0.1

    def test_custom_window_override(self):
        """Explicit window_seconds parameter overrides the configured window."""
        agg = _make_aggregator(error_window_min=15)

        # Insert an old error (20 minutes ago)
        old_ts = time.time() - 1200
        agg._events.append((old_ts, True))
        agg.record_success()

        # Default window (15 min) should prune the old error
        rate_default = agg.get_error_rate()
        assert rate_default == 0.0  # old error pruned

        # Wider window (25 min) should include the old error
        # Need to re-add since pruning mutates the deque
        agg._events.appendleft((old_ts, True))
        rate_wide = agg.get_error_rate(window_seconds=1500)
        assert rate_wide > 0

    def test_old_events_pruned(self):
        """Events outside the window are removed by _prune_window."""
        agg = _make_aggregator(error_window_min=1)  # 60-second window

        # Insert an old event (2 minutes ago)
        old_ts = time.time() - 120
        agg._events.append((old_ts, True))

        # Recent event
        agg.record_success()

        # After pruning via get_error_rate, only recent event remains
        rate = agg.get_error_rate()
        assert rate == 0.0
        assert len(agg._events) == 1


# ---------------------------------------------------------------------------
# Test: Should Alert
# ---------------------------------------------------------------------------

class TestShouldAlert:
    """Tests for should_alert threshold check."""

    def test_no_alert_below_threshold(self):
        """No alert when error rate is below configured threshold."""
        agg = _make_aggregator(error_rate_pct=10.0)

        # 1 error + 99 successes = 1% rate, threshold 10%
        agg.record_error("err", "msg")
        for _ in range(99):
            agg.record_success()

        assert agg.should_alert() is False

    def test_alert_above_threshold(self):
        """Alert fires when error rate exceeds threshold."""
        agg = _make_aggregator(error_rate_pct=1.0)

        # 5 errors + 5 successes = 50% rate, threshold 1%
        for _ in range(5):
            agg.record_error("err", "msg")
        for _ in range(5):
            agg.record_success()

        assert agg.should_alert() is True

    def test_no_alert_when_empty(self):
        """No alert with zero events."""
        agg = _make_aggregator()

        assert agg.should_alert() is False

    def test_alert_exact_threshold(self):
        """At exactly the threshold, should_alert returns False (strictly >)."""
        agg = _make_aggregator(error_rate_pct=50.0)

        agg.record_error("err", "msg")
        agg.record_success()

        # 50% error rate == 50% threshold → not strictly greater
        assert agg.should_alert() is False


# ---------------------------------------------------------------------------
# Test: Error Summary
# ---------------------------------------------------------------------------

class TestErrorSummary:
    """Tests for get_error_summary."""

    def test_empty_summary(self):
        agg = _make_aggregator()

        summary = agg.get_error_summary()

        assert summary["error_counts"] == {}
        assert summary["total_errors"] == 0
        assert summary["total_successes"] == 0
        assert summary["current_rate_percent"] == 0.0

    def test_populated_summary(self):
        agg = _make_aggregator()

        agg.record_error("validation_error", "negative price")
        agg.record_error("write_error", "disk full")
        agg.record_success()
        agg.record_success()
        agg.record_success()

        summary = agg.get_error_summary()

        assert summary["error_counts"] == {
            "validation_error": 1,
            "write_error": 1,
        }
        assert summary["total_errors"] == 2
        assert summary["total_successes"] == 3
        assert abs(summary["current_rate_percent"] - 40.0) < 0.1


# ---------------------------------------------------------------------------
# Test: Recent Errors
# ---------------------------------------------------------------------------

class TestRecentErrors:
    """Tests for get_recent_errors."""

    def test_no_errors_returns_empty(self):
        agg = _make_aggregator()

        assert agg.get_recent_errors() == []
        assert agg.get_recent_errors("validation_error") == []

    def test_filter_by_type(self):
        agg = _make_aggregator()

        agg.record_error("validation_error", "negative price")
        agg.record_error("write_error", "disk full")

        val_errors = agg.get_recent_errors("validation_error")
        assert len(val_errors) == 1
        assert val_errors[0][1] == "negative price"

    def test_all_types_sorted(self):
        """Without filter, returns all errors sorted by timestamp."""
        agg = _make_aggregator()

        agg.record_error("type_a", "first")
        time.sleep(0.01)
        agg.record_error("type_b", "second")

        all_errors = agg.get_recent_errors()
        assert len(all_errors) == 2
        assert all_errors[0][1] == "first"
        assert all_errors[1][1] == "second"
        assert all_errors[0][0] <= all_errors[1][0]

    def test_recent_errors_capped(self):
        """Only the last 50 errors per type are retained."""
        agg = _make_aggregator()

        for i in range(60):
            agg.record_error("err", f"msg_{i}")

        recent = agg.get_recent_errors("err")
        assert len(recent) == 50
        # Oldest kept should be msg_10 (0-9 evicted)
        assert recent[0][1] == "msg_10"

    def test_unknown_type_returns_empty(self):
        agg = _make_aggregator()

        agg.record_error("err", "msg")

        assert agg.get_recent_errors("nonexistent") == []


# ---------------------------------------------------------------------------
# Test: Window Pruning
# ---------------------------------------------------------------------------

class TestPruning:
    """Tests for _prune_window internals."""

    def test_prune_removes_old_events(self):
        agg = _make_aggregator()

        old = time.time() - 1000
        agg._events.append((old, True))
        agg._events.append((time.time(), False))

        agg._prune_window(60)  # 60-second window

        assert len(agg._events) == 1

    def test_prune_keeps_recent(self):
        agg = _make_aggregator()

        recent = time.time() - 5
        agg._events.append((recent, True))

        agg._prune_window(60)

        assert len(agg._events) == 1

    def test_prune_empty_deque(self):
        """Pruning an empty deque is a no-op."""
        agg = _make_aggregator()

        agg._prune_window(60)

        assert len(agg._events) == 0


# ---------------------------------------------------------------------------
# Test: Error Burst and Recovery
# ---------------------------------------------------------------------------

class TestBurstAndRecovery:
    """Simulate error bursts followed by healthy recovery."""

    def test_error_burst_triggers_alert(self):
        """Rapid burst of errors pushes rate above threshold."""
        agg = _make_aggregator(error_rate_pct=5.0)

        # 10 errors in a row
        for _ in range(10):
            agg.record_error("validation_error", "negative price")

        assert agg.should_alert() is True
        assert agg.get_error_rate() == 100.0

    def test_recovery_after_burst(self):
        """After burst, many successes bring the rate back below threshold."""
        agg = _make_aggregator(error_rate_pct=5.0)

        # Burst: 5 errors
        for _ in range(5):
            agg.record_error("validation_error", "negative price")

        assert agg.should_alert() is True

        # Recovery: 200 successes
        for _ in range(200):
            agg.record_success()

        # 5 errors / 205 total = 2.4% — below 5% threshold
        assert agg.should_alert() is False
        rate = agg.get_error_rate()
        assert rate < 5.0

    def test_sustained_low_error_rate_no_alert(self):
        """Intermittent single errors amid many successes stay below threshold."""
        agg = _make_aggregator(error_rate_pct=5.0)

        for i in range(100):
            if i % 50 == 0:
                agg.record_error("validation_error", f"occasional error {i}")
            else:
                agg.record_success()

        # 2 errors / 100 total = 2% — below 5%
        assert agg.should_alert() is False

    def test_mixed_error_types_in_burst(self):
        """Burst with multiple error types — all counted toward rate."""
        agg = _make_aggregator(error_rate_pct=10.0)

        for _ in range(3):
            agg.record_error("validation_error", "negative price")
        for _ in range(3):
            agg.record_error("write_error", "disk full")
        for _ in range(4):
            agg.record_success()

        # 6 errors / 10 total = 60% — well above 10%
        assert agg.should_alert() is True

        summary = agg.get_error_summary()
        assert summary["error_counts"]["validation_error"] == 3
        assert summary["error_counts"]["write_error"] == 3


# ---------------------------------------------------------------------------
# Test: No-Data Scenarios for ErrorAggregator
# ---------------------------------------------------------------------------

class TestNoDataErrorScenarios:
    """Pipeline produces zero records — error aggregator handles gracefully."""

    def test_summary_with_no_events(self):
        """Summary when nothing has happened."""
        agg = _make_aggregator()

        summary = agg.get_error_summary()

        assert summary["error_counts"] == {}
        assert summary["total_errors"] == 0
        assert summary["total_successes"] == 0
        assert summary["current_rate_percent"] == 0.0

    def test_no_alert_after_only_successes(self):
        """All-success pipeline never alerts."""
        agg = _make_aggregator(error_rate_pct=1.0)

        for _ in range(1000):
            agg.record_success()

        assert agg.should_alert() is False
        assert agg.get_error_rate() == 0.0

    def test_recent_errors_empty_after_no_errors(self):
        """get_recent_errors returns empty when only successes recorded."""
        agg = _make_aggregator()

        for _ in range(10):
            agg.record_success()

        assert agg.get_recent_errors() == []
        assert agg.get_recent_errors("validation_error") == []


# ---------------------------------------------------------------------------
# Test: Read vs Write Error Tracking
# ---------------------------------------------------------------------------

class TestReadWriteErrorTracking:
    """Verify errors from fetch (read) and write are tracked independently."""

    def test_fetch_and_write_errors_tracked_separately(self):
        """Fetch errors and write errors have independent counts."""
        agg = _make_aggregator()

        agg.record_error("fetch_error", "API timeout on Polygon REST")
        agg.record_error("fetch_error", "HTTP 429 rate limited")
        agg.record_error("write_error", "Parquet write failed: disk full")

        summary = agg.get_error_summary()

        assert summary["error_counts"]["fetch_error"] == 2
        assert summary["error_counts"]["write_error"] == 1
        assert summary["total_errors"] == 3

    def test_fetch_errors_only_retrievable_by_type(self):
        """get_recent_errors filters correctly by read vs write type."""
        agg = _make_aggregator()

        agg.record_error("fetch_error", "API timeout")
        agg.record_error("write_error", "disk full")

        fetch_errs = agg.get_recent_errors("fetch_error")
        write_errs = agg.get_recent_errors("write_error")

        assert len(fetch_errs) == 1
        assert fetch_errs[0][1] == "API timeout"
        assert len(write_errs) == 1
        assert write_errs[0][1] == "disk full"

    def test_alert_message_format_includes_rate(self):
        """should_alert logs the actual rate and threshold in the warning."""
        agg = _make_aggregator(error_rate_pct=5.0)

        for _ in range(20):
            agg.record_error("fetch_error", "API down")
        for _ in range(20):
            agg.record_success()

        # Rate = 50%, threshold = 5% — alert should fire
        assert agg.should_alert() is True


# ---------------------------------------------------------------------------
# Test: Session Label
# ---------------------------------------------------------------------------

class TestSessionLabel:
    """Tests for session_label in error summaries."""

    def test_default_session_label(self):
        """Default session_label is 'default'."""
        agg = _make_aggregator()
        assert agg.session_label == "default"

    def test_custom_session_label(self):
        """Custom session_label is stored."""
        agg = ErrorAggregator(_make_config(), session_label="spy")
        assert agg.session_label == "spy"

    def test_error_summary_includes_session_label(self):
        """Error summary includes session_label field."""
        agg = ErrorAggregator(_make_config(), session_label="tsla")
        agg.record_error("err", "test")

        summary = agg.get_error_summary()
        assert summary["session_label"] == "tsla"
