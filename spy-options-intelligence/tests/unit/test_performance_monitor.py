# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PerformanceMonitor."""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from src.monitoring.performance_monitor import PerformanceMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    commit_latency=300,
    throughput_min=100,
    memory_mb=1000,
    dump_interval_min=15,
    window_size=100,
):
    """Build a minimal config dict with monitoring.performance settings."""
    return {
        "monitoring": {
            "performance": {
                "commit_latency_seconds": commit_latency,
                "throughput_min_records_per_sec": throughput_min,
                "memory_usage_mb_threshold": memory_mb,
                "metrics_dump_interval_minutes": dump_interval_min,
                "latency_window_size": window_size,
            }
        }
    }


def _make_monitor(**kwargs):
    """Create a PerformanceMonitor with default config (overridable)."""
    return PerformanceMonitor(_make_config(**kwargs))


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for __init__ configuration parsing."""

    def test_defaults_from_empty_config(self):
        """Empty config falls back to hardcoded defaults."""
        monitor = PerformanceMonitor({})

        assert monitor.commit_latency_threshold == 300
        assert monitor.throughput_min == 100
        assert monitor.memory_threshold_mb == 1000
        assert monitor.metrics_dump_interval == 15 * 60
        assert monitor.latency_window_size == 100

    def test_custom_thresholds(self):
        """Config values override defaults."""
        monitor = _make_monitor(
            commit_latency=60,
            throughput_min=500,
            memory_mb=2048,
            dump_interval_min=5,
            window_size=50,
        )

        assert monitor.commit_latency_threshold == 60
        assert monitor.throughput_min == 500
        assert monitor.memory_threshold_mb == 2048
        assert monitor.metrics_dump_interval == 300
        assert monitor.latency_window_size == 50

    def test_counters_start_at_zero(self):
        monitor = _make_monitor()

        assert monitor.total_records_processed == 0
        assert monitor.total_operations == 0


# ---------------------------------------------------------------------------
# Test: Timer API
# ---------------------------------------------------------------------------

class TestTimerAPI:
    """Tests for start_operation / end_operation."""

    def test_basic_timing(self):
        """end_operation records a latency sample."""
        monitor = _make_monitor()

        monitor.start_operation("write")
        # Minimal sleep to ensure non-zero elapsed
        time.sleep(0.01)
        alerts = monitor.end_operation("write", record_count=100)

        assert monitor.total_records_processed == 100
        assert monitor.total_operations == 1
        stats = monitor.get_latency_stats("write")
        assert stats["p50"] is not None
        assert stats["p50"] > 0

    def test_end_without_start_returns_empty(self):
        """Calling end_operation without start logs a warning and returns []."""
        monitor = _make_monitor()

        alerts = monitor.end_operation("missing_op", 50)

        assert alerts == []
        assert monitor.total_records_processed == 0

    def test_multiple_operations_tracked_separately(self):
        """Different operation names have independent latency windows."""
        monitor = _make_monitor()

        monitor.start_operation("validate")
        monitor.end_operation("validate", 10)

        monitor.start_operation("write")
        monitor.end_operation("write", 20)

        assert monitor.total_operations == 2
        assert monitor.total_records_processed == 30

        val_stats = monitor.get_latency_stats("validate")
        write_stats = monitor.get_latency_stats("write")
        assert val_stats["p50"] is not None
        assert write_stats["p50"] is not None

    def test_cumulative_counters(self):
        """Repeated operations accumulate correctly."""
        monitor = _make_monitor()

        for i in range(5):
            monitor.start_operation("write")
            monitor.end_operation("write", 100)

        assert monitor.total_records_processed == 500
        assert monitor.total_operations == 5


# ---------------------------------------------------------------------------
# Test: Alert Checks
# ---------------------------------------------------------------------------

class TestAlertChecks:
    """Tests for check_alerts threshold evaluation."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_no_alerts_when_within_thresholds(self, mock_mem):
        """No alerts fired when all metrics are within bounds."""
        monitor = _make_monitor(commit_latency=300, throughput_min=10)

        alerts = monitor.check_alerts("write", elapsed=1.0, record_count=100)

        assert alerts == []

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_latency_alert(self, mock_mem):
        """Alert fires when elapsed exceeds commit_latency_threshold."""
        monitor = _make_monitor(commit_latency=5, throughput_min=1)

        alerts = monitor.check_alerts("write", elapsed=10.0, record_count=100)

        assert len(alerts) == 1
        assert "latency" in alerts[0]
        assert "10.0s" in alerts[0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_throughput_alert(self, mock_mem):
        """Alert fires when throughput drops below minimum."""
        monitor = _make_monitor(throughput_min=1000)

        # 10 records in 1 second = 10 rec/s, threshold is 1000
        alerts = monitor.check_alerts("write", elapsed=1.0, record_count=10)

        assert any("throughput" in a for a in alerts)

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_no_throughput_alert_zero_records(self, mock_mem):
        """Throughput alert skipped when record_count is 0."""
        monitor = _make_monitor(throughput_min=1000)

        alerts = monitor.check_alerts("write", elapsed=1.0, record_count=0)

        assert not any("throughput" in a for a in alerts)

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=2000.0)
    def test_memory_alert(self, mock_mem):
        """Alert fires when memory exceeds threshold."""
        monitor = _make_monitor(memory_mb=1000)

        alerts = monitor.check_alerts("write", elapsed=0.1, record_count=100)

        assert any("Memory" in a for a in alerts)

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=2000.0)
    def test_multiple_alerts_simultaneously(self, mock_mem):
        """Multiple thresholds can trigger in a single check."""
        monitor = _make_monitor(commit_latency=1, throughput_min=10000, memory_mb=500)

        alerts = monitor.check_alerts("write", elapsed=5.0, record_count=10)

        # Latency + throughput + memory = 3 alerts
        assert len(alerts) == 3


# ---------------------------------------------------------------------------
# Test: Latency Percentiles
# ---------------------------------------------------------------------------

class TestLatencyStats:
    """Tests for get_latency_stats percentile computation."""

    def test_empty_returns_none(self):
        """No data yields all-None percentiles."""
        monitor = _make_monitor()

        stats = monitor.get_latency_stats("unknown_op")

        assert stats == {"p50": None, "p95": None, "p99": None}

    def test_single_sample(self):
        """One sample: all percentiles equal that sample."""
        monitor = _make_monitor()
        monitor._latencies["write"].append(0.5)

        stats = monitor.get_latency_stats("write")

        assert stats["p50"] == 0.5
        assert stats["p95"] == 0.5
        assert stats["p99"] == 0.5

    def test_ordered_samples(self):
        """Known ordered data produces expected percentiles."""
        monitor = _make_monitor(window_size=200)

        # 100 evenly spaced values: 0.01, 0.02, ..., 1.00
        for i in range(1, 101):
            monitor._latencies["write"].append(i / 100.0)

        stats = monitor.get_latency_stats("write")

        # p50 ≈ 0.50, p95 ≈ 0.95, p99 ≈ 0.99
        assert 0.49 <= stats["p50"] <= 0.51
        assert 0.94 <= stats["p95"] <= 0.96
        assert 0.98 <= stats["p99"] <= 1.00

    def test_rolling_window_evicts_old(self):
        """Window maxlen caps the number of samples retained."""
        monitor = _make_monitor(window_size=5)

        for val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            monitor._latencies["write"].append(val)

        # Only last 5 should remain: 3, 4, 5, 6, 7
        assert len(monitor._latencies["write"]) == 5
        # Sorted: [3, 4, 5, 6, 7] — p50 nearest-rank = index 1 → 4.0
        stats = monitor.get_latency_stats("write")
        assert stats["p50"] == 4.0


# ---------------------------------------------------------------------------
# Test: Throughput
# ---------------------------------------------------------------------------

class TestThroughput:
    """Tests for get_throughput."""

    def test_no_data_returns_zero(self):
        monitor = _make_monitor()

        assert monitor.get_throughput("write") == 0.0

    def test_recent_data_counted(self):
        """Entries within the window contribute to throughput."""
        monitor = _make_monitor()

        now = time.time()
        monitor._throughput["write"].append((now - 5, 500))
        monitor._throughput["write"].append((now - 2, 300))

        tp = monitor.get_throughput("write", window_seconds=60)

        assert tp > 0

    def test_old_data_excluded(self):
        """Entries outside the window are ignored."""
        monitor = _make_monitor()

        old = time.time() - 120  # 2 minutes ago
        monitor._throughput["write"].append((old, 1000))

        tp = monitor.get_throughput("write", window_seconds=60)

        assert tp == 0.0


# ---------------------------------------------------------------------------
# Test: Memory
# ---------------------------------------------------------------------------

class TestMemory:
    """Tests for get_memory_usage_mb."""

    def test_returns_positive_float(self):
        monitor = _make_monitor()

        mem = monitor.get_memory_usage_mb()

        assert isinstance(mem, float)
        assert mem > 0


# ---------------------------------------------------------------------------
# Test: Metrics Dump
# ---------------------------------------------------------------------------

class TestMetricsDump:
    """Tests for should_dump_metrics and dump_metrics."""

    def test_should_dump_first_time(self):
        """First call always returns True (no previous dump)."""
        monitor = _make_monitor()

        assert monitor.should_dump_metrics() is True

    def test_should_not_dump_immediately_after(self):
        """Right after a dump, should_dump returns False."""
        monitor = _make_monitor(dump_interval_min=15)
        monitor._last_dump_time = time.time()

        assert monitor.should_dump_metrics() is False

    def test_should_dump_after_interval(self):
        """Returns True when interval has elapsed."""
        monitor = _make_monitor(dump_interval_min=1)
        monitor._last_dump_time = time.time() - 120  # 2 minutes ago

        assert monitor.should_dump_metrics() is True

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=250.5)
    def test_dump_writes_json(self, mock_mem, tmp_path):
        """dump_metrics creates a valid JSON file with expected structure."""
        monitor = _make_monitor()
        monitor._metrics_dir = tmp_path

        # Add some data
        monitor._latencies["write"].append(0.1)
        monitor._latencies["write"].append(0.2)
        monitor._throughput["write"].append((time.time(), 100))
        monitor.total_records_processed = 200
        monitor.total_operations = 3

        path = monitor.dump_metrics()

        assert path.exists()
        data = json.loads(path.read_text())

        assert data["total_records_processed"] == 200
        assert data["total_operations"] == 3
        assert data["memory_usage_mb"] == 250.5
        assert "write" in data["operations"]
        assert "latency_p50" in data["operations"]["write"]
        assert "timestamp" in data

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=100.0)
    def test_dump_updates_last_dump_time(self, mock_mem, tmp_path):
        """After dump_metrics, _last_dump_time is set."""
        monitor = _make_monitor()
        monitor._metrics_dir = tmp_path

        assert monitor._last_dump_time is None

        monitor.dump_metrics()

        assert monitor._last_dump_time is not None
        assert monitor.should_dump_metrics() is False


# ---------------------------------------------------------------------------
# Test: Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for get_summary."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=300.0)
    def test_summary_structure(self, mock_mem):
        """Summary has expected top-level keys and per-operation stats."""
        monitor = _make_monitor()
        monitor.total_records_processed = 1000
        monitor.total_operations = 5
        monitor._latencies["validate"].append(0.05)
        monitor._latencies["write"].append(0.3)

        summary = monitor.get_summary()

        assert summary["total_records_processed"] == 1000
        assert summary["total_operations"] == 5
        assert summary["memory_usage_mb"] == 300.0
        assert "validate" in summary["operations"]
        assert "write" in summary["operations"]
        assert "p50" in summary["operations"]["write"]


# ---------------------------------------------------------------------------
# Test: Percentile Edge Cases
# ---------------------------------------------------------------------------

class TestPercentileEdgeCases:
    """Tests for _percentile static method."""

    def test_empty_list(self):
        assert PerformanceMonitor._percentile([], 50) == 0.0

    def test_single_element(self):
        assert PerformanceMonitor._percentile([42.0], 99) == 42.0

    def test_two_elements(self):
        result = PerformanceMonitor._percentile([1.0, 2.0], 50)
        assert result == 1.0


# ---------------------------------------------------------------------------
# Test: End-to-End Timer Flow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Integration-style test verifying the timer → stats → dump flow."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_timer_to_dump(self, mock_mem, tmp_path):
        """Full flow: start → end → percentiles → dump."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)
        monitor._metrics_dir = tmp_path

        for _ in range(10):
            monitor.start_operation("write")
            monitor.end_operation("write", 50)

        assert monitor.total_records_processed == 500
        assert monitor.total_operations == 10

        stats = monitor.get_latency_stats("write")
        assert stats["p50"] is not None

        path = monitor.dump_metrics()
        data = json.loads(path.read_text())
        assert data["operations"]["write"]["samples"] == 10


# ---------------------------------------------------------------------------
# Test: Stale / Hung Operation Detection
# ---------------------------------------------------------------------------

class TestStaleOperations:
    """Tests for check_stale_operations — detecting hung reads or writes."""

    def test_stale_write_detected(self):
        """A write operation that exceeds timeout is reported as stale."""
        monitor = _make_monitor(commit_latency=5)

        # Simulate a write that started 10 seconds ago
        monitor._timers["write"] = time.monotonic() - 10

        alerts = monitor.check_stale_operations()

        assert len(alerts) == 1
        assert "write" in alerts[0]
        assert "stale" in alerts[0]
        assert "possible hang" in alerts[0]

    def test_stale_fetch_detected(self):
        """A fetch (read) operation that exceeds timeout is reported as stale."""
        monitor = _make_monitor(commit_latency=5)

        monitor._timers["fetch"] = time.monotonic() - 10

        alerts = monitor.check_stale_operations()

        assert len(alerts) == 1
        assert "fetch" in alerts[0]
        assert "stale" in alerts[0]

    def test_multiple_stale_operations(self):
        """Both a stale fetch and a stale write are detected in one call."""
        monitor = _make_monitor(commit_latency=5)

        monitor._timers["fetch"] = time.monotonic() - 20
        monitor._timers["write"] = time.monotonic() - 15

        alerts = monitor.check_stale_operations()

        assert len(alerts) == 2
        ops_mentioned = " ".join(alerts)
        assert "fetch" in ops_mentioned
        assert "write" in ops_mentioned

    def test_no_stale_when_recent(self):
        """Operations that just started are not flagged as stale."""
        monitor = _make_monitor(commit_latency=300)

        monitor.start_operation("write")
        monitor.start_operation("fetch")

        alerts = monitor.check_stale_operations()

        assert alerts == []

    def test_no_stale_when_no_operations(self):
        """No timers → no stale alerts."""
        monitor = _make_monitor()

        alerts = monitor.check_stale_operations()

        assert alerts == []

    def test_custom_timeout_override(self):
        """Custom timeout overrides the commit_latency_threshold default."""
        monitor = _make_monitor(commit_latency=300)

        # 10s old — well under 300s threshold, but over 5s custom timeout
        monitor._timers["write"] = time.monotonic() - 10

        alerts_default = monitor.check_stale_operations()
        assert alerts_default == []  # not stale at 300s default

        alerts_custom = monitor.check_stale_operations(timeout_seconds=5)
        assert len(alerts_custom) == 1
        assert "write" in alerts_custom[0]

    def test_stale_alert_message_includes_elapsed_time(self):
        """Alert message shows how long the operation has been running."""
        monitor = _make_monitor(commit_latency=5)

        monitor._timers["write"] = time.monotonic() - 60

        alerts = monitor.check_stale_operations()

        assert len(alerts) == 1
        # Should mention ~60 seconds elapsed
        assert "60" in alerts[0] or "59" in alerts[0]
        assert "timeout 5s" in alerts[0]


# ---------------------------------------------------------------------------
# Test: Realistic SLA Overrun (Read + Write)
# ---------------------------------------------------------------------------

class TestSLAOverrun:
    """Realistic overrun scenarios at production-like thresholds."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_write_overrun_5min_sla(self, mock_mem):
        """Write taking 400s with 300s SLA produces actionable alert."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)

        alerts = monitor.check_alerts("write", elapsed=400.0, record_count=10000)

        assert len(alerts) == 1
        assert "write" in alerts[0]
        assert "400.0s" in alerts[0]
        assert "300s" in alerts[0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_fetch_overrun_5min_sla(self, mock_mem):
        """Fetch (read) taking 600s with 300s SLA produces actionable alert."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)

        alerts = monitor.check_alerts("fetch", elapsed=600.0, record_count=50000)

        assert len(alerts) == 1
        assert "fetch" in alerts[0]
        assert "600.0s" in alerts[0]
        assert "300s" in alerts[0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_fetch_slow_but_write_fast(self, mock_mem):
        """Read is the bottleneck while write is healthy — only fetch alerts."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)

        # Slow fetch
        fetch_alerts = monitor.check_alerts("fetch", elapsed=400.0, record_count=10000)
        # Fast write
        write_alerts = monitor.check_alerts("write", elapsed=2.0, record_count=10000)

        assert len(fetch_alerts) == 1
        assert "fetch" in fetch_alerts[0]
        assert write_alerts == []

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_write_slow_but_fetch_fast(self, mock_mem):
        """Write is the bottleneck while fetch is healthy — only write alerts."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)

        fetch_alerts = monitor.check_alerts("fetch", elapsed=5.0, record_count=10000)
        write_alerts = monitor.check_alerts("write", elapsed=350.0, record_count=10000)

        assert fetch_alerts == []
        assert len(write_alerts) == 1
        assert "write" in write_alerts[0]


# ---------------------------------------------------------------------------
# Test: Progressive Degradation / Backlog Scenarios
# ---------------------------------------------------------------------------

class TestProgressiveDegradation:
    """Simulate batches getting slower over time (backlog buildup)."""

    def test_rising_latency_reflected_in_p99(self):
        """As batches get slower, p99 rises above p50 significantly."""
        monitor = _make_monitor(window_size=100)

        # 90 fast batches (0.1s each), then 10 slow batches (5s each)
        for _ in range(90):
            monitor._latencies["write"].append(0.1)
        for _ in range(10):
            monitor._latencies["write"].append(5.0)

        stats = monitor.get_latency_stats("write")

        # p50 should still be ~0.1 (the majority)
        assert stats["p50"] < 1.0
        # p99 should capture the slow tail
        assert stats["p99"] >= 5.0

    def test_fetch_degradation_separate_from_write(self):
        """Degradation in fetch shows in fetch stats, not write stats."""
        monitor = _make_monitor(window_size=100)

        # Fetch gets progressively slower
        for i in range(20):
            monitor._latencies["fetch"].append(0.5 + i * 0.5)

        # Write stays fast
        for _ in range(20):
            monitor._latencies["write"].append(0.1)

        fetch_stats = monitor.get_latency_stats("fetch")
        write_stats = monitor.get_latency_stats("write")

        # Fetch p99 should be high (10s range)
        assert fetch_stats["p99"] > 5.0
        # Write p99 should be low
        assert write_stats["p99"] <= 0.1

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_backlog_throughput_drops(self, mock_mem):
        """Per-record extra processing causes throughput to drop below minimum."""
        monitor = _make_monitor(throughput_min=100)

        # First batch: 1000 records in 2s = 500 rec/s (healthy)
        alerts_fast = monitor.check_alerts("write", elapsed=2.0, record_count=1000)
        assert not any("throughput" in a for a in alerts_fast)

        # Backlogged batch: 1000 records in 60s = 16.7 rec/s (slow)
        alerts_slow = monitor.check_alerts("write", elapsed=60.0, record_count=1000)
        assert any("throughput" in a for a in alerts_slow)
        # Verify the alert is actionable — shows actual vs minimum
        tp_alert = [a for a in alerts_slow if "throughput" in a][0]
        assert "16.7 rec/s" in tp_alert
        assert "100 rec/s" in tp_alert


# ---------------------------------------------------------------------------
# Test: No-Data Scenarios
# ---------------------------------------------------------------------------

class TestNoDataScenarios:
    """Pipeline produces zero records — monitoring should handle gracefully."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_summary_with_zero_operations(self, mock_mem):
        """Summary for an empty run has all zeros and empty operations."""
        monitor = _make_monitor()

        summary = monitor.get_summary()

        assert summary["total_records_processed"] == 0
        assert summary["total_operations"] == 0
        assert summary["operations"] == {}

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_dump_with_zero_operations(self, mock_mem, tmp_path):
        """Dumping metrics when nothing happened produces valid empty JSON."""
        monitor = _make_monitor()
        monitor._metrics_dir = tmp_path

        path = monitor.dump_metrics()
        data = json.loads(path.read_text())

        assert data["total_records_processed"] == 0
        assert data["total_operations"] == 0
        assert data["operations"] == {}
        assert data["stale_operations"] == []

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_fetch_returns_zero_records(self, mock_mem):
        """Fetch completed but returned 0 records — no false throughput alert."""
        monitor = _make_monitor(throughput_min=100)

        alerts = monitor.check_alerts("fetch", elapsed=5.0, record_count=0)

        # Throughput alert is skipped for 0 records
        assert not any("throughput" in a for a in alerts)

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_write_zero_records(self, mock_mem):
        """Write called with 0 records (all filtered out) — no false alert."""
        monitor = _make_monitor(throughput_min=100)

        alerts = monitor.check_alerts("write", elapsed=0.5, record_count=0)

        assert not any("throughput" in a for a in alerts)

    def test_throughput_with_no_recent_operations(self):
        """get_throughput returns 0 when all entries are outside the window."""
        monitor = _make_monitor()

        # All entries are old
        old = time.time() - 300
        monitor._throughput["fetch"].append((old, 1000))
        monitor._throughput["write"].append((old, 500))

        assert monitor.get_throughput("fetch", 60) == 0.0
        assert monitor.get_throughput("write", 60) == 0.0


# ---------------------------------------------------------------------------
# Test: Alert Message Quality
# ---------------------------------------------------------------------------

class TestAlertMessageQuality:
    """Verify alert messages are actionable and diagnostic."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_latency_alert_shows_operation_name(self, mock_mem):
        """Latency alert includes which operation overran."""
        monitor = _make_monitor(commit_latency=10, throughput_min=1)

        alerts = monitor.check_alerts("fetch", elapsed=30.0, record_count=100)

        assert "fetch" in alerts[0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_latency_alert_shows_actual_vs_threshold(self, mock_mem):
        """Alert shows both the actual elapsed time and the configured threshold."""
        monitor = _make_monitor(commit_latency=300, throughput_min=1)

        alerts = monitor.check_alerts("write", elapsed=450.0, record_count=10000)

        assert "450.0s" in alerts[0]
        assert "300s" in alerts[0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=500.0)
    def test_throughput_alert_shows_actual_vs_minimum(self, mock_mem):
        """Throughput alert shows the actual rate and the minimum threshold."""
        monitor = _make_monitor(throughput_min=500)

        alerts = monitor.check_alerts("write", elapsed=10.0, record_count=100)

        tp_alert = [a for a in alerts if "throughput" in a][0]
        assert "10.0 rec/s" in tp_alert
        assert "500 rec/s" in tp_alert

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=1500.0)
    def test_memory_alert_shows_actual_vs_threshold(self, mock_mem):
        """Memory alert shows the actual usage and the configured limit."""
        monitor = _make_monitor(memory_mb=1000)

        alerts = monitor.check_alerts("write", elapsed=1.0, record_count=100)

        mem_alert = [a for a in alerts if "Memory" in a][0]
        assert "1500" in mem_alert
        assert "1000" in mem_alert

    def test_stale_alert_shows_operation_and_elapsed(self):
        """Stale alert names the operation and how long it's been hung."""
        monitor = _make_monitor(commit_latency=5)
        monitor._timers["fetch"] = time.monotonic() - 120

        alerts = monitor.check_stale_operations()

        assert "fetch" in alerts[0]
        assert "120" in alerts[0] or "119" in alerts[0]
        assert "timeout 5s" in alerts[0]
        assert "possible hang" in alerts[0]


# ---------------------------------------------------------------------------
# Test: Metrics Dump with Stale Operations
# ---------------------------------------------------------------------------

class TestDumpWithStaleOps:
    """Verify metrics dump captures stale operations."""

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_dump_includes_stale_alerts(self, mock_mem, tmp_path):
        """If a timer is stale at dump time, stale_operations is populated."""
        monitor = _make_monitor(commit_latency=5)
        monitor._metrics_dir = tmp_path

        # Simulate a hung write
        monitor._timers["write"] = time.monotonic() - 60

        path = monitor.dump_metrics()
        data = json.loads(path.read_text())

        assert len(data["stale_operations"]) == 1
        assert "write" in data["stale_operations"][0]

    @patch.object(PerformanceMonitor, "get_memory_usage_mb", return_value=200.0)
    def test_dump_no_stale_when_clean(self, mock_mem, tmp_path):
        """With no in-flight timers, stale_operations is empty."""
        monitor = _make_monitor()
        monitor._metrics_dir = tmp_path

        path = monitor.dump_metrics()
        data = json.loads(path.read_text())

        assert data["stale_operations"] == []
