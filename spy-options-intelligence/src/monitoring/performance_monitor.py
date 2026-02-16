# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Performance monitoring with latency tracking, throughput, and alerting.

Tracks operation latencies in a rolling window, computes percentiles
(p50, p95, p99), monitors throughput and memory usage, and raises
alerts when configurable thresholds are exceeded.  Periodically dumps
metrics to a JSON file under data/logs/performance/.
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from src.utils.logger import get_logger

logger = get_logger()


class PerformanceMonitor:
    """Track pipeline performance metrics and alert on threshold breaches.

    Usage::

        monitor = PerformanceMonitor(config)
        monitor.start_operation("write")
        sink.write_batch(records, key)
        alerts = monitor.end_operation("write", len(records))
        if alerts:
            for alert in alerts:
                logger.warning(alert)
    """

    def __init__(self, config: Dict[str, Any], session_label: str = "default"):
        """
        Args:
            config: Full merged configuration dict.  Reads thresholds from
                    ``config["monitoring"]["performance"]``.
            session_label: Label for this monitoring session (e.g. ticker name).
                           Used in metrics dump filenames.
        """
        self.session_label = session_label
        perf = config.get("monitoring", {}).get("performance", {})

        # Configurable thresholds
        self.commit_latency_threshold = perf.get("commit_latency_seconds", 300)
        self.throughput_min = perf.get("throughput_min_records_per_sec", 100)
        self.memory_threshold_mb = perf.get("memory_usage_mb_threshold", 1000)
        self.metrics_dump_interval = perf.get("metrics_dump_interval_minutes", 15) * 60
        self.latency_window_size = perf.get("latency_window_size", 100)

        # Metrics directory
        self._metrics_dir = Path("data/logs/performance")

        # Rolling latency windows keyed by operation name
        self._latencies: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.latency_window_size)
        )

        # Throughput tracking: list of (timestamp, record_count) per operation
        self._throughput: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        # In-flight timers: operation_name → start_time
        self._timers: Dict[str, float] = {}

        # Cumulative counters
        self.total_records_processed = 0
        self.total_operations = 0

        # Dump tracking
        self._last_dump_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Timer API
    # ------------------------------------------------------------------

    def start_operation(self, operation: str) -> None:
        """Begin timing an operation.

        Args:
            operation: Name of the operation (e.g. "write", "validate").
        """
        self._timers[operation] = time.monotonic()

    def end_operation(
        self, operation: str, record_count: int = 0
    ) -> List[str]:
        """Stop the timer for *operation* and record metrics.

        Args:
            operation: Must match a previous ``start_operation`` call.
            record_count: Number of records processed in this operation.

        Returns:
            List of alert messages (empty if no thresholds breached).
        """
        start = self._timers.pop(operation, None)
        if start is None:
            logger.warning(f"end_operation called without start for '{operation}'")
            return []

        elapsed = time.monotonic() - start
        self._latencies[operation].append(elapsed)
        self._throughput[operation].append((time.time(), record_count))
        self._prune_throughput(operation)
        self.total_records_processed += record_count
        self.total_operations += 1

        return self.check_alerts(operation, elapsed, record_count)

    def check_stale_operations(self, timeout_seconds: Optional[float] = None) -> List[str]:
        """Detect operations that started but never finished (hung).

        Compares each in-flight timer against the current monotonic clock.
        If an operation has been running longer than *timeout_seconds*
        (defaults to ``commit_latency_threshold``), it is reported as stale.

        Args:
            timeout_seconds: Override the staleness threshold.  Defaults to
                             the configured commit latency threshold.

        Returns:
            List of alert messages for stale operations.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.commit_latency_threshold
        now = time.monotonic()
        alerts: List[str] = []

        for operation, start in self._timers.items():
            elapsed = now - start
            if elapsed > timeout:
                alerts.append(
                    f"ALERT: {operation} appears stale — started {elapsed:.1f}s ago, "
                    f"exceeds timeout {timeout:.0f}s (possible hang)"
                )

        for alert in alerts:
            logger.warning(alert)

        return alerts

    # ------------------------------------------------------------------
    # Alert Checks
    # ------------------------------------------------------------------

    def check_alerts(
        self,
        operation: str,
        elapsed: float,
        record_count: int,
    ) -> List[str]:
        """Evaluate alert thresholds for a completed operation.

        Args:
            operation: Operation name.
            elapsed: Duration in seconds.
            record_count: Records processed.

        Returns:
            List of human-readable alert strings.
        """
        alerts: List[str] = []

        # 1. Commit latency
        if elapsed > self.commit_latency_threshold:
            alerts.append(
                f"ALERT: {operation} latency {elapsed:.1f}s exceeds "
                f"threshold {self.commit_latency_threshold}s"
            )

        # 2. Throughput
        if record_count > 0 and elapsed > 0:
            throughput = record_count / elapsed
            if throughput < self.throughput_min:
                alerts.append(
                    f"ALERT: {operation} throughput {throughput:.1f} rec/s "
                    f"below minimum {self.throughput_min} rec/s"
                )

        # 3. Memory
        mem_mb = self.get_memory_usage_mb()
        if mem_mb > self.memory_threshold_mb:
            alerts.append(
                f"ALERT: Memory usage {mem_mb:.0f} MB exceeds "
                f"threshold {self.memory_threshold_mb} MB"
            )

        for alert in alerts:
            logger.warning(alert)

        return alerts

    # ------------------------------------------------------------------
    # Latency Percentiles
    # ------------------------------------------------------------------

    def get_latency_stats(self, operation: str) -> Dict[str, Optional[float]]:
        """Compute p50, p95, p99 latency for an operation.

        Args:
            operation: Operation name.

        Returns:
            Dict with keys ``p50``, ``p95``, ``p99`` (None if no data).
        """
        window = list(self._latencies.get(operation, []))
        if not window:
            return {"p50": None, "p95": None, "p99": None}

        window_sorted = sorted(window)
        return {
            "p50": self._percentile(window_sorted, 50),
            "p95": self._percentile(window_sorted, 95),
            "p99": self._percentile(window_sorted, 99),
        }

    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Compute the *pct*-th percentile from pre-sorted data.

        Uses nearest-rank method.
        """
        if not sorted_data:
            return 0.0
        k = max(0, int(len(sorted_data) * pct / 100) - 1)
        return sorted_data[min(k, len(sorted_data) - 1)]

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    def _prune_throughput(self, operation: str, window_seconds: float = 3600.0) -> None:
        """Remove throughput entries older than *window_seconds*.

        Always keeps at least one entry per operation so that
        ``get_throughput`` can still return a meaningful value.
        """
        entries = self._throughput.get(operation)
        if not entries or len(entries) <= 1:
            return
        cutoff = time.time() - window_seconds
        # Find the first index that is within the window
        first_valid = 0
        for i, (ts, _) in enumerate(entries):
            if ts >= cutoff:
                first_valid = i
                break
        else:
            # All entries are old — keep the last one
            first_valid = len(entries) - 1
        if first_valid > 0:
            del entries[:first_valid]

    def get_throughput(self, operation: str, window_seconds: float = 60.0) -> float:
        """Average throughput (records/second) over a recent time window.

        Args:
            operation: Operation name.
            window_seconds: How far back to look (default 60 s).

        Returns:
            Records per second (0.0 if no data in window).
        """
        self._prune_throughput(operation)
        entries = self._throughput.get(operation, [])
        if not entries:
            return 0.0

        cutoff = time.time() - window_seconds
        recent = [(ts, cnt) for ts, cnt in entries if ts >= cutoff]
        if not recent:
            return 0.0

        total_records = sum(cnt for _, cnt in recent)
        span = time.time() - recent[0][0]
        if span <= 0:
            return float(total_records)
        return total_records / span

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_memory_usage_mb(self) -> float:
        """Current process RSS in megabytes."""
        return psutil.Process().memory_info().rss / (1024 * 1024)

    # ------------------------------------------------------------------
    # Metrics Dump
    # ------------------------------------------------------------------

    def should_dump_metrics(self) -> bool:
        """True if the dump interval has elapsed since last dump."""
        if self._last_dump_time is None:
            return True
        return (time.time() - self._last_dump_time) >= self.metrics_dump_interval

    def dump_metrics(self) -> Path:
        """Write current metrics snapshot to a JSON file.

        Returns:
            Path to the written metrics file.
        """
        self._metrics_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.utcnow()
        filename = f"metrics_{self.session_label}_{now.strftime('%Y-%m-%d_%H%M%S')}.json"
        path = self._metrics_dir / filename

        # Detect any stale in-flight operations
        stale_alerts = self.check_stale_operations()

        snapshot = {
            "timestamp": now.isoformat() + "Z",
            "total_records_processed": self.total_records_processed,
            "total_operations": self.total_operations,
            "memory_usage_mb": round(self.get_memory_usage_mb(), 1),
            "stale_operations": stale_alerts,
            "operations": {},
        }

        for op_name in self._latencies:
            stats = self.get_latency_stats(op_name)
            snapshot["operations"][op_name] = {
                "latency_p50": round(stats["p50"], 4) if stats["p50"] is not None else None,
                "latency_p95": round(stats["p95"], 4) if stats["p95"] is not None else None,
                "latency_p99": round(stats["p99"], 4) if stats["p99"] is not None else None,
                "samples": len(self._latencies[op_name]),
                "throughput_60s": round(self.get_throughput(op_name, 60.0), 1),
            }

        path.write_text(json.dumps(snapshot, indent=2))
        self._last_dump_time = time.time()

        logger.debug(f"Metrics dumped to {path}")
        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for logging at end of a run."""
        summary: Dict[str, Any] = {
            "total_records_processed": self.total_records_processed,
            "total_operations": self.total_operations,
            "memory_usage_mb": round(self.get_memory_usage_mb(), 1),
            "operations": {},
        }
        for op_name in self._latencies:
            summary["operations"][op_name] = self.get_latency_stats(op_name)
        return summary
