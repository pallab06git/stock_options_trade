# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Error aggregation with sliding-window rate tracking and alerting.

Collects errors by type, computes error rates over a configurable
sliding window, and generates summary reports for operational analysis.
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()


class ErrorAggregator:
    """Aggregate errors and alert when the error rate exceeds a threshold.

    Errors are grouped by type and tracked in a sliding time window.
    When the error rate (errors / total operations) exceeds the
    configured threshold within the window, an alert is raised.

    Usage::

        agg = ErrorAggregator(config)
        agg.record_error("validation_error", "negative price field: open=-1.0")
        agg.record_success()  # count a successful operation
        if agg.should_alert():
            logger.warning("Error rate threshold exceeded!")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged configuration dict.  Reads thresholds from
                    ``config["monitoring"]["performance"]``.
        """
        perf = config.get("monitoring", {}).get("performance", {})

        # Configurable thresholds
        self.error_rate_threshold = perf.get("error_rate_percent", 1.0)
        self.error_window_seconds = perf.get("error_window_minutes", 15) * 60

        # Sliding window: deque of (timestamp, is_error) tuples
        self._events: deque = deque()

        # Cumulative error counts by type
        self._error_counts: Dict[str, int] = defaultdict(int)

        # Recent error messages for reporting (keep last N per type)
        self._recent_errors: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Cumulative totals
        self.total_errors = 0
        self.total_successes = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_error(self, error_type: str, error_msg: str) -> None:
        """Record an error occurrence.

        Args:
            error_type: Category key (e.g. "validation_error", "write_error").
            error_msg: Human-readable description.
        """
        now = time.time()
        self._events.append((now, True))
        self._error_counts[error_type] += 1
        self._recent_errors[error_type].append((now, error_msg))
        self.total_errors += 1

    def record_success(self) -> None:
        """Record a successful operation (used to compute error rate)."""
        self._events.append((time.time(), False))
        self.total_successes += 1

    # ------------------------------------------------------------------
    # Error Rate
    # ------------------------------------------------------------------

    def get_error_rate(self, window_seconds: Optional[float] = None) -> float:
        """Compute error rate as a percentage over the sliding window.

        Args:
            window_seconds: Override the default window.  If None, uses
                            the configured ``error_window_minutes``.

        Returns:
            Error rate as a percentage (0.0 – 100.0).  Returns 0.0 if
            no events exist in the window.
        """
        window = window_seconds if window_seconds is not None else self.error_window_seconds
        self._prune_window(window)

        total = len(self._events)
        if total == 0:
            return 0.0

        errors = sum(1 for _, is_err in self._events if is_err)
        return (errors / total) * 100.0

    def should_alert(self) -> bool:
        """Check if the error rate exceeds the configured threshold.

        Returns:
            True if the current error rate within the sliding window
            exceeds ``error_rate_percent``.
        """
        rate = self.get_error_rate()
        if rate > self.error_rate_threshold:
            logger.warning(
                f"Error rate {rate:.1f}% exceeds threshold "
                f"{self.error_rate_threshold}%"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_error_summary(self) -> Dict[str, Any]:
        """Return grouped error counts and current rate.

        Returns:
            Dict with ``error_counts`` (by type), ``total_errors``,
            ``total_successes``, and ``current_rate_percent``.
        """
        return {
            "error_counts": dict(self._error_counts),
            "total_errors": self.total_errors,
            "total_successes": self.total_successes,
            "current_rate_percent": round(self.get_error_rate(), 2),
        }

    def get_recent_errors(self, error_type: Optional[str] = None) -> List[Tuple[float, str]]:
        """Return recent error messages.

        Args:
            error_type: Filter to a specific type.  If None, returns
                        errors across all types sorted by timestamp.

        Returns:
            List of (timestamp, message) tuples.
        """
        if error_type is not None:
            return list(self._recent_errors.get(error_type, []))

        all_errors: List[Tuple[float, str]] = []
        for msgs in self._recent_errors.values():
            all_errors.extend(msgs)
        all_errors.sort(key=lambda x: x[0])
        return all_errors

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_window(self, window_seconds: float) -> None:
        """Remove events older than the sliding window."""
        cutoff = time.time() - window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
