# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unified health dashboard for all worker sessions.

Aggregates per-session metrics files and process registry into a
single view for monitoring all running or completed workers.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from src.orchestrator.parallel_runner import ParallelRunner
from src.utils.logger import get_logger

logger = get_logger()


class HealthDashboard:
    """Aggregate metrics from all per-session monitoring files.

    Scans ``data/logs/performance/`` for per-session JSON files
    (named ``metrics_{session_label}_{timestamp}.json``) and combines
    them with the process registry for a unified health view.
    """

    def __init__(self, metrics_dir: str = "data/logs/performance"):
        self.metrics_dir = Path(metrics_dir)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Scan metrics dir for latest per-session JSON files.

        Returns:
            Dict keyed by session_label with latest metrics snapshot.
            Filenames follow pattern: metrics_{session_label}_{YYYY-MM-DD_HHMMSS}.json
        """
        sessions: Dict[str, Dict[str, Any]] = {}
        if not self.metrics_dir.exists():
            return sessions

        for f in sorted(self.metrics_dir.glob("metrics_*.json"), reverse=True):
            # Filename: metrics_{label}_{YYYY-MM-DD}_{HHMMSS}.json
            stem = f.stem  # e.g. "metrics_spy_2025-01-27_143000"
            parts = stem.split("_", 1)  # ["metrics", "spy_2025-01-27_143000"]
            if len(parts) < 2:
                continue

            label_and_ts = parts[1]
            # Label is everything before the last two _-separated segments (date + time)
            segments = label_and_ts.rsplit("_", 2)
            if len(segments) >= 3:
                label = "_".join(segments[:-2])
            else:
                label = label_and_ts

            if label and label not in sessions:
                try:
                    sessions[label] = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

        return sessions

    def get_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Combine process registry + latest metrics into unified view.

        Returns:
            Dict keyed by ticker with process info and metrics.
        """
        registry = ParallelRunner.load_registry()
        sessions = self.get_all_sessions()

        summary: Dict[str, Dict[str, Any]] = {}

        # Merge: for each ticker in registry, attach its metrics
        for ticker, proc_info in registry.items():
            label = ticker.lower()
            pid = proc_info.get("pid", 0)
            alive = psutil.pid_exists(pid) if pid else False
            metrics = sessions.get(label, {})
            summary[ticker] = {
                "pid": pid,
                "alive": alive,
                "started_at": proc_info.get("started_at", ""),
                "status": proc_info.get("status", "unknown"),
                "total_records": metrics.get("total_records_processed", 0),
                "total_operations": metrics.get("total_operations", 0),
                "memory_mb": metrics.get("memory_usage_mb", 0),
                "stale_operations": metrics.get("stale_operations", []),
                "operations": metrics.get("operations", {}),
            }

        # Also include sessions not in registry (e.g. single-ticker runs)
        for label, metrics in sessions.items():
            ticker_upper = label.upper()
            if ticker_upper not in summary:
                summary[ticker_upper] = {
                    "pid": None,
                    "alive": False,
                    "started_at": "",
                    "status": "no_process",
                    "total_records": metrics.get("total_records_processed", 0),
                    "total_operations": metrics.get("total_operations", 0),
                    "memory_mb": metrics.get("memory_usage_mb", 0),
                    "stale_operations": metrics.get("stale_operations", []),
                    "operations": metrics.get("operations", {}),
                }

        return summary

    def get_session_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a single session.

        Args:
            ticker: Ticker symbol (case-insensitive).

        Returns:
            Detailed metrics dict or None if not found.
        """
        sessions = self.get_all_sessions()
        return sessions.get(ticker.lower())

    def format_table(self, summary: Dict[str, Dict[str, Any]]) -> str:
        """Format the health summary as a text table.

        Args:
            summary: Output from get_health_summary().

        Returns:
            Formatted table string.
        """
        if not summary:
            return "No sessions found."

        header = (
            f"{'Ticker':<8} {'PID':<8} {'Status':<12} {'Records':<10} "
            f"{'Ops':<6} {'Memory':<10} {'Stale':<6}"
        )
        separator = "-" * len(header)
        lines = [header, separator]

        for ticker, info in sorted(summary.items()):
            status = "running" if info["alive"] else info.get("status", "stopped")
            pid = str(info["pid"] or "-")
            mem = f"{info['memory_mb']:.0f} MB" if info["memory_mb"] else "-"
            stale = str(len(info["stale_operations"]))
            lines.append(
                f"{ticker:<8} {pid:<8} {status:<12} "
                f"{info['total_records']:<10} {info['total_operations']:<6} "
                f"{mem:<10} {stale:<6}"
            )

        return "\n".join(lines)
