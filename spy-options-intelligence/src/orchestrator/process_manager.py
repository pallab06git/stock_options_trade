# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Process manager for controlling worker subprocesses.

Provides start, stop, and status queries for worker processes
tracked by the ParallelRunner's process registry.
"""

import os
import signal
from typing import Any, Dict, List

import psutil

from src.orchestrator.parallel_runner import ParallelRunner
from src.utils.logger import get_logger

logger = get_logger()


class ProcessManager:
    """Start, stop, and query worker processes via the registry."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def list_workers(self) -> List[Dict[str, Any]]:
        """Read registry, check each PID is alive, return status list.

        Returns:
            List of dicts with ticker, pid, started_at, and alive status.
        """
        registry = ParallelRunner.load_registry()
        workers = []
        for ticker, info in registry.items():
            pid = info.get("pid", 0)
            alive = psutil.pid_exists(pid) if pid else False
            workers.append({
                "ticker": ticker,
                "pid": pid,
                "started_at": info.get("started_at", ""),
                "alive": alive,
                "status": info.get("status", "unknown"),
            })
        return workers

    def stop_worker(self, ticker: str) -> bool:
        """Send SIGTERM to a specific worker by ticker name.

        Args:
            ticker: Ticker symbol of the worker to stop.

        Returns:
            True if signal was sent, False if ticker not found or process dead.
        """
        registry = ParallelRunner.load_registry()
        if ticker not in registry:
            logger.warning(f"No worker found for ticker '{ticker}'")
            return False

        pid = registry[ticker].get("pid", 0)
        if not pid or not psutil.pid_exists(pid):
            logger.warning(f"Worker {ticker} (PID {pid}) is not running")
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            registry[ticker]["status"] = "stopped"
            ParallelRunner.REGISTRY_PATH.write_text(
                __import__("json").dumps(registry, indent=2)
            )
            logger.info(f"Sent SIGTERM to worker {ticker} (PID {pid})")
            return True
        except (ProcessLookupError, PermissionError) as e:
            logger.error(f"Failed to stop worker {ticker} (PID {pid}): {e}")
            return False

    def stop_all(self) -> Dict[str, bool]:
        """Send SIGTERM to all running workers.

        Returns:
            Dict mapping ticker to success/failure of stop signal.
        """
        registry = ParallelRunner.load_registry()
        results = {}
        for ticker in registry:
            results[ticker] = self.stop_worker(ticker)
        return results
