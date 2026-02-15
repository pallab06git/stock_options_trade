# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Parallel runner for multi-ticker backfill.

Spawns one subprocess per ticker for parallel historical data ingestion.
Tracks processes in a JSON registry for management and monitoring.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()


class ParallelRunner:
    """Spawn one subprocess per ticker for parallel backfill.

    Each subprocess runs ``python -m src.cli backfill --ticker <TICKER>``
    with a proportional share of the rate limit budget. Process PIDs are
    tracked in a JSON registry file for management via ProcessManager.
    """

    REGISTRY_PATH = Path("data/logs/process_registry.json")

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged configuration dict. Reads tickers from
                    ``config["orchestrator"]["tickers"]`` and rate limit
                    from ``config["polygon"]["rate_limiting"]``.
        """
        self.config = config
        self.tickers: List[str] = config.get("orchestrator", {}).get("tickers", ["SPY"])
        total_rate = (
            config.get("polygon", {})
            .get("rate_limiting", {})
            .get("total_requests_per_minute", 5)
        )
        self.per_worker_rate = max(1, total_rate / len(self.tickers))

    def run(
        self,
        start_date: str,
        end_date: str,
        resume: bool = False,
        config_dir: str = "config",
    ) -> Dict[str, Dict[str, Any]]:
        """Spawn subprocesses, register PIDs, wait for results.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            resume: If True, pass --resume to each subprocess.
            config_dir: Path to config directory.

        Returns:
            Dict mapping ticker to result info (exit_code, stdout, stderr).
        """
        processes: Dict[str, subprocess.Popen] = {}
        registry: Dict[str, Dict[str, Any]] = {}

        for ticker in self.tickers:
            cmd = [
                sys.executable, "-m", "src.cli", "backfill",
                "--ticker", ticker,
                "--config-dir", config_dir,
                "--start-date", start_date,
                "--end-date", end_date,
                "--rate-limit", str(self.per_worker_rate),
            ]
            if resume:
                cmd.append("--resume")

            logger.info(f"Spawning worker for {ticker}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            processes[ticker] = proc
            registry[ticker] = {
                "pid": proc.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "command": cmd,
                "status": "running",
            }

        self._save_registry(registry)
        logger.info(
            f"ParallelRunner: {len(processes)} workers spawned, "
            f"rate limit {self.per_worker_rate:.1f} req/min each"
        )

        # Wait for all subprocesses and collect results
        results: Dict[str, Dict[str, Any]] = {}
        for ticker, proc in processes.items():
            stdout, stderr = proc.communicate()
            exit_code = proc.returncode
            registry[ticker]["status"] = "completed" if exit_code == 0 else "failed"
            registry[ticker]["exit_code"] = exit_code
            registry[ticker]["finished_at"] = datetime.now(timezone.utc).isoformat()
            results[ticker] = {
                "exit_code": exit_code,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
            if exit_code == 0:
                logger.info(f"Worker {ticker} completed successfully")
            else:
                logger.error(f"Worker {ticker} failed with exit code {exit_code}")

        self._save_registry(registry)
        return results

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Persist registry to JSON file."""
        self.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    @classmethod
    def load_registry(cls) -> Dict[str, Any]:
        """Load the process registry from disk.

        Returns:
            Dict mapping ticker to process info. Empty dict if no registry.
        """
        if cls.REGISTRY_PATH.exists():
            try:
                return json.loads(cls.REGISTRY_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
