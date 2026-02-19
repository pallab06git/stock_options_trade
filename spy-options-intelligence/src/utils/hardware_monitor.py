# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Hardware resource monitor using psutil.

Snapshots CPU, memory, and disk I/O at the start and end of a command
to produce a per-run JSON record. Also provides a @track_hardware
decorator for CLI commands.

Output: data/reports/hardware/{YYYY-MM-DD}_{command_name}.json
Schema:
  {
    "command": str,
    "start_time": str (ISO 8601),
    "end_time":   str (ISO 8601),
    "elapsed_sec": float,
    "cpu_pct_avg": float,
    "mem_rss_start_mb": float,
    "mem_rss_end_mb": float,
    "mem_delta_mb": float,
    "disk_read_mb": float,
    "disk_write_mb": float
  }
"""

import functools
import json
import time
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import psutil

from src.utils.logger import get_logger

logger = get_logger()


class HardwareMonitor:
    """Monitor CPU, memory, and disk I/O for a CLI command.

    Usage:
        monitor = HardwareMonitor(config)
        monitor.start("engineer-features")
        # ... run work ...
        result = monitor.stop()

    Or via the decorator:
        @HardwareMonitor.track_hardware("download-minute")
        def my_fn():
            ...
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged config dict.
        """
        v2 = config.get("pipeline_v2", {})
        hw_cfg = v2.get("hardware", {})
        self.enabled: bool = hw_cfg.get("enabled", True)
        self.output_dir = Path(
            hw_cfg.get("output_dir", "data/reports/hardware")
        )

        self._proc: Optional[psutil.Process] = None
        self._command_name: Optional[str] = None
        self._start_time: Optional[float] = None
        self._start_dt: Optional[datetime] = None
        self._mem_start: float = 0.0
        self._disk_read_start: int = 0
        self._disk_write_start: int = 0
        self._cpu_samples: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, command_name: str) -> None:
        """Take a baseline snapshot of CPU, memory, and disk I/O.

        Args:
            command_name: Human-readable name for the command being tracked.
        """
        if not self.enabled:
            return

        self._command_name = command_name
        self._proc = psutil.Process()
        self._start_time = time.monotonic()
        self._start_dt = datetime.now(tz=timezone.utc)

        self._mem_start = self._proc.memory_info().rss / (1024 ** 2)

        counters = self._disk_counters()
        self._disk_read_start = counters.read_bytes if counters else 0
        self._disk_write_start = counters.write_bytes if counters else 0

        # First CPU sample (interval=None returns instantaneous)
        self._cpu_samples = [self._proc.cpu_percent(interval=None)]

        logger.debug(
            f"HardwareMonitor started for '{command_name}' "
            f"(mem_start={self._mem_start:.1f} MB)"
        )

    def stop(self) -> Dict[str, Any]:
        """Compute deltas, write JSON, and return the metrics dict.

        Returns:
            Metrics dict. Empty dict if monitoring is disabled or start()
            was never called.
        """
        if not self.enabled or self._proc is None:
            return {}

        end_dt = datetime.now(tz=timezone.utc)
        elapsed = time.monotonic() - self._start_time

        mem_end = self._proc.memory_info().rss / (1024 ** 2)

        # Final CPU sample
        self._cpu_samples.append(self._proc.cpu_percent(interval=None))
        cpu_avg = (
            sum(s for s in self._cpu_samples if s is not None) / len(self._cpu_samples)
            if self._cpu_samples
            else 0.0
        )

        counters = self._disk_counters()
        disk_read = (
            (counters.read_bytes - self._disk_read_start) / (1024 ** 2)
            if counters else 0.0
        )
        disk_write = (
            (counters.write_bytes - self._disk_write_start) / (1024 ** 2)
            if counters else 0.0
        )

        metrics: Dict[str, Any] = {
            "command": self._command_name,
            "start_time": self._start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "cpu_pct_avg": round(cpu_avg, 2),
            "mem_rss_start_mb": round(self._mem_start, 2),
            "mem_rss_end_mb": round(mem_end, 2),
            "mem_delta_mb": round(mem_end - self._mem_start, 2),
            "disk_read_mb": round(max(disk_read, 0.0), 3),
            "disk_write_mb": round(max(disk_write, 0.0), 3),
        }

        self._write_json(metrics)
        logger.info(
            f"HardwareMonitor '{self._command_name}': "
            f"{elapsed:.1f}s, CPU avg {cpu_avg:.1f}%, "
            f"mem +{metrics['mem_delta_mb']:.1f} MB"
        )
        return metrics

    def daily_summary(self, date: Optional[str] = None) -> pd.DataFrame:
        """Load all hardware JSONs for a given date.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            DataFrame with one row per command run.
        """
        if date is None:
            date = date_cls.today().strftime("%Y-%m-%d")

        if not self.output_dir.exists():
            return pd.DataFrame()

        records: List[Dict] = []
        for path in sorted(self.output_dir.glob(f"{date}_*.json")):
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except Exception as exc:
                logger.warning(f"Could not load hardware JSON {path}: {exc}")

        return pd.DataFrame(records) if records else pd.DataFrame()

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @staticmethod
    def track_hardware(command_name: str):
        """Decorator factory: wraps a function with start/stop monitoring.

        Usage:
            @HardwareMonitor.track_hardware("my-command")
            def run_something(monitor, *args, **kwargs):
                ...

        The wrapped function receives the HardwareMonitor instance as its
        first positional argument only if it is listed in the signature
        with the name 'monitor'. Otherwise no extra argument is injected
        and monitoring runs silently around the call.

        In practice CLI commands do not take a 'monitor' argument, so
        the decorator just instruments start/stop transparently.
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Build a minimal config from kwargs or empty
                config = kwargs.get("config", {})
                hw_monitor = HardwareMonitor(config)
                hw_monitor.start(command_name)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    hw_monitor.stop()
                return result
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _disk_counters(self):
        """Return per-process disk I/O counters (None if unavailable)."""
        try:
            return self._proc.io_counters()
        except (psutil.AccessDenied, AttributeError, NotImplementedError):
            # io_counters() is not available on macOS without root
            return None

    def _write_json(self, metrics: Dict[str, Any]) -> None:
        """Persist metrics dict to a dated JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use local date for filename (filesystem organization by local calendar day)
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_cmd = (self._command_name or "unknown").replace(" ", "_").replace("/", "_")
        out_path = self.output_dir / f"{date_str}_{safe_cmd}.json"

        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.debug(f"Hardware metrics written to {out_path}")
