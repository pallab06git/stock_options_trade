# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Config-driven data purge manager for date-partitioned files.

Scans project data directories and deletes files older than the
configured retention period (in days).  Supports per-category
retention, dry-run mode, and graceful error handling.
"""

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()

# Default retention in days per category (0 = disabled)
_DEFAULT_RETENTION: Dict[str, int] = {
    "raw_data": 3,
    "processed_data": 3,
    "performance_metrics": 3,
    "schema_drift": 7,
    "checkpoints": 7,
    "heartbeat": 1,
}

# Category → (directories, file pattern or None for all files)
_CATEGORY_PATHS: Dict[str, List[Dict[str, Any]]] = {
    "raw_data": [
        {"dir": "data/raw/spy", "pattern": None},
        {"dir": "data/raw/vix", "pattern": None},
        {"dir": "data/raw/options", "pattern": None},
        {"dir": "data/raw/options/contracts", "pattern": None},
        {"dir": "data/raw/news", "pattern": None},
    ],
    "processed_data": [
        {"dir": "data/processed/consolidated", "pattern": None},
        {"dir": "data/processed/training", "pattern": None},
    ],
    "performance_metrics": [
        {"dir": "data/logs/performance", "pattern": None},
    ],
    "schema_drift": [
        {"dir": "data/logs/schema/drift", "pattern": None},
    ],
    "checkpoints": [
        {"dir": "data/logs/execution", "pattern": "checkpoint_*.json"},
    ],
    "heartbeat": [
        {"dir": "data/logs/heartbeat", "pattern": None},
    ],
}


class PurgeManager:
    """Delete old data files based on configurable per-category retention.

    Usage::

        pm = PurgeManager(config)
        summary = pm.purge_all(dry_run=True)   # preview
        summary = pm.purge_all(dry_run=False)   # delete
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged configuration dict.  Reads retention
                    settings from ``config["retention"]``.
        """
        retention_cfg = config.get("retention", {})
        self._retention: Dict[str, int] = {}
        for category in _CATEGORY_PATHS:
            self._retention[category] = retention_cfg.get(
                category, _DEFAULT_RETENTION.get(category, 0)
            )

    def purge_category(
        self,
        category: str,
        retention_days_override: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Purge files in a single category older than retention.

        Args:
            category: Category name (e.g. "raw_data", "heartbeat").
            retention_days_override: Override config retention for this run.
            dry_run: If True, report but do not delete.

        Returns:
            Summary dict with files_scanned, files_purged, bytes_freed,
            files_failed.
        """
        retention_days = (
            retention_days_override
            if retention_days_override is not None
            else self._retention.get(category, 0)
        )

        if retention_days == 0:
            return {
                "category": category,
                "retention_days": 0,
                "files_scanned": 0,
                "files_purged": 0,
                "bytes_freed": 0,
                "files_failed": 0,
                "skipped": True,
            }

        paths_cfg = _CATEGORY_PATHS.get(category, [])
        if not paths_cfg:
            logger.warning(f"Unknown purge category: {category}")
            return {
                "category": category,
                "retention_days": retention_days,
                "files_scanned": 0,
                "files_purged": 0,
                "bytes_freed": 0,
                "files_failed": 0,
            }

        import time

        cutoff = time.time() - (retention_days * 86400)

        files_scanned = 0
        files_purged = 0
        bytes_freed = 0
        files_failed = 0

        for path_entry in paths_cfg:
            dir_path = Path(path_entry["dir"])
            pattern = path_entry.get("pattern")

            if not dir_path.exists():
                continue

            for entry in dir_path.iterdir():
                if not entry.is_file():
                    continue

                # Apply pattern filter if specified
                if pattern and not fnmatch.fnmatch(entry.name, pattern):
                    continue

                files_scanned += 1

                try:
                    mtime = os.path.getmtime(entry)
                except OSError:
                    files_failed += 1
                    continue

                if mtime < cutoff:
                    file_size = entry.stat().st_size
                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete: {entry} ({file_size} bytes)")
                    else:
                        try:
                            entry.unlink()
                            logger.info(f"Deleted: {entry} ({file_size} bytes)")
                        except PermissionError:
                            logger.warning(f"Permission denied: {entry}")
                            files_failed += 1
                            continue
                        except OSError as e:
                            logger.warning(f"Failed to delete {entry}: {e}")
                            files_failed += 1
                            continue
                    files_purged += 1
                    bytes_freed += file_size

        return {
            "category": category,
            "retention_days": retention_days,
            "files_scanned": files_scanned,
            "files_purged": files_purged,
            "bytes_freed": bytes_freed,
            "files_failed": files_failed,
        }

    def purge_all(
        self,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Purge all categories according to configured retention.

        Args:
            dry_run: If True, report but do not delete.

        Returns:
            Aggregated summary with per-category breakdowns.
        """
        total_scanned = 0
        total_purged = 0
        total_bytes = 0
        total_failed = 0
        categories: Dict[str, Dict[str, Any]] = {}

        for category in _CATEGORY_PATHS:
            result = self.purge_category(category, dry_run=dry_run)
            categories[category] = result
            total_scanned += result["files_scanned"]
            total_purged += result["files_purged"]
            total_bytes += result["bytes_freed"]
            total_failed += result["files_failed"]

        return {
            "files_scanned": total_scanned,
            "files_purged": total_purged,
            "bytes_freed": total_bytes,
            "files_failed": total_failed,
            "categories": categories,
            "dry_run": dry_run,
        }
