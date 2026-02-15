# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Schema drift detection for Parquet data sources.

Extracts column schemas from Parquet metadata (no data loading), stores
baselines as JSON, and detects new/missing columns or type changes.
Drift triggers warnings but does not block processing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq

from src.utils.logger import get_logger

logger = get_logger()


class SchemaMonitor:
    """Detect schema drift between Parquet files and stored baselines.

    Usage::

        monitor = SchemaMonitor(config)
        alerts = monitor.check_drift("spy", "data/raw/spy/2026-02-14.parquet")
        for alert in alerts:
            logger.warning(alert)
    """

    def __init__(self, config: Dict[str, Any], session_label: str = "default"):
        """
        Args:
            config: Full merged configuration dict.  Reads toggles from
                    ``config["monitoring"]["schema"]``.
            session_label: Label for this monitoring session.
        """
        self.session_label = session_label
        schema_cfg = config.get("monitoring", {}).get("schema", {})

        # Alert toggles
        self.alert_on_new_columns = schema_cfg.get("alert_on_new_columns", True)
        self.alert_on_missing_columns = schema_cfg.get("alert_on_missing_columns", True)
        self.alert_on_type_changes = schema_cfg.get("alert_on_type_changes", True)
        self.auto_update_baseline = schema_cfg.get("auto_update_baseline", False)

        # Storage paths
        self._baseline_dir = Path("data/logs/schema")
        self._drift_dir = Path("data/logs/schema/drift")

    # ------------------------------------------------------------------
    # Schema extraction
    # ------------------------------------------------------------------

    def capture_baseline(self, source: str, parquet_path: str) -> Dict[str, Any]:
        """Extract schema from Parquet file metadata.

        Args:
            source: Data source name (e.g. "spy", "vix", "options", "news",
                    "consolidated").
            parquet_path: Path to a Parquet file.

        Returns:
            Baseline dict with ``source``, ``captured_at``, ``sample_file``,
            ``schema`` (column→dtype mapping), and ``column_count``.
        """
        arrow_schema = pq.read_schema(parquet_path)
        schema_map = {
            field.name: str(field.type) for field in arrow_schema
        }

        baseline = {
            "source": source,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sample_file": str(parquet_path),
            "schema": schema_map,
            "column_count": len(schema_map),
        }
        return baseline

    # ------------------------------------------------------------------
    # Baseline persistence
    # ------------------------------------------------------------------

    def save_baseline(self, source: str, baseline: Dict[str, Any]) -> Path:
        """Write baseline JSON to disk.

        Args:
            source: Data source name.
            baseline: Baseline dict from ``capture_baseline``.

        Returns:
            Path to the written file.
        """
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{source}_baseline.json"
        path.write_text(json.dumps(baseline, indent=2))
        logger.info(f"Schema baseline saved for '{source}' → {path}")
        return path

    def load_baseline(self, source: str) -> Optional[Dict[str, Any]]:
        """Load baseline from disk.

        Args:
            source: Data source name.

        Returns:
            Baseline dict, or None if no baseline file exists.
        """
        path = self._baseline_dir / f"{source}_baseline.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(self, source: str, parquet_path: str) -> List[str]:
        """Compare a Parquet file's schema against the stored baseline.

        If no baseline exists, auto-captures one and returns empty alerts.

        Args:
            source: Data source name.
            parquet_path: Path to the Parquet file to check.

        Returns:
            List of alert messages (empty if no drift or first capture).
        """
        baseline = self.load_baseline(source)

        if baseline is None:
            # First time — auto-capture baseline
            new_baseline = self.capture_baseline(source, parquet_path)
            self.save_baseline(source, new_baseline)
            logger.info(
                f"No baseline for '{source}' — auto-captured from {parquet_path}"
            )
            return []

        current = self.capture_baseline(source, parquet_path)
        changes = self.detect_schema_changes(baseline["schema"], current["schema"])

        if not changes["new_columns"] and not changes["missing_columns"] and not changes["type_changes"]:
            return []

        alerts = self.format_alerts(source, changes)
        for alert in alerts:
            logger.warning(alert)

        # Log drift event to disk
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_drift(source, date_str, changes)

        # Auto-update baseline if configured
        if self.auto_update_baseline:
            self.save_baseline(source, current)
            logger.info(f"Baseline auto-updated for '{source}'")

        return alerts

    def detect_schema_changes(
        self,
        baseline: Dict[str, str],
        current: Dict[str, str],
    ) -> Dict[str, Any]:
        """Pure diff between two schema dicts.

        Args:
            baseline: Column→dtype mapping from the stored baseline.
            current: Column→dtype mapping from the current file.

        Returns:
            Dict with ``new_columns``, ``missing_columns``, and
            ``type_changes`` lists.
        """
        baseline_cols = set(baseline.keys())
        current_cols = set(current.keys())

        new_columns = sorted(current_cols - baseline_cols)
        missing_columns = sorted(baseline_cols - current_cols)

        type_changes = []
        for col in sorted(baseline_cols & current_cols):
            if baseline[col] != current[col]:
                type_changes.append({
                    "column": col,
                    "baseline_type": baseline[col],
                    "current_type": current[col],
                })

        return {
            "new_columns": new_columns,
            "missing_columns": missing_columns,
            "type_changes": type_changes,
        }

    def format_alerts(self, source: str, changes: Dict[str, Any]) -> List[str]:
        """Convert a changes dict into human-readable alert messages.

        Respects the ``alert_on_*`` config toggles — suppressed categories
        produce no alerts.

        Args:
            source: Data source name.
            changes: Dict from ``detect_schema_changes``.

        Returns:
            List of alert strings.
        """
        alerts: List[str] = []

        if self.alert_on_new_columns and changes["new_columns"]:
            cols = ", ".join(changes["new_columns"])
            alerts.append(
                f"SCHEMA DRIFT [{source}]: New columns detected: {cols}"
            )

        if self.alert_on_missing_columns and changes["missing_columns"]:
            cols = ", ".join(changes["missing_columns"])
            alerts.append(
                f"SCHEMA DRIFT [{source}]: Missing columns: {cols}"
            )

        if self.alert_on_type_changes and changes["type_changes"]:
            for tc in changes["type_changes"]:
                alerts.append(
                    f"SCHEMA DRIFT [{source}]: Type change on '{tc['column']}' — "
                    f"{tc['baseline_type']} → {tc['current_type']}"
                )

        return alerts

    # ------------------------------------------------------------------
    # Drift logging
    # ------------------------------------------------------------------

    def log_drift(self, source: str, date: str, changes: Dict[str, Any]) -> Path:
        """Write a drift event JSON to the drift log directory.

        Args:
            source: Data source name.
            date: Date string (YYYY-MM-DD).
            changes: Dict from ``detect_schema_changes``.

        Returns:
            Path to the written drift file.
        """
        self._drift_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{source}_{date}_{timestamp}.json"
        path = self._drift_dir / filename

        event = {
            "source": source,
            "date": date,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "new_columns": changes["new_columns"],
            "missing_columns": changes["missing_columns"],
            "type_changes": changes["type_changes"],
        }
        path.write_text(json.dumps(event, indent=2))
        logger.info(f"Schema drift logged → {path}")
        return path
