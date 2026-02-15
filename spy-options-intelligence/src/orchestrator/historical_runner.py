# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Historical data ingestion runner.

Orchestrates the end-to-end pipeline for fetching equity historical data:
Polygon REST API → Validator → Deduplicator → Parquet sink.
Processes one date at a time with optional checkpoint/resume support.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.data_sources.polygon_client import PolygonEquityClient
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator
from src.sinks.parquet_sink import ParquetSink
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger

logger = get_logger()


class HistoricalRunner:
    """Orchestrate historical equity data ingestion.

    Wires together the Polygon client, record validator, deduplicator,
    and Parquet sink into a batch pipeline. Processes one date at a time,
    with optional checkpoint/resume to skip already-completed dates.
    """

    def __init__(self, config: Dict[str, Any], ticker: str = "SPY"):
        """
        Args:
            config: Full merged configuration dict.
            ticker: Equity ticker symbol (e.g. "SPY", "TSLA").
        """
        self.config = config
        self.ticker = ticker

        # Pipeline components
        self.connection_manager = ConnectionManager(config)
        self.client = PolygonEquityClient(config, self.connection_manager, ticker=self.ticker)
        self.validator = RecordValidator.for_equity(self.ticker)
        self.deduplicator = Deduplicator(key_field="timestamp")
        self.sink = ParquetSink(config)

        # Backfill settings
        backfill = config.get("historical", {}).get("backfill", {})
        self.batch_size = backfill.get("batch_size", 10000)
        self.trading_days = backfill.get("trading_days", 30)
        self.start_date_cfg = backfill.get("start_date")
        self.end_date_cfg = backfill.get("end_date")

        # Checkpoint directory from logging config
        log_cfg = config.get("logging", {})
        self._checkpoint_dir = Path(
            log_cfg.get("execution_log_path", "data/logs/execution")
        )

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """Execute the full historical ingestion pipeline.

        Iterates date-by-date, processing each date through the full
        validate → deduplicate → write pipeline. When resume=True,
        previously completed dates (from checkpoint file) are skipped.

        Args:
            resume: If True, load checkpoint and skip completed dates.

        Returns:
            Stats dict with counts for fetched, valid, invalid,
            duplicates, and written records.
        """
        start_date, end_date = self._resolve_date_range()
        all_dates = self._date_range_list(start_date, end_date)

        # Load checkpoint if resuming
        completed_dates: Set[str] = set()
        if resume:
            completed_dates = self._load_checkpoint(start_date, end_date)
            if completed_dates:
                logger.info(
                    f"Resuming {self.ticker}: {len(completed_dates)} dates already completed, "
                    f"skipping them"
                )

        remaining_dates = [d for d in all_dates if d not in completed_dates]

        stats: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "dates_processed": 0,
            "dates_skipped": len(completed_dates),
            "total_fetched": 0,
            "total_valid": 0,
            "total_invalid": 0,
            "total_duplicates": 0,
            "total_written": 0,
        }

        logger.info(
            f"Historical run for {self.ticker}: {start_date} → {end_date} "
            f"({len(remaining_dates)} dates to process)"
        )

        try:
            self.client.connect()
            self.sink.connect()

            for date_str in remaining_dates:
                self.deduplicator.reset()
                date_fetched = 0
                buffer: List[Dict[str, Any]] = []

                for record in self.client.fetch_historical(date_str, date_str):
                    stats["total_fetched"] += 1
                    date_fetched += 1
                    buffer.append(record)

                    # Batch size threshold — flush within same date
                    if len(buffer) >= self.batch_size:
                        batch_stats = self._process_batch(buffer, date_str)
                        self._accumulate_stats(stats, batch_stats)
                        buffer = []

                # Flush remaining records for this date
                if buffer:
                    batch_stats = self._process_batch(buffer, date_str)
                    self._accumulate_stats(stats, batch_stats)

                if date_fetched > 0:
                    stats["dates_processed"] += 1

                # Save checkpoint after each date
                self._save_checkpoint(date_str, start_date, end_date)

        finally:
            self.client.disconnect()
            self.sink.disconnect()
            self.connection_manager.close()

        logger.info(
            f"Historical run for {self.ticker} complete: {stats['dates_processed']} dates, "
            f"{stats['total_written']}/{stats['total_fetched']} records written"
        )
        return stats

    def _process_batch(
        self, records: List[Dict[str, Any]], partition_key: str
    ) -> Dict[str, int]:
        """Validate, deduplicate, and write one batch.

        Args:
            records: Raw records for a single partition.
            partition_key: Date string (YYYY-MM-DD) for the Parquet partition.

        Returns:
            Per-batch stats dict.
        """
        valid, invalid = self.validator.validate_batch(records)
        deduplicated = self.deduplicator.deduplicate_batch(valid)

        if deduplicated:
            self.sink.write_batch(deduplicated, partition_key)

        return {
            "valid": len(valid),
            "invalid": len(invalid),
            "duplicates": len(valid) - len(deduplicated),
            "written": len(deduplicated),
        }

    def _resolve_date_range(self) -> Tuple[str, str]:
        """Compute start and end dates from config.

        Uses explicit start_date/end_date if provided, otherwise
        computes from trading_days relative to today.

        Returns:
            Tuple of (start_date, end_date) as YYYY-MM-DD strings.
        """
        if self.start_date_cfg and self.end_date_cfg:
            return self.start_date_cfg, self.end_date_cfg

        end = datetime.now()
        start = end - timedelta(days=self.trading_days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    @staticmethod
    def _date_range_list(start_date: str, end_date: str) -> List[str]:
        """Generate all dates in the range [start_date, end_date].

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of date strings in YYYY-MM-DD format.
        """
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    def _checkpoint_path(self, start_date: str, end_date: str) -> Path:
        """Return the checkpoint file path for a given date range.

        Includes ticker to prevent collision between parallel runs.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Path to the checkpoint JSON file.
        """
        return self._checkpoint_dir / f"checkpoint_{self.ticker}_{start_date}_{end_date}.json"

    def _load_checkpoint(self, start_date: str, end_date: str) -> Set[str]:
        """Load completed dates from checkpoint file.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Set of completed date strings. Empty set if no checkpoint exists.
        """
        path = self._checkpoint_path(start_date, end_date)
        if not path.exists():
            return set()

        try:
            data = json.loads(path.read_text())
            return set(data.get("completed_dates", []))
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Corrupt checkpoint file {path}, starting fresh")
            return set()

    def _save_checkpoint(
        self, date: str, start_date: str, end_date: str
    ) -> None:
        """Append a completed date to the checkpoint file.

        Args:
            date: The date just completed (YYYY-MM-DD).
            start_date: Range start for checkpoint file naming.
            end_date: Range end for checkpoint file naming.
        """
        path = self._checkpoint_path(start_date, end_date)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint or start fresh
        completed = list(self._load_checkpoint(start_date, end_date))
        if date not in completed:
            completed.append(date)

        data = {
            "completed_dates": completed,
            "last_updated": datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _partition_from_timestamp(timestamp_ms: int) -> str:
        """Convert a Unix-ms timestamp to a YYYY-MM-DD partition key."""
        return datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")

    @staticmethod
    def _accumulate_stats(stats: Dict[str, Any], batch: Dict[str, int]) -> None:
        """Merge per-batch stats into the running totals."""
        stats["total_valid"] += batch["valid"]
        stats["total_invalid"] += batch["invalid"]
        stats["total_duplicates"] += batch["duplicates"]
        stats["total_written"] += batch["written"]
