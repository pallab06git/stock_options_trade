# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Historical data ingestion runner.

Orchestrates the end-to-end pipeline for fetching SPY historical data:
Polygon REST API → Validator → Deduplicator → Parquet sink.
Records are batched by date partition and flushed at configurable thresholds.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from src.data_sources.polygon_client import PolygonSPYClient
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator
from src.sinks.parquet_sink import ParquetSink
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger

logger = get_logger()


class HistoricalRunner:
    """Orchestrate historical SPY data ingestion.

    Wires together the Polygon client, record validator, deduplicator,
    and Parquet sink into a batch pipeline. Records are accumulated
    per-date and flushed when the date boundary changes or the batch
    size threshold is reached.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged configuration dict.
        """
        self.config = config

        # Pipeline components
        self.connection_manager = ConnectionManager(config)
        self.client = PolygonSPYClient(config, self.connection_manager)
        self.validator = RecordValidator("spy")
        self.deduplicator = Deduplicator(key_field="timestamp")
        self.sink = ParquetSink(config)

        # Backfill settings
        backfill = config.get("historical", {}).get("backfill", {})
        self.batch_size = backfill.get("batch_size", 10000)
        self.trading_days = backfill.get("trading_days", 30)
        self.start_date_cfg = backfill.get("start_date")
        self.end_date_cfg = backfill.get("end_date")

    def run(self) -> Dict[str, Any]:
        """Execute the full historical ingestion pipeline.

        Returns:
            Stats dict with counts for fetched, valid, invalid,
            duplicates, and written records.
        """
        start_date, end_date = self._resolve_date_range()

        stats: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "dates_processed": 0,
            "total_fetched": 0,
            "total_valid": 0,
            "total_invalid": 0,
            "total_duplicates": 0,
            "total_written": 0,
        }

        logger.info(f"Historical run starting: {start_date} → {end_date}")

        try:
            self.client.connect()
            self.sink.connect()

            buffer: List[Dict[str, Any]] = []
            current_partition: str | None = None

            for record in self.client.fetch_historical(start_date, end_date):
                stats["total_fetched"] += 1

                partition = self._partition_from_timestamp(record["timestamp"])

                # Date boundary changed — flush previous partition
                if current_partition is not None and partition != current_partition:
                    batch_stats = self._process_batch(buffer, current_partition)
                    self._accumulate_stats(stats, batch_stats)
                    stats["dates_processed"] += 1
                    buffer = []
                    self.deduplicator.reset()

                current_partition = partition
                buffer.append(record)

                # Batch size threshold — flush within same partition
                if len(buffer) >= self.batch_size:
                    batch_stats = self._process_batch(buffer, current_partition)
                    self._accumulate_stats(stats, batch_stats)
                    buffer = []

            # Flush remaining records
            if buffer and current_partition is not None:
                batch_stats = self._process_batch(buffer, current_partition)
                self._accumulate_stats(stats, batch_stats)
                stats["dates_processed"] += 1

        finally:
            self.client.disconnect()
            self.sink.disconnect()
            self.connection_manager.close()

        logger.info(
            f"Historical run complete: {stats['dates_processed']} dates, "
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
