# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Parquet storage sink for date-partitioned SPY aggregate data.

Writes standardized records to local Parquet files partitioned by date.
Uses pyarrow engine with configurable compression (default: Snappy).
Supports resumable fetches via append-with-dedup and full overwrite.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.sinks.base_sink import BaseSink, SinkType
from src.utils.logger import get_logger

logger = get_logger()


class ParquetSink(BaseSink):
    """
    Write records to date-partitioned Parquet files.

    File layout: {base_path}/{source}/{YYYY-MM-DD}.parquet
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sink_type = SinkType.PARQUET

        parquet_config = config.get("sinks", {}).get("parquet", {})
        self.base_path = Path(parquet_config.get("base_path", "data/raw"))
        self.compression = parquet_config.get("compression", "snappy")
        self.row_group_size = parquet_config.get("row_group_size", 10000)

    def connect(self) -> None:
        """Ensure the base directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ParquetSink connected (base_path={self.base_path})")

    def disconnect(self) -> None:
        """No-op — file-based sink has no persistent connection."""
        logger.debug("ParquetSink disconnect (no-op)")

    def write_batch(
        self,
        records: List[Dict[str, Any]],
        partition_key: Optional[str] = None,
    ) -> None:
        """
        Write a batch of records to a date-partitioned Parquet file.

        If the partition file already exists, new records are merged with
        existing data and deduplicated by timestamp.

        Args:
            records: List of standardized data records.
            partition_key: Date string (YYYY-MM-DD). Derived from first
                           record's timestamp if not provided.
        """
        if not records:
            return

        source = records[0].get("source", "unknown")
        if partition_key is None:
            partition_key = self._derive_partition_key(records[0])

        path = self._partition_path(source, partition_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame(records)

        if path.exists():
            existing_df = pd.read_parquet(path)
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=["timestamp"], keep="last")
            merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
        else:
            merged_df = new_df

        merged_df.to_parquet(
            path,
            engine="pyarrow",
            compression=self.compression,
            row_group_size=self.row_group_size,
            index=False,
        )
        logger.info(f"Wrote {len(merged_df)} records to {path}")

    def write_single(
        self,
        record: Dict[str, Any],
        partition_key: Optional[str] = None,
    ) -> None:
        """Write a single record by delegating to write_batch."""
        self.write_batch([record], partition_key)

    def check_duplicate(self, record: Dict[str, Any]) -> bool:
        """
        Check if a record's timestamp already exists in its partition file.

        Args:
            record: Data record to check.

        Returns:
            True if a record with the same timestamp exists, False otherwise.
        """
        source = record.get("source", "unknown")
        partition_key = self._derive_partition_key(record)
        path = self._partition_path(source, partition_key)

        if not path.exists():
            return False

        existing_df = pd.read_parquet(path, columns=["timestamp"])
        return int(record["timestamp"]) in existing_df["timestamp"].values

    def overwrite(
        self,
        records: List[Dict[str, Any]],
        partition_key: str,
    ) -> None:
        """
        Overwrite all data in a partition with new records.

        Args:
            records: New records to write.
            partition_key: Date partition to overwrite (YYYY-MM-DD).
        """
        if not records:
            return

        source = records[0].get("source", "unknown")
        path = self._partition_path(source, partition_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(records)
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=self.compression,
            row_group_size=self.row_group_size,
            index=False,
        )
        logger.info(f"Overwrote partition {partition_key} with {len(df)} records at {path}")

    def _partition_path(self, source: str, partition_key: str) -> Path:
        """Build the file path for a given source and date partition."""
        return self.base_path / source / f"{partition_key}.parquet"

    def _derive_partition_key(self, record: Dict[str, Any]) -> str:
        """Convert a record's timestamp (Unix ms) to a YYYY-MM-DD string."""
        ts_ms = record["timestamp"]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
