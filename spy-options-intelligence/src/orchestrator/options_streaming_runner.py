# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Real-time streaming runner for options WebSocket data.

Orchestrates the continuous pipeline:
WebSocket → buffer → validate → deduplicate → Parquet sink.
Uses compound dedup key (ticker + timestamp) since multiple
contracts share the same stream.
"""

import signal
import threading
from datetime import datetime
from typing import Any, Dict, List

from src.data_sources.polygon_options_client import PolygonOptionsClient
from src.monitoring.error_aggregator import ErrorAggregator
from src.monitoring.heartbeat_monitor import HeartbeatMonitor
from src.monitoring.performance_monitor import PerformanceMonitor
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator
from src.sinks.parquet_sink import ParquetSink
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger
from src.utils.market_hours import MarketHours

logger = get_logger()


class OptionsStreamingRunner:
    """Orchestrate real-time options streaming via WebSocket.

    Loads discovered contracts for a given date, subscribes to
    Polygon options WebSocket, and writes validated/deduplicated
    aggregate records to Parquet with compound dedup (ticker, timestamp).
    """

    def __init__(self, config: Dict[str, Any], date: str):
        """
        Args:
            config: Full merged configuration dict.
            date: Trading date (YYYY-MM-DD) for contract loading.
        """
        self.config = config
        self.date = date
        self.session_label = "streaming_options"

        # Pipeline components
        self.connection_manager = ConnectionManager(config)
        self.client = PolygonOptionsClient(config, self.connection_manager)
        self.validator = RecordValidator("options")
        dedup_max = config.get("processing", {}).get("deduplication", {}).get("max_size", 100000)
        self.deduplicator = Deduplicator(key_field="dedup_key", max_size=dedup_max)
        self.sink = ParquetSink(config, dedup_subset=["ticker", "timestamp"])

        # Market hours
        streaming_cfg = config.get("streaming", {})
        market_hours_cfg = streaming_cfg.get("market_hours", {})
        api_key = config.get("polygon", {}).get("api_key", "")
        self.market_hours = MarketHours(market_hours_cfg, api_key)

        # Monitoring
        monitoring_cfg = streaming_cfg.get("monitoring", {})
        self.heartbeat = HeartbeatMonitor(monitoring_cfg, self.session_label)
        self.perf_monitor = PerformanceMonitor(config, session_label=self.session_label)
        max_err_types = config.get("monitoring", {}).get("performance", {}).get("max_error_types", 100)
        self.error_aggregator = ErrorAggregator(config, session_label=self.session_label, max_error_types=max_err_types)

        # Streaming settings
        self.batch_size = streaming_cfg.get("batch_size", 1000)

        # Shutdown control
        self.stop_event = threading.Event()

        # Stats
        self._stats: Dict[str, Any] = {
            "messages_received": 0,
            "messages_written": 0,
            "messages_invalid": 0,
            "messages_duplicates": 0,
            "batches_flushed": 0,
        }

    def run(self) -> Dict[str, Any]:
        """Execute the options streaming pipeline.

        Returns:
            Stats dict with message counts.
        """
        if not self.market_hours.is_market_open():
            logger.info("Market closed — options streaming not started")
            return {"status": "market_closed", **self._stats}

        self._register_signal_handlers()

        logger.info(f"Starting options streaming for {self.date}")
        self._stats["start_time"] = datetime.utcnow().isoformat() + "Z"

        buffer: List[Dict[str, Any]] = []

        try:
            self.sink.connect()

            for record in self.client.stream_realtime(
                date=self.date, stop_event=self.stop_event
            ):
                self._stats["messages_received"] += 1
                self.heartbeat.record_message()
                buffer.append(record)

                # Flush at batch_size threshold
                if len(buffer) >= self.batch_size:
                    self._flush_buffer(buffer)
                    buffer = []

                # Periodic heartbeat
                if self.heartbeat.should_send_heartbeat():
                    self.heartbeat.send_heartbeat()

                # Check market hours periodically
                if not self.market_hours.is_market_open():
                    logger.info("Market closed — stopping options stream")
                    break

                # Check for stalled stream
                if self.heartbeat.check_stalled_stream():
                    logger.error("Options stream appears stalled")

            # Flush remaining buffer
            if buffer:
                self._flush_buffer(buffer)

        except Exception as e:
            logger.error(f"Options streaming error: {e}")
            self.error_aggregator.record_error("streaming_error", str(e))
            if buffer:
                self._flush_buffer(buffer)
        finally:
            self._stats["end_time"] = datetime.utcnow().isoformat() + "Z"
            self.sink.disconnect()
            self.connection_manager.close()

        self._stats["status"] = "completed"
        logger.info(
            f"Options streaming complete: "
            f"{self._stats['messages_received']} received, "
            f"{self._stats['messages_written']} written"
        )
        return self._stats

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> None:
        """Validate, deduplicate, and write a batch of records.

        Adds a temporary compound dedup_key (ticker_timestamp) for
        in-memory deduplication, then removes it before writing.

        Args:
            buffer: Raw records to process.
        """
        self.perf_monitor.start_operation("flush")

        # Add compound dedup key
        for r in buffer:
            r["dedup_key"] = f"{r.get('ticker', '')}_{r.get('timestamp', '')}"

        valid, invalid = self.validator.validate_batch(buffer)
        self._stats["messages_invalid"] += len(invalid)

        for _ in invalid:
            self.error_aggregator.record_error("validation_error", "invalid record")
        for _ in valid:
            self.error_aggregator.record_success()

        deduplicated = self.deduplicator.deduplicate_batch(valid)
        duplicates = len(valid) - len(deduplicated)
        self._stats["messages_duplicates"] += duplicates

        # Remove temporary dedup_key before writing
        for r in deduplicated:
            r.pop("dedup_key", None)
        # Also clean invalid records so buffer isn't polluted
        for r in invalid:
            r.pop("dedup_key", None)

        if deduplicated:
            # Partition by UTC date from record timestamp
            partitions: Dict[str, List[Dict[str, Any]]] = {}
            for record in deduplicated:
                ts = record.get("timestamp", 0)
                partition_key = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                partitions.setdefault(partition_key, []).append(record)

            for partition_key, records in partitions.items():
                self.sink.write_batch(records, partition_key)
                self._stats["messages_written"] += len(records)

        self._stats["batches_flushed"] += 1
        self.perf_monitor.end_operation("flush", len(deduplicated))

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown."""
        def _signal_handler(signum, frame):
            logger.info(f"Signal {signum} received — stopping options stream")
            self.stop_event.set()

        try:
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except ValueError:
            pass
