# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Real-time streaming runner for equity WebSocket data.

Orchestrates the continuous pipeline:
WebSocket → buffer → validate → deduplicate → Parquet sink.
Manages market hours, heartbeat monitoring, and graceful shutdown.
"""

import signal
import threading
from datetime import datetime
from typing import Any, Dict, List

from src.data_sources.polygon_client import PolygonEquityClient
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


class StreamingRunner:
    """Orchestrate real-time equity streaming via WebSocket.

    Connects to Polygon WebSocket, buffers incoming records, and
    periodically flushes validated/deduplicated batches to Parquet.
    Respects market hours and monitors stream health via heartbeats.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ticker: str = "SPY",
        connection_manager: "ConnectionManager | None" = None,
        client: "PolygonEquityClient | None" = None,
        validator: "RecordValidator | None" = None,
    ):
        """
        Args:
            config: Full merged configuration dict.
            ticker: Equity ticker symbol to stream (e.g. "SPY") or index (e.g. "I:VIX").
            connection_manager: Optional pre-built ConnectionManager (for DI).
            client: Optional pre-built data source client (for DI).
            validator: Optional pre-built RecordValidator (for DI).
        """
        self.config = config
        self.ticker = ticker
        self.session_label = f"streaming_{ticker.lower()}"

        # Pipeline components — use injected dependencies or create defaults
        self.connection_manager = connection_manager or ConnectionManager(config)
        self.client = client or PolygonEquityClient(config, self.connection_manager, ticker=self.ticker)
        self.validator = validator or RecordValidator.for_equity(self.ticker)
        self.deduplicator = Deduplicator(key_field="timestamp")
        self.sink = ParquetSink(config)

        # Market hours
        streaming_cfg = config.get("streaming", {})
        market_hours_cfg = streaming_cfg.get("market_hours", {})
        api_key = config.get("polygon", {}).get("api_key", "")
        self.market_hours = MarketHours(market_hours_cfg, api_key)

        # Monitoring
        monitoring_cfg = streaming_cfg.get("monitoring", {})
        self.heartbeat = HeartbeatMonitor(monitoring_cfg, self.session_label)
        self.perf_monitor = PerformanceMonitor(config, session_label=self.session_label)
        self.error_aggregator = ErrorAggregator(config, session_label=self.session_label)

        # Streaming settings
        self.batch_size = config.get("streaming", {}).get("batch_size", 1000)

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
        """Execute the streaming pipeline.

        Returns:
            Stats dict with message counts.
        """
        if not self.market_hours.is_market_open():
            logger.info(f"Market closed — streaming for {self.ticker} not started")
            return {"status": "market_closed", **self._stats}

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        logger.info(f"Starting streaming for {self.ticker}")
        self._stats["start_time"] = datetime.utcnow().isoformat() + "Z"

        buffer: List[Dict[str, Any]] = []

        try:
            self.sink.connect()

            for record in self.client.stream_realtime(stop_event=self.stop_event):
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
                    logger.info(f"Market closed — stopping {self.ticker} stream")
                    break

                # Check for stalled stream
                if self.heartbeat.check_stalled_stream():
                    logger.error(f"Stream appears stalled for {self.ticker}")

            # Flush remaining buffer
            if buffer:
                self._flush_buffer(buffer)

        except Exception as e:
            logger.error(f"Streaming error for {self.ticker}: {e}")
            self.error_aggregator.record_error("streaming_error", str(e))
            # Flush whatever is in the buffer
            if buffer:
                self._flush_buffer(buffer)
        finally:
            self._stats["end_time"] = datetime.utcnow().isoformat() + "Z"
            self.sink.disconnect()
            self.connection_manager.close()

        self._stats["status"] = "completed"
        logger.info(
            f"Streaming for {self.ticker} complete: "
            f"{self._stats['messages_received']} received, "
            f"{self._stats['messages_written']} written"
        )
        return self._stats

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> None:
        """Validate, deduplicate, and write a batch of records.

        Args:
            buffer: Raw records to process.
        """
        self.perf_monitor.start_operation("flush")

        valid, invalid = self.validator.validate_batch(buffer)
        self._stats["messages_invalid"] += len(invalid)

        for _ in invalid:
            self.error_aggregator.record_error("validation_error", "invalid record")
        for _ in valid:
            self.error_aggregator.record_success()

        deduplicated = self.deduplicator.deduplicate_batch(valid)
        duplicates = len(valid) - len(deduplicated)
        self._stats["messages_duplicates"] += duplicates

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
            logger.info(f"Signal {signum} received — stopping {self.ticker} stream")
            self.stop_event.set()

        try:
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except ValueError:
            # Can't set signal handlers outside main thread
            pass
