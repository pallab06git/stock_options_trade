# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Polygon.io VIX index client for historical and real-time aggregate data.

Fetches VIX (I:VIX) per-minute aggregates via the Polygon REST API and
streams real-time data via WebSocket (Market.Indices). Implements the
BaseSource interface.
"""

import queue
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Dict, Any, Generator, List

from polygon.websocket.models import Market

from src.data_sources.base_source import BaseSource, ExecutionMode
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()

# Required fields for a valid VIX aggregate record
_REQUIRED_FIELDS = {"timestamp", "open", "high", "low", "close"}

# Price fields that must be positive
_PRICE_FIELDS = ("open", "high", "low", "close")


class PolygonVIXClient(BaseSource):
    """
    Fetch VIX index aggregates from Polygon.io.

    Supports both historical (REST) and real-time (WebSocket) modes.
    Uses Market.Indices for WebSocket subscriptions.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        connection_manager: ConnectionManager,
    ):
        """
        Args:
            config: Full merged config dict.
            connection_manager: Shared ConnectionManager for REST client and rate limiting.
        """
        super().__init__(config)
        self.connection_manager = connection_manager

        vix_config = config.get("polygon", {}).get("vix", {})
        self.ticker = vix_config.get("ticker", "I:VIX")
        self.multiplier = vix_config.get("multiplier", 1)
        self.timespan = vix_config.get("timespan", "minute")
        self.limit_per_request = vix_config.get("limit_per_request", 50000)

    def connect(self) -> None:
        """Verify REST client is available and set mode to HISTORICAL."""
        self.connection_manager.get_rest_client()
        self.mode = ExecutionMode.HISTORICAL
        logger.info(f"PolygonVIXClient connected (ticker={self.ticker}, mode=historical)")

    def disconnect(self) -> None:
        """No-op — ConnectionManager owns the REST client lifecycle."""
        logger.debug(f"PolygonVIXClient disconnect for {self.ticker} (no-op)")

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch VIX aggregates for a date range.

        Iterates date-by-date, yielding individual records as dicts.
        On error for a single date, logs and skips to the next date.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Yields:
            Standardized VIX aggregate records.
        """
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                records = self._fetch_single_date(date_str)
                count = len(records)
                if count > 0:
                    logger.info(f"Fetched {count} VIX bars for {date_str}")
                else:
                    logger.debug(f"No VIX data for {date_str}")
                for record in records:
                    yield record
            except Exception as e:
                logger.error(
                    f"Failed to fetch VIX data for {date_str}: {e}. Skipping date."
                )
            current += timedelta(days=1)

    def _fetch_single_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Fetch all VIX aggregates for a single date.

        Args:
            date: Date string (YYYY-MM-DD).

        Returns:
            List of standardized aggregate records.
        """
        self.connection_manager.acquire_rate_limit(source="vix")

        @with_retry(source="polygon", config=self.config)
        def _api_call():
            client = self.connection_manager.get_rest_client()
            try:
                aggs = client.get_aggs(
                    ticker=self.ticker,
                    multiplier=self.multiplier,
                    timespan=self.timespan,
                    from_=date,
                    to=date,
                    limit=self.limit_per_request,
                    sort="asc",
                )
                return aggs if aggs else []
            except Exception as e:
                status_code = getattr(e, "status_code", getattr(e, "status", 0))
                if isinstance(status_code, int) and status_code > 0:
                    raise RetryableError(str(e), status_code=status_code) from e
                raise

        raw_aggs = _api_call()
        return [self._transform_agg(agg) for agg in raw_aggs]

    def _transform_agg(self, agg) -> Dict[str, Any]:
        """
        Transform a Polygon Agg object to a standardized dict.

        VIX indices have no volume/vwap/transactions — these are set to None.

        Args:
            agg: Polygon Agg object with fields o, h, l, c, t, etc.

        Returns:
            Standardized record dict with source="vix".
        """
        return {
            "timestamp": getattr(agg, "timestamp", None),
            "open": getattr(agg, "open", None),
            "high": getattr(agg, "high", None),
            "low": getattr(agg, "low", None),
            "close": getattr(agg, "close", None),
            "volume": getattr(agg, "volume", None),
            "vwap": getattr(agg, "vwap", None),
            "transactions": getattr(agg, "transactions", None),
            "source": "vix",
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a VIX aggregate record.

        Checks that all required fields are present, timestamp is positive,
        and OHLC prices are positive. Volume/vwap are not required for indices.

        Args:
            record: Record dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        for field in _REQUIRED_FIELDS:
            if record.get(field) is None:
                return False

        if record["timestamp"] <= 0:
            return False

        for field in _PRICE_FIELDS:
            if record[field] <= 0:
                return False

        return True

    # Sentinel object to signal the generator thread has exited
    _SENTINEL = object()

    def stream_realtime(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Stream real-time VIX aggregates via Polygon WebSocket.

        Uses Market.Indices for VIX data and subscribes to "A.I:VIX".

        Args:
            stop_event: threading.Event to signal shutdown from outside.

        Yields:
            Standardized VIX aggregate dicts (same schema as REST).
        """
        stop_event = kwargs.get("stop_event") or threading.Event()
        msg_queue: queue.Queue = queue.Queue(maxsize=10000)
        reconnect_delay = 5

        def _ws_thread():
            """Background thread: connect, subscribe, and push to queue."""
            while not stop_event.is_set():
                try:
                    ws_client = self.connection_manager.get_ws_client(
                        market=Market.Indices,
                    )
                    ws_client.subscribe(f"A.{self.ticker}")

                    def _handle_msg(msgs):
                        for msg in msgs:
                            if stop_event.is_set():
                                return
                            record = self._transform_agg(msg)
                            try:
                                msg_queue.put_nowait(record)
                            except queue.Full:
                                pass  # Drop oldest — backpressure

                    ws_client.run(handle_msg=_handle_msg)
                except Exception as e:
                    if stop_event.is_set():
                        break
                    logger.warning(
                        f"WebSocket disconnected for VIX: {e}. "
                        f"Reconnecting in {reconnect_delay}s"
                    )
                    _time.sleep(reconnect_delay)

            msg_queue.put(self._SENTINEL)

        thread = threading.Thread(target=_ws_thread, daemon=True)
        thread.start()
        self.mode = ExecutionMode.REALTIME
        logger.info(f"PolygonVIXClient streaming started (ticker={self.ticker})")

        try:
            while not stop_event.is_set():
                try:
                    item = msg_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is self._SENTINEL:
                    break
                yield item
        finally:
            stop_event.set()
            thread.join(timeout=5.0)
            logger.info(f"PolygonVIXClient streaming stopped (ticker={self.ticker})")
