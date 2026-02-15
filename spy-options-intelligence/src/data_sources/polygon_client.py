# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Polygon.io equity client for historical per-second aggregate data.

Fetches equity (SPY, TSLA, etc.) per-second bars via the Polygon REST API
using the official polygon-api-client SDK. Implements the BaseSource interface
for historical mode. Streaming (WebSocket) is deferred to Step 10.
"""

import queue
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Dict, Any, Generator, List

from src.data_sources.base_source import BaseSource, ExecutionMode
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()

# Required fields for a valid equity aggregate record
_REQUIRED_FIELDS = {"timestamp", "open", "high", "low", "close"}

# Price fields that must be positive
_PRICE_FIELDS = ("open", "high", "low", "close")


class PolygonEquityClient(BaseSource):
    """
    Fetch equity per-second aggregate bars from Polygon.io REST API.

    Uses ConnectionManager for shared RESTClient and rate limiting.
    Generator-based processing for memory efficiency.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        connection_manager: ConnectionManager,
        ticker: str = "SPY",
    ):
        """
        Args:
            config: Full merged config dict.
            connection_manager: Shared ConnectionManager for REST client and rate limiting.
            ticker: Equity ticker symbol (e.g. "SPY", "TSLA").
        """
        super().__init__(config)
        self.connection_manager = connection_manager
        self.ticker = ticker

        # Try new equities config first, fall back to legacy polygon.spy
        equities_config = config.get("polygon", {}).get("equities", {}).get(ticker, {})
        if equities_config:
            self.multiplier = equities_config.get("multiplier", 1)
            self.timespan = equities_config.get("timespan", "second")
            self.limit_per_request = equities_config.get("limit_per_request", 50000)
        else:
            spy_config = config.get("polygon", {}).get("spy", {})
            self.multiplier = spy_config.get("multiplier", 1)
            self.timespan = spy_config.get("timespan", "second")
            self.limit_per_request = spy_config.get("limit_per_request", 50000)

    def connect(self) -> None:
        """Verify REST client is available and set mode to HISTORICAL."""
        self.connection_manager.get_rest_client()
        self.mode = ExecutionMode.HISTORICAL
        logger.info(f"PolygonEquityClient connected (ticker={self.ticker}, mode=historical)")

    def disconnect(self) -> None:
        """No-op — ConnectionManager owns the REST client lifecycle."""
        logger.debug(f"PolygonEquityClient disconnect for {self.ticker} (no-op)")

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch equity per-second aggregates for a date range.

        Iterates date-by-date, yielding individual records as dicts.
        On error for a single date, logs and skips to the next date.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Yields:
            Standardized equity aggregate records.
        """
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                records = self._fetch_single_date(date_str)
                count = len(records)
                if count > 0:
                    logger.info(f"Fetched {count} {self.ticker} bars for {date_str}")
                else:
                    logger.debug(f"No {self.ticker} data for {date_str}")
                for record in records:
                    yield record
            except Exception as e:
                logger.error(
                    f"Failed to fetch {self.ticker} data for {date_str}: {e}. Skipping date."
                )
            current += timedelta(days=1)

    def _fetch_single_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Fetch all equity aggregates for a single date.

        Acquires rate limit before making the API call. Retries on
        server errors via the with_retry decorator.

        Args:
            date: Date string (YYYY-MM-DD).

        Returns:
            List of standardized aggregate records.
        """
        self.connection_manager.acquire_rate_limit(source=self.ticker.lower())

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

        Args:
            agg: Polygon Agg object with fields o, h, l, c, v, vw, t, n.

        Returns:
            Standardized record dict.
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
            "source": self.ticker.lower(),
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate an equity aggregate record.

        Checks that all required fields are present, timestamp is positive,
        and OHLC prices are positive.

        Args:
            record: Record dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Check required fields present and not None
        for field in _REQUIRED_FIELDS:
            if record.get(field) is None:
                return False

        # Timestamp must be positive
        if record["timestamp"] <= 0:
            return False

        # Price fields must be positive
        for field in _PRICE_FIELDS:
            if record[field] <= 0:
                return False

        return True

    # Sentinel object to signal the generator thread has exited
    _SENTINEL = object()

    def stream_realtime(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Stream real-time equity aggregates via Polygon WebSocket.

        Uses a background daemon thread running the WebSocket client.
        Messages are bridged to the caller via a bounded queue.

        Args:
            stop_event: threading.Event to signal shutdown from outside.

        Yields:
            Standardized equity aggregate dicts (same schema as REST).
        """
        stop_event = kwargs.get("stop_event") or threading.Event()
        msg_queue: queue.Queue = queue.Queue(maxsize=10000)
        reconnect_delay = 5  # seconds between reconnect attempts

        def _ws_thread():
            """Background thread: connect, subscribe, and push to queue."""
            while not stop_event.is_set():
                try:
                    ws_client = self.connection_manager.get_ws_client()
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
                        f"WebSocket disconnected for {self.ticker}: {e}. "
                        f"Reconnecting in {reconnect_delay}s"
                    )
                    _time.sleep(reconnect_delay)

            # Signal generator to stop
            msg_queue.put(self._SENTINEL)

        thread = threading.Thread(target=_ws_thread, daemon=True)
        thread.start()
        self.mode = ExecutionMode.REALTIME
        logger.info(f"PolygonEquityClient streaming started (ticker={self.ticker})")

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
            logger.info(f"PolygonEquityClient streaming stopped (ticker={self.ticker})")


# Backward-compatible alias
PolygonSPYClient = PolygonEquityClient
