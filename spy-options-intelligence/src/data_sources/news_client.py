# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Polygon.io news client for historical backfill and polling-based streaming.

Fetches news articles via Polygon's list_ticker_news() REST API. Supports
both historical date-range backfill and polling-based "streaming" (news has
no WebSocket endpoint). Implements the BaseSource interface.
"""

import queue
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

from src.data_sources.base_source import BaseSource, ExecutionMode
from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()

# Required fields for a valid news record
_REQUIRED_FIELDS = {"timestamp", "title"}


class PolygonNewsClient(BaseSource):
    """
    Fetch news articles from Polygon.io REST API.

    Supports historical backfill via date-range queries and polling-based
    "streaming" via periodic REST calls. Uses article_id for deduplication
    (multiple articles can share the same published second).
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

        news_config = config.get("polygon", {}).get("news", {})
        self.tickers = news_config.get("tickers", ["SPY"])
        self.sort = news_config.get("sort", "published_utc")
        self.order = news_config.get("order", "asc")
        self.limit_per_request = news_config.get("limit_per_request", 100)
        self.poll_interval_seconds = news_config.get("poll_interval_seconds", 300)

    def connect(self) -> None:
        """Verify REST client is available and set mode to HISTORICAL."""
        self.connection_manager.get_rest_client()
        self.mode = ExecutionMode.HISTORICAL
        logger.info("PolygonNewsClient connected (mode=historical)")

    def disconnect(self) -> None:
        """No-op — ConnectionManager owns the REST client lifecycle."""
        logger.debug("PolygonNewsClient disconnect (no-op)")

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch news articles for a date range.

        Iterates date-by-date, yielding transformed news records.
        On error for a single date, logs and skips to the next date.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Yields:
            Standardized news records.
        """
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                records = self._fetch_single_date(date_str, next_day)
                count = len(records)
                if count > 0:
                    logger.info(f"Fetched {count} news articles for {date_str}")
                else:
                    logger.debug(f"No news data for {date_str}")
                for record in records:
                    yield record
            except Exception as e:
                logger.error(
                    f"Failed to fetch news for {date_str}: {e}. Skipping date."
                )
            current += timedelta(days=1)

    def _fetch_single_date(
        self, date_str: str, next_day_str: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch all news articles for a single date.

        Args:
            date_str: Date string (YYYY-MM-DD) — inclusive lower bound.
            next_day_str: Next date string (YYYY-MM-DD) — exclusive upper bound.

        Returns:
            List of standardized news records.
        """
        self.connection_manager.acquire_rate_limit(source="news")

        @with_retry(source="polygon", config=self.config)
        def _api_call():
            client = self.connection_manager.get_rest_client()
            try:
                results = list(client.list_ticker_news(
                    ticker=self.tickers[0] if self.tickers else "SPY",
                    published_utc_gte=date_str,
                    published_utc_lt=next_day_str,
                    sort=self.sort,
                    order=self.order,
                    limit=self.limit_per_request,
                ))
                return results
            except Exception as e:
                status_code = getattr(e, "status_code", getattr(e, "status", 0))
                if isinstance(status_code, int) and status_code > 0:
                    raise RetryableError(str(e), status_code=status_code) from e
                raise

        raw_articles = _api_call()
        return [self._transform_news(item) for item in raw_articles]

    def _transform_news(self, item: Any) -> Dict[str, Any]:
        """
        Transform a Polygon news object to a standardized dict.

        Args:
            item: Polygon news object with attributes like title, published_utc, etc.

        Returns:
            Standardized news record dict with source="news".
        """
        published_utc = getattr(item, "published_utc", None)
        timestamp = self._parse_published_utc(published_utc)

        # Extract insights for the primary ticker
        insights = getattr(item, "insights", None) or []
        primary_ticker = self.tickers[0] if self.tickers else "SPY"
        sentiment, sentiment_reasoning = self._extract_sentiment(
            insights, primary_ticker
        )

        publisher = getattr(item, "publisher", None)
        publisher_name = getattr(publisher, "name", None) if publisher else None

        return {
            "timestamp": timestamp,
            "article_id": getattr(item, "id", None),
            "title": getattr(item, "title", None),
            "description": getattr(item, "description", None),
            "author": getattr(item, "author", None),
            "article_url": getattr(item, "article_url", None),
            "tickers": getattr(item, "tickers", None),
            "keywords": getattr(item, "keywords", None),
            "sentiment": sentiment,
            "sentiment_reasoning": sentiment_reasoning,
            "publisher_name": publisher_name,
            "source": "news",
        }

    @staticmethod
    def _parse_published_utc(utc_str: Optional[str]) -> Optional[int]:
        """
        Convert an ISO 8601 UTC string to Unix milliseconds.

        Handles formats with and without timezone info.

        Args:
            utc_str: ISO 8601 date-time string (e.g. "2026-02-10T14:30:00Z").

        Returns:
            Unix timestamp in milliseconds, or None if input is None.
        """
        if utc_str is None:
            return None

        # Strip trailing 'Z' and any timezone suffix for parsing
        clean = utc_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(clean)
            # Convert to UTC timestamp in ms
            return int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _extract_sentiment(
        insights: List[Any], ticker: str
    ) -> tuple:
        """
        Extract sentiment and reasoning from Polygon insights for a ticker.

        Args:
            insights: List of insight objects from Polygon news response.
            ticker: Ticker to match in insights.

        Returns:
            Tuple of (sentiment, reasoning). Both None if not found.
        """
        if not insights:
            return None, None

        for insight in insights:
            insight_ticker = getattr(insight, "ticker", None)
            if insight_ticker == ticker:
                return (
                    getattr(insight, "sentiment", None),
                    getattr(insight, "sentiment_reasoning", None),
                )

        # If no exact ticker match, use the first insight
        first = insights[0]
        return (
            getattr(first, "sentiment", None),
            getattr(first, "sentiment_reasoning", None),
        )

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a news record.

        Checks that required fields (timestamp, title) are present and not None.

        Args:
            record: Record dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        for field in _REQUIRED_FIELDS:
            if record.get(field) is None:
                return False
        return True

    # Sentinel object to signal the polling thread has exited
    _SENTINEL = object()

    def stream_realtime(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Poll for new news articles at a configurable interval.

        News has no WebSocket — this uses periodic REST calls instead.
        Tracks last_published_utc to yield only new articles.
        Uses a queue-based bridge to the main thread (same pattern as
        equity WebSocket streaming but with time.sleep polling loop).

        Args:
            stop_event: threading.Event to signal shutdown from outside.

        Yields:
            Standardized news record dicts.
        """
        stop_event = kwargs.get("stop_event") or threading.Event()
        msg_queue: queue.Queue = queue.Queue(maxsize=10000)

        # Track the most recent published_utc seen to avoid re-yielding
        last_published_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        def _poll_thread():
            """Background thread: poll REST API and push new articles to queue."""
            nonlocal last_published_utc
            while not stop_event.is_set():
                try:
                    self.connection_manager.acquire_rate_limit(source="news")
                    client = self.connection_manager.get_rest_client()
                    results = list(client.list_ticker_news(
                        ticker=self.tickers[0] if self.tickers else "SPY",
                        published_utc_gte=last_published_utc,
                        sort=self.sort,
                        order=self.order,
                        limit=self.limit_per_request,
                    ))

                    for item in results:
                        if stop_event.is_set():
                            break
                        record = self._transform_news(item)
                        pub_utc = getattr(item, "published_utc", None)
                        if pub_utc and pub_utc > last_published_utc:
                            last_published_utc = pub_utc
                        try:
                            msg_queue.put_nowait(record)
                        except queue.Full:
                            pass  # Drop — backpressure

                except Exception as e:
                    if stop_event.is_set():
                        break
                    logger.warning(f"News polling error: {e}. Retrying in {self.poll_interval_seconds}s")

                # Wait for next poll interval (check stop_event frequently)
                for _ in range(self.poll_interval_seconds):
                    if stop_event.is_set():
                        break
                    _time.sleep(1)

            msg_queue.put(self._SENTINEL)

        thread = threading.Thread(target=_poll_thread, daemon=True)
        thread.start()
        self.mode = ExecutionMode.REALTIME
        logger.info("PolygonNewsClient polling started")

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
            logger.info("PolygonNewsClient polling stopped")
