# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PolygonNewsClient."""

import queue
import threading
import time

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tickers=None, sort=None, order=None, limit=None, poll_interval=None):
    """Build a minimal config dict for PolygonNewsClient."""
    news_cfg = {}
    if tickers is not None:
        news_cfg["tickers"] = tickers
    if sort is not None:
        news_cfg["sort"] = sort
    if order is not None:
        news_cfg["order"] = order
    if limit is not None:
        news_cfg["limit_per_request"] = limit
    if poll_interval is not None:
        news_cfg["poll_interval_seconds"] = poll_interval

    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "rate_limiting": {"total_requests_per_minute": 5},
            "news": news_cfg,
        },
        "retry": {
            "polygon": {
                "max_retries": 1,
                "base_delay_seconds": 0,
                "max_delay_seconds": 0,
                "retryable_status_codes": [429, 500, 502, 503],
            },
        },
    }


def _make_news_item(
    article_id="abc123",
    title="SPY drops 2%",
    description="Markets tumble on...",
    published_utc="2026-02-10T14:30:00Z",
    author="John Doe",
    article_url="https://example.com/article",
    tickers=None,
    keywords=None,
    insights=None,
    publisher_name="Reuters",
):
    """Create a mock Polygon news object."""
    item = MagicMock()
    item.id = article_id
    item.title = title
    item.description = description
    item.published_utc = published_utc
    item.author = author
    item.article_url = article_url
    item.tickers = tickers or ["SPY"]
    item.keywords = keywords or ["markets", "spy"]
    item.insights = insights

    publisher = MagicMock()
    publisher.name = publisher_name
    item.publisher = publisher

    return item


def _make_insight(ticker="SPY", sentiment="negative", reasoning="Price decline"):
    """Create a mock insight object."""
    insight = MagicMock()
    insight.ticker = ticker
    insight.sentiment = sentiment
    insight.sentiment_reasoning = reasoning
    return insight


# ---------------------------------------------------------------------------
# Test: Config Defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_config_defaults(self):
        """Default tickers/sort/order/limit/poll_interval when no config provided."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        assert client.tickers == ["SPY"]
        assert client.sort == "published_utc"
        assert client.order == "asc"
        assert client.limit_per_request == 100
        assert client.poll_interval_seconds == 300

    def test_config_override(self):
        """Custom config values are used when provided."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config(
            tickers=["AAPL", "TSLA"],
            sort="relevance",
            order="desc",
            limit=50,
            poll_interval=60,
        )
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        assert client.tickers == ["AAPL", "TSLA"]
        assert client.sort == "relevance"
        assert client.order == "desc"
        assert client.limit_per_request == 50
        assert client.poll_interval_seconds == 60


# ---------------------------------------------------------------------------
# Test: Connect
# ---------------------------------------------------------------------------

class TestConnect:

    def test_connect(self):
        """Connect verifies REST client acquired and sets mode."""
        from src.data_sources.base_source import ExecutionMode
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        client.connect()

        cm.get_rest_client.assert_called_once()
        assert client.mode == ExecutionMode.HISTORICAL


# ---------------------------------------------------------------------------
# Test: Transform
# ---------------------------------------------------------------------------

class TestTransformNews:

    def test_transform_news(self):
        """Transform maps all fields correctly with source='news'."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        insight = _make_insight("SPY", "negative", "Price decline")
        item = _make_news_item(insights=[insight])

        record = client._transform_news(item)

        assert record["source"] == "news"
        assert record["article_id"] == "abc123"
        assert record["title"] == "SPY drops 2%"
        assert record["description"] == "Markets tumble on..."
        assert record["author"] == "John Doe"
        assert record["article_url"] == "https://example.com/article"
        assert record["tickers"] == ["SPY"]
        assert record["keywords"] == ["markets", "spy"]
        assert record["sentiment"] == "negative"
        assert record["sentiment_reasoning"] == "Price decline"
        assert record["publisher_name"] == "Reuters"
        assert isinstance(record["timestamp"], int)

    def test_transform_news_no_insights(self):
        """No insights → sentiment=None, reasoning=None."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        item = _make_news_item(insights=None)

        record = client._transform_news(item)

        assert record["sentiment"] is None
        assert record["sentiment_reasoning"] is None


# ---------------------------------------------------------------------------
# Test: Parse Published UTC
# ---------------------------------------------------------------------------

class TestParsePublishedUtc:

    def test_parse_published_utc(self):
        """ISO 8601 string → Unix ms conversion."""
        from src.data_sources.news_client import PolygonNewsClient

        result = PolygonNewsClient._parse_published_utc("2026-02-10T14:30:00Z")

        assert isinstance(result, int)
        assert result > 0
        # 2026-02-10 14:30:00 UTC
        assert result == 1770733800000

    def test_parse_published_utc_none(self):
        """None input → None output."""
        from src.data_sources.news_client import PolygonNewsClient

        assert PolygonNewsClient._parse_published_utc(None) is None

    def test_parse_published_utc_with_offset(self):
        """ISO 8601 with timezone offset parses correctly."""
        from src.data_sources.news_client import PolygonNewsClient

        result = PolygonNewsClient._parse_published_utc("2026-02-10T14:30:00+00:00")
        assert isinstance(result, int)
        assert result == 1770733800000


# ---------------------------------------------------------------------------
# Test: Validate Record
# ---------------------------------------------------------------------------

class TestValidateRecord:

    def test_validate_record_valid(self):
        """Valid news record with required fields passes."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        record = {
            "timestamp": 1770857400000,
            "title": "SPY drops 2%",
            "source": "news",
        }
        assert client.validate_record(record) is True

    def test_validate_record_missing_title(self):
        """Missing title fails validation."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        record = {"timestamp": 1770857400000, "title": None, "source": "news"}
        assert client.validate_record(record) is False

    def test_validate_record_missing_timestamp(self):
        """Missing timestamp fails validation."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        record = {"title": "SPY drops 2%", "source": "news"}
        assert client.validate_record(record) is False


# ---------------------------------------------------------------------------
# Test: Extract Sentiment
# ---------------------------------------------------------------------------

class TestExtractSentiment:

    def test_extract_sentiment_matching_ticker(self):
        """Extracts sentiment from insight matching the ticker."""
        from src.data_sources.news_client import PolygonNewsClient

        insight_spy = _make_insight("SPY", "negative", "Price decline")
        insight_aapl = _make_insight("AAPL", "positive", "Earnings beat")

        sentiment, reasoning = PolygonNewsClient._extract_sentiment(
            [insight_aapl, insight_spy], "SPY"
        )
        assert sentiment == "negative"
        assert reasoning == "Price decline"

    def test_extract_sentiment_no_match_uses_first(self):
        """When no ticker match, uses first insight."""
        from src.data_sources.news_client import PolygonNewsClient

        insight = _make_insight("AAPL", "positive", "Earnings beat")

        sentiment, reasoning = PolygonNewsClient._extract_sentiment(
            [insight], "SPY"
        )
        assert sentiment == "positive"
        assert reasoning == "Earnings beat"

    def test_extract_sentiment_empty_insights(self):
        """Empty insights → (None, None)."""
        from src.data_sources.news_client import PolygonNewsClient

        sentiment, reasoning = PolygonNewsClient._extract_sentiment([], "SPY")
        assert sentiment is None
        assert reasoning is None


# ---------------------------------------------------------------------------
# Test: Fetch Historical
# ---------------------------------------------------------------------------

class TestFetchHistorical:

    def test_fetch_historical_single_date(self):
        """Mock list_ticker_news, verify records are transformed and yielded."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        mock_rest = MagicMock()
        cm.get_rest_client.return_value = mock_rest

        items = [_make_news_item("id1", "Article 1"), _make_news_item("id2", "Article 2")]
        mock_rest.list_ticker_news.return_value = items

        client.connect()
        records = list(client.fetch_historical("2026-02-10", "2026-02-10"))

        assert len(records) == 2
        assert records[0]["article_id"] == "id1"
        assert records[1]["article_id"] == "id2"
        assert all(r["source"] == "news" for r in records)

    def test_fetch_historical_empty(self):
        """No data returns nothing."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        mock_rest = MagicMock()
        cm.get_rest_client.return_value = mock_rest
        mock_rest.list_ticker_news.return_value = []

        client.connect()
        records = list(client.fetch_historical("2026-02-10", "2026-02-10"))

        assert len(records) == 0

    def test_fetch_historical_skips_error_dates(self):
        """Error on one date, continues to next."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config()
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        mock_rest = MagicMock()
        cm.get_rest_client.return_value = mock_rest

        # First date raises, second succeeds
        items = [_make_news_item("id1", "Article 1")]
        mock_rest.list_ticker_news.side_effect = [
            RuntimeError("API error"),
            items,
        ]

        client.connect()
        records = list(client.fetch_historical("2026-02-10", "2026-02-11"))

        assert len(records) == 1
        assert records[0]["article_id"] == "id1"


# ---------------------------------------------------------------------------
# Test: Stream Realtime (Polling)
# ---------------------------------------------------------------------------

class TestStreamRealtime:

    def test_stream_realtime_polls_and_yields(self):
        """Mock polling loop yields records via queue."""
        from src.data_sources.news_client import PolygonNewsClient

        config = _make_config(poll_interval=1)
        cm = MagicMock()
        client = PolygonNewsClient(config, cm)

        mock_rest = MagicMock()
        cm.get_rest_client.return_value = mock_rest

        items = [_make_news_item("id1", "Article 1")]
        mock_rest.list_ticker_news.return_value = items

        stop_event = threading.Event()
        collected = []

        def collect():
            for record in client.stream_realtime(stop_event=stop_event):
                collected.append(record)
                if len(collected) >= 1:
                    stop_event.set()
                    break

        t = threading.Thread(target=collect)
        t.start()
        t.join(timeout=10)

        assert len(collected) >= 1
        assert collected[0]["source"] == "news"
        assert collected[0]["article_id"] == "id1"
