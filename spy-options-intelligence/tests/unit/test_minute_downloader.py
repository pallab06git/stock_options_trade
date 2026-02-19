# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for MinuteDownloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_sources.minute_downloader import MinuteDownloader, _source_name


# ---------------------------------------------------------------------------
# Source name helper
# ---------------------------------------------------------------------------


class TestSourceName:
    def test_spy_lower(self):
        assert _source_name("SPY") == "spy"

    def test_spy_already_lower(self):
        assert _source_name("spy") == "spy"

    def test_vix_mapped(self):
        assert _source_name("I:VIX") == "vix"

    def test_vix_lowercase(self):
        assert _source_name("i:vix") == "vix"

    def test_other_ticker_colon_replaced(self):
        result = _source_name("O:SPY250307C00625000")
        assert ":" not in result
        assert result == "o_spy250307c00625000"

    def test_plain_ticker(self):
        assert _source_name("TSLA") == "tsla"


# ---------------------------------------------------------------------------
# MinuteDownloader
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return {
        "polygon": {
            "api_key": "test",
            "equities": {
                "SPY": {"multiplier": 1, "timespan": "minute", "limit_per_request": 50000}
            },
        },
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "raw"),
                "compression": "snappy",
            }
        },
        "retry": {"polygon": {"max_attempts": 1, "base_delay_seconds": 0}},
    }


@pytest.fixture
def mock_cm():
    cm = MagicMock()
    cm.acquire_rate_limit.return_value = True
    return cm


class TestMinuteDownloaderDateRange:
    def test_single_date(self):
        dl = MinuteDownloader({}, MagicMock())
        dl.config = {}
        dates = dl._date_range("2025-03-01", "2025-03-01")
        assert dates == ["2025-03-01"]

    def test_three_days(self):
        dl = MinuteDownloader({}, MagicMock())
        dates = dl._date_range("2025-03-01", "2025-03-03")
        assert dates == ["2025-03-01", "2025-03-02", "2025-03-03"]

    def test_end_before_start_empty(self):
        dl = MinuteDownloader({}, MagicMock())
        dates = dl._date_range("2025-03-05", "2025-03-01")
        assert dates == []


class TestMinuteDownloaderDownload:
    def _make_record(self, ts=1740000000000):
        return {
            "timestamp": ts,
            "open": 600.0,
            "high": 601.0,
            "low": 599.0,
            "close": 600.5,
            "volume": 1000,
            "vwap": 600.2,
            "transactions": 50,
            "source": "spy",
        }

    def test_skip_existing_file(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)
        # Pre-create file
        out_dir = tmp_path / "raw" / "spy"
        out_dir.mkdir(parents=True)
        (out_dir / "2025-03-03.parquet").write_text("")

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.return_value = iter([])

            stats = dl.download("SPY", "2025-03-03", "2025-03-03", resume=True)

        assert stats["dates_skipped"] == 1
        assert stats["dates_downloaded"] == 0
        # fetch_historical should not have been called
        mock_client.fetch_historical.assert_not_called()

    def test_download_writes_parquet(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)
        record = self._make_record()

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.return_value = iter([record])

            stats = dl.download("SPY", "2025-03-03", "2025-03-03", resume=True)

        assert stats["dates_downloaded"] == 1
        assert stats["total_bars"] == 1
        out_path = tmp_path / "raw" / "spy" / "2025-03-03.parquet"
        assert out_path.exists()

    def test_vix_source_normalized(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)
        record = self._make_record()
        record["source"] = "i:vix"  # raw source before normalization

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.return_value = iter([record])

            stats = dl.download("I:VIX", "2025-03-03", "2025-03-03", resume=True)

        out_path = tmp_path / "raw" / "vix" / "2025-03-03.parquet"
        assert out_path.exists()
        # Source field in the written file should be "vix"
        df = pd.read_parquet(out_path)
        assert df.iloc[0]["source"] == "vix"

    def test_empty_result_no_file(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.return_value = iter([])

            stats = dl.download("SPY", "2025-03-03", "2025-03-03", resume=True)

        assert stats["dates_downloaded"] == 0
        assert stats["total_bars"] == 0
        out_path = tmp_path / "raw" / "spy" / "2025-03-03.parquet"
        assert not out_path.exists()

    def test_no_resume_overwrites(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)
        record = self._make_record()

        # Pre-create file
        out_dir = tmp_path / "raw" / "spy"
        out_dir.mkdir(parents=True)
        pd.DataFrame([record]).to_parquet(out_dir / "2025-03-03.parquet")

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.return_value = iter([record])

            stats = dl.download("SPY", "2025-03-03", "2025-03-03", resume=False)

        assert stats["dates_downloaded"] == 1
        assert stats["dates_skipped"] == 0

    def test_multi_date_stats(self, config, mock_cm, tmp_path):
        dl = MinuteDownloader(config, mock_cm)

        def _fake_historical(start, end, **kw):
            if start == "2025-03-01":
                return iter([self._make_record(1740000000000), self._make_record(1740000060000)])
            return iter([])

        with patch("src.data_sources.minute_downloader.PolygonEquityClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.fetch_historical.side_effect = _fake_historical

            stats = dl.download("SPY", "2025-03-01", "2025-03-02", resume=True)

        assert stats["dates_downloaded"] == 1
        assert stats["total_bars"] == 2
