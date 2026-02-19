# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for TargetedOptionsDownloader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_sources.targeted_options_downloader import TargetedOptionsDownloader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return {
        "polygon": {
            "api_key": "test",
            "equities": {"SPY": {"multiplier": 1}},
        },
        "pipeline_v2": {
            "options": {
                "n_calls": 2,
                "n_puts": 2,
                "discovery_range_pct": 0.05,
                "expiration_search_days": 3,
            }
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
    cm.get_rest_client.return_value = MagicMock()
    return cm


def _make_spy_df(tmp_path, date="2025-03-03", open_price=600.0):
    """Write a minimal SPY parquet for the given date."""
    spy_dir = tmp_path / "raw" / "spy"
    spy_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "timestamp": 1740920400000,  # ~09:00 ET
        "open": open_price,
        "high": open_price + 1,
        "low": open_price - 1,
        "close": open_price + 0.5,
        "volume": 5000,
        "source": "spy",
    }])
    df.to_parquet(spy_dir / f"{date}.parquet", index=False)


def _make_contract(ticker, strike, ctype):
    c = MagicMock()
    c.ticker = ticker
    c.strike_price = strike
    c.contract_type = ctype
    c.expiration_date = "2025-03-05"
    c.underlying_ticker = "SPY"
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetOpeningPrice:
    def test_reads_first_bar(self, config, mock_cm, tmp_path):
        _make_spy_df(tmp_path, open_price=612.34)
        dl = TargetedOptionsDownloader(config, mock_cm)
        price = dl.get_opening_price("2025-03-03")
        assert price == pytest.approx(612.34)

    def test_missing_file_raises(self, config, mock_cm):
        dl = TargetedOptionsDownloader(config, mock_cm)
        with pytest.raises(FileNotFoundError):
            dl.get_opening_price("2025-01-01")


class TestDiscoverTargeted:
    def test_selects_n_calls_n_puts(self, config, mock_cm, tmp_path):
        contracts = [
            _make_contract("O:SPY250305C00595000", 595.0, "call"),
            _make_contract("O:SPY250305C00600000", 600.0, "call"),
            _make_contract("O:SPY250305C00605000", 605.0, "call"),  # above open
            _make_contract("O:SPY250305C00610000", 610.0, "call"),  # above open
            _make_contract("O:SPY250305P00590000", 590.0, "put"),   # below open
            _make_contract("O:SPY250305P00585000", 585.0, "put"),   # below open
            _make_contract("O:SPY250305P00580000", 580.0, "put"),   # below open
        ]
        mock_cm.get_rest_client.return_value.list_options_contracts.return_value = contracts

        dl = TargetedOptionsDownloader(config, mock_cm)
        result = dl.discover_targeted("2025-03-03", 603.0)

        calls = [c for c in result if c["contract_type"] == "call"]
        puts = [c for c in result if c["contract_type"] == "put"]
        assert len(calls) == 2  # n_calls
        assert len(puts) == 2   # n_puts
        # Calls should have lowest strikes above opening
        assert all(c["strike_price"] > 603.0 for c in calls)
        # Puts should have highest strikes below opening
        assert all(p["strike_price"] < 603.0 for p in puts)

    def test_empty_result_on_no_contracts(self, config, mock_cm):
        mock_cm.get_rest_client.return_value.list_options_contracts.return_value = []
        dl = TargetedOptionsDownloader(config, mock_cm)
        result = dl.discover_targeted("2025-03-03", 600.0)
        assert result == []

    def test_tries_multiple_expirations(self, config, mock_cm):
        rest = mock_cm.get_rest_client.return_value
        # First expiry returns nothing, second returns contracts
        call = _make_contract("O:SPY250305C00605000", 605.0, "call")
        put = _make_contract("O:SPY250305P00595000", 595.0, "put")
        rest.list_options_contracts.side_effect = [[], [call, put]]

        dl = TargetedOptionsDownloader(config, mock_cm)
        result = dl.discover_targeted("2025-03-03", 600.0)
        assert len(result) > 0
        # Called twice (two expiration attempts)
        assert rest.list_options_contracts.call_count == 2


class TestTransformContract:
    def test_transform(self, config, mock_cm):
        c = _make_contract("O:SPY250305C00605000", 605.0, "call")
        dl = TargetedOptionsDownloader(config, mock_cm)
        result = dl._transform_contract(c)
        assert result["ticker"] == "O:SPY250305C00605000"
        assert result["strike_price"] == 605.0
        assert result["contract_type"] == "call"
        assert result["expiration_date"] == "2025-03-05"


class TestDownloadMinute:
    def test_writes_parquet(self, config, mock_cm, tmp_path):
        agg = MagicMock()
        agg.timestamp = 1740923400000
        agg.open = 10.0
        agg.high = 11.0
        agg.low = 9.5
        agg.close = 10.5
        agg.volume = 100
        agg.vwap = 10.2
        agg.transactions = 5

        mock_cm.get_rest_client.return_value.get_aggs.return_value = [agg]

        dl = TargetedOptionsDownloader(config, mock_cm)
        bars = dl.download_minute("O:SPY250305C00605000", "2025-03-03")

        assert bars == 1
        out_path = (
            tmp_path / "raw" / "options" / "minute"
            / "O_SPY250305C00605000" / "2025-03-03.parquet"
        )
        assert out_path.exists()

    def test_no_data_returns_zero(self, config, mock_cm):
        mock_cm.get_rest_client.return_value.get_aggs.return_value = []
        dl = TargetedOptionsDownloader(config, mock_cm)
        bars = dl.download_minute("O:SPY250305C00605000", "2025-03-03")
        assert bars == 0


class TestRun:
    def test_skips_missing_spy(self, config, mock_cm, tmp_path):
        dl = TargetedOptionsDownloader(config, mock_cm)
        stats = dl.run("2025-03-03", "2025-03-03")
        assert stats["dates_skipped"] == 1
        assert stats["dates_processed"] == 0

    def test_processes_with_spy(self, config, mock_cm, tmp_path):
        _make_spy_df(tmp_path, open_price=600.0)

        call = _make_contract("O:SPY250305C00605000", 605.0, "call")
        put = _make_contract("O:SPY250305P00595000", 595.0, "put")
        mock_cm.get_rest_client.return_value.list_options_contracts.return_value = [call, put]

        agg = MagicMock()
        agg.timestamp = 1740923400000
        agg.open = agg.high = agg.low = agg.close = 10.0
        agg.volume = 100
        agg.vwap = 10.0
        agg.transactions = 5
        mock_cm.get_rest_client.return_value.get_aggs.return_value = [agg]

        dl = TargetedOptionsDownloader(config, mock_cm)
        stats = dl.run("2025-03-03", "2025-03-03")

        assert stats["dates_processed"] == 1
        assert stats["contracts_found"] >= 1
        assert stats["total_bars"] >= 1

    def test_discovery_error_skips_date(self, config, mock_cm, tmp_path):
        """429 / other errors in discover_targeted skip the date instead of crashing."""
        _make_spy_df(tmp_path, open_price=600.0)

        mock_cm.get_rest_client.return_value.list_options_contracts.side_effect = (
            RuntimeError("too many 429 error responses")
        )

        dl = TargetedOptionsDownloader(config, mock_cm)
        stats = dl.run("2025-03-03", "2025-03-03")

        assert stats["dates_skipped"] == 1
        assert stats["dates_processed"] == 0

    def test_download_minute_error_skips_contract(self, config, mock_cm, tmp_path):
        """A failed minute download for one contract does not abort the date."""
        _make_spy_df(tmp_path, open_price=600.0)

        call = _make_contract("O:SPY250305C00605000", 605.0, "call")
        put = _make_contract("O:SPY250305P00595000", 595.0, "put")
        mock_cm.get_rest_client.return_value.list_options_contracts.return_value = [call, put]

        # get_aggs raises on the call contract, succeeds on the put
        agg = MagicMock()
        agg.timestamp = 1740923400000
        agg.open = agg.high = agg.low = agg.close = 10.0
        agg.volume = 100
        agg.vwap = 10.0
        agg.transactions = 5
        mock_cm.get_rest_client.return_value.get_aggs.side_effect = [
            RuntimeError("rate limited"),
            [agg],
        ]

        dl = TargetedOptionsDownloader(config, mock_cm)
        stats = dl.run("2025-03-03", "2025-03-03")

        # Date is still marked processed even though one contract failed
        assert stats["dates_processed"] == 1
        assert stats["contracts_found"] == 2
        # Only 1 bar written (the put succeeded)
        assert stats["total_bars"] == 1
