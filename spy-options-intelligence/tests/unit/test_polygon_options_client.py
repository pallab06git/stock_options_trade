# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PolygonOptionsClient."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data_sources.polygon_options_client import PolygonOptionsClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    underlying_ticker="SPY",
    strike_range_pct=0.01,
    max_contracts=100,
    expiration_lookahead_days=1,
):
    """Create a minimal config dict for PolygonOptionsClient."""
    return {
        "polygon": {
            "api_key": "pk_test_12345678",
            "options": {
                "underlying_ticker": underlying_ticker,
                "strike_range_pct": strike_range_pct,
                "max_contracts": max_contracts,
                "expiration_lookahead_days": expiration_lookahead_days,
            },
        },
        "retry": {
            "polygon": {
                "max_attempts": 1,
                "initial_wait_seconds": 0.01,
                "max_wait_seconds": 0.05,
                "exponential_base": 2,
                "jitter": False,
                "retry_on_status_codes": [500, 502, 503, 504, 429],
            },
        },
    }


def _make_connection_manager():
    """Create a mock ConnectionManager."""
    cm = MagicMock()
    cm.acquire_rate_limit.return_value = True
    cm.get_rest_client.return_value = MagicMock()
    return cm


def _make_contract(
    ticker="O:SPY260210C00430000",
    underlying_ticker="SPY",
    strike_price=430.0,
    expiration_date="2026-02-10",
    contract_type="call",
    exercise_style="american",
    primary_exchange="CBOE",
    shares_per_contract=100,
):
    """Create a mock Polygon OptionsContract object."""
    return SimpleNamespace(
        ticker=ticker,
        underlying_ticker=underlying_ticker,
        strike_price=strike_price,
        expiration_date=expiration_date,
        contract_type=contract_type,
        exercise_style=exercise_style,
        primary_exchange=primary_exchange,
        shares_per_contract=shares_per_contract,
    )


# ---------------------------------------------------------------------------
# Tests: fetch_opening_price
# ---------------------------------------------------------------------------


class TestFetchOpeningPrice:

    def test_fetch_opening_price(self):
        """Returns the open field from get_daily_open_close."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_daily_open_close.return_value = (
            SimpleNamespace(open=428.50, close=430.00)
        )

        client = PolygonOptionsClient(config, cm)
        price = client.fetch_opening_price("2026-02-09")

        assert price == 428.50
        cm.get_rest_client.return_value.get_daily_open_close.assert_called_once_with(
            ticker="SPY", date="2026-02-09"
        )

    def test_fetch_opening_price_no_data(self):
        """Raises ValueError when API returns None."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_daily_open_close.return_value = None

        client = PolygonOptionsClient(config, cm)
        with pytest.raises(ValueError, match="No daily open/close data"):
            client.fetch_opening_price("2026-02-09")

    def test_fetch_opening_price_no_open_field(self):
        """Raises ValueError when result has no open attribute."""
        config = _make_config()
        cm = _make_connection_manager()
        # Object without .open attribute
        cm.get_rest_client.return_value.get_daily_open_close.return_value = (
            SimpleNamespace(close=430.00)
        )

        client = PolygonOptionsClient(config, cm)
        with pytest.raises(ValueError, match="No opening price"):
            client.fetch_opening_price("2026-02-09")


# ---------------------------------------------------------------------------
# Tests: strike range and discovery
# ---------------------------------------------------------------------------


class TestStrikeRange:

    def test_strike_range_calculation(self):
        """Verify ±1% strike range math."""
        config = _make_config(strike_range_pct=0.01)
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.list_options_contracts.return_value = []

        client = PolygonOptionsClient(config, cm)
        opening_price = 428.50

        client.discover_contracts("2026-02-09", opening_price)

        expected_lower = round(428.50 * 0.99, 2)  # 424.22
        expected_upper = round(428.50 * 1.01, 2)  # 432.79

        call_kwargs = (
            cm.get_rest_client.return_value.list_options_contracts.call_args
        )
        assert call_kwargs.kwargs["strike_price_gte"] == expected_lower
        assert call_kwargs.kwargs["strike_price_lte"] == expected_upper


class TestDiscoverContracts:

    def test_discover_contracts(self):
        """Mock list_options_contracts, verify transform and return."""
        config = _make_config()
        cm = _make_connection_manager()

        raw_contracts = [
            _make_contract(ticker="O:SPY260210C00428000", strike_price=428.0),
            _make_contract(ticker="O:SPY260210P00429000", strike_price=429.0, contract_type="put"),
        ]
        cm.get_rest_client.return_value.list_options_contracts.return_value = raw_contracts

        client = PolygonOptionsClient(config, cm)
        result = client.discover_contracts("2026-02-09", 428.50)

        assert len(result) == 2
        assert result[0]["ticker"] == "O:SPY260210C00428000"
        assert result[0]["strike_price"] == 428.0
        assert result[1]["contract_type"] == "put"

    def test_discover_contracts_empty(self):
        """No contracts in range returns empty list."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.list_options_contracts.return_value = []

        client = PolygonOptionsClient(config, cm)
        result = client.discover_contracts("2026-02-09", 428.50)

        assert result == []

    def test_discover_contracts_capped(self):
        """Respects max_contracts limit."""
        config = _make_config(max_contracts=3)
        cm = _make_connection_manager()

        raw_contracts = [_make_contract(strike_price=427.0 + i) for i in range(10)]
        cm.get_rest_client.return_value.list_options_contracts.return_value = raw_contracts

        client = PolygonOptionsClient(config, cm)
        result = client.discover_contracts("2026-02-09", 428.50)

        assert len(result) == 3

    def test_expiration_date_calculation(self):
        """Lookahead days are applied to compute expiration date."""
        config = _make_config(expiration_lookahead_days=2)
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.list_options_contracts.return_value = []

        client = PolygonOptionsClient(config, cm)
        client.discover_contracts("2026-02-09", 428.50)

        call_kwargs = (
            cm.get_rest_client.return_value.list_options_contracts.call_args
        )
        # 2026-02-09 + 2 days = 2026-02-11
        assert call_kwargs.kwargs["expiration_date"] == "2026-02-11"


# ---------------------------------------------------------------------------
# Tests: transform
# ---------------------------------------------------------------------------


class TestTransformContract:

    def test_transform_contract(self):
        """All fields are correctly mapped."""
        contract = _make_contract(
            ticker="O:SPY260210C00430000",
            underlying_ticker="SPY",
            strike_price=430.0,
            expiration_date="2026-02-10",
            contract_type="call",
            exercise_style="american",
            primary_exchange="CBOE",
            shares_per_contract=100,
        )

        result = PolygonOptionsClient._transform_contract(contract)

        assert result == {
            "ticker": "O:SPY260210C00430000",
            "underlying_ticker": "SPY",
            "strike_price": 430.0,
            "expiration_date": "2026-02-10",
            "contract_type": "call",
            "exercise_style": "american",
            "primary_exchange": "CBOE",
            "shares_per_contract": 100,
        }


# ---------------------------------------------------------------------------
# Tests: save and load
# ---------------------------------------------------------------------------


class TestSaveContracts:

    def test_save_contracts(self, tmp_path, monkeypatch):
        """Writes JSON to correct path."""
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonOptionsClient(config, cm)

        # Redirect output to tmp_path
        monkeypatch.chdir(tmp_path)

        contracts = [
            {"ticker": "O:SPY260210C00430000", "strike_price": 430.0},
            {"ticker": "O:SPY260210P00429000", "strike_price": 429.0},
        ]
        path = client.save_contracts(contracts, "2026-02-09")

        assert path.exists()
        assert path.name == "2026-02-09_contracts.json"

        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["ticker"] == "O:SPY260210C00430000"


class TestLoadContracts:

    def test_load_contracts(self, tmp_path, monkeypatch):
        """Reads JSON back from the expected path."""
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonOptionsClient(config, cm)

        monkeypatch.chdir(tmp_path)

        # Create file in expected location
        contracts_dir = tmp_path / "data" / "raw" / "options" / "contracts"
        contracts_dir.mkdir(parents=True)
        contracts = [{"ticker": "O:SPY260210C00430000", "strike_price": 430.0}]
        with open(contracts_dir / "2026-02-09_contracts.json", "w") as f:
            json.dump(contracts, f)

        result = client.load_contracts("2026-02-09")

        assert len(result) == 1
        assert result[0]["ticker"] == "O:SPY260210C00430000"

    def test_load_contracts_not_found(self, tmp_path, monkeypatch):
        """Raises FileNotFoundError when file doesn't exist."""
        config = _make_config()
        cm = _make_connection_manager()
        client = PolygonOptionsClient(config, cm)

        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="No contract file for"):
            client.load_contracts("2026-02-09")


# ---------------------------------------------------------------------------
# Tests: rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:

    def test_rate_limit_acquired_on_fetch(self):
        """acquire_rate_limit called when fetching opening price."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.get_daily_open_close.return_value = (
            SimpleNamespace(open=428.50)
        )

        client = PolygonOptionsClient(config, cm)
        client.fetch_opening_price("2026-02-09")

        cm.acquire_rate_limit.assert_called_with(source="options")

    def test_rate_limit_acquired_on_discover(self):
        """acquire_rate_limit called when discovering contracts."""
        config = _make_config()
        cm = _make_connection_manager()
        cm.get_rest_client.return_value.list_options_contracts.return_value = []

        client = PolygonOptionsClient(config, cm)
        client.discover_contracts("2026-02-09", 428.50)

        cm.acquire_rate_limit.assert_called_with(source="options")
