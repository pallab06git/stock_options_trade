# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for ContractSelector.

All Massive API calls are mocked — no network required.
Interactive input() is replaced with a controlled sequence via _input_fn.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch `massive` before importing the module under test
# ---------------------------------------------------------------------------
_mock_massive_cls = MagicMock()

with patch.dict(
    "sys.modules",
    {"massive": MagicMock(RESTClient=_mock_massive_cls)},
):
    from src.data_sources.contract_selector import (
        ContractSelector,
        EXPIRY_NEXT_CALENDAR,
        EXPIRY_NEXT_TRADING,
        EXPIRY_NEXT_FRIDAY,
        EXPIRY_FIXED,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "pipeline_v2": {
        "contract_selector": {
            "mode": "test",
            "prod": {
                "underlying": "SPY",
                "n_calls": 2,
                "n_puts": 2,
                "strike_increment": 0.5,
                "expiry_convention": "next_trading_day",
                "expiration_search_days": 3,
            },
        }
    }
}


def _make_selector(inputs=None, mode="test", api_key="test-key", config=None):
    """Build a ContractSelector with a controlled input sequence."""
    cfg = config or _BASE_CONFIG
    responses = iter(inputs or [])
    input_fn = lambda prompt: next(responses, "")

    with patch.dict(
        "sys.modules",
        {"massive": MagicMock(RESTClient=_mock_massive_cls)},
    ):
        return ContractSelector(cfg, mode=mode, api_key=api_key, _input_fn=input_fn)


def _make_contract(ticker, strike, contract_type="call"):
    c = MagicMock()
    c.ticker = ticker
    c.strike_price = strike
    c.contract_type = contract_type
    c.expiration_date = "2025-03-04"
    return c


# ===========================================================================
# Constructor validation
# ===========================================================================

class TestConstructor:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            _make_selector(mode="live")

    def test_prod_mode_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            _make_selector(mode="prod", api_key="")

    def test_test_mode_no_api_key_needed(self):
        sel = _make_selector(mode="test", api_key=None)
        assert sel.mode == "test"

    def test_prod_mode_missing_underlying_raises(self):
        cfg = {"pipeline_v2": {"contract_selector": {"prod": {}}}}
        with pytest.raises(ValueError, match="underlying is required"):
            _make_selector(mode="prod", api_key="key", config=cfg)


# ===========================================================================
# prompt_once — TEST mode interactive input
# ===========================================================================

class TestPromptOnce:
    def _sel_with_inputs(self, *inputs):
        return _make_selector(inputs=list(inputs))

    def test_stores_params_after_prompt(self):
        sel = self._sel_with_inputs(
            "TSLA",   # underlying
            "1.0",    # increment
            "1",      # n_calls
            "1",      # n_puts
            "2",      # expiry convention → next_trading_day
        )
        sel.prompt_once()
        p = sel._test_params
        assert p["underlying"] == "TSLA"
        assert p["strike_increment"] == pytest.approx(1.0)
        assert p["n_calls"] == 1
        assert p["n_puts"] == 1
        assert p["expiry_convention"] == EXPIRY_NEXT_TRADING
        assert p["fixed_expiry_date"] is None

    def test_defaults_used_when_input_blank(self):
        sel = self._sel_with_inputs("", "", "", "", "")
        sel.prompt_once()
        p = sel._test_params
        assert p["underlying"] == "SPY"
        assert p["strike_increment"] == pytest.approx(1.0)
        assert p["n_calls"] == 1
        assert p["n_puts"] == 1
        assert p["expiry_convention"] == EXPIRY_NEXT_TRADING

    def test_next_calendar_day_convention(self):
        sel = self._sel_with_inputs("SPY", "1.0", "1", "1", "1")
        sel.prompt_once()
        assert sel._test_params["expiry_convention"] == EXPIRY_NEXT_CALENDAR

    def test_next_friday_convention(self):
        sel = self._sel_with_inputs("SPY", "1.0", "1", "1", "3")
        sel.prompt_once()
        assert sel._test_params["expiry_convention"] == EXPIRY_NEXT_FRIDAY

    def test_fixed_date_convention_stores_date(self):
        sel = self._sel_with_inputs(
            "SPY", "1.0", "1", "1",
            "4",             # fixed
            "2025-12-19",    # fixed expiry date
        )
        sel.prompt_once()
        p = sel._test_params
        assert p["expiry_convention"] == EXPIRY_FIXED
        assert p["fixed_expiry_date"] == "2025-12-19"

    def test_fixed_date_missing_raises(self):
        sel = self._sel_with_inputs("SPY", "1.0", "1", "1", "4", "")
        with pytest.raises(ValueError, match="fixed expiry date is required"):
            sel.prompt_once()

    def test_underlying_uppercased(self):
        sel = self._sel_with_inputs("tsla", "1.0", "1", "1", "2")
        sel.prompt_once()
        assert sel._test_params["underlying"] == "TSLA"

    def test_raises_in_prod_mode(self):
        sel = _make_selector(mode="prod", api_key="key")
        with pytest.raises(RuntimeError, match="TEST mode"):
            sel.prompt_once()


# ===========================================================================
# _resolve_expiry
# ===========================================================================

class TestResolveExpiry:
    def _sel(self):
        return _make_selector()

    def test_next_trading_day_monday(self):
        sel = self._sel()
        result = sel._resolve_expiry("2025-03-03", EXPIRY_NEXT_TRADING)
        assert result == "2025-03-04"

    def test_next_trading_day_friday_skips_weekend(self):
        sel = self._sel()
        result = sel._resolve_expiry("2025-03-07", EXPIRY_NEXT_TRADING)
        assert result == "2025-03-10"

    def test_next_friday(self):
        sel = self._sel()
        result = sel._resolve_expiry("2025-03-03", EXPIRY_NEXT_FRIDAY)
        assert result == "2025-03-07"

    def test_next_calendar_day(self):
        sel = self._sel()
        result = sel._resolve_expiry("2025-03-03", EXPIRY_NEXT_CALENDAR)
        assert result == "2025-03-04"

    def test_fixed_date_offset_1(self):
        sel = self._sel()
        result = sel._resolve_expiry(
            "2025-03-03", EXPIRY_FIXED, offset=1, fixed_date="2025-12-19"
        )
        assert result == "2025-12-19"

    def test_fixed_date_offset_2_shifts_forward(self):
        sel = self._sel()
        result = sel._resolve_expiry(
            "2025-03-03", EXPIRY_FIXED, offset=2, fixed_date="2025-12-19"
        )
        assert result == "2025-12-20"

    def test_offset_shifts_base_expiry(self):
        sel = self._sel()
        r1 = sel._resolve_expiry("2025-03-03", EXPIRY_NEXT_TRADING, offset=1)
        r2 = sel._resolve_expiry("2025-03-03", EXPIRY_NEXT_TRADING, offset=2)
        from datetime import datetime
        diff = (
            datetime.strptime(r2, "%Y-%m-%d") - datetime.strptime(r1, "%Y-%m-%d")
        ).days
        assert diff == 1

    def test_fixed_date_without_date_raises(self):
        sel = self._sel()
        with pytest.raises(ValueError, match="fixed_date is required"):
            sel._resolve_expiry("2025-03-03", EXPIRY_FIXED)


# ===========================================================================
# get_contracts — TEST mode
# ===========================================================================

class TestGetContractsTestMode:
    def _primed_selector(self, underlying="SPY", increment=1.0,
                         n_calls=1, n_puts=1, convention="2"):
        sel = _make_selector(inputs=[
            underlying, str(increment), str(n_calls), str(n_puts), convention
        ])
        sel.prompt_once()
        return sel

    def test_returns_n_calls_plus_n_puts_contracts(self):
        sel = self._primed_selector(n_calls=1, n_puts=1)
        contracts = sel.get_contracts("2025-03-03", 600.25)
        assert len(contracts) == 2

    def test_correct_contract_types(self):
        sel = self._primed_selector(n_calls=1, n_puts=1)
        contracts = sel.get_contracts("2025-03-03", 600.25)
        types = {c["contract_type"] for c in contracts}
        assert types == {"call", "put"}

    def test_call_strike_above_opening(self):
        sel = self._primed_selector(n_calls=1, n_puts=0, increment=1.0)
        contracts = sel.get_contracts("2025-03-03", 600.25)
        call = contracts[0]
        assert call["contract_type"] == "call"
        assert call["strike"] > 600.25

    def test_put_strike_at_or_below_opening(self):
        sel = self._primed_selector(n_calls=0, n_puts=1, increment=1.0)
        contracts = sel.get_contracts("2025-03-03", 600.25)
        put = contracts[0]
        assert put["contract_type"] == "put"
        assert put["strike"] <= 600.25

    def test_ticker_format_correct(self):
        sel = self._primed_selector(underlying="SPY", increment=1.0,
                                    n_calls=1, n_puts=0, convention="2")
        contracts = sel.get_contracts("2025-03-03", 600.25)
        # 2025-03-03 → next_trading_day → 2025-03-04 → "250304"
        # strike 601.0 → "00601000"
        assert contracts[0]["ticker"] == "O:SPY250304C00601000"

    def test_expiry_next_trading_day(self):
        sel = self._primed_selector(convention="2")   # next_trading_day
        contracts = sel.get_contracts("2025-03-07", 600.0)  # Friday
        # next trading day after Friday = Monday 2025-03-10
        assert all(c["expiry_date"] == "2025-03-10" for c in contracts)

    def test_expiry_next_friday(self):
        sel = self._primed_selector(convention="3")   # next_friday
        contracts = sel.get_contracts("2025-03-03", 600.0)
        assert all(c["expiry_date"] == "2025-03-07" for c in contracts)

    def test_expiry_next_calendar_day(self):
        sel = self._primed_selector(convention="1")   # next_calendar_day
        contracts = sel.get_contracts("2025-03-03", 600.0)
        assert all(c["expiry_date"] == "2025-03-04" for c in contracts)

    def test_expiry_fixed_same_across_dates(self):
        sel = _make_selector(inputs=[
            "SPY", "1.0", "1", "1", "4", "2025-12-19"
        ])
        sel.prompt_once()
        c1 = sel.get_contracts("2025-03-03", 600.0)
        c2 = sel.get_contracts("2025-03-04", 601.0)
        assert all(c["expiry_date"] == "2025-12-19" for c in c1 + c2)

    def test_output_schema_has_all_keys(self):
        sel = self._primed_selector()
        contracts = sel.get_contracts("2025-03-03", 600.0)
        expected = {"ticker", "strike", "contract_type", "expiry_date", "underlying"}
        for c in contracts:
            assert set(c.keys()) == expected

    def test_auto_prompts_if_not_called(self):
        """get_contracts should auto-call prompt_once() if params not set."""
        sel = _make_selector(inputs=["SPY", "1.0", "1", "1", "2"])
        # Do NOT call prompt_once() — get_contracts should do it
        contracts = sel.get_contracts("2025-03-03", 600.25)
        assert len(contracts) == 2

    def test_same_params_used_across_entire_cycle(self):
        """Params from prompt_once() must persist across multiple get_contracts calls."""
        sel = self._primed_selector(underlying="TSLA", increment=1.0,
                                    n_calls=1, n_puts=1)
        dates = ["2025-03-03", "2025-03-04", "2025-03-05"]
        prices = [250.25, 251.75, 249.50]
        for date, price in zip(dates, prices):
            contracts = sel.get_contracts(date, price)
            assert all(c["underlying"] == "TSLA" for c in contracts)

    def test_different_opening_prices_give_different_strikes(self):
        sel = self._primed_selector(n_calls=1, n_puts=1, increment=1.0)
        c1 = sel.get_contracts("2025-03-03", 600.0)
        c2 = sel.get_contracts("2025-03-04", 605.0)
        strikes1 = {c["strike"] for c in c1}
        strikes2 = {c["strike"] for c in c2}
        assert strikes1 != strikes2


# ===========================================================================
# get_contracts — PROD mode
# ===========================================================================

class TestGetContractsProdMode:
    def _prod_selector(self, contracts_by_expiry=None):
        """Build a PROD selector with a mocked Massive client."""
        sel = _make_selector(mode="prod", api_key="test-key")

        def _list_contracts(**kwargs):
            expiry = kwargs.get("expiration_date", "")
            return iter(contracts_by_expiry.get(expiry, []))

        sel._client.list_options_contracts.side_effect = (
            lambda **kw: iter(contracts_by_expiry.get(
                kw.get("expiration_date", ""), []
            ))
        )
        return sel

    def test_returns_filtered_contracts(self):
        c_call = _make_contract("O:SPY250304C00601000", 601.0, "call")
        c_put  = _make_contract("O:SPY250304P00600000", 600.0, "put")
        sel = self._prod_selector({"2025-03-04": [c_call, c_put]})
        result = sel.get_contracts("2025-03-03", 600.25)
        assert len(result) == 2
        tickers = {r["ticker"] for r in result}
        assert "O:SPY250304C00601000" in tickers
        assert "O:SPY250304P00600000" in tickers

    def test_tries_next_expiry_if_first_empty(self):
        c_call = _make_contract("O:SPY250305C00601000", 601.0, "call")
        c_put  = _make_contract("O:SPY250305P00600000", 600.0, "put")
        # First expiry (2025-03-04) empty, second (2025-03-05) has contracts
        sel = self._prod_selector({
            "2025-03-04": [],
            "2025-03-05": [c_call, c_put],
        })
        result = sel.get_contracts("2025-03-03", 600.25)
        assert len(result) == 2

    def test_returns_empty_when_all_expiries_fail(self):
        sel = self._prod_selector({})   # always empty
        result = sel.get_contracts("2025-03-03", 600.25)
        assert result == []

    def test_output_schema_matches_test_mode(self):
        c = _make_contract("O:SPY250304C00601000", 601.0, "call")
        sel = self._prod_selector({"2025-03-04": [c]})
        result = sel.get_contracts("2025-03-03", 600.25)
        if result:
            expected = {"ticker", "strike", "contract_type", "expiry_date", "underlying"}
            assert set(result[0].keys()) == expected

    def test_api_exception_skips_to_next_expiry(self):
        c = _make_contract("O:SPY250305C00601000", 601.0, "call")
        sel = _make_selector(mode="prod", api_key="key")
        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("connection error")
            return iter([c])

        sel._client.list_options_contracts.side_effect = _side_effect
        result = sel.get_contracts("2025-03-03", 600.25)
        assert call_count == 2   # retried on second expiry
