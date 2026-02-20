# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Handshake module between contract discovery and bar downloading.

Sits between two Massive API calls:
  Block 1 → list_options_contracts()  (discovery)
  Block 2 → list_aggs()               (bar fetch, handled by MassiveOptionsDownloader)

ContractSelector filters Block 1's output down to the exact contracts
MassiveOptionsDownloader should fetch, using two modes:

PROD mode (paid Massive tier):
  - Calls Massive list_options_contracts() for the underlying + expiry date
  - Receives the full active contract list
  - Filters to the nearest n_calls strikes above and n_puts strikes below
    the day's SPY opening price
  - Returns filtered list — ready for MassiveOptionsDownloader

TEST mode (free tier / no contract list available):
  - prompt_once() collects selection criteria interactively at the start
    of each run cycle (not once per date — once per cycle)
  - For each date, uses OptionsTickerBuilder to compute target strikes
    and construct ticker strings mathematically from opening price
  - Returns constructed candidates — MassiveOptionsDownloader probes
    whether bars actually exist for each

Both modes return the same output schema:
  List[{ticker, strike, contract_type, expiry_date, underlying}]

Config (pipeline_v2.yaml → pipeline_v2.contract_selector):
  mode:  "test" | "prod"
  prod:
    underlying:             "SPY"
    n_calls:                2
    n_puts:                 2
    strike_increment:       0.5
    expiry_convention:      "next_trading_day"
    expiration_search_days: 5

TEST mode params come from interactive input, not config.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.data_sources.options_ticker_builder import OptionsTickerBuilder
from src.utils.logger import get_logger

logger = get_logger()

try:
    from massive import RESTClient as MassiveRESTClient  # type: ignore
    _MASSIVE_AVAILABLE = True
except ImportError:
    MassiveRESTClient = None  # type: ignore
    _MASSIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Expiry convention constants
# ---------------------------------------------------------------------------
EXPIRY_NEXT_CALENDAR = "next_calendar_day"
EXPIRY_NEXT_TRADING  = "next_trading_day"
EXPIRY_NEXT_FRIDAY   = "next_friday"
EXPIRY_FIXED         = "fixed"

_EXPIRY_LABELS = {
    "1": EXPIRY_NEXT_CALENDAR,
    "2": EXPIRY_NEXT_TRADING,
    "3": EXPIRY_NEXT_FRIDAY,
    "4": EXPIRY_FIXED,
}


class ContractSelector:
    """Filter or construct the exact option contracts to download for each date.

    Instantiate once per run, call prompt_once() for TEST mode, then call
    get_contracts(date, opening_price) for every trading date in the cycle.

    Args:
        config:    Full merged config dict.
        mode:      "test" (default) or "prod".
        api_key:   Massive.com API key — required only in PROD mode.
        _input_fn: Injectable input function for testing (default: built-in input).
    """

    MODE_PROD = "prod"
    MODE_TEST = "test"

    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = MODE_TEST,
        api_key: Optional[str] = None,
        _input_fn: Callable[[str], str] = input,
    ) -> None:
        if mode not in (self.MODE_PROD, self.MODE_TEST):
            raise ValueError(f"mode must be 'prod' or 'test', got: {mode!r}")

        self.config = config
        self.mode = mode
        self._input_fn = _input_fn
        self._test_params: Optional[Dict[str, Any]] = None

        # ---- PROD: build Massive client + read config ----
        if mode == self.MODE_PROD:
            if not _MASSIVE_AVAILABLE:
                raise ImportError(
                    "The `massive` package is required for PROD mode.\n"
                    "Install with:  pip install massive"
                )
            if not api_key:
                raise ValueError("api_key is required in PROD mode.")
            self._client = MassiveRESTClient(api_key)
            self._prod_cfg = self._load_prod_cfg()
        else:
            self._client = None
            self._prod_cfg = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Properties for external consumers (avoids accessing private attrs)
    # ------------------------------------------------------------------

    @property
    def needs_prompt(self) -> bool:
        """True if TEST mode and prompt_once() has not yet been called."""
        return self.mode == self.MODE_TEST and self._test_params is None

    @property
    def underlying(self) -> str:
        """The configured underlying ticker for the current mode/params."""
        if self.mode == self.MODE_PROD:
            return self._prod_cfg.get("underlying", "SPY")
        if self._test_params:
            return self._test_params.get("underlying", "SPY")
        return "SPY"  # default before prompt_once() is called

    # ------------------------------------------------------------------

    def prompt_once(self) -> None:
        """Collect selection criteria interactively — TEST mode only.

        Must be called once before the first get_contracts() call in TEST mode.
        Stores params in self._test_params for the entire run cycle so the
        user is not prompted again for subsequent dates.

        Raises:
            RuntimeError: If called in PROD mode.
        """
        if self.mode == self.MODE_PROD:
            raise RuntimeError("prompt_once() is only valid in TEST mode.")

        print("\n" + "=" * 60)
        print("  CONTRACT SELECTOR — TEST MODE")
        print("  Params will be used for the entire download cycle.")
        print("=" * 60)

        underlying      = self._ask("Underlying ticker",       "SPY",  str).upper()
        strike_incr     = self._ask("Strike increment (USD)",  "1.0",  float)
        n_calls         = self._ask("Calls per day",           "1",    int)
        n_puts          = self._ask("Puts per day",            "1",    int)

        print("\nExpiry convention:")
        print("  1. Next calendar day")
        print("  2. Next trading day  (default)")
        print("  3. Next Friday")
        print("  4. Fixed date — same expiry for entire cycle")
        convention_key  = self._ask("Choice", "2", str).strip()
        convention      = _EXPIRY_LABELS.get(convention_key, EXPIRY_NEXT_TRADING)

        fixed_expiry: Optional[str] = None
        if convention == EXPIRY_FIXED:
            fixed_expiry = self._ask("Fixed expiry date (YYYY-MM-DD)", "", str).strip()
            if not fixed_expiry:
                raise ValueError("A fixed expiry date is required when convention=fixed.")

        self._test_params = {
            "underlying":        underlying,
            "strike_increment":  strike_incr,
            "n_calls":           n_calls,
            "n_puts":            n_puts,
            "expiry_convention": convention,
            "fixed_expiry_date": fixed_expiry,
        }

        print("\nUsing for entire cycle:")
        print(f"  Underlying   : {underlying}")
        print(f"  Increment    : ${strike_incr}")
        print(f"  Calls/day    : {n_calls}    Puts/day: {n_puts}")
        print(f"  Expiry       : {convention}" + (
            f" → {fixed_expiry}" if fixed_expiry else ""
        ))
        print("=" * 60 + "\n")

        logger.info(
            f"ContractSelector TEST params set — underlying={underlying}, "
            f"increment={strike_incr}, calls={n_calls}, puts={n_puts}, "
            f"expiry_convention={convention}"
        )

    def get_contracts(
        self, date: str, opening_price: float
    ) -> List[Dict[str, Any]]:
        """Return the filtered/constructed contract list for one trading date.

        In PROD mode: calls Massive, filters by nearest strikes.
        In TEST mode: constructs candidates from opening price using
                      params collected by prompt_once().

        Auto-calls prompt_once() if in TEST mode and params are not yet set.

        Args:
            date:          Trading date (YYYY-MM-DD).
            opening_price: Underlying opening price for the date.

        Returns:
            List of contract dicts:
              {ticker, strike, contract_type, expiry_date, underlying}
            May be empty in PROD mode if the API returns nothing.
            Never empty in TEST mode (candidates always constructed).
        """
        if self.mode == self.MODE_PROD:
            return self._get_contracts_prod(date, opening_price)

        # TEST mode
        if self._test_params is None:
            self.prompt_once()
        return self._get_contracts_test(date, opening_price)

    # ------------------------------------------------------------------
    # PROD mode internals
    # ------------------------------------------------------------------

    def _get_contracts_prod(
        self, date: str, opening_price: float
    ) -> List[Dict[str, Any]]:
        """Call Massive list_options_contracts, filter by nearest strikes.

        Tries successive expiry dates (up to expiration_search_days) until
        at least one contract is returned by the API.
        """
        underlying  = self._prod_cfg["underlying"]
        n_calls     = self._prod_cfg["n_calls"]
        n_puts      = self._prod_cfg["n_puts"]
        increment   = self._prod_cfg["strike_increment"]
        convention  = self._prod_cfg["expiry_convention"]
        search_days = self._prod_cfg["expiration_search_days"]

        call_strikes, put_strikes = OptionsTickerBuilder.compute_strikes(
            opening_price, n_calls, n_puts, increment
        )
        strike_lo = min(put_strikes)
        strike_hi = max(call_strikes)

        for offset in range(1, search_days + 1):
            expiry = self._resolve_expiry(date, convention, offset=offset)

            try:
                raw = list(self._client.list_options_contracts(
                    underlying_ticker=underlying,
                    expiration_date=expiry,
                    strike_price_gte=strike_lo,
                    strike_price_lte=strike_hi,
                    limit=500,
                    sort="ticker",
                    order="asc",
                ))
            except Exception as exc:
                logger.warning(
                    f"[{date}] list_options_contracts error (expiry {expiry}): {exc}"
                )
                continue

            if not raw:
                logger.debug(
                    f"[{date}] No contracts from API for expiry {expiry} "
                    f"(strikes {strike_lo}–{strike_hi})"
                )
                continue

            filtered = self._filter_by_strikes(
                raw, call_strikes, put_strikes, underlying, expiry, n_calls, n_puts
            )
            logger.info(
                f"[{date}] PROD: {len(filtered)} contracts selected "
                f"(expiry {expiry}, opening {opening_price})"
            )
            return filtered

        logger.warning(
            f"[{date}] PROD: no contracts found after {search_days} expiry attempts"
        )
        return []

    def _filter_by_strikes(
        self,
        raw_contracts: list,
        call_strikes: List[float],
        put_strikes: List[float],
        underlying: str,
        expiry: str,
        n_calls: int,
        n_puts: int,
    ) -> List[Dict[str, Any]]:
        """From the raw API list, select the nearest n_calls calls and n_puts puts.

        Matches contracts whose strike_price is within 1 cent of a target strike.
        """
        result: List[Dict[str, Any]] = []

        for target_strike in call_strikes[:n_calls]:
            match = next(
                (
                    c for c in raw_contracts
                    if (getattr(c, "contract_type", "") or "").lower() == "call"
                    and abs(float(getattr(c, "strike_price", 0) or 0) - target_strike) < 0.01
                ),
                None,
            )
            if match:
                result.append(self._normalise(match, underlying, expiry))
            else:
                logger.debug(
                    f"PROD filter: no call match for strike {target_strike} "
                    f"(expiry {expiry})"
                )

        for target_strike in put_strikes[:n_puts]:
            match = next(
                (
                    c for c in raw_contracts
                    if (getattr(c, "contract_type", "") or "").lower() == "put"
                    and abs(float(getattr(c, "strike_price", 0) or 0) - target_strike) < 0.01
                ),
                None,
            )
            if match:
                result.append(self._normalise(match, underlying, expiry))
            else:
                logger.debug(
                    f"PROD filter: no put match for strike {target_strike} "
                    f"(expiry {expiry})"
                )

        return result

    @staticmethod
    def _normalise(
        contract: Any, underlying: str, expiry: str
    ) -> Dict[str, Any]:
        """Convert a Massive contract object to the standard output dict."""
        return {
            "ticker":         getattr(contract, "ticker", ""),
            "strike":         float(getattr(contract, "strike_price", 0) or 0),
            "contract_type":  (getattr(contract, "contract_type", "") or "").lower(),
            "expiry_date":    getattr(contract, "expiration_date", expiry),
            "underlying":     underlying,
        }

    # ------------------------------------------------------------------
    # TEST mode internals
    # ------------------------------------------------------------------

    def _get_contracts_test(
        self, date: str, opening_price: float
    ) -> List[Dict[str, Any]]:
        """Construct candidate ticker strings from opening price and test params."""
        p = self._test_params  # guaranteed non-None by get_contracts()

        underlying  = p["underlying"]
        increment   = p["strike_increment"]
        n_calls     = p["n_calls"]
        n_puts      = p["n_puts"]
        convention  = p["expiry_convention"]
        fixed_expiry = p["fixed_expiry_date"]

        call_strikes, put_strikes = OptionsTickerBuilder.compute_strikes(
            opening_price, n_calls, n_puts, increment
        )
        expiry = self._resolve_expiry(date, convention, fixed_date=fixed_expiry)

        result: List[Dict[str, Any]] = []

        for strike in call_strikes[:n_calls]:
            ticker = OptionsTickerBuilder.build_ticker(
                underlying, strike, "call", expiry
            )
            result.append({
                "ticker":        ticker,
                "strike":        strike,
                "contract_type": "call",
                "expiry_date":   expiry,
                "underlying":    underlying,
            })

        for strike in put_strikes[:n_puts]:
            ticker = OptionsTickerBuilder.build_ticker(
                underlying, strike, "put", expiry
            )
            result.append({
                "ticker":        ticker,
                "strike":        strike,
                "contract_type": "put",
                "expiry_date":   expiry,
                "underlying":    underlying,
            })

        logger.debug(
            f"[{date}] TEST: {len(result)} candidates constructed "
            f"(opening {opening_price}, expiry {expiry})"
        )
        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_expiry(
        self,
        date: str,
        convention: str,
        offset: int = 1,
        fixed_date: Optional[str] = None,
    ) -> str:
        """Convert a convention name to a concrete expiry date string.

        Args:
            date:       Trading date (YYYY-MM-DD).
            convention: One of the EXPIRY_* constants.
            offset:     For PROD mode multi-attempt loop — shifts the base
                        expiry forward by (offset - 1) extra days so successive
                        calls probe different expiry dates.
            fixed_date: Required when convention == EXPIRY_FIXED.

        Returns:
            Expiry date as YYYY-MM-DD.
        """
        from datetime import datetime, timedelta

        if convention == EXPIRY_FIXED:
            if not fixed_date:
                raise ValueError("fixed_date is required for EXPIRY_FIXED convention.")
            if offset > 1:
                # Shift fixed date forward for subsequent PROD attempts
                dt = datetime.strptime(fixed_date, "%Y-%m-%d") + timedelta(days=offset - 1)
                return dt.strftime("%Y-%m-%d")
            return fixed_date

        if convention == EXPIRY_NEXT_TRADING:
            base = OptionsTickerBuilder.next_trading_day(date)
        elif convention == EXPIRY_NEXT_FRIDAY:
            base = OptionsTickerBuilder.next_friday(date)
        else:  # EXPIRY_NEXT_CALENDAR (default)
            from datetime import datetime, timedelta
            dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
            base = dt.strftime("%Y-%m-%d")

        # Apply offset shift for PROD multi-attempt loop
        if offset > 1:
            from datetime import datetime, timedelta
            dt = datetime.strptime(base, "%Y-%m-%d") + timedelta(days=offset - 1)
            return dt.strftime("%Y-%m-%d")
        return base

    def _load_prod_cfg(self) -> Dict[str, Any]:
        """Read PROD mode params from config. Raises ValueError if missing."""
        cs = (
            self.config
            .get("pipeline_v2", {})
            .get("contract_selector", {})
            .get("prod", {})
        )
        underlying = cs.get("underlying")
        if not underlying:
            raise ValueError(
                "pipeline_v2.contract_selector.prod.underlying is required in PROD mode."
            )
        return {
            "underlying":           underlying,
            "n_calls":              cs.get("n_calls", 2),
            "n_puts":               cs.get("n_puts", 2),
            "strike_increment":     cs.get("strike_increment", 0.5),
            "expiry_convention":    cs.get("expiry_convention", EXPIRY_NEXT_TRADING),
            "expiration_search_days": cs.get("expiration_search_days", 5),
        }

    def _ask(self, prompt: str, default: Any, cast: type) -> Any:
        """Display a prompt with a default value, cast and return the result."""
        raw = self._input_fn(f"  {prompt:<30} [{default}]: ").strip()
        value = raw if raw else str(default)
        try:
            return cast(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid input {value!r} for {prompt!r}, using default {default!r}"
            )
            return cast(default)
