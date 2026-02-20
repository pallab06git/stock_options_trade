# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Pure-math options ticker utilities — no API calls, no I/O, no config.

Provides four stateless helpers consumed by ContractSelector in both
PROD and TEST modes, and usable standalone for any underlying.

  build_ticker      — construct the standard O: ticker string
  compute_strikes   — derive call/put strikes from an opening price
  next_trading_day  — first weekday strictly after a given date (Mon-Fri)
  next_friday       — first Friday strictly after a given date

Ticker format: O:{underlying}{YYMMDD}{C|P}{XXXXXXXX}
  YYMMDD   — expiry as 2-digit year, 2-digit month, 2-digit day
  C | P    — call or put
  XXXXXXXX — strike price × 1000, zero-padded to 8 digits

Examples:
  SPY  $650.00 call  expiry 2025-12-19 → O:SPY251219C00650000
  SPY  $600.50 put   expiry 2025-03-07 → O:SPY250307P00600500
  TSLA $250.00 call  expiry 2025-03-07 → O:TSLA250307C00250000
  XSP  $520.00 put   expiry 2025-03-28 → O:XSP250328P00520000
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple


class OptionsTickerBuilder:
    """Stateless utility for constructing options ticker strings.

    All methods are @staticmethod — no instantiation or config needed.
    The class wrapper keeps the namespace tidy and makes mocking straightforward
    in tests that depend on this module.
    """

    @staticmethod
    def build_ticker(
        underlying: str,
        strike: float,
        contract_type: str,
        expiry_date: str,
    ) -> str:
        """Construct the standard options ticker string.

        Args:
            underlying:    Underlying ticker symbol (e.g. "SPY", "TSLA", "XSP").
                           Converted to uppercase automatically.
            strike:        Strike price in USD (e.g. 650.0, 600.5).
            contract_type: "call" or "put" (case-insensitive; only first char used).
            expiry_date:   Expiry date as YYYY-MM-DD.

        Returns:
            Ticker string conforming to the O: options format.

        Examples:
            >>> OptionsTickerBuilder.build_ticker("SPY", 650.0, "call", "2025-12-19")
            'O:SPY251219C00650000'
            >>> OptionsTickerBuilder.build_ticker("SPY", 600.5, "put", "2025-03-07")
            'O:SPY250307P00600500'
            >>> OptionsTickerBuilder.build_ticker("TSLA", 250.0, "call", "2025-03-07")
            'O:TSLA250307C00250000'
            >>> OptionsTickerBuilder.build_ticker("SPY", 1.0, "call", "2025-01-01")
            'O:SPY250101C00001000'
        """
        dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        yymmdd = dt.strftime("%y%m%d")                     # e.g. "251219"
        cp = "C" if contract_type.upper().startswith("C") else "P"
        strike_int = round(strike * 1000)                  # $650.00 → 650000
        strike_str = f"{strike_int:08d}"                   # → "00650000"
        return f"O:{underlying.upper()}{yymmdd}{cp}{strike_str}"

    @staticmethod
    def compute_strikes(
        opening_price: float,
        n_calls: int,
        n_puts: int,
        strike_increment: float,
    ) -> Tuple[List[float], List[float]]:
        """Derive call and put strike prices from an opening price.

        Call strikes: n_calls multiples of strike_increment strictly above
                      opening_price, sorted nearest-first.
        Put  strikes: n_puts  multiples of strike_increment at or below
                      opening_price, sorted nearest-first (highest first).

        Boundary rule: if opening_price falls exactly on a multiple of
        strike_increment, calls start one full step above so that the call
        strike is always strictly above the opening price.

        Args:
            opening_price:    Underlying price at market open.
            n_calls:          Number of call strike levels to return.
            n_puts:           Number of put  strike levels to return.
            strike_increment: Dollar step between adjacent strikes
                              (e.g. 0.5 for SPY, 1.0 for TSLA, 5.0 for XSP).

        Returns:
            Tuple of (call_strikes, put_strikes).

        Examples:
            Between increments:
              compute_strikes(600.25, 2, 2, 1.0) → ([601.0, 602.0], [600.0, 599.0])
            Exactly on boundary:
              compute_strikes(600.0,  2, 2, 1.0) → ([601.0, 602.0], [600.0, 599.0])
            Half-dollar increment:
              compute_strikes(600.25, 2, 2, 0.5) → ([600.5, 601.0], [600.0, 599.5])
        """
        inc = strike_increment

        # Lowest call strike: first multiple of inc strictly above opening_price
        call_base = math.ceil(opening_price / inc) * inc
        if call_base <= opening_price + 1e-9:   # exact boundary → step up once
            call_base += inc
        call_base = round(call_base, 2)
        call_strikes = [round(call_base + i * inc, 2) for i in range(n_calls)]

        # Highest put strike: highest multiple of inc at or below opening_price
        put_base = math.floor(opening_price / inc) * inc
        put_base = round(put_base, 2)
        put_strikes = [round(put_base - i * inc, 2) for i in range(n_puts)]

        return call_strikes, put_strikes

    @staticmethod
    def next_trading_day(date: str) -> str:
        """Return the first weekday (Mon–Fri) strictly after *date*.

        Weekday-only logic — does not account for market holidays.
        Holiday awareness can be layered on top by callers if needed.

        Args:
            date: Reference date as YYYY-MM-DD.

        Returns:
            Next weekday as YYYY-MM-DD.

        Examples:
            "2025-03-03" (Mon) → "2025-03-04" (Tue)
            "2025-03-07" (Fri) → "2025-03-10" (Mon)
            "2025-03-08" (Sat) → "2025-03-10" (Mon)
            "2025-03-09" (Sun) → "2025-03-10" (Mon)
        """
        dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        while dt.weekday() >= 5:        # 5 = Saturday, 6 = Sunday
            dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def next_friday(date: str) -> str:
        """Return the first Friday strictly after *date*.

        Args:
            date: Reference date as YYYY-MM-DD.

        Returns:
            Next Friday as YYYY-MM-DD.

        Examples:
            "2025-03-03" (Mon) → "2025-03-07" (that week's Fri)
            "2025-03-07" (Fri) → "2025-03-14" (following Fri)
            "2025-03-08" (Sat) → "2025-03-14" (following Fri)
        """
        dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        while dt.weekday() != 4:        # 4 = Friday
            dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
