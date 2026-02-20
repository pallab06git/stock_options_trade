# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for OptionsTickerBuilder.

Pure-math module — no mocking required.
"""

import pytest

from src.data_sources.options_ticker_builder import OptionsTickerBuilder as OTB


# ===========================================================================
# build_ticker
# ===========================================================================

class TestBuildTicker:

    # --- call contracts ---

    def test_spy_call_whole_dollar(self):
        assert OTB.build_ticker("SPY", 650.0, "call", "2025-12-19") == "O:SPY251219C00650000"

    def test_spy_call_half_dollar(self):
        assert OTB.build_ticker("SPY", 600.5, "call", "2025-03-07") == "O:SPY250307C00600500"

    def test_spy_call_low_strike(self):
        # $1.00 → strike_int=1000 → "00001000"
        assert OTB.build_ticker("SPY", 1.0, "call", "2025-01-01") == "O:SPY250101C00001000"

    # --- put contracts ---

    def test_spy_put_whole_dollar(self):
        assert OTB.build_ticker("SPY", 649.0, "put", "2025-12-19") == "O:SPY251219P00649000"

    def test_spy_put_half_dollar(self):
        assert OTB.build_ticker("SPY", 600.5, "put", "2025-03-07") == "O:SPY250307P00600500"

    # --- other underlyings ---

    def test_tsla_call(self):
        assert OTB.build_ticker("TSLA", 250.0, "call", "2025-03-07") == "O:TSLA250307C00250000"

    def test_xsp_put_five_dollar_strike(self):
        assert OTB.build_ticker("XSP", 520.0, "put", "2025-03-28") == "O:XSP250328P00520000"

    # --- case insensitivity ---

    def test_contract_type_uppercase(self):
        assert OTB.build_ticker("SPY", 600.0, "CALL", "2025-03-07") == "O:SPY250307C00600000"

    def test_contract_type_mixed_case(self):
        assert OTB.build_ticker("SPY", 600.0, "Put", "2025-03-07") == "O:SPY250307P00600000"

    def test_underlying_lowercase_normalised(self):
        assert OTB.build_ticker("spy", 600.0, "call", "2025-03-07") == "O:SPY250307C00600000"

    # --- strike encoding edge cases ---

    def test_strike_zero_padded_to_8_digits(self):
        # $10.00 → 10000 → "00010000"
        assert OTB.build_ticker("SPY", 10.0, "call", "2025-01-01") == "O:SPY250101C00010000"

    def test_high_strike_fits_8_digits(self):
        # $9999.00 → 9999000 → "09999000"
        assert OTB.build_ticker("SPY", 9999.0, "put", "2025-01-01") == "O:SPY250101P09999000"

    def test_expiry_year_boundary(self):
        # 2030-01-15 → "300115"
        assert OTB.build_ticker("SPY", 600.0, "call", "2030-01-15") == "O:SPY300115C00600000"

    # --- YYMMDD format ---

    def test_yymmdd_single_digit_month_padded(self):
        # March = "03"
        assert OTB.build_ticker("SPY", 600.0, "call", "2025-03-07").startswith("O:SPY250307")

    def test_yymmdd_december(self):
        assert OTB.build_ticker("SPY", 600.0, "call", "2025-12-31").startswith("O:SPY251231")


# ===========================================================================
# compute_strikes
# ===========================================================================

class TestComputeStrikes:

    # --- between-increment opening prices ---

    def test_1usd_increment_between(self):
        calls, puts = OTB.compute_strikes(600.25, 2, 2, 1.0)
        assert calls == [601.0, 602.0]
        assert puts  == [600.0, 599.0]

    def test_half_dollar_increment_between(self):
        calls, puts = OTB.compute_strikes(600.25, 2, 2, 0.5)
        assert calls == [600.5, 601.0]
        assert puts  == [600.0, 599.5]

    def test_5usd_increment_between(self):
        calls, puts = OTB.compute_strikes(522.0, 2, 2, 5.0)
        assert calls == [525.0, 530.0]
        assert puts  == [520.0, 515.0]

    # --- exact boundary opening prices ---

    def test_1usd_on_exact_boundary(self):
        # 600.0 is exactly on increment → calls must start at 601, not 600
        calls, puts = OTB.compute_strikes(600.0, 2, 2, 1.0)
        assert calls == [601.0, 602.0]
        assert puts  == [600.0, 599.0]

    def test_half_dollar_on_exact_boundary(self):
        calls, puts = OTB.compute_strikes(600.5, 2, 2, 0.5)
        assert calls == [601.0, 601.5]
        assert puts  == [600.5, 600.0]

    def test_5usd_on_exact_boundary(self):
        calls, puts = OTB.compute_strikes(520.0, 2, 2, 5.0)
        assert calls == [525.0, 530.0]
        assert puts  == [520.0, 515.0]

    # --- n_calls / n_puts counts ---

    def test_n_calls_1_n_puts_1(self):
        calls, puts = OTB.compute_strikes(600.25, 1, 1, 1.0)
        assert calls == [601.0]
        assert puts  == [600.0]

    def test_n_calls_3_n_puts_3(self):
        calls, puts = OTB.compute_strikes(600.0, 3, 3, 1.0)
        assert len(calls) == 3
        assert len(puts)  == 3
        assert calls == [601.0, 602.0, 603.0]
        assert puts  == [600.0, 599.0, 598.0]

    def test_asymmetric_n_calls_n_puts(self):
        calls, puts = OTB.compute_strikes(600.0, 1, 3, 1.0)
        assert len(calls) == 1
        assert len(puts)  == 3

    # --- sorting (nearest-first) ---

    def test_call_strikes_sorted_nearest_first(self):
        calls, _ = OTB.compute_strikes(600.0, 4, 1, 1.0)
        assert calls == sorted(calls)           # ascending = nearest-first for calls

    def test_put_strikes_sorted_nearest_first(self):
        _, puts = OTB.compute_strikes(600.0, 1, 4, 1.0)
        assert puts == sorted(puts, reverse=True)  # descending = nearest-first for puts

    # --- rounding ---

    def test_no_floating_point_drift(self):
        # 0.1 increments can cause float drift — confirm rounding holds
        calls, puts = OTB.compute_strikes(600.1, 2, 2, 0.1)
        for v in calls + puts:
            assert round(v, 2) == v


# ===========================================================================
# next_trading_day
# ===========================================================================

class TestNextTradingDay:

    def test_monday_gives_tuesday(self):
        assert OTB.next_trading_day("2025-03-03") == "2025-03-04"   # Mon → Tue

    def test_tuesday_gives_wednesday(self):
        assert OTB.next_trading_day("2025-03-04") == "2025-03-05"

    def test_wednesday_gives_thursday(self):
        assert OTB.next_trading_day("2025-03-05") == "2025-03-06"

    def test_thursday_gives_friday(self):
        assert OTB.next_trading_day("2025-03-06") == "2025-03-07"

    def test_friday_gives_next_monday(self):
        assert OTB.next_trading_day("2025-03-07") == "2025-03-10"   # Fri → Mon

    def test_saturday_gives_monday(self):
        assert OTB.next_trading_day("2025-03-08") == "2025-03-10"   # Sat → Mon

    def test_sunday_gives_monday(self):
        assert OTB.next_trading_day("2025-03-09") == "2025-03-10"   # Sun → Mon

    def test_result_is_always_weekday(self):
        for day_offset in range(14):
            from datetime import datetime, timedelta
            base = "2025-03-03"
            dt = datetime.strptime(base, "%Y-%m-%d") + timedelta(days=day_offset)
            result = OTB.next_trading_day(dt.strftime("%Y-%m-%d"))
            result_dt = datetime.strptime(result, "%Y-%m-%d")
            assert result_dt.weekday() < 5, f"{result} is not a weekday"


# ===========================================================================
# next_friday
# ===========================================================================

class TestNextFriday:

    def test_monday_gives_same_week_friday(self):
        assert OTB.next_friday("2025-03-03") == "2025-03-07"   # Mon → Fri same week

    def test_tuesday_gives_same_week_friday(self):
        assert OTB.next_friday("2025-03-04") == "2025-03-07"

    def test_thursday_gives_same_week_friday(self):
        assert OTB.next_friday("2025-03-06") == "2025-03-07"

    def test_friday_gives_next_week_friday(self):
        assert OTB.next_friday("2025-03-07") == "2025-03-14"   # Fri → following Fri

    def test_saturday_gives_next_friday(self):
        assert OTB.next_friday("2025-03-08") == "2025-03-14"   # Sat → following Fri

    def test_sunday_gives_next_friday(self):
        assert OTB.next_friday("2025-03-09") == "2025-03-14"   # Sun → following Fri

    def test_result_is_always_friday(self):
        from datetime import datetime, timedelta
        for day_offset in range(14):
            base = "2025-03-03"
            dt = datetime.strptime(base, "%Y-%m-%d") + timedelta(days=day_offset)
            result = OTB.next_friday(dt.strftime("%Y-%m-%d"))
            result_dt = datetime.strptime(result, "%Y-%m-%d")
            assert result_dt.weekday() == 4, f"{result} is not a Friday"

    def test_result_is_strictly_after_input(self):
        from datetime import datetime
        for date_str in ["2025-03-03", "2025-03-07", "2025-03-14"]:
            result = OTB.next_friday(date_str)
            assert datetime.strptime(result, "%Y-%m-%d") > datetime.strptime(date_str, "%Y-%m-%d")
