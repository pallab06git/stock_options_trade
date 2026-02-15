# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for Consolidator — per-option-per-minute flat schema."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processing.consolidator import Consolidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 1707480000000  # 2024-02-09 12:00:00 UTC (approx)


def _config(tmp_path, **overrides):
    """Build a minimal config dict pointing at tmp_path for data."""
    cfg = {
        "sinks": {
            "parquet": {
                "base_path": str(tmp_path / "raw"),
                "consolidated_path": str(tmp_path / "consolidated"),
                "compression": "snappy",
            }
        },
        "consolidation": {
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
            "indicators": {
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std_dev": 2.0},
            },
            "momentum": {"windows": [5, 30, 60]},
            "greeks": {
                "min_time_to_expiry_days": 0.01,
                "min_iv": 0.01,
                "max_iv": 5.0,
                "fallback_iv": 0.20,
            },
            "news": {"max_lookback_hours": 24},
        },
    }
    cfg.update(overrides)
    return cfg


def _spy_df(n=100, start_ts=BASE_TS):
    """Create a synthetic per-second SPY DataFrame with n rows."""
    np.random.seed(42)
    prices = 450.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": [start_ts + i * 1000 for i in range(n)],
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices + 0.05,
            "volume": np.random.randint(100, 10000, n),
            "vwap": prices + 0.03,
            "transactions": np.random.randint(10, 500, n),
            "source": "spy",
        }
    )


def _vix_df(n=50, start_ts=BASE_TS):
    """Create a synthetic per-second VIX DataFrame."""
    np.random.seed(99)
    prices = 18.0 + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame(
        {
            "timestamp": [start_ts + i * 2000 for i in range(n)],
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices + 0.02,
            "volume": [0] * n,
            "vwap": prices,
            "transactions": [0] * n,
            "source": "vix",
        }
    )


def _options_df(tickers, n=20, start_ts=BASE_TS):
    """Create synthetic per-second options data for given tickers."""
    rows = []
    for i in range(n):
        for ticker in tickers:
            rows.append(
                {
                    "timestamp": start_ts + i * 5000,
                    "open": 5.0 + np.random.rand(),
                    "high": 6.0,
                    "low": 4.5,
                    "close": 5.2 + np.random.rand() * 0.5,
                    "volume": 100,
                    "vwap": 5.1,
                    "transactions": 10,
                    "ticker": ticker,
                    "source": "options",
                }
            )
    return pd.DataFrame(rows)


def _contracts(tickers, strike=451.0, exp="2024-03-15", ctype="call"):
    """Create contract metadata list."""
    return [
        {
            "ticker": t,
            "underlying_ticker": "SPY",
            "strike_price": strike,
            "expiration_date": exp,
            "contract_type": ctype,
            "exercise_style": "american",
            "primary_exchange": "CBOE",
            "shares_per_contract": 100,
        }
        for t in tickers
    ]


def _news_df(n=5, start_ts=BASE_TS):
    """Create synthetic news DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": [start_ts + i * 60000 for i in range(n)],
            "article_id": [f"art_{i}" for i in range(n)],
            "title": [f"Headline {i}" for i in range(n)],
            "description": [f"Description {i}" for i in range(n)],
            "author": ["Author"] * n,
            "article_url": ["https://example.com"] * n,
            "tickers": [["SPY"]] * n,
            "keywords": [["market"]] * n,
            "sentiment": [0.5 + i * 0.05 for i in range(n)],
            "sentiment_reasoning": [f"Reason {i}" for i in range(n)],
            "publisher_name": ["Publisher"] * n,
            "source": ["news"] * n,
        }
    )


def _write_parquet(df, base_path, source, date):
    """Write a DataFrame as a Parquet file in the expected layout."""
    out = Path(base_path) / source / f"{date}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def _write_contracts(contracts, base_path, date):
    """Write contracts JSON file."""
    out = Path(base_path) / "options" / "contracts" / f"{date}_contracts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(contracts, f)
    return out


# ---------------------------------------------------------------------------
# Config / Init
# ---------------------------------------------------------------------------


class TestInit:

    def test_init_loads_config(self, tmp_path):
        cfg = _config(tmp_path)
        c = Consolidator(cfg)
        assert c.risk_free_rate == 0.045
        assert c.dividend_yield == 0.015
        assert c.momentum_windows == [5, 30, 60]
        assert c.rsi_period == 14
        assert c.macd_fast == 12
        assert c.macd_slow == 26
        assert c.bb_period == 20

    def test_init_defaults(self, tmp_path):
        c = Consolidator({"sinks": {"parquet": {"base_path": str(tmp_path)}}})
        assert c.risk_free_rate == 0.045
        assert c.momentum_windows == [5, 30, 60]
        assert c.fallback_iv == 0.20
        assert c.news_lookback_hours == 24

    def test_init_no_signal_validation_needed(self, tmp_path):
        """Consolidator does not read signal_validation config."""
        cfg = _config(tmp_path)
        c = Consolidator(cfg)
        assert not hasattr(c, "prediction_window_minutes")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:

    def test_load_spy_missing(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        df = c._load_spy("2024-02-09")
        assert df.empty

    def test_load_spy_existing(self, tmp_path):
        raw = tmp_path / "raw"
        spy = _spy_df(10)
        _write_parquet(spy, raw, "spy", "2024-02-09")
        c = Consolidator(_config(tmp_path))
        df = c._load_spy("2024-02-09")
        assert len(df) == 10
        assert "timestamp" in df.columns

    def test_load_contracts_missing(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        result = c._load_contracts("2024-02-09")
        assert result == []

    def test_load_contracts_existing(self, tmp_path):
        raw = tmp_path / "raw"
        contracts = _contracts(["O:SPY240315C00451000"])
        _write_contracts(contracts, raw, "2024-02-09")
        c = Consolidator(_config(tmp_path))
        result = c._load_contracts("2024-02-09")
        assert len(result) == 1
        assert result[0]["ticker"] == "O:SPY240315C00451000"


# ---------------------------------------------------------------------------
# Per-Minute Aggregation
# ---------------------------------------------------------------------------


class TestPerMinuteAggregation:

    def test_spy_aggregation_basic(self, tmp_path):
        """60 per-second rows in one minute → 1 row."""
        c = Consolidator(_config(tmp_path))
        spy = _spy_df(60, start_ts=BASE_TS)
        result = c._aggregate_spy_per_minute(spy)
        assert len(result) == 1
        assert result["timestamp"].iloc[0] == (BASE_TS // 60000) * 60000

    def test_spy_aggregation_ohlc(self, tmp_path):
        """Verify OHLC aggregation: open=first, high=max, low=min, close=last."""
        c = Consolidator(_config(tmp_path))
        ts = BASE_TS
        df = pd.DataFrame({
            "timestamp": [ts, ts + 1000, ts + 2000],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 103.0, 106.0],
            "low": [99.0, 100.0, 98.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [100, 200, 300],
            "vwap": [100.5, 101.5, 102.5],
            "transactions": [10, 20, 30],
        })
        result = c._aggregate_spy_per_minute(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 100.0
        assert row["high"] == 106.0
        assert row["low"] == 98.0
        assert row["close"] == 103.0
        assert row["volume"] == 600
        assert row["transactions"] == 60

    def test_spy_aggregation_vwap(self, tmp_path):
        """Volume-weighted average price calculation."""
        c = Consolidator(_config(tmp_path))
        ts = BASE_TS
        df = pd.DataFrame({
            "timestamp": [ts, ts + 1000],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [100, 300],
            "vwap": [100.0, 102.0],
            "transactions": [10, 30],
        })
        result = c._aggregate_spy_per_minute(df)
        # Expected VWAP: (100*100 + 102*300) / 400 = 40600/400 = 101.5
        expected_vwap = (100.0 * 100 + 102.0 * 300) / 400
        assert abs(result["vwap"].iloc[0] - expected_vwap) < 0.01

    def test_spy_aggregation_multiple_minutes(self, tmp_path):
        """Data spanning 3 minutes → 3 rows."""
        c = Consolidator(_config(tmp_path))
        # 180 seconds = 3 minutes
        spy = _spy_df(180, start_ts=BASE_TS)
        result = c._aggregate_spy_per_minute(spy)
        assert len(result) == 3

    def test_spy_aggregation_timestamps_aligned(self, tmp_path):
        """Output timestamps are minute-aligned (divisible by 60000)."""
        c = Consolidator(_config(tmp_path))
        spy = _spy_df(120, start_ts=BASE_TS)
        result = c._aggregate_spy_per_minute(spy)
        for ts in result["timestamp"]:
            assert ts % 60000 == 0

    def test_vix_aggregation(self, tmp_path):
        """VIX aggregation to per-minute OHLC."""
        c = Consolidator(_config(tmp_path))
        ts = BASE_TS
        df = pd.DataFrame({
            "timestamp": [ts, ts + 1000, ts + 2000],
            "open": [18.0, 18.1, 18.2],
            "high": [18.5, 18.3, 18.6],
            "low": [17.8, 18.0, 17.9],
            "close": [18.1, 18.2, 18.3],
        })
        result = c._aggregate_vix_per_minute(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 18.0
        assert row["high"] == 18.6
        assert row["low"] == 17.8
        assert row["close"] == 18.3

    def test_vix_aggregation_empty(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        result = c._aggregate_vix_per_minute(pd.DataFrame())
        assert result.empty

    def test_options_aggregation_per_ticker(self, tmp_path):
        """Options aggregate per (minute, ticker) — two tickers in same minute → 2 rows."""
        c = Consolidator(_config(tmp_path))
        ts = BASE_TS
        df = pd.DataFrame({
            "timestamp": [ts, ts + 1000, ts, ts + 1000],
            "ticker": ["OPT_A", "OPT_A", "OPT_B", "OPT_B"],
            "open": [5.0, 5.1, 3.0, 3.1],
            "high": [5.5, 5.6, 3.5, 3.6],
            "low": [4.5, 4.6, 2.5, 2.6],
            "close": [5.2, 5.3, 3.2, 3.3],
            "volume": [100, 200, 150, 250],
        })
        result = c._aggregate_options_per_minute(df)
        assert len(result) == 2
        assert set(result["ticker"]) == {"OPT_A", "OPT_B"}
        # Check option_avg_price for OPT_A: mean(5.2, 5.3) = 5.25
        opt_a = result[result["ticker"] == "OPT_A"].iloc[0]
        assert abs(opt_a["option_avg_price"] - 5.25) < 0.01

    def test_options_aggregation_empty(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        result = c._aggregate_options_per_minute(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# VIX alignment
# ---------------------------------------------------------------------------


class TestVIXAlignment:

    def test_align_vix_empty(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        spy = pd.DataFrame({
            "timestamp": [BASE_TS],
            "spy_close": [450.0],
        })
        result = c._align_vix(spy, pd.DataFrame())
        assert "vix_open" in result.columns
        assert result["vix_close"].isna().all()

    def test_align_vix_forward_fill(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        spy = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000],
                "spy_close": [450.0, 451.0, 452.0, 453.0],
            }
        )
        vix = pd.DataFrame(
            {
                "timestamp": [1000, 3000],
                "open": [18.0, 19.0],
                "high": [18.5, 19.5],
                "low": [17.5, 18.5],
                "close": [18.2, 19.2],
            }
        )
        result = c._align_vix(spy, vix)
        assert len(result) == 4
        # ts=2000 should get forward-filled from ts=1000
        assert result.loc[result["timestamp"] == 2000, "vix_close"].values[0] == 18.2
        # ts=3000 should get the value at ts=3000
        assert result.loc[result["timestamp"] == 3000, "vix_close"].values[0] == 19.2


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


class TestIndicators:

    def test_indicators_insufficient_data(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame(
            {
                "timestamp": range(10),
                "spy_close": [450.0 + i for i in range(10)],
            }
        )
        result = c._compute_indicators(df)
        assert "spy_rsi_14" in result.columns
        assert result["spy_rsi_14"].isna().all()
        assert result["spy_macd"].isna().all()

    def test_indicators_rsi(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": range(n),
                "spy_close": 450.0 + np.cumsum(np.random.randn(n) * 0.1),
            }
        )
        result = c._compute_indicators(df)
        assert "spy_rsi_14" in result.columns
        assert result["spy_rsi_14"].notna().any()
        valid = result["spy_rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_indicators_macd(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": range(n),
                "spy_close": 450.0 + np.cumsum(np.random.randn(n) * 0.1),
            }
        )
        result = c._compute_indicators(df)
        for col in ["spy_macd", "spy_macd_signal", "spy_macd_histogram"]:
            assert col in result.columns
            assert result[col].notna().any()

    def test_indicators_bollinger(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": range(n),
                "spy_close": 450.0 + np.cumsum(np.random.randn(n) * 0.1),
            }
        )
        result = c._compute_indicators(df)
        for col in ["spy_bb_upper", "spy_bb_middle", "spy_bb_lower", "spy_bb_width"]:
            assert col in result.columns
            assert result[col].notna().any()
        valid_mask = result["spy_bb_upper"].notna()
        assert (result.loc[valid_mask, "spy_bb_upper"] >= result.loc[valid_mask, "spy_bb_middle"]).all()
        assert (result.loc[valid_mask, "spy_bb_middle"] >= result.loc[valid_mask, "spy_bb_lower"]).all()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestMomentum:

    def test_momentum_windows(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": range(n),
                "spy_close": [450.0 + i * 0.1 for i in range(n)],
            }
        )
        result = c._compute_momentum(df)
        for w in [5, 30, 60]:
            assert f"spy_price_change_{w}" in result.columns
            assert f"spy_roc_{w}" in result.columns

    def test_momentum_first_rows_nan(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": range(n),
                "spy_close": [450.0 + i * 0.1 for i in range(n)],
            }
        )
        result = c._compute_momentum(df)
        assert result["spy_price_change_5"].iloc[:5].isna().all()
        assert result["spy_price_change_5"].iloc[5:].notna().all()


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


class TestNews:

    def test_news_empty(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame(
            {"timestamp": [1000, 2000], "spy_close": [450.0, 451.0]}
        )
        result = c._attach_news(df, pd.DataFrame())
        assert "news_sentiment" in result.columns
        assert result["news_sentiment"].isna().all()

    def test_news_within_lookback(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame(
            {"timestamp": [BASE_TS, BASE_TS + 1000], "spy_close": [450.0, 451.0]}
        )
        news = pd.DataFrame(
            {
                "timestamp": [BASE_TS - 500],
                "sentiment": [0.75],
                "sentiment_reasoning": ["Bullish outlook"],
                "article_id": ["art_1"],
            }
        )
        result = c._attach_news(df, news)
        assert result["news_sentiment"].iloc[0] == 0.75
        assert result["news_article_id"].iloc[0] == "art_1"


# ---------------------------------------------------------------------------
# Flatten to per-option
# ---------------------------------------------------------------------------


class TestFlattenToPerOption:

    def test_flatten_basic(self, tmp_path):
        """2 options × 2 SPY minutes → 4 rows."""
        c = Consolidator(_config(tmp_path))
        ts1 = (BASE_TS // 60000) * 60000
        ts2 = ts1 + 60000

        spy = pd.DataFrame({
            "timestamp": [ts1, ts2],
            "spy_close": [450.0, 451.0],
            "spy_open": [449.5, 450.5],
        })

        options = pd.DataFrame({
            "timestamp": [ts1, ts1, ts2, ts2],
            "ticker": ["OPT_A", "OPT_B", "OPT_A", "OPT_B"],
            "option_avg_price": [5.0, 3.0, 5.1, 3.1],
            "open": [4.9, 2.9, 5.0, 3.0],
            "high": [5.5, 3.5, 5.6, 3.6],
            "low": [4.5, 2.5, 4.6, 2.6],
            "close": [5.0, 3.0, 5.1, 3.1],
            "volume": [100, 200, 150, 250],
        })

        contracts = [
            {"ticker": "OPT_A", "strike_price": 451.0, "expiration_date": "2024-03-15", "contract_type": "call"},
            {"ticker": "OPT_B", "strike_price": 449.0, "expiration_date": "2024-03-15", "contract_type": "put"},
        ]

        result = c._flatten_to_per_option(spy, options, contracts)
        assert len(result) == 4
        assert set(result["ticker"]) == {"OPT_A", "OPT_B"}
        # Each row has SPY context
        assert (result["spy_close"].notna()).all()

    def test_flatten_spy_only(self, tmp_path):
        """No options → SPY-only rows with null option columns."""
        c = Consolidator(_config(tmp_path))
        spy = pd.DataFrame({
            "timestamp": [BASE_TS, BASE_TS + 60000],
            "spy_close": [450.0, 451.0],
        })
        result = c._flatten_to_per_option(spy, pd.DataFrame(), [])
        assert len(result) == 2
        assert result["ticker"].isna().all()
        assert result["option_avg_price"].isna().all()

    def test_flatten_contract_metadata(self, tmp_path):
        """Verify strike_price, contract_type, time_to_expiry are populated."""
        c = Consolidator(_config(tmp_path))
        ts = (BASE_TS // 60000) * 60000
        spy = pd.DataFrame({
            "timestamp": [ts],
            "spy_close": [450.0],
        })
        options = pd.DataFrame({
            "timestamp": [ts],
            "ticker": ["OPT_A"],
            "option_avg_price": [5.0],
            "open": [4.9], "high": [5.5], "low": [4.5], "close": [5.0], "volume": [100],
        })
        contracts = [
            {"ticker": "OPT_A", "strike_price": 451.0, "expiration_date": "2024-03-15", "contract_type": "call"},
        ]
        result = c._flatten_to_per_option(spy, options, contracts)
        assert result["strike_price"].iloc[0] == 451.0
        assert result["contract_type"].iloc[0] == "call"
        assert result["time_to_expiry_days"].iloc[0] > 0

    def test_flatten_unknown_ticker_filtered(self, tmp_path):
        """Options with tickers not in contracts are filtered out."""
        c = Consolidator(_config(tmp_path))
        ts = (BASE_TS // 60000) * 60000
        spy = pd.DataFrame({
            "timestamp": [ts],
            "spy_close": [450.0],
        })
        options = pd.DataFrame({
            "timestamp": [ts],
            "ticker": ["UNKNOWN"],
            "option_avg_price": [5.0],
            "open": [4.9], "high": [5.5], "low": [4.5], "close": [5.0], "volume": [100],
        })
        contracts = [
            {"ticker": "OPT_A", "strike_price": 451.0, "expiration_date": "2024-03-15", "contract_type": "call"},
        ]
        result = c._flatten_to_per_option(spy, options, contracts)
        # Falls back to SPY-only
        assert result["ticker"].isna().all()

    def test_flatten_inner_join_only_matching_timestamps(self, tmp_path):
        """Options at timestamps not in SPY are dropped (inner join)."""
        c = Consolidator(_config(tmp_path))
        ts1 = (BASE_TS // 60000) * 60000
        ts2 = ts1 + 60000

        spy = pd.DataFrame({
            "timestamp": [ts1],  # Only 1 minute
            "spy_close": [450.0],
        })
        options = pd.DataFrame({
            "timestamp": [ts1, ts2],
            "ticker": ["OPT_A", "OPT_A"],
            "option_avg_price": [5.0, 5.1],
            "open": [4.9, 5.0], "high": [5.5, 5.6], "low": [4.5, 4.6],
            "close": [5.0, 5.1], "volume": [100, 150],
        })
        contracts = [
            {"ticker": "OPT_A", "strike_price": 451.0, "expiration_date": "2024-03-15", "contract_type": "call"},
        ]
        result = c._flatten_to_per_option(spy, options, contracts)
        assert len(result) == 1  # Only ts1 matches


# ---------------------------------------------------------------------------
# Greeks (flat)
# ---------------------------------------------------------------------------


class TestGreeksFlat:

    def test_greeks_call_delta_range(self, tmp_path):
        """Call delta should be between 0 and 1."""
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame({
            "ticker": ["OPT_A"],
            "spy_close": [450.0],
            "strike_price": [451.0],
            "time_to_expiry_days": [35.0],
            "contract_type": ["call"],
            "option_avg_price": [5.0],
        })
        result = c._compute_greeks_flat(df)
        assert 0 <= result["delta"].iloc[0] <= 1
        assert result["implied_volatility"].iloc[0] > 0

    def test_greeks_put_delta_range(self, tmp_path):
        """Put delta should be between -1 and 0."""
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame({
            "ticker": ["OPT_B"],
            "spy_close": [450.0],
            "strike_price": [449.0],
            "time_to_expiry_days": [35.0],
            "contract_type": ["put"],
            "option_avg_price": [3.0],
        })
        result = c._compute_greeks_flat(df)
        assert -1 <= result["delta"].iloc[0] <= 0

    def test_greeks_spy_only_nan(self, tmp_path):
        """Rows with no ticker (SPY-only) should get NaN Greeks."""
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame({
            "ticker": [None],
            "spy_close": [450.0],
            "strike_price": [np.nan],
            "time_to_expiry_days": [np.nan],
            "contract_type": [None],
            "option_avg_price": [np.nan],
        })
        result = c._compute_greeks_flat(df)
        assert np.isnan(result["delta"].iloc[0])
        assert np.isnan(result["gamma"].iloc[0])
        assert np.isnan(result["implied_volatility"].iloc[0])

    def test_greeks_expired_option_nan(self, tmp_path):
        """Options near expiry (tte < min) should get NaN Greeks."""
        c = Consolidator(_config(tmp_path))
        df = pd.DataFrame({
            "ticker": ["OPT_A"],
            "spy_close": [450.0],
            "strike_price": [451.0],
            "time_to_expiry_days": [0.001],  # Below min_tte_days (0.01)
            "contract_type": ["call"],
            "option_avg_price": [5.0],
        })
        result = c._compute_greeks_flat(df)
        assert np.isnan(result["delta"].iloc[0])

    def test_iv_fallback(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        iv = c._calc_iv(0.0, 450.0, 451.0, 0.1, "c")
        assert iv == 0.20


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestConsolidate:

    def test_consolidate_spy_only(self, tmp_path):
        """SPY-only consolidation produces per-minute rows with null option columns."""
        raw = tmp_path / "raw"
        # 3 minutes of per-second data
        spy = _spy_df(180, start_ts=BASE_TS)
        _write_parquet(spy, raw, "spy", "2024-02-09")
        c = Consolidator(_config(tmp_path))
        stats = c.consolidate("2024-02-09")
        assert stats["status"] == "success"
        assert stats["minutes"] == 3
        assert stats["total_rows"] == 3  # 1 row per minute
        assert stats["unique_options"] == 0
        assert stats["vix_available"] is False
        assert stats["news_available"] is False

        out = tmp_path / "consolidated" / "2024-02-09.parquet"
        assert out.exists()
        result = pd.read_parquet(out)
        assert "spy_close" in result.columns
        assert "source" in result.columns
        assert (result["source"] == "consolidated").all()
        assert "ticker" in result.columns
        assert result["ticker"].isna().all()

    def test_consolidate_missing_spy(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        stats = c.consolidate("2024-02-09")
        assert stats["status"] == "failed"
        assert stats["reason"] == "missing_spy_data"

    def test_consolidate_with_options_returns_stats(self, tmp_path):
        """Verify minutes, unique_options in return stats."""
        raw = tmp_path / "raw"
        # 2 minutes of data, 1 option
        ts_base = (BASE_TS // 60000) * 60000
        spy = pd.DataFrame({
            "timestamp": [ts_base + i * 1000 for i in range(120)],
            "open": [450.0] * 120,
            "high": [451.0] * 120,
            "low": [449.0] * 120,
            "close": [450.5] * 120,
            "volume": [1000] * 120,
            "vwap": [450.3] * 120,
            "transactions": [50] * 120,
            "source": ["spy"] * 120,
        })
        _write_parquet(spy, raw, "spy", "2024-02-09")

        ticker = "O:SPY240315C00451000"
        options = pd.DataFrame({
            "timestamp": [ts_base + i * 1000 for i in range(120)],
            "open": [5.0] * 120,
            "high": [5.5] * 120,
            "low": [4.5] * 120,
            "close": [5.2] * 120,
            "volume": [100] * 120,
            "vwap": [5.1] * 120,
            "transactions": [10] * 120,
            "ticker": [ticker] * 120,
            "source": ["options"] * 120,
        })
        _write_parquet(options, raw, "options", "2024-02-09")
        _write_contracts(_contracts([ticker]), raw, "2024-02-09")

        c = Consolidator(_config(tmp_path))
        stats = c.consolidate("2024-02-09")
        assert stats["status"] == "success"
        assert stats["minutes"] == 2
        assert stats["unique_options"] == 1
        assert "target_future_prices" not in stats  # Not in consolidator anymore
