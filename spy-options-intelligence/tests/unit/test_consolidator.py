# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for Consolidator."""

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
    """Create a synthetic SPY DataFrame with n rows."""
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
    """Create a synthetic VIX DataFrame."""
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
    """Create synthetic options data for given tickers."""
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
# VIX alignment
# ---------------------------------------------------------------------------


class TestVIXAlignment:

    def test_align_vix_empty(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        spy = _spy_df(10).rename(
            columns={
                "open": "spy_open",
                "high": "spy_high",
                "low": "spy_low",
                "close": "spy_close",
                "volume": "spy_volume",
                "vwap": "spy_vwap",
                "transactions": "spy_transactions",
            }
        )
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
        # After warmup, RSI should have non-NaN values
        assert result["spy_rsi_14"].notna().any()
        # RSI is bounded 0-100
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
        # Upper > Middle > Lower
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
        # First 5 rows should be NaN for window=5
        assert result["spy_price_change_5"].iloc[:5].isna().all()
        assert result["spy_price_change_5"].iloc[5:].notna().all()


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:

    def test_greeks_no_options(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        spy = pd.DataFrame(
            {"timestamp": [1000, 2000], "spy_close": [450.0, 451.0]}
        )
        result = c._compute_greeks(pd.DataFrame(), spy, [])
        assert result.empty

    def test_greeks_single_call(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        ticker = "O:SPY240315C00451000"
        spy = pd.DataFrame(
            {"timestamp": [BASE_TS], "spy_close": [450.0]}
        )
        opts = pd.DataFrame(
            [
                {
                    "timestamp": BASE_TS,
                    "ticker": ticker,
                    "close": 5.0,
                    "open": 4.8,
                    "high": 5.5,
                    "low": 4.5,
                }
            ]
        )
        contracts = _contracts([ticker], strike=451.0, exp="2024-03-15", ctype="call")
        result = c._compute_greeks(opts, spy, contracts)
        assert len(result) == 1
        # Call delta should be between 0 and 1
        d = result["option_deltas"].iloc[0]
        assert len(d) == 1
        assert 0 <= d[0] <= 1

    def test_greeks_iv_fallback(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        ticker = "O:SPY240315C00451000"
        spy = pd.DataFrame(
            {"timestamp": [BASE_TS], "spy_close": [450.0]}
        )
        # Price=0 forces IV fallback
        opts = pd.DataFrame(
            [
                {
                    "timestamp": BASE_TS,
                    "ticker": ticker,
                    "close": 0.0,
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                }
            ]
        )
        contracts = _contracts([ticker], strike=451.0, exp="2024-03-15", ctype="call")
        # _calc_iv should return fallback
        iv = c._calc_iv(0.0, 450.0, 451.0, 0.1, "c")
        assert iv == 0.20


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
# Full pipeline
# ---------------------------------------------------------------------------


class TestConsolidate:

    def test_consolidate_spy_only(self, tmp_path):
        raw = tmp_path / "raw"
        spy = _spy_df(50)
        _write_parquet(spy, raw, "spy", "2024-02-09")
        c = Consolidator(_config(tmp_path))
        stats = c.consolidate("2024-02-09")
        assert stats["status"] == "success"
        assert stats["total_rows"] == 50
        assert stats["vix_available"] is False
        assert stats["news_available"] is False
        # Output file exists
        out = tmp_path / "consolidated" / "2024-02-09.parquet"
        assert out.exists()
        result = pd.read_parquet(out)
        assert "spy_close" in result.columns
        assert "source" in result.columns
        assert (result["source"] == "consolidated").all()

    def test_consolidate_missing_spy(self, tmp_path):
        c = Consolidator(_config(tmp_path))
        stats = c.consolidate("2024-02-09")
        assert stats["status"] == "failed"
        assert stats["reason"] == "missing_spy_data"
