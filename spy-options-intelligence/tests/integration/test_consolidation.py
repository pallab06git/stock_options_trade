# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration test for the full consolidation pipeline — flat per-option-per-minute schema."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processing.consolidator import Consolidator


# Use a minute-aligned base timestamp to keep math clean
BASE_TS = (1707480000000 // 60000) * 60000  # 2024-02-09 ~12:00 UTC, minute-aligned
DATE = "2024-02-09"


def _make_spy(raw_path, n_minutes=100):
    """Write synthetic per-second SPY Parquet (n_minutes × 60 rows)."""
    np.random.seed(42)
    n = n_minutes * 60
    prices = 450.0 + np.cumsum(np.random.randn(n) * 0.01)
    df = pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * 1000 for i in range(n)],
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
    out = raw_path / "spy" / f"{DATE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return n_minutes


def _make_vix(raw_path, n=3000):
    """Write synthetic per-second VIX Parquet."""
    np.random.seed(99)
    prices = 18.0 + np.cumsum(np.random.randn(n) * 0.005)
    df = pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * 2000 for i in range(n)],
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
    out = raw_path / "vix" / f"{DATE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)


def _make_options(raw_path, tickers, n_minutes=100):
    """Write synthetic per-second options Parquet (every 5s per ticker)."""
    rows = []
    for i in range(n_minutes * 12):  # 12 ticks per minute (every 5 sec)
        for ticker in tickers:
            rows.append(
                {
                    "timestamp": BASE_TS + i * 5000,
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
    df = pd.DataFrame(rows)
    out = raw_path / "options" / f"{DATE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)


def _make_contracts(raw_path, tickers):
    """Write contracts JSON."""
    contracts = []
    for i, t in enumerate(tickers):
        ctype = "call" if i % 2 == 0 else "put"
        contracts.append(
            {
                "ticker": t,
                "underlying_ticker": "SPY",
                "strike_price": 449.0 + i,
                "expiration_date": "2024-03-15",
                "contract_type": ctype,
                "exercise_style": "american",
                "primary_exchange": "CBOE",
                "shares_per_contract": 100,
            }
        )
    out = raw_path / "options" / "contracts" / f"{DATE}_contracts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(contracts, f)


def _make_news(raw_path, n=5):
    """Write synthetic news Parquet."""
    df = pd.DataFrame(
        {
            "timestamp": [BASE_TS + i * 300000 for i in range(n)],
            "article_id": [f"art_{i}" for i in range(n)],
            "title": [f"Market Update {i}" for i in range(n)],
            "description": [f"Description {i}" for i in range(n)],
            "author": ["Reporter"] * n,
            "article_url": ["https://example.com"] * n,
            "tickers": [["SPY"]] * n,
            "keywords": [["market"]] * n,
            "sentiment": [0.6 + i * 0.05 for i in range(n)],
            "sentiment_reasoning": [f"Positive outlook {i}" for i in range(n)],
            "publisher_name": ["News Corp"] * n,
            "source": ["news"] * n,
        }
    )
    out = raw_path / "news" / f"{DATE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)


class TestFullConsolidationPipeline:

    def test_full_consolidation_flat_schema(self, tmp_path):
        """End-to-end: create all 4 raw sources, run consolidation, verify flat output."""
        raw_path = tmp_path / "raw"
        consolidated_path = tmp_path / "consolidated"

        tickers = ["O:SPY240315C00449000", "O:SPY240315P00450000"]
        n_minutes = 50  # Enough for indicators (need >36)

        # Create all raw data
        spy_minutes = _make_spy(raw_path, n_minutes=n_minutes)
        _make_vix(raw_path, n=1500)
        _make_options(raw_path, tickers, n_minutes=n_minutes)
        _make_contracts(raw_path, tickers)
        _make_news(raw_path, n=5)

        config = {
            "sinks": {
                "parquet": {
                    "base_path": str(raw_path),
                    "consolidated_path": str(consolidated_path),
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

        consolidator = Consolidator(config)
        stats = consolidator.consolidate(DATE)

        # --- Status check ---
        assert stats["status"] == "success"
        assert stats["minutes"] == spy_minutes
        assert stats["unique_options"] == len(tickers)
        assert stats["options_contracts_processed"] == len(tickers)
        assert stats["vix_available"] is True
        assert stats["news_available"] is True
        # Total rows = minutes × options (inner join)
        assert stats["total_rows"] == spy_minutes * len(tickers)

        # --- Output file ---
        out_path = consolidated_path / f"{DATE}.parquet"
        assert out_path.exists()

        result = pd.read_parquet(out_path)

        # Row count: per-option-per-minute
        assert len(result) == spy_minutes * len(tickers)

        # --- SPY columns (flat scalars) ---
        for col in ["spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "spy_vwap"]:
            assert col in result.columns
            assert result[col].notna().all()

        # --- VIX columns (forward-filled, flat scalars) ---
        for col in ["vix_open", "vix_high", "vix_low", "vix_close"]:
            assert col in result.columns
        assert result["vix_close"].notna().any()

        # --- Indicator columns (flat scalars) ---
        for col in [
            "spy_rsi_14",
            "spy_macd",
            "spy_macd_signal",
            "spy_macd_histogram",
            "spy_bb_upper",
            "spy_bb_middle",
            "spy_bb_lower",
            "spy_bb_width",
        ]:
            assert col in result.columns

        # --- Momentum columns (flat scalars) ---
        for w in [5, 30, 60]:
            assert f"spy_price_change_{w}" in result.columns
            assert f"spy_roc_{w}" in result.columns

        # --- Option columns (flat scalars, NOT lists) ---
        assert "ticker" in result.columns
        assert "contract_type" in result.columns
        assert "strike_price" in result.columns
        assert "time_to_expiry_days" in result.columns
        assert "option_avg_price" in result.columns

        # Verify these are scalars, not lists
        assert result["ticker"].iloc[0] is not None and isinstance(result["ticker"].iloc[0], str)
        assert result["strike_price"].dtype in (np.float64, float)
        assert result["option_avg_price"].dtype in (np.float64, float)

        # --- Greeks (flat scalars, NOT lists) ---
        for col in ["delta", "gamma", "theta", "vega", "rho", "implied_volatility"]:
            assert col in result.columns
            assert result[col].dtype in (np.float64, float)
        # At least some rows should have valid Greeks
        assert result["delta"].notna().any()

        # --- No list columns from old schema ---
        old_list_cols = [
            "option_tickers", "option_strikes", "option_types",
            "option_close_prices", "option_deltas", "option_gammas",
            "option_thetas", "option_vegas", "option_rhos", "option_ivs",
        ]
        for col in old_list_cols:
            assert col not in result.columns

        # --- News columns (flat scalars) ---
        for col in ["news_sentiment", "news_sentiment_reasoning", "news_article_id"]:
            assert col in result.columns
        assert result["news_sentiment"].notna().any()

        # --- No target_future_prices (handled by TrainingDataPrep) ---
        assert "target_future_prices" not in result.columns

        # --- Source tag ---
        assert (result["source"] == "consolidated").all()

        # --- Timestamps are minute-aligned ---
        for ts in result["timestamp"]:
            assert ts % 60000 == 0
