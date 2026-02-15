# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration test for the full consolidation pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processing.consolidator import Consolidator


BASE_TS = 1707480000000  # 2024-02-09 12:00:00 UTC (approx)
DATE = "2024-02-09"


def _make_spy(raw_path, n=100):
    """Write synthetic SPY Parquet to raw_path/spy/."""
    np.random.seed(42)
    prices = 450.0 + np.cumsum(np.random.randn(n) * 0.1)
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
    return n


def _make_vix(raw_path, n=50):
    """Write synthetic VIX Parquet."""
    np.random.seed(99)
    prices = 18.0 + np.cumsum(np.random.randn(n) * 0.05)
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


def _make_options(raw_path, tickers, n=20):
    """Write synthetic options Parquet."""
    rows = []
    for i in range(n):
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
            "timestamp": [BASE_TS + i * 30000 for i in range(n)],
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

    def test_full_consolidation_pipeline(self, tmp_path):
        """End-to-end: create all 4 raw sources, run consolidation, verify output."""
        raw_path = tmp_path / "raw"
        consolidated_path = tmp_path / "consolidated"

        tickers = ["O:SPY240315C00449000", "O:SPY240315P00450000"]

        # Create all raw data
        spy_rows = _make_spy(raw_path, n=100)
        _make_vix(raw_path, n=50)
        _make_options(raw_path, tickers, n=20)
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

        # Status check
        assert stats["status"] == "success"
        assert stats["total_rows"] == spy_rows
        assert stats["options_contracts_processed"] == len(tickers)
        assert stats["vix_available"] is True
        assert stats["news_available"] is True

        # Output file exists
        out_path = consolidated_path / f"{DATE}.parquet"
        assert out_path.exists()

        result = pd.read_parquet(out_path)

        # Row count matches SPY
        assert len(result) == spy_rows

        # SPY columns
        for col in ["spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "spy_vwap"]:
            assert col in result.columns

        # VIX columns (forward-filled)
        for col in ["vix_open", "vix_high", "vix_low", "vix_close"]:
            assert col in result.columns
        assert result["vix_close"].notna().any()

        # Indicator columns
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

        # Momentum columns
        for w in [5, 30, 60]:
            assert f"spy_price_change_{w}" in result.columns
            assert f"spy_roc_{w}" in result.columns

        # Greeks list columns
        for col in [
            "option_tickers",
            "option_strikes",
            "option_types",
            "option_close_prices",
            "option_deltas",
            "option_gammas",
            "option_thetas",
            "option_vegas",
            "option_rhos",
            "option_ivs",
        ]:
            assert col in result.columns

        # At least some rows should have populated Greeks
        # Parquet roundtrip converts lists to numpy arrays
        has_greeks = result["option_tickers"].apply(lambda x: len(x) > 0 if hasattr(x, "__len__") else False)
        assert has_greeks.any()

        # News columns
        for col in ["news_sentiment", "news_sentiment_reasoning", "news_article_id", "news_timestamp"]:
            assert col in result.columns
        assert result["news_sentiment"].notna().any()

        # Source tag
        assert (result["source"] == "consolidated").all()
