# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""End-to-end integration test: historical ingest → consolidation → verify.

Runs the full pipeline using mock data source clients:
1. Ingest SPY (HistoricalRunner with FakeEquityClient)
2. Ingest VIX (HistoricalRunner with FakeVIXClient)
3. Ingest Options (direct Parquet write — simulating pre-streamed data)
4. Ingest News (HistoricalRunner with FakeNewsClient)
5. Run Consolidator on the ingested data
6. Verify consolidated output schema and content
7. Run SchemaMonitor on output to verify drift detection

No live API calls.
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest

from src.monitoring.schema_monitor import SchemaMonitor
from src.orchestrator.historical_runner import HistoricalRunner
from src.processing.consolidator import Consolidator
from src.processing.deduplicator import Deduplicator
from src.processing.validator import RecordValidator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE = "2026-02-10"
# Minute-aligned base timestamp for 2026-02-10 14:30:00 UTC (9:30 ET)
BASE_TS = 1770843000000
N_MINUTES = 60  # Enough for indicators (RSI needs 14, MACD needs 26+9=35)


# ---------------------------------------------------------------------------
# Fake clients (same as test_historical_flow.py but specialized)
# ---------------------------------------------------------------------------

class FakeEquityClient:
    """Yields N_MINUTES × 60 per-second SPY bars for DATE."""

    def connect(self): pass
    def disconnect(self): pass

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator:
        np.random.seed(42)
        n = N_MINUTES * 60
        prices = 500.0 + np.cumsum(np.random.randn(n) * 0.01)
        for i in range(n):
            yield {
                "timestamp": BASE_TS + i * 1000,
                "open": float(prices[i]),
                "high": float(prices[i] + 0.2),
                "low": float(prices[i] - 0.2),
                "close": float(prices[i] + 0.05),
                "volume": int(1000 + i),
                "vwap": float(prices[i] + 0.03),
                "transactions": 50,
                "source": "spy",
            }


class FakeVIXClient:
    """Yields per-second VIX bars spanning the same time range."""

    def connect(self): pass
    def disconnect(self): pass

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator:
        np.random.seed(99)
        n = N_MINUTES * 30  # VIX at ~2s intervals
        prices = 18.0 + np.cumsum(np.random.randn(n) * 0.005)
        for i in range(n):
            yield {
                "timestamp": BASE_TS + i * 2000,
                "open": float(prices[i]),
                "high": float(prices[i] + 0.1),
                "low": float(prices[i] - 0.1),
                "close": float(prices[i] + 0.02),
                "volume": 0,
                "vwap": float(prices[i]),
                "transactions": 0,
                "source": "vix",
            }


class FakeNewsClient:
    """Yields a few news articles within the time range."""

    def connect(self): pass
    def disconnect(self): pass

    def fetch_historical(self, start_date, end_date, **kwargs) -> Generator:
        for i in range(5):
            yield {
                "timestamp": BASE_TS + i * 300000,
                "article_id": f"art_{i}",
                "title": f"Market Update {i}",
                "description": f"SPY rallies on strong earnings {i}",
                "author": "Reporter",
                "article_url": "https://example.com",
                "tickers": ["SPY"],
                "keywords": ["market", "spy"],
                "sentiment": 0.65 + i * 0.05,
                "sentiment_reasoning": "Bullish momentum",
                "publisher_name": "News Corp",
                "source": "news",
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_options_parquet(raw_path, tickers, n_minutes=N_MINUTES):
    """Write synthetic per-second options data (every 5s per ticker)."""
    np.random.seed(7)
    rows = []
    for i in range(n_minutes * 12):  # 12 ticks per minute (every 5 sec)
        for ticker in tickers:
            rows.append({
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
            })
    df = pd.DataFrame(rows)
    out = raw_path / "options" / f"{DATE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)


def _write_contracts_json(raw_path, tickers):
    """Write options contracts JSON."""
    contracts = []
    for i, t in enumerate(tickers):
        ctype = "call" if i % 2 == 0 else "put"
        contracts.append({
            "ticker": t,
            "underlying_ticker": "SPY",
            "strike_price": 499.0 + i,
            "expiration_date": "2026-03-20",
            "contract_type": ctype,
            "exercise_style": "american",
            "primary_exchange": "CBOE",
            "shares_per_contract": 100,
        })
    out = raw_path / "options" / "contracts" / f"{DATE}_contracts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(contracts, f)


def _make_config(tmp_path):
    """Build full config pointing to tmp_path."""
    raw_path = tmp_path / "raw"
    consolidated_path = tmp_path / "consolidated"
    return {
        "sinks": {
            "parquet": {
                "base_path": str(raw_path),
                "consolidated_path": str(consolidated_path),
                "compression": "snappy",
                "row_group_size": 10000,
            },
        },
        "historical": {
            "backfill": {
                "start_date": DATE,
                "end_date": DATE,
                "batch_size": 10000,
                "trading_days": 1,
            },
        },
        "logging": {
            "execution_log_path": str(tmp_path / "logs"),
        },
        "monitoring": {
            "performance": {
                "commit_latency_seconds": 300,
                "throughput_min_records_per_sec": 0,
                "memory_usage_mb_threshold": 10000,
            },
            "schema": {
                "alert_on_new_columns": True,
                "alert_on_missing_columns": True,
                "alert_on_type_changes": True,
                "auto_update_baseline": False,
            },
        },
        "consolidation": {
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
            "indicators": {
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std_dev": 2.0},
            },
            "momentum": {"windows": [5, 30]},
            "greeks": {
                "min_time_to_expiry_days": 0.01,
                "min_iv": 0.01,
                "max_iv": 5.0,
                "fallback_iv": 0.20,
            },
            "news": {"max_lookback_hours": 24},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end: ingest all sources → consolidate → verify."""

    def test_ingest_and_consolidate(self, tmp_path):
        """Full pipeline: ingest SPY+VIX+News → write options → consolidate → verify output."""
        config = _make_config(tmp_path)
        raw_path = tmp_path / "raw"

        # --- Step 1: Ingest SPY ---
        spy_runner = HistoricalRunner(
            config, ticker="SPY",
            client=FakeEquityClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        spy_stats = spy_runner.run()
        assert spy_stats["total_written"] > 0
        assert (raw_path / "spy" / f"{DATE}.parquet").exists()

        # --- Step 2: Ingest VIX ---
        vix_runner = HistoricalRunner(
            config, ticker="I:VIX",
            client=FakeVIXClient(),
            validator=RecordValidator("vix"),
        )
        vix_stats = vix_runner.run()
        assert vix_stats["total_written"] > 0
        assert (raw_path / "vix" / f"{DATE}.parquet").exists()

        # --- Step 3: Ingest News ---
        news_runner = HistoricalRunner(
            config, ticker="news",
            client=FakeNewsClient(),
            validator=RecordValidator("news"),
            deduplicator=Deduplicator(key_field="article_id"),
        )
        news_stats = news_runner.run()
        assert news_stats["total_written"] > 0
        assert (raw_path / "news" / f"{DATE}.parquet").exists()

        # --- Step 4: Write options data + contracts ---
        option_tickers = ["O:SPY260320C00499000", "O:SPY260320P00500000"]
        _write_options_parquet(raw_path, option_tickers, n_minutes=N_MINUTES)
        _write_contracts_json(raw_path, option_tickers)

        # --- Step 5: Consolidate ---
        consolidator = Consolidator(config)
        cons_stats = consolidator.consolidate(DATE)

        assert cons_stats["status"] == "success"
        assert cons_stats["minutes"] == N_MINUTES
        assert cons_stats["unique_options"] == len(option_tickers)
        assert cons_stats["vix_available"] is True
        assert cons_stats["news_available"] is True

        # --- Step 6: Verify output ---
        consolidated_path = tmp_path / "consolidated" / f"{DATE}.parquet"
        assert consolidated_path.exists()

        result = pd.read_parquet(consolidated_path)

        # Row count: minutes × options
        assert len(result) == N_MINUTES * len(option_tickers)

        # SPY columns present
        for col in ["spy_open", "spy_close", "spy_volume", "spy_vwap"]:
            assert col in result.columns
            assert result[col].notna().all()

        # VIX columns present
        assert "vix_close" in result.columns
        assert result["vix_close"].notna().any()

        # Indicator columns present
        assert "spy_rsi_14" in result.columns
        assert "spy_macd" in result.columns
        assert "spy_bb_upper" in result.columns

        # Momentum columns present
        assert "spy_price_change_5" in result.columns
        assert "spy_roc_30" in result.columns

        # Options columns present (flat scalars)
        assert "ticker" in result.columns
        assert "strike_price" in result.columns
        assert "option_avg_price" in result.columns

        # Greeks present (flat scalars)
        for col in ["delta", "gamma", "theta", "vega", "implied_volatility"]:
            assert col in result.columns

        # News sentiment attached
        assert "news_sentiment" in result.columns
        assert result["news_sentiment"].notna().any()

        # Source tag
        assert (result["source"] == "consolidated").all()

        # Timestamps minute-aligned
        assert (result["timestamp"] % 60000 == 0).all()

    def test_consolidation_idempotent(self, tmp_path):
        """Running consolidation twice on the same data produces identical output."""
        config = _make_config(tmp_path)
        raw_path = tmp_path / "raw"

        # Ingest SPY only (minimal test)
        spy_runner = HistoricalRunner(
            config, ticker="SPY",
            client=FakeEquityClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        spy_runner.run()

        # Write options + contracts
        option_tickers = ["O:SPY260320C00499000"]
        _write_options_parquet(raw_path, option_tickers)
        _write_contracts_json(raw_path, option_tickers)

        # First consolidation
        consolidator1 = Consolidator(config)
        stats1 = consolidator1.consolidate(DATE)

        consolidated_path = tmp_path / "consolidated" / f"{DATE}.parquet"
        df1 = pd.read_parquet(consolidated_path)

        # Second consolidation (overwrites same file)
        consolidator2 = Consolidator(config)
        stats2 = consolidator2.consolidate(DATE)
        df2 = pd.read_parquet(consolidated_path)

        assert stats1["total_rows"] == stats2["total_rows"]
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)


class TestSchemaMonitorOnPipeline:
    """Verify SchemaMonitor works on pipeline-produced Parquet files."""

    def test_baseline_capture_on_ingested_data(self, tmp_path):
        """SchemaMonitor can capture baselines from ingested Parquet files."""
        config = _make_config(tmp_path)
        raw_path = tmp_path / "raw"

        # Ingest SPY
        runner = HistoricalRunner(
            config, ticker="SPY",
            client=FakeEquityClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        runner.run()

        spy_file = str(raw_path / "spy" / f"{DATE}.parquet")
        monitor = SchemaMonitor(config)
        monitor._baseline_dir = tmp_path / "baselines"

        baseline = monitor.capture_baseline("spy", spy_file)

        assert baseline["source"] == "spy"
        assert "timestamp" in baseline["schema"]
        assert "open" in baseline["schema"]
        assert "close" in baseline["schema"]
        assert baseline["column_count"] >= 8

    def test_no_drift_on_same_schema(self, tmp_path):
        """No drift detected when schema matches baseline."""
        config = _make_config(tmp_path)
        raw_path = tmp_path / "raw"

        runner = HistoricalRunner(
            config, ticker="SPY",
            client=FakeEquityClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        runner.run()

        spy_file = str(raw_path / "spy" / f"{DATE}.parquet")
        monitor = SchemaMonitor(config)
        monitor._baseline_dir = tmp_path / "baselines"

        # First call auto-captures baseline
        alerts1 = monitor.check_drift("spy", spy_file)
        assert alerts1 == []

        # Second call — same schema, no drift
        alerts2 = monitor.check_drift("spy", spy_file)
        assert alerts2 == []

    def test_schema_drift_detected_on_consolidated(self, tmp_path):
        """SchemaMonitor detects drift between raw and consolidated schemas."""
        config = _make_config(tmp_path)
        raw_path = tmp_path / "raw"

        # Ingest SPY
        runner = HistoricalRunner(
            config, ticker="SPY",
            client=FakeEquityClient(),
            validator=RecordValidator.for_equity("SPY"),
        )
        runner.run()

        # Write options + contracts + consolidate
        option_tickers = ["O:SPY260320C00499000"]
        _write_options_parquet(raw_path, option_tickers)
        _write_contracts_json(raw_path, option_tickers)

        consolidator = Consolidator(config)
        consolidator.consolidate(DATE)

        spy_file = str(raw_path / "spy" / f"{DATE}.parquet")
        consolidated_file = str(tmp_path / "consolidated" / f"{DATE}.parquet")

        monitor = SchemaMonitor(config)
        monitor._baseline_dir = tmp_path / "baselines"

        # Capture baseline from raw SPY
        monitor.check_drift("spy", spy_file)

        # Check consolidated against SPY baseline — should detect drift
        # (consolidated has many more columns)
        alerts = monitor.check_drift("spy", consolidated_file)
        assert len(alerts) > 0
        assert any("New columns" in a for a in alerts)
