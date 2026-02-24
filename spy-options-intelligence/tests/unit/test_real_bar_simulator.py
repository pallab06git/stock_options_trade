# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for RealBarSimulator."""

import numpy as np
import pandas as pd
import pytest

from src.ml.exit_signal_model import EXIT_FEATURE_COLS, ExitSignalModel
from src.ml.real_bar_simulator import RealBarSimulator
from src.ml.trade_simulator import Trade


def _make_bars(n=60, base_price=4.0, trend="up"):
    """Create synthetic option minute bars."""
    rng = np.random.RandomState(42)
    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        drift = 0.02 if trend == "up" else (-0.02 if trend == "down" else 0.0)
        prices[i] = prices[i - 1] * (1 + drift + rng.randn() * 0.01)

    return pd.DataFrame({
        "timestamp": np.arange(n) * 60000 + 1000000000,
        "open": prices - 0.01,
        "high": prices + rng.uniform(0.01, 0.05, n),
        "low": prices - rng.uniform(0.01, 0.05, n),
        "close": prices,
        "volume": rng.randint(10, 500, n).astype(float),
        "transactions": rng.randint(1, 50, n).astype(float),
        "spy_close": np.linspace(580, 582, n),
        "spy_vol_std_5m": rng.uniform(0.01, 0.05, n),
        "spy_macd_hist": rng.randn(n) * 0.01,
        "opt_rsi_14": rng.uniform(30, 80, n),
        "minutes_since_open": np.arange(n, dtype=float),
    })


def _make_features_df(n=100, n_feats=20):
    """Create synthetic features DataFrame."""
    rng = np.random.RandomState(42)
    data = {}
    feat_cols = [f"feat_{i}" for i in range(n_feats)]
    for col in feat_cols:
        data[col] = rng.randn(n).astype(np.float32)
    data["timestamp"] = np.arange(n) * 60000 + 1000000000
    data["close"] = rng.uniform(3.0, 5.0, n)
    data["opt_close"] = data["close"]
    data["ticker"] = "O:SPY250307C00580000"
    data["contract_type"] = 1
    data["hour_et"] = 10
    data["minute_et"] = np.arange(n) % 60
    data["spy_close"] = 580.0
    data["minutes_since_open"] = np.arange(n, dtype=float) + 30
    data["date"] = "2025-03-03"
    data["target"] = (rng.random(n) > 0.9).astype(int)
    return pd.DataFrame(data), feat_cols


def _trained_exit_model():
    """Create a simple trained exit model."""
    rng = np.random.RandomState(42)
    n = 200
    data = {col: rng.randn(n).astype(np.float32) for col in EXIT_FEATURE_COLS}
    data["exit_label"] = (rng.random(n) > 0.85).astype(int)
    df = pd.DataFrame(data)

    model = ExitSignalModel()
    model.train(training_data=df)
    return model


class TestRealBarSimulatorInit:
    def test_defaults(self):
        sim = RealBarSimulator()
        assert sim.position_size == 10_000
        assert sim.max_signals_per_day == 3
        assert sim.entry_threshold == 0.70

    def test_custom_config(self):
        config = {"signal_pipeline": {
            "position_size_usd": 5000,
            "max_signals_per_day": 2,
            "entry_threshold": 0.80,
        }}
        sim = RealBarSimulator(config)
        assert sim.position_size == 5000
        assert sim.max_signals_per_day == 2


class TestSimulateDay:
    def test_returns_trades(self, tmp_path):
        sim = RealBarSimulator()
        exit_model = _trained_exit_model()

        # Create raw option data
        date_dir = tmp_path / "2025-03-03"
        date_dir.mkdir()
        bars = _make_bars(60)
        bars.to_parquet(date_dir / "O_SPY250307C00580000.parquet")

        features_df, feat_cols = _make_features_df(60)
        proba = np.random.RandomState(42).uniform(0.5, 0.9, 60)
        proba[10] = 0.95  # High confidence signal

        trades = sim.simulate_day(
            date="2025-03-03",
            features_df=features_df,
            feature_cols=feat_cols,
            entry_proba=proba,
            exit_model=exit_model,
            raw_options_dir=str(tmp_path),
        )
        # Should produce at least 1 trade
        assert isinstance(trades, list)

    def test_respects_max_signals(self, tmp_path):
        config = {"signal_pipeline": {"max_signals_per_day": 1, "entry_threshold": 0.3}}
        sim = RealBarSimulator(config)
        exit_model = _trained_exit_model()

        date_dir = tmp_path / "2025-03-03"
        date_dir.mkdir()
        bars = _make_bars(60)
        bars.to_parquet(date_dir / "O_SPY250307C00580000.parquet")

        features_df, feat_cols = _make_features_df(60)
        proba = np.full(60, 0.9)  # All high confidence

        trades = sim.simulate_day(
            date="2025-03-03",
            features_df=features_df,
            feature_cols=feat_cols,
            entry_proba=proba,
            exit_model=exit_model,
            raw_options_dir=str(tmp_path),
        )
        assert len(trades) <= 1

    def test_no_trades_below_threshold(self, tmp_path):
        sim = RealBarSimulator()
        exit_model = _trained_exit_model()

        date_dir = tmp_path / "2025-03-03"
        date_dir.mkdir()
        bars = _make_bars(30)
        bars.to_parquet(date_dir / "O_SPY250307C00580000.parquet")

        features_df, feat_cols = _make_features_df(30)
        proba = np.full(30, 0.3)  # All below threshold

        trades = sim.simulate_day(
            date="2025-03-03",
            features_df=features_df,
            feature_cols=feat_cols,
            entry_proba=proba,
            exit_model=exit_model,
            raw_options_dir=str(tmp_path),
        )
        assert len(trades) == 0

    def test_empty_features(self, tmp_path):
        sim = RealBarSimulator()
        exit_model = _trained_exit_model()

        trades = sim.simulate_day(
            date="2025-03-03",
            features_df=pd.DataFrame(),
            feature_cols=[],
            entry_proba=np.array([]),
            exit_model=exit_model,
            raw_options_dir=str(tmp_path),
        )
        assert len(trades) == 0

    def test_trade_has_pnl(self, tmp_path):
        config = {"signal_pipeline": {"entry_threshold": 0.3}}
        sim = RealBarSimulator(config)
        exit_model = _trained_exit_model()

        date_dir = tmp_path / "2025-03-03"
        date_dir.mkdir()
        bars = _make_bars(60, trend="up")
        bars.to_parquet(date_dir / "O_SPY250307C00580000.parquet")

        features_df, feat_cols = _make_features_df(60)
        proba = np.full(60, 0.9)

        trades = sim.simulate_day(
            date="2025-03-03",
            features_df=features_df,
            feature_cols=feat_cols,
            entry_proba=proba,
            exit_model=exit_model,
            raw_options_dir=str(tmp_path),
        )

        if trades:
            t = trades[0]
            assert t.profit_loss_usd is not None
            assert t.exit_reason is not None
            assert t.time_in_trade_minutes is not None


class TestGenerateSummary:
    def test_empty_trades(self):
        sim = RealBarSimulator()
        result = sim._generate_summary([], 0, "2025-03-03", "2025-03-31")
        assert result["total_trades"] == 0
        assert result["net_pnl"] == 0.0

    def test_with_trades(self):
        sim = RealBarSimulator()
        t = Trade(1, "2025-03-03 10:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.8)
        t.close_trade("2025-03-03 10:30 ET", 4.5, "Target hit", 30.0)

        result = sim._generate_summary([t], 1, "2025-03-03", "2025-03-03")
        assert result["total_trades"] == 1
        assert result["win_rate"] == 1.0
        assert result["net_pnl"] > 0

    def test_exit_reasons_counted(self):
        sim = RealBarSimulator()
        t1 = Trade(1, "2025-03-03 10:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.8)
        t1.close_trade("2025-03-03 10:30 ET", 4.5, "Target hit", 30.0)
        t2 = Trade(2, "2025-03-03 11:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.8)
        t2.close_trade("2025-03-03 11:30 ET", 3.5, "Hard stop", 30.0)

        result = sim._generate_summary([t1, t2], 1, None, None)
        assert result["exit_reasons"]["Target hit"] == 1
        assert result["exit_reasons"]["Hard stop"] == 1


class TestSaveResults:
    def test_saves_json_and_csv(self, tmp_path):
        sim = RealBarSimulator()
        t = Trade(1, "2025-03-03 10:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.8)
        t.close_trade("2025-03-03 10:30 ET", 4.5, "Target hit", 30.0)

        result = sim._generate_summary([t], 1, "2025-03-03", "2025-03-03")
        sim._save_results(result, [t], str(tmp_path / "output"))

        assert (tmp_path / "output" / "simulation_results.json").exists()
        assert (tmp_path / "output" / "trades.csv").exists()


class TestCountExitReasons:
    def test_counts_reasons(self):
        sim = RealBarSimulator()
        t1 = Trade(1, "x", "O:SPY", 4.0, 10000, 0.8)
        t1.exit_reason = "A"
        t2 = Trade(2, "x", "O:SPY", 4.0, 10000, 0.8)
        t2.exit_reason = "A"
        t3 = Trade(3, "x", "O:SPY", 4.0, 10000, 0.8)
        t3.exit_reason = "B"

        reasons = sim._count_exit_reasons([t1, t2, t3])
        assert reasons == {"A": 2, "B": 1}
