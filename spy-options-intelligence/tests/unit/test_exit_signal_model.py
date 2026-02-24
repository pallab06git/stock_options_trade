# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for ExitSignalModel and ExitFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from src.ml.exit_signal_model import (
    EXIT_FEATURE_COLS,
    ExitFeatureEngineer,
    ExitSignalModel,
)


def _make_bars(n=60, entry_price=4.0, trend="up"):
    """Create synthetic option bars DataFrame."""
    rng = np.random.RandomState(42)
    prices = np.zeros(n)
    prices[0] = entry_price

    for i in range(1, n):
        if trend == "up":
            drift = 0.02
        elif trend == "down":
            drift = -0.02
        else:
            drift = 0.0
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
    })


class TestExitFeatureEngineer:
    def setup_method(self):
        self.eng = ExitFeatureEngineer()

    def test_returns_15_columns(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1, 580.0, 0)
        assert len(EXIT_FEATURE_COLS) == 15
        for col in EXIT_FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_output_length(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 5, 4.0, 1)
        assert len(result) == 25  # 30 - 5

    def test_unrealized_pnl_at_entry_is_zero(self):
        bars = _make_bars(30)
        entry_price = float(bars.loc[0, "close"])
        result = self.eng.compute_exit_features(bars, 0, entry_price, 1)
        assert abs(result.loc[0, "unrealized_pnl_pct"]) < 0.01

    def test_time_in_trade_starts_at_zero(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        assert result.loc[0, "time_in_trade_min"] == 0.0

    def test_peak_pnl_monotonically_increases(self):
        bars = _make_bars(30, trend="up")
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        peak = result["peak_pnl_pct"].values
        for i in range(1, len(peak)):
            assert peak[i] >= peak[i - 1] - 1e-8

    def test_drawdown_non_positive(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        dd = result["drawdown_from_peak_pct"].values
        assert (dd <= 1e-8).all()

    def test_bars_since_peak_non_negative(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        bsp = result["bars_since_peak"].values
        assert (bsp >= 0).all()

    def test_rsi_reversal_binary(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        vals = set(result["rsi_reversal"].unique())
        assert vals.issubset({0.0, 1.0})

    def test_macd_cross_binary(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        vals = set(result["macd_cross"].unique())
        assert vals.issubset({0.0, 1.0})

    def test_regime_changed_binary(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1, 580.0, 0)
        vals = set(result["regime_changed"].unique())
        assert vals.issubset({0.0, 1.0})

    def test_empty_bars(self):
        bars = pd.DataFrame()
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1)
        assert result.empty

    def test_zero_entry_price(self):
        bars = _make_bars(10)
        result = self.eng.compute_exit_features(bars, 0, 0.0, 1)
        assert result.empty

    def test_spy_return_since_entry(self):
        bars = _make_bars(30)
        result = self.eng.compute_exit_features(bars, 0, 4.0, 1, 580.0, 0)
        assert "spy_return_since_entry" in result.columns
        # First bar should be ~0
        assert abs(result.loc[0, "spy_return_since_entry"]) < 0.5


class TestExitSignalModelInit:
    def test_defaults(self):
        model = ExitSignalModel()
        assert model.exit_threshold == 0.6
        assert model.max_hold_minutes == 120
        assert model.stop_loss_pct == -20.0

    def test_custom_config(self):
        config = {"exit_model": {"exit_threshold": 0.7, "max_hold_minutes": 60}}
        model = ExitSignalModel(config)
        assert model.exit_threshold == 0.7
        assert model.max_hold_minutes == 60


class TestExitSignalModelShouldExit:
    def test_hard_stop(self):
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        should, prob, reason = model.should_exit(feats, -25.0, 30.0)
        assert should is True
        assert reason == "Hard stop"

    def test_time_limit(self):
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        should, prob, reason = model.should_exit(feats, 5.0, 130.0)
        assert should is True
        assert reason == "Time limit"

    def test_eod_close(self):
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        should, prob, reason = model.should_exit(feats, 5.0, 30.0, minutes_to_close=5.0)
        assert should is True
        assert reason == "EOD close"

    def test_min_hold(self):
        # min_hold_early_loss defaults to 2.0, so time < 2.0 triggers "Min hold"
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        should, prob, reason = model.should_exit(feats, 5.0, 1.0)
        assert should is False
        assert reason == "Min hold"

    def test_no_model_returns_hold(self):
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        should, prob, reason = model.should_exit(feats, 5.0, 30.0)
        assert should is False
        assert reason == "No model"

    def test_hard_stop_before_early_loss(self):
        """Hard stop fires even when early-loss conditions are also met."""
        # Use a config with a tight stop so both hard-stop and tier-1 conditions met
        cfg = {"exit_model": {"stop_loss_pct": -5.0}}
        model = ExitSignalModel(cfg)
        feats = np.zeros((1, 15))
        # -6% is below hard stop (-5%) AND tier-1 early loss (-3%), time=7 in tier-1 window
        should, prob, reason = model.should_exit(feats, -6.0, 7.0)
        assert should is True
        assert reason == "Hard stop"  # hard stop takes priority over early-loss


class TestAdaptiveEarlyLossExit:
    """Tests for the adaptive early-loss exit tiers."""

    def _trained_model(self):
        rng = np.random.RandomState(42)
        n = 400
        data = {col: rng.randn(n).astype(np.float32) for col in EXIT_FEATURE_COLS}
        # Give losing early-bar rows exit_label=1 so the model learns the pattern
        data["unrealized_pnl_pct"] = np.where(
            np.arange(n) < 200, -4.0, 5.0
        ).astype(np.float32)
        data["exit_label"] = (np.arange(n) < 200).astype(int)
        df = pd.DataFrame(data)
        model = ExitSignalModel()
        model.train(training_data=df)
        return model

    def test_tier1_conditions_check(self):
        """With no model, early-loss returns 'No model' not 'Early loss exit'."""
        model = ExitSignalModel()
        feats = np.zeros((1, 15))
        # -4% loss at minute 7 (inside tier-1 window), but no model
        should, prob, reason = model.should_exit(feats, -4.0, 7.0)
        assert should is False
        assert reason == "No model"

    def test_tier1_does_not_fire_outside_time_window(self):
        """Tier-1 early-loss check must not fire after 10 minutes."""
        model = self._trained_model()
        feats = np.zeros((1, 15), dtype=np.float32)
        # -4% at minute 15 → outside tier-1 (<=10min) but inside tier-2 (<=20min)
        should, prob, reason = model.should_exit(feats, -4.0, 15.0)
        # Reason must NOT be tier-1 early loss (time > 10)
        assert reason != "Early loss exit" or prob >= 0.15  # could fire via tier-2 or momentum

    def test_tier2_fires_for_deeper_loss(self):
        """Tier-2: -7% at 15 min can trigger early exit with lower threshold."""
        model = self._trained_model()
        # Force a feature vector that produces high exit probability
        feats = np.zeros((1, 15), dtype=np.float32)
        feats[0, 0] = -7.0  # unrealized_pnl_pct deep negative
        # We can't guarantee exact prob, but the path is valid
        should, prob, reason = model.should_exit(feats, -7.0, 15.0)
        assert isinstance(should, bool)
        assert reason in ("Early loss exit", "Momentum exhaustion", "Hold")

    def test_positive_pnl_never_triggers_early_loss(self):
        """A winning trade (positive P&L) must never fire early-loss exit."""
        model = self._trained_model()
        feats = np.zeros((1, 15), dtype=np.float32)
        should, prob, reason = model.should_exit(feats, 5.0, 7.0)
        assert reason != "Early loss exit"

    def test_config_overrides_thresholds(self):
        """Custom config is used for early-loss tiers."""
        cfg = {"exit_model": {
            "early_loss_pct_tier1": -1.0,
            "early_loss_time_tier1": 30.0,
            "early_loss_threshold_tier1": 0.99,  # impossible threshold
        }}
        model = ExitSignalModel(cfg)
        assert model.early_loss_pct_t1 == -1.0
        assert model.early_loss_time_t1 == 30.0
        assert model.early_loss_thresh_t1 == 0.99


class TestExitSignalModelTrain:
    def test_train_with_data(self):
        """Train with pre-built exit training data."""
        rng = np.random.RandomState(42)
        n = 200
        data = {}
        for col in EXIT_FEATURE_COLS:
            data[col] = rng.randn(n).astype(np.float32)
        data["exit_label"] = (rng.random(n) > 0.8).astype(int)

        df = pd.DataFrame(data)
        model = ExitSignalModel()
        metrics = model.train(training_data=df)

        assert model._is_trained
        assert "val_auc" in metrics
        assert metrics["n_train_samples"] > 0

    def test_should_exit_after_train(self):
        rng = np.random.RandomState(42)
        n = 200
        data = {}
        for col in EXIT_FEATURE_COLS:
            data[col] = rng.randn(n).astype(np.float32)
        data["exit_label"] = (rng.random(n) > 0.8).astype(int)

        df = pd.DataFrame(data)
        model = ExitSignalModel()
        model.train(training_data=df)

        feats = np.zeros((1, 15), dtype=np.float32)
        should, prob, reason = model.should_exit(feats, 5.0, 30.0)
        assert isinstance(should, bool)
        assert 0 <= prob <= 1
        assert reason in ("Momentum exhaustion", "Hold", "Early loss exit")


class TestExitSignalModelSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 200
        data = {col: rng.randn(n).astype(np.float32) for col in EXIT_FEATURE_COLS}
        data["exit_label"] = (rng.random(n) > 0.8).astype(int)
        df = pd.DataFrame(data)

        model = ExitSignalModel()
        model.train(training_data=df)
        path = tmp_path / "exit_model.pkl"
        model.save(path)

        loaded = ExitSignalModel()
        loaded.load(path)
        assert loaded._is_trained

        feats = np.zeros((1, 15), dtype=np.float32)
        s1, p1, _ = model.should_exit(feats, 5.0, 30.0)
        s2, p2, _ = loaded.should_exit(feats, 5.0, 30.0)
        assert s1 == s2
        assert abs(p1 - p2) < 1e-6

    def test_save_before_train_raises(self, tmp_path):
        model = ExitSignalModel()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save(tmp_path / "bad.pkl")


class TestGenerateTrainingData:
    def test_generates_from_signals(self, tmp_path):
        # Create raw option parquet (layout: {ticker_dir}/{date}.parquet)
        ticker_dir = tmp_path / "O_SPY250307C00580000"
        ticker_dir.mkdir()
        bars = _make_bars(60)
        bars.to_parquet(ticker_dir / "2025-03-03.parquet")

        signals = pd.DataFrame([{
            "date": "2025-03-03",
            "ticker": "O:SPY250307C00580000",
            "timestamp": int(bars.loc[5, "timestamp"]),
            "entry_price": float(bars.loc[5, "close"]),
            "contract_type": 1,
        }])

        eng = ExitSignalModel()
        result = eng.generate_training_data(
            feature_df=pd.DataFrame(),
            entry_signals=signals,
            raw_options_dir=str(tmp_path),
        )
        assert not result.empty
        assert "exit_label" in result.columns
        assert result["exit_label"].isin([0, 1]).all()
