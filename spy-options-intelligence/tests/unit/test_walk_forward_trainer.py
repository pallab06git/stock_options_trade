# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for WalkForwardTrainer."""

import numpy as np
import pandas as pd
import pytest

from src.ml.walk_forward_trainer import WalkForwardTrainer, _get_monthly_boundaries


class TestGetMonthlyBoundaries:
    def test_basic(self):
        dates = ["2025-03-03", "2025-03-04", "2025-04-01", "2025-04-15"]
        result = _get_monthly_boundaries(dates)
        assert len(result) == 2
        assert result[0] == ("2025-03-03", "2025-03-04")
        assert result[1] == ("2025-04-01", "2025-04-15")

    def test_empty(self):
        assert _get_monthly_boundaries([]) == []

    def test_single_date(self):
        result = _get_monthly_boundaries(["2025-03-15"])
        assert result == [("2025-03-15", "2025-03-15")]

    def test_sorted_output(self):
        dates = ["2025-05-01", "2025-03-01", "2025-04-01"]
        result = _get_monthly_boundaries(dates)
        months = [r[0][:7] for r in result]
        assert months == ["2025-03", "2025-04", "2025-05"]

    def test_many_months(self):
        dates = [f"2025-{m:02d}-15" for m in range(1, 13)]
        result = _get_monthly_boundaries(dates)
        assert len(result) == 12


class TestWalkForwardTrainerInit:
    def test_defaults(self):
        trainer = WalkForwardTrainer()
        assert trainer.position_size == 10_000
        assert trainer.max_signals_per_day == 3
        assert trainer.entry_threshold == 0.70
        assert trainer.min_agreement == 2

    def test_custom_config(self):
        cfg = {
            "signal_pipeline": {
                "position_size_usd": 5000,
                "max_signals_per_day": 2,
                "entry_threshold": 0.80,
                "min_model_agreement": 3,
            }
        }
        trainer = WalkForwardTrainer(cfg)
        assert trainer.position_size == 5000
        assert trainer.max_signals_per_day == 2
        assert trainer.entry_threshold == 0.80
        assert trainer.min_agreement == 3


class TestWalkForwardTrainerValidation:
    def test_no_features_raises(self, tmp_path):
        trainer = WalkForwardTrainer()
        with pytest.raises(ValueError, match="No feature files"):
            trainer.run(features_dir=str(tmp_path))

    def test_insufficient_months_raises(self, tmp_path):
        # Create 4 months of empty feature files (need 7 for min_train=6)
        for month in range(3, 7):
            fp = tmp_path / f"2025-{month:02d}-01_features.csv"
            fp.write_text("timestamp,target\n1,0\n")

        trainer = WalkForwardTrainer()
        with pytest.raises(ValueError, match="Need at least"):
            trainer.run(features_dir=str(tmp_path), min_train_months=6)


class TestComputeAgreementMask:
    def test_agreement_mask_shape(self):
        """Agreement mask returns correct shape."""
        trainer = WalkForwardTrainer()
        trainer.min_agreement = 2

        # Create a mock ensemble with known base models
        class MockModel:
            def predict_proba(self, X):
                return np.column_stack([1 - np.ones(len(X)) * 0.7,
                                         np.ones(len(X)) * 0.7])

        class MockEnsemble:
            _base_models = {"xgboost": MockModel(), "lightgbm": MockModel(),
                            "random_forest": MockModel()}
            _feature_cols = ["f1", "f2"]

        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        mask = trainer._compute_agreement_mask(MockEnsemble(), df, ["f1", "f2"])
        assert mask.shape == (3,)
        assert mask.all()  # all 3 models > 0.5

    def test_disagreement_filtered(self):
        """When models disagree, mask is False."""
        trainer = WalkForwardTrainer()
        trainer.min_agreement = 2

        class HighModel:
            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.3,
                                         np.ones(len(X)) * 0.7])

        class LowModel:
            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.8,
                                         np.ones(len(X)) * 0.2])

        class MockEnsemble:
            _base_models = {"xgboost": HighModel(), "lightgbm": LowModel(),
                            "random_forest": LowModel()}
            _feature_cols = ["f1"]

        df = pd.DataFrame({"f1": [1, 2]})
        mask = trainer._compute_agreement_mask(MockEnsemble(), df, ["f1"])
        assert not mask.any()  # only 1/3 > 0.5, need 2


class TestSaveResults:
    def test_saves_files(self, tmp_path):
        from src.ml.trade_simulator import Trade
        trainer = WalkForwardTrainer()

        trade = Trade(
            trade_id=1,
            entry_time="2025-10-01 10:30 ET",
            contract_symbol="O:SPY251001C00580000",
            entry_price_per_share=5.0,
            position_size_usd=10000,
            confidence=0.75,
        )
        trade.close_trade(
            exit_time="2025-10-01 10:40 ET",
            exit_price_per_share=5.5,
            exit_reason="Hold",
            time_in_trade_minutes=10.0,
        )

        result = {"method": "walk_forward", "folds": []}
        trainer._save_results(result, [trade], str(tmp_path))

        assert (tmp_path / "walk_forward_results.json").exists()
        assert (tmp_path / "walk_forward_trades.csv").exists()

    def test_empty_trades(self, tmp_path):
        trainer = WalkForwardTrainer()
        result = {"method": "walk_forward", "folds": []}
        trainer._save_results(result, [], str(tmp_path))
        assert (tmp_path / "walk_forward_results.json").exists()
        assert not (tmp_path / "walk_forward_trades.csv").exists()
