# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for src/ml/cli.py — ML CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.ml.cli import ml_cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> Dict[str, Any]:
    """Minimal merged config that satisfies all ML module constructors."""
    return {
        "feature_engineering": {
            "start_date": "2025-03-01",
            "end_date": "2025-03-05",
            "features_dir": "data/processed/features",
            "target_threshold_pct": 20.0,
            "target_lookforward_minutes": 120,
            "lookback_windows_minutes": [1, 5],
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
        },
        "data_preparation": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "balance_method": "undersample",
            "random_state": 42,
            "target_col": "target",
        },
        "ml_training": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.10,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_weight": 1,
                "gamma": 0.0,
                "random_state": 42,
                "early_stopping_rounds": 5,
                "eval_metric": "logloss",
                "threshold": 0.50,
                "model_version": "v1",
            },
            "feature_importance": {
                "importance_type": "gain",
                "top_n": 5,
                "output_dir": "data/reports/feature_importance",
            },
            "backtest": {
                "output_dir": "data/reports/backtest",
            },
        },
        "ml_paths": {
            "models_dir": "models",
            "training_logs_dir": "data/logs/training",
        },
    }


def _runner() -> CliRunner:
    return CliRunner()


# Patch ConfigLoader so no real YAML files are needed
def _patch_config_loader(config: Dict[str, Any]):
    mock_loader = MagicMock()
    mock_loader.load.return_value = config
    return patch("src.ml.cli.ConfigLoader", return_value=mock_loader)


def _patch_setup_logger():
    return patch("src.ml.cli.setup_logger")


# ---------------------------------------------------------------------------
# TestGroupHelp
# ---------------------------------------------------------------------------


class TestGroupHelp:
    def test_group_help_exits_zero(self):
        result = _runner().invoke(ml_cli, ["--help"])
        assert result.exit_code == 0

    def test_group_help_lists_all_commands(self):
        result = _runner().invoke(ml_cli, ["--help"])
        for cmd in ("generate-features", "train", "feature-importance", "backtest"):
            assert cmd in result.output

    def test_generate_features_help(self):
        result = _runner().invoke(ml_cli, ["generate-features", "--help"])
        assert result.exit_code == 0
        assert "--start-date" in result.output
        assert "--end-date" in result.output

    def test_train_help(self):
        result = _runner().invoke(ml_cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--model-version" in result.output

    def test_feature_importance_help(self):
        result = _runner().invoke(ml_cli, ["feature-importance", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output
        assert "--importance-type" in result.output
        assert "--top-n" in result.output

    def test_backtest_help(self):
        result = _runner().invoke(ml_cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output


# ---------------------------------------------------------------------------
# TestGenerateFeatures
# ---------------------------------------------------------------------------


class TestGenerateFeatures:
    def _mock_engineer_stats(self) -> Dict[str, Any]:
        return {
            "dates_processed": 5,
            "dates_skipped": 0,
            "dates_failed": 0,
            "total_rows": 1000,
            "n_features": 66,
            "positive_rate": 0.58,
        }

    def test_success_prints_summary(self):
        config = _minimal_config()
        stats = self._mock_engineer_stats()

        mock_engineer = MagicMock()
        mock_engineer.run.return_value = stats

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.processing.ml_feature_engineer.MLFeatureEngineer", return_value=mock_engineer):
                result = _runner().invoke(ml_cli, ["generate-features"])

        assert result.exit_code == 0
        assert "Feature Engineering" in result.output
        assert "5" in result.output  # dates_processed

    def test_start_date_override_passed_to_engineer(self):
        config = _minimal_config()
        mock_engineer = MagicMock()
        mock_engineer.run.return_value = self._mock_engineer_stats()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.processing.ml_feature_engineer.MLFeatureEngineer", return_value=mock_engineer):
                _runner().invoke(
                    ml_cli, ["generate-features", "--start-date", "2025-04-01"]
                )

        # engineer.run() called with the overridden start date
        call_kwargs = mock_engineer.run.call_args
        assert call_kwargs.kwargs.get("start_date") == "2025-04-01" or \
               (call_kwargs.args and "2025-04-01" in str(call_kwargs))

    def test_end_date_override_applied(self):
        config = _minimal_config()
        mock_engineer = MagicMock()
        mock_engineer.run.return_value = self._mock_engineer_stats()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.processing.ml_feature_engineer.MLFeatureEngineer", return_value=mock_engineer):
                result = _runner().invoke(
                    ml_cli, ["generate-features", "--end-date", "2025-06-30"]
                )

        assert result.exit_code == 0

    def test_exception_exits_nonzero(self):
        config = _minimal_config()
        mock_engineer = MagicMock()
        mock_engineer.run.side_effect = RuntimeError("disk full")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.processing.ml_feature_engineer.MLFeatureEngineer", return_value=mock_engineer):
                result = _runner().invoke(ml_cli, ["generate-features"])

        assert result.exit_code == 1
        assert "disk full" in result.output or "Error" in result.output

    def test_positive_rate_formatted_as_percentage(self):
        config = _minimal_config()
        mock_engineer = MagicMock()
        mock_engineer.run.return_value = self._mock_engineer_stats()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.processing.ml_feature_engineer.MLFeatureEngineer", return_value=mock_engineer):
                result = _runner().invoke(ml_cli, ["generate-features"])

        assert "%" in result.output


# ---------------------------------------------------------------------------
# TestTrain
# ---------------------------------------------------------------------------


class TestTrain:
    def _mock_train_metrics(self) -> Dict[str, Any]:
        return {
            "train_rows": 700,
            "train_rows_balanced": 400,
            "val_rows": 150,
            "test_rows": 150,
            "n_features": 15,
            "best_iteration": 42,
            "best_score": 0.45,
            "threshold": 0.5,
            "val_accuracy": 0.72,
            "val_precision": 0.68,
            "val_recall": 0.61,
            "val_f1": 0.64,
            "val_roc_auc": 0.75,
            "model_path": "models/xgboost_v1.pkl",
            "log_path": "data/logs/training/training_20260220T120000.json",
        }

    def test_success_prints_summary(self):
        config = _minimal_config()
        metrics = self._mock_train_metrics()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = metrics

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.train_xgboost.XGBoostTrainer", return_value=mock_trainer):
                result = _runner().invoke(ml_cli, ["train"])

        assert result.exit_code == 0
        assert "Training Summary" in result.output
        assert "0.75" in result.output  # ROC-AUC

    def test_model_version_override(self):
        config = _minimal_config()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = self._mock_train_metrics()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.train_xgboost.XGBoostTrainer") as MockTrainer:
                MockTrainer.return_value = mock_trainer
                _runner().invoke(ml_cli, ["train", "--model-version", "v99"])
                # Config should have been patched with "v99"
                call_config = MockTrainer.call_args[0][0]
                assert call_config["ml_training"]["xgboost"]["model_version"] == "v99"

    def test_start_end_dates_passed_to_trainer(self):
        config = _minimal_config()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = self._mock_train_metrics()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.train_xgboost.XGBoostTrainer", return_value=mock_trainer):
                _runner().invoke(
                    ml_cli,
                    ["train", "--start-date", "2025-03-03", "--end-date", "2025-12-31"],
                )

        call_kwargs = mock_trainer.train.call_args
        assert "2025-03-03" in str(call_kwargs)
        assert "2025-12-31" in str(call_kwargs)

    def test_exception_exits_nonzero(self):
        config = _minimal_config()
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = ValueError("no data loaded")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.train_xgboost.XGBoostTrainer", return_value=mock_trainer):
                result = _runner().invoke(ml_cli, ["train"])

        assert result.exit_code == 1

    def test_model_path_printed_in_output(self):
        config = _minimal_config()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = self._mock_train_metrics()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.train_xgboost.XGBoostTrainer", return_value=mock_trainer):
                result = _runner().invoke(ml_cli, ["train"])

        assert "xgboost_v1.pkl" in result.output


# ---------------------------------------------------------------------------
# TestFeatureImportance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def _mock_df(self):
        import pandas as pd
        return pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c"],
            "importance": [10.0, 5.0, 1.0],
            "importance_pct": [0.625, 0.3125, 0.0625],
            "rank": [1, 2, 3],
        })

    def test_model_path_required(self):
        result = _runner().invoke(ml_cli, ["feature-importance"])
        assert result.exit_code != 0
        assert "model-path" in result.output.lower() or "Missing" in result.output

    def test_success_prints_chart(self):
        config = _minimal_config()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._mock_df()
        mock_analyzer.plot_summary.return_value = "Feature Importance (gain)\n feat_a ████ 62.5%"

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.feature_importance.FeatureImportanceAnalyzer", return_value=mock_analyzer):
                result = _runner().invoke(
                    ml_cli, ["feature-importance", "--model-path", "models/xgboost_v1.pkl"]
                )

        assert result.exit_code == 0
        assert "Feature Importance" in result.output

    def test_importance_type_override_applied(self):
        config = _minimal_config()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._mock_df()
        mock_analyzer.plot_summary.return_value = "chart"

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.feature_importance.FeatureImportanceAnalyzer") as MockAnalyzer:
                MockAnalyzer.return_value = mock_analyzer
                _runner().invoke(
                    ml_cli,
                    [
                        "feature-importance",
                        "--model-path", "models/xgboost_v1.pkl",
                        "--importance-type", "weight",
                    ],
                )
                call_config = MockAnalyzer.call_args[0][0]
                assert call_config["ml_training"]["feature_importance"]["importance_type"] == "weight"

    def test_top_n_override_applied(self):
        config = _minimal_config()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._mock_df()
        mock_analyzer.plot_summary.return_value = "chart"

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.feature_importance.FeatureImportanceAnalyzer") as MockAnalyzer:
                MockAnalyzer.return_value = mock_analyzer
                _runner().invoke(
                    ml_cli,
                    [
                        "feature-importance",
                        "--model-path", "models/xgboost_v1.pkl",
                        "--top-n", "10",
                    ],
                )
                call_config = MockAnalyzer.call_args[0][0]
                assert call_config["ml_training"]["feature_importance"]["top_n"] == 10

    def test_exception_exits_nonzero(self):
        config = _minimal_config()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = FileNotFoundError("model not found")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.feature_importance.FeatureImportanceAnalyzer", return_value=mock_analyzer):
                result = _runner().invoke(
                    ml_cli, ["feature-importance", "--model-path", "missing.pkl"]
                )

        assert result.exit_code == 1

    def test_invalid_importance_type_rejected(self):
        """Click should reject an invalid --importance-type value."""
        result = _runner().invoke(
            ml_cli,
            ["feature-importance", "--model-path", "m.pkl", "--importance-type", "shap"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# TestBacktest
# ---------------------------------------------------------------------------


class TestBacktest:
    def _mock_result(self) -> Dict[str, Any]:
        return {
            "metrics": {
                "n_test_rows": 200,
                "n_signals": 40,
                "n_true_positives": 22,
                "n_false_positives": 18,
                "signal_rate": 0.20,
                "positive_rate_test": 0.30,
                "precision": 0.55,
                "recall": 0.37,
                "f1": 0.44,
                "roc_auc": 0.72,
                "avg_gain_all_bars": 0.35,
                "avg_gain_signals": 0.60,
                "avg_gain_tp": 0.95,
                "avg_gain_fp": 0.18,
                "lift": 1.71,
            },
            "model_path": "models/xgboost_v1.pkl",
            "trades_path": "data/reports/backtest/xgboost_v1_trades_20260220T120000.csv",
            "report_path": "data/reports/backtest/xgboost_v1_backtest_20260220T120000.json",
        }

    def test_model_path_required(self):
        result = _runner().invoke(ml_cli, ["backtest"])
        assert result.exit_code != 0
        assert "model-path" in result.output.lower() or "Missing" in result.output

    def test_success_prints_summary(self):
        config = _minimal_config()
        mock_backtester = MagicMock()
        mock_backtester.run.return_value = self._mock_result()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                result = _runner().invoke(
                    ml_cli, ["backtest", "--model-path", "models/xgboost_v1.pkl"]
                )

        assert result.exit_code == 0
        assert "Backtest Summary" in result.output
        assert "200" in result.output   # n_test_rows
        assert "40" in result.output    # n_signals

    def test_lift_shown_in_output(self):
        config = _minimal_config()
        mock_backtester = MagicMock()
        mock_backtester.run.return_value = self._mock_result()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                result = _runner().invoke(
                    ml_cli, ["backtest", "--model-path", "models/xgboost_v1.pkl"]
                )

        assert "1.71" in result.output  # lift

    def test_none_lift_shows_na(self):
        config = _minimal_config()
        result_data = self._mock_result()
        result_data["metrics"]["lift"] = None
        result_data["metrics"]["avg_gain_all_bars"] = None
        result_data["metrics"]["avg_gain_signals"] = None
        mock_backtester = MagicMock()
        mock_backtester.run.return_value = result_data

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                result = _runner().invoke(
                    ml_cli, ["backtest", "--model-path", "models/xgboost_v1.pkl"]
                )

        assert result.exit_code == 0
        assert "n/a" in result.output

    def test_date_range_passed_to_backtester(self):
        config = _minimal_config()
        mock_backtester = MagicMock()
        mock_backtester.run.return_value = self._mock_result()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                _runner().invoke(
                    ml_cli,
                    [
                        "backtest",
                        "--model-path", "models/xgboost_v1.pkl",
                        "--start-date", "2025-03-03",
                        "--end-date", "2025-12-31",
                    ],
                )

        call_kwargs = mock_backtester.run.call_args
        assert "2025-03-03" in str(call_kwargs)
        assert "2025-12-31" in str(call_kwargs)

    def test_exception_exits_nonzero(self):
        config = _minimal_config()
        mock_backtester = MagicMock()
        mock_backtester.run.side_effect = FileNotFoundError("model not found")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                result = _runner().invoke(
                    ml_cli, ["backtest", "--model-path", "missing.pkl"]
                )

        assert result.exit_code == 1

    def test_trades_and_report_paths_in_output(self):
        config = _minimal_config()
        mock_backtester = MagicMock()
        mock_backtester.run.return_value = self._mock_result()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("src.ml.backtest.ModelBacktester", return_value=mock_backtester):
                result = _runner().invoke(
                    ml_cli, ["backtest", "--model-path", "models/xgboost_v1.pkl"]
                )

        assert "trades" in result.output
        assert "backtest" in result.output


# ---------------------------------------------------------------------------
# TestMainCliIntegration
# ---------------------------------------------------------------------------


class TestMainCliIntegration:
    """Verify ml subgroup is reachable from the top-level CLI."""

    def test_ml_subgroup_registered_in_main_cli(self):
        from src.cli import cli

        result = _runner().invoke(cli, ["ml", "--help"])
        assert result.exit_code == 0
        assert "generate-features" in result.output

    def test_all_four_commands_accessible_from_main_cli(self):
        from src.cli import cli

        for cmd in ("generate-features", "train", "feature-importance", "backtest"):
            result = _runner().invoke(cli, ["ml", cmd, "--help"])
            assert result.exit_code == 0, f"ml {cmd} --help failed: {result.output}"
