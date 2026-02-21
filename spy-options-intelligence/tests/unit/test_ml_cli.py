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

    def test_full_comparison_accessible_from_main_cli(self):
        from src.cli import cli

        result = _runner().invoke(cli, ["ml", "full-comparison", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output


# ---------------------------------------------------------------------------
# TestFullComparison
# ---------------------------------------------------------------------------


def _build_mock_artifact(feature_cols=None):
    """Return a fake joblib artifact dict."""
    import numpy as np

    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b", "feat_c"]

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
    mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

    return {
        "model": mock_model,
        "feature_cols": feature_cols,
        "params": {"n_estimators": 100},
        "optimization_score": 0.91,
        "model_type": "xgboost",
    }


def _build_mock_test_df(feature_cols=None, n_rows=10):
    """Return a minimal feature DataFrame compatible with ModelComparator."""
    import numpy as np
    import pandas as pd

    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b", "feat_c"]

    data = {col: np.random.randn(n_rows) for col in feature_cols}
    data.update(
        {
            "date": ["2026-01-02"] * n_rows,
            "ticker": ["O:SPY260102C00500000"] * n_rows,
            "close": np.random.uniform(1.0, 5.0, n_rows),
            "max_gain_120m": np.random.uniform(0.0, 0.5, n_rows),
            "min_loss_120m": np.random.uniform(-0.3, 0.0, n_rows),
            "target": np.random.randint(0, 2, n_rows),
        }
    )
    return pd.DataFrame(data)


class TestFullComparisonHelp:
    """Help text and argument validation tests (no real IO)."""

    def test_help_exits_zero(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert result.exit_code == 0

    def test_help_shows_model_path_option(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert "--model-path" in result.output

    def test_help_shows_test_start_date(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert "--test-start-date" in result.output

    def test_help_shows_test_end_date(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert "--test-end-date" in result.output

    def test_help_shows_thresholds(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert "--thresholds" in result.output

    def test_help_shows_output(self):
        result = _runner().invoke(ml_cli, ["full-comparison", "--help"])
        assert "--output" in result.output

    def test_missing_model_path_fails(self):
        result = _runner().invoke(
            ml_cli,
            [
                "full-comparison",
                "--test-start-date", "2026-01-01",
                "--test-end-date", "2026-01-31",
            ],
        )
        assert result.exit_code != 0

    def test_missing_test_start_date_fails(self):
        result = _runner().invoke(
            ml_cli,
            [
                "full-comparison",
                "--model-path", "xgb=models/xgb.pkl",
                "--test-end-date", "2026-01-31",
            ],
        )
        assert result.exit_code != 0

    def test_missing_test_end_date_fails(self):
        result = _runner().invoke(
            ml_cli,
            [
                "full-comparison",
                "--model-path", "xgb=models/xgb.pkl",
                "--test-start-date", "2026-01-01",
            ],
        )
        assert result.exit_code != 0

    def test_bad_model_path_format_exits_nonzero(self):
        """NAME=PATH format required; plain path should fail gracefully."""
        config = _minimal_config()
        with _patch_config_loader(config), _patch_setup_logger():
            result = _runner().invoke(
                ml_cli,
                [
                    "full-comparison",
                    "--model-path", "models/xgb.pkl",  # missing NAME=
                    "--test-start-date", "2026-01-01",
                    "--test-end-date", "2026-01-31",
                ],
            )
        assert result.exit_code != 0

    def test_bad_thresholds_format_exits_nonzero(self):
        """Non-numeric thresholds should fail gracefully."""
        config = _minimal_config()
        with _patch_config_loader(config), _patch_setup_logger():
            result = _runner().invoke(
                ml_cli,
                [
                    "full-comparison",
                    "--model-path", "xgb=models/xgb.pkl",
                    "--test-start-date", "2026-01-01",
                    "--test-end-date", "2026-01-31",
                    "--thresholds", "high,medium,low",
                ],
            )
        assert result.exit_code != 0


class TestFullComparisonRun:
    """End-to-end tests with mocked ML dependencies."""

    def _base_args(self, model_path: str, output: str) -> list:
        return [
            "full-comparison",
            "--model-path", f"xgboost={model_path}",
            "--test-start-date", "2026-01-01",
            "--test-end-date", "2026-01-31",
            "--thresholds", "0.80,0.90",
            "--output", output,
        ]

    def test_success_exit_code(self, tmp_path):
        config = _minimal_config()
        artifact = _build_mock_artifact()
        test_df = _build_mock_test_df(artifact["feature_cols"])

        mock_comparator = MagicMock()
        mock_comparator.model_names = ["xgboost"]
        mock_comparator.evaluate_at_thresholds.return_value = {
            0.80: {
                "total_signals": 5,
                "win_rate": 0.6,
                "net_profit_usd": 1500.0,
            },
            0.90: {
                "total_signals": 2,
                "win_rate": 0.8,
                "net_profit_usd": 800.0,
            },
        }
        mock_comparator.get_best_threshold_per_model.return_value = {
            "xgboost": {
                "best_threshold": 0.80,
                "total_signals": 5,
                "win_rate": 0.6,
                "net_profit_usd": 1500.0,
            }
        }
        import pandas as pd

        mock_comparator.generate_comparison_report.return_value = pd.DataFrame(
            [{"Model": "xgboost", "Net Profit": "$+1,500", "Meets Target": "NO"}]
        )
        mock_comparator.find_signal_overlap.return_value = {
            "total_unique_signals": 5,
            "all_models_agree": 5,
            "majority_agree": 5,
            "overlap_breakdown": {},
        }
        mock_comparator.save_results.return_value = None

        model_pkl = tmp_path / "xgb.pkl"
        model_pkl.write_bytes(b"fake")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", return_value=artifact):
                with patch("src.ml.model_comparator.ModelComparator", return_value=mock_comparator):
                    with patch("src.ml.train_xgboost.load_features", return_value=test_df):
                        result = _runner().invoke(
                            ml_cli,
                            self._base_args(str(model_pkl), str(tmp_path)),
                        )

        assert result.exit_code == 0, result.output

    def test_comparison_table_printed(self, tmp_path):
        config = _minimal_config()
        artifact = _build_mock_artifact()
        test_df = _build_mock_test_df(artifact["feature_cols"])

        import pandas as pd

        mock_comparator = MagicMock()
        mock_comparator.model_names = ["xgboost"]
        mock_comparator.evaluate_at_thresholds.return_value = {
            0.80: {"total_signals": 3, "win_rate": 0.7, "net_profit_usd": 900.0},
        }
        mock_comparator.get_best_threshold_per_model.return_value = {
            "xgboost": {"best_threshold": 0.80, "total_signals": 3,
                        "win_rate": 0.7, "net_profit_usd": 900.0}
        }
        mock_comparator.generate_comparison_report.return_value = pd.DataFrame(
            [{"Model": "xgboost", "Net Profit": "$+900", "Meets Target": "NO"}]
        )
        mock_comparator.save_results.return_value = None

        model_pkl = tmp_path / "xgb.pkl"
        model_pkl.write_bytes(b"fake")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", return_value=artifact):
                with patch("src.ml.model_comparator.ModelComparator", return_value=mock_comparator):
                    with patch("src.ml.train_xgboost.load_features", return_value=test_df):
                        result = _runner().invoke(
                            ml_cli,
                            self._base_args(str(model_pkl), str(tmp_path)),
                        )

        assert "Side-by-Side Comparison" in result.output

    def test_dashboard_hint_printed(self, tmp_path):
        config = _minimal_config()
        artifact = _build_mock_artifact()
        test_df = _build_mock_test_df(artifact["feature_cols"])

        import pandas as pd

        mock_comparator = MagicMock()
        mock_comparator.model_names = ["xgboost"]
        mock_comparator.evaluate_at_thresholds.return_value = {
            0.80: {"total_signals": 2, "win_rate": 0.5, "net_profit_usd": 200.0},
        }
        mock_comparator.get_best_threshold_per_model.return_value = {
            "xgboost": {"best_threshold": 0.80, "total_signals": 2,
                        "win_rate": 0.5, "net_profit_usd": 200.0}
        }
        mock_comparator.generate_comparison_report.return_value = pd.DataFrame(
            [{"Model": "xgboost"}]
        )
        mock_comparator.save_results.return_value = None

        model_pkl = tmp_path / "xgb.pkl"
        model_pkl.write_bytes(b"fake")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", return_value=artifact):
                with patch("src.ml.model_comparator.ModelComparator", return_value=mock_comparator):
                    with patch("src.ml.train_xgboost.load_features", return_value=test_df):
                        result = _runner().invoke(
                            ml_cli,
                            self._base_args(str(model_pkl), str(tmp_path)),
                        )

        assert "streamlit run src/ml/dashboard.py" in result.output

    def test_missing_model_file_exits_nonzero(self, tmp_path):
        config = _minimal_config()
        test_df = _build_mock_test_df()

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", side_effect=FileNotFoundError("not found")):
                with patch("src.ml.train_xgboost.load_features", return_value=test_df):
                    result = _runner().invoke(
                        ml_cli,
                        self._base_args("/no/such/file.pkl", str(tmp_path)),
                    )

        assert result.exit_code != 0

    def test_empty_test_features_exits_nonzero(self, tmp_path):
        import pandas as pd

        config = _minimal_config()
        artifact = _build_mock_artifact()

        model_pkl = tmp_path / "xgb.pkl"
        model_pkl.write_bytes(b"fake")

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", return_value=artifact):
                with patch("src.ml.train_xgboost.load_features", return_value=pd.DataFrame()):
                    result = _runner().invoke(
                        ml_cli,
                        self._base_args(str(model_pkl), str(tmp_path)),
                    )

        assert result.exit_code != 0

    def test_multiple_models_registered(self, tmp_path):
        config = _minimal_config()
        artifact = _build_mock_artifact()
        test_df = _build_mock_test_df(artifact["feature_cols"])

        import pandas as pd

        mock_comparator = MagicMock()
        mock_comparator.model_names = ["xgboost", "lightgbm"]
        mock_comparator.evaluate_at_thresholds.return_value = {
            0.80: {"total_signals": 5, "win_rate": 0.6, "net_profit_usd": 1000.0},
        }
        mock_comparator.get_best_threshold_per_model.return_value = {
            "xgboost": {"best_threshold": 0.80, "total_signals": 5,
                        "win_rate": 0.6, "net_profit_usd": 1000.0},
            "lightgbm": {"best_threshold": 0.80, "total_signals": 4,
                         "win_rate": 0.7, "net_profit_usd": 1200.0},
        }
        mock_comparator.generate_comparison_report.return_value = pd.DataFrame(
            [{"Model": "xgboost"}, {"Model": "lightgbm"}]
        )
        mock_comparator.find_signal_overlap.return_value = {
            "total_unique_signals": 8,
            "all_models_agree": 3,
            "majority_agree": 3,
            "overlap_breakdown": {"1_models": 5, "2_models": 3},
        }
        mock_comparator.save_results.return_value = None

        model_pkl = tmp_path / "xgb.pkl"
        model_pkl.write_bytes(b"fake")

        args = [
            "full-comparison",
            "--model-path", f"xgboost={model_pkl}",
            "--model-path", f"lightgbm={model_pkl}",
            "--test-start-date", "2026-01-01",
            "--test-end-date", "2026-01-31",
            "--thresholds", "0.80",
            "--output", str(tmp_path),
        ]

        with _patch_config_loader(config), _patch_setup_logger():
            with patch("joblib.load", return_value=artifact):
                with patch("src.ml.model_comparator.ModelComparator", return_value=mock_comparator):
                    with patch("src.ml.train_xgboost.load_features", return_value=test_df):
                        result = _runner().invoke(ml_cli, args)

        assert result.exit_code == 0, result.output
        # Overlap section should appear when ≥2 models
        assert "Signal Overlap" in result.output
        # add_model called twice (once per model)
        assert mock_comparator.add_model.call_count == 2

    def test_full_comparison_in_group_help(self):
        result = _runner().invoke(ml_cli, ["--help"])
        assert "full-comparison" in result.output
