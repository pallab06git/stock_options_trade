# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for train_xgboost module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.ml.train_xgboost import (
    XGBoostTrainer,
    _NON_FEATURE_COLS,
    load_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features_df(
    n_dates: int = 10,
    bars_per_date: int = 20,
    positive_rate: float = 0.30,
    random_state: int = 0,
) -> pd.DataFrame:
    """Synthetic feature DataFrame that mimics the real features CSV."""
    rng = np.random.default_rng(random_state)
    base_ts = 1_740_992_400_000  # 2025-03-03 09:30 ET in ms

    rows = []
    for d in range(n_dates):
        date_str = f"2025-03-{d + 1:02d}"
        for b in range(bars_per_date):
            ts = base_ts + d * 86_400_000 + b * 60_000
            rows.append(
                {
                    "timestamp": ts,
                    "date": date_str,
                    "ticker": "O:SPY250303C00580000",
                    "source": "polygon",
                    # raw OHLCV (excluded from features)
                    "open": rng.uniform(1.0, 5.0),
                    "high": rng.uniform(1.0, 5.0),
                    "low": rng.uniform(1.0, 5.0),
                    "close": rng.uniform(1.0, 5.0),
                    "volume": int(rng.integers(100, 1000)),
                    "vwap": rng.uniform(1.0, 5.0),
                    "transactions": int(rng.integers(10, 100)),
                    "opt_close": rng.uniform(1.0, 5.0),
                    # engineered features (included in model)
                    "opt_return_1m": rng.normal(0, 0.02),
                    "opt_return_5m": rng.normal(0, 0.05),
                    "spy_return_1m": rng.normal(0, 0.005),
                    "spy_return_5m": rng.normal(0, 0.01),
                    "spy_rsi_14": rng.uniform(30, 70),
                    "spy_vol_ratio_5m": rng.uniform(0.5, 2.0),
                    "hour_et": 9 + b // 60,
                    "minute_et": b % 60,
                    "minutes_since_open": b,
                    "moneyness": rng.uniform(0.95, 1.05),
                    "time_to_expiry_days": rng.uniform(0, 5),
                    "implied_volatility": rng.uniform(0.15, 0.60),
                    "strike": 580.0,
                    "contract_type": 1,  # int encoded
                    "is_0dte": 1,
                    # label columns (excluded from features)
                    "target": int(rng.random() < positive_rate),
                    "max_gain_120m": rng.uniform(0, 1.5),
                    "time_to_max_min": rng.uniform(0, 120),
                }
            )

    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, directory: Path, date: str) -> Path:
    """Write df as ``{date}_features.csv`` in directory."""
    path = directory / f"{date}_features.csv"
    df.to_csv(path, index=False)
    return path


def _make_config(
    n_estimators: int = 10,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    return {
        "ml_training": {
            "xgboost": {
                "n_estimators": n_estimators,
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
                "model_version": "test",
            }
        },
        "data_preparation": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "balance_method": "undersample",
            "random_state": 42,
            "target_col": "target",
        },
    }


# ---------------------------------------------------------------------------
# TestLoadFeatures
# ---------------------------------------------------------------------------


class TestLoadFeatures:
    def test_loads_csv_in_range(self, tmp_path):
        df1 = _make_features_df(n_dates=1)
        df2 = _make_features_df(n_dates=1)
        _write_csv(df1, tmp_path, "2025-03-01")
        _write_csv(df2, tmp_path, "2025-03-02")

        result = load_features(tmp_path, start_date="2025-03-01", end_date="2025-03-02")
        assert len(result) == len(df1) + len(df2)

    def test_filters_by_start_date(self, tmp_path):
        df_early = _make_features_df(n_dates=1)
        df_late = _make_features_df(n_dates=1)
        _write_csv(df_early, tmp_path, "2025-03-01")
        _write_csv(df_late, tmp_path, "2025-03-10")

        result = load_features(tmp_path, start_date="2025-03-05")
        assert len(result) == len(df_late)

    def test_filters_by_end_date(self, tmp_path):
        _write_csv(_make_features_df(n_dates=1), tmp_path, "2025-03-01")
        _write_csv(_make_features_df(n_dates=1), tmp_path, "2025-03-10")

        result = load_features(tmp_path, end_date="2025-03-05")
        assert len(result) == len(_make_features_df(n_dates=1))

    def test_no_bounds_loads_all(self, tmp_path):
        for day in ("2025-03-01", "2025-03-02", "2025-03-03"):
            _write_csv(_make_features_df(n_dates=1), tmp_path, day)
        result = load_features(tmp_path)
        assert len(result) == 3 * 20  # 3 files × 20 bars default

    def test_empty_dir_returns_empty_df(self, tmp_path):
        result = load_features(tmp_path)
        assert result.empty

    def test_sorted_by_timestamp(self, tmp_path):
        # Write in reverse order — result must still be ascending
        _write_csv(_make_features_df(n_dates=1, random_state=1), tmp_path, "2025-03-02")
        _write_csv(_make_features_df(n_dates=1, random_state=0), tmp_path, "2025-03-01")
        result = load_features(tmp_path)
        assert list(result["timestamp"]) == sorted(result["timestamp"].tolist())

    def test_raises_if_dir_missing(self):
        with pytest.raises(FileNotFoundError, match="Features directory"):
            load_features("/nonexistent/path/to/features")

    def test_ignores_non_feature_csvs(self, tmp_path):
        """Files that don't match *_features.csv pattern are ignored."""
        _write_csv(_make_features_df(n_dates=1), tmp_path, "2025-03-01")
        # Write an unrelated CSV
        (tmp_path / "summary.csv").write_text("a,b\n1,2\n")
        result = load_features(tmp_path)
        assert len(result) == 20  # only the features file


# ---------------------------------------------------------------------------
# TestNonFeatureCols
# ---------------------------------------------------------------------------


class TestNonFeatureCols:
    def test_target_excluded(self):
        assert "target" in _NON_FEATURE_COLS

    def test_label_cols_excluded(self):
        for col in ("max_gain_120m", "max_gain_pct", "time_to_max_min"):
            assert col in _NON_FEATURE_COLS

    def test_raw_ohlcv_excluded(self):
        for col in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
            assert col in _NON_FEATURE_COLS

    def test_metadata_excluded(self):
        for col in ("date", "ticker", "timestamp", "source", "opt_close"):
            assert col in _NON_FEATURE_COLS


# ---------------------------------------------------------------------------
# TestXGBoostTrainerInit
# ---------------------------------------------------------------------------


class TestXGBoostTrainerInit:
    def test_default_params_applied(self):
        trainer = XGBoostTrainer({})
        assert trainer.n_estimators == 300
        assert trainer.max_depth == 6
        assert trainer.learning_rate == 0.05
        assert trainer.threshold == 0.50
        assert trainer.model_version == "v1"

    def test_config_overrides_applied(self):
        trainer = XGBoostTrainer(_make_config(n_estimators=50))
        assert trainer.n_estimators == 50
        assert trainer.max_depth == 3

    def test_xgb_params_dict_populated(self):
        trainer = XGBoostTrainer(_make_config())
        assert "n_estimators" in trainer.xgb_params
        assert "max_depth" in trainer.xgb_params
        assert "learning_rate" in trainer.xgb_params


# ---------------------------------------------------------------------------
# TestGetFeatureCols
# ---------------------------------------------------------------------------


class TestGetFeatureCols:
    def test_excludes_non_feature_cols(self):
        df = _make_features_df(n_dates=2)
        trainer = XGBoostTrainer(_make_config())
        feature_cols = trainer.get_feature_cols(df)
        for col in _NON_FEATURE_COLS:
            assert col not in feature_cols, f"Non-feature col found: {col}"

    def test_includes_engineered_features(self):
        df = _make_features_df(n_dates=2)
        trainer = XGBoostTrainer(_make_config())
        feature_cols = trainer.get_feature_cols(df)
        for col in ("opt_return_1m", "spy_rsi_14", "moneyness", "implied_volatility"):
            assert col in feature_cols

    def test_returns_sorted_list(self):
        df = _make_features_df(n_dates=2)
        trainer = XGBoostTrainer(_make_config())
        feature_cols = trainer.get_feature_cols(df)
        assert feature_cols == sorted(feature_cols)

    def test_empty_df_returns_empty_list(self):
        trainer = XGBoostTrainer(_make_config())
        # DataFrame with only non-feature columns
        df = pd.DataFrame(columns=list(_NON_FEATURE_COLS))
        feature_cols = trainer.get_feature_cols(df)
        assert feature_cols == []


# ---------------------------------------------------------------------------
# TestXGBoostTrainerTrain  (full pipeline, uses tmp_path)
# ---------------------------------------------------------------------------


class TestXGBoostTrainerTrain:
    """Integration-style tests for XGBoostTrainer.train().

    These tests use real (small) DataFrames and real XGBoost fits to verify
    the pipeline end-to-end without mocking the model.
    """

    def _prepare(self, tmp_path, n_dates: int = 15, bars: int = 30):
        """Write a single multi-day features CSV and return trainer + dirs."""
        # One big CSV for all dates, named for the first date
        df = _make_features_df(n_dates=n_dates, bars_per_date=bars, positive_rate=0.30)
        # Write one CSV per date so load_features picks them up
        for day_idx in range(n_dates):
            date_str = f"2025-03-{day_idx + 1:02d}"
            day_df = df[df["date"] == date_str]
            if not day_df.empty:
                _write_csv(day_df, tmp_path / "features", date_str)
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        return _make_config(), tmp_path / "features", models_dir, logs_dir

    @pytest.fixture(autouse=True)
    def _make_dirs(self, tmp_path):
        (tmp_path / "features").mkdir()

    def test_returns_metrics_dict_with_required_keys(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)

        required = {
            "val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc",
            "train_rows", "val_rows", "test_rows", "n_features",
            "model_path", "log_path",
        }
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_model_file_is_created(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        assert Path(metrics["model_path"]).exists()

    def test_log_file_is_created(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        assert Path(metrics["log_path"]).exists()

    def test_log_file_is_valid_json(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        with open(metrics["log_path"]) as fh:
            data = json.load(fh)
        assert "val_roc_auc" in data

    def test_model_artifact_has_required_keys(self, tmp_path):
        import joblib
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        artifact = joblib.load(metrics["model_path"])
        for key in ("model", "feature_cols", "threshold", "xgb_params", "saved_at"):
            assert key in artifact, f"Artifact missing key: {key}"

    def test_feature_cols_in_artifact_match_trainer(self, tmp_path):
        import joblib
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        artifact = joblib.load(metrics["model_path"])
        assert isinstance(artifact["feature_cols"], list)
        assert len(artifact["feature_cols"]) == metrics["n_features"]

    def test_val_metrics_are_floats_in_range(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        for key in ("val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc"):
            val = metrics[key]
            assert isinstance(val, float), f"{key} is not float"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_train_val_test_rows_are_positive(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        assert metrics["train_rows"] > 0
        assert metrics["val_rows"] > 0
        assert metrics["test_rows"] > 0

    def test_date_range_filter_respected(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path, n_dates=10)
        trainer = XGBoostTrainer(config)
        # Only first 5 dates
        metrics_full = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        metrics_partial = trainer.train(
            feat_dir,
            start_date="2025-03-01",
            end_date="2025-03-05",
            models_dir=models_dir,
            logs_dir=logs_dir,
        )
        total_partial = (
            metrics_partial["train_rows"]
            + metrics_partial["val_rows"]
            + metrics_partial["test_rows"]
        )
        total_full = (
            metrics_full["train_rows"]
            + metrics_full["val_rows"]
            + metrics_full["test_rows"]
        )
        assert total_partial < total_full

    def test_raises_on_empty_features_dir(self, tmp_path):
        trainer = XGBoostTrainer(_make_config())
        with pytest.raises(ValueError, match="no data loaded"):
            trainer.train(
                tmp_path / "features",
                models_dir=tmp_path / "models",
                logs_dir=tmp_path / "logs",
            )

    def test_raises_when_target_column_missing(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        # Remove target column from all CSVs
        for csv_path in feat_dir.glob("*_features.csv"):
            df = pd.read_csv(csv_path)
            df = df.drop(columns=["target"], errors="ignore")
            df.to_csv(csv_path, index=False)

        trainer = XGBoostTrainer(config)
        with pytest.raises(ValueError, match="target"):
            trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)

    def test_model_version_in_filename(self, tmp_path):
        config, feat_dir, models_dir, logs_dir = self._prepare(tmp_path)
        config["ml_training"]["xgboost"]["model_version"] = "v99"
        trainer = XGBoostTrainer(config)
        metrics = trainer.train(feat_dir, models_dir=models_dir, logs_dir=logs_dir)
        assert "v99" in metrics["model_path"]


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Unit tests for XGBoostTrainer._evaluate (via a tiny trained model)."""

    @pytest.fixture()
    def tiny_model(self, tmp_path):
        """Train a tiny XGBoost model on synthetic data."""
        import xgboost as xgb

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int8)
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
        model.fit(X, y)
        return model, X, y

    def test_returns_dict_with_five_metrics(self, tiny_model):
        model, X, y = tiny_model
        trainer = XGBoostTrainer(_make_config())
        metrics = trainer._evaluate(model, X, y, threshold=0.5)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    def test_all_metrics_are_floats(self, tiny_model):
        model, X, y = tiny_model
        trainer = XGBoostTrainer(_make_config())
        metrics = trainer._evaluate(model, X, y, threshold=0.5)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} not float"

    def test_all_metrics_in_unit_interval(self, tiny_model):
        model, X, y = tiny_model
        trainer = XGBoostTrainer(_make_config())
        metrics = trainer._evaluate(model, X, y, threshold=0.5)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"

    def test_threshold_affects_precision_recall_tradeoff(self, tiny_model):
        model, X, y = tiny_model
        trainer = XGBoostTrainer(_make_config())
        low_thresh = trainer._evaluate(model, X, y, threshold=0.10)
        high_thresh = trainer._evaluate(model, X, y, threshold=0.90)
        # High threshold → fewer positives predicted → lower recall (guaranteed)
        assert high_thresh["recall"] <= low_thresh["recall"]
        # Precision at high threshold may be 0.0 (no predictions) via zero_division=0;
        # only assert the tradeoff direction when predictions exist at both thresholds.
        if high_thresh["recall"] > 0 and low_thresh["recall"] > 0:
            assert high_thresh["precision"] >= low_thresh["precision"]
