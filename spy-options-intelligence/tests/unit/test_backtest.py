# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for the backtest module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.ml.backtest import (
    ModelBacktester,
    _build_trades_df,
    _compute_metrics,
    backtest_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_tiny_model(n_features: int = 6, n_samples: int = 300) -> xgb.XGBClassifier:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int8)
    model = xgb.XGBClassifier(
        n_estimators=10, max_depth=3, random_state=0, eval_metric="logloss"
    )
    model.fit(X, y)
    return model


def _make_feature_cols(n: int = 6) -> List[str]:
    return [f"feat_{i}" for i in range(n)]


def _make_test_df(
    n_rows: int = 100,
    n_features: int = 6,
    positive_rate: float = 0.30,
    random_state: int = 0,
    with_gain: bool = True,
    with_meta: bool = True,
) -> pd.DataFrame:
    """Synthetic test-split DataFrame."""
    rng = np.random.default_rng(random_state)
    base_ts = 1_740_992_400_000

    data: Dict[str, Any] = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows)

    data["target"] = (rng.random(n_rows) < positive_rate).astype(np.int8)

    if with_gain:
        data["max_gain_120m"] = rng.uniform(0, 1.5, n_rows)
        data["time_to_max_min"] = rng.uniform(0, 120, n_rows)

    if with_meta:
        data["timestamp"] = [base_ts + i * 60_000 for i in range(n_rows)]
        data["date"] = "2025-06-01"
        data["ticker"] = "O:SPY250601C00580000"

    return pd.DataFrame(data)


def _make_artifact(model: xgb.XGBClassifier, feature_cols: List[str]) -> Dict[str, Any]:
    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": 0.5,
        "xgb_params": {},
        "saved_at": "2026-02-20T12:00:00",
    }


def _save_artifact(artifact: Dict[str, Any], path: Path) -> Path:
    joblib.dump(artifact, path)
    return path


def _make_config(train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, Any]:
    return {
        "ml_training": {
            "backtest": {
                "output_dir": "data/reports/backtest",
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


def _write_features_csv(df: pd.DataFrame, directory: Path, date: str) -> Path:
    path = directory / f"{date}_features.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# TestBacktestModel — module-level function
# ---------------------------------------------------------------------------


class TestBacktestModel:
    @pytest.fixture()
    def model_cols_df(self):
        model = _train_tiny_model()
        cols = _make_feature_cols()
        df = _make_test_df(n_rows=100)
        return model, cols, df

    def test_returns_tuple_of_metrics_and_df(self, model_cols_df):
        model, cols, df = model_cols_df
        result = backtest_model(model, cols, df)
        assert isinstance(result, tuple) and len(result) == 2
        metrics, trades = result
        assert isinstance(metrics, dict)
        assert isinstance(trades, pd.DataFrame)

    def test_metrics_has_required_keys(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        required = {
            "n_test_rows", "n_signals", "n_true_positives", "n_false_positives",
            "signal_rate", "positive_rate_test", "precision", "recall", "f1",
            "roc_auc", "avg_gain_all_bars", "avg_gain_signals", "lift",
        }
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_n_test_rows_equals_df_length(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        assert metrics["n_test_rows"] == len(df)

    def test_signals_split_into_tp_and_fp(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        assert metrics["n_true_positives"] + metrics["n_false_positives"] == metrics["n_signals"]

    def test_precision_in_unit_interval(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        assert 0.0 <= metrics["precision"] <= 1.0

    def test_recall_in_unit_interval(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_signal_rate_in_unit_interval(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        assert 0.0 <= metrics["signal_rate"] <= 1.0

    def test_trades_df_only_contains_predicted_positives(self, model_cols_df):
        model, cols, df = model_cols_df
        metrics, trades = backtest_model(model, cols, df)
        assert len(trades) == metrics["n_signals"]

    def test_trades_has_predicted_proba_column(self, model_cols_df):
        model, cols, df = model_cols_df
        _, trades = backtest_model(model, cols, df)
        if not trades.empty:
            assert "predicted_proba" in trades.columns

    def test_trades_predicted_proba_above_threshold(self, model_cols_df):
        """All trades must have proba >= threshold (default 0.5)."""
        model, cols, df = model_cols_df
        _, trades = backtest_model(model, cols, df, threshold=0.5)
        if not trades.empty:
            assert (trades["predicted_proba"] >= 0.5).all()

    def test_custom_threshold_reduces_signals(self, model_cols_df):
        """Higher threshold → fewer (more selective) signals."""
        model, cols, df = model_cols_df
        m_low, _ = backtest_model(model, cols, df, threshold=0.20)
        m_high, _ = backtest_model(model, cols, df, threshold=0.80)
        assert m_high["n_signals"] <= m_low["n_signals"]

    def test_raises_on_empty_df(self, model_cols_df):
        model, cols, _ = model_cols_df
        empty = pd.DataFrame(columns=[f"feat_{i}" for i in range(6)] + ["target"])
        with pytest.raises(ValueError, match="empty"):
            backtest_model(model, cols, empty)

    def test_raises_on_missing_target_column(self, model_cols_df):
        model, cols, df = model_cols_df
        df_no_target = df.drop(columns=["target"])
        with pytest.raises(ValueError, match="target"):
            backtest_model(model, cols, df_no_target)

    def test_raises_on_missing_feature_column(self, model_cols_df):
        model, cols, df = model_cols_df
        df_missing = df.drop(columns=["feat_0"])
        with pytest.raises(ValueError, match="feat_0"):
            backtest_model(model, cols, df_missing)

    def test_lift_is_ratio_of_signal_gain_to_baseline(self, model_cols_df):
        """lift = avg_gain_signals / avg_gain_all_bars (when both are positive)."""
        model, cols, df = model_cols_df
        metrics, _ = backtest_model(model, cols, df)
        if (
            metrics["lift"] is not None
            and metrics["avg_gain_all_bars"] is not None
            and metrics["avg_gain_signals"] is not None
            and metrics["avg_gain_all_bars"] > 0
        ):
            expected_lift = metrics["avg_gain_signals"] / metrics["avg_gain_all_bars"]
            assert abs(metrics["lift"] - expected_lift) < 1e-5

    def test_no_max_gain_col_produces_none_gain_metrics(self):
        """When max_gain_120m is absent, gain metrics must be None."""
        model = _train_tiny_model()
        cols = _make_feature_cols()
        df = _make_test_df(n_rows=80, with_gain=False)
        metrics, _ = backtest_model(model, cols, df)
        assert metrics["avg_gain_all_bars"] is None
        assert metrics["avg_gain_signals"] is None
        assert metrics["lift"] is None

    def test_meta_columns_in_trades_when_present(self, model_cols_df):
        model, cols, df = model_cols_df
        _, trades = backtest_model(model, cols, df)
        if not trades.empty:
            for col in ("date", "ticker", "timestamp"):
                if col in df.columns:
                    assert col in trades.columns

    def test_is_true_positive_flag_correct(self, model_cols_df):
        model, cols, df = model_cols_df
        _, trades = backtest_model(model, cols, df)
        if not trades.empty and "actual_target" in trades.columns:
            expected = trades["actual_target"] == 1
            assert (trades["is_true_positive"] == expected).all()


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _dummy_df(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame({"max_gain_120m": rng.uniform(0, 1.5, n)})

    def test_all_correct_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        probas = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
        df = self._dummy_df(5)
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0])
        probas = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
        df = self._dummy_df(5)
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_n_signals_counts_predicted_positives(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        probas = np.array([0.1, 0.9, 0.1, 0.4, 0.2])
        df = self._dummy_df(5)
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert m["n_signals"] == 1

    def test_tp_fp_sum_to_signals(self):
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        probas = rng.uniform(0, 1, n)
        df = self._dummy_df(n)
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert m["n_true_positives"] + m["n_false_positives"] == m["n_signals"]

    def test_no_gain_col_gives_none(self):
        y_true = np.array([0, 1])
        y_pred = np.array([1, 1])
        probas = np.array([0.8, 0.9])
        df = pd.DataFrame({"other_col": [1.0, 2.0]})
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert m["avg_gain_all_bars"] is None
        assert m["lift"] is None

    def test_positive_rate_test_computed_correctly(self):
        y_true = np.array([1, 1, 0, 0, 0])  # 2/5 = 0.4
        y_pred = np.array([1, 0, 0, 0, 0])
        probas = np.array([0.9, 0.3, 0.1, 0.2, 0.4])
        df = self._dummy_df(5)
        m = _compute_metrics(y_true, y_pred, probas, df)
        assert abs(m["positive_rate_test"] - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# TestBuildTradesDf
# ---------------------------------------------------------------------------


class TestBuildTradesDf:
    def test_empty_when_no_signals(self):
        df = _make_test_df(n_rows=10)
        y_pred = np.zeros(10, dtype=int)
        probas = np.zeros(10, dtype=float)
        trades = _build_trades_df(df, y_pred, probas)
        assert trades.empty

    def test_row_count_equals_signals(self):
        df = _make_test_df(n_rows=50)
        rng = np.random.default_rng(0)
        y_pred = rng.integers(0, 2, 50)
        probas = rng.uniform(0, 1, 50)
        trades = _build_trades_df(df, y_pred, probas)
        assert len(trades) == int(y_pred.sum())

    def test_has_predicted_proba_column(self):
        df = _make_test_df(n_rows=30)
        y_pred = np.ones(30, dtype=int)
        probas = np.ones(30, dtype=float) * 0.9
        trades = _build_trades_df(df, y_pred, probas)
        assert "predicted_proba" in trades.columns

    def test_has_is_true_positive_column(self):
        df = _make_test_df(n_rows=30)
        y_pred = np.ones(30, dtype=int)
        probas = np.ones(30, dtype=float) * 0.9
        trades = _build_trades_df(df, y_pred, probas)
        assert "is_true_positive" in trades.columns

    def test_index_reset(self):
        df = _make_test_df(n_rows=20)
        y_pred = np.ones(20, dtype=int)
        probas = np.ones(20, dtype=float) * 0.8
        trades = _build_trades_df(df, y_pred, probas)
        assert list(trades.index) == list(range(len(trades)))


# ---------------------------------------------------------------------------
# TestModelBacktester — full pipeline
# ---------------------------------------------------------------------------


class TestModelBacktester:
    def _setup(self, tmp_path, n_dates: int = 15, bars: int = 30):
        """Write feature CSVs and a saved model artifact."""
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()

        model = _train_tiny_model(n_features=6)
        cols = _make_feature_cols(6)
        artifact = _make_artifact(model, cols)
        model_path = tmp_path / "xgboost_v1.pkl"
        _save_artifact(artifact, model_path)

        base_ts = 1_740_992_400_000
        rng = np.random.default_rng(0)
        for d in range(n_dates):
            date_str = f"2025-03-{d + 1:02d}"
            rows = []
            for b in range(bars):
                row = {
                    "timestamp": base_ts + d * 86_400_000 + b * 60_000,
                    "date": date_str,
                    "ticker": "O:SPY250303C00580000",
                }
                for i in range(6):
                    row[f"feat_{i}"] = float(rng.standard_normal(1)[0])
                row["target"] = int(rng.random() < 0.30)
                row["max_gain_120m"] = float(rng.uniform(0, 1.5))
                row["time_to_max_min"] = float(rng.uniform(0, 120))
                rows.append(row)
            day_df = pd.DataFrame(rows)
            _write_features_csv(day_df, feat_dir, date_str)

        return model_path, feat_dir

    def test_returns_dict_with_required_keys(self, tmp_path):
        model_path, feat_dir = self._setup(tmp_path)
        config = _make_config()
        backtester = ModelBacktester(config)
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        for key in ("metrics", "model_path", "trades_path", "report_path"):
            assert key in result

    def test_metrics_dict_has_scalar_values(self, tmp_path):
        model_path, feat_dir = self._setup(tmp_path)
        backtester = ModelBacktester(_make_config())
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        m = result["metrics"]
        assert isinstance(m["n_test_rows"], int)
        assert isinstance(m["n_signals"], int)
        assert isinstance(m["precision"], float)

    def test_trades_csv_is_created(self, tmp_path):
        model_path, feat_dir = self._setup(tmp_path)
        backtester = ModelBacktester(_make_config())
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        assert Path(result["trades_path"]).exists()

    def test_report_json_is_created(self, tmp_path):
        model_path, feat_dir = self._setup(tmp_path)
        backtester = ModelBacktester(_make_config())
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        assert Path(result["report_path"]).exists()

    def test_report_json_is_valid(self, tmp_path):
        model_path, feat_dir = self._setup(tmp_path)
        backtester = ModelBacktester(_make_config())
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        with open(result["report_path"]) as fh:
            data = json.load(fh)
        assert "n_test_rows" in data

    def test_raises_if_model_path_missing(self, tmp_path):
        _, feat_dir = self._setup(tmp_path)
        backtester = ModelBacktester(_make_config())
        with pytest.raises(FileNotFoundError, match="model artifact not found"):
            backtester.run(tmp_path / "missing.pkl", feat_dir, output_dir=tmp_path)

    def test_raises_if_no_feature_data(self, tmp_path):
        model = _train_tiny_model()
        artifact = _make_artifact(model, _make_feature_cols())
        model_path = tmp_path / "xgboost_v1.pkl"
        _save_artifact(artifact, model_path)
        empty_dir = tmp_path / "empty_features"
        empty_dir.mkdir()
        backtester = ModelBacktester(_make_config())
        with pytest.raises(ValueError, match="no feature data loaded"):
            backtester.run(model_path, empty_dir, output_dir=tmp_path)

    def test_test_rows_less_than_total_rows(self, tmp_path):
        """Test split must be smaller than the full dataset."""
        model_path, feat_dir = self._setup(tmp_path, n_dates=15)
        backtester = ModelBacktester(_make_config())
        result = backtester.run(model_path, feat_dir, output_dir=tmp_path / "out")
        total_rows = 15 * 30
        assert result["metrics"]["n_test_rows"] < total_rows

    def test_default_config_creates_backtester(self):
        backtester = ModelBacktester({})
        assert "backtest" in backtester.output_dir or "report" in backtester.output_dir
