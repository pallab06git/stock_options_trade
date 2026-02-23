# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for the sustained-movement prediction experiment.

Coverage
--------
  TestSustainedMovementLabeler  (10 tests)
    - basic positive label when price rises and sustains
    - negative label when price rises but doesn't sustain
    - negative label when price falls at confirmation
    - NaN / zero entry price is skipped
    - empty DataFrame returns four empty columns
    - missing columns raises ValueError
    - per-ticker isolation prevents cross-contract leakage
    - magnitude bucket classification (all six buckets)
    - validate() returns correct stats dict
    - config-driven wrapper reads confirmation/sustain params

  TestClassifyMagnitude  (6 tests)
    - below_zero for negative gains
    - 0-1% bucket
    - 1-5% bucket
    - 5-10% bucket
    - 10-20% bucket
    - 20%+ bucket

  TestMultiModelTrainer  (8 tests)
    - raises ValueError when target_col missing
    - raises ValueError when feature cols missing
    - raises ValueError when all-zero target
    - raises ValueError when training portion has only one class
    - returns dict with three model type keys
    - each artifact has required keys
    - model_type key matches the dict key
    - save_artifacts writes .pkl files

  TestSustainedMovementEvaluator  (10 tests)
    - raises ValueError when target_col missing
    - evaluate() returns required top-level keys
    - models key contains one entry per artifact
    - threshold_results has one entry per threshold
    - precision in [0, 1] for every model/threshold
    - generate_report returns DataFrame with one row per model
    - report DataFrame has required columns
    - save_results writes expected files
    - model agreement computed when two models provided
    - precision_by_magnitude returns all six magnitude buckets
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.processing.sustained_movement_labeler import (
    MAGNITUDE_BUCKETS,
    SustainedMovementLabeler,
    classify_magnitude,
    generate_sustained_labels,
)
from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONE_MIN_MS = 60_000  # 1 minute in milliseconds
BASE_TS    = 1_700_000_000_000  # arbitrary base timestamp (ms)


def _make_df(closes, base_ts=BASE_TS, one_min_ms=ONE_MIN_MS, ticker=None):
    """Build a minimal DataFrame from a list of close prices."""
    n = len(closes)
    timestamps = [base_ts + i * one_min_ms for i in range(n)]
    data = {"timestamp": timestamps, "close": closes}
    if ticker is not None:
        data["ticker"] = [ticker] * n
    return pd.DataFrame(data)


def _make_two_ticker_df():
    """Two tickers with independent price series."""
    ts_a = [BASE_TS + i * ONE_MIN_MS for i in range(30)]
    ts_b = [BASE_TS + i * ONE_MIN_MS for i in range(30)]
    closes_a = [1.0] * 30
    closes_b = [2.0] * 30
    # Ticker B: price jumps at bar 15
    closes_b[15:] = [4.0] * 15  # 100% gain
    return pd.DataFrame(
        {
            "timestamp": ts_a + ts_b,
            "close":     closes_a + closes_b,
            "ticker":    ["A"] * 30 + ["B"] * 30,
        }
    )


def _make_minimal_artifact(n_features=5, model=None):
    """Return a minimal artifact dict with a mock model."""
    if model is None:
        model = MagicMock()
        # predict_proba always returns 0.6 for class=1
        model.predict_proba.return_value = np.full((10, 2), 0.5)
        model.predict_proba.return_value[:, 1] = 0.6
    return {
        "model":              model,
        "feature_cols":       [f"feat_{i}" for i in range(n_features)],
        "best_params":        {},
        "optimization_score": 0.5,
        "model_type":         "xgboost",
        "threshold":          0.50,
        "target_col":         "target_sustained",
    }


def _make_test_df_for_evaluator(n=10, n_features=5):
    """Build a test DataFrame for SustainedMovementEvaluator."""
    np.random.seed(42)
    data = {f"feat_{i}": np.random.randn(n) for i in range(n_features)}
    data["target_sustained"] = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0][:n])
    data["magnitude_bucket"] = (["0-1%", "below_zero", "1-5%", "below_zero",
                                  "5-10%", "below_zero", "0-1%", "10-20%",
                                  "below_zero", "below_zero"][:n])
    return pd.DataFrame(data)


# ===========================================================================
# TestSustainedMovementLabeler
# ===========================================================================


class TestSustainedMovementLabeler:
    """Tests for SustainedMovementLabeler and generate_sustained_labels."""

    def test_basic_positive_label(self):
        """Price rises at confirmation AND sustains → target_sustained = 1."""
        # Entry price = 1.0, confirmation bar = bar 15 (price = 2.0)
        # Bars 15..19 all > 1.0 → sustain ≥ 5
        closes = [1.0] * 15 + [2.0] * 15
        df = _make_df(closes)
        labeled = generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=5)
        # Bar 0: conf bar at index 15 (price=2.0 > 1.0), sustain=5+ → positive
        assert labeled.loc[0, "target_sustained"] == 1

    def test_negative_label_no_sustain(self):
        """Price rises at confirmation but only 3 bars sustain → label = 0."""
        # confirmation bar 15: price=2.0, bars 15-17 > 1.0, bar 18 falls
        closes = [1.0] * 15 + [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        df = _make_df(closes)
        labeled = generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=5)
        assert labeled.loc[0, "target_sustained"] == 0

    def test_negative_label_price_falls_at_confirmation(self):
        """Price below entry at confirmation bar → label = 0."""
        closes = [1.0] * 15 + [0.8] * 15
        df = _make_df(closes)
        labeled = generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=5)
        assert labeled.loc[0, "target_sustained"] == 0

    def test_nan_entry_price_skipped(self):
        """Bar with NaN close is skipped — no label error."""
        closes = [float("nan")] + [1.0] * 29
        df = _make_df(closes)
        labeled = generate_sustained_labels(df)
        assert labeled.loc[0, "target_sustained"] == 0

    def test_zero_entry_price_skipped(self):
        """Bar with zero close is skipped — no division-by-zero."""
        closes = [0.0] * 15 + [2.0] * 15
        df = _make_df(closes)
        labeled = generate_sustained_labels(df)
        assert labeled.loc[0, "target_sustained"] == 0

    def test_empty_dataframe(self):
        """Empty DataFrame returns four empty label columns."""
        df = pd.DataFrame({"timestamp": pd.Series(dtype="int64"),
                            "close":     pd.Series(dtype="float64")})
        labeled = generate_sustained_labels(df)
        assert "target_sustained" in labeled.columns
        assert "gain_pct_at_confirmation" in labeled.columns
        assert "magnitude_bucket" in labeled.columns
        assert "sustain_minutes_actual" in labeled.columns
        assert len(labeled) == 0

    def test_missing_columns_raises(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"close": [1.0, 2.0]})  # no timestamp
        with pytest.raises(ValueError, match="timestamp"):
            generate_sustained_labels(df)

    def test_per_ticker_isolation(self):
        """Tickers are labeled independently — B's future bars don't affect A."""
        df = _make_two_ticker_df()
        labeled = generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=5)
        # Ticker A's closes are all 1.0 → no gain → all labels = 0
        a_labels = labeled[labeled["ticker"] == "A"]["target_sustained"].values
        assert all(v == 0 for v in a_labels)

    def test_gain_pct_at_confirmation_computed(self):
        """gain_pct_at_confirmation is populated for rows with a confirmation bar."""
        closes = [1.0] * 15 + [1.5] * 15   # 50% gain at bar 15
        df = _make_df(closes)
        labeled = generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=1)
        gp = labeled.loc[0, "gain_pct_at_confirmation"]
        assert abs(gp - 50.0) < 1e-6

    def test_config_driven_wrapper(self):
        """SustainedMovementLabeler reads confirmation/sustain from config."""
        cfg = {"sustained_movement": {"confirmation_minutes": 5, "sustain_minutes": 3}}
        labeler = SustainedMovementLabeler(cfg)
        assert labeler.confirmation_minutes == 5
        assert labeler.sustain_minutes == 3

    def test_validate_returns_stats_dict(self):
        """SustainedMovementLabeler.validate() returns required keys."""
        closes = [1.0] * 15 + [2.0] * 15
        df = _make_df(closes)
        labeler = SustainedMovementLabeler()
        labeled = labeler.label(df)
        stats = labeler.validate(labeled)
        for key in ("n_total", "n_positive", "positive_rate", "magnitude_breakdown"):
            assert key in stats

    def test_validate_magnitude_breakdown_has_all_buckets(self):
        """validate() includes all six magnitude buckets in the breakdown."""
        closes = [1.0] * 15 + [2.0] * 15
        df = _make_df(closes)
        labeler = SustainedMovementLabeler()
        labeled = labeler.label(df)
        stats = labeler.validate(labeled)
        for bucket in MAGNITUDE_BUCKETS:
            assert bucket in stats["magnitude_breakdown"]


# ===========================================================================
# TestClassifyMagnitude
# ===========================================================================


class TestClassifyMagnitude:
    """Tests for the classify_magnitude helper."""

    def test_below_zero(self):
        assert classify_magnitude(-5.0) == "below_zero"

    def test_zero_to_one(self):
        assert classify_magnitude(0.5) == "0-1%"

    def test_one_to_five(self):
        assert classify_magnitude(3.0) == "1-5%"

    def test_five_to_ten(self):
        assert classify_magnitude(7.0) == "5-10%"

    def test_ten_to_twenty(self):
        assert classify_magnitude(15.0) == "10-20%"

    def test_twenty_plus(self):
        assert classify_magnitude(25.0) == "20%+"

    def test_nan_returns_below_zero(self):
        assert classify_magnitude(float("nan")) == "below_zero"


# ===========================================================================
# TestMultiModelTrainer
# ===========================================================================


class TestMultiModelTrainer:
    """Tests for MultiModelTrainer (mocking heavy ML dependencies)."""

    def _make_training_df(self, n=100, n_features=5, positive_rate=0.3):
        np.random.seed(0)
        data = {f"feat_{i}": np.random.randn(n) for i in range(n_features)}
        # Ensure two classes in both first 80% (train) and last 20% (val)
        target = np.zeros(n, dtype=np.int8)
        train_end = int(n * 0.80)
        # Place positives in train portion
        pos_indices_train = np.random.choice(
            train_end, size=max(2, int(train_end * positive_rate)), replace=False
        )
        target[pos_indices_train] = 1
        # Place at least one positive in val portion
        target[train_end + 1] = 1
        data["target_sustained"] = target
        timestamps = list(range(n))
        data["timestamp"] = timestamps
        return pd.DataFrame(data)

    def test_raises_missing_target_col(self):
        from src.ml.multi_model_trainer import MultiModelTrainer
        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        with pytest.raises(ValueError, match="target_col"):
            trainer.train(df, target_col="nonexistent", feature_cols=["feat_0"])

    def test_raises_missing_feature_col(self):
        from src.ml.multi_model_trainer import MultiModelTrainer
        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        with pytest.raises(ValueError, match="Missing feature"):
            trainer.train(df, target_col="target_sustained", feature_cols=["ghost_col"])

    def test_raises_all_zero_target(self):
        from src.ml.multi_model_trainer import MultiModelTrainer
        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        df["target_sustained"] = 0
        with pytest.raises(ValueError, match="no positive"):
            trainer.train(df, target_col="target_sustained", feature_cols=["feat_0"])

    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_xgboost")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_lightgbm")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_random_forest")
    def test_returns_three_model_keys(self, mock_rf, mock_lgbm, mock_xgb):
        from src.ml.multi_model_trainer import MultiModelTrainer
        mock_xgb.return_value = {"model": MagicMock(), "feature_cols": [], "model_type": "xgboost",
                                  "best_params": {}, "optimization_score": 0.5,
                                  "threshold": 0.5, "target_col": "t", "saved_at": "x",
                                  "val_precision_at_0_70": 0.5}
        mock_lgbm.return_value = {**mock_xgb.return_value, "model_type": "lightgbm"}
        mock_rf.return_value   = {**mock_xgb.return_value, "model_type": "random_forest"}

        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        artifacts = trainer.train(df, "target_sustained", ["feat_0", "feat_1"])
        assert set(artifacts.keys()) == {"xgboost", "lightgbm", "random_forest"}

    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_xgboost")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_lightgbm")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_random_forest")
    def test_each_artifact_has_required_keys(self, mock_rf, mock_lgbm, mock_xgb):
        from src.ml.multi_model_trainer import MultiModelTrainer
        required = {"model", "feature_cols", "best_params",
                    "optimization_score", "model_type", "threshold"}
        base = {k: MagicMock() if k == "model" else ([] if k == "feature_cols" else 0.5)
                for k in required}
        base["target_col"] = "target_sustained"
        base["saved_at"] = "x"
        base["val_precision_at_0_70"] = 0.5
        base["optimization_history"] = []
        mock_xgb.return_value = {**base, "model_type": "xgboost"}
        mock_lgbm.return_value = {**base, "model_type": "lightgbm"}
        mock_rf.return_value   = {**base, "model_type": "random_forest"}

        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        artifacts = trainer.train(df, "target_sustained", ["feat_0"])
        for name, artifact in artifacts.items():
            for key in required:
                assert key in artifact, f"Missing '{key}' in {name} artifact"

    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_xgboost")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_lightgbm")
    @patch("src.ml.multi_model_trainer.MultiModelTrainer._train_random_forest")
    def test_model_type_key_matches(self, mock_rf, mock_lgbm, mock_xgb):
        from src.ml.multi_model_trainer import MultiModelTrainer
        base = {"model": MagicMock(), "feature_cols": [], "best_params": {},
                "optimization_score": 0.5, "threshold": 0.5,
                "target_col": "t", "saved_at": "x",
                "val_precision_at_0_70": 0.5, "optimization_history": []}
        mock_xgb.return_value  = {**base, "model_type": "xgboost"}
        mock_lgbm.return_value = {**base, "model_type": "lightgbm"}
        mock_rf.return_value   = {**base, "model_type": "random_forest"}

        trainer = MultiModelTrainer(n_trials=1, cv_splits=2)
        df = self._make_training_df()
        artifacts = trainer.train(df, "target_sustained", ["feat_0"])
        for name in ("xgboost", "lightgbm", "random_forest"):
            assert artifacts[name]["model_type"] == name

    def test_save_artifacts_writes_pkl(self, tmp_path):
        """save_artifacts() writes one .pkl file per model type."""
        from sklearn.dummy import DummyClassifier
        from src.ml.multi_model_trainer import MultiModelTrainer

        # Use a real picklable sklearn model
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit([[0, 1], [1, 0]], [0, 1])  # minimal fit so it's serialisable

        base = {"model": dummy, "feature_cols": ["feat_0"], "best_params": {},
                "optimization_score": 0.5, "threshold": 0.5,
                "target_col": "t", "saved_at": "x",
                "val_precision_at_0_70": 0.5, "optimization_history": []}
        artifacts = {
            "xgboost":      {**base, "model_type": "xgboost"},
            "lightgbm":     {**base, "model_type": "lightgbm"},
            "random_forest":{**base, "model_type": "random_forest"},
        }

        trainer = MultiModelTrainer()
        saved = trainer.save_artifacts(artifacts, tmp_path)
        for name in ("xgboost", "lightgbm", "random_forest"):
            assert (tmp_path / f"{name}.pkl").exists()

    def test_val_precision_empty_val(self):
        from src.ml.multi_model_trainer import MultiModelTrainer
        trainer = MultiModelTrainer()
        model = MagicMock()
        result = trainer._val_precision(model, np.array([]), np.array([]))
        assert result == 0.0


# ===========================================================================
# TestSustainedMovementEvaluator
# ===========================================================================


class TestSustainedMovementEvaluator:
    """Tests for SustainedMovementEvaluator."""

    def _make_artifacts(self, n=10, n_features=5, proba=0.6):
        """Two dummy artifacts that always predict ``proba`` for class=1."""
        def _make_model(p):
            m = MagicMock()
            probas = np.full((n, 2), 1 - p)
            probas[:, 1] = p
            m.predict_proba.return_value = probas
            return m

        feat_cols = [f"feat_{i}" for i in range(n_features)]
        art1 = {**_make_minimal_artifact(n_features, _make_model(proba)),
                "model_type": "xgboost"}
        art2 = {**_make_minimal_artifact(n_features, _make_model(proba - 0.1)),
                "model_type": "lightgbm"}
        return {"xgboost": art1, "lightgbm": art2}

    def test_raises_missing_target_col(self):
        evaluator = SustainedMovementEvaluator()
        df = pd.DataFrame({"feat_0": [1.0, 2.0]})
        with pytest.raises(ValueError, match="target_col"):
            evaluator.evaluate({}, df)

    def test_evaluate_returns_required_top_level_keys(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        for key in ("models", "model_agreement", "target_col",
                    "n_test_rows", "n_positives", "positive_rate"):
            assert key in results

    def test_models_key_has_one_entry_per_artifact(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        assert set(results["models"].keys()) == set(artifacts.keys())

    def test_threshold_results_one_per_threshold(self):
        thresholds = [0.50, 0.60, 0.70]
        evaluator = SustainedMovementEvaluator(thresholds=thresholds)
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        for name, mdata in results["models"].items():
            assert set(mdata["threshold_results"].keys()) == set(thresholds)

    def test_precision_in_zero_one_range(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50, 0.70])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        for name, mdata in results["models"].items():
            for t, r in mdata["threshold_results"].items():
                assert 0.0 <= r["precision"] <= 1.0

    def test_generate_report_returns_dataframe(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        report = evaluator.generate_report(results, comparison_threshold=0.50)
        assert isinstance(report, pd.DataFrame)

    def test_report_has_one_row_per_model(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        report = evaluator.generate_report(results, comparison_threshold=0.50)
        assert len(report) == len(artifacts)

    def test_report_has_required_columns(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        report = evaluator.generate_report(results, comparison_threshold=0.50)
        required = {"Model", "Threshold", "Signals", "Precision", "Recall", "F1"}
        assert required.issubset(set(report.columns))

    def test_save_results_writes_expected_files(self, tmp_path):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        report = evaluator.generate_report(results)
        saved = evaluator.save_results(results, report, tmp_path)
        assert (tmp_path / "full_results.json").exists()
        assert (tmp_path / "precision_by_magnitude.json").exists()
        assert (tmp_path / "model_agreement.json").exists()

    def test_model_agreement_computed_for_two_models(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        agr = results["model_agreement"]
        assert 0.50 in agr
        assert "n_models_compared" in agr[0.50]

    def test_precision_by_magnitude_has_all_buckets(self):
        evaluator = SustainedMovementEvaluator(thresholds=[0.50])
        df = _make_test_df_for_evaluator()
        artifacts = self._make_artifacts(n=10)
        results = evaluator.evaluate(artifacts, df)
        for name, mdata in results["models"].items():
            pbm = mdata["precision_by_magnitude"]
            by_t = pbm.get("by_threshold", {})
            if by_t:
                for t_data in by_t.values():
                    for bucket in MAGNITUDE_BUCKETS:
                        assert bucket in t_data

    # ------------------------------------------------------------------
    # test_on_random_data
    # ------------------------------------------------------------------

    def test_random_data_no_leakage_for_low_confidence_model(self):
        """Model that always predicts 0.5 should never trigger leakage flag."""
        evaluator = SustainedMovementEvaluator()
        model = MagicMock()
        # Always return proba 0.5 — no high-confidence signals
        model.predict_proba.side_effect = lambda X: np.column_stack([
            np.full(len(X), 0.5), np.full(len(X), 0.5)
        ])
        feature_cols = [f"feat_{i}" for i in range(5)]
        result = evaluator.test_on_random_data(model, feature_cols, n_samples=1000)

        assert result["n_samples"] == 1000
        assert result["n_features"] == 5
        assert result["high_conf_signals"] == 0
        assert result["leakage_suspected"] is False
        assert "OK" in result["verdict"]

    def test_random_data_leakage_flagged_for_high_confidence_model(self):
        """Model that always predicts 0.95 should trigger the leakage flag."""
        evaluator = SustainedMovementEvaluator()
        model = MagicMock()
        # Always return proba 0.95 — all 10 000 rows are high-confidence
        model.predict_proba.side_effect = lambda X: np.column_stack([
            np.full(len(X), 0.05), np.full(len(X), 0.95)
        ])
        feature_cols = [f"feat_{i}" for i in range(3)]
        result = evaluator.test_on_random_data(model, feature_cols, n_samples=10_000)

        assert result["leakage_suspected"] is True
        assert result["high_conf_signals"] == 10_000
        assert "LEAKAGE SUSPECTED" in result["verdict"]

    # ------------------------------------------------------------------
    # print_feature_importance
    # ------------------------------------------------------------------

    def test_feature_importance_returns_dataframe(self):
        """Model with feature_importances_ returns ranked DataFrame."""
        evaluator = SustainedMovementEvaluator()
        model = MagicMock()
        model.feature_importances_ = np.array([0.1, 0.5, 0.2, 0.05, 0.15])
        feature_cols = [f"feat_{i}" for i in range(5)]
        df = evaluator.print_feature_importance(model, feature_cols, top_n=3)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # top_n applied
        assert "Feature" in df.columns
        assert "Importance" in df.columns
        # Top feature should be the one with importance 0.5
        assert df.iloc[0]["Feature"] == "feat_1"

    def test_feature_importance_empty_when_no_attribute(self):
        """Model without feature_importances_ returns empty DataFrame."""
        evaluator = SustainedMovementEvaluator()
        model = MagicMock(spec=[])  # no attributes at all
        result = evaluator.print_feature_importance(model, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty
