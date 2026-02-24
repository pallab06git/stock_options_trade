# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for StackedEnsemble."""

import numpy as np
import pandas as pd
import pytest

from src.ml.stacked_ensemble import StackedEnsemble


def _make_feature_df(n=500, n_features=20, seed=42, positive_rate=0.08):
    """Create synthetic feature DataFrame with regime columns."""
    rng = np.random.RandomState(seed)
    data = {}

    # Feature columns
    feat_names = [f"feat_{i}" for i in range(n_features)]
    for name in feat_names:
        data[name] = rng.randn(n).astype(np.float32)

    # Required regime features
    data["spy_vol_std_30m"] = rng.uniform(0.01, 0.08, n)
    data["spy_hl_range_30m"] = rng.uniform(0.1, 1.0, n)
    data["spy_vol_regime"] = rng.uniform(0.5, 2.0, n)
    data["spy_return_30m"] = rng.randn(n) * 0.5
    data["spy_rsi_14"] = rng.uniform(20, 80, n)
    data["spy_bb_pct_b"] = rng.uniform(-0.2, 1.2, n)

    # Target
    data["target"] = (rng.random(n) < positive_rate).astype(int)

    return pd.DataFrame(data), feat_names


class TestStackedEnsembleInit:
    def test_default_init(self):
        ensemble = StackedEnsemble()
        assert not ensemble._is_trained

    def test_custom_config(self):
        config = {"stacked_ensemble": {"n_regimes": 3, "contamination": 0.10}}
        ensemble = StackedEnsemble(config)
        assert ensemble._n_regimes == 3
        assert ensemble._contamination == 0.10


class TestStackedEnsembleTrain:
    def test_train_returns_metrics(self):
        df, feat_cols = _make_feature_df(400)
        train_df = df.iloc[:280]
        val_df = df.iloc[280:]

        ensemble = StackedEnsemble()
        metrics = ensemble.train(train_df, val_df, feat_cols)

        assert "val_precision" in metrics
        assert "val_recall" in metrics
        assert "val_roc_auc" in metrics
        assert "val_f1" in metrics
        assert metrics["n_features"] == len(feat_cols)
        assert metrics["train_rows"] == 280

    def test_train_marks_trained(self):
        df, feat_cols = _make_feature_df(400)
        train_df = df.iloc[:280]
        val_df = df.iloc[280:]

        ensemble = StackedEnsemble()
        ensemble.train(train_df, val_df, feat_cols)
        assert ensemble._is_trained

    def test_base_models_populated(self):
        df, feat_cols = _make_feature_df(400)
        train_df = df.iloc[:280]
        val_df = df.iloc[280:]

        ensemble = StackedEnsemble()
        ensemble.train(train_df, val_df, feat_cols)
        assert len(ensemble._base_models) >= 1  # At least RF
        assert "random_forest" in ensemble._base_models


class TestStackedEnsemblePredict:
    @pytest.fixture
    def trained_ensemble(self):
        df, feat_cols = _make_feature_df(500)
        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        ensemble = StackedEnsemble()
        ensemble.train(train_df, val_df, feat_cols)
        return ensemble, df, feat_cols

    def test_predict_proba_shape(self, trained_ensemble):
        ensemble, df, feat_cols = trained_ensemble
        test_df = df.iloc[400:]
        proba = ensemble.predict_proba(test_df, feat_cols)
        assert len(proba) == len(test_df)

    def test_predict_proba_range(self, trained_ensemble):
        ensemble, df, feat_cols = trained_ensemble
        test_df = df.iloc[400:]
        proba = ensemble.predict_proba(test_df, feat_cols)
        assert (proba >= 0).all()
        assert (proba <= 1.001).all()

    def test_predict_proba_without_anomaly(self, trained_ensemble):
        ensemble, df, feat_cols = trained_ensemble
        test_df = df.iloc[400:]
        proba_with = ensemble.predict_proba(test_df, feat_cols, apply_anomaly_filter=True)
        proba_without = ensemble.predict_proba(test_df, feat_cols, apply_anomaly_filter=False)
        # With anomaly filter, some values should be discounted (≤ without)
        assert (proba_with <= proba_without + 1e-6).all()

    def test_predict_before_train_raises(self):
        ensemble = StackedEnsemble()
        df, feat_cols = _make_feature_df(50)
        with pytest.raises(RuntimeError, match="train"):
            ensemble.predict_proba(df, feat_cols)


class TestPredictWithDetail:
    @pytest.fixture
    def trained_ensemble(self):
        df, feat_cols = _make_feature_df(500)
        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        ensemble = StackedEnsemble()
        ensemble.train(train_df, val_df, feat_cols)
        return ensemble, df, feat_cols

    def test_detail_columns_present(self, trained_ensemble):
        ensemble, df, feat_cols = trained_ensemble
        test_df = df.iloc[400:]
        result = ensemble.predict_with_detail(test_df, feat_cols)
        assert "rf_proba" in result.columns
        assert "market_regime" in result.columns
        assert "regime_confidence" in result.columns
        assert "meta_proba" in result.columns
        assert "calibrated_proba" in result.columns
        assert "is_anomalous" in result.columns
        assert "final_proba" in result.columns

    def test_detail_length_matches(self, trained_ensemble):
        ensemble, df, feat_cols = trained_ensemble
        test_df = df.iloc[400:]
        result = ensemble.predict_with_detail(test_df, feat_cols)
        assert len(result) == len(test_df)


class TestStackedEnsembleSaveLoad:
    @pytest.fixture
    def trained_ensemble(self):
        df, feat_cols = _make_feature_df(500)
        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        ensemble = StackedEnsemble()
        ensemble.train(train_df, val_df, feat_cols)
        return ensemble, df, feat_cols

    def test_save_creates_file(self, trained_ensemble, tmp_path):
        ensemble, _, _ = trained_ensemble
        path = tmp_path / "test_ensemble.pkl"
        ensemble.save(path)
        assert path.exists()

    def test_save_before_train_raises(self, tmp_path):
        ensemble = StackedEnsemble()
        with pytest.raises(RuntimeError, match="untrained"):
            ensemble.save(tmp_path / "bad.pkl")

    def test_load_restores_state(self, trained_ensemble, tmp_path):
        ensemble, df, feat_cols = trained_ensemble
        path = tmp_path / "test_ensemble.pkl"
        ensemble.save(path)

        loaded = StackedEnsemble()
        loaded.load(path)
        assert loaded._is_trained

        test_df = df.iloc[400:]
        proba_orig = ensemble.predict_proba(test_df, feat_cols)
        proba_loaded = loaded.predict_proba(test_df, feat_cols)
        assert np.allclose(proba_orig, proba_loaded, atol=1e-6)

    def test_load_feature_cols_preserved(self, trained_ensemble, tmp_path):
        ensemble, _, feat_cols = trained_ensemble
        path = tmp_path / "test_ensemble.pkl"
        ensemble.save(path)

        loaded = StackedEnsemble()
        loaded.load(path)
        assert loaded._feature_cols == feat_cols


class TestStackedEnsembleEdgeCases:
    def test_all_same_class_train(self):
        """Train when all targets are 0 (or 1)."""
        df, feat_cols = _make_feature_df(400, positive_rate=0.0)
        # Force at least 1 positive to avoid div-by-zero
        df.loc[0, "target"] = 1
        train_df = df.iloc[:280]
        val_df = df.iloc[280:]

        ensemble = StackedEnsemble()
        metrics = ensemble.train(train_df, val_df, feat_cols)
        assert metrics["n_features"] == len(feat_cols)

    def test_small_dataset(self):
        df, feat_cols = _make_feature_df(60, n_features=5, positive_rate=0.15)
        train_df = df.iloc[:40]
        val_df = df.iloc[40:]

        ensemble = StackedEnsemble()
        metrics = ensemble.train(train_df, val_df, feat_cols)
        assert ensemble._is_trained

    def test_nan_features(self):
        df, feat_cols = _make_feature_df(400)
        df.loc[10:20, feat_cols[0]] = np.nan
        train_df = df.iloc[:280]
        val_df = df.iloc[280:]

        ensemble = StackedEnsemble()
        metrics = ensemble.train(train_df, val_df, feat_cols)
        assert ensemble._is_trained
