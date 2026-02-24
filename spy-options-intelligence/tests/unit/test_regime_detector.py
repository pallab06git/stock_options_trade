# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for RegimeDetector."""

import numpy as np
import pandas as pd
import pytest

from src.processing.regime_detector import RegimeDetector, _REGIME_FEATURES


def _make_regime_df(n=200):
    """Create synthetic DataFrame with all regime features."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "spy_vol_std_30m": rng.uniform(0.01, 0.08, n),
        "spy_hl_range_30m": rng.uniform(0.1, 1.0, n),
        "spy_vol_regime": rng.uniform(0.5, 2.0, n),
        "spy_return_30m": rng.randn(n) * 0.5,
        "spy_rsi_14": rng.uniform(20, 80, n),
        "spy_bb_pct_b": rng.uniform(-0.2, 1.2, n),
    })


class TestRegimeDetectorInit:
    def test_default_params(self):
        det = RegimeDetector()
        assert det.n_regimes == 4
        assert det.random_state == 42
        assert not det._is_fitted

    def test_custom_params(self):
        det = RegimeDetector(n_regimes=3, random_state=99)
        assert det.n_regimes == 3
        assert det.random_state == 99

    def test_feature_names(self):
        det = RegimeDetector()
        assert det.feature_names == list(_REGIME_FEATURES)


class TestRegimeDetectorFit:
    def test_fit_marks_fitted(self):
        det = RegimeDetector()
        df = _make_regime_df()
        det.fit(df)
        assert det._is_fitted

    def test_fit_returns_self(self):
        det = RegimeDetector()
        df = _make_regime_df()
        result = det.fit(df)
        assert result is det

    def test_fit_with_nan_values(self):
        det = RegimeDetector()
        df = _make_regime_df()
        df.loc[10:15, "spy_rsi_14"] = np.nan
        det.fit(df)
        assert det._is_fitted


class TestRegimeDetectorPredict:
    def test_predict_before_fit_raises(self):
        det = RegimeDetector()
        df = _make_regime_df()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(df)

    def test_predict_adds_two_columns(self):
        det = RegimeDetector()
        df = _make_regime_df()
        det.fit(df)
        result = det.predict(df)
        assert "market_regime" in result.columns
        assert "regime_confidence" in result.columns

    def test_regime_labels_in_range(self):
        det = RegimeDetector(n_regimes=4)
        df = _make_regime_df()
        det.fit(df)
        result = det.predict(df)
        labels = result["market_regime"].unique()
        assert all(0 <= l < 4 for l in labels)

    def test_confidence_between_0_and_1(self):
        det = RegimeDetector()
        df = _make_regime_df()
        det.fit(df)
        result = det.predict(df)
        conf = result["regime_confidence"]
        assert (conf >= 0).all()
        assert (conf <= 1.0001).all()  # small float tolerance

    def test_does_not_modify_input(self):
        det = RegimeDetector()
        df = _make_regime_df()
        det.fit(df)
        orig_cols = set(df.columns)
        det.predict(df)
        assert set(df.columns) == orig_cols

    def test_predict_on_different_data(self):
        det = RegimeDetector()
        train_df = _make_regime_df(200)
        det.fit(train_df)
        test_df = _make_regime_df(50)
        result = det.predict(test_df)
        assert len(result) == 50

    def test_missing_feature_columns(self):
        det = RegimeDetector()
        df = _make_regime_df()
        det.fit(df)
        # Predict on df missing one column
        partial = df.drop(columns=["spy_rsi_14"])
        result = det.predict(partial)
        assert "market_regime" in result.columns


class TestRegimeDetectorGetParams:
    def test_get_params_unfitted(self):
        det = RegimeDetector()
        params = det.get_params()
        assert params["is_fitted"] is False

    def test_get_params_fitted(self):
        det = RegimeDetector()
        det.fit(_make_regime_df())
        params = det.get_params()
        assert params["is_fitted"] is True
        assert params["n_regimes"] == 4
