# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for AnomalyFilter."""

import numpy as np
import pytest

from src.ml.anomaly_filter import AnomalyFilter


def _make_data(n=200, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, n_features).astype(np.float32)


class TestAnomalyFilterInit:
    def test_defaults(self):
        af = AnomalyFilter()
        assert af.contamination == 0.05
        assert af.discount_factor == 0.5

    def test_custom_params(self):
        af = AnomalyFilter(contamination=0.10, discount_factor=0.3)
        assert af.contamination == 0.10
        assert af.discount_factor == 0.3


class TestAnomalyFilterFit:
    def test_fit_marks_fitted(self):
        af = AnomalyFilter()
        X = _make_data()
        af.fit(X)
        assert af._is_fitted

    def test_fit_returns_self(self):
        af = AnomalyFilter()
        result = af.fit(_make_data())
        assert result is af

    def test_fit_handles_nan(self):
        af = AnomalyFilter()
        X = _make_data()
        X[10:15, 3] = np.nan
        af.fit(X)
        assert af._is_fitted


class TestIsAnomalous:
    def test_raises_before_fit(self):
        af = AnomalyFilter()
        with pytest.raises(RuntimeError, match="fit"):
            af.is_anomalous(_make_data(50))

    def test_returns_boolean_mask(self):
        af = AnomalyFilter()
        X = _make_data()
        af.fit(X)
        mask = af.is_anomalous(X)
        assert mask.dtype == bool
        assert len(mask) == len(X)

    def test_contamination_rate_approximate(self):
        af = AnomalyFilter(contamination=0.10)
        X = _make_data(500)
        af.fit(X)
        mask = af.is_anomalous(X)
        # Should be roughly 10% anomalous (allow tolerance)
        rate = mask.mean()
        assert 0.02 < rate < 0.25


class TestApplyDiscount:
    def test_discounts_anomalous_bars(self):
        af = AnomalyFilter(contamination=0.10, discount_factor=0.5)
        X = _make_data(500)
        af.fit(X)

        proba = np.full(500, 0.8)
        discounted = af.apply_discount(proba, X)

        mask = af.is_anomalous(X)
        # Normal bars unchanged
        assert np.allclose(discounted[~mask], 0.8)
        # Anomalous bars discounted
        if mask.any():
            assert np.allclose(discounted[mask], 0.4)

    def test_does_not_modify_input(self):
        af = AnomalyFilter()
        X = _make_data()
        af.fit(X)

        proba = np.full(len(X), 0.9)
        proba_copy = proba.copy()
        af.apply_discount(proba, X)
        assert np.allclose(proba, proba_copy)


class TestAnomalyScores:
    def test_raises_before_fit(self):
        af = AnomalyFilter()
        with pytest.raises(RuntimeError, match="fit"):
            af.anomaly_scores(_make_data(50))

    def test_returns_scores(self):
        af = AnomalyFilter()
        X = _make_data()
        af.fit(X)
        scores = af.anomaly_scores(X)
        assert len(scores) == len(X)
        assert np.isfinite(scores).all()
