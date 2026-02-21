# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for src/ml/evaluate.py — threshold optimisation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.evaluate import find_optimal_threshold_for_precision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(probas: np.ndarray) -> MagicMock:
    """Mock model whose predict_proba always returns fixed probabilities."""
    m = MagicMock()
    m.predict_proba.return_value = np.column_stack([1 - probas, probas])
    return m


def _perfect_model(n: int = 200) -> tuple:
    """Returns (model, X_val, y_val) where model outputs correct class probabilities."""
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    # Perfect model: positives get proba=1.0, negatives get proba=0.0
    probas = np.where(y == 1, 1.0, 0.0)
    return _make_model(probas), np.zeros((n, 5)), y


def _weak_model(n: int = 200) -> tuple:
    """Returns (model, X_val, y_val) where model outputs near-random probabilities."""
    rng = np.random.default_rng(42)
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    # Weak model: probabilities are uniformly distributed — hard to get 90% precision
    probas = rng.uniform(0.40, 0.60, size=n)
    return _make_model(probas), np.zeros((n, 5)), y


# ---------------------------------------------------------------------------
# TestFindOptimalThreshold — achievable cases
# ---------------------------------------------------------------------------


class TestFindOptimalThresholdAchievable:
    def test_perfect_model_achieves_90pct(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.90)
        assert result["achievable"] is True

    def test_returns_correct_keys(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val)
        expected = {
            "achievable", "optimal_threshold", "achieved_precision",
            "achieved_recall", "n_signals", "signal_rate", "analysis_df",
        }
        assert set(result.keys()) == expected

    def test_achieved_precision_meets_requirement(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.90)
        assert result["achieved_precision"] >= 0.90

    def test_optimal_threshold_in_range(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.80)
        assert 0.50 <= result["optimal_threshold"] <= 0.99

    def test_n_signals_positive(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.80)
        assert result["n_signals"] > 0

    def test_signal_rate_consistent_with_n_signals(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.80)
        n = len(y_val)
        assert abs(result["signal_rate"] - result["n_signals"] / n) < 1e-6

    def test_analysis_df_has_correct_columns(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val)
        df = result["analysis_df"]
        for col in ("threshold", "precision", "recall", "n_signals", "signal_rate", "meets_requirement"):
            assert col in df.columns

    def test_analysis_df_covers_threshold_range(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val)
        df = result["analysis_df"]
        assert df["threshold"].min() == pytest.approx(0.50, abs=0.015)
        assert df["threshold"].max() >= 0.95

    def test_lower_min_precision_gives_lower_or_equal_threshold(self):
        """Lower precision requirement → threshold ≤ higher requirement threshold."""
        model, X_val, y_val = _perfect_model(n=400)
        r_high = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.95)
        r_low = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.70)
        if r_high["achievable"] and r_low["achievable"]:
            assert r_low["optimal_threshold"] <= r_high["optimal_threshold"]


# ---------------------------------------------------------------------------
# TestFindOptimalThreshold — not achievable
# ---------------------------------------------------------------------------


class TestFindOptimalThresholdNotAchievable:
    def test_weak_model_cannot_achieve_99pct(self):
        model, X_val, y_val = _weak_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.99)
        assert result["achievable"] is False

    def test_not_achievable_returns_none_for_threshold(self):
        model, X_val, y_val = _weak_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.99)
        assert result["optimal_threshold"] is None
        assert result["achieved_precision"] is None
        assert result["achieved_recall"] is None

    def test_not_achievable_n_signals_is_zero(self):
        model, X_val, y_val = _weak_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.99)
        assert result["n_signals"] == 0

    def test_not_achievable_still_returns_analysis_df(self):
        model, X_val, y_val = _weak_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.99)
        assert isinstance(result["analysis_df"], pd.DataFrame)
        assert len(result["analysis_df"]) > 0


# ---------------------------------------------------------------------------
# TestFindOptimalThreshold — input validation
# ---------------------------------------------------------------------------


class TestFindOptimalThresholdValidation:
    def test_raises_on_empty_X_val(self):
        model = _make_model(np.array([]))
        model.predict_proba.return_value = np.empty((0, 2))
        with pytest.raises(ValueError, match="non-empty"):
            find_optimal_threshold_for_precision(
                model, np.empty((0, 5)), np.array([]), min_precision=0.90
            )

    def test_raises_on_min_precision_above_1(self):
        model, X_val, y_val = _perfect_model()
        with pytest.raises(ValueError, match="min_precision"):
            find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=1.01)

    def test_raises_on_min_precision_zero(self):
        model, X_val, y_val = _perfect_model()
        with pytest.raises(ValueError, match="min_precision"):
            find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.0)

    def test_min_precision_exactly_1_is_valid(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=1.0)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestFindOptimalThreshold — analysis DataFrame integrity
# ---------------------------------------------------------------------------


class TestAnalysisDfIntegrity:
    def test_thresholds_strictly_increasing(self):
        model, X_val, y_val = _perfect_model()
        df = find_optimal_threshold_for_precision(model, X_val, y_val)["analysis_df"]
        assert (df["threshold"].diff().dropna() > 0).all()

    def test_recall_decreases_as_threshold_increases(self):
        """Higher threshold → fewer positives predicted → recall cannot increase."""
        model, X_val, y_val = _perfect_model()
        df = find_optimal_threshold_for_precision(model, X_val, y_val)["analysis_df"]
        # Allow for equal values but never strict increase
        assert (df["recall"].diff().dropna() <= 0).all()

    def test_n_signals_monotonically_decreasing(self):
        model, X_val, y_val = _perfect_model()
        df = find_optimal_threshold_for_precision(model, X_val, y_val)["analysis_df"]
        assert (df["n_signals"].diff().dropna() <= 0).all()

    def test_meets_requirement_column_consistent(self):
        model, X_val, y_val = _perfect_model()
        result = find_optimal_threshold_for_precision(model, X_val, y_val, min_precision=0.80)
        df = result["analysis_df"]
        for _, row in df.iterrows():
            if row["n_signals"] > 0 and row["precision"] == row["precision"]:  # not NaN
                expected = row["precision"] >= 0.80
                assert row["meets_requirement"] == expected
