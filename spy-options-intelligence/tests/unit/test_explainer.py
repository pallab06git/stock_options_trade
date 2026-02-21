# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for src/ml/explainer.py (SignalExplainer).

Strategy
--------
* Mock ``shap.TreeExplainer`` so tests run without a real fitted model.
* Test every public method: ``from_artifact``, ``explain_signal``,
  ``_interpret_feature``, ``_format_explanation``.
* Cover edge cases: missing features, marginal threshold, all-positive/
  all-negative SHAP, zero-impact features.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "contract_type", "hour_et", "implied_volatility", "is_0dte",
    "is_last_hour", "is_morning", "iv_change_1m", "iv_change_5m",
    "spy_rsi_14", "spy_return_1m", "spy_vol_ratio_5m", "time_to_expiry_days",
]


def _make_mock_model():
    """Return a minimal mock that passes shap.TreeExplainer construction."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.05, 0.95]])
    return model


def _make_shap_values(n_features: int = 12, values: list | None = None):
    """Return a (1, n_features) numpy array of SHAP values."""
    if values is not None:
        return np.array([values], dtype=float)
    # Default: first half positive, second half negative
    sv = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        sv[i] = 0.05 * (1 if i % 2 == 0 else -1) * (i + 1)
    return np.array([sv])


def _make_explainer(shap_return_value=None, feature_names=None):
    """Construct a SignalExplainer with a mocked shap.TreeExplainer."""
    from src.ml.explainer import SignalExplainer

    if feature_names is None:
        feature_names = FEATURE_NAMES

    model = _make_mock_model()
    n = len(feature_names)
    sv = shap_return_value if shap_return_value is not None else _make_shap_values(n)

    with patch("shap.TreeExplainer") as mock_tree:
        instance = MagicMock()
        instance.shap_values.return_value = sv
        mock_tree.return_value = instance
        explainer = SignalExplainer(model, feature_names)

    # Replace the explainer's shap_explainer with the fully configured mock
    explainer.shap_explainer = instance
    return explainer


def _basic_features(feature_names=None):
    """Return a features dict with a sensible value for each feature."""
    fn = feature_names or FEATURE_NAMES
    return {name: float(i + 1) for i, name in enumerate(fn)}


# ---------------------------------------------------------------------------
# TestSignalExplainer — construction
# ---------------------------------------------------------------------------


class TestSignalExplainerConstruction:
    def test_init_stores_model_and_feature_names(self):
        explainer = _make_explainer()
        assert explainer.model is not None
        assert explainer.feature_names == FEATURE_NAMES

    def test_init_creates_shap_explainer_attribute(self):
        explainer = _make_explainer()
        assert explainer.shap_explainer is not None

    def test_from_artifact(self):
        from src.ml.explainer import SignalExplainer

        model = _make_mock_model()
        artifact = {"model": model, "feature_cols": FEATURE_NAMES, "threshold": 0.85}
        sv = _make_shap_values(len(FEATURE_NAMES))

        with patch("shap.TreeExplainer") as mock_tree:
            instance = MagicMock()
            instance.shap_values.return_value = sv
            mock_tree.return_value = instance
            explainer = SignalExplainer.from_artifact(artifact)

        assert explainer.feature_names == FEATURE_NAMES
        assert explainer.model is model

    def test_from_artifact_uses_feature_cols(self):
        from src.ml.explainer import SignalExplainer

        model = _make_mock_model()
        custom_cols = ["spy_close", "spy_volume", "opt_return_1m"]
        artifact = {"model": model, "feature_cols": custom_cols}
        sv = _make_shap_values(3)

        with patch("shap.TreeExplainer") as mock_tree:
            instance = MagicMock()
            instance.shap_values.return_value = sv
            mock_tree.return_value = instance
            explainer = SignalExplainer.from_artifact(artifact)

        assert explainer.feature_names == custom_cols

    def test_missing_shap_raises_import_error(self):
        from src.ml import explainer as mod

        original = mod._SHAP_AVAILABLE
        try:
            mod._SHAP_AVAILABLE = False
            from src.ml.explainer import SignalExplainer
            with pytest.raises(ImportError, match="shap is required"):
                SignalExplainer(_make_mock_model(), FEATURE_NAMES)
        finally:
            mod._SHAP_AVAILABLE = original


# ---------------------------------------------------------------------------
# TestExplainSignal
# ---------------------------------------------------------------------------


class TestExplainSignal:
    def test_returns_string(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert isinstance(result, str)

    def test_contains_confidence(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "94.0%" in result

    def test_contains_threshold(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(
            _basic_features(), prediction_proba=0.94, threshold=0.90
        )
        assert "90.0%" in result

    def test_contains_signal_detected(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "SIGNAL DETECTED" in result

    def test_contains_top_contributing_factors(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "TOP CONTRIBUTING FACTORS" in result

    def test_shows_at_most_10_features(self):
        # Use 15 features; only top 10 should appear
        fnames = [f"feat_{i}" for i in range(15)]
        sv = _make_shap_values(15)
        explainer = _make_explainer(shap_return_value=sv, feature_names=fnames)
        features = {f"feat_{i}": float(i) for i in range(15)}
        result = explainer.explain_signal(features, prediction_proba=0.94)
        # At most entries 1–10
        assert "10." in result
        assert "11." not in result

    def test_margin_above_threshold_shown(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(
            _basic_features(), prediction_proba=0.96, threshold=0.90
        )
        assert "+6.0%" in result

    def test_close_call_warning_near_threshold(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(
            _basic_features(), prediction_proba=0.905, threshold=0.90
        )
        assert "CLOSE CALL" in result or "CAUTION" in result

    def test_no_close_call_warning_well_above_threshold(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(
            _basic_features(), prediction_proba=0.97, threshold=0.90
        )
        assert "CLOSE CALL" not in result

    def test_risk_factors_shown_when_negative_shap(self):
        # Force top-10 to have mix of positive and negative
        sv = np.array([[-0.3, 0.2, -0.25, 0.1, 0.15, -0.05, 0.08, 0.09, 0.07, -0.06, 0.04, 0.03]])
        explainer = _make_explainer(shap_return_value=sv)
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "RISK FACTORS" in result

    def test_risk_factors_not_shown_when_all_positive(self):
        # All positive SHAP
        n = len(FEATURE_NAMES)
        sv = np.array([[0.1 * (i + 1) for i in range(n)]])
        explainer = _make_explainer(shap_return_value=sv)
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "RISK FACTORS" not in result

    def test_missing_feature_defaults_to_zero(self):
        explainer = _make_explainer()
        # Pass only partial features dict
        partial = {"hour_et": 10.0}
        result = explainer.explain_signal(partial, prediction_proba=0.94)
        # Should not raise; result is still a string
        assert isinstance(result, str)

    def test_shap_explainer_called_with_feature_array(self):
        explainer = _make_explainer()
        features = _basic_features()
        explainer.explain_signal(features, prediction_proba=0.92)
        assert explainer.shap_explainer.shap_values.called

    def test_caution_header_when_marginal(self):
        explainer = _make_explainer()
        result = explainer.explain_signal(
            _basic_features(), prediction_proba=0.901, threshold=0.90
        )
        assert "CAUTION" in result

    def test_recommendation_shown_with_multiple_risk_factors(self):
        # Force ≥2 negative top impacts
        sv = np.array([[-0.4, -0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]])
        explainer = _make_explainer(shap_return_value=sv)
        result = explainer.explain_signal(_basic_features(), prediction_proba=0.94)
        assert "RECOMMENDATION" in result


# ---------------------------------------------------------------------------
# TestInterpretFeature
# ---------------------------------------------------------------------------


class TestInterpretFeature:
    """Test _interpret_feature for key feature groups."""

    def setup_method(self):
        self.explainer = _make_explainer()

    # ── Option returns ──────────────────────────────────────────────────

    def test_opt_return_1m_sharp_move(self):
        result = self.explainer._interpret_feature("opt_return_1m", 0.15, 0.1)
        assert "15" in result or "sharp" in result.lower()

    def test_opt_return_5m_declining(self):
        result = self.explainer._interpret_feature("opt_return_5m", -0.08, -0.05)
        assert "declin" in result.lower() or "-8" in result

    # ── SPY technicals ──────────────────────────────────────────────────

    def test_spy_rsi_overbought(self):
        result = self.explainer._interpret_feature("spy_rsi_14", 75.0, 0.05)
        assert "overbought" in result.lower()

    def test_spy_rsi_oversold(self):
        result = self.explainer._interpret_feature("spy_rsi_14", 25.0, -0.03)
        assert "oversold" in result.lower()

    def test_spy_rsi_neutral(self):
        result = self.explainer._interpret_feature("spy_rsi_14", 50.0, 0.01)
        assert "neutral" in result.lower()

    def test_spy_macd_hist_bullish(self):
        result = self.explainer._interpret_feature("spy_macd_hist", 0.05, 0.1)
        assert "bullish" in result.lower()

    def test_spy_macd_hist_bearish(self):
        result = self.explainer._interpret_feature("spy_macd_hist", -0.03, -0.08)
        assert "bearish" in result.lower()

    def test_spy_ema_diff_bullish_trend(self):
        result = self.explainer._interpret_feature("spy_ema_diff", 0.5, 0.05)
        assert "bullish" in result.lower()

    # ── Volume features ─────────────────────────────────────────────────

    def test_spy_vol_ratio_spike(self):
        result = self.explainer._interpret_feature("spy_vol_ratio_5m", 3.5, 0.1)
        assert "surge" in result.lower() or "3.5" in result

    def test_spy_vol_regime_high(self):
        result = self.explainer._interpret_feature("spy_vol_regime", 2.0, 0.06)
        assert "high" in result.lower() or "institutional" in result.lower()

    def test_spy_vol_zscore_abnormal(self):
        result = self.explainer._interpret_feature("spy_vol_zscore", 3.2, 0.07)
        assert "abnormal" in result.lower() or "3.2" in result

    # ── IV features ─────────────────────────────────────────────────────

    def test_iv_high(self):
        result = self.explainer._interpret_feature("implied_volatility", 0.50, 0.08)
        assert "elevated" in result.lower() or "expensive" in result.lower()

    def test_iv_low(self):
        result = self.explainer._interpret_feature("implied_volatility", 0.10, -0.03)
        assert "compres" in result.lower() or "cheap" in result.lower()

    def test_iv_change_1m_rising(self):
        result = self.explainer._interpret_feature("iv_change_1m", 0.02, 0.04)
        assert "rising" in result.lower() or "demand" in result.lower()

    # ── Time features ────────────────────────────────────────────────────

    def test_hour_et_prime_window(self):
        result = self.explainer._interpret_feature("hour_et", 10.0, 0.15)
        assert "prime" in result.lower() or "10" in result

    def test_hour_et_power_hour(self):
        result = self.explainer._interpret_feature("hour_et", 15.0, 0.08)
        assert "power hour" in result.lower() or "3-4" in result

    def test_is_morning_true(self):
        result = self.explainer._interpret_feature("is_morning", 1.0, 0.05)
        assert "morning" in result.lower() or "volatility" in result.lower()

    def test_is_last_hour_true(self):
        result = self.explainer._interpret_feature("is_last_hour", 1.0, 0.04)
        assert "last" in result.lower() or "3-4" in result

    # ── Moneyness ────────────────────────────────────────────────────────

    def test_moneyness_atm(self):
        result = self.explainer._interpret_feature("moneyness", 1.000, 0.04)
        assert "ATM" in result

    def test_moneyness_itm(self):
        result = self.explainer._interpret_feature("moneyness", 0.98, 0.05)
        assert "ITM" in result

    def test_moneyness_otm(self):
        result = self.explainer._interpret_feature("moneyness", 1.02, 0.03)
        assert "OTM" in result

    # ── Contract type ────────────────────────────────────────────────────

    def test_contract_type_call(self):
        result = self.explainer._interpret_feature("contract_type", 1.0, 0.05)
        assert "Call" in result

    def test_contract_type_put(self):
        result = self.explainer._interpret_feature("contract_type", 0.0, -0.03)
        assert "Put" in result

    # ── 0DTE ─────────────────────────────────────────────────────────────

    def test_is_0dte_true(self):
        result = self.explainer._interpret_feature("is_0dte", 1.0, 0.1)
        assert "0DTE" in result or "same-day" in result.lower()

    # ── Fallback ─────────────────────────────────────────────────────────

    def test_unknown_feature_fallback(self):
        result = self.explainer._interpret_feature("some_unknown_feature_xyz", 3.14, 0.02)
        assert "some_unknown_feature_xyz" in result
        assert "3.14" in result

    def test_all_known_features_return_strings(self):
        """Every feature in the model's feature list must return a non-empty string."""
        import joblib
        art = joblib.load("models/xgboost_v2.pkl")
        for col in art["feature_cols"]:
            result = self.explainer._interpret_feature(col, 1.0, 0.05)
            assert isinstance(result, str), f"_interpret_feature returned non-str for '{col}'"
            assert len(result) > 0, f"_interpret_feature returned empty string for '{col}'"


# ---------------------------------------------------------------------------
# TestFormatExplanation
# ---------------------------------------------------------------------------


class TestFormatExplanation:
    def setup_method(self):
        self.explainer = _make_explainer()
        self.top_impacts = [
            {
                "feature": "opt_return_1m",
                "value": 0.12,
                "impact": 0.35,
                "interpretation": "Option return last 1 min: +12.00%",
            },
            {
                "feature": "spy_rsi_14",
                "value": 72.0,
                "impact": 0.22,
                "interpretation": "SPY RSI-14: 72 — overbought",
            },
            {
                "feature": "spy_vol_ratio_5m",
                "value": 3.1,
                "impact": -0.10,
                "interpretation": "SPY volume 3.1x its 5-min MA — volume surge",
            },
        ]

    def test_returns_string(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert isinstance(result, str)

    def test_contains_signal_detected(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "SIGNAL DETECTED" in result

    def test_shows_confidence(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "94.0%" in result

    def test_shows_threshold(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "90.0%" in result

    def test_shows_margin(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "+4.0%" in result

    def test_feature_names_present(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "opt_return_1m" in result
        assert "spy_rsi_14" in result

    def test_risk_factors_section_shown(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "RISK FACTORS" in result

    def test_no_risk_factors_when_all_positive(self):
        pos_impacts = [x.copy() for x in self.top_impacts]
        for item in pos_impacts:
            item["impact"] = abs(item["impact"])
        result = self.explainer._format_explanation(0.94, 0.90, pos_impacts, {})
        assert "RISK FACTORS" not in result

    def test_caution_header_marginal(self):
        result = self.explainer._format_explanation(0.901, 0.90, self.top_impacts, {})
        assert "CAUTION" in result

    def test_no_caution_header_comfortably_above(self):
        result = self.explainer._format_explanation(0.96, 0.90, self.top_impacts, {})
        assert "CAUTION" not in result or "Near Threshold" not in result

    def test_empty_impacts_list(self):
        result = self.explainer._format_explanation(0.94, 0.90, [], {})
        assert "SIGNAL DETECTED" in result
        assert isinstance(result, str)

    def test_bullish_bearish_labels(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "BULLISH" in result
        assert "BEARISH" in result

    def test_red_dot_indicators_present(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "🔴" in result

    def test_interpretation_text_included(self):
        result = self.explainer._format_explanation(0.94, 0.90, self.top_impacts, {})
        assert "Option return last 1 min" in result
        assert "overbought" in result
