# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for src/ml/leakage_detector.py."""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.leakage_detector import LeakageDetector, _KNOWN_LOOKAHEAD_FEATURES, _TARGET_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(probas: np.ndarray) -> MagicMock:
    """Return a mock sklearn classifier with fixed predict_proba output."""
    m = MagicMock()
    m.n_features_in_ = 10
    col0 = 1.0 - probas
    m.predict_proba.return_value = np.column_stack([col0, probas])
    return m


def _low_conf_model(n: int = 500) -> MagicMock:
    """Model that always outputs ~0.50 confidence — no leakage expected."""
    return _make_model(np.full(n, 0.50))


def _high_conf_model(n: int = 500) -> MagicMock:
    """Model that outputs 0.90 confidence on every sample — leakage expected."""
    return _make_model(np.full(n, 0.90))


# ---------------------------------------------------------------------------
# TestRandomDataTest
# ---------------------------------------------------------------------------


class TestRandomDataTest:
    def test_no_leakage_low_confidence(self):
        """Model that is always ~50% confident passes the random data test."""
        detector = LeakageDetector()
        model = _low_conf_model(1_000)
        result = detector.test_on_random_data(model, n_samples=1_000)

        assert result["leakage_detected"] is False
        assert result["high_confidence_count"] == 0
        assert result["verdict"] == "PASS"

    def test_leakage_high_confidence(self):
        """Model that outputs 0.90 on every sample fails the random data test."""
        detector = LeakageDetector()
        model = _high_conf_model(1_000)
        result = detector.test_on_random_data(model, n_samples=1_000)

        assert result["leakage_detected"] is True
        assert result["high_confidence_count"] == 1_000
        assert result["verdict"] == "LEAKAGE DETECTED"

    def test_borderline_exactly_at_threshold(self):
        """Exactly max_acceptable_signals high-conf predictions → PASS (not >)."""
        detector = LeakageDetector()
        # 5 high-confidence, 995 low-confidence
        probas = np.concatenate([np.full(5, 0.90), np.full(995, 0.45)])
        model = _make_model(probas)
        result = detector.test_on_random_data(
            model, n_samples=1_000, max_acceptable_signals=5
        )
        # 5 is NOT > 5, so no leakage
        assert result["leakage_detected"] is False

    def test_one_over_threshold_triggers_leakage(self):
        """6 high-confidence signals (> 5 max) → leakage detected."""
        detector = LeakageDetector()
        probas = np.concatenate([np.full(6, 0.90), np.full(994, 0.45)])
        model = _make_model(probas)
        result = detector.test_on_random_data(
            model, n_samples=1_000, max_acceptable_signals=5
        )
        assert result["leakage_detected"] is True

    def test_uses_feature_cols_length_when_provided(self):
        """n_features is taken from feature_cols list, not model attribute."""
        detector = LeakageDetector()
        model = _low_conf_model(50)
        feature_cols = [f"feat_{i}" for i in range(20)]
        result = detector.test_on_random_data(model, feature_cols=feature_cols, n_samples=50)
        assert result["n_features"] == 20
        # predict_proba was called with shape (50, 20)
        call_args = model.predict_proba.call_args[0][0]
        assert call_args.shape == (50, 20)

    def test_predict_proba_error_returns_error_dict(self):
        """If predict_proba raises, result contains 'error' key."""
        detector = LeakageDetector()
        model = MagicMock()
        model.n_features_in_ = 5
        model.predict_proba.side_effect = ValueError("boom")
        result = detector.test_on_random_data(model, n_samples=10)

        assert result["leakage_detected"] is None
        assert "error" in result

    def test_result_stored_in_results_dict(self):
        """test_on_random_data stores result under 'random_data_test' key."""
        detector = LeakageDetector()
        model = _low_conf_model(100)
        detector.test_on_random_data(model, n_samples=100)
        assert "random_data_test" in detector.results


# ---------------------------------------------------------------------------
# TestAuditFeatureDefinitions
# ---------------------------------------------------------------------------


class TestAuditFeatureDefinitions:
    def test_clean_code_no_patterns(self, tmp_path):
        """Code with no lookahead patterns returns empty suspicious list."""
        code = textwrap.dedent("""\
            import pandas as pd
            close = df['close']
            df['ret_1m'] = close.shift(1)
            df['vol'] = close.rolling(5).std()
        """)
        p = tmp_path / "fe.py"
        p.write_text(code)
        detector = LeakageDetector()
        result = detector.audit_feature_definitions(str(p))

        assert result["leakage_likely"] is False
        assert result["pattern_count"] == 0

    def test_negative_shift_flagged(self, tmp_path):
        """Code using .shift(-5) is flagged as HIGH severity."""
        code = "df['future'] = df['close'].shift(-5)"
        p = tmp_path / "fe.py"
        p.write_text(code)
        detector = LeakageDetector()
        result = detector.audit_feature_definitions(str(p))

        assert result["leakage_likely"] is True
        severities = [pt["severity"] for pt in result["suspicious_patterns"]]
        assert "HIGH" in severities

    def test_day_vol_total_sum_flagged(self, tmp_path):
        """opt_vol.sum() used as denominator triggers MEDIUM pattern."""
        code = textwrap.dedent("""\
            day_vol_total = opt_vol.sum()
            df['opt_vol_pct_cumday'] = opt_vol.cumsum() / day_vol_total
        """)
        p = tmp_path / "fe.py"
        p.write_text(code)
        detector = LeakageDetector()
        result = detector.audit_feature_definitions(str(p))

        patterns = result["suspicious_patterns"]
        assert any("day_vol_total" in pt["pattern"] for pt in patterns)

    def test_file_not_found_returns_error(self):
        """Missing file returns error key, leakage_likely False."""
        detector = LeakageDetector()
        result = detector.audit_feature_definitions("/nonexistent/path.py")
        assert "error" in result
        assert result["leakage_likely"] is False


# ---------------------------------------------------------------------------
# TestKnownLookaheadFeatures
# ---------------------------------------------------------------------------


class TestKnownLookaheadFeatures:
    def test_clean_feature_list_passes(self):
        """Feature list without known lookahead features passes."""
        detector = LeakageDetector()
        cols = ["spy_return_1m", "spy_rsi_14", "opt_return_5m", "moneyness"]
        result = detector.check_known_lookahead_features(cols)

        assert result["leakage_detected"] is False
        assert result["lookahead_features"] == []

    def test_opt_vol_pct_cumday_flagged(self):
        """opt_vol_pct_cumday is a known lookahead feature."""
        detector = LeakageDetector()
        cols = ["spy_return_1m", "opt_vol_pct_cumday", "moneyness"]
        result = detector.check_known_lookahead_features(cols)

        assert result["leakage_detected"] is True
        assert any(f["feature"] == "opt_vol_pct_cumday" for f in result["lookahead_features"])

    def test_features_checked_count(self):
        """features_checked reflects the length of the input list."""
        detector = LeakageDetector()
        cols = [f"feat_{i}" for i in range(30)]
        result = detector.check_known_lookahead_features(cols)
        assert result["features_checked"] == 30


# ---------------------------------------------------------------------------
# TestVerifyTargetNotInFeatures
# ---------------------------------------------------------------------------


class TestVerifyTargetNotInFeatures:
    def test_no_target_cols_passes(self):
        """Feature list without any target column passes."""
        detector = LeakageDetector()
        cols = ["spy_return_1m", "moneyness", "opt_rsi_14"]
        result = detector.verify_target_not_in_features(cols)
        assert result["leakage_detected"] is False
        assert result["contaminated_cols"] == []

    def test_target_col_flagged(self):
        """'target' appearing in feature list is flagged."""
        detector = LeakageDetector()
        cols = ["spy_return_1m", "target", "moneyness"]
        result = detector.verify_target_not_in_features(cols)
        assert result["leakage_detected"] is True
        assert "target" in result["contaminated_cols"]

    def test_max_gain_flagged(self):
        """'max_gain_120m' appearing in feature list is flagged."""
        detector = LeakageDetector()
        cols = ["spy_return_1m", "max_gain_120m"]
        result = detector.verify_target_not_in_features(cols)
        assert result["leakage_detected"] is True
        assert "max_gain_120m" in result["contaminated_cols"]


# ---------------------------------------------------------------------------
# TestVerifyTemporalOrdering
# ---------------------------------------------------------------------------


class TestVerifyTemporalOrdering:
    def test_sorted_timestamps_pass(self):
        """Strictly ascending timestamps → no violations."""
        detector = LeakageDetector()
        df = pd.DataFrame({"timestamp": [1000, 2000, 3000, 4000]})
        result = detector.verify_temporal_ordering(df)
        assert result["ordering_valid"] is True
        assert result["violations"] == 0

    def test_out_of_order_timestamps_fail(self):
        """Descending timestamp pair → 1 violation."""
        detector = LeakageDetector()
        df = pd.DataFrame({"timestamp": [1000, 3000, 2000, 4000]})
        result = detector.verify_temporal_ordering(df)
        assert result["ordering_valid"] is False
        assert result["violations"] == 1

    def test_missing_timestamp_column(self):
        """DataFrame without timestamp column returns error."""
        detector = LeakageDetector()
        df = pd.DataFrame({"close": [1, 2, 3]})
        result = detector.verify_temporal_ordering(df)
        assert "error" in result
        assert result["ordering_valid"] is None


# ---------------------------------------------------------------------------
# TestDetectTrainTestContamination
# ---------------------------------------------------------------------------


class TestDetectTrainTestContamination:
    def test_clean_split_no_overlap(self):
        """Non-overlapping date sets with positive gap → no contamination."""
        detector = LeakageDetector()
        train = ["2025-01-02", "2025-01-03", "2025-01-06"]
        test  = ["2025-03-03", "2025-03-04", "2025-03-05"]
        result = detector.detect_train_test_contamination(train, test)

        assert result["contamination_detected"] is False
        assert result["overlap_count"] == 0
        assert result["gap_days"] > 0

    def test_overlap_detected(self):
        """Shared date between train and test → contamination."""
        detector = LeakageDetector()
        train = ["2025-01-02", "2025-01-03", "2025-01-06"]
        test  = ["2025-01-06", "2025-01-07", "2025-01-08"]
        result = detector.detect_train_test_contamination(train, test)

        assert result["contamination_detected"] is True
        assert "2025-01-06" in result["overlap_dates"]

    def test_negative_gap_detected(self):
        """Test set starting before train set ends → contamination."""
        detector = LeakageDetector()
        train = ["2025-01-10", "2025-01-11", "2025-01-12"]
        test  = ["2025-01-09", "2025-01-10"]   # test_min < train_max
        result = detector.detect_train_test_contamination(train, test)
        assert result["contamination_detected"] is True


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_written_to_disk(self, tmp_path):
        """generate_report writes a valid JSON file."""
        detector = LeakageDetector()
        # Populate with a passing random-data result
        detector.results["random_data_test"] = {
            "leakage_detected": False,
            "high_confidence_count": 0,
            "verdict": "PASS",
        }
        out = tmp_path / "report.json"
        report = detector.generate_report(str(out))

        assert out.exists()
        with open(out) as fh:
            loaded = json.load(fh)
        assert loaded["overall_verdict"] == "NO LEAKAGE"
        assert loaded["safe_to_proceed"] is True

    def test_leakage_sets_safe_to_proceed_false(self, tmp_path):
        """Random-data failure sets safe_to_proceed = False."""
        detector = LeakageDetector()
        detector.results["random_data_test"] = {
            "leakage_detected": True,
            "high_confidence_count": 120,
            "max_acceptable_signals": 5,
            "verdict": "LEAKAGE DETECTED",
        }
        out = tmp_path / "report.json"
        report = detector.generate_report(str(out))

        assert report["safe_to_proceed"] is False
        assert report["overall_verdict"] == "LEAKAGE DETECTED"
        assert len(report["critical_issues"]) >= 1

    def test_no_results_produces_clean_report(self, tmp_path):
        """Empty results dict → no leakage, safe to proceed."""
        detector = LeakageDetector()
        out = tmp_path / "report.json"
        report = detector.generate_report(str(out))
        assert report["safe_to_proceed"] is True
        assert report["critical_issues"] == []
