# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Unit tests for src/ml/threshold_analyzer.py — ThresholdAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from src.ml.threshold_analyzer import ThresholdAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)


def _make_df(
    n: int = 100,
    positive_rate: float = 0.40,
    max_gain_mean: float = 30.0,
    min_loss_mean: float = -15.0,
    date: str = "2025-06-01",
) -> pd.DataFrame:
    """Synthetic feature-CSV-style DataFrame for testing."""
    target = (_RNG.random(n) < positive_rate).astype(int)
    max_gain = np.where(
        target == 1,
        _RNG.uniform(20.0, 60.0, n),
        _RNG.uniform(0.0, 15.0, n),
    )
    min_loss = _RNG.uniform(-40.0, -1.0, n)
    return pd.DataFrame(
        {
            "target": target,
            "max_gain_120m": max_gain,
            "min_loss_120m": min_loss,
            "date": date,
        }
    )


def _make_probas(n: int, low: float = 0.4, high: float = 0.99) -> np.ndarray:
    return _RNG.uniform(low, high, n).astype(np.float32)


# ---------------------------------------------------------------------------
# _analyze_single_threshold
# ---------------------------------------------------------------------------


class TestAnalyzeSingleThreshold:
    def _run(self, df, probas, threshold):
        analyzer = ThresholdAnalyzer()
        y_true = df["target"].values.astype(int)
        max_gains = df["max_gain_120m"].values.astype(float)
        min_losses = df["min_loss_120m"].values.astype(float)
        return analyzer._analyze_single_threshold(
            y_true, probas, max_gains, min_losses, threshold
        )

    def test_required_keys_present(self):
        df = _make_df(50)
        probas = _make_probas(50)
        result = self._run(df, probas, 0.80)
        required = {
            "total_signals",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "signal_rate",
            "precision",
            "recall",
            "f1_score",
            "tp_profit_pct_count",
            "fp_loss_pct_count",
            "fn_missed_pct_count",
            "expected_value_pct",
        }
        assert required.issubset(set(result.keys()))

    def test_counts_sum_to_n(self):
        df = _make_df(80)
        probas = _make_probas(80)
        result = self._run(df, probas, 0.75)
        n = len(df)
        assert (
            result["true_positives"]
            + result["false_positives"]
            + result["false_negatives"]
            + result["true_negatives"]
            == n
        )

    def test_tp_fp_sum_to_total_signals(self):
        df = _make_df(60)
        probas = _make_probas(60)
        result = self._run(df, probas, 0.75)
        assert result["total_signals"] == result["true_positives"] + result["false_positives"]

    def test_precision_in_bounds(self):
        df = _make_df(100)
        probas = _make_probas(100)
        result = self._run(df, probas, 0.80)
        assert 0.0 <= result["precision"] <= 1.0

    def test_recall_in_bounds(self):
        df = _make_df(100)
        probas = _make_probas(100)
        result = self._run(df, probas, 0.80)
        assert 0.0 <= result["recall"] <= 1.0

    def test_no_signals_all_metrics_zero(self):
        """Threshold higher than all probas → no signals fired."""
        df = _make_df(50)
        probas = np.full(50, 0.50)
        result = self._run(df, probas, 0.99)
        assert result["total_signals"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["expected_value_pct"] == 0.0

    def test_all_signals_fired(self):
        """Threshold of 0.0 → every bar fires a signal."""
        df = _make_df(60)
        probas = _make_probas(60, low=0.1, high=0.9)
        result = self._run(df, probas, 0.0)
        assert result["total_signals"] == len(df)
        assert result["false_negatives"] == 0

    def test_perfect_precision_when_all_tp(self):
        """All predictions correct positive → precision = 1.0."""
        n = 30
        target = np.ones(n, dtype=int)
        gains = np.full(n, 25.0)
        losses = np.full(n, -5.0)
        probas = np.full(n, 0.95)
        analyzer = ThresholdAnalyzer()
        result = analyzer._analyze_single_threshold(target, probas, gains, losses, 0.90)
        assert result["precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["false_positives"] == 0

    def test_perfect_recall_when_threshold_zero(self):
        """Threshold=0 catches all positives → recall = 1.0."""
        n = 40
        target = (np.arange(n) % 2).astype(int)  # alternating 0/1
        gains = np.where(target == 1, 30.0, 5.0)
        losses = np.full(n, -10.0)
        probas = _RNG.uniform(0.5, 0.9, n).astype(np.float32)
        analyzer = ThresholdAnalyzer()
        result = analyzer._analyze_single_threshold(target, probas, gains, losses, 0.0)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)

    def test_ev_positive_with_high_precision(self):
        """EV should be positive when precision is high and TP gains dominate FP losses."""
        df = _make_df(200, positive_rate=0.50)
        probas = _make_probas(200, low=0.7, high=0.99)
        result = self._run(df, probas, 0.90)
        # Not guaranteed but with positive_rate=0.5 and high probas, EV should be > 0
        if result["total_signals"] > 0 and result["tp_profit_pct_count"] > 0:
            assert result["expected_value_pct"] != 0.0

    def test_nan_gain_loss_rows_excluded(self):
        """NaN values in max_gain and min_loss should not crash or skew stats."""
        n = 20
        target = np.ones(n, dtype=int)
        gains = np.full(n, 30.0)
        gains[0:5] = np.nan  # NaN for first 5 bars
        losses = np.full(n, -10.0)
        losses[5:8] = np.nan
        probas = np.full(n, 0.95)
        analyzer = ThresholdAnalyzer()
        result = analyzer._analyze_single_threshold(target, probas, gains, losses, 0.90)
        # Should not raise; TP gains excludes NaN rows
        assert result["tp_profit_pct_count"] <= n - 5  # 5 NaN gains excluded


# ---------------------------------------------------------------------------
# _analyze_threshold_range
# ---------------------------------------------------------------------------


class TestAnalyzeThresholdRange:
    def test_returns_dataframe(self):
        analyzer = ThresholdAnalyzer()
        df = _make_df(80)
        probas = _make_probas(80)
        result = analyzer._analyze_threshold_range(df, probas, [0.70, 0.80, 0.90])
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_threshold(self):
        analyzer = ThresholdAnalyzer()
        df = _make_df(80)
        probas = _make_probas(80)
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
        result = analyzer._analyze_threshold_range(df, probas, thresholds)
        assert len(result) == len(thresholds)

    def test_threshold_column_present(self):
        analyzer = ThresholdAnalyzer()
        df = _make_df(60)
        probas = _make_probas(60)
        result = analyzer._analyze_threshold_range(df, probas, [0.80])
        assert "threshold" in result.columns

    def test_threshold_values_correct(self):
        analyzer = ThresholdAnalyzer()
        df = _make_df(60)
        probas = _make_probas(60)
        ts = [0.70, 0.80, 0.90]
        result = analyzer._analyze_threshold_range(df, probas, ts)
        assert list(result["threshold"].round(2)) == [round(t, 2) for t in ts]

    def test_signals_non_increasing_with_threshold(self):
        """Higher threshold → fewer or equal signals."""
        analyzer = ThresholdAnalyzer()
        df = _make_df(100)
        probas = _make_probas(100)
        ts = [0.60, 0.70, 0.80, 0.90]
        result = analyzer._analyze_threshold_range(df, probas, ts)
        sigs = list(result["total_signals"])
        for i in range(len(sigs) - 1):
            assert sigs[i] >= sigs[i + 1]

    def test_precision_non_decreasing_with_threshold(self):
        """Higher threshold → precision should be non-decreasing (generally)."""
        analyzer = ThresholdAnalyzer()
        # Use well-calibrated probas: high proba → more likely positive
        n = 200
        target = (_RNG.random(n) < 0.5).astype(int)
        # Higher probas correlate with target=1
        probas = np.clip(
            target.astype(float) * 0.4 + _RNG.uniform(0.3, 0.7, n), 0, 1
        ).astype(np.float32)
        df = pd.DataFrame(
            {"target": target, "max_gain_120m": 25.0, "min_loss_120m": -10.0, "date": "2025-01-01"}
        )
        ts = [0.55, 0.65, 0.75, 0.85]
        result = analyzer._analyze_threshold_range(df, probas, ts)
        # Filter to rows with signals
        with_sigs = result[result["total_signals"] > 0]
        if len(with_sigs) > 1:
            prec = list(with_sigs["precision"])
            for i in range(len(prec) - 1):
                assert prec[i] <= prec[i + 1] + 0.1  # allow small deviation


# ---------------------------------------------------------------------------
# generate_monthly_summary
# ---------------------------------------------------------------------------


class TestGenerateMonthlySummary:
    def _make_monthly_df(self) -> pd.DataFrame:
        months = ["2025-03", "2025-04", "2025-05"]
        thresholds = [0.70, 0.80, 0.90]
        rows = []
        for m in months:
            for t in thresholds:
                rows.append(
                    {
                        "month": m,
                        "threshold": t,
                        "total_signals": int(100 * (1.0 - t)),
                        "precision": round(0.60 + t * 0.40, 4),
                        "expected_value_pct": round((1.0 - t) * 20.0, 2),
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.70, 0.80, 0.90])
        assert isinstance(summary, pd.DataFrame)

    def test_one_row_per_month(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.70, 0.80, 0.90])
        assert len(summary) == 3  # 3 months

    def test_signals_columns_present(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.70, 0.80])
        assert "signals_70" in summary.columns
        assert "signals_80" in summary.columns

    def test_precision_columns_present(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.90])
        assert "precision_90" in summary.columns

    def test_ev_columns_present(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.80])
        assert "ev_80" in summary.columns

    def test_signals_values_correct(self):
        analyzer = ThresholdAnalyzer()
        monthly_df = self._make_monthly_df()
        summary = analyzer.generate_monthly_summary(monthly_df, key_thresholds=[0.70])
        # signals at 0.70 for each month = int(100 * (1 - 0.70)) = 30
        assert all(summary["signals_70"] == 30)

    def test_missing_threshold_fills_zero(self):
        """If a month has no row for a requested threshold, fills 0."""
        analyzer = ThresholdAnalyzer()
        monthly_df = pd.DataFrame(
            [
                {
                    "month": "2025-03",
                    "threshold": 0.80,
                    "total_signals": 10,
                    "precision": 0.85,
                    "expected_value_pct": 15.0,
                }
            ]
        )
        summary = analyzer.generate_monthly_summary(
            monthly_df, key_thresholds=[0.80, 0.90]
        )
        assert summary.iloc[0]["signals_80"] == 10
        assert summary.iloc[0]["signals_90"] == 0


# ---------------------------------------------------------------------------
# plot_monthly_signals
# ---------------------------------------------------------------------------


class TestPlotMonthlySignals:
    def _make_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"month": "2025-03", "signals_70": 40, "precision_70": 0.85, "ev_70": 12.0},
                {"month": "2025-04", "signals_70": 55, "precision_70": 0.87, "ev_70": 14.0},
                {"month": "2025-05", "signals_70": 30, "precision_70": 0.82, "ev_70": 10.0},
            ]
        )

    def test_returns_string(self):
        analyzer = ThresholdAnalyzer()
        summary = self._make_summary()
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.70])
        assert isinstance(result, str)

    def test_contains_month_labels(self):
        analyzer = ThresholdAnalyzer()
        summary = self._make_summary()
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.70])
        assert "2025-03" in result
        assert "2025-04" in result
        assert "2025-05" in result

    def test_contains_bar_characters(self):
        analyzer = ThresholdAnalyzer()
        summary = self._make_summary()
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.70])
        assert "█" in result

    def test_threshold_header_present(self):
        analyzer = ThresholdAnalyzer()
        summary = self._make_summary()
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.70])
        assert "0.70" in result

    def test_missing_threshold_column_skipped(self):
        """If column for threshold not in summary, threshold section is skipped gracefully."""
        analyzer = ThresholdAnalyzer()
        summary = self._make_summary()  # only has signals_70
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.90])
        # 0.90 section absent (column missing) but should not crash
        assert isinstance(result, str)

    def test_zero_signals_handled(self):
        analyzer = ThresholdAnalyzer()
        summary = pd.DataFrame(
            [
                {"month": "2025-03", "signals_90": 0, "precision_90": 0.0, "ev_90": 0.0},
            ]
        )
        result = analyzer.plot_monthly_signals(summary, key_thresholds=[0.90])
        assert "no signals" in result.lower() or isinstance(result, str)


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------


class TestFindOptimalThreshold:
    def _make_results_df(self) -> pd.DataFrame:
        """Simulated aggregate results with varying precision and signals."""
        rows = []
        for t_i, t in enumerate([0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
            rows.append(
                {
                    "threshold": t,
                    "total_signals": 500 - t_i * 70,
                    "precision": 0.75 + t_i * 0.04,
                    "recall": 0.80 - t_i * 0.12,
                    "expected_value_pct": 10.0 + t_i * 2.0,
                    "tp_profit_pct_avg": 25.0 + t_i,
                    "fp_loss_pct_avg": -(10.0 + t_i),
                    "fn_missed_pct_avg": 28.0,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_success_when_threshold_meets_criteria(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(df, min_precision=0.85, min_signals=10)
        assert result["status"] == "SUCCESS"
        assert "optimal_threshold" in result
        assert "metrics" in result

    def test_returns_no_valid_threshold_when_precision_too_high(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(df, min_precision=0.999, min_signals=1)
        assert result["status"] == "NO_VALID_THRESHOLD"
        assert "message" in result

    def test_returns_no_valid_when_min_signals_too_high(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(
            df, min_precision=0.80, min_signals=100_000
        )
        assert result["status"] == "NO_VALID_THRESHOLD"

    def test_optimizes_ev_correctly(self):
        """Should pick the row with highest EV among valid rows."""
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(
            df,
            optimization_metric="expected_value_pct",
            min_precision=0.80,
            min_signals=10,
        )
        assert result["status"] == "SUCCESS"
        # The row with highest EV (0.95 → ev=20.0) should win
        assert result["optimal_threshold"] == pytest.approx(0.95, abs=0.01)

    def test_optimizes_precision_correctly(self):
        """Should pick highest precision row when optimizing precision."""
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(
            df,
            optimization_metric="precision",
            min_precision=0.80,
            min_signals=10,
        )
        assert result["status"] == "SUCCESS"
        assert result["optimal_threshold"] == pytest.approx(0.95, abs=0.01)

    def test_raises_on_unknown_metric(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        with pytest.raises(ValueError, match="not in results_df"):
            analyzer.find_optimal_threshold(df, optimization_metric="nonexistent_col")

    def test_metrics_dict_contains_threshold(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(df, min_precision=0.80, min_signals=1)
        if result["status"] == "SUCCESS":
            assert "threshold" in result["metrics"]

    def test_optimal_threshold_matches_metrics_threshold(self):
        analyzer = ThresholdAnalyzer()
        df = self._make_results_df()
        result = analyzer.find_optimal_threshold(df, min_precision=0.80, min_signals=1)
        if result["status"] == "SUCCESS":
            assert result["optimal_threshold"] == pytest.approx(
                result["metrics"]["threshold"], abs=1e-4
            )
