# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Unit tests for src/ml/error_analyzer.py — PredictionErrorAnalyzer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.error_analyzer import PredictionErrorAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades_df(
    n_fp: int = 10,
    n_tp: int = 5,
    fp_min_losses: list | None = None,
    tp_min_losses: list | None = None,
) -> pd.DataFrame:
    """Build a minimal trades DataFrame with required columns."""
    if fp_min_losses is None:
        fp_min_losses = list(np.linspace(-30.0, 0.0, n_fp))
    if tp_min_losses is None:
        tp_min_losses = [5.0] * n_tp

    rows = []
    for loss in fp_min_losses:
        rows.append(
            {
                "is_true_positive": False,
                "min_loss_120m": loss,
                "max_gain_120m": max(0.0, loss + 5.0),
                "predicted_proba": 0.6,
            }
        )
    for gain in tp_min_losses:
        rows.append(
            {
                "is_true_positive": True,
                "min_loss_120m": -2.0,
                "max_gain_120m": gain,
                "predicted_proba": 0.85,
            }
        )
    return pd.DataFrame(rows)


def _write_trades_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "trades.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_false_positives
# ---------------------------------------------------------------------------


class TestLoadFalsePositives:
    def test_returns_only_fp_rows(self, tmp_path):
        df = _make_trades_df(n_fp=5, n_tp=10)
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        fp_df = analyzer.load_false_positives(path)
        assert len(fp_df) == 5
        assert fp_df["is_true_positive"].all() == False  # noqa: E712

    def test_empty_fp_when_all_tp(self, tmp_path):
        df = _make_trades_df(n_fp=0, n_tp=8, fp_min_losses=[])
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        fp_df = analyzer.load_false_positives(path)
        assert fp_df.empty

    def test_raises_file_not_found(self):
        analyzer = PredictionErrorAnalyzer()
        with pytest.raises(FileNotFoundError, match="file not found"):
            analyzer.load_false_positives("/nonexistent/path/trades.csv")

    def test_raises_missing_columns(self, tmp_path):
        df = pd.DataFrame({"is_true_positive": [False], "max_gain_120m": [5.0]})
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        with pytest.raises(ValueError, match="missing columns"):
            analyzer.load_false_positives(path)

    def test_index_reset_after_filter(self, tmp_path):
        df = _make_trades_df(n_fp=3, n_tp=7)
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        fp_df = analyzer.load_false_positives(path)
        assert list(fp_df.index) == list(range(len(fp_df)))

    def test_accepts_path_object(self, tmp_path):
        df = _make_trades_df(n_fp=4, n_tp=2)
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        fp_df = analyzer.load_false_positives(Path(path))
        assert len(fp_df) == 4

    def test_accepts_string_path(self, tmp_path):
        df = _make_trades_df(n_fp=4, n_tp=2)
        path = _write_trades_csv(tmp_path, df)
        analyzer = PredictionErrorAnalyzer()
        fp_df = analyzer.load_false_positives(str(path))
        assert len(fp_df) == 4


# ---------------------------------------------------------------------------
# generate_risk_report
# ---------------------------------------------------------------------------


class TestGenerateRiskReport:
    def _fp_df_with_losses(self, losses: list) -> pd.DataFrame:
        rows = [
            {"is_true_positive": False, "min_loss_120m": v, "max_gain_120m": 0.0}
            for v in losses
        ]
        return pd.DataFrame(rows)

    def test_raises_on_empty_df(self):
        analyzer = PredictionErrorAnalyzer()
        with pytest.raises(ValueError, match="no false positives"):
            analyzer.generate_risk_report(pd.DataFrame())

    def test_raises_on_missing_column(self):
        analyzer = PredictionErrorAnalyzer()
        df = pd.DataFrame({"is_true_positive": [False]})
        with pytest.raises(ValueError, match="min_loss_120m"):
            analyzer.generate_risk_report(df)

    def test_total_false_positives_count(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -10.0, -15.0])
        report = analyzer.generate_risk_report(fp_df)
        assert report["total_false_positives"] == 3

    def test_pct_never_below_entry(self):
        analyzer = PredictionErrorAnalyzer()
        # 2 losses above entry (0.0, 3.0), 2 losses below (-5, -10)
        fp_df = self._fp_df_with_losses([0.0, 3.0, -5.0, -10.0])
        report = analyzer.generate_risk_report(fp_df)
        assert report["pct_price_never_below_entry"] == pytest.approx(0.5, abs=1e-6)

    def test_all_never_below_entry(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([0.0, 5.0, 10.0])
        report = analyzer.generate_risk_report(fp_df)
        assert report["pct_price_never_below_entry"] == pytest.approx(1.0, abs=1e-6)

    def test_mean_median_drawdown(self):
        analyzer = PredictionErrorAnalyzer()
        losses = [-10.0, -20.0, -30.0]
        fp_df = self._fp_df_with_losses(losses)
        report = analyzer.generate_risk_report(fp_df)
        assert report["mean_worst_drawdown_pct"] == pytest.approx(-20.0, abs=0.01)
        assert report["median_worst_drawdown_pct"] == pytest.approx(-20.0, abs=0.01)

    def test_max_worst_drawdown_is_minimum_value(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -50.0, -15.0])
        report = analyzer.generate_risk_report(fp_df)
        assert report["max_worst_drawdown_pct"] == pytest.approx(-50.0, abs=0.01)

    def test_loss_buckets_sum_to_one(self):
        analyzer = PredictionErrorAnalyzer()
        # 10 evenly-spaced losses from -25 to 0
        losses = list(np.linspace(-25.0, 0.0, 10))
        fp_df = self._fp_df_with_losses(losses)
        report = analyzer.generate_risk_report(fp_df)
        bucket_sum = (
            report["pct_never_below_entry"]
            + report["pct_0_to_5pct"]
            + report["pct_5_to_10pct"]
            + report["pct_10_to_15pct"]
            + report["pct_15_to_20pct"]
            + report["pct_over_20pct"]
        )
        assert bucket_sum == pytest.approx(1.0, abs=1e-6)

    def test_stop_loss_trigger_pcts_monotone_decreasing(self):
        """Looser stops catch more trades — triggered% must be non-decreasing as stop tightens."""
        analyzer = PredictionErrorAnalyzer()
        losses = list(np.linspace(-30.0, -1.0, 50))
        fp_df = self._fp_df_with_losses(losses)
        report = analyzer.generate_risk_report(fp_df)
        # tighter stop = smaller (more negative) threshold = more triggered
        # stop_5 < stop_10 < stop_15 < stop_20  (all catch fewer trades)
        assert report["stop_5pct_triggered_pct"] >= report["stop_10pct_triggered_pct"]
        assert report["stop_10pct_triggered_pct"] >= report["stop_15pct_triggered_pct"]
        assert report["stop_15pct_triggered_pct"] >= report["stop_20pct_triggered_pct"]

    def test_stop_recommendations_ordered(self):
        """Conservative stop is widest (most negative, p75 < p90 < p95 numerically)."""
        analyzer = PredictionErrorAnalyzer()
        losses = list(np.linspace(-30.0, -1.0, 30))
        fp_df = self._fp_df_with_losses(losses)
        report = analyzer.generate_risk_report(fp_df)
        # p75 is more negative than p90 which is more negative than p95
        assert report["stop_loss_conservative_pct"] <= report["stop_loss_moderate_pct"]
        assert report["stop_loss_moderate_pct"] <= report["stop_loss_aggressive_pct"]

    def test_report_keys_present(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -10.0, -20.0])
        report = analyzer.generate_risk_report(fp_df)
        expected_keys = {
            "total_false_positives",
            "pct_price_never_below_entry",
            "mean_worst_drawdown_pct",
            "median_worst_drawdown_pct",
            "p25_worst_drawdown_pct",
            "p50_worst_drawdown_pct",
            "p75_worst_drawdown_pct",
            "p90_worst_drawdown_pct",
            "max_worst_drawdown_pct",
            "pct_0_to_5pct",
            "pct_5_to_10pct",
            "pct_10_to_15pct",
            "pct_15_to_20pct",
            "pct_over_20pct",
            "pct_never_below_entry",
            "stop_5pct_triggered_pct",
            "stop_10pct_triggered_pct",
            "stop_15pct_triggered_pct",
            "stop_20pct_triggered_pct",
            "stop_loss_conservative_pct",
            "stop_loss_moderate_pct",
            "stop_loss_aggressive_pct",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_nan_values_dropped(self):
        """NaN min_loss_120m values should be dropped and not affect counts."""
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-10.0, float("nan"), -20.0])
        # Should not raise, should use 2 valid values
        report = analyzer.generate_risk_report(fp_df)
        # n in report is len(fp_df) = 3 but losses.dropna() = 2
        assert report["total_false_positives"] == 3


# ---------------------------------------------------------------------------
# stop_loss_impact
# ---------------------------------------------------------------------------


class TestStopLossImpact:
    def _fp_df_with_losses(self, losses: list) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"is_true_positive": False, "min_loss_120m": v, "max_gain_120m": 0.0}
                for v in losses
            ]
        )

    def test_default_stop_levels_present(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -15.0, -25.0])
        result = analyzer.stop_loss_impact(fp_df)
        assert set(result.keys()) == {-5.0, -10.0, -15.0, -20.0, -25.0}

    def test_custom_stop_levels(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -15.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-7.0, -12.0])
        assert set(result.keys()) == {-7.0, -12.0}

    def test_triggered_count_correct(self):
        analyzer = PredictionErrorAnalyzer()
        # Losses: -3, -8, -12, -18 → only 2 exceed -10 stop
        fp_df = self._fp_df_with_losses([-3.0, -8.0, -12.0, -18.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-10.0])
        assert result[-10.0]["triggered_count"] == 2
        assert result[-10.0]["uncaught_count"] == 2

    def test_triggered_pct_correct(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -15.0, -25.0, -35.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-20.0])
        # 2 out of 4 exceed -20 stop
        assert result[-20.0]["triggered_pct"] == pytest.approx(0.5, abs=1e-6)

    def test_no_triggered_when_all_above_stop(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-1.0, -2.0, -3.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-10.0])
        assert result[-10.0]["triggered_count"] == 0
        assert result[-10.0]["triggered_pct"] == 0.0

    def test_all_triggered_when_all_below_stop(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-15.0, -20.0, -30.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-5.0])
        assert result[-5.0]["triggered_count"] == 3
        assert result[-5.0]["triggered_pct"] == pytest.approx(1.0, abs=1e-6)

    def test_result_keys_per_stop(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-10.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-10.0])
        expected = {
            "stop_level_pct",
            "triggered_count",
            "triggered_pct",
            "exit_loss_pct",
            "uncaught_max_loss_pct",
            "uncaught_count",
        }
        assert expected.issubset(set(result[-10.0].keys()))

    def test_exit_loss_equals_stop_level(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df_with_losses([-5.0, -15.0])
        result = analyzer.stop_loss_impact(fp_df, stop_losses=[-8.0])
        assert result[-8.0]["exit_loss_pct"] == -8.0


# ---------------------------------------------------------------------------
# plot_ascii
# ---------------------------------------------------------------------------


class TestPlotAscii:
    def _fp_df(self, losses: list) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"is_true_positive": False, "min_loss_120m": v, "max_gain_120m": 0.0}
                for v in losses
            ]
        )

    def test_returns_string(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df([-5.0, -10.0, -15.0])
        result = analyzer.plot_ascii(fp_df)
        assert isinstance(result, str)

    def test_empty_df_returns_placeholder(self):
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.plot_ascii(pd.DataFrame())
        assert "no false positive" in result

    def test_missing_column_returns_placeholder(self):
        analyzer = PredictionErrorAnalyzer()
        df = pd.DataFrame({"is_true_positive": [False]})
        result = analyzer.plot_ascii(df)
        assert "no false positive" in result

    def test_all_nan_returns_placeholder(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df([float("nan"), float("nan")])
        result = analyzer.plot_ascii(fp_df)
        assert "NaN" in result

    def test_contains_summary_stats(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df([-5.0, -10.0, -15.0, -20.0])
        result = analyzer.plot_ascii(fp_df)
        assert "median=" in result
        assert "p90=" in result
        assert "worst=" in result

    def test_contains_histogram_bars(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df(list(np.linspace(-30.0, -1.0, 20)))
        result = analyzer.plot_ascii(fp_df, bins=5)
        assert "█" in result

    def test_custom_bin_count(self):
        analyzer = PredictionErrorAnalyzer()
        fp_df = self._fp_df(list(np.linspace(-30.0, 0.0, 30)))
        result5 = analyzer.plot_ascii(fp_df, bins=5)
        result10 = analyzer.plot_ascii(fp_df, bins=10)
        # 5 bins → fewer lines than 10 bins (both have header + footer)
        assert result5.count("\n") < result10.count("\n")

    def test_n_count_in_output(self):
        analyzer = PredictionErrorAnalyzer()
        losses = [-5.0, -10.0, -15.0]
        fp_df = self._fp_df(losses)
        result = analyzer.plot_ascii(fp_df)
        assert f"n={len(losses)}" in result


# ---------------------------------------------------------------------------
# calculate_expected_value
# ---------------------------------------------------------------------------


class TestCalculateExpectedValue:
    def test_profitable_high_precision(self):
        """90% precision, +20% win, -10% loss → EV = 0.90*20 + 0.10*(-10) = +17%."""
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(
            precision=0.90, avg_win_pct=20.0, stop_loss_pct=-10.0
        )
        assert result["profitable"] is True
        assert result["expected_value_pct"] == pytest.approx(17.0, abs=0.01)

    def test_unprofitable_low_precision(self):
        """40% precision, +20% win, -20% loss → EV = 0.40*20 + 0.60*(-20) = -4%."""
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(
            precision=0.40, avg_win_pct=20.0, stop_loss_pct=-20.0
        )
        assert result["profitable"] is False
        assert result["expected_value_pct"] == pytest.approx(-4.0, abs=0.01)

    def test_breakeven_precision(self):
        """At exactly the breakeven win rate EV should be 0."""
        analyzer = PredictionErrorAnalyzer()
        # breakeven = -(-10) / (20 - (-10)) = 10/30 ≈ 0.3333
        breakeven_p = 10.0 / 30.0
        result = analyzer.calculate_expected_value(
            precision=breakeven_p, avg_win_pct=20.0, stop_loss_pct=-10.0
        )
        assert result["expected_value_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_avg_loss_pct_overrides_stop_loss(self):
        """When avg_loss_pct is provided it takes precedence over stop_loss_pct."""
        analyzer = PredictionErrorAnalyzer()
        result_stop = analyzer.calculate_expected_value(
            precision=0.80, avg_win_pct=20.0, stop_loss_pct=-5.0
        )
        result_avg = analyzer.calculate_expected_value(
            precision=0.80, avg_win_pct=20.0, avg_loss_pct=-5.0, stop_loss_pct=-99.0
        )
        assert result_stop["expected_value_pct"] == pytest.approx(
            result_avg["expected_value_pct"], abs=1e-6
        )

    def test_win_loss_rate_sum_to_one(self):
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(precision=0.75)
        assert result["win_rate"] + result["loss_rate"] == pytest.approx(1.0, abs=1e-6)

    def test_raises_on_precision_above_one(self):
        analyzer = PredictionErrorAnalyzer()
        with pytest.raises(ValueError, match="precision must be in"):
            analyzer.calculate_expected_value(precision=1.5)

    def test_raises_on_negative_precision(self):
        analyzer = PredictionErrorAnalyzer()
        with pytest.raises(ValueError, match="precision must be in"):
            analyzer.calculate_expected_value(precision=-0.1)

    def test_precision_zero_all_losses(self):
        """Precision=0: 100% false positives, EV = stop_loss_pct."""
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(
            precision=0.0, avg_win_pct=20.0, stop_loss_pct=-15.0
        )
        assert result["expected_value_pct"] == pytest.approx(-15.0, abs=0.01)
        assert result["profitable"] is False

    def test_precision_one_all_wins(self):
        """Precision=1: 100% true positives, EV = avg_win_pct."""
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(
            precision=1.0, avg_win_pct=25.0, stop_loss_pct=-10.0
        )
        assert result["expected_value_pct"] == pytest.approx(25.0, abs=0.01)
        assert result["profitable"] is True

    def test_result_keys_present(self):
        analyzer = PredictionErrorAnalyzer()
        result = analyzer.calculate_expected_value(precision=0.85)
        expected_keys = {
            "win_rate",
            "loss_rate",
            "avg_win_pct",
            "avg_loss_pct",
            "expected_value_pct",
            "profitable",
            "breakeven_win_rate",
        }
        assert expected_keys == set(result.keys())
