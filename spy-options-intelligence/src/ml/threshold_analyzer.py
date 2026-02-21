# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Threshold sensitivity analysis with temporal breakdown.

Purpose
-------
Sweeps model probability thresholds from *min_threshold* to *max_threshold*
and computes — for each threshold — signal counts, precision, recall, and
detailed gain/loss distributions for true positives (TP), false positives (FP),
and false negatives (FN).  Results are broken down three ways:

  1. **Full-year aggregate** — all dates combined
  2. **Monthly breakdown** — statistics per calendar month
  3. **Daily breakdown** — statistics per trading day

NOTE: This module operates on the **full** loaded dataset (including the
training period).  Precision / recall on training dates will be optimistic
because the model has seen that data.  Use ``ml backtest --threshold X``
for held-out test-set evaluation.

Public API
----------
  ThresholdAnalyzer
    .analyze_full_year(artifact, features_dir, thresholds, start_date, end_date)
        → dict with keys: aggregate (DataFrame), monthly (DataFrame),
          daily (DataFrame), date_range, total_samples, n_dates, n_months

    .generate_monthly_summary(monthly_df, key_thresholds)
        → DataFrame — pivot: month × threshold showing signals/precision/EV

    .plot_monthly_signals(monthly_summary, key_thresholds)
        → str  (ASCII bar chart, suitable for click.echo)

    .find_optimal_threshold(results_df, optimization_metric, min_precision, min_signals)
        → dict with status / optimal_threshold / metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Defaults
_DEFAULT_THRESHOLDS: List[float] = [round(t, 2) for t in np.arange(0.70, 0.96, 0.01)]
_KEY_THRESHOLDS: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


class ThresholdAnalyzer:
    """Sweep probability thresholds and report TP / FP / FN gain-loss stats.

    Designed to work with the output of ``MLFeatureEngineer`` — feature CSVs
    that include ``target``, ``max_gain_120m``, ``min_loss_120m``, and
    ``date`` columns alongside the model feature columns.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_full_year(
        self,
        artifact: Dict[str, Any],
        features_dir: Union[str, Path],
        thresholds: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load features, predict, and run threshold sweep across date range.

        Args:
            artifact:     Model artifact dict from ``joblib.load()`` — must
                          contain keys ``model`` and ``feature_cols``.
            features_dir: Directory containing ``*_features.csv`` files.
            thresholds:   Thresholds to sweep.  Defaults to 0.70–0.95 (step 0.01).
            start_date:   Earliest date to include (``"YYYY-MM-DD"``).  No lower
                          bound if omitted.
            end_date:     Latest date to include (``"YYYY-MM-DD"``).  No upper
                          bound if omitted.

        Returns:
            Dict with keys:
              ``aggregate``     pd.DataFrame — one row per threshold, full dataset
              ``monthly``       pd.DataFrame — one row per (month × threshold)
              ``daily``         pd.DataFrame — one row per (date × threshold)
              ``date_range``    (str, str)   — min and max date in dataset
              ``total_samples`` int
              ``n_dates``       int
              ``n_months``      int

        Raises:
            ValueError: If no feature data is found or required columns are missing.
        """
        from src.ml.train_xgboost import load_features  # deferred — heavy import

        if thresholds is None:
            thresholds = _DEFAULT_THRESHOLDS

        model = artifact["model"]
        feature_cols: List[str] = artifact["feature_cols"]

        # ── Load feature data ────────────────────────────────────────────
        df = load_features(features_dir, start_date, end_date)
        if df.empty:
            raise ValueError(
                f"ThresholdAnalyzer: no feature data in {features_dir} "
                f"for [{start_date}, {end_date}]"
            )

        required_cols = {"target", "max_gain_120m", "min_loss_120m", "date"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"ThresholdAnalyzer: required columns missing: {missing_cols!r}"
            )

        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(
                f"ThresholdAnalyzer: model feature columns missing from data: "
                f"{missing_feats[:5]!r}{'…' if len(missing_feats) > 5 else ''}"
            )

        # ── Predict on full dataset in one batch ─────────────────────────
        df = df.reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)
        probas: np.ndarray = model.predict_proba(X)[:, 1]
        df["_proba"] = probas

        # Month column for groupby
        df["_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

        n_dates = df["date"].nunique()
        n_months = df["_month"].nunique()

        logger.info(
            f"ThresholdAnalyzer.analyze_full_year: {len(df)} rows | "
            f"{n_dates} dates | {n_months} months | "
            f"thresholds {thresholds[0]:.2f}–{thresholds[-1]:.2f}"
        )

        # ── Aggregate ────────────────────────────────────────────────────
        aggregate_df = self._analyze_threshold_range(df, probas, thresholds)

        # ── Monthly ──────────────────────────────────────────────────────
        monthly_parts: List[pd.DataFrame] = []
        for month, grp in df.groupby("_month", sort=True):
            grp = grp.reset_index(drop=True)
            month_df = self._analyze_threshold_range(
                grp, grp["_proba"].values, thresholds
            )
            month_df.insert(0, "month", str(month))
            monthly_parts.append(month_df)
        monthly_df = pd.concat(monthly_parts, ignore_index=True)

        # ── Daily ────────────────────────────────────────────────────────
        daily_parts: List[pd.DataFrame] = []
        for date, grp in df.groupby("date", sort=True):
            grp = grp.reset_index(drop=True)
            date_df = self._analyze_threshold_range(
                grp, grp["_proba"].values, thresholds
            )
            date_df.insert(0, "date", str(date))
            daily_parts.append(date_df)
        daily_df = pd.concat(daily_parts, ignore_index=True)

        return {
            "aggregate": aggregate_df,
            "monthly": monthly_df,
            "daily": daily_df,
            "date_range": (df["date"].min(), df["date"].max()),
            "total_samples": len(df),
            "n_dates": n_dates,
            "n_months": n_months,
        }

    def generate_monthly_summary(
        self,
        monthly_df: pd.DataFrame,
        key_thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Pivot monthly_df to a compact summary at key thresholds.

        Args:
            monthly_df:     Output of ``analyze_full_year["monthly"]``.
            key_thresholds: Thresholds to include.  Defaults to
                            ``[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]``.

        Returns:
            DataFrame with one row per month and three columns per threshold:
            ``signals_{pct}``, ``precision_{pct}``, ``ev_{pct}``.
        """
        if key_thresholds is None:
            key_thresholds = _KEY_THRESHOLDS

        rows: List[Dict[str, Any]] = []
        for month in sorted(monthly_df["month"].unique()):
            month_slice = monthly_df[monthly_df["month"] == month]
            row: Dict[str, Any] = {"month": month}
            for t in key_thresholds:
                t_key = round(t, 2)
                pct = int(t * 100)
                match = month_slice[month_slice["threshold"].round(2) == t_key]
                if not match.empty:
                    row[f"signals_{pct}"] = int(match.iloc[0]["total_signals"])
                    row[f"precision_{pct}"] = round(
                        float(match.iloc[0]["precision"]), 4
                    )
                    row[f"ev_{pct}"] = round(
                        float(match.iloc[0]["expected_value_pct"]), 2
                    )
                else:
                    row[f"signals_{pct}"] = 0
                    row[f"precision_{pct}"] = 0.0
                    row[f"ev_{pct}"] = 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_monthly_signals(
        self,
        monthly_summary: pd.DataFrame,
        key_thresholds: Optional[List[float]] = None,
    ) -> str:
        """Return an ASCII bar chart of monthly signal counts at key thresholds.

        Args:
            monthly_summary: Output of ``generate_monthly_summary``.
            key_thresholds:  Thresholds to plot.  Defaults to key thresholds.

        Returns:
            Multi-line string suitable for ``click.echo`` / ``print``.
        """
        if key_thresholds is None:
            key_thresholds = _KEY_THRESHOLDS

        bar_width = 28
        lines = ["─" * 72, "MONTHLY SIGNAL DISTRIBUTION", "─" * 72]

        for t in key_thresholds:
            pct = int(t * 100)
            col_sig = f"signals_{pct}"
            col_prec = f"precision_{pct}"
            if col_sig not in monthly_summary.columns:
                continue

            lines.append(f"\n  Threshold {t:.2f} ({pct}%)")
            lines.append("  " + "─" * 52)

            max_sig = int(monthly_summary[col_sig].max())
            if max_sig == 0:
                lines.append("  (no signals)")
                continue

            for _, row in monthly_summary.iterrows():
                month = row["month"]
                sigs = int(row[col_sig])
                prec = float(row.get(col_prec, 0.0))
                bar_len = int(sigs / max_sig * bar_width)
                bar = "█" * bar_len + "░" * (bar_width - bar_len)
                lines.append(f"  {month}  {bar}  {sigs:4d}  ({prec:.1%})")

        lines.append("\n" + "─" * 72)
        return "\n".join(lines)

    def find_optimal_threshold(
        self,
        results_df: pd.DataFrame,
        optimization_metric: str = "expected_value_pct",
        min_precision: float = 0.90,
        min_signals: int = 10,
    ) -> Dict[str, Any]:
        """Find the threshold that maximises *optimization_metric* under constraints.

        Args:
            results_df:           Aggregate threshold results DataFrame.
            optimization_metric:  Column to maximise.  Default ``expected_value_pct``.
            min_precision:        Minimum acceptable precision.  Default 0.90.
            min_signals:          Minimum number of total signals.  Default 10.

        Returns:
            Dict with keys:
              ``status``             ``"SUCCESS"`` or ``"NO_VALID_THRESHOLD"``
              ``optimal_threshold``  float  (only when SUCCESS)
              ``metrics``            dict of all metric values  (only when SUCCESS)
              ``message``            str   (only when NO_VALID_THRESHOLD)

        Raises:
            ValueError: If *optimization_metric* is not a column in *results_df*.
        """
        if optimization_metric not in results_df.columns:
            raise ValueError(
                f"find_optimal_threshold: '{optimization_metric}' not in results_df; "
                f"available: {list(results_df.columns)}"
            )

        valid = results_df[
            (results_df["precision"] >= min_precision)
            & (results_df["total_signals"] >= min_signals)
        ].copy()

        if valid.empty:
            return {
                "status": "NO_VALID_THRESHOLD",
                "message": (
                    f"No threshold meets precision≥{min_precision:.0%} "
                    f"and total_signals≥{min_signals}"
                ),
            }

        best_idx = valid[optimization_metric].idxmax()
        best = valid.loc[best_idx]

        def _to_python(v: Any) -> Any:
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        return {
            "status": "SUCCESS",
            "optimal_threshold": round(float(best["threshold"]), 4),
            "metrics": {k: _to_python(v) for k, v in best.to_dict().items()},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_threshold_range(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        thresholds: List[float],
    ) -> pd.DataFrame:
        """Run ``_analyze_single_threshold`` for every threshold in *thresholds*.

        Args:
            df:          DataFrame with ``target``, ``max_gain_120m``,
                         ``min_loss_120m`` columns.
            predictions: Probability array aligned with *df* (post reset_index).
            thresholds:  Thresholds to evaluate.

        Returns:
            DataFrame with one row per threshold.
        """
        y_true = df["target"].fillna(0).values.astype(int)
        max_gains = df["max_gain_120m"].values.astype(float)
        min_losses = df["min_loss_120m"].values.astype(float)

        rows = []
        for t in thresholds:
            row = self._analyze_single_threshold(
                y_true, predictions, max_gains, min_losses, t
            )
            row["threshold"] = round(float(t), 4)
            rows.append(row)

        cols = ["threshold"] + [c for c in rows[0] if c != "threshold"]
        return pd.DataFrame(rows)[cols]

    def _analyze_single_threshold(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        max_gains: np.ndarray,
        min_losses: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """Compute all metrics at a single probability threshold.

        Args:
            y_true:      True binary labels.
            predictions: Predicted probabilities.
            max_gains:   Max % gain in 120-min forward window (per bar).
            min_losses:  Min % change (worst drawdown) in 120-min forward window.
            threshold:   Probability cutoff for a positive prediction.

        Returns:
            Dict of scalar metrics.
        """
        y_pred = (predictions >= threshold).astype(int)

        signal_mask = y_pred == 1
        tp_mask = signal_mask & (y_true == 1)
        fp_mask = signal_mask & (y_true == 0)
        fn_mask = (~signal_mask) & (y_true == 1)
        tn_mask = (~signal_mask) & (y_true == 0)

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        tn = int(tn_mask.sum())
        n_signals = tp + fp
        n_actual_pos = tp + fn
        n_total = len(y_true)

        precision = tp / n_signals if n_signals > 0 else 0.0
        recall = tp / n_actual_pos if n_actual_pos > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Extract gain/loss arrays — drop NaN (end-of-day bars with no fwd window)
        tp_gains = pd.Series(max_gains[tp_mask]).dropna()
        fp_losses = pd.Series(min_losses[fp_mask]).dropna()
        fn_missed = pd.Series(max_gains[fn_mask]).dropna()

        def _stat(s: pd.Series, prefix: str) -> Dict[str, Any]:
            if s.empty:
                return {
                    f"{prefix}_count": 0,
                    f"{prefix}_max": None,
                    f"{prefix}_avg": None,
                    f"{prefix}_median": None,
                    f"{prefix}_min": None,
                    f"{prefix}_std": None,
                }
            return {
                f"{prefix}_count": len(s),
                f"{prefix}_max": round(float(s.max()), 2),
                f"{prefix}_avg": round(float(s.mean()), 2),
                f"{prefix}_median": round(float(s.median()), 2),
                f"{prefix}_min": round(float(s.min()), 2),
                f"{prefix}_std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
            }

        # Expected value = precision × avg_tp_gain + (1 − precision) × avg_fp_loss
        ev = 0.0
        if n_signals > 0:
            if not tp_gains.empty:
                ev += precision * float(tp_gains.mean())
            if not fp_losses.empty:
                ev += (1.0 - precision) * float(fp_losses.mean())

        result: Dict[str, Any] = {
            # Counts
            "total_signals": n_signals,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "signal_rate": round(n_signals / n_total, 6) if n_total > 0 else 0.0,
            # Classification metrics
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            # TP profit stats
            **_stat(tp_gains, "tp_profit_pct"),
            # FP loss stats  (min_loss_120m is negative for actual losses)
            **_stat(fp_losses, "fp_loss_pct"),
            # FN missed opportunity stats
            **_stat(fn_missed, "fn_missed_pct"),
            # Expected value
            "expected_value_pct": round(ev, 4),
        }
        return result
