# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Walk-forward validation for time-series model stability assessment.

Tests whether the current 91.9% backtest precision is typical or an outlier
by re-training XGBoost from scratch on each rolling window and evaluating
on the following unseen month.

Walk-forward scheme (default: 3-month train, 1-month test, 1-month slide):
    Split 1: Train [Mar, Apr, May] → Test [Jun]
    Split 2: Train [Apr, May, Jun] → Test [Jul]
    ...
    Split 9: Train [Nov, Dec, Jan] → Test [Feb]

All test windows are non-overlapping; training windows overlap by
(train_window_months - test_window_months) months.

Usage
-----
    from src.ml.walk_forward_validator import WalkForwardValidator

    validator = WalkForwardValidator(features_dir="data/processed/features")
    summary = validator.run_validation(threshold=0.67)
    print(validator.plot_results(summary))
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import xgboost as xgb
from sklearn.metrics import confusion_matrix

from src.ml.data_balancer import undersample_majority
from src.ml.train_xgboost import _NON_FEATURE_COLS, load_features


# ---------------------------------------------------------------------------
# Defaults — identical to xgboost_v2 training config
# ---------------------------------------------------------------------------

_DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "gamma": 0.10,
    "random_state": 42,
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
}

# Minimum positive examples required in a training fold to proceed
_MIN_POSITIVES = 10


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Perform rolling walk-forward validation for SPY options spike prediction.

    Parameters
    ----------
    features_dir:
        Directory containing ``{YYYY-MM-DD}_features.csv`` files.
    xgb_params:
        XGBoost constructor keyword arguments.  Defaults to the same params
        used to train ``xgboost_v2``.
    train_window_months:
        Number of calendar months in each training window (default 3).
    test_window_months:
        Number of calendar months in each test window (default 1).  The
        training window slides forward by this amount each iteration, so
        test windows are non-overlapping.
    """

    def __init__(
        self,
        features_dir: str | Path,
        xgb_params: Optional[Dict[str, Any]] = None,
        train_window_months: int = 3,
        test_window_months: int = 1,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.xgb_params: Dict[str, Any] = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_date_splits(self) -> List[Tuple[str, str, str, str]]:
        """Generate ``(train_start, train_end, test_start, test_end)`` tuples.

        Uses calendar-month boundaries.  The training window slides forward
        by ``test_window_months`` each iteration, producing non-overlapping
        test windows.

        Returns
        -------
        List of 4-tuples with ISO date strings.
        """
        feature_files = sorted(self.features_dir.glob("*_features.csv"))
        if not feature_files:
            return []

        all_dates = sorted(f.stem.split("_")[0] for f in feature_files)
        first_dt = pd.Timestamp(all_dates[0])
        last_dt = pd.Timestamp(all_dates[-1])

        # Align to the first day of the month of the earliest date
        current_start = first_dt.replace(day=1)

        splits: List[Tuple[str, str, str, str]] = []
        while True:
            # Training window: [current_start, current_start + train_months)
            train_end_dt = current_start + relativedelta(
                months=self.train_window_months
            ) - timedelta(days=1)

            # Test window immediately follows
            test_start_dt = train_end_dt + timedelta(days=1)
            test_end_dt = test_start_dt + relativedelta(
                months=self.test_window_months
            ) - timedelta(days=1)

            # Stop when test window would exceed available data
            if test_end_dt > last_dt:
                break

            splits.append(
                (
                    current_start.strftime("%Y-%m-%d"),
                    train_end_dt.strftime("%Y-%m-%d"),
                    test_start_dt.strftime("%Y-%m-%d"),
                    test_end_dt.strftime("%Y-%m-%d"),
                )
            )

            # Slide forward by one test window (non-overlapping tests)
            current_start += relativedelta(months=self.test_window_months)

        return splits

    def load_date_range(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load all feature CSVs whose date falls in ``[start_date, end_date]``.

        Delegates to ``load_features`` from ``train_xgboost`` so the same
        filtering and concatenation logic is used everywhere.

        Parameters
        ----------
        start_date, end_date:
            Inclusive ISO date strings (``"YYYY-MM-DD"``).

        Returns
        -------
        Concatenated DataFrame, or an empty DataFrame if no files match.
        """
        return load_features(self.features_dir, start_date=start_date, end_date=end_date)

    def evaluate_split(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        threshold: float = 0.67,
    ) -> Dict[str, Any]:
        """Train on ``[train_start, train_end]``, evaluate on ``[test_start, test_end]``.

        Training pipeline mirrors ``XGBoostTrainer``:
        1. Chronological 80/20 split within the training window for early
           stopping (val is never undersampled).
        2. Undersample majority class in the training portion.
        3. Fit XGBoost with early stopping.
        4. Threshold predictions on the test set and compute precision/recall/EV.

        Parameters
        ----------
        train_start, train_end, test_start, test_end:
            Inclusive ISO date strings.
        threshold:
            Probability threshold for signal firing (default 0.67).

        Returns
        -------
        Result dict with ``status`` key (``"SUCCESS"`` | ``"INSUFFICIENT_DATA"``).
        """
        train_df = self.load_date_range(train_start, train_end)
        test_df = self.load_date_range(test_start, test_end)

        if train_df.empty or test_df.empty:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": "empty train or test set",
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_period": f"{train_start} → {train_end}",
                "test_period": f"{test_start} → {test_end}",
            }

        feature_cols = self._get_feature_cols(train_df)

        # Check minimum positives in training data
        n_pos_train = int((train_df["target"] == 1).sum())
        if n_pos_train < _MIN_POSITIVES:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": f"only {n_pos_train} positives in training window (need {_MIN_POSITIVES})",
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_period": f"{train_start} → {train_end}",
                "test_period": f"{test_start} → {test_end}",
            }

        # Train model on this fold
        model = self._train_fold(train_df, feature_cols)

        # ── Evaluate on test set ──────────────────────────────────────────
        # Align to the exact features the model was trained on
        missing_cols = [c for c in feature_cols if c not in test_df.columns]
        for col in missing_cols:
            test_df[col] = 0.0

        X_test = test_df[feature_cols].fillna(0.0).values.astype(np.float32)
        y_test = test_df["target"].values.astype(np.int8)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(np.int8)

        # Guard: single-class test set
        unique_preds = set(np.unique(y_pred))
        unique_true = set(np.unique(y_test))
        if len(unique_preds) < 2 or len(unique_true) < 2:
            # No signals fired or test set has only one class
            n_signals = int(y_pred.sum())
            tp = int(((y_pred == 1) & (y_test == 1)).sum())
            fp = n_signals - tp
            fn = int(((y_pred == 0) & (y_test == 1)).sum())
            tn = int(((y_pred == 0) & (y_test == 0)).sum())
        else:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()

        n_signals = tp + fp
        precision = tp / n_signals if n_signals > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Profit analysis from label columns
        tp_mask = (y_pred == 1) & (y_test == 1)
        fp_mask = (y_pred == 1) & (y_test == 0)

        tp_gains = (
            test_df.loc[tp_mask, "max_gain_120m"].dropna()
            if "max_gain_120m" in test_df.columns
            else pd.Series(dtype=float)
        )
        fp_losses = (
            test_df.loc[fp_mask, "min_loss_120m"].dropna()
            if "min_loss_120m" in test_df.columns
            else pd.Series(dtype=float)
        )

        tp_avg_gain = float(tp_gains.mean()) if len(tp_gains) > 0 else 0.0
        fp_avg_loss = float(fp_losses.mean()) if len(fp_losses) > 0 else 0.0

        # EV = precision × avg_tp_gain + (1−precision) × avg_fp_loss
        expected_value = (
            precision * tp_avg_gain + (1.0 - precision) * fp_avg_loss
            if n_signals > 0
            else 0.0
        )

        return {
            "status": "SUCCESS",
            "train_period": f"{train_start} → {train_end}",
            "test_period": f"{test_start} → {test_end}",
            "test_month": test_start[:7],  # "YYYY-MM" label for charts
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_positives": n_pos_train,
            "test_positives": int((y_test == 1).sum()),
            "threshold": threshold,
            # Signal counts
            "total_signals": n_signals,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            # Classifier metrics
            "precision": precision,
            "recall": recall,
            "signal_rate": n_signals / len(test_df) if len(test_df) > 0 else 0.0,
            # Profit metrics
            "tp_avg_gain_pct": tp_avg_gain,
            "fp_avg_loss_pct": fp_avg_loss,
            "expected_value_pct": expected_value,
        }

    def run_validation(self, threshold: float = 0.67) -> Dict[str, Any]:
        """Run complete walk-forward validation across all generated splits.

        Parameters
        ----------
        threshold:
            Probability threshold applied to every split's test predictions.

        Returns
        -------
        Summary dict with aggregated statistics and per-split results.
        ``status`` is ``"SUCCESS"``, ``"NO_SPLITS"``, or ``"ALL_SPLITS_FAILED"``.
        """
        splits = self.get_date_splits()

        if not splits:
            return {
                "status": "NO_SPLITS",
                "message": "No date splits generated — insufficient data range",
            }

        results: List[Dict[str, Any]] = []

        for i, (ts, te, vs, ve) in enumerate(splits, 1):
            result = self.evaluate_split(ts, te, vs, ve, threshold)
            result["split_index"] = i
            results.append(result)

        successful = [r for r in results if r["status"] == "SUCCESS"]

        if not successful:
            return {
                "status": "ALL_SPLITS_FAILED",
                "message": "No splits had sufficient data",
                "splits": results,
            }

        # ── Aggregate statistics ──────────────────────────────────────────
        precisions = [r["precision"] for r in successful]
        signals = [r["total_signals"] for r in successful]
        evs = [r["expected_value_pct"] for r in successful]

        return {
            "status": "SUCCESS",
            "threshold": threshold,
            "total_splits": len(splits),
            "successful_splits": len(successful),
            "failed_splits": len(splits) - len(successful),
            # Precision
            "precision_mean": float(np.mean(precisions)),
            "precision_median": float(np.median(precisions)),
            "precision_std": float(np.std(precisions)),
            "precision_min": float(np.min(precisions)),
            "precision_max": float(np.max(precisions)),
            # Signals per test month
            "signals_mean": float(np.mean(signals)),
            "signals_median": float(np.median(signals)),
            "signals_min": int(np.min(signals)),
            "signals_max": int(np.max(signals)),
            # Expected value
            "ev_mean": float(np.mean(evs)),
            "ev_median": float(np.median(evs)),
            "ev_std": float(np.std(evs)),
            # Per-split detail
            "splits": results,
        }

    def plot_results(self, summary: Dict[str, Any]) -> str:
        """Render an ASCII bar chart of precision per test month.

        Parameters
        ----------
        summary:
            Return value of ``run_validation()``.

        Returns
        -------
        Multi-line string suitable for ``print()`` or file output.
        """
        if summary.get("status") != "SUCCESS":
            return f"[No results to plot: {summary.get('status')}]"

        successful = [r for r in summary["splits"] if r["status"] == "SUCCESS"]
        if not successful:
            return "[No successful splits to plot]"

        width = 30
        lines = [
            "─" * 70,
            "PRECISION BY TEST MONTH",
            "─" * 70,
            "",
        ]

        mean_prec = summary["precision_mean"]

        for r in successful:
            month = r.get("test_month", r["test_period"][:7])
            prec = r["precision"]
            n_sig = r["total_signals"]

            filled = round(prec * width)
            bar = "█" * filled + "░" * (width - filled)
            marker = " <-- mean" if abs(prec - mean_prec) < 0.005 else ""

            lines.append(
                f"  {month}  {bar}  {prec:.1%}  ({n_sig} signals){marker}"
            )

        lines += [
            "",
            f"  Mean:    {'─' * round(mean_prec * width)}|  {mean_prec:.1%}",
            "─" * 70,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return sorted feature column list, excluding non-feature columns."""
        return sorted(c for c in df.columns if c not in _NON_FEATURE_COLS)

    def _train_fold(
        self, train_df: pd.DataFrame, feature_cols: List[str]
    ) -> xgb.XGBClassifier:
        """Train XGBoost on one fold.

        Internally splits the training window 80/20 by date order:
        - First 80%: undersampled, used for gradient updates.
        - Last 20%: unmodified, used as ``eval_set`` for early stopping.

        Parameters
        ----------
        train_df:
            Full training data for this fold (sorted by date implicitly from
            ``load_features``).
        feature_cols:
            Ordered list of feature column names.

        Returns
        -------
        Fitted ``XGBClassifier``.
        """
        # Chronological split: last 20% of dates → early-stopping val
        unique_dates = sorted(train_df["date"].unique()) if "date" in train_df.columns else []

        if len(unique_dates) >= 5:
            cutoff_idx = int(len(unique_dates) * 0.80)
            cutoff_date = unique_dates[cutoff_idx]
            fit_df = train_df[train_df["date"] < cutoff_date]
            val_df = train_df[train_df["date"] >= cutoff_date]
        else:
            # Too few dates: use all data for training, no early stopping
            fit_df = train_df
            val_df = pd.DataFrame()

        # Undersample majority class in training portion
        if len(fit_df) > 0:
            try:
                fit_df = undersample_majority(fit_df, target_col="target", random_state=42)
            except Exception:
                pass  # Fall through if already balanced or single class

        X_fit = fit_df[feature_cols].fillna(0.0).values.astype(np.float32)
        y_fit = fit_df["target"].values.astype(np.int8)

        model = xgb.XGBClassifier(**self.xgb_params)

        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if not val_df.empty and (val_df["target"] == 1).sum() > 0:
            X_val = val_df[feature_cols].fillna(0.0).values.astype(np.float32)
            y_val = val_df["target"].values.astype(np.int8)
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        model.fit(X_fit, y_fit, **fit_kwargs)
        return model
