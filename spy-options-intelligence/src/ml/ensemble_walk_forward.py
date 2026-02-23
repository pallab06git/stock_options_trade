# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Walk-forward stability test for the pre-trained magnitude ensemble.

Unlike the directional WalkForwardValidator (Step 43), this module does
NOT retrain on each fold.  It tests a *fixed* pre-trained ensemble
(LightGBM + RandomForest, AVG strategy) across n time-series windows to
verify that the 100% precision observed in the single 72-day test period
holds across different market regimes.

Walk-forward scheme (sklearn TimeSeriesSplit, n_splits folds):
    Fold 1: train idx [0 .. T1], test idx [T1 .. T2]
    Fold 2: train idx [0 .. T2], test idx [T2 .. T3]
    ...
    Fold 5: train idx [0 .. T4], test idx [T4 .. T5]

Train indices are only used to determine the test window boundaries;
no retraining occurs.  The same frozen models predict on every fold.

Verdict thresholds
------------------
    STABLE         : mean_precision ≥ 95% AND std_precision < 5%
    GOOD_BUT_VARIABLE : mean_precision ≥ 90% (but std ≥ 5%)
    UNSTABLE       : mean_precision < 90%
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import get_logger

logger = get_logger()


class EnsembleWalkForwardValidator:
    """Walk-forward stability test for a fixed pre-trained ensemble.

    Parameters
    ----------
    feature_cols:
        Ordered list of feature column names that both models were
        trained on (loaded from the saved artifact).
    """

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_ensemble_strategy(
        self,
        lgbm_model,
        rf_model,
        full_df: pd.DataFrame,
        threshold: float = 0.97,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Test the AVG-threshold ensemble across *n_splits* time windows.

        Parameters
        ----------
        lgbm_model:
            Fitted LightGBM classifier (``predict_proba`` interface).
        rf_model:
            Fitted RandomForest classifier (``predict_proba`` interface).
        full_df:
            DataFrame that already has ``target_magnitude``,
            ``abs_max_move_pct``, and ``move_direction`` columns
            (apply ``MagnitudeLabeler.label()`` before calling).
        threshold:
            AVG-probability threshold for firing a signal (default 0.97).
        n_splits:
            Number of TimeSeriesSplit folds (default 5).

        Returns
        -------
        dict with keys:
            results  — List[Dict] per fold
            summary  — aggregate stats + verdict string
        """
        all_dates = sorted(full_df["date"].unique())
        tscv = TimeSeriesSplit(n_splits=n_splits)

        width = 80
        print("\n" + "=" * width)
        print("  WALK-FORWARD CROSS-VALIDATION  (Step 67)")
        print("=" * width)
        print(f"\n  Total dates : {len(all_dates)}")
        print(f"  Splits      : {n_splits}")
        print(f"  Threshold   : {threshold} (AVG strategy)")
        print(f"  Features    : {len(self.feature_cols)}")

        results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(all_dates), 1):
            train_dates = [all_dates[i] for i in train_idx]
            test_dates  = [all_dates[i] for i in test_idx]

            print(f"\n{'=' * width}")
            print(f"  FOLD {fold}/{n_splits}")
            print(f"{'=' * width}")
            print(
                f"  Train: {train_dates[0]} → {train_dates[-1]}"
                f"  ({len(train_dates)} days)"
            )
            print(
                f"  Test:  {test_dates[0]} → {test_dates[-1]}"
                f"  ({len(test_dates)} days)"
            )

            test_fold = (
                full_df[full_df["date"].isin(set(test_dates))]
                .reset_index(drop=True)
            )

            if test_fold.empty:
                print("  (no rows — skipping)")
                continue

            X_test = (
                test_fold[self.feature_cols]
                .fillna(0.0)
                .values
                .astype(np.float32)
            )
            y_test = test_fold["target_magnitude"].values.astype(np.int8)

            # Ensemble: average probability of both models
            lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
            rf_proba   = rf_model.predict_proba(X_test)[:, 1]
            avg_proba  = (lgbm_proba + rf_proba) / 2.0

            signals = avg_proba >= threshold

            tp = int(( signals & (y_test == 1)).sum())
            fp = int(( signals & (y_test == 0)).sum())
            fn = int((~signals & (y_test == 1)).sum())

            total_signals  = tp + fp
            precision      = tp / total_signals if total_signals > 0 else 0.0
            recall         = tp / (tp + fn)     if (tp + fn) > 0     else 0.0
            signals_per_day = total_signals / max(len(test_dates), 1)

            # Magnitude and direction for signal rows
            signal_df = test_fold[signals]
            if len(signal_df) > 0:
                avg_mag    = float(signal_df["abs_max_move_pct"].mean())
                median_mag = float(signal_df["abs_max_move_pct"].median())
                up_moves   = int((signal_df["move_direction"] == "up").sum())
                down_moves = int((signal_df["move_direction"] == "down").sum())
            else:
                avg_mag = median_mag = 0.0
                up_moves = down_moves = 0

            print(f"\n  Results @ {threshold}:")
            print(f"    Signals    : {total_signals}  ({signals_per_day:.2f}/day)")
            print(f"    Precision  : {precision:.1%}")
            print(f"    Recall     : {recall:.1%}")
            print(f"    TP: {tp}  FP: {fp}  FN: {fn}")
            if total_signals > 0:
                print(f"\n    Magnitude  : avg={avg_mag:.1f}%  median={median_mag:.1f}%")
                print(f"    Direction  : {up_moves} UP  {down_moves} DOWN")

            results.append({
                "fold":             fold,
                "train_period":     f"{train_dates[0]} to {train_dates[-1]}",
                "test_period":      f"{test_dates[0]} to {test_dates[-1]}",
                "test_days":        len(test_dates),
                "signals":          total_signals,
                "signals_per_day":  float(signals_per_day),
                "precision":        float(precision),
                "recall":           float(recall),
                "avg_magnitude":    avg_mag,
                "median_magnitude": median_mag,
                "tp":               tp,
                "fp":               fp,
                "fn":               fn,
            })

        # ── Aggregate ──────────────────────────────────────────────────
        print(f"\n{'=' * width}")
        print("  CROSS-VALIDATION SUMMARY")
        print(f"{'=' * width}")

        folds_with_signals = [r for r in results if r["signals"] > 0]
        precisions  = [r["precision"]       for r in folds_with_signals]
        spd_vals    = [r["signals_per_day"] for r in results]
        magnitudes  = [r["avg_magnitude"]   for r in folds_with_signals]

        if precisions:
            mean_prec = float(np.mean(precisions))
            std_prec  = float(np.std(precisions))
            min_prec  = float(np.min(precisions))
            max_prec  = float(np.max(precisions))

            print(f"\n  Precision across {len(precisions)} folds (with signals):")
            print(f"    Mean   : {mean_prec:.1%}")
            print(f"    Median : {np.median(precisions):.1%}")
            print(f"    Std    : {std_prec:.1%}")
            print(f"    Min    : {min_prec:.1%}")
            print(f"    Max    : {max_prec:.1%}")
        else:
            mean_prec = std_prec = min_prec = max_prec = 0.0
            print("\n  No signals in any fold.")

        print(f"\n  Signals/day  :  mean={np.mean(spd_vals):.2f}  std={np.std(spd_vals):.2f}")

        if magnitudes:
            print(
                f"  Avg magnitude:  mean={np.mean(magnitudes):.1f}%"
                f"  std={np.std(magnitudes):.1f}%"
            )

        # ── Verdict ───────────────────────────────────────────────────
        print(f"\n{'=' * width}")
        print("  VERDICT")
        print(f"{'=' * width}")

        if precisions and mean_prec >= 0.95 and std_prec < 0.05:
            verdict = "STABLE"
            print("\n  ✅ STABLE & RELIABLE")
            print(f"     Precision consistently ≥95% across folds")
            print(f"     Low variance ({std_prec:.1%})")
            print(f"     Safe to deploy")
        elif precisions and mean_prec >= 0.90:
            verdict = "GOOD_BUT_VARIABLE"
            print("\n  ⚠️  GOOD BUT VARIABLE")
            print(f"     Mean precision ≥90% but variance = {std_prec:.1%}")
            print(f"     Monitor carefully in production")
        else:
            verdict = "UNSTABLE"
            print("\n  ❌ UNSTABLE")
            print(f"     Precision varies too much across periods")
            print(f"     May be overfit to specific regime")
            print(f"     DO NOT DEPLOY")

        summary: Dict[str, Any] = {
            "mean_precision":       mean_prec,
            "std_precision":        std_prec,
            "min_precision":        min_prec,
            "max_precision":        max_prec,
            "mean_signals_per_day": float(np.mean(spd_vals)),
            "std_signals_per_day":  float(np.std(spd_vals)),
            "n_folds_with_signals": len(folds_with_signals),
            "verdict":              verdict,
        }

        logger.info(
            "EnsembleWalkForward: verdict=%s  mean_prec=%.3f  std=%.3f  "
            "spd=%.2f  folds=%d/%d",
            verdict, mean_prec, std_prec,
            float(np.mean(spd_vals)), len(folds_with_signals), len(results),
        )

        return {"results": results, "summary": summary}
