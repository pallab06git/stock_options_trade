# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Evaluation framework for sustained-movement models.

Compares multiple trained models (XGBoost, LightGBM, RandomForest) on a
shared test DataFrame that contains both features and the
``target_sustained`` label column.  Produces:

  1. Per-model precision / recall / F1 / signal count across thresholds.
  2. Precision-by-magnitude: for True Positives, what magnitude bucket
     were they in?  And what fraction of signals in each bucket are TPs?
  3. Model agreement: signals predicted by ≥ N models at a given threshold.
  4. TP / FP / FN summary statistics.
  5. An overall comparison report (DataFrame) saved as CSV.

Usage
-----
    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator

    evaluator = SustainedMovementEvaluator(
        thresholds=[0.60, 0.70, 0.80],
        target_col="target_sustained",
    )
    results = evaluator.evaluate(artifacts, test_df)
    report_df = evaluator.generate_report(results)
    evaluator.save_results(results, report_df, output_dir="data/reports/sustained")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.processing.sustained_movement_labeler import MAGNITUDE_BUCKETS

logger = logging.getLogger(__name__)

# Default thresholds to sweep
_DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# ---------------------------------------------------------------------------
# SustainedMovementEvaluator
# ---------------------------------------------------------------------------


class SustainedMovementEvaluator:
    """Evaluate and compare sustained-movement models across thresholds.

    Parameters
    ----------
    thresholds:
        Probability thresholds to evaluate.  Default: 0.50 → 0.90 step 0.05.
    target_col:
        Column name of the binary sustained-movement label.
        Default: ``"target_sustained"``.
    """

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        target_col: str = "target_sustained",
        magnitude_buckets: Optional[List[str]] = None,
    ) -> None:
        self.thresholds = thresholds or _DEFAULT_THRESHOLDS
        self.target_col = target_col
        self._magnitude_buckets = magnitude_buckets or MAGNITUDE_BUCKETS

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        artifacts: Dict[str, Dict[str, Any]],
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Evaluate all models on the test DataFrame.

        Args:
            artifacts: Dict of {model_name → artifact_dict} as returned by
                       ``MultiModelTrainer.train()``.
            test_df:   Test DataFrame with feature columns and
                       ``target_col`` + optionally ``magnitude_bucket``.

        Returns:
            Nested results dict::

                {
                  "models": {
                      "<name>": {
                          "threshold_results": {t: {precision, recall, ...}},
                          "precision_by_magnitude": {...},
                          "signal_predictions": {t: [row_indices]},
                      },
                      ...
                  },
                  "model_agreement": {t: {1_models: [...], 2_models: [...], ...}},
                  "target_col": str,
                  "n_test_rows": int,
                  "n_positives": int,
                  "positive_rate": float,
                  "magnitude_distribution": {bucket: count, ...},
                }
        """
        if self.target_col not in test_df.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not in test_df columns"
            )

        y_true = test_df[self.target_col].values.astype(np.int8)
        n_total = len(test_df)
        n_pos = int(y_true.sum())
        pos_rate = float(n_pos / max(n_total, 1))

        # Magnitude distribution on test set
        mag_dist = {b: 0 for b in self._magnitude_buckets}
        if "magnitude_bucket" in test_df.columns:
            for b in self._magnitude_buckets:
                mag_dist[b] = int((test_df["magnitude_bucket"] == b).sum())

        model_results: Dict[str, Any] = {}
        # Also keep per-threshold signal sets for agreement analysis
        # signal_sets[threshold][model_name] = set of row indices
        signal_sets: Dict[float, Dict[str, Set[int]]] = {
            t: {} for t in self.thresholds
        }

        for name, artifact in artifacts.items():
            model = artifact["model"]
            feature_cols = artifact.get("feature_cols") or []

            # Ensure all feature_cols are in test_df
            available = [c for c in feature_cols if c in test_df.columns]
            if len(available) < len(feature_cols):
                missing = set(feature_cols) - set(available)
                logger.warning(
                    f"SustainedMovementEvaluator: model '{name}' missing "
                    f"{len(missing)} feature(s) — will fill with 0"
                )

            X_test = (
                test_df[available].reindex(columns=feature_cols, fill_value=0.0)
                .fillna(0.0)
                .values.astype(np.float32)
                if feature_cols
                else test_df.fillna(0.0).values.astype(np.float32)
            )

            probas = model.predict_proba(X_test)[:, 1]

            threshold_results: Dict[float, Dict[str, Any]] = {}
            for t in self.thresholds:
                preds = (probas >= t).astype(np.int8)
                tp = int(((preds == 1) & (y_true == 1)).sum())
                fp = int(((preds == 1) & (y_true == 0)).sum())
                fn = int(((preds == 0) & (y_true == 1)).sum())
                tn = int(((preds == 0) & (y_true == 0)).sum())
                n_sig = tp + fp

                prec = float(precision_score(y_true, preds, zero_division=0))
                rec  = float(recall_score(y_true, preds, zero_division=0))
                f1   = float(f1_score(y_true, preds, zero_division=0))
                try:
                    auc = float(roc_auc_score(y_true, probas))
                except Exception:
                    auc = float("nan")

                threshold_results[t] = {
                    "threshold":       t,
                    "n_signals":       n_sig,
                    "true_positives":  tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives":  tn,
                    "precision":       prec,
                    "recall":          rec,
                    "f1":              f1,
                    "roc_auc":         auc,
                    "signal_rate":     float(n_sig / max(n_total, 1)),
                }

                # Track which rows were signalled (for agreement analysis)
                sig_indices = set(np.where(preds == 1)[0].tolist())
                signal_sets[t][name] = sig_indices

            # Precision-by-magnitude analysis
            prec_by_mag = self._precision_by_magnitude(
                test_df, y_true, probas
            )

            model_results[name] = {
                "threshold_results":    threshold_results,
                "precision_by_magnitude": prec_by_mag,
            }

        # Model agreement
        agreement = self._compute_model_agreement(signal_sets)

        return {
            "models":               model_results,
            "model_agreement":      agreement,
            "target_col":           self.target_col,
            "n_test_rows":          n_total,
            "n_positives":          n_pos,
            "positive_rate":        pos_rate,
            "magnitude_distribution": mag_dist,
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        comparison_threshold: float = 0.70,
    ) -> pd.DataFrame:
        """Build a side-by-side comparison DataFrame at a given threshold.

        Args:
            results:              Dict returned by ``evaluate()``.
            comparison_threshold: Threshold to use for the comparison table.
                                  Defaults to 0.70.

        Returns:
            DataFrame with one row per model, columns:
              Model, Threshold, Signals, Precision, Recall, F1, ROC-AUC,
              TP, FP, FN, Signal Rate
        """
        rows = []
        for name, mdata in results.get("models", {}).items():
            t_res = mdata.get("threshold_results", {})
            # Find closest threshold
            closest_t = min(
                t_res.keys(),
                key=lambda t: abs(t - comparison_threshold),
                default=None,
            )
            if closest_t is None:
                continue
            r = t_res[closest_t]
            rows.append(
                {
                    "Model":       name,
                    "Threshold":   r["threshold"],
                    "Signals":     r["n_signals"],
                    "Precision":   round(r["precision"], 4),
                    "Recall":      round(r["recall"], 4),
                    "F1":          round(r["f1"], 4),
                    "ROC-AUC":     round(r["roc_auc"], 4) if not np.isnan(r["roc_auc"]) else None,
                    "TP":          r["true_positives"],
                    "FP":          r["false_positives"],
                    "FN":          r["false_negatives"],
                    "Signal Rate": round(r["signal_rate"], 4),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("Precision", ascending=False).reset_index(drop=True)
        return df

    def save_results(
        self,
        results: Dict[str, Any],
        report_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """Save evaluation results to disk.

        Saves:
          - ``full_results.json``       — complete nested results dict
          - ``comparison_report.csv``   — side-by-side DataFrame
          - ``precision_by_magnitude.json`` — per-model mag breakdown
          - ``model_agreement.json``    — signal overlap stats

        Args:
            results:    Dict returned by ``evaluate()``.
            report_df:  DataFrame returned by ``generate_report()``.
            output_dir: Directory to write files.

        Returns:
            Dict mapping filename stem → saved Path.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}

        # JSON-serialise results (convert float keys to strings)
        def _serialise(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return sorted(obj)
            return str(obj)

        # Convert float-keyed threshold dicts to str-keyed for JSON
        def _normalise_results(d):
            if isinstance(d, dict):
                return {
                    (str(k) if isinstance(k, float) else k): _normalise_results(v)
                    for k, v in d.items()
                }
            return d

        norm = _normalise_results(results)

        p = out_dir / "full_results.json"
        with open(p, "w") as fh:
            json.dump(norm, fh, indent=2, default=_serialise)
        saved["full_results"] = p

        if not report_df.empty:
            p = out_dir / "comparison_report.csv"
            report_df.to_csv(p, index=False)
            saved["comparison_report"] = p

        # Precision by magnitude summary
        mag_summary = {
            name: mdata.get("precision_by_magnitude", {})
            for name, mdata in results.get("models", {}).items()
        }
        p = out_dir / "precision_by_magnitude.json"
        with open(p, "w") as fh:
            json.dump(mag_summary, fh, indent=2, default=_serialise)
        saved["precision_by_magnitude"] = p

        # Model agreement
        p = out_dir / "model_agreement.json"
        with open(p, "w") as fh:
            json.dump(
                _normalise_results(results.get("model_agreement", {})),
                fh, indent=2, default=_serialise,
            )
        saved["model_agreement"] = p

        logger.info(
            f"SustainedMovementEvaluator: saved {len(saved)} file(s) to {out_dir}"
        )
        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precision_by_magnitude(
        self,
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        probas: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute precision broken down by magnitude bucket.

        For each threshold × magnitude bucket, return:
          - n_signals: predicted positives in that bucket
          - n_true_positives: correct predictions in that bucket
          - precision: TP / (TP + FP) in that bucket

        Also returns:
          - tp_magnitude_distribution: for TP signals, which bucket were they in?

        Args:
            test_df: Test DataFrame with optional ``magnitude_bucket`` column.
            y_true:  True label array (length n).
            probas:  Predicted probability array (length n).

        Returns:
            Dict with keys ``"by_threshold"`` and ``"tp_magnitude_distribution"``.
        """
        has_mag = "magnitude_bucket" in test_df.columns

        # TP magnitude distribution (across all thresholds at 0.70 for clarity)
        tp_dist: Dict[str, int] = {b: 0 for b in self._magnitude_buckets}
        if has_mag:
            ref_preds = (probas >= 0.70).astype(np.int8)
            tp_mask = (ref_preds == 1) & (y_true == 1)
            for b in self._magnitude_buckets:
                bucket_mask = test_df["magnitude_bucket"] == b
                tp_dist[b] = int((tp_mask & bucket_mask.values).sum())

        by_threshold: Dict[float, Dict[str, Dict[str, Any]]] = {}
        for t in self.thresholds:
            preds = (probas >= t).astype(np.int8)
            bucket_stats: Dict[str, Dict[str, Any]] = {}

            for b in self._magnitude_buckets:
                if has_mag:
                    bucket_mask = (test_df["magnitude_bucket"] == b).values
                    bucket_preds = preds[bucket_mask]
                    bucket_y     = y_true[bucket_mask]
                else:
                    bucket_stats[b] = {"n_signals": 0, "n_tp": 0, "precision": 0.0}
                    continue

                n_sig = int(bucket_preds.sum())
                n_tp  = int(((bucket_preds == 1) & (bucket_y == 1)).sum())
                prec  = float(n_tp / max(n_sig, 1)) if n_sig > 0 else 0.0

                bucket_stats[b] = {
                    "n_signals": n_sig,
                    "n_tp":      n_tp,
                    "precision": prec,
                }

            by_threshold[t] = bucket_stats

        return {
            "by_threshold":              by_threshold,
            "tp_magnitude_distribution": tp_dist,
        }

    def _compute_model_agreement(
        self,
        signal_sets: Dict[float, Dict[str, Set[int]]],
    ) -> Dict[float, Dict[str, Any]]:
        """Find row indices where ≥ N models agree at each threshold.

        Args:
            signal_sets: {threshold → {model_name → set of row indices}}.

        Returns:
            {threshold → {"n_models_compared": int,
                           "total_unique_signals": int,
                           "agreement_breakdown": {"1_models": count, ...},
                           "all_agree_count": int,
                           "majority_agree_count": int}}
        """
        agreement: Dict[float, Dict[str, Any]] = {}

        for t, model_signals in signal_sets.items():
            if not model_signals:
                agreement[t] = {
                    "n_models_compared":     0,
                    "total_unique_signals":  0,
                    "agreement_breakdown":   {},
                    "all_agree_count":       0,
                    "majority_agree_count":  0,
                }
                continue

            n_models = len(model_signals)
            all_row_idxs: Set[int] = set()
            for s in model_signals.values():
                all_row_idxs |= s

            total_unique = len(all_row_idxs)
            majority_threshold = (n_models // 2) + 1

            breakdown: Dict[str, int] = {}
            all_agree = 0
            majority_agree = 0

            for idx in all_row_idxs:
                count = sum(1 for s in model_signals.values() if idx in s)
                key = f"{count}_models"
                breakdown[key] = breakdown.get(key, 0) + 1
                if count == n_models:
                    all_agree += 1
                if count >= majority_threshold:
                    majority_agree += 1

            agreement[t] = {
                "n_models_compared":    n_models,
                "total_unique_signals": total_unique,
                "agreement_breakdown":  breakdown,
                "all_agree_count":      all_agree,
                "majority_agree_count": majority_agree,
            }

        return agreement

    # ------------------------------------------------------------------
    # Direction-split analysis
    # ------------------------------------------------------------------

    def analyze_precision_by_current_direction(
        self,
        model_name: str,
        model,
        feature_cols: List[str],
        test_df: pd.DataFrame,
        threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """Compare model precision when price is currently rising vs falling.

        Splits signals (predicted positives at ``threshold``) into two groups:

        - **Rising**: ``opt_return_1m > 0`` AND ``opt_return_5m > 0``
        - **Falling**: ``opt_return_1m < 0`` OR  ``opt_return_5m < 0``

        For each group reports: total signals, TP, FP, precision, and the
        magnitude-bucket breakdown so we can see where ``below_zero`` signals
        concentrate.

        Args:
            model_name:   Human-readable name used in printed output.
            model:        Trained sklearn-compatible model with ``predict_proba``.
            feature_cols: Feature column names matching the model's training order.
            test_df:      Test DataFrame; must contain ``opt_return_1m``,
                          ``opt_return_5m``, ``target_sustained`` (or the
                          configured ``target_col``), and optionally
                          ``magnitude_bucket``.
            threshold:    Probability threshold for firing a signal. Default 0.70.

        Returns:
            Dict with keys ``"rising"``, ``"falling"``, and ``"improvement"``::

                {
                  "rising":  {"total_signals": int, "precision": float,
                               "tp": int, "fp": int},
                  "falling": {"total_signals": int, "precision": float,
                               "tp": int, "fp": int},
                  "improvement": float,   # rising_precision - falling_precision
                }
        """
        # --- Build feature matrix ------------------------------------------------
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(np.float32)
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(np.int8)
        y_true  = test_df[self.target_col].values.astype(np.int8)

        # --- Direction mask ------------------------------------------------------
        r1m = test_df.get("opt_return_1m", pd.Series(0.0, index=test_df.index))
        r5m = test_df.get("opt_return_5m", pd.Series(0.0, index=test_df.index))
        currently_rising = ((r1m > 0) & (r5m > 0)).values

        rising_signals  = (y_pred == 1) & currently_rising
        falling_signals = (y_pred == 1) & (~currently_rising)

        has_mag = "magnitude_bucket" in test_df.columns

        def _bucket_breakdown(mask: np.ndarray) -> Dict[str, Dict[str, Any]]:
            total = int(mask.sum())
            rows: Dict[str, Dict[str, Any]] = {}
            for b in self._magnitude_buckets:
                if has_mag:
                    bucket_mask = (test_df["magnitude_bucket"] == b).values
                    count = int((mask & bucket_mask).sum())
                else:
                    count = 0
                rows[b] = {"count": count, "pct": count / max(total, 1) * 100}
            return rows

        def _stats(mask: np.ndarray) -> Dict[str, Any]:
            tp    = int(((mask) & (y_true == 1)).sum())
            fp    = int(((mask) & (y_true == 0)).sum())
            total = tp + fp
            prec  = tp / max(total, 1)
            return {"total_signals": total, "precision": prec, "tp": tp, "fp": fp}

        rising_stats   = _stats(rising_signals)
        falling_stats  = _stats(falling_signals)
        rising_buckets = _bucket_breakdown(rising_signals)
        falling_buckets = _bucket_breakdown(falling_signals)

        improvement = rising_stats["precision"] - falling_stats["precision"]

        # --- Print report --------------------------------------------------------
        w = 80
        click.echo(f"\n{'=' * w}")
        click.echo(f"  PRECISION BY CURRENT DIRECTION: {model_name}  (threshold={threshold})")
        click.echo(f"{'=' * w}")

        for label, stats, buckets in [
            ("RISING  (1m>0 AND 5m>0)", rising_stats,  rising_buckets),
            ("FALLING (1m<0 OR  5m<0)", falling_stats, falling_buckets),
        ]:
            arrow = "^" if "RISING" in label else "v"
            click.echo(f"\n[{arrow}] CURRENTLY {label}")
            click.echo(f"    Total signals : {stats['total_signals']:>8,}")
            click.echo(f"    True positives: {stats['tp']:>8,}")
            click.echo(f"    False positives:{stats['fp']:>8,}")
            click.echo(f"    Precision     : {stats['precision']:>8.1%}")
            if has_mag:
                click.echo(f"    Magnitude breakdown:")
                for b, bdata in buckets.items():
                    click.echo(
                        f"      {b:<12}  {bdata['count']:>6,}  ({bdata['pct']:>5.1f}%)"
                    )

        click.echo(f"\n{'=' * w}")
        click.echo("  SUMMARY COMPARISON")
        click.echo(f"{'=' * w}")
        click.echo(f"\n  {'Metric':<30} {'Rising':>12} {'Falling':>12}")
        click.echo(f"  {'-' * 54}")
        click.echo(
            f"  {'Total signals':<30} "
            f"{rising_stats['total_signals']:>12,} "
            f"{falling_stats['total_signals']:>12,}"
        )
        click.echo(
            f"  {'Precision':<30} "
            f"{rising_stats['precision']:>12.1%} "
            f"{falling_stats['precision']:>12.1%}"
        )
        if has_mag:
            rz = rising_buckets.get("below_zero", {}).get("count", 0)
            fz = falling_buckets.get("below_zero", {}).get("count", 0)
            click.echo(f"  {'Below-zero count':<30} {rz:>12,} {fz:>12,}")
        click.echo(
            f"\n  Precision improvement when RISING: {improvement:+.1%}"
        )
        click.echo(f"{'=' * w}\n")

        return {
            "rising":      {**rising_stats,  "magnitude_breakdown": rising_buckets},
            "falling":     {**falling_stats, "magnitude_breakdown": falling_buckets},
            "improvement": float(improvement),
        }

    # ------------------------------------------------------------------
    # Consolidation-breakout analysis
    # ------------------------------------------------------------------

    def analyze_precision_by_consolidation(
        self,
        model_name: str,
        model,
        feature_cols: List[str],
        test_df: pd.DataFrame,
        threshold: float = 0.70,
        consol_5m_pct: float = 1.0,
        consol_15m_pct: float = 2.0,
    ) -> Dict[str, Any]:
        """Test whether signals during consolidation have better precision.

        Hypothesis: breakouts from tight trading ranges are more predictable
        than moves that happen during already-volatile periods.

        Consolidation is defined as:
          |opt_return_5m| < ``consol_5m_pct``  AND
          |opt_return_15m| < ``consol_15m_pct``

        Falls back to ``opt_return_5m`` alone if 15m is absent, or to
        ``opt_return_1m`` if neither is available.

        Args:
            model_name:      Human-readable label used in printed output.
            model:           Trained sklearn-compatible model with ``predict_proba``.
            feature_cols:    Feature column names matching the model's training order.
            test_df:         Test DataFrame; must contain ``target_col`` and
                             optionally ``magnitude_bucket``, ``opt_return_5m``,
                             ``opt_return_15m``.
            threshold:       Probability threshold for firing a signal.  Default 0.70.
            consol_5m_pct:   Max absolute 5-minute return to count as consolidating.
                             Default 1.0 (%).
            consol_15m_pct:  Max absolute 15-minute return to count as consolidating.
                             Default 2.0 (%).

        Returns:
            Dict with keys ``"consolidating"``, ``"non_consolidating"``,
            ``"improvement"``, and ``"hypothesis_confirmed"``::

                {
                  "consolidating":     {"total_signals", "precision", "tp", "fp"},
                  "non_consolidating": {"total_signals", "precision", "tp", "fp"},
                  "improvement":       float,   # consol_prec - non_consol_prec
                  "hypothesis_confirmed": bool, # True if improvement > 10 pp
                }
        """
        # --- Build feature matrix ------------------------------------------------
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(np.float32)
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(np.int8)
        y_true  = test_df[self.target_col].values.astype(np.int8)

        # --- Consolidation mask --------------------------------------------------
        has_5m  = "opt_return_5m"  in test_df.columns
        has_15m = "opt_return_15m" in test_df.columns

        if has_5m and has_15m:
            defn = (
                f"|opt_return_5m| < {consol_5m_pct}%  AND  "
                f"|opt_return_15m| < {consol_15m_pct}%"
            )
            consol_mask = (
                (test_df["opt_return_5m"].abs()  < consol_5m_pct) &
                (test_df["opt_return_15m"].abs() < consol_15m_pct)
            ).values
        elif has_5m:
            defn = f"|opt_return_5m| < {consol_5m_pct}%  (15m not available)"
            consol_mask = (test_df["opt_return_5m"].abs() < consol_5m_pct).values
        else:
            defn = "|opt_return_1m| < 0.5%  (5m/15m not available)"
            r1m  = test_df.get("opt_return_1m", pd.Series(0.0, index=test_df.index))
            consol_mask = (r1m.abs() < 0.5).values

        consol_signals     = (y_pred == 1) & consol_mask
        non_consol_signals = (y_pred == 1) & (~consol_mask)

        has_mag = "magnitude_bucket" in test_df.columns

        def _stats(mask: np.ndarray) -> Dict[str, Any]:
            tp    = int(((mask) & (y_true == 1)).sum())
            fp    = int(((mask) & (y_true == 0)).sum())
            total = tp + fp
            prec  = tp / max(total, 1)
            return {"total_signals": total, "precision": prec, "tp": tp, "fp": fp}

        def _bucket_breakdown(mask: np.ndarray) -> Dict[str, Dict[str, Any]]:
            total = int(mask.sum())
            rows: Dict[str, Dict[str, Any]] = {}
            for b in self._magnitude_buckets:
                if has_mag:
                    count = int((mask & (test_df["magnitude_bucket"] == b).values).sum())
                else:
                    count = 0
                rows[b] = {"count": count, "pct": count / max(total, 1) * 100}
            return rows

        consol_stats      = _stats(consol_signals)
        non_consol_stats  = _stats(non_consol_signals)
        consol_buckets    = _bucket_breakdown(consol_signals)
        non_consol_buckets = _bucket_breakdown(non_consol_signals)

        improvement        = consol_stats["precision"] - non_consol_stats["precision"]
        hypothesis_confirmed = improvement > 0.10

        # --- Print report --------------------------------------------------------
        w = 80
        click.echo(f"\n{'=' * w}")
        click.echo(f"  CONSOLIDATION -> BREAKOUT HYPOTHESIS TEST: {model_name}  (threshold={threshold})")
        click.echo(f"{'=' * w}")

        for label, stats, buckets, defn_str in [
            ("CONSOLIDATING   (tight range, small recent moves)", consol_stats,
             consol_buckets,
             f"  Definition: {defn}"),
            ("NON-CONSOLIDATING (already volatile, large recent moves)", non_consol_stats,
             non_consol_buckets,
             f"  Definition: NOT ({defn})"),
        ]:
            click.echo(f"\n[~] {label}")
            click.echo(defn_str)
            click.echo(f"    Total signals : {stats['total_signals']:>8,}")
            click.echo(f"    True positives: {stats['tp']:>8,}")
            click.echo(f"    False positives:{stats['fp']:>8,}")
            click.echo(f"    Precision     : {stats['precision']:>8.1%}")
            if has_mag:
                click.echo("    Magnitude breakdown:")
                for b, bdata in buckets.items():
                    click.echo(
                        f"      {b:<12}  {bdata['count']:>6,}  ({bdata['pct']:>5.1f}%)"
                    )

        click.echo(f"\n{'=' * w}")
        click.echo("  HYPOTHESIS TEST RESULT")
        click.echo(f"{'=' * w}")
        click.echo(
            f"\n  Precision during CONSOLIDATION:    {consol_stats['precision']:>6.1%}"
        )
        click.echo(
            f"  Precision during HIGH VOLATILITY:  {non_consol_stats['precision']:>6.1%}"
        )
        click.echo(f"  Difference:                        {improvement:>+6.1%}")

        if hypothesis_confirmed:
            click.echo(
                f"\n  RESULT: STRONG consolidation pattern confirmed."
                f"  Breakouts from tight ranges have {improvement:.1%} better precision."
                f"\n  RECOMMENDATION: Add consolidation features and retrain."
            )
        elif improvement > 0.05:
            click.echo(
                f"\n  RESULT: Moderate consolidation effect ({improvement:.1%})."
                f"  May be worth adding consolidation features."
            )
        else:
            click.echo(
                f"\n  RESULT: No significant consolidation pattern (diff={improvement:+.1%})."
                f"  Consolidation filtering would not meaningfully improve results."
            )
        click.echo(f"{'=' * w}\n")

        return {
            "consolidating": {
                **consol_stats,
                "magnitude_breakdown": consol_buckets,
            },
            "non_consolidating": {
                **non_consol_stats,
                "magnitude_breakdown": non_consol_buckets,
            },
            "improvement":          float(improvement),
            "hypothesis_confirmed": bool(hypothesis_confirmed),
        }

    # ------------------------------------------------------------------
    # Consolidation parameter sweep
    # ------------------------------------------------------------------

    def analyze_consolidation_parameter_sweep(
        self,
        model_name: str,
        model,
        feature_cols: List[str],
        test_df: pd.DataFrame,
        threshold: float = 0.70,
        min_signals: int = 10,
    ) -> Dict[str, Any]:
        """Sweep window × tightness-threshold combinations to find optimal consolidation.

        Tests all combinations of:
          - Windows : 3, 5, 7, 10, 15, 20 minutes
          - Tightness: price_range < 0.5%, 1.0%, 1.5%, 2.0%, 3.0%

        For each combination that produces at least ``min_signals`` signals,
        reports precision, signal count, and 20%+ moves.

        Args:
            model_name:   Label for printed output.
            model:        Trained sklearn-compatible model.
            feature_cols: Feature columns matching model training order.
            test_df:      Test DataFrame with ``target_col``, ``price_range_{N}m``
                          cols, and optionally ``magnitude_bucket``.
            threshold:    Confidence threshold for firing a signal.
            min_signals:  Skip combinations with fewer signals than this.

        Returns:
            Dict with ``"all_results"`` (list of dicts), ``"best"`` (top row),
            and ``"top5"`` (top-5 by precision).
        """
        # --- Predictions ---------------------------------------------------------
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(np.float32)
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(np.int8)
        y_true  = test_df[self.target_col].values.astype(np.int8)
        has_mag = "magnitude_bucket" in test_df.columns

        windows    = [3, 5, 7, 10, 15, 20]
        thr_values = [0.5, 1.0, 1.5, 2.0, 3.0]
        results: List[Dict[str, Any]] = []

        click.echo(f"\n{'=' * 80}")
        click.echo(f"  CONSOLIDATION PARAMETER SWEEP: {model_name}  (signal threshold={threshold})")
        click.echo(f"{'=' * 80}")
        click.echo(
            f"\n  {'Window':<8} {'RngThr%':<10} {'Signals':>8} "
            f"{'Precision':>10} {'20%+ Count':>12} {'20%+ %':>8}"
        )
        click.echo(f"  {'-' * 58}")

        for w in windows:
            col = f"price_range_{w}m"
            if col not in test_df.columns:
                continue
            for thr_pct in thr_values:
                consol_mask    = (test_df[col] < thr_pct).values
                consol_signals = (y_pred == 1) & consol_mask

                tp    = int(((consol_signals) & (y_true == 1)).sum())
                fp    = int(((consol_signals) & (y_true == 0)).sum())
                total = tp + fp
                if total < min_signals:
                    continue

                prec = tp / total
                count_20p = 0
                pct_20p   = 0.0
                if has_mag:
                    count_20p = int(
                        (consol_signals & (test_df["magnitude_bucket"] == "20%+").values).sum()
                    )
                    pct_20p = count_20p / max(total, 1)

                click.echo(
                    f"  {w}m{'':<6} <{thr_pct}%{'':<5} {total:>8,} "
                    f"{prec:>10.1%} {count_20p:>12,} {pct_20p:>8.1%}"
                )
                results.append(
                    {
                        "window_m":       w,
                        "range_thr_pct":  thr_pct,
                        "signals":        total,
                        "precision":      float(prec),
                        "tp":             tp,
                        "fp":             fp,
                        "count_20plus":   count_20p,
                        "pct_20plus":     float(pct_20p),
                    }
                )

        if not results:
            click.echo("\n  No combinations produced enough signals.")
            return {"all_results": [], "best": None, "top5": []}

        results_df = pd.DataFrame(results).sort_values("precision", ascending=False)
        top5       = results_df.head(5).to_dict("records")
        best       = top5[0]

        click.echo(f"\n{'=' * 80}")
        click.echo("  TOP 5 BY PRECISION")
        click.echo(f"{'=' * 80}")
        for i, row in enumerate(top5, 1):
            click.echo(
                f"  #{i}  window={row['window_m']}m  range<{row['range_thr_pct']}%"
                f"  signals={row['signals']:,}  precision={row['precision']:.1%}"
                f"  20%+={row['count_20plus']:,}"
            )

        click.echo(f"\n{'=' * 80}")
        click.echo("  OPTIMAL CONSOLIDATION DEFINITION")
        click.echo(f"{'=' * 80}")
        click.echo(f"\n  Window    : {best['window_m']} minutes")
        click.echo(f"  Threshold : price_range < {best['range_thr_pct']}%")
        click.echo(f"  Signals   : {best['signals']:,}")
        click.echo(f"  Precision : {best['precision']:.1%}")
        click.echo(f"  20%+ moves: {best['count_20plus']:,}  ({best['pct_20plus']:.1%})")
        click.echo(f"{'=' * 80}\n")

        return {"all_results": results, "best": best, "top5": top5}

    # ------------------------------------------------------------------
    # Leakage verification helpers
    # ------------------------------------------------------------------

    def test_on_random_data(
        self,
        model,
        feature_cols: List[str],
        n_samples: int = 10_000,
    ) -> Dict[str, Any]:
        """Leakage check: run the model on pure Gaussian noise.

        If the model generates many high-confidence (≥ 70 %) signals on
        completely random data it likely absorbed spurious patterns from
        future-leaking features.

        Args:
            model:        Trained sklearn-compatible model with ``predict_proba``.
            feature_cols: Feature column names (used to size the noise matrix).
            n_samples:    Number of random rows to generate.  Default: 10_000.

        Returns:
            Dict with keys:
              - ``n_samples``           int
              - ``n_features``          int
              - ``high_conf_signals``   int   (predicted proba ≥ 0.70)
              - ``high_conf_rate``      float
              - ``mean_proba``          float
              - ``max_proba``           float
              - ``leakage_suspected``   bool  (True if > 100 high-conf signals)
              - ``verdict``             str   human-readable summary
        """
        n_features = len(feature_cols)
        rng = np.random.RandomState(42)
        X_noise = rng.randn(n_samples, n_features).astype(np.float32)

        probas = model.predict_proba(X_noise)[:, 1]
        high_conf = int((probas >= 0.70).sum())
        high_conf_rate = float(high_conf / max(n_samples, 1))
        mean_proba = float(probas.mean())
        max_proba = float(probas.max())

        leakage = high_conf > 100

        if leakage:
            verdict = (
                f"LEAKAGE SUSPECTED: {high_conf:,} high-confidence signals on random noise "
                f"({high_conf_rate:.2%} of {n_samples:,} rows).  "
                f"Mean proba={mean_proba:.4f}, max proba={max_proba:.4f}."
            )
        else:
            verdict = (
                f"OK: only {high_conf} high-confidence signals on random noise "
                f"({high_conf_rate:.4%} of {n_samples:,} rows).  "
                f"Mean proba={mean_proba:.4f}, max proba={max_proba:.4f}."
            )

        logger.info(f"test_on_random_data: {verdict}")
        return {
            "n_samples":         n_samples,
            "n_features":        n_features,
            "high_conf_signals": high_conf,
            "high_conf_rate":    high_conf_rate,
            "mean_proba":        mean_proba,
            "max_proba":         max_proba,
            "leakage_suspected": leakage,
            "verdict":           verdict,
        }

    def evaluate_split_models(
        self,
        call_artifact: Dict[str, Any],
        put_artifact: Dict[str, Any],
        test_df: pd.DataFrame,
        threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """Evaluate separately-trained call and put models and combine results.

        Each model is evaluated only on its corresponding option-type subset of
        ``test_df`` (``contract_type == 1`` for calls, ``== 0`` for puts), then
        results are merged to give a combined precision figure.

        Args:
            call_artifact:  Artifact dict for the CALL model (from
                            ``train_call_put_models_separately``).
            put_artifact:   Artifact dict for the PUT model.
            test_df:        Full test DataFrame containing ``contract_type``
                            (1=call, 0=put), ``target_col``, and optionally
                            ``magnitude_bucket``.
            threshold:      Probability threshold for firing a signal.

        Returns:
            Dict with per-type and combined precision/recall/TP/FP metrics plus
            an ``improvement_vs_mixed`` float (pp above 34.3% baseline).
        """
        SEP = "=" * 70

        call_model     = call_artifact["model"]
        put_model      = put_artifact["model"]
        call_feat_cols = call_artifact["feature_cols"]
        put_feat_cols  = put_artifact["feature_cols"]

        # Split test data
        test_calls = test_df[test_df["contract_type"] == 1].reset_index(drop=True)
        test_puts  = test_df[test_df["contract_type"] == 0].reset_index(drop=True)

        print(f"\n{SEP}")
        print(f"  SPLIT-MODEL EVALUATION  (threshold={threshold:.0%})")
        print(SEP)
        print(f"\n  Test CALL rows : {len(test_calls):,}  "
              f"positives={int(test_calls[self.target_col].sum())}")
        print(f"  Test PUT  rows : {len(test_puts):,}  "
              f"positives={int(test_puts[self.target_col].sum())}")

        def _eval_type(df_type, feat_cols, model):
            avail = [c for c in feat_cols if c in df_type.columns]
            X = (
                df_type[avail]
                .reindex(columns=feat_cols, fill_value=0.0)
                .fillna(0.0)
                .values.astype(np.float32)
            )
            y_true  = df_type[self.target_col].values
            y_proba = model.predict_proba(X)[:, 1]
            y_pred  = (y_proba >= threshold).astype(int)

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Magnitude breakdown
            mag_stats: Dict[str, Any] = {}
            if "magnitude_bucket" in df_type.columns:
                for b in self._magnitude_buckets:
                    sig_b = (y_pred == 1) & (df_type["magnitude_bucket"] == b).values
                    n_sig = int(sig_b.sum())
                    n_tp  = int((sig_b & (y_true == 1)).sum())
                    mag_stats[b] = {
                        "n_signals":  n_sig,
                        "n_tp":       n_tp,
                        "precision":  n_tp / n_sig if n_sig > 0 else 0.0,
                    }

            return {
                "signals":       tp + fp,
                "tp":            tp,
                "fp":            fp,
                "fn":            fn,
                "precision":     float(prec),
                "recall":        float(recall),
                "magnitude":     mag_stats,
            }

        call_res = _eval_type(test_calls, call_feat_cols, call_model)
        put_res  = _eval_type(test_puts,  put_feat_cols,  put_model)

        # Combined
        total_tp  = call_res["tp"] + put_res["tp"]
        total_fp  = call_res["fp"] + put_res["fp"]
        total_sig = total_tp + total_fp
        combined_prec = total_tp / total_sig if total_sig > 0 else 0.0

        _BASELINE = 0.343  # mixed-model XGBoost precision (Step 56 Enhanced)
        improvement = combined_prec - _BASELINE

        print(f"\n  CALL model   precision={call_res['precision']:.1%}  "
              f"signals={call_res['signals']:,}  TP={call_res['tp']:,}  FP={call_res['fp']:,}")
        print(f"  PUT  model   precision={put_res['precision']:.1%}  "
              f"signals={put_res['signals']:,}  TP={put_res['tp']:,}  FP={put_res['fp']:,}")
        print(f"\n  COMBINED     precision={combined_prec:.1%}  "
              f"signals={total_sig:,}  TP={total_tp:,}  FP={total_fp:,}")
        print(f"  vs baseline  {improvement:+.1%}  (baseline={_BASELINE:.1%})")

        # Magnitude breakdown for below_zero
        print(f"\n  MAGNITUDE BREAKDOWN:")
        for label, res in [("CALL", call_res), ("PUT", put_res)]:
            if res["magnitude"]:
                bz = res["magnitude"].get("below_zero", {})
                print(f"    {label}  below_zero: "
                      f"{bz.get('n_signals',0):,} signals  "
                      f"{bz.get('precision',0):.1%} precision")

        print()

        return {
            "call":                 call_res,
            "put":                  put_res,
            "combined_precision":   float(combined_prec),
            "combined_signals":     total_sig,
            "combined_tp":          total_tp,
            "combined_fp":          total_fp,
            "improvement_vs_mixed": float(improvement),
            "threshold":            threshold,
        }

    def analyze_signals_by_option_type(
        self,
        model_name: str,
        model,
        feature_cols: List[str],
        test_df: pd.DataFrame,
        threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """Analyze whether model signals equally on calls and puts.

        If the dataset contains both calls and puts and the model fires on both
        equally, ~50% of signals will be wrong by construction when the
        underlying moves (calls succeed on up-moves, puts succeed on down-moves).
        This would explain persistently poor precision regardless of feature set.

        Args:
            model_name:   Human-readable label for printed output.
            model:        Trained sklearn-compatible model with ``predict_proba``.
            feature_cols: Feature columns matching the model's training order.
            test_df:      Test DataFrame; must contain ``target_col`` and ideally
                          ``contract_type`` ('C'/'P') or ``ticker`` column.
            threshold:    Probability threshold for a signal.  Default 0.70.

        Returns:
            Dict with ``call_signals``, ``put_signals``, ``call_precision``,
            ``put_precision``, ``call_pct``, ``put_pct``, and ``below_zero_by_type``.
        """
        # --- Build feature matrix ---------------------------------------------------
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(np.float32)
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        y_true = test_df[self.target_col].values

        SEP = "=" * 80
        print(f"\n{SEP}")
        print(f"  CALL/PUT SIGNAL DISTRIBUTION: {model_name}  (threshold={threshold})")
        print(SEP)

        # --- Infer/normalise contract_type ------------------------------------------
        df = test_df.copy()
        if "contract_type" not in df.columns:
            if "ticker" in df.columns:
                extracted = df["ticker"].str.extract(r"\d([CP])\d", expand=False)
                df["contract_type"] = extracted
            else:
                print("\n  No 'contract_type' or 'ticker' column found.")
                print(f"  Available columns: {df.columns.tolist()[:20]}")
                return {"error": "no_contract_type_column"}

        # Normalize: if numeric (1=call, 0=put) convert to 'C'/'P'
        ct = df["contract_type"]
        if pd.api.types.is_numeric_dtype(ct):
            df["contract_type"] = ct.map({1: "C", 0: "P"})

        total_signals = int(y_pred.sum())
        print(f"\n  Total signals: {total_signals:,}")

        if total_signals == 0:
            print("  No signals at this threshold.")
            return {"error": "no_signals"}

        # --- Per-type counts --------------------------------------------------------
        sig_mask = y_pred == 1
        call_mask = sig_mask & (df["contract_type"] == "C")
        put_mask  = sig_mask & (df["contract_type"] == "P")

        call_signals = int(call_mask.sum())
        put_signals  = int(put_mask.sum())
        call_pct = call_signals / total_signals * 100
        put_pct  = put_signals  / total_signals * 100

        print(f"\n  CALL signals : {call_signals:,}  ({call_pct:.1f}%)")
        print(f"  PUT  signals : {put_signals:,}  ({put_pct:.1f}%)")

        # --- Precision by type ------------------------------------------------------
        def _prec(mask):
            tp = int(((mask) & (y_true == 1)).sum())
            fp = int(((mask) & (y_true == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return tp, fp, prec

        call_tp, call_fp, call_prec = _prec(call_mask)
        put_tp,  put_fp,  put_prec  = _prec(put_mask)

        print(f"\n  PRECISION BY OPTION TYPE:")
        print(f"    CALL  precision={call_prec:.1%}  TP={call_tp:,}  FP={call_fp:,}")
        print(f"    PUT   precision={put_prec:.1%}  TP={put_tp:,}  FP={put_fp:,}")

        # --- Below-zero breakdown by type -------------------------------------------
        has_mag = "magnitude_bucket" in df.columns
        below_zero_call = below_zero_put = 0
        if has_mag:
            below_zero_call = int((call_mask & (df["magnitude_bucket"] == "below_zero")).sum())
            below_zero_put  = int((put_mask  & (df["magnitude_bucket"] == "below_zero")).sum())
            call_bz_pct = below_zero_call / call_signals * 100 if call_signals > 0 else 0.0
            put_bz_pct  = below_zero_put  / put_signals  * 100 if put_signals  > 0 else 0.0
            print(f"\n  BELOW-ZERO SIGNALS BY TYPE:")
            print(f"    CALL  below-zero: {below_zero_call:,}  ({call_bz_pct:.1f}% of CALL signals)")
            print(f"    PUT   below-zero: {below_zero_put:,}  ({put_bz_pct:.1f}% of PUT signals)")

        # --- Key insight ------------------------------------------------------------
        print(f"\n{SEP}")
        print("  KEY INSIGHT")
        print(SEP)

        if abs(call_pct - 50) < 15 and call_signals > 100 and put_signals > 100:
            print(f"\n  *** MODEL SIGNALS ~EQUALLY ON CALLS AND PUTS ({call_pct:.0f}% / {put_pct:.0f}%) ***")
            print("  This is the root cause of ~50% precision!")
            print("  When SPY rises  → CALL signals succeed, PUT signals fail.")
            print("  When SPY falls  → PUT signals succeed, CALL signals fail.")
            print("  FIX: add is_call / is_put feature so model knows option type.")
        elif call_pct > 80:
            print(f"\n  Model is {call_pct:.0f}% CALL signals — call/put confusion NOT the issue.")
            print("  The problem is predicting CALL price direction.")
        elif put_pct > 80:
            print(f"\n  Model is {put_pct:.0f}% PUT signals — call/put confusion NOT the issue.")
            print("  The problem is predicting PUT price direction.")
        else:
            print(f"\n  Mixed signal: {call_pct:.0f}% calls / {put_pct:.0f}% puts.")
            print("  Partial call/put confusion may be contributing.")

        print()

        return {
            "call_signals":    call_signals,
            "put_signals":     put_signals,
            "call_precision":  float(call_prec),
            "put_precision":   float(put_prec),
            "call_pct":        float(call_pct),
            "put_pct":         float(put_pct),
            "below_zero_by_type": {
                "call": below_zero_call,
                "put":  below_zero_put,
            },
        }

    def print_feature_importance(
        self,
        model,
        feature_cols: List[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Print a ranked feature importance table.

        Works with any model exposing ``feature_importances_``
        (XGBoost, LightGBM, RandomForest).

        Args:
            model:        Trained model with ``feature_importances_`` attribute.
            feature_cols: Feature column names (must match model's training order).
            top_n:        Number of top features to show.  Default: 20.

        Returns:
            DataFrame with columns [Rank, Feature, Importance] sorted DESC.
            Returns empty DataFrame if the model has no ``feature_importances_``.
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning(
                "print_feature_importance: model has no feature_importances_ attribute"
            )
            return pd.DataFrame()

        importances = model.feature_importances_
        cols = list(feature_cols)
        if len(importances) != len(cols):
            logger.warning(
                f"print_feature_importance: importances length {len(importances)} "
                f"!= feature_cols length {len(cols)} — truncating to min"
            )
            n = min(len(importances), len(cols))
            importances = importances[:n]
            cols = cols[:n]

        df = (
            pd.DataFrame({"Feature": cols, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------

    def comprehensive_threshold_sweep(
        self,
        artifact: Dict[str, Any],
        test_df: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
        position_size_usd: float = 12500.0,
        stop_loss_pct: float = -10.0,
    ) -> Dict[str, Any]:
        """Test a model at multiple confidence thresholds.

        For each threshold, computes signals/day, precision, TP/FP counts,
        and a rough monthly profit estimate.  Identifies the optimal threshold
        where precision ≥ 50% and signals/day are in the 10–50 range.

        Args:
            artifact:          Model artifact dict (keys: ``model``,
                               ``feature_cols``, ``target_col``,
                               optionally ``model_name``, ``option_type``).
            test_df:           Test DataFrame with feature columns,
                               ``self.target_col``, ``date``, and optionally
                               ``magnitude_bucket``.
            thresholds:        Probability thresholds to test.
                               Defaults to 0.70 → 0.95 in 0.05 steps.
            position_size_usd: Notional per trade for monthly profit estimate.
            stop_loss_pct:     Assumed FP loss % (e.g. -10 for a −10% stop).

        Returns:
            {
                'results':  [list of per-threshold dicts],
                'optimal':  best viable dict (or None),
                'model_name': str,
                'test_days': int,
            }
        """
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        clf          = artifact["model"]
        feature_cols = artifact.get("feature_cols") or []
        model_name   = artifact.get("model_name") or artifact.get("model_type", "model")
        option_type  = artifact.get("option_type", "mixed")

        # Build X_test
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(np.float32)
        )

        y_true  = test_df[self.target_col].values.astype(np.int8)
        y_proba = clf.predict_proba(X_test)[:, 1]

        has_mag = "magnitude_bucket" in test_df.columns

        test_dates = sorted(test_df["date"].unique()) if "date" in test_df.columns else []
        test_days  = len(test_dates)
        date_range = (
            f"{test_dates[0]} to {test_dates[-1]}" if test_dates else "unknown"
        )

        # Magnitude → midpoint gain % for monthly profit estimate
        _MAG_MIDPOINT = {
            "below_zero": 0.0,
            "0-1%":       0.5,
            "1-5%":       3.0,
            "5-10%":      7.5,
            "10-20%":    15.0,
            "20%+":      30.0,
        }

        print(f"\n{'='*72}")
        print(f"  THRESHOLD SWEEP: {model_name}  [{option_type}]")
        print(f"  Test period: {date_range} ({test_days} days)")
        print(f"  Baseline positive rate: {y_true.mean():.2%}")
        print(f"{'='*72}")
        print(
            f"\n  {'Thresh':>7}  {'Signals':>8}  {'Per Day':>8}  "
            f"{'Prec':>8}  {'TP':>6}  {'FP':>6}  {'Monthly $':>12}"
        )
        print("  " + "-" * 64)

        results: List[Dict[str, Any]] = []

        for t in thresholds:
            preds  = (y_proba >= t).astype(np.int8)
            tp     = int(((preds == 1) & (y_true == 1)).sum())
            fp     = int(((preds == 1) & (y_true == 0)).sum())
            fn     = int(((preds == 0) & (y_true == 1)).sum())
            n_sig  = tp + fp
            prec   = tp / max(n_sig, 1)
            recall = tp / max(tp + fn, 1)
            spd    = n_sig / max(test_days, 1)

            # Magnitude breakdown for signals
            mag_breakdown: Dict[str, Dict[str, Any]] = {}
            if has_mag and n_sig > 0:
                sig_mask = preds == 1
                sig_df   = test_df[sig_mask]
                for b in self._magnitude_buckets:
                    cnt = int((sig_df["magnitude_bucket"] == b).sum())
                    mag_breakdown[b] = {
                        "count":     cnt,
                        "pct":       float(cnt / n_sig * 100),
                    }

            # Monthly profit estimate
            # Average TP gain: magnitude-weighted midpoint of positive buckets
            avg_tp_gain = 0.0
            if has_mag and tp > 0:
                tp_mask = (preds == 1) & (y_true == 1)
                tp_df   = test_df[tp_mask]
                weighted = 0.0
                for b, mid in _MAG_MIDPOINT.items():
                    cnt = int((tp_df["magnitude_bucket"] == b).sum())
                    weighted += cnt * mid
                avg_tp_gain = weighted / max(tp, 1)

            monthly_scale = 22.0 / max(test_days, 1)
            monthly_tp    = tp  * monthly_scale
            monthly_fp    = fp  * monthly_scale
            monthly_profit = (
                monthly_tp * position_size_usd * avg_tp_gain / 100.0
                + monthly_fp * position_size_usd * stop_loss_pct / 100.0
            )

            print(
                f"  {t:>7.2f}  {n_sig:>8,}  {spd:>8.1f}  "
                f"{prec:>7.1%}  {tp:>6,}  {fp:>6,}  ${monthly_profit:>11,.0f}"
            )

            results.append(
                {
                    "threshold":          t,
                    "n_signals":          n_sig,
                    "signals_per_day":    float(spd),
                    "precision":          float(prec),
                    "recall":             float(recall),
                    "tp":                 tp,
                    "fp":                 fp,
                    "fn":                 fn,
                    "avg_tp_gain_pct":    float(avg_tp_gain),
                    "monthly_profit_usd": float(monthly_profit),
                    "magnitude_breakdown": mag_breakdown,
                }
            )

        # Find optimal: precision >= 50% AND 10 <= signals/day <= 50
        viable = [
            r for r in results
            if r["precision"] >= 0.50 and 10 <= r["signals_per_day"] <= 50
        ]

        print(f"\n{'='*72}")
        print("  OPTIMAL THRESHOLD ANALYSIS")
        print(f"{'='*72}")

        if viable:
            # Best by monthly profit among viable
            optimal = max(viable, key=lambda r: r["monthly_profit_usd"])
            print(
                f"\n  VIABLE threshold found: {optimal['threshold']:.2f}\n"
                f"  Signals: {optimal['n_signals']:,} ({optimal['signals_per_day']:.1f}/day)\n"
                f"  Precision: {optimal['precision']:.1%}\n"
                f"  Monthly profit est: ${optimal['monthly_profit_usd']:,.0f}"
            )
            if optimal.get("magnitude_breakdown"):
                print("\n  Signal magnitude breakdown:")
                for b, d in optimal["magnitude_breakdown"].items():
                    print(f"    {b:<12}: {d['count']:>5,}  ({d['pct']:>5.1f}%)")
        else:
            optimal = None
            best_prec = max(results, key=lambda r: r["precision"])
            print(
                f"\n  No threshold meets criteria (>=50% precision, 10-50 signals/day)"
                f"\n  Best precision: {best_prec['precision']:.1%} @ {best_prec['threshold']:.2f}"
                f"\n  Signals: {best_prec['n_signals']:,} ({best_prec['signals_per_day']:.1f}/day)"
                f"\n  Monthly profit est: ${best_prec['monthly_profit_usd']:,.0f}"
            )

        print(f"{'='*72}")

        return {
            "results":    results,
            "optimal":    optimal,
            "model_name": model_name,
            "option_type": option_type,
            "test_days":  test_days,
        }
