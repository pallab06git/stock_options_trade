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
    ) -> None:
        self.thresholds = thresholds or _DEFAULT_THRESHOLDS
        self.target_col = target_col

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
        mag_dist = {b: 0 for b in MAGNITUDE_BUCKETS}
        if "magnitude_bucket" in test_df.columns:
            for b in MAGNITUDE_BUCKETS:
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
        tp_dist: Dict[str, int] = {b: 0 for b in MAGNITUDE_BUCKETS}
        if has_mag:
            ref_preds = (probas >= 0.70).astype(np.int8)
            tp_mask = (ref_preds == 1) & (y_true == 1)
            for b in MAGNITUDE_BUCKETS:
                bucket_mask = test_df["magnitude_bucket"] == b
                tp_dist[b] = int((tp_mask & bucket_mask.values).sum())

        by_threshold: Dict[float, Dict[str, Dict[str, Any]]] = {}
        for t in self.thresholds:
            preds = (probas >= t).astype(np.int8)
            bucket_stats: Dict[str, Dict[str, Any]] = {}

            for b in MAGNITUDE_BUCKETS:
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
