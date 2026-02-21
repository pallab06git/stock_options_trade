# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""ML model backtesting for options spike prediction.

Backtesting approach
--------------------
The model predicts whether an options bar will see a ≥20% price spike in
the next 120 minutes.  A "trade" is triggered every time the model's
predicted probability exceeds *threshold*.

For each predicted-positive bar the backtest records the **actual** maximum
gain available in the next 120 minutes (``max_gain_120m``).  This is a
simplified, idealized backtest — it assumes a perfect exit at the peak
and no transaction costs.  Its purpose is to measure the model's ability
to *identify bars that precede large moves*, not to simulate real P&L.

Key design decisions
--------------------
1. **Test set only**: predictions are made exclusively on the chronological
   *test* split — the set the model never saw during training.  Using the
   full dataset or the training split would produce inflated metrics.

2. **No look-ahead**: ``DataSplitter`` enforces chronological ordering so
   the test set always contains only future dates relative to training.

3. **Lift over random baseline**: the random baseline enters on every bar in
   the test set (regardless of model prediction).  Lift = ML average gain /
   baseline average gain.  Lift > 1 means the model adds value.

Metrics produced
----------------
  n_test_rows       : total rows in test split
  n_signals         : bars where model predicted positive (trades taken)
  n_true_positives  : signals where actual target = 1
  n_false_positives : signals where actual target = 0
  signal_rate       : n_signals / n_test_rows
  precision         : n_true_pos / n_signals  (fraction of trades that win)
  recall            : n_true_pos / total_positives_in_test
  f1                : harmonic mean of precision and recall
  roc_auc           : area under ROC curve for full test set
  avg_gain_all_bars : mean(max_gain_120m) across all test bars  (baseline)
  avg_gain_signals  : mean(max_gain_120m) for predicted-positive bars
  avg_gain_tp       : mean(max_gain_120m) where true positive (win trades)
  avg_gain_fp       : mean(max_gain_120m) where false positive (loss trades)
  lift              : avg_gain_signals / avg_gain_all_bars
  positive_rate_test: fraction of test bars with target=1

Per-trade output columns (trades DataFrame / CSV)
--------------------------------------------------
  date, ticker, timestamp, predicted_proba, predicted_label,
  actual_target, max_gain_120m, time_to_max_min, is_true_positive

Configuration keys (all under ``ml_training.backtest``, optional)
------------------------------------------------------------------
  output_dir  str    "data/reports/backtest"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.data_splitter import DataSplitter
from src.ml.train_xgboost import _NON_FEATURE_COLS, load_features
from src.utils.logger import get_logger

logger = get_logger()

# Columns carried into the per-trade output (if present in the test DataFrame)
_TRADE_META_COLS: List[str] = [
    "date",
    "ticker",
    "timestamp",
    "max_gain_120m",
    "min_loss_120m",
    "time_to_max_min",
]


# ---------------------------------------------------------------------------
# Module-level function (primary API)
# ---------------------------------------------------------------------------


def backtest_model(
    model: xgb.XGBClassifier,
    feature_cols: List[str],
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run a backtest on *df* using a pre-loaded XGBClassifier.

    The DataFrame must contain ``target`` and ``max_gain_120m`` columns in
    addition to all columns listed in *feature_cols*.

    Args:
        model:        Trained ``XGBClassifier``.
        feature_cols: Ordered feature columns used during training (must be
                      present in *df*).
        df:           DataFrame to evaluate (typically the test split from
                      ``DataSplitter.split``).  Must contain ``target`` and
                      ``max_gain_120m``.
        threshold:    Probability cutoff for binary prediction (default 0.5).

    Returns:
        ``(metrics_dict, trades_df)`` where *metrics_dict* contains all
        scalar backtest metrics and *trades_df* contains one row per
        predicted-positive bar (the "trades" taken by the model).

    Raises:
        ValueError: If *df* is empty.
        ValueError: If ``target`` is missing from *df*.
        ValueError: If any column in *feature_cols* is missing from *df*.
    """
    if df.empty:
        raise ValueError("backtest_model: input DataFrame is empty")

    if "target" not in df.columns:
        raise ValueError(
            "backtest_model: 'target' column required for backtesting. "
            f"Available: {df.columns.tolist()!r}"
        )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"backtest_model: feature columns missing from df: {missing!r}"
        )

    df = df.reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    y_true = df["target"].values.astype(int)

    probas: np.ndarray = model.predict_proba(X)[:, 1]
    y_pred: np.ndarray = (probas >= threshold).astype(int)

    metrics = _compute_metrics(y_true, y_pred, probas, df)
    trades_df = _build_trades_df(df, y_pred, probas)

    lift_str = f"{metrics['lift']:.2f}x" if metrics["lift"] is not None else "n/a"
    logger.info(
        f"backtest_model: {len(df)} test rows | "
        f"{metrics['n_signals']} signals | "
        f"precision={metrics['precision']:.3f} | "
        f"lift={lift_str}"
    )
    return metrics, trades_df


# ---------------------------------------------------------------------------
# ModelBacktester class (config-driven)
# ---------------------------------------------------------------------------


class ModelBacktester:
    """Config-driven backtest runner for saved XGBoost model artifacts.

    Loads a ``.pkl`` artifact, applies ``DataSplitter`` to take the test
    split, runs predictions, and produces both a metrics dict and a
    per-trade CSV.

    Usage::

        backtester = ModelBacktester(config)
        results = backtester.run(
            model_path="models/xgboost_v1.pkl",
            features_dir="data/processed/features",
            start_date="2025-03-03",
            end_date="2026-01-31",
        )
        print(f"Lift: {results['metrics']['lift']:.2f}x")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged application config dict.  Settings from
                    ``config["ml_training"]["backtest"]``.
                    Split ratios from ``config["data_preparation"]``.
        """
        bt_cfg = config.get("ml_training", {}).get("backtest", {})
        self.output_dir: str = bt_cfg.get("output_dir", "data/reports/backtest")

        self._splitter = DataSplitter(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model_path: str | Path,
        features_dir: str | Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load artifact + features, backtest on test split, save reports.

        Args:
            model_path:   Path to a ``.pkl`` artifact from ``XGBoostTrainer``.
            features_dir: Directory with ``*_features.csv`` files.
            start_date:   Earliest date to include (optional).
            end_date:     Latest date to include (optional).
            output_dir:   Override the configured output directory.
            threshold:    Override the model artifact's prediction threshold
                          (0.0–1.0).  ``None`` uses the artifact value.

        Returns:
            Dict with keys:
              ``metrics``   — scalar backtest metrics dict
              ``model_path``— str path to the model artifact
              ``trades_path``— str path to the per-trade CSV (or None)
              ``report_path``— str path to the JSON metrics file

        Raises:
            FileNotFoundError: If model_path or features_dir is missing.
            ValueError: If no data is loaded or the test split is empty.
        """
        # ── Load artifact ─────────────────────────────────────────────
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ModelBacktester: model artifact not found: {model_path}"
            )
        artifact = joblib.load(model_path)
        model: xgb.XGBClassifier = artifact["model"]
        feature_cols: List[str] = artifact["feature_cols"]
        # Use caller-supplied threshold; fall back to artifact value
        threshold = float(threshold) if threshold is not None else float(artifact.get("threshold", 0.5))
        model_version: str = model_path.stem  # e.g. "xgboost_v1"

        # ── Load features ─────────────────────────────────────────────
        df = load_features(features_dir, start_date, end_date)
        if df.empty:
            raise ValueError(
                f"ModelBacktester: no feature data loaded from {features_dir} "
                f"for range [{start_date}, {end_date}]"
            )

        # ── Split — take test set only ────────────────────────────────
        _, _, test_df = self._splitter.split(df)
        if test_df.empty:
            raise ValueError("ModelBacktester: test split is empty after chronological split")

        logger.info(
            f"ModelBacktester: {len(df)} total rows | "
            f"{len(test_df)} test rows | threshold={threshold:.2f}"
            + (" (CLI override)" if threshold != float(artifact.get("threshold", 0.5)) else "")
        )

        # ── Backtest ──────────────────────────────────────────────────
        metrics, trades_df = backtest_model(model, feature_cols, test_df, threshold)

        # ── Save reports ──────────────────────────────────────────────
        out_dir = Path(output_dir) if output_dir is not None else Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        trades_path = self._save_trades(trades_df, model_version, run_ts, out_dir)
        report_path = self._save_report(metrics, model_version, run_ts, out_dir)

        return {
            "metrics": metrics,
            "model_path": str(model_path),
            "trades_path": str(trades_path),
            "report_path": str(report_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_trades(
        self,
        trades_df: pd.DataFrame,
        model_version: str,
        run_ts: str,
        out_dir: Path,
    ) -> Path:
        """Save per-trade DataFrame as CSV."""
        path = out_dir / f"{model_version}_trades_{run_ts}.csv"
        trades_df.to_csv(path, index=False)
        logger.info(f"Trades CSV saved → {path}")
        return path

    def _save_report(
        self,
        metrics: Dict[str, Any],
        model_version: str,
        run_ts: str,
        out_dir: Path,
    ) -> Path:
        """Save metrics dict as JSON."""
        path = out_dir / f"{model_version}_backtest_{run_ts}.json"
        with open(path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Backtest report saved → {path}")
        return path


# ---------------------------------------------------------------------------
# Internal computation helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute all scalar backtest metrics."""
    n_total = len(y_true)
    n_signals = int(y_pred.sum())
    n_positives_in_test = int(y_true.sum())

    signal_mask = y_pred == 1
    tp_mask = signal_mask & (y_true == 1)
    fp_mask = signal_mask & (y_true == 0)

    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, probas))
    except ValueError:
        # Only one class in test set — AUC undefined
        roc_auc = float("nan")

    # Gain metrics (require max_gain_120m column)
    avg_gain_all = float("nan")
    avg_gain_signals = float("nan")
    avg_gain_tp = float("nan")
    avg_gain_fp = float("nan")
    lift = float("nan")

    if "max_gain_120m" in df.columns:
        gains = df["max_gain_120m"].values.astype(float)
        valid_all = ~np.isnan(gains)

        if valid_all.any():
            avg_gain_all = float(np.nanmean(gains))

        if signal_mask.any():
            sig_gains = gains[signal_mask]
            avg_gain_signals = float(np.nanmean(sig_gains)) if len(sig_gains) > 0 else float("nan")

        if tp_mask.any():
            avg_gain_tp = float(np.nanmean(gains[tp_mask]))

        if fp_mask.any():
            avg_gain_fp = float(np.nanmean(gains[fp_mask]))

        if not np.isnan(avg_gain_all) and avg_gain_all > 0 and not np.isnan(avg_gain_signals):
            lift = avg_gain_signals / avg_gain_all

    positive_rate_test = n_positives_in_test / n_total if n_total > 0 else 0.0
    signal_rate = n_signals / n_total if n_total > 0 else 0.0

    return {
        "n_test_rows": n_total,
        "n_signals": n_signals,
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "signal_rate": round(signal_rate, 6),
        "positive_rate_test": round(positive_rate_test, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6) if not np.isnan(roc_auc) else None,
        "avg_gain_all_bars": round(avg_gain_all, 6) if not np.isnan(avg_gain_all) else None,
        "avg_gain_signals": round(avg_gain_signals, 6) if not np.isnan(avg_gain_signals) else None,
        "avg_gain_tp": round(avg_gain_tp, 6) if not np.isnan(avg_gain_tp) else None,
        "avg_gain_fp": round(avg_gain_fp, 6) if not np.isnan(avg_gain_fp) else None,
        "lift": round(lift, 6) if not np.isnan(lift) else None,
    }


def _build_trades_df(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    probas: np.ndarray,
) -> pd.DataFrame:
    """Build a DataFrame of predicted-positive bars (the "trades" taken).

    Carries through available metadata columns and the actual outcome.
    """
    signal_mask = y_pred == 1

    if not signal_mask.any():
        # Return an empty DataFrame with the expected schema
        cols = (
            [c for c in _TRADE_META_COLS if c in df.columns]
            + ["predicted_proba", "predicted_label", "actual_target", "is_true_positive"]
        )
        return pd.DataFrame(columns=cols)

    trades = df[signal_mask].copy()
    trades["predicted_proba"] = probas[signal_mask]
    trades["predicted_label"] = 1

    if "target" in trades.columns:
        trades["actual_target"] = trades["target"].astype(int)
        trades["is_true_positive"] = (trades["actual_target"] == 1)
    else:
        trades["actual_target"] = None
        trades["is_true_positive"] = None

    # Keep only the useful columns (metadata + outcome)
    keep = [c for c in _TRADE_META_COLS if c in trades.columns]
    keep += ["predicted_proba", "predicted_label", "actual_target", "is_true_positive"]
    trades = trades[keep].reset_index(drop=True)

    return trades
