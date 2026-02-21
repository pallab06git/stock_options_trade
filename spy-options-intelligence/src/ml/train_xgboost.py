# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""XGBoost training pipeline for options spike prediction.

Pipeline order (critical for preventing data leakage)
------------------------------------------------------
  1. Load all feature CSVs in [start_date, end_date]
  2. Split chronologically → train / val / test (BEFORE any balancing)
  3. Balance training set only (undersample majority class)
  4. Fit XGBoostClassifier on balanced training set
  5. Evaluate on (unbalanced) validation set
  6. Save model artifact + feature column list to models/
  7. Log training metrics as JSON to data/logs/training/

Why split before balance
------------------------
If the full dataset were balanced first, the resampling would alter which
rows are present before the chronological cut is applied.  All three splits
would then reflect the resampled distribution rather than the real-world
1% positive rate — a subtle form of distribution leakage.  By splitting
first we guarantee that:

  * Only the *training* set distribution is modified.
  * Validation and test sets remain at their natural (skewed) distribution,
    which is the distribution the deployed model will encounter in production.

Model artifact layout
---------------------
Saved via ``joblib.dump`` as a plain dict so the artifact is self-describing::

    {
        "model":       <trained XGBClassifier>,
        "feature_cols": [...],
        "threshold":   0.5,
        "xgb_params":  {...},
        "saved_at":    "2026-02-20T14:32:00",
    }

Configuration keys (all under ``ml_training.xgboost``, all optional)
---------------------------------------------------------------------
  n_estimators          int    300
  max_depth             int    6
  learning_rate         float  0.05
  subsample             float  0.80
  colsample_bytree      float  0.80
  min_child_weight      int    5
  gamma                 float  0.10
  random_state          int    42
  early_stopping_rounds int    20
  eval_metric           str    "logloss"
  threshold             float  0.50   (classification cutoff for binary labels)
  model_version         str    "v1"
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.data_balancer import DataBalancer
from src.ml.data_splitter import DataSplitter
from src.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Columns that must never be used as model input features
# ---------------------------------------------------------------------------

#: Raw OHLCV from options bars, metadata, and all label columns.
_NON_FEATURE_COLS: frozenset = frozenset(
    {
        # time / metadata
        "date",
        "ticker",
        "timestamp",
        "source",
        # raw options price/volume (computed features already capture these)
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "opt_close",
        # label columns — must never leak into features
        "target",
        "max_gain_120m",
        "min_loss_120m",
        "max_gain_pct",
        "time_to_max_min",
    }
)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def load_features(
    features_dir: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate feature CSVs from *features_dir*.

    Each CSV is expected to follow the naming convention
    ``{YYYY-MM-DD}_features.csv``.  Only files whose date component falls
    within ``[start_date, end_date]`` (both inclusive) are loaded.  When
    either bound is ``None`` the corresponding end is unbounded.

    Args:
        features_dir: Directory containing ``*_features.csv`` files.
        start_date:   Earliest date to include (``"YYYY-MM-DD"``), or
                      ``None`` for no lower bound.
        end_date:     Latest date to include (``"YYYY-MM-DD"``), or
                      ``None`` for no upper bound.

    Returns:
        Concatenated DataFrame sorted by ``timestamp``.  Returns an empty
        DataFrame if no matching files are found.

    Raises:
        FileNotFoundError: If ``features_dir`` does not exist.
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    csv_files = sorted(features_dir.glob("*_features.csv"))

    selected: List[Path] = []
    for path in csv_files:
        # Extract date from file name: "2025-03-03_features.csv" → "2025-03-03"
        stem = path.stem  # e.g. "2025-03-03_features"
        date_part = stem.split("_features")[0]  # "2025-03-03"
        if start_date and date_part < start_date:
            continue
        if end_date and date_part > end_date:
            continue
        selected.append(path)

    if not selected:
        logger.warning(
            f"load_features: no CSV files found in {features_dir} "
            f"for range [{start_date}, {end_date}]"
        )
        return pd.DataFrame()

    frames = []
    for path in selected:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"load_features: skipping {path.name} — {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"load_features: loaded {len(selected)} file(s) → "
        f"{len(combined)} rows, {len(combined.columns)} columns"
    )
    return combined


# ---------------------------------------------------------------------------
# XGBoostTrainer class
# ---------------------------------------------------------------------------


class XGBoostTrainer:
    """Config-driven XGBoost training pipeline.

    Reads XGBoost hyper-parameters and training settings from
    ``config["ml_training"]["xgboost"]`` and orchestrates the full
    train → evaluate → save pipeline.

    Usage::

        trainer = XGBoostTrainer(config)
        metrics = trainer.train(
            features_dir="data/processed/features",
            start_date="2025-03-03",
            end_date="2026-01-31",
        )
        print(metrics)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged application config dict.  XGBoost parameters are
                    read from ``config["ml_training"]["xgboost"]``.  Split
                    ratios are read from ``config["data_preparation"]``.
        """
        xgb_cfg = config.get("ml_training", {}).get("xgboost", {})

        self.n_estimators: int = xgb_cfg.get("n_estimators", 300)
        self.max_depth: int = xgb_cfg.get("max_depth", 6)
        self.learning_rate: float = xgb_cfg.get("learning_rate", 0.05)
        self.subsample: float = xgb_cfg.get("subsample", 0.80)
        self.colsample_bytree: float = xgb_cfg.get("colsample_bytree", 0.80)
        self.min_child_weight: int = xgb_cfg.get("min_child_weight", 5)
        self.gamma: float = xgb_cfg.get("gamma", 0.10)
        self.random_state: int = xgb_cfg.get("random_state", 42)
        self.early_stopping_rounds: int = xgb_cfg.get("early_stopping_rounds", 20)
        self.eval_metric: str = xgb_cfg.get("eval_metric", "logloss")
        self.threshold: float = xgb_cfg.get("threshold", 0.50)
        self.model_version: str = xgb_cfg.get("model_version", "v1")

        self._splitter = DataSplitter(config)
        self._balancer = DataBalancer(config)

        self.xgb_params: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            # early_stopping_rounds belongs in the constructor (XGBoost ≥2.0)
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        features_dir: str | Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        models_dir: str | Path = "models",
        logs_dir: str | Path = "data/logs/training",
    ) -> Dict[str, Any]:
        """Run the full training pipeline and return a metrics dict.

        Steps
        -----
        1. Load feature CSVs from *features_dir*.
        2. Split chronologically → (train, val, test).
        3. Balance training split by undersampling majority class.
        4. Fit XGBoostClassifier with early stopping on validation set.
        5. Evaluate on unbalanced validation set.
        6. Save model artifact to *models_dir*.
        7. Log metrics JSON to *logs_dir*.

        Args:
            features_dir: Directory containing ``*_features.csv`` files.
            start_date:   Earliest date to include (optional).
            end_date:     Latest date to include (optional).
            models_dir:   Directory to write the ``.pkl`` model artifact.
            logs_dir:     Directory to write the JSON metrics log.

        Returns:
            Metrics dict with keys:
              ``val_accuracy``, ``val_precision``, ``val_recall``,
              ``val_f1``, ``val_roc_auc``, ``train_rows``, ``val_rows``,
              ``test_rows``, ``n_features``, ``model_path``, ``log_path``.

        Raises:
            ValueError: If the loaded DataFrame has no ``target`` column.
            ValueError: If the training split is empty after balancing.
        """
        # ── 1. Load ──────────────────────────────────────────────────────
        df = load_features(features_dir, start_date, end_date)
        if df.empty:
            raise ValueError(
                f"XGBoostTrainer.train: no data loaded from {features_dir} "
                f"for range [{start_date}, {end_date}]"
            )

        if "target" not in df.columns:
            raise ValueError(
                "XGBoostTrainer.train: 'target' column missing from features "
                f"DataFrame.  Available: {df.columns.tolist()!r}"
            )

        feature_cols = self._get_feature_cols(df)
        logger.info(
            f"XGBoostTrainer.train: {len(df)} rows | "
            f"{len(feature_cols)} features | "
            f"positive rate {df['target'].mean():.2%}"
        )

        # ── 2. Split ─────────────────────────────────────────────────────
        train_df, val_df, test_df = self._splitter.split(df)
        logger.info(
            f"Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
        )

        if train_df.empty:
            raise ValueError("XGBoostTrainer.train: training split is empty")

        # ── 3. Balance training split only ───────────────────────────────
        train_balanced = self._balancer.balance(train_df)
        logger.info(
            f"Training set after balance: {len(train_balanced)} rows | "
            f"positive rate {train_balanced['target'].mean():.2%}"
        )

        # ── 4. Prepare arrays ────────────────────────────────────────────
        X_train = train_balanced[feature_cols].values.astype(np.float32)
        y_train = train_balanced["target"].values.astype(np.int8)

        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["target"].values.astype(np.int8)

        # ── 5. Fit ───────────────────────────────────────────────────────
        model = xgb.XGBClassifier(**self.xgb_params)

        fit_kwargs: dict = {"eval_set": [(X_val, y_val)], "verbose": False}
        if self._balancer.balance_method == "class_weights":
            cw = self._balancer.get_class_weights(train_balanced)
            fit_kwargs["sample_weight"] = np.array(
                [cw[int(label)] for label in y_train], dtype=np.float32
            )
            logger.info(
                f"Applying class weights to fit: "
                + ", ".join(f"class {k}: {v:.4f}" for k, v in sorted(cw.items()))
            )

        model.fit(X_train, y_train, **fit_kwargs)
        logger.info(
            f"XGBoost fitted | best_iteration={model.best_iteration} | "
            f"best_score={model.best_score:.4f}"
        )

        # ── 6. Evaluate on validation set ────────────────────────────────
        val_metrics = self._evaluate(model, X_val, y_val, self.threshold)
        logger.info(
            f"Val metrics: "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"prec={val_metrics['precision']:.4f} | "
            f"rec={val_metrics['recall']:.4f} | "
            f"f1={val_metrics['f1']:.4f} | "
            f"auc={val_metrics['roc_auc']:.4f}"
        )

        # ── 7. Save model artifact ────────────────────────────────────────
        artifact = {
            "model": model,
            "feature_cols": feature_cols,
            "threshold": self.threshold,
            "xgb_params": self.xgb_params,
            "saved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        model_path = self._save_model(artifact, self.model_version, models_dir)

        # ── 8. Log metrics ────────────────────────────────────────────────
        run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        full_metrics: Dict[str, Any] = {
            "run_timestamp": run_ts,
            "start_date": start_date,
            "end_date": end_date,
            "train_rows": len(train_df),
            "train_rows_balanced": len(train_balanced),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "n_features": len(feature_cols),
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "threshold": self.threshold,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "model_path": str(model_path),
            "xgb_params": self.xgb_params,
        }
        log_path = self._log_metrics(full_metrics, run_ts, logs_dir)
        full_metrics["log_path"] = str(log_path)

        return full_metrics

    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return the list of columns to use as model features.

        Convenience wrapper over ``_get_feature_cols`` for external callers.
        """
        return self._get_feature_cols(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return sorted list of feature columns (excludes non-feature cols)."""
        return sorted(
            [c for c in df.columns if c not in _NON_FEATURE_COLS]
        )

    def _evaluate(
        self,
        model: xgb.XGBClassifier,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """Compute binary classification metrics at *threshold*.

        Returns dict with keys: ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``.
        """
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        }

    def _save_model(
        self,
        artifact: Dict[str, Any],
        version: str,
        models_dir: str | Path,
    ) -> Path:
        """Persist model artifact to ``{models_dir}/xgboost_{version}.pkl``.

        The directory is created if it does not exist.
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"xgboost_{version}.pkl"
        joblib.dump(artifact, path)
        logger.info(f"Model saved → {path}")
        return path

    def _log_metrics(
        self,
        metrics: Dict[str, Any],
        run_ts: str,
        logs_dir: str | Path,
    ) -> Path:
        """Write metrics dict as JSON to ``{logs_dir}/training_{run_ts}.json``.

        The directory is created if it does not exist.
        """
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"training_{run_ts}.json"
        with open(path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Training metrics logged → {path}")
        return path
