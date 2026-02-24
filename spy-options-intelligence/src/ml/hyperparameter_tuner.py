# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Exhaustive hyperparameter optimization for the stacked ensemble.

Objective
---------
Maximize ``precision @ top-k signals per day`` on the validation set.
This is the direct business metric: of the k signals the pipeline selects
each day, what fraction are actually profitable?

Strategy
--------
  1. Chronological 70 / 20 / 10 split of all feature data.
  2. Optuna TPE sampler with MedianPruner (skips clearly bad trials early).
  3. Each base model is tuned independently (faster, avoids interaction noise).
  4. After model tuning, a grid sweep over entry thresholds finds the
     optimal operating point on the VALIDATION set.
  5. A final report compares default vs tuned params across all metrics.

Models tuned
------------
  - XGBoost   : 75 trials  (~10-15 min)
  - LightGBM  : 75 trials  (~6-10 min)
  - RandomForest: 30 trials (~20-25 min, slower — no early stopping)
  - Entry threshold: grid sweep after model tuning

New search spaces (vs current defaults)
----------------------------------------
  XGBoost    : adds reg_alpha, reg_lambda  (L1/L2 regularisation)
  LightGBM   : adds num_leaves (primary complexity knob), min_child_samples,
                feature_fraction, bagging_fraction, reg_alpha, reg_lambda
  RandomForest: adds max_features (currently using sklearn sqrt default)

Usage
-----
    from src.ml.hyperparameter_tuner import HyperparameterTuner

    tuner = HyperparameterTuner("data/processed/features", n_trials=75)
    results = tuner.run("reports/hyperparameter_tuning")
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from src.utils.logger import get_logger

logger = get_logger()

_XGB_AVAILABLE = False
_LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    pass

# Columns that are never model features
_NON_FEATURE_COLS = frozenset([
    "timestamp", "ticker", "source", "date", "target",
    "max_gain_120m", "min_loss_120m", "time_to_max_min",
    "open", "high", "low", "close", "volume", "vwap",
    "transactions", "opt_close", "market_regime", "regime_confidence",
])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_features(features_dir: str | Path,
                   start_date: str = "2025-03-03",
                   end_date: str = "2026-02-19") -> pd.DataFrame:
    """Load and concatenate feature CSVs within date range."""
    fdir = Path(features_dir)
    dfs: List[pd.DataFrame] = []
    for f in sorted(fdir.glob("*_features.csv")):
        date_str = f.stem.replace("_features", "")
        if start_date <= date_str <= end_date:
            try:
                dfs.append(pd.read_csv(f, low_memory=False))
            except Exception:
                pass
    if not dfs:
        raise ValueError(f"No features found in {fdir} for {start_date}..{end_date}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date.astype(str)
    return df


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c not in _NON_FEATURE_COLS])


def _chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date boundaries (whole trading days kept together)."""
    dates = sorted(df["date"].unique())
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(val_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )


def _top_k_precision(
    df: pd.DataFrame,
    proba: np.ndarray,
    k: int = 3,
    target_col: str = "target",
) -> float:
    """Precision at top-k signals per day.

    For each calendar day in df, take the k rows with highest predicted
    probability.  Return the fraction of those rows where target == 1.
    This is the direct business metric for the trading pipeline.
    """
    tmp = df[["date", target_col]].copy()
    tmp["_p"] = proba
    top_rows = (
        tmp.groupby("date", group_keys=False)
           .apply(lambda g: g.nlargest(k, "_p"))
    )
    if top_rows.empty:
        return 0.0
    return float(top_rows[target_col].mean())


def _metrics_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (proba >= threshold).astype(int)
    n_signals = int(preds.sum())
    if n_signals == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_signals": 0, "roc_auc": float(roc_auc_score(y_true, proba))}
    return {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "n_signals": n_signals,
        "roc_auc":   float(roc_auc_score(y_true, proba)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main tuner
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterTuner:
    """Exhaustive hyperparameter optimisation for the stacked ensemble.

    Parameters
    ----------
    features_dir : str or Path
        Directory containing ``*_features.csv`` files.
    n_trials : int
        Number of Optuna trials per model (RF uses min(n_trials, 30)).
    top_k : int
        Number of signals to select per day for the precision metric.
    train_start, train_end : str
        Date range filter for feature files.
    """

    def __init__(
        self,
        features_dir: str,
        n_trials: int = 75,
        top_k: int = 3,
        train_start: str = "2025-03-03",
        train_end: str = "2026-02-19",
    ) -> None:
        self.features_dir = Path(features_dir)
        self.n_trials = n_trials
        self.top_k = top_k
        self.train_start = train_start
        self.train_end = train_end

        self._study_results: Dict[str, Any] = {}
        self._best_params: Dict[str, Any] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _prepare_data(self):
        """Load features and return train/val/test splits."""
        logger.info(f"Loading features from {self.features_dir} "
                    f"({self.train_start} → {self.train_end})...")
        df = _load_features(self.features_dir, self.train_start, self.train_end)
        feature_cols = _get_feature_cols(df)

        logger.info(f"  Total: {len(df):,} rows | {len(feature_cols)} features | "
                    f"target rate {df['target'].mean():.3f}")

        train_df, val_df, test_df = _chronological_split(df)

        def _to_xy(split_df):
            X = np.nan_to_num(
                split_df[feature_cols].values.astype(np.float32),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            y = split_df["target"].values.astype(int)
            return X, y

        X_tr, y_tr = _to_xy(train_df)
        X_val, y_val = _to_xy(val_df)
        X_test, y_test = _to_xy(test_df)

        # Class imbalance weight for XGBoost / LightGBM
        scale_pw = max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)

        logger.info(
            f"  Train: {len(X_tr):,} rows ({(y_tr==1).sum()} pos) | "
            f"Val: {len(X_val):,} rows ({(y_val==1).sum()} pos) | "
            f"Test: {len(X_test):,} rows ({(y_test==1).sum()} pos) | "
            f"scale_pos_weight={scale_pw:.2f}"
        )

        return X_tr, y_tr, X_val, y_val, X_test, y_test, val_df, test_df, feature_cols, scale_pw

    # ── Baseline (default params) ─────────────────────────────────────────────

    def _compute_baseline(self, X_tr, y_tr, X_val, y_val, val_df, scale_pw, feature_cols):
        """Train with current default params and compute baseline precision@top-k."""
        results = {}

        if _XGB_AVAILABLE:
            m = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.80, min_child_weight=5,
                gamma=0.10, scale_pos_weight=scale_pw, random_state=42,
                eval_metric="logloss", early_stopping_rounds=20,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            results["xgboost"] = {
                "precision_top_k": _top_k_precision(val_df, p, self.top_k),
                "roc_auc": float(roc_auc_score(y_val, p)),
                "params": "defaults",
            }
            logger.info(f"Baseline XGBoost  precision@top-{self.top_k}: "
                        f"{results['xgboost']['precision_top_k']:.4f}")

        if _LGBM_AVAILABLE:
            m = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.80, min_child_weight=5,
                scale_pos_weight=scale_pw, random_state=42, verbosity=-1,
            )
            m.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
            )
            p = m.predict_proba(X_val)[:, 1]
            results["lightgbm"] = {
                "precision_top_k": _top_k_precision(val_df, p, self.top_k),
                "roc_auc": float(roc_auc_score(y_val, p)),
                "params": "defaults",
            }
            logger.info(f"Baseline LightGBM precision@top-{self.top_k}: "
                        f"{results['lightgbm']['precision_top_k']:.4f}")

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p = rf.predict_proba(X_val)[:, 1]
        results["random_forest"] = {
            "precision_top_k": _top_k_precision(val_df, p, self.top_k),
            "roc_auc": float(roc_auc_score(y_val, p)),
            "params": "defaults",
        }
        logger.info(f"Baseline RF        precision@top-{self.top_k}: "
                    f"{results['random_forest']['precision_top_k']:.4f}")

        return results

    # ── XGBoost tuning ────────────────────────────────────────────────────────

    def tune_xgboost(
        self, X_tr, y_tr, X_val, y_val, val_df, scale_pw
    ) -> Dict[str, Any]:
        """Tune XGBoost with Optuna. Returns best params dict."""
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed")

        logger.info(f"\n{'─'*60}")
        logger.info(f"Tuning XGBoost  ({self.n_trials} trials, "
                    f"objective=precision@top-{self.top_k}/day)")
        logger.info(f"{'─'*60}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(
                n_estimators       = trial.suggest_int("n_estimators", 100, 700),
                max_depth          = trial.suggest_int("max_depth", 3, 9),
                learning_rate      = trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
                min_child_weight   = trial.suggest_int("min_child_weight", 1, 30),
                gamma              = trial.suggest_float("gamma", 0.0, 0.6),
                subsample          = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree   = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_alpha          = trial.suggest_float("reg_alpha", 0.0, 3.0),
                reg_lambda         = trial.suggest_float("reg_lambda", 0.5, 6.0),
                scale_pos_weight   = scale_pw,
                random_state       = 42,
                eval_metric        = "logloss",
                early_stopping_rounds = 25,
            )
            try:
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                proba = m.predict_proba(X_val)[:, 1]
                return _top_k_precision(val_df, proba, self.top_k)
            except Exception:
                return 0.0

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        elapsed = time.time() - t0

        best_params = study.best_params
        best_val = study.best_value
        logger.info(f"\nXGBoost  — best precision@top-{self.top_k}: {best_val:.4f}  "
                    f"({elapsed/60:.1f} min, {len(study.trials)} trials)")
        logger.info(f"  Best params: {best_params}")

        self._study_results["xgboost"] = {
            "best_precision_top_k": best_val,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": round(elapsed),
        }
        return best_params

    # ── LightGBM tuning ───────────────────────────────────────────────────────

    def tune_lightgbm(
        self, X_tr, y_tr, X_val, y_val, val_df, scale_pw
    ) -> Dict[str, Any]:
        """Tune LightGBM with Optuna. Returns best params dict."""
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm not installed")

        logger.info(f"\n{'─'*60}")
        logger.info(f"Tuning LightGBM ({self.n_trials} trials, "
                    f"objective=precision@top-{self.top_k}/day)")
        logger.info(f"{'─'*60}")

        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 100, 700),
                num_leaves        = trial.suggest_int("num_leaves", 20, 150),
                learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
                min_child_samples = trial.suggest_int("min_child_samples", 10, 120),
                feature_fraction  = trial.suggest_float("feature_fraction", 0.4, 1.0),
                bagging_fraction  = trial.suggest_float("bagging_fraction", 0.4, 1.0),
                bagging_freq      = trial.suggest_int("bagging_freq", 1, 7),
                reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 3.0),
                reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 6.0),
                max_depth         = trial.suggest_int("max_depth", 4, 12),
                scale_pos_weight  = scale_pw,
                random_state      = 42,
                verbosity         = -1,
            )
            try:
                m = lgb.LGBMClassifier(**params)
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(25, verbose=False),
                               lgb.log_evaluation(0)],
                )
                proba = m.predict_proba(X_val)[:, 1]
                return _top_k_precision(val_df, proba, self.top_k)
            except Exception:
                return 0.0

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        elapsed = time.time() - t0

        best_params = study.best_params
        best_val = study.best_value
        logger.info(f"\nLightGBM — best precision@top-{self.top_k}: {best_val:.4f}  "
                    f"({elapsed/60:.1f} min, {len(study.trials)} trials)")
        logger.info(f"  Best params: {best_params}")

        self._study_results["lightgbm"] = {
            "best_precision_top_k": best_val,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": round(elapsed),
        }
        return best_params

    # ── RandomForest tuning ───────────────────────────────────────────────────

    def tune_random_forest(
        self, X_tr, y_tr, X_val, y_val, val_df
    ) -> Dict[str, Any]:
        """Tune RandomForest with Optuna. Returns best params dict."""
        rf_trials = min(self.n_trials, 30)   # RF is slow — cap at 30

        logger.info(f"\n{'─'*60}")
        logger.info(f"Tuning RandomForest ({rf_trials} trials, "
                    f"objective=precision@top-{self.top_k}/day)")
        logger.info(f"{'─'*60}")

        def objective(trial):
            params = dict(
                n_estimators    = trial.suggest_int("n_estimators", 100, 400),
                max_depth       = trial.suggest_int("max_depth", 5, 25),
                min_samples_leaf= trial.suggest_int("min_samples_leaf", 5, 60),
                max_features    = trial.suggest_float("max_features", 0.15, 0.80),
                class_weight    = "balanced",
                random_state    = 42,
                n_jobs          = -1,
            )
            try:
                m = RandomForestClassifier(**params)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_val)[:, 1]
                return _top_k_precision(val_df, proba, self.top_k)
            except Exception:
                return 0.0

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=rf_trials, show_progress_bar=True)
        elapsed = time.time() - t0

        best_params = study.best_params
        best_val = study.best_value
        logger.info(f"\nRF       — best precision@top-{self.top_k}: {best_val:.4f}  "
                    f"({elapsed/60:.1f} min, {len(study.trials)} trials)")
        logger.info(f"  Best params: {best_params}")

        self._study_results["random_forest"] = {
            "best_precision_top_k": best_val,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": round(elapsed),
        }
        return best_params

    # ── Entry threshold sweep ─────────────────────────────────────────────────

    def sweep_entry_threshold(
        self,
        X_tr, y_tr, X_val, y_val, val_df, scale_pw,
        xgb_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Grid sweep over entry thresholds using the tuned ensemble (simple avg).

        For each threshold in [0.50 .. 0.75], computes:
          - precision at threshold
          - recall at threshold
          - signals per day
          - precision@top-3
        Returns the sweep table + recommended threshold.
        """
        logger.info(f"\n{'─'*60}")
        logger.info("Entry threshold sweep (tuned ensemble simple average)")
        logger.info(f"{'─'*60}")

        # Train each base model with tuned (or default) params
        all_probas = []

        if _XGB_AVAILABLE:
            p = xgb_params or {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.80, "colsample_bytree": 0.80,
                "min_child_weight": 5, "gamma": 0.10,
                "scale_pos_weight": scale_pw, "random_state": 42,
                "eval_metric": "logloss", "early_stopping_rounds": 25,
            }
            p = {**p, "scale_pos_weight": scale_pw,
                 "eval_metric": "logloss", "early_stopping_rounds": 25}
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            all_probas.append(m.predict_proba(X_val)[:, 1])

        if _LGBM_AVAILABLE:
            p = lgbm_params or {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 5,
                "scale_pos_weight": scale_pw, "random_state": 42, "verbosity": -1,
            }
            p = {**p, "scale_pos_weight": scale_pw, "verbosity": -1}
            m = lgb.LGBMClassifier(**p)
            m.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
            )
            all_probas.append(m.predict_proba(X_val)[:, 1])

        p = rf_params or {
            "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
        }
        m = RandomForestClassifier(**{k: v for k, v in p.items()
                                      if k not in ("scale_pos_weight",
                                                   "eval_metric",
                                                   "early_stopping_rounds",
                                                   "verbosity",
                                                   "bagging_freq",
                                                   "feature_fraction",
                                                   "bagging_fraction",
                                                   "reg_alpha",
                                                   "reg_lambda",
                                                   "num_leaves",
                                                   "min_child_samples")})
        m.set_params(class_weight="balanced", n_jobs=-1)
        m.fit(X_tr, y_tr)
        all_probas.append(m.predict_proba(X_val)[:, 1])

        ensemble_proba = np.mean(all_probas, axis=0)
        n_days_val = val_df["date"].nunique()

        thresholds = [0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]
        rows = []
        for thr in thresholds:
            m = _metrics_at_threshold(y_val, ensemble_proba, thr)
            sigs_per_day = m["n_signals"] / max(n_days_val, 1)
            prec_topk = _top_k_precision(val_df, ensemble_proba, self.top_k)
            rows.append({
                "threshold":      thr,
                "precision":      round(m["precision"], 4),
                "recall":         round(m["recall"], 4),
                "f1":             round(m["f1"], 4),
                "n_signals":      m["n_signals"],
                "signals_per_day":round(sigs_per_day, 2),
                "roc_auc":        round(m["roc_auc"], 4),
                "precision_top_k":round(prec_topk, 4),
            })

        df_sweep = pd.DataFrame(rows)
        logger.info(f"\n{'Thresh':>7}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
                    f"{'Sigs/day':>9}  {'P@top-k':>9}")
        for _, r in df_sweep.iterrows():
            logger.info(f"  {r.threshold:.2f}  {r.precision:.4f}  {r.recall:.4f}  "
                        f"{r.f1:.4f}  {r.signals_per_day:9.2f}  {r.precision_top_k:.4f}")

        # Recommend: highest precision with >= 1 signal/day
        valid = df_sweep[df_sweep["signals_per_day"] >= 1.0]
        if not valid.empty:
            best_row = valid.loc[valid["precision"].idxmax()]
            recommended_threshold = float(best_row["threshold"])
        else:
            recommended_threshold = 0.65

        logger.info(f"\nRecommended entry threshold: {recommended_threshold}")

        return {
            "sweep_table": df_sweep.to_dict(orient="records"),
            "recommended_threshold": recommended_threshold,
            "ensemble_roc_auc": float(roc_auc_score(y_val, ensemble_proba)),
        }

    # ── Main orchestrator ─────────────────────────────────────────────────────

    def run(self, output_dir: str | Path = "reports/hyperparameter_tuning") -> Dict[str, Any]:
        """Run full hyperparameter tuning pipeline.

        Steps:
          1. Load data, chronological split.
          2. Compute baseline precision@top-k for each model.
          3. Tune XGBoost (Optuna, 75 trials).
          4. Tune LightGBM (Optuna, 75 trials).
          5. Tune RandomForest (Optuna, 30 trials).
          6. Entry threshold sweep with tuned ensemble.
          7. Save comprehensive JSON report.

        Returns:
            Dict with ``best_params`` keyed by model name + ``threshold_sweep``.
        """
        t_total = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ─ Data ────────────────────────────────────────────────────────────────
        data = self._prepare_data()
        X_tr, y_tr, X_val, y_val, X_test, y_test, val_df, test_df, feature_cols, scale_pw = data

        # ─ Baseline ────────────────────────────────────────────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("BASELINE (current default params)")
        logger.info(f"{'═'*60}")
        baseline = self._compute_baseline(X_tr, y_tr, X_val, y_val, val_df, scale_pw, feature_cols)

        # ─ Model tuning ────────────────────────────────────────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("HYPERPARAMETER OPTIMISATION")
        logger.info(f"{'═'*60}")

        xgb_params  = self.tune_xgboost(X_tr, y_tr, X_val, y_val, val_df, scale_pw)
        lgbm_params = self.tune_lightgbm(X_tr, y_tr, X_val, y_val, val_df, scale_pw)
        rf_params   = self.tune_random_forest(X_tr, y_tr, X_val, y_val, val_df)

        self._best_params = {
            "xgboost": xgb_params,
            "lightgbm": lgbm_params,
            "random_forest": rf_params,
        }

        # ─ Threshold sweep ─────────────────────────────────────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("ENTRY THRESHOLD SWEEP")
        logger.info(f"{'═'*60}")
        threshold_sweep = self.sweep_entry_threshold(
            X_tr, y_tr, X_val, y_val, val_df, scale_pw,
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            rf_params=rf_params,
        )

        # ─ Summary ─────────────────────────────────────────────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("TUNING SUMMARY")
        logger.info(f"{'═'*60}")
        logger.info(f"{'Model':<16} {'Baseline':>10}  {'Tuned':>10}  {'Lift':>10}")
        logger.info("─" * 52)
        for model_name in ["xgboost", "lightgbm", "random_forest"]:
            baseline_v = baseline.get(model_name, {}).get("precision_top_k", 0.0)
            tuned_v    = self._study_results.get(model_name, {}).get("best_precision_top_k", 0.0)
            lift       = tuned_v - baseline_v
            logger.info(f"  {model_name:<14}  {baseline_v:>10.4f}  {tuned_v:>10.4f}  "
                        f"{lift:>+10.4f}")

        rec_thr = threshold_sweep["recommended_threshold"]
        logger.info(f"\nRecommended entry threshold: {rec_thr}")
        logger.info(f"Total tuning time: {(time.time()-t_total)/60:.1f} min")

        # ─ Save report ─────────────────────────────────────────────────────────
        report = {
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "n_trials_per_model":   self.n_trials,
            "objective":            f"precision@top-{self.top_k}/day",
            "n_features":           len(feature_cols),
            "train_rows":           len(X_tr),
            "val_rows":             len(X_val),
            "baseline":             baseline,
            "study_results":        self._study_results,
            "best_params":          self._best_params,
            "threshold_sweep":      threshold_sweep,
            "recommended_threshold":rec_thr,
            "total_elapsed_sec":    round(time.time() - t_total),
        }

        report_path = out / "tuning_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nFull report saved to {report_path}")

        # Also save a concise YAML-ready param block for easy copy-paste
        yaml_block_path = out / "best_params_yaml.txt"
        with open(yaml_block_path, "w") as f:
            f.write("# Paste into config/ml_settings.yaml → stacked_ensemble:\n\n")
            f.write("stacked_ensemble:\n")
            f.write("  xgboost:\n")
            for k, v in xgb_params.items():
                f.write(f"    {k}: {v}\n")
            f.write("  lightgbm:\n")
            for k, v in lgbm_params.items():
                f.write(f"    {k}: {v}\n")
            f.write("  random_forest:\n")
            for k, v in rf_params.items():
                f.write(f"    {k}: {v}\n")
            f.write(f"\n# Recommended entry threshold: {rec_thr}\n")
        logger.info(f"YAML param block saved to {yaml_block_path}")

        return report
