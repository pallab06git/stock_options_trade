# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Multi-model trainer: XGBoost + LightGBM + RandomForest with Optuna optimization.

Trains three model types simultaneously on the same feature set and target
column, each with Bayesian hyperparameter search via Optuna.  The Optuna
objective optimises precision at operating thresholds (0.70–0.90) rather
than at the default 0.50 cut-off, so the search directly targets the
high-precision regime used in production.

Model artifacts
---------------
Each trained model is returned as a dict compatible with the existing
pipeline (ModelComparator, full-comparison CLI, ML dashboard):

    {
        "model":              <fitted sklearn-compatible estimator>,
        "feature_cols":       [...],
        "best_params":        {...},
        "optimization_score": float,
        "model_type":         "xgboost" | "lightgbm" | "random_forest",
        "threshold":          float,   # default 0.50; caller may override
        "saved_at":           "ISO-8601",
        "target_col":         str,
    }

Usage
-----
    from src.ml.multi_model_trainer import MultiModelTrainer

    trainer = MultiModelTrainer(n_trials=30, cv_splits=3)
    artifacts = trainer.train(
        df=features_df,          # must include feature cols + target col
        target_col="target_sustained",
        feature_cols=feat_cols,  # list of column names to use as features
    )
    # artifacts is a dict: {"xgboost": artifact, "lightgbm": ..., "random_forest": ...}

    trainer.save_artifacts(artifacts, output_dir="models/sustained")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.ml.data_balancer import undersample_majority
from src.ml.data_splitter import time_based_split
from src.ml.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    XGBOOST_PARAM_SPACE,
    LIGHTGBM_PARAM_SPACE,
    RF_PARAM_SPACE,
    _OPTUNA_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Thresholds swept inside the Optuna objective — same range as production use
_OPERATING_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

# Minimum signals per fold for the objective to return a non-zero score
_MIN_SIGNALS_PER_FOLD = 10


# ---------------------------------------------------------------------------
# MultiModelTrainer
# ---------------------------------------------------------------------------


class MultiModelTrainer:
    """Train XGBoost, LightGBM, and RandomForest with Optuna optimisation.

    Parameters
    ----------
    n_trials:
        Number of Optuna trials per model (default 50).
    cv_splits:
        TimeSeriesSplit folds inside each Optuna trial (default 3).
    operating_thresholds:
        List of probability thresholds the objective evaluates.
        The score returned to Optuna is the *average* precision across these
        thresholds on folds that produce ≥ min_signals signals.
    min_signals:
        Minimum predicted positives per fold × threshold to count the
        fold's precision; folds with fewer signals contribute 0.0 to the
        average score.
    random_state:
        Seed for Optuna sampler and model random states (default 42).
    train_ratio:
        Fraction of data used for training in the internal split (default 0.80).
        The remainder becomes the internal validation set used only for
        computing the final optimisation score.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_splits: int = 3,
        operating_thresholds: Optional[List[float]] = None,
        min_signals: int = _MIN_SIGNALS_PER_FOLD,
        random_state: int = 42,
        train_ratio: float = 0.80,
    ) -> None:
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.operating_thresholds = operating_thresholds or _OPERATING_THRESHOLDS
        self.min_signals = min_signals
        self.random_state = random_state
        self.train_ratio = train_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Train all three model types and return their artifacts.

        Pipeline per model type:
          1. Chronological 80/20 split (training / internal-val).
          2. Undersample majority in *training* portion only.
          3. Optuna search (cv_splits-fold TimeSeriesSplit on training set).
          4. Retrain best params on full (undersampled) training set.
          5. Evaluate on (unbalanced) internal validation set.

        Args:
            df:           Full feature DataFrame containing ``feature_cols``
                          and ``target_col``.  Should be sorted by timestamp.
            target_col:   Column name of the binary label (0/1).
            feature_cols: List of feature column names to use.

        Returns:
            Dict mapping model type name → artifact dict.
            Keys: ``"xgboost"``, ``"lightgbm"``, ``"random_forest"``.
            Each artifact is compatible with ``ModelComparator.add_model()``.

        Raises:
            ValueError: If required columns are missing or target is all-zero.
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in df")

        missing_feat = [c for c in feature_cols if c not in df.columns]
        if missing_feat:
            raise ValueError(f"Missing feature columns: {missing_feat}")

        if int(df[target_col].sum()) == 0:
            raise ValueError(
                f"target_col '{target_col}' has no positive examples — "
                "cannot train a classifier"
            )

        X_all = df[feature_cols].fillna(0.0).values.astype(np.float32)
        y_all = df[target_col].values.astype(np.int8)

        # Chronological split: training portion vs internal-val
        n = len(df)
        train_end = int(n * self.train_ratio)
        X_train, y_train = X_all[:train_end], y_all[:train_end]
        X_val,   y_val   = X_all[train_end:], y_all[train_end:]

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                "Training portion has only one class — cannot train. "
                "Use a larger dataset or different split ratio."
            )

        # Balance training set (undersample majority)
        train_df_tmp = pd.DataFrame(X_train)
        train_df_tmp["__target__"] = y_train
        balanced_df = undersample_majority(train_df_tmp, "__target__", self.random_state)
        X_train_bal = balanced_df.drop(columns=["__target__"]).values.astype(np.float32)
        y_train_bal = balanced_df["__target__"].values.astype(np.int8)

        logger.info(
            f"MultiModelTrainer: {len(df)} rows | {len(X_train)} train "
            f"({len(X_train_bal)} after balance) | {len(X_val)} val | "
            f"{int(y_train_bal.sum())} train positives"
        )

        artifacts: Dict[str, Dict[str, Any]] = {}
        saved_at = datetime.now().isoformat(timespec="seconds")

        # ── XGBoost ───────────────────────────────────────────────────────
        artifacts["xgboost"] = self._train_xgboost(
            X_train_bal, y_train_bal, X_val, y_val,
            feature_cols, target_col, saved_at,
        )

        # ── LightGBM ──────────────────────────────────────────────────────
        artifacts["lightgbm"] = self._train_lightgbm(
            X_train_bal, y_train_bal, X_val, y_val,
            feature_cols, target_col, saved_at,
        )

        # ── RandomForest ──────────────────────────────────────────────────
        artifacts["random_forest"] = self._train_random_forest(
            X_train_bal, y_train_bal, X_val, y_val,
            feature_cols, target_col, saved_at,
        )

        return artifacts

    def train_call_put_models_separately(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Train separate XGBoost and LightGBM models for CALL and PUT options.

        Eliminates directional confusion: when a single model trains on both
        calls and puts simultaneously, SPY directional features become noise
        (rising SPY helps calls, hurts puts — equal and opposite).  Training
        type-specific models lets the directional features work as intended.

        ``contract_type`` (1=call, 0=put) is excluded from each model's
        feature set because it is constant within each subset.

        Args:
            df:           Full feature DataFrame with ``contract_type`` (1/0),
                          ``feature_cols``, and ``target_col``.
            target_col:   Column name of the binary label (0/1).
            feature_cols: Feature column names; ``contract_type`` is stripped
                          automatically.

        Returns:
            Dict with four keys — ``"xgboost_call"``, ``"xgboost_put"``,
            ``"lightgbm_call"``, ``"lightgbm_put"`` — each an artifact dict
            compatible with ``ModelComparator`` / ``save_artifacts``.

        Raises:
            ValueError: If ``contract_type`` column is missing or either type
                        subset has too few positive examples.
        """
        if "contract_type" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'contract_type' column (1=call, 0=put)."
            )
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in df")

        # Drop contract_type from features — it is constant within each subset
        feature_cols_split = [f for f in feature_cols if f != "contract_type"]

        SEP = "=" * 70
        print(f"\n{SEP}")
        print("  TRAINING SEPARATE CALL / PUT MODELS")
        print(SEP)

        call_df = df[df["contract_type"] == 1].reset_index(drop=True)
        put_df  = df[df["contract_type"] == 0].reset_index(drop=True)

        print(f"\n  Training data split:")
        print(f"    CALL rows : {len(call_df):,}  "
              f"({len(call_df)/max(len(df),1):.1%})  "
              f"positives={int(call_df[target_col].sum())}")
        print(f"    PUT  rows : {len(put_df):,}  "
              f"({len(put_df)/max(len(df),1):.1%})  "
              f"positives={int(put_df[target_col].sum())}")

        results: Dict[str, Dict[str, Any]] = {}
        saved_at = datetime.now().isoformat(timespec="seconds")

        for option_type, type_df in [("call", call_df), ("put", put_df)]:
            label = option_type.upper()
            print(f"\n{SEP}")
            print(f"  {label} MODEL TRAINING  ({len(type_df):,} rows)")
            print(SEP)

            if int(type_df[target_col].sum()) < 20:
                raise ValueError(
                    f"{label} subset has fewer than 20 positive examples "
                    f"({int(type_df[target_col].sum())}) — cannot train reliably."
                )

            X_all = type_df[feature_cols_split].fillna(0.0).values.astype(np.float32)
            y_all = type_df[target_col].values.astype(np.int8)

            # Chronological 80/20 split
            n = len(type_df)
            train_end = int(n * self.train_ratio)
            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_val,   y_val   = X_all[train_end:], y_all[train_end:]

            # Balance training portion only
            train_tmp = pd.DataFrame(X_train, columns=feature_cols_split)
            train_tmp["__target__"] = y_train
            balanced = undersample_majority(train_tmp, "__target__", self.random_state)
            X_train_bal = (
                balanced.drop(columns=["__target__"]).values.astype(np.float32)
            )
            y_train_bal = balanced["__target__"].values.astype(np.int8)

            logger.info(
                f"{label}: {len(type_df)} rows | {len(X_train)} train "
                f"({len(X_train_bal)} balanced) | {len(X_val)} val"
            )

            # ── XGBoost ────────────────────────────────────────────────
            print(f"\n  [{option_type}] XGBoost ({self.n_trials} Optuna trials)…")
            xgb_artifact = self._train_xgboost(
                X_train_bal, y_train_bal, X_val, y_val,
                feature_cols_split, target_col, saved_at,
            )
            xgb_artifact["model_name"]  = f"xgboost_{option_type}"
            xgb_artifact["option_type"] = option_type
            results[f"xgboost_{option_type}"] = xgb_artifact
            print(f"    Optuna score : {xgb_artifact['optimization_score']:.4f}")
            print(f"    Val prec@0.7 : {xgb_artifact['val_precision_at_0_70']:.4f}")

            # ── LightGBM ───────────────────────────────────────────────
            print(f"\n  [{option_type}] LightGBM ({self.n_trials} Optuna trials)…")
            lgbm_artifact = self._train_lightgbm(
                X_train_bal, y_train_bal, X_val, y_val,
                feature_cols_split, target_col, saved_at,
            )
            lgbm_artifact["model_name"]  = f"lightgbm_{option_type}"
            lgbm_artifact["option_type"] = option_type
            results[f"lightgbm_{option_type}"] = lgbm_artifact
            print(f"    Optuna score : {lgbm_artifact['optimization_score']:.4f}")
            print(f"    Val prec@0.7 : {lgbm_artifact['val_precision_at_0_70']:.4f}")

        print(f"\n{SEP}")
        print("  TRAINING COMPLETE")
        print(SEP)
        return results

    def save_artifacts(
        self,
        artifacts: Dict[str, Dict[str, Any]],
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """Persist each artifact as a .pkl file via joblib.

        Args:
            artifacts:  Dict returned by ``train()``.
            output_dir: Directory to save ``{model_type}.pkl`` files.

        Returns:
            Dict mapping model type name → saved path.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}
        for name, artifact in artifacts.items():
            path = out_dir / f"{name}.pkl"
            joblib.dump(artifact, path)
            logger.info(f"MultiModelTrainer: saved {name} → {path}")
            saved[name] = path
        return saved

    # ------------------------------------------------------------------
    # Internal per-model trainers
    # ------------------------------------------------------------------

    def _run_optuna(
        self,
        model_class,
        param_space: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        extra_params: Dict,
    ) -> Dict[str, Any]:
        """Run Optuna search and return best_params + best_score."""
        if not _OPTUNA_AVAILABLE:
            logger.warning(
                "optuna not available — skipping hyperparameter search, "
                "using default params."
            )
            return {"best_params": {}, "best_score": 0.0, "optimization_history": []}

        # Custom objective: precision averaged across operating thresholds
        def _objective(trial):
            params = {}
            for name, spec in param_space.items():
                if isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)
                elif isinstance(spec, tuple) and all(isinstance(v, int) for v in spec):
                    params[name] = trial.suggest_int(name, spec[0], spec[1])
                else:
                    params[name] = trial.suggest_float(name, spec[0], spec[1])

            all_params = {**params, **extra_params}

            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            fold_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train[train_idx], X_train[val_idx]
                y_tr, y_v = y_train[train_idx], y_train[val_idx]

                try:
                    model = model_class(**all_params)
                    model.fit(X_tr, y_tr)
                except Exception:
                    fold_scores.append(0.0)
                    continue

                probas = model.predict_proba(X_v)[:, 1]
                threshold_scores = []
                for t in self.operating_thresholds:
                    preds = (probas >= t).astype(int)
                    n_sig = int(preds.sum())
                    if n_sig < self.min_signals:
                        threshold_scores.append(0.0)
                        continue
                    try:
                        prec = precision_score(y_v, preds, zero_division=0)
                        threshold_scores.append(prec)
                    except Exception:
                        threshold_scores.append(0.0)

                fold_scores.append(float(np.mean(threshold_scores)))

            return float(np.mean(fold_scores)) if fold_scores else 0.0

        import optuna
        from optuna.samplers import TPESampler

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=False)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "optimization_history": [
                {"trial": t.number, "value": t.value}
                for t in study.trials
                if t.value is not None
            ],
        }

    def _train_xgboost(
        self,
        X_train, y_train, X_val, y_val,
        feature_cols, target_col, saved_at,
    ) -> Dict[str, Any]:
        from xgboost import XGBClassifier

        extra = {"verbosity": 0, "eval_metric": "logloss", "random_state": self.random_state}
        opt = self._run_optuna(XGBClassifier, XGBOOST_PARAM_SPACE, X_train, y_train, extra)

        best_params = {**opt["best_params"], **extra}
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        val_score = self._val_precision(model, X_val, y_val)
        logger.info(
            f"XGBoost: best_score={opt['best_score']:.4f} | "
            f"val_precision@0.70={val_score:.4f}"
        )

        return {
            "model": model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "optimization_score": opt["best_score"],
            "optimization_history": opt["optimization_history"],
            "model_type": "xgboost",
            "threshold": 0.50,
            "target_col": target_col,
            "val_precision_at_0_70": val_score,
            "saved_at": saved_at,
        }

    def _train_lightgbm(
        self,
        X_train, y_train, X_val, y_val,
        feature_cols, target_col, saved_at,
    ) -> Dict[str, Any]:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError(
                "lightgbm is required for MultiModelTrainer. "
                "Install with: pip install lightgbm"
            )

        extra = {"verbose": -1, "random_state": self.random_state}
        opt = self._run_optuna(LGBMClassifier, LIGHTGBM_PARAM_SPACE, X_train, y_train, extra)

        best_params = {**opt["best_params"], **extra}
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)

        val_score = self._val_precision(model, X_val, y_val)
        logger.info(
            f"LightGBM: best_score={opt['best_score']:.4f} | "
            f"val_precision@0.70={val_score:.4f}"
        )

        return {
            "model": model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "optimization_score": opt["best_score"],
            "optimization_history": opt["optimization_history"],
            "model_type": "lightgbm",
            "threshold": 0.50,
            "target_col": target_col,
            "val_precision_at_0_70": val_score,
            "saved_at": saved_at,
        }

    def _train_random_forest(
        self,
        X_train, y_train, X_val, y_val,
        feature_cols, target_col, saved_at,
    ) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier

        extra = {"random_state": self.random_state, "n_jobs": -1}
        opt = self._run_optuna(
            RandomForestClassifier, RF_PARAM_SPACE, X_train, y_train, extra
        )

        best_params = {**opt["best_params"], **extra}
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        val_score = self._val_precision(model, X_val, y_val)
        logger.info(
            f"RandomForest: best_score={opt['best_score']:.4f} | "
            f"val_precision@0.70={val_score:.4f}"
        )

        return {
            "model": model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "optimization_score": opt["best_score"],
            "optimization_history": opt["optimization_history"],
            "model_type": "random_forest",
            "threshold": 0.50,
            "target_col": target_col,
            "val_precision_at_0_70": val_score,
            "saved_at": saved_at,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _val_precision(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        threshold: float = 0.70,
    ) -> float:
        """Compute precision on the internal validation set at a given threshold."""
        if len(X_val) == 0:
            return 0.0
        try:
            probas = model.predict_proba(X_val)[:, 1]
            preds = (probas >= threshold).astype(int)
            return float(precision_score(y_val, preds, zero_division=0))
        except Exception:
            return 0.0
