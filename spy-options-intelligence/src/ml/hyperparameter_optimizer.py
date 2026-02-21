# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Automated hyperparameter optimization using Optuna Bayesian search.

Supports tree-based models (XGBoost, LightGBM, RandomForest) via
TimeSeriesSplit cross-validation and LSTM via PyTorch-based search.

Parameter space format
----------------------
Each entry in ``param_space`` is either:

* A **list of discrete values** → ``suggest_categorical``  (e.g. ``[100, 200, 300]``)
* A **2-tuple (low, high)** of ints  → ``suggest_int``     (e.g. ``(1, 8)``)
* A **2-tuple (low, high)** of floats → ``suggest_float``  (e.g. ``(0.01, 0.2)``)

Usage
-----
    from src.ml.hyperparameter_optimizer import HyperparameterOptimizer
    from xgboost import XGBClassifier

    optimizer = HyperparameterOptimizer(
        model_class=XGBClassifier,
        param_space={
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
        },
        X_train=X,
        y_train=y,
        n_trials=50,
        model_extra_params={'verbosity': 0},
    )
    results = optimizer.optimize()
    optimizer.save_results('reports/xgb_optimization.json')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Supported param spec formats
ParamSpec = Union[List, Tuple[int, int], Tuple[float, float]]

_SUPPORTED_METRICS = {"precision", "f1", "roc_auc"}

# Default XGBoost search space (matches xgboost_v2 neighbourhood)
XGBOOST_PARAM_SPACE: Dict[str, ParamSpec] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5],
    "scale_pos_weight": [5, 10, 15, 20],
}

LIGHTGBM_PARAM_SPACE: Dict[str, ParamSpec] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 30, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0, 0.1, 0.5, 1.0],
}

RF_PARAM_SPACE: Dict[str, ParamSpec] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
}

LSTM_PARAM_SPACE: Dict[str, ParamSpec] = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [32, 64, 128],
    "sequence_length": [30, 60, 90, 120],
    "bidirectional": [True, False],
}


# ---------------------------------------------------------------------------
# HyperparameterOptimizer — tree-based models
# ---------------------------------------------------------------------------


class HyperparameterOptimizer:
    """Optimize tree-based model hyperparameters using Optuna TPE sampler.

    Parameters
    ----------
    model_class:
        Uninstantiated model class (e.g. ``XGBClassifier``).
    param_space:
        Dict mapping parameter name → discrete list OR (low, high) range.
    X_train, y_train:
        Training data.  Uses ``cv_splits``-fold TimeSeriesSplit for CV.
    optimization_metric:
        One of ``"precision"``, ``"f1"``, ``"roc_auc"``.
    n_trials:
        Number of Optuna optimization trials.
    cv_splits:
        Number of time-series CV folds (default 5).
    min_signals_per_fold:
        Minimum predicted positives per fold; trials producing fewer are
        scored 0.0 (effectively pruned).
    model_extra_params:
        Fixed kwargs always passed to the model constructor, e.g.
        ``{"verbosity": 0}`` for XGBoost or ``{"verbose": -1}`` for LGBM.
    predict_threshold:
        Threshold for ``predict_proba[:, 1]`` → binary prediction
        (default 0.5 = standard ``predict`` behaviour).
    random_state:
        Optuna sampler seed (default 42).
    """

    def __init__(
        self,
        model_class: Callable,
        param_space: Dict[str, ParamSpec],
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimization_metric: str = "precision",
        n_trials: int = 100,
        cv_splits: int = 5,
        min_signals_per_fold: int = 3,
        model_extra_params: Optional[Dict[str, Any]] = None,
        predict_threshold: float = 0.5,
        random_state: int = 42,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for HyperparameterOptimizer. "
                "Install with: pip install optuna"
            )
        if optimization_metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"optimization_metric must be one of {_SUPPORTED_METRICS}, "
                f"got '{optimization_metric}'"
            )

        self.model_class = model_class
        self.param_space = param_space
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.min_signals_per_fold = min_signals_per_fold
        self.model_extra_params: Dict[str, Any] = model_extra_params or {}
        self.predict_threshold = predict_threshold
        self.random_state = random_state

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.study: Optional["optuna.Study"] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Sample one set of hyperparameters from the param_space."""
        params: Dict[str, Any] = {}
        for name, spec in self.param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(
                        name, float(low), float(high)
                    )
            else:
                raise ValueError(
                    f"Invalid param spec for '{name}': {spec!r}. "
                    "Use a list of discrete values or a (low, high) tuple."
                )
        return params

    def _predict_binary(
        self, model: Any, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (y_pred_binary, y_proba_or_None)."""
        try:
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= self.predict_threshold).astype(np.int8)
            return y_pred, y_proba
        except (AttributeError, NotImplementedError):
            y_pred = model.predict(X).astype(np.int8)
            return y_pred, None

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> float:
        """Compute the chosen metric for one CV fold."""
        try:
            if self.optimization_metric == "precision":
                return float(precision_score(y_true, y_pred, zero_division=0.0))
            elif self.optimization_metric == "f1":
                return float(f1_score(y_true, y_pred, zero_division=0.0))
            elif self.optimization_metric == "roc_auc":
                if y_proba is None or len(np.unique(y_true)) < 2:
                    return 0.5
                return float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective: mean metric across TimeSeriesSplit folds."""
        params = {**self._sample_params(trial), **self.model_extra_params}
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        fold_scores: List[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_tr, y_tr = self.X_train[tr_idx], self.y_train[tr_idx]
            X_val, y_val = self.X_train[val_idx], self.y_train[val_idx]

            if (y_tr == 1).sum() < 10:
                return 0.0

            try:
                model = self.model_class(**params)
                model.fit(X_tr, y_tr)
            except Exception as exc:
                logger.debug(
                    "Trial %d fold %d fit failed: %s", trial.number, fold_idx, exc
                )
                return 0.0

            y_pred, y_proba = self._predict_binary(model, X_val)
            n_signals = int(y_pred.sum())

            if n_signals < self.min_signals_per_fold:
                fold_scores.append(0.0)
                continue

            fold_scores.append(self._compute_metric(y_val, y_pred, y_proba))

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization.

        Returns
        -------
        Dict with ``best_params``, ``best_score``, ``n_trials``, and
        ``optimization_history``.
        """
        model_name = getattr(self.model_class, "__name__", str(self.model_class))
        logger.info(
            "Starting optimization: model=%s metric=%s trials=%d",
            model_name,
            self.optimization_metric,
            self.n_trials,
        )

        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = {**self.study.best_params, **self.model_extra_params}
        self.best_score = self.study.best_value

        logger.info(
            "Optimization complete: best_%s=%.4f params=%s",
            self.optimization_metric,
            self.best_score,
            self.study.best_params,
        )

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "model_class": model_name,
            "optimization_metric": self.optimization_metric,
            "n_trials": len(self.study.trials),
            "optimization_history": [
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in self.study.trials
            ],
        }

    def save_results(self, output_path: str | Path) -> None:
        """Persist optimization results to a JSON file."""
        if self.study is None:
            raise RuntimeError("Call optimize() before save_results()")

        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "model_class": getattr(self.model_class, "__name__", str(self.model_class)),
            "optimization_metric": self.optimization_metric,
            "n_trials": len(self.study.trials),
            "optimization_history": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in self.study.trials
            ],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Saved optimization results → %s", path)


# ---------------------------------------------------------------------------
# LSTMHyperparameterOptimizer — PyTorch LSTM
# ---------------------------------------------------------------------------


class LSTMHyperparameterOptimizer:
    """Optimize LSTM hyperparameters using Optuna (requires PyTorch).

    Parameters
    ----------
    X_train, y_train:
        2-D feature array and labels.  Sequences are created internally.
    param_space:
        Same format as ``HyperparameterOptimizer``.
        Must include ``sequence_length``, ``learning_rate``, ``batch_size``.
    n_trials:
        Optuna trials (default 50 — fewer than tree models, slower training).
    cv_splits:
        Time-series CV splits (default 3).
    train_epochs:
        Fixed number of training epochs per fold during search (default 15).
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if not specified.
    random_state:
        Optuna sampler seed.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict[str, ParamSpec]] = None,
        n_trials: int = 50,
        cv_splits: int = 3,
        train_epochs: int = 15,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required. Install with: pip install optuna")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for LSTM optimization. "
                "Install with: pip install torch"
            ) from exc

        import torch

        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        self.param_space = param_space or dict(LSTM_PARAM_SPACE)
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.train_epochs = train_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self.param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
        return params

    def objective(self, trial: "optuna.Trial") -> float:
        """Train a small LSTM and return mean precision across folds."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from src.ml.lstm_model import LSTMModel

        params = self._sample_params(trial)
        seq_len: int = int(params.pop("sequence_length", 60))
        lr: float = float(params.pop("learning_rate", 0.001))
        batch_size: int = int(params.pop("batch_size", 64))

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        fold_scores: List[float] = []

        for tr_idx, val_idx in tscv.split(self.X_train):
            X_tr_seq, y_tr_seq = make_sequences(self.X_train[tr_idx], self.y_train[tr_idx], seq_len)
            X_val_seq, y_val_seq = make_sequences(self.X_train[val_idx], self.y_train[val_idx], seq_len)

            if len(X_tr_seq) == 0 or (y_tr_seq == 1).sum() < 5:
                return 0.0

            model = LSTMModel(input_size=X_tr_seq.shape[2], **params).to(self.device)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()

            ds = TensorDataset(
                torch.FloatTensor(X_tr_seq),
                torch.FloatTensor(y_tr_seq.reshape(-1, 1)),
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            model.train()
            for _ in range(self.train_epochs):
                for Xb, yb in loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimiser.zero_grad()
                    criterion(model(Xb), yb).backward()
                    optimiser.step()

            model.eval()
            with torch.no_grad():
                proba = (
                    model(torch.FloatTensor(X_val_seq).to(self.device))
                    .cpu()
                    .numpy()
                    .flatten()
                )
            y_pred = (proba >= 0.5).astype(np.int8)
            n_signals = int(y_pred.sum())
            if n_signals < 3:
                fold_scores.append(0.0)
            else:
                fold_scores.append(
                    float(precision_score(y_val_seq, y_pred, zero_division=0))
                )

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    def optimize(self) -> Dict[str, Any]:
        """Run LSTM hyperparameter optimization."""
        logger.info("Starting LSTM optimization: n_trials=%d", self.n_trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        self.best_score = study.best_value
        logger.info("LSTM optimization complete: best_precision=%.4f", self.best_score)
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "model_class": "LSTMModel",
            "n_trials": len(study.trials),
        }


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D feature array into (N, seq_len, n_features) sequences.

    The label for each sequence is the target of the *last* row in the window.
    Sequences are created with a stride of 1 (dense sliding window).

    Parameters
    ----------
    X:
        2-D array of shape (n_samples, n_features).
    y:
        1-D label array of length n_samples.
    seq_len:
        Number of timesteps per sequence.

    Returns
    -------
    (X_seq, y_seq) where X_seq has shape (n_samples - seq_len, seq_len, n_features)
    and y_seq has shape (n_samples - seq_len,).
    Returns empty arrays if ``len(X) <= seq_len``.
    """
    n = len(X)
    if n <= seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )
    X_seq = np.stack(
        [X[i : i + seq_len] for i in range(n - seq_len)], axis=0
    ).astype(np.float32)
    y_seq = y[seq_len:].astype(np.int8)
    return X_seq, y_seq
