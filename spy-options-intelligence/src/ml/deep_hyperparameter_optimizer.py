# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Deep hyperparameter search targeting maximum precision at 2–3 signals/day.

Three classes
-------------
DeepHyperparameterOptimizer
    Optuna-based exhaustive search for LightGBM and RandomForest.
    Optimises a composite score that rewards both high precision and a
    target signal frequency (default: 2.5 signals/day).

EnsembleStrategy
    Tests AND / OR / average-probability combinations of the two
    best-found models across multiple thresholds.

MonteCarloSimulator
    Simulates monthly straddle P&L with randomised position sizes,
    theta decay, and slippage noise across 1 000+ iterations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# DeepHyperparameterOptimizer
# ---------------------------------------------------------------------------


class DeepHyperparameterOptimizer:
    """Exhaustive Optuna search for maximum-precision magnitude predictors.

    The objective sweeps thresholds 0.85 → 0.999 and rewards configurations
    that produce 2–5 signals/day at the highest possible precision.

    Parameters
    ----------
    X_train, y_train:
        Training feature matrix and binary labels (numpy arrays).
    X_val, y_val:
        Validation feature matrix and binary labels (numpy arrays).
    val_days:
        Number of trading days represented by the validation set.
        Used to convert raw signal counts into signals-per-day.
    target_signals_per_day:
        Desired average daily signal count (default 2.5).
    min_precision:
        Minimum acceptable precision threshold — informational; not
        enforced as a hard cut inside the objective function.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_days: int,
        target_signals_per_day: float = 2.5,
        min_precision: float = 0.98,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.val_days = max(int(val_days), 1)
        self.target_signals_per_day = target_signals_per_day
        self.min_precision = min_precision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _objective_score(self, y_proba: np.ndarray) -> float:
        """Composite precision-volume score evaluated over a threshold sweep.

        Sweeps thresholds 0.85 → 0.999 in steps of 0.005 and returns the
        best composite score found across all thresholds.

        Scoring rules
        -------------
        signals/day < 1  → precision * 0.5
        signals/day > 5  → precision * 0.7
        1 ≤ spd ≤ 5      → precision * (0.7 + 0.3 * volume_score)

        where  volume_score = 1 − min(|spd − target| / target, 1.0)
        """
        best_score = 0.0
        target = self.target_signals_per_day

        for thresh in np.arange(0.85, 1.0, 0.005):
            preds = (y_proba >= thresh).astype(int)
            n_signals = int(preds.sum())
            if n_signals == 0:
                continue

            tp = int(((preds == 1) & (self.y_val == 1)).sum())
            precision = tp / n_signals
            spd = n_signals / self.val_days

            if spd < 1.0:
                score = precision * 0.5
            elif spd > 5.0:
                score = precision * 0.7
            else:
                volume_score = 1.0 - min(abs(spd - target) / target, 1.0)
                score = precision * (0.7 + 0.3 * volume_score)

            if score > best_score:
                best_score = score

        return best_score

    # ------------------------------------------------------------------
    # LightGBM optimisation
    # ------------------------------------------------------------------

    def optimize_lightgbm_precision(self, n_trials: int = 200) -> Dict[str, Any]:
        """Run Optuna Bayesian search for LightGBM hyperparameters.

        Parameter space
        ---------------
        n_estimators      : 100 – 1 000
        max_depth         : 3 – 15
        learning_rate     : 0.001 – 0.3  (log scale)
        num_leaves        : 15 – 255
        min_child_samples : 5 – 100
        min_child_weight  : 1e-3 – 10.0  (log scale)
        reg_alpha         : 0.0 – 10.0
        reg_lambda        : 0.0 – 10.0
        subsample         : 0.4 – 1.0
        colsample_bytree  : 0.4 – 1.0
        subsample_freq    : 0 – 10
        min_split_gain    : 0.0 – 1.0
        max_bin           : 63 – 511

        Fixed: random_state=42, verbose=-1, n_jobs=-1

        Returns
        -------
        dict with keys:  best_params, best_value, study
        """
        import optuna
        from lightgbm import LGBMClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params: Dict[str, Any] = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
                "max_depth":         trial.suggest_int("max_depth", 3, 15),
                "learning_rate":     trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight":  trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 10.0),
                "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "subsample_freq":    trial.suggest_int("subsample_freq", 0, 10),
                "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
                "max_bin":           trial.suggest_int("max_bin", 63, 511),
                # fixed
                "random_state": 42,
                "verbose":      -1,
                "n_jobs":       -1,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LGBMClassifier(**params)
                clf.fit(self.X_train, self.y_train)
                y_proba = clf.predict_proba(self.X_val)[:, 1]
            return self._objective_score(y_proba)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(
            "LightGBM optimisation complete: best_value=%.4f  (trial %d/%d)",
            study.best_value,
            study.best_trial.number,
            n_trials,
        )
        return {
            "best_params": study.best_params,
            "best_value":  study.best_value,
            "study":       study,
        }

    # ------------------------------------------------------------------
    # RandomForest optimisation
    # ------------------------------------------------------------------

    def optimize_randomforest_precision(self, n_trials: int = 200) -> Dict[str, Any]:
        """Run Optuna Bayesian search for RandomForest hyperparameters.

        Parameter space
        ---------------
        n_estimators         : 100 – 1 000
        max_depth            : 10 – 50
        min_samples_split    : 2 – 50
        min_samples_leaf     : 1 – 20
        max_features         : ['sqrt','log2', 0.3, 0.5, 0.7, 0.9]
        min_impurity_decrease: 0.0 – 0.1
        ccp_alpha            : 0.0 – 0.1
        bootstrap            : True | False
        max_samples          : 0.5 – 1.0  (only when bootstrap=True)

        Note: oob_score is NOT included — it fails when bootstrap=False.

        Returns
        -------
        dict with keys:  best_params, best_value, study
        """
        import optuna
        from sklearn.ensemble import RandomForestClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            params: Dict[str, Any] = {
                "n_estimators":           trial.suggest_int("n_estimators", 100, 1000),
                "max_depth":              trial.suggest_int("max_depth", 10, 50),
                "min_samples_split":      trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf":       trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features":           trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
                ),
                "min_impurity_decrease":  trial.suggest_float(
                    "min_impurity_decrease", 0.0, 0.1
                ),
                "ccp_alpha":              trial.suggest_float("ccp_alpha", 0.0, 0.1),
                "bootstrap":              bootstrap,
                "random_state":           42,
                "n_jobs":                 -1,
            }

            if bootstrap:
                params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = RandomForestClassifier(**params)
                clf.fit(self.X_train, self.y_train)
                y_proba = clf.predict_proba(self.X_val)[:, 1]

            return self._objective_score(y_proba)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(
            "RandomForest optimisation complete: best_value=%.4f  (trial %d/%d)",
            study.best_value,
            study.best_trial.number,
            n_trials,
        )
        return {
            "best_params": study.best_params,
            "best_value":  study.best_value,
            "study":       study,
        }


# ---------------------------------------------------------------------------
# EnsembleStrategy
# ---------------------------------------------------------------------------


class EnsembleStrategy:
    """Test AND / OR / avg-probability combinations of two fitted models.

    Three strategies are tested:
        AND  — fires a signal only when BOTH models predict ≥ threshold.
        OR   — fires when EITHER model predicts ≥ threshold.
        AVG  — fires when the average of both probabilities ≥ threshold.

    Each strategy is tested at the class-level threshold lists below.
    A combination is "viable" if its signals-per-day falls in
    [target * 0.5, target * 2.0].  The best viable combination is the
    one with the highest (precision, avg_magnitude) lexicographically.
    """

    _THRESHOLDS_AND: List[float] = [0.85, 0.90, 0.95, 0.97, 0.99]
    _THRESHOLDS_OR:  List[float] = [0.90, 0.95, 0.97]
    _THRESHOLDS_AVG: List[float] = [0.90, 0.95, 0.97, 0.99]

    def test_ensemble_combinations(
        self,
        lgbm_model,
        rf_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
        n_test_days: int,
        target_signals_per_day: float = 2.5,
    ) -> Dict[str, Any]:
        """Test all ensemble combinations across the three strategies.

        Parameters
        ----------
        lgbm_model:
            Fitted LightGBM classifier (has ``predict_proba`` method).
        rf_model:
            Fitted RandomForest classifier (has ``predict_proba`` method).
        X_test:
            Feature matrix for the test set.
        y_test:
            True binary labels for the test set.
        test_df:
            Test DataFrame with an ``abs_max_move_pct`` column aligned
            to X_test rows.  Must be ``reset_index(drop=True)`` before
            passing.
        n_test_days:
            Number of trading days in the test set.
        target_signals_per_day:
            Target average signals per day for the viability range.

        Returns
        -------
        dict with keys:
            results — List[Dict] of all strategy/threshold combinations.
            best    — Dict of the best viable result (None if none viable).
        """
        n_test_days = max(int(n_test_days), 1)
        min_spd = target_signals_per_day * 0.5
        max_spd = target_signals_per_day * 2.0

        lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
        rf_proba   = rf_model.predict_proba(X_test)[:, 1]
        avg_proba  = (lgbm_proba + rf_proba) / 2.0

        abs_magnitude = (
            test_df["abs_max_move_pct"].values
            if "abs_max_move_pct" in test_df.columns
            else np.zeros(len(y_test))
        )

        results: List[Dict[str, Any]] = []

        def _evaluate(mask: np.ndarray, strategy: str, threshold: float) -> Dict[str, Any]:
            n_signals = int(mask.sum())
            spd = n_signals / n_test_days

            if n_signals == 0:
                return {
                    "strategy":        strategy,
                    "threshold":       threshold,
                    "n_signals":       0,
                    "signals_per_day": 0.0,
                    "precision":       0.0,
                    "avg_magnitude":   0.0,
                    "viable":          False,
                    "signals_mask":    mask,
                }

            tp        = int(((mask) & (y_test == 1)).sum())
            precision = float(tp) / n_signals
            avg_mag   = float(abs_magnitude[mask].mean())
            viable    = min_spd <= spd <= max_spd

            return {
                "strategy":        strategy,
                "threshold":       threshold,
                "n_signals":       n_signals,
                "signals_per_day": float(spd),
                "precision":       float(precision),
                "avg_magnitude":   float(avg_mag),
                "viable":          viable,
                "signals_mask":    mask,
            }

        # AND: both models must predict >= threshold
        for thresh in self._THRESHOLDS_AND:
            mask = (lgbm_proba >= thresh) & (rf_proba >= thresh)
            results.append(_evaluate(mask, "AND", thresh))

        # OR: either model predicts >= threshold
        for thresh in self._THRESHOLDS_OR:
            mask = (lgbm_proba >= thresh) | (rf_proba >= thresh)
            results.append(_evaluate(mask, "OR", thresh))

        # AVG: average probability >= threshold
        for thresh in self._THRESHOLDS_AVG:
            mask = avg_proba >= thresh
            results.append(_evaluate(mask, "AVG", thresh))

        viable = [r for r in results if r["viable"]]
        best = (
            max(viable, key=lambda r: (r["precision"], r["avg_magnitude"]))
            if viable
            else None
        )

        logger.info(
            "Ensemble combinations tested: %d total, %d viable",
            len(results),
            len(viable),
        )
        if best is not None:
            logger.info(
                "Best: strategy=%s  threshold=%.2f  precision=%.3f  "
                "spd=%.2f  avg_mag=%.1f%%",
                best["strategy"],
                best["threshold"],
                best["precision"],
                best["signals_per_day"],
                best["avg_magnitude"],
            )

        return {"results": results, "best": best}


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------


class MonteCarloSimulator:
    """Monte Carlo P&L simulation for straddle option signals.

    Each simulation run samples randomised position sizes, theta decay,
    and slippage for every signal, then accumulates total P&L.  Running
    many simulations (default 1 000) produces a distribution of monthly
    P&L outcomes.
    """

    def simulate_monthly_pnl(
        self,
        test_df: pd.DataFrame,
        signals_mask: np.ndarray,
        position_size_range: Tuple[float, float] = (10_000.0, 15_000.0),
        n_simulations: int = 1000,
        n_test_months: float = 2.4,
    ) -> Dict[str, Any]:
        """Simulate monthly straddle P&L with randomised trade parameters.

        For each signal row the simulation samples:
            position  ~ Uniform(pos_min, pos_max)   [USD]
            theta     ~ Normal(-6%, 2%)              [% decay of losing leg]
            slippage  ~ Normal(-1%, 0.5%)            [% execution slippage]
            fill      ~ Bernoulli(0.95)              [95% fill rate]

        P&L per trade:
            TP  winner_pnl = position * (abs_max_move_pct + slippage) / 100
            FP  loser_pnl  = position * theta / 100
            Unfilled trades contribute $0.

        Monthly P&L is annualised as:
            monthly_pnl = total_simulation_pnl / n_test_months

        Parameters
        ----------
        test_df:
            Test DataFrame with ``abs_max_move_pct`` and
            ``target_magnitude`` columns, reset_index(drop=True) and
            aligned to ``signals_mask``.
        signals_mask:
            Boolean array (len == len(test_df)); True = model fires a
            signal on that row.
        position_size_range:
            (min, max) USD straddle budget — sampled Uniform per trade.
        n_simulations:
            Number of Monte Carlo iterations (default 1 000).
        n_test_months:
            Length of the test period in trading months — used to
            annualise total P&L to a monthly figure.

        Returns
        -------
        dict with keys:
            mean, median, std,
            percentiles (p5, p25, p75, p95),
            win_rate, n_signals, n_months
        """
        rng = np.random.default_rng(42)

        signal_rows = test_df[signals_mask].reset_index(drop=True)
        n_signals   = len(signal_rows)

        if n_signals == 0:
            return {
                "mean":   0.0,
                "median": 0.0,
                "std":    0.0,
                "percentiles": {"p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0},
                "win_rate": 0.0,
                "n_signals": 0,
                "n_months":  float(n_test_months),
            }

        abs_moves = signal_rows["abs_max_move_pct"].values.astype(float)
        is_tp     = (signal_rows["target_magnitude"].values == 1)

        pos_min, pos_max = position_size_range
        monthly_pnls: List[float] = []
        sim_win_rates: List[float] = []

        for _ in range(n_simulations):
            positions = rng.uniform(pos_min, pos_max, size=n_signals)
            thetas    = rng.normal(-6.0, 2.0, size=n_signals)    # % (always negative intent)
            slippages = rng.normal(-1.0, 0.5, size=n_signals)    # % execution slippage
            fills     = rng.random(size=n_signals) < 0.95        # 95% fill rate

            winner_pnl = positions * (abs_moves + slippages) / 100.0
            loser_pnl  = positions * thetas / 100.0

            trade_pnl = np.where(is_tp, winner_pnl, loser_pnl)
            trade_pnl = trade_pnl * fills   # unfilled → $0

            total_pnl  = float(trade_pnl.sum())
            monthly_pnl = total_pnl / max(float(n_test_months), 1e-9)
            monthly_pnls.append(monthly_pnl)

            filled         = fills.sum()
            filled_tp      = (fills & is_tp).sum()
            sim_win_rates.append(float(filled_tp) / max(float(filled), 1.0))

        arr = np.array(monthly_pnls)
        return {
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "percentiles": {
                "p5":  float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            },
            "win_rate":  float(np.mean(sim_win_rates)),
            "n_signals": int(n_signals),
            "n_months":  float(n_test_months),
        }
