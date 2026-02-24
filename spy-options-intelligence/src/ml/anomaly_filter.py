# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Anomaly filter using Isolation Forest to detect unusual market conditions.

Prevents the entry model from firing signals during market dislocations
(flash crashes, circuit breakers, extreme gap opens) where learned patterns
are unlikely to generalise.

Usage
-----
    from src.ml.anomaly_filter import AnomalyFilter

    af = AnomalyFilter(contamination=0.05, discount_factor=0.5)
    af.fit(X_train)  # fit on training SPY features only
    discounted_proba = af.apply_discount(proba, X_test)
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyFilter:
    """Isolation Forest anomaly detector for market microstructure.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalous samples in training data (0–0.5).
    discount_factor : float
        Multiplier applied to predicted probabilities for anomalous bars.
        E.g. 0.5 halves the signal confidence during unusual conditions.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        discount_factor: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.discount_factor = discount_factor
        self.random_state = random_state
        self._model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyFilter":
        """Fit the Isolation Forest on training features.

        Args:
            X: Feature matrix (n_samples, n_features). Should use the same
               feature columns as the entry model.

        Returns:
            self (for method chaining).
        """
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.fit(X_clean)
        self._is_fitted = True
        return self

    def is_anomalous(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask where True = anomalous bar.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Boolean array of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyFilter must be fit() before is_anomalous()")
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = self._model.predict(X_clean)
        return preds == -1  # -1 = anomaly, 1 = normal

    def apply_discount(
        self,
        probabilities: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Discount predicted probabilities for anomalous bars.

        Normal bars keep their original probability.
        Anomalous bars get probability * discount_factor.

        Args:
            probabilities: Predicted probabilities (n_samples,).
            X: Feature matrix (n_samples, n_features).

        Returns:
            Adjusted probabilities (n_samples,).
        """
        mask = self.is_anomalous(X)
        result = probabilities.copy()
        result[mask] = result[mask] * self.discount_factor
        return result

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (lower = more anomalous).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Score array of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyFilter must be fit() before anomaly_scores()")
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._model.decision_function(X_clean)
