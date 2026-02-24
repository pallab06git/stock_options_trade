# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Market regime detection via KMeans clustering on SPY features.

Identifies four market regimes based on volatility and trend
characteristics, allowing the stacked ensemble to specialize its
predictions per market condition.

Regimes
-------
  0 — Low-vol Trending     : quiet market with a clear directional bias
  1 — Low-vol Mean-revert  : quiet market oscillating around a mean
  2 — High-vol Trending    : volatile market with directional momentum
  3 — High-vol Choppy      : volatile market with no clear direction

Output columns
--------------
  market_regime      : int (0–3), cluster assignment
  regime_confidence  : float (0–1), inverse normalized distance to centroid

Usage
-----
    from src.processing.regime_detector import RegimeDetector

    detector = RegimeDetector(n_regimes=4)
    detector.fit(train_spy_features)
    df = detector.predict(test_spy_features)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Features used for regime detection (must exist in the feature DataFrame)
_REGIME_FEATURES: List[str] = [
    "spy_vol_std_30m",
    "spy_hl_range_30m",
    "spy_vol_regime",
    "spy_return_30m",
    "spy_rsi_14",
    "spy_bb_pct_b",
]


class RegimeDetector:
    """KMeans-based market regime classifier.

    Parameters
    ----------
    n_regimes : int
        Number of clusters (default 4).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=random_state,
            n_init=10,
        )
        self._is_fitted = False
        self._feature_names = list(_REGIME_FEATURES)

    @property
    def feature_names(self) -> List[str]:
        """Return the list of SPY feature names used for regime detection."""
        return list(self._feature_names)

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit the regime detector on training SPY features.

        Args:
            df: Feature DataFrame containing the regime feature columns.

        Returns:
            self (for method chaining).
        """
        X = self._extract_features(df)
        X_scaled = self._scaler.fit_transform(X)
        self._kmeans.fit(X_scaled)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign regime labels and confidence to each row.

        Args:
            df: Feature DataFrame containing the regime feature columns.

        Returns:
            DataFrame with two added columns: market_regime, regime_confidence.
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector must be fit() before predict()")

        df = df.copy()
        X = self._extract_features(df)
        X_scaled = self._scaler.transform(X)

        labels = self._kmeans.predict(X_scaled)
        df["market_regime"] = labels

        # Confidence: inverse normalized distance to assigned centroid
        distances = self._kmeans.transform(X_scaled)  # (n_samples, n_clusters)
        assigned_distances = distances[np.arange(len(labels)), labels]

        # Normalize: 0 = farthest from centroid, 1 = at centroid
        max_dist = assigned_distances.max()
        if max_dist > 0:
            df["regime_confidence"] = 1.0 - (assigned_distances / max_dist)
        else:
            df["regime_confidence"] = 1.0

        return df

    def get_params(self) -> Dict[str, Any]:
        """Return serializable parameters for artifact storage."""
        return {
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate regime feature matrix.

        Missing columns are filled with 0.0 to handle warm-up rows
        at the start of a trading day.
        """
        cols = []
        for feat in self._feature_names:
            if feat in df.columns:
                cols.append(df[feat].fillna(0.0).values)
            else:
                cols.append(np.zeros(len(df)))
        return np.column_stack(cols)
