# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Threshold optimisation and precision-recall sweep for options spike models.

Motivation
----------
Standard ML classification at threshold=0.50 maximises the F1 / accuracy
trade-off, but a trading signal generator has a very different loss function:
a **false positive** (entering a trade that doesn't pay off) is far more
expensive than a **false negative** (missing a profitable trade).

This module sweeps probability thresholds from 0.50 to 0.99 on a held-out
**validation** set (never the test set) and finds the lowest threshold that
meets a minimum precision requirement while retaining maximum recall.

Usage example
-------------
    from src.ml.evaluate import find_optimal_threshold_for_precision

    result = find_optimal_threshold_for_precision(
        model, X_val, y_val, min_precision=0.90
    )
    if result["achievable"]:
        print(f"Use threshold={result['optimal_threshold']:.2f} → "
              f"precision={result['achieved_precision']:.2%}, "
              f"recall={result['achieved_recall']:.2%}")
    else:
        print("90% precision not achievable — model not discriminating enough")

Public API
----------
  find_optimal_threshold_for_precision(model, X_val, y_val,
                                       min_precision=0.90, step=0.01)
      → dict with keys:
          achievable            bool
          optimal_threshold     float | None
          achieved_precision    float | None
          achieved_recall       float | None
          n_signals             int
          signal_rate           float
          analysis_df           pd.DataFrame  (full sweep; all thresholds)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger()


def find_optimal_threshold_for_precision(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_precision: float = 0.90,
    step: float = 0.01,
) -> Dict[str, Any]:
    """Sweep thresholds on the validation set to meet a minimum precision.

    Thresholds are tested from 0.50 to 0.99 (inclusive) in steps of *step*.
    For each threshold the model's probability outputs are binarised and
    precision / recall are computed.  The function then selects the **lowest**
    threshold whose precision ≥ *min_precision* — lowest threshold = highest
    recall = fewest missed opportunities while staying above the precision bar.

    Args:
        model:         Fitted classifier with a ``predict_proba`` method.
        X_val:         Feature array for the validation set (n_samples × n_feat).
        y_val:         Binary target array for the validation set (n_samples,).
        min_precision: Minimum acceptable precision (0.0–1.0).  Default 0.90.
        step:          Threshold sweep increment.  Default 0.01.

    Returns:
        Dict with the following keys:

        ``achievable``          bool — whether min_precision is reachable.
        ``optimal_threshold``   float | None — lowest threshold achieving it.
        ``achieved_precision``  float | None — actual precision at that threshold.
        ``achieved_recall``     float | None — recall at that threshold.
        ``n_signals``           int — number of signals fired at optimal threshold.
        ``signal_rate``         float — signals / total validation rows.
        ``analysis_df``         pd.DataFrame — full sweep table:
                                  threshold, precision, recall, n_signals,
                                  signal_rate, meets_requirement.

    Raises:
        ValueError: If X_val / y_val are empty or min_precision is outside (0, 1].
    """
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError(
            "find_optimal_threshold_for_precision: X_val and y_val must be non-empty"
        )
    if not (0.0 < min_precision <= 1.0):
        raise ValueError(
            f"find_optimal_threshold_for_precision: min_precision must be in (0, 1], "
            f"got {min_precision}"
        )

    y_proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    n_total = len(y_val)

    rows = []
    thresholds = np.round(np.arange(0.50, 1.00, step), decimals=2)
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        n_signals = int(y_pred.sum())

        if n_signals == 0:
            prec = float("nan")
            rec = 0.0
        else:
            prec = float(precision_score(y_val, y_pred, zero_division=0.0))
            rec = float(recall_score(y_val, y_pred, zero_division=0.0))

        rows.append(
            {
                "threshold": float(thresh),
                "precision": prec,
                "recall": rec,
                "n_signals": n_signals,
                "signal_rate": n_signals / n_total,
                "meets_requirement": (not np.isnan(prec)) and (prec >= min_precision),
            }
        )

    analysis_df = pd.DataFrame(rows)

    # Filter: must meet precision AND produce at least 1 signal
    valid = analysis_df[analysis_df["meets_requirement"] & (analysis_df["n_signals"] > 0)]

    logger.info(
        f"find_optimal_threshold_for_precision: swept {len(thresholds)} thresholds "
        f"[0.50–0.99] | min_precision={min_precision:.0%} | "
        f"valid thresholds={len(valid)}"
    )

    if valid.empty:
        logger.warning(
            f"find_optimal_threshold_for_precision: "
            f"min_precision={min_precision:.0%} is NOT achievable on this validation set. "
            f"Best achievable precision: "
            f"{analysis_df['precision'].max(skipna=True):.2%}"
        )
        return {
            "achievable": False,
            "optimal_threshold": None,
            "achieved_precision": None,
            "achieved_recall": None,
            "n_signals": 0,
            "signal_rate": 0.0,
            "analysis_df": analysis_df,
        }

    # Pick the row with highest recall among valid thresholds
    best_idx = valid["recall"].idxmax()
    best = valid.loc[best_idx]

    logger.info(
        f"find_optimal_threshold_for_precision: optimal threshold={best['threshold']:.2f} | "
        f"precision={best['precision']:.2%} | recall={best['recall']:.2%} | "
        f"n_signals={int(best['n_signals'])}"
    )

    return {
        "achievable": True,
        "optimal_threshold": float(best["threshold"]),
        "achieved_precision": float(best["precision"]),
        "achieved_recall": float(best["recall"]),
        "n_signals": int(best["n_signals"]),
        "signal_rate": float(best["signal_rate"]),
        "analysis_df": analysis_df,
    }
