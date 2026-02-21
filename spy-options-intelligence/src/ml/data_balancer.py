# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Class-imbalance handling for options spike prediction training data.

The raw dataset is heavily imbalanced:
  ~1% positive (option spikes ≥20%)   vs   ~99% negative

Without balancing a classifier trained on this data will default to predicting
"no spike" for every row and still achieve 99% accuracy — which is useless.
This module provides two complementary strategies:

1. ``undersample_majority``
   Randomly downsample the majority class (no-spike rows) to match the
   minority class (spike rows) count.  Simple, fast, and works well as an
   initial baseline.

2. ``calculate_class_weights``
   Compute per-class weights so that the model penalises errors on the
   minority class more heavily.  This uses all training data (no rows are
   discarded) and is preferable when data is scarce.
   Formula (equivalent to sklearn's ``compute_class_weight('balanced', ...)``):
     weight_i = n_total / (n_classes × count_i)

Module-level functions
----------------------
  undersample_majority(df, target_col, random_state)  → DataFrame
  calculate_class_weights(df, target_col)              → {class: weight, ...}

DataBalancer class
------------------
Config-driven wrapper that reads strategy from config and exposes:
  balance(df)          → DataFrame (applies undersample or returns unchanged)
  get_class_weights(df)→ dict
  get_summary(df)      → dict (distribution stats, imbalance ratio)

Configuration keys (all optional):
  data_preparation.balance_method   str  "undersample" | "class_weights"
                                         default "undersample"
  data_preparation.random_state     int  default 42
  data_preparation.target_col       str  default "target"
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Module-level functions (primary API)
# ---------------------------------------------------------------------------


def undersample_majority(
    df: pd.DataFrame,
    target_col: str = "target",
    random_state: int = 42,
) -> pd.DataFrame:
    """Downsample the majority class to match the minority class size.

    Produces a balanced dataset where both classes have equal representation.
    The minority class is kept in full; a random subset of the majority class
    of the same size is drawn without replacement.  The result is shuffled.

    Args:
        df:           DataFrame containing the target column.
        target_col:   Name of the binary target column (default ``"target"``).
        random_state: Seed for reproducible sampling (default 42).

    Returns:
        Balanced DataFrame with ``len(minority_class) × 2`` rows.
        Row order is shuffled.  Index is reset to 0..n-1.

    Raises:
        ValueError: If ``target_col`` is not in ``df.columns``.
        ValueError: If ``df`` contains fewer than 2 distinct classes.

    Notes:
        - If the two classes already have equal counts the DataFrame is
          returned shuffled without any rows being discarded.
        - Empty input returns an empty DataFrame immediately.
    """
    _check_target_col(df, target_col)

    if df.empty:
        return df.copy()

    value_counts = df[target_col].value_counts()
    n_classes = len(value_counts)

    if n_classes < 2:
        logger.warning(
            f"undersample_majority: only {n_classes} class found in "
            f"'{target_col}' — returning df unchanged"
        )
        return df.copy().sample(frac=1, random_state=random_state).reset_index(drop=True)

    minority_class = value_counts.idxmin()
    majority_class = value_counts.idxmax()
    minority_count = int(value_counts[minority_class])
    majority_count = int(value_counts[majority_class])

    minority_df = df[df[target_col] == minority_class]
    majority_df = df[df[target_col] == majority_class]

    if majority_count <= minority_count:
        # Already balanced (or majority is actually smaller — unusual)
        logger.info(
            f"undersample_majority: classes already balanced "
            f"({minority_count} vs {majority_count}) — shuffling only"
        )
        return df.copy().sample(frac=1, random_state=random_state).reset_index(drop=True)

    majority_sampled = majority_df.sample(n=minority_count, random_state=random_state)

    balanced = pd.concat([minority_df, majority_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        f"undersample_majority: {len(df)} → {len(balanced)} rows "
        f"(majority class {majority_class} downsampled {majority_count} → {minority_count}; "
        f"minority class {minority_class} kept at {minority_count})"
    )
    return balanced


def calculate_class_weights(
    df: pd.DataFrame,
    target_col: str = "target",
) -> Dict[Any, float]:
    """Compute balanced class weights for imbalanced learning.

    Weights are calculated as:
      weight_i = n_total / (n_classes × count_i)

    This is numerically equivalent to sklearn's
    ``compute_class_weight('balanced', classes=classes, y=y)`` and can be
    passed directly to XGBoost's ``sample_weight`` parameter or used to
    derive ``scale_pos_weight``.

    Args:
        df:         DataFrame containing the target column.
        target_col: Name of the binary target column (default ``"target"``).

    Returns:
        Dict mapping each class label to its float weight, e.g.
        ``{0: 0.505, 1: 50.0}`` for a 99:1 imbalance.

    Raises:
        ValueError: If ``target_col`` is not in ``df.columns``.
        ValueError: If ``df`` is empty or contains zero samples for any class.
    """
    _check_target_col(df, target_col)

    if df.empty:
        raise ValueError("calculate_class_weights: DataFrame is empty")

    value_counts = df[target_col].value_counts()
    n_total = len(df)
    n_classes = len(value_counts)

    weights: Dict[Any, float] = {}
    for cls, count in value_counts.items():
        if count == 0:
            raise ValueError(
                f"calculate_class_weights: class {cls!r} has 0 samples"
            )
        weights[cls] = n_total / (n_classes * count)

    logger.info(
        f"calculate_class_weights: {dict(value_counts.items())} samples → "
        f"weights {{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(weights.items()))}}}"
    )
    return weights


# ---------------------------------------------------------------------------
# DataBalancer class (config-driven wrapper)
# ---------------------------------------------------------------------------


class DataBalancer:
    """Config-driven wrapper around ``undersample_majority`` and
    ``calculate_class_weights``.

    Reads balancing strategy from ``config["data_preparation"]`` and exposes
    a unified ``balance(df)`` entry point.

    Supported strategies
    --------------------
    ``"undersample"``    (default)
        Downsample the majority class to match the minority class size.
        The resulting DataFrame has equal class representation.

    ``"class_weights"``
        Do not modify the DataFrame.  Instead, use ``get_class_weights()``
        to retrieve per-sample weights that can be passed to the model
        (e.g. XGBoost ``sample_weight`` or ``scale_pos_weight``).
        Calling ``balance()`` with this strategy returns the DataFrame
        unchanged with a log message.

    Usage::

        balancer = DataBalancer(config)
        balanced_df = balancer.balance(train_df)
        # -- or --
        weights = balancer.get_class_weights(train_df)
        stats = balancer.get_summary(train_df)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged config dict.  Keys read from ``data_preparation``.
        """
        prep_cfg = config.get("data_preparation", {})
        self.balance_method: str = prep_cfg.get("balance_method", "undersample")
        self.random_state: int = prep_cfg.get("random_state", 42)
        self.target_col: str = prep_cfg.get("target_col", "target")

        valid = ("undersample", "class_weights")
        if self.balance_method not in valid:
            raise ValueError(
                f"DataBalancer: unknown balance_method {self.balance_method!r}. "
                f"Expected one of {valid}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured balancing strategy.

        ``"undersample"``   → calls ``undersample_majority`` and returns the
                              downsampled, shuffled DataFrame.
        ``"class_weights"`` → returns ``df`` unchanged (use ``get_class_weights``
                              separately to retrieve the weights).

        Args:
            df: Training DataFrame containing the target column.

        Returns:
            Balanced (or unchanged) DataFrame.
        """
        if self.balance_method == "undersample":
            return undersample_majority(df, self.target_col, self.random_state)

        # class_weights: DataFrame is not modified; caller gets weights separately
        logger.info(
            "DataBalancer.balance: strategy is 'class_weights' — "
            "DataFrame returned unchanged. Use get_class_weights() for weights."
        )
        return df.copy()

    def get_class_weights(self, df: pd.DataFrame) -> Dict[Any, float]:
        """Compute balanced class weights for the provided DataFrame.

        Returns a dict usable as XGBoost ``sample_weight`` after expansion,
        e.g. ``{0: 0.505, 1: 50.0}``.

        Args:
            df: DataFrame containing the target column.

        Returns:
            Dict mapping class label → float weight.
        """
        return calculate_class_weights(df, self.target_col)

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute class distribution statistics before balancing.

        Useful for deciding which balancing strategy is appropriate and for
        logging the imbalance level.

        Args:
            df: DataFrame containing the target column.

        Returns:
            Stats dict with keys:
              n_total          : total row count
              n_positive       : count of minority class (label=1)
              n_negative       : count of majority class (label=0)
              positive_rate    : fraction of positive labels (0.0–1.0)
              imbalance_ratio  : n_negative / n_positive (higher = more imbalanced)
              class_weights    : dict from ``calculate_class_weights``
              balance_method   : currently configured strategy
        """
        _check_target_col(df, target_col=self.target_col)

        n_total = len(df)
        n_positive = int((df[self.target_col] == 1).sum())
        n_negative = int((df[self.target_col] == 0).sum())
        positive_rate = n_positive / n_total if n_total > 0 else 0.0
        imbalance_ratio = (n_negative / n_positive) if n_positive > 0 else float("inf")

        try:
            weights = calculate_class_weights(df, self.target_col)
        except ValueError:
            weights = {}

        stats = {
            "n_total": n_total,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_rate": positive_rate,
            "imbalance_ratio": imbalance_ratio,
            "class_weights": weights,
            "balance_method": self.balance_method,
        }

        logger.info(
            f"DataBalancer summary: {n_total} rows | "
            f"{n_positive} positive ({positive_rate:.2%}) | "
            f"imbalance ratio {imbalance_ratio:.1f}:1"
        )
        return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_target_col(df: pd.DataFrame, target_col: str) -> None:
    """Raise ValueError if target_col is missing from df."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()!r}"
        )
