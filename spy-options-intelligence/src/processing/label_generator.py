# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Binary label generation for options spike prediction.

Module-level function
---------------------
``generate_labels(df, threshold_pct, lookforward_minutes)`` is the primary
entry point.  It works on any DataFrame that contains at minimum:
  - ``timestamp`` : Unix millisecond timestamps (sorted ascending per ticker)
  - ``close``     : option close prices

If a ``ticker`` column is present the function computes labels independently
for each ticker group and then reassembles the rows in their original order.
This prevents a look-forward bar from one contract being used to label a bar
from a different contract.

Returned columns
----------------
  target          : int8  — 1 if price rose ≥ threshold_pct% within the
                             forward window, else 0
  max_gain_pct    : float — maximum % gain achieved in the forward window
                             (NaN if no future bars exist within the window)
  time_to_max_min : float — minutes from the entry bar to the bar with the
                             highest gain (NaN if no future bars)

LabelGenerator class
--------------------
Config-driven wrapper that reads ``threshold_pct`` and ``lookforward_minutes``
from the config dict and exposes two additional methods:

  ``generate(df)``             — apply generate_labels with stored params
  ``generate_for_file(path)``  — load a CSV / Parquet, apply labels, return df
  ``validate(df)``             — check label distribution & metadata completeness
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Module-level function (primary API)
# ---------------------------------------------------------------------------


def generate_labels(
    df: pd.DataFrame,
    threshold_pct: float = 20.0,
    lookforward_minutes: int = 120,
) -> pd.DataFrame:
    """Generate binary spike-prediction labels for an options price DataFrame.

    For each row at time T with option close price P, looks forward up to
    ``lookforward_minutes`` minutes and asks:
      "Does the close price exceed P × (1 + threshold_pct / 100) at any point?"

    If yes  → target = 1  (spike detected within the window)
    If no   → target = 0  (no qualifying spike)

    The computation uses numpy ``searchsorted`` for O(n log n) performance per
    ticker group, rather than an O(n²) nested loop.

    Args:
        df: DataFrame containing at minimum:
              ``timestamp`` (int, Unix ms, sorted ascending per ticker) and
              ``close``     (float, option close price).
            If a ``ticker`` column is present, labels are computed per-ticker
            group to prevent cross-contract data leakage.
        threshold_pct: Minimum % gain required for a positive label.
                       Default 20.0 (i.e. 20% gain).
        lookforward_minutes: Size of the forward look window in minutes.
                             Default 120.

    Returns:
        Copy of ``df`` with three new columns appended:
          ``target``          (int8)  — 1 / 0 spike indicator
          ``max_gain_pct``    (float) — best % gain achievable in the window
          ``time_to_max_min`` (float) — minutes to the bar with peak gain

    Raises:
        ValueError: If ``df`` is missing the required ``timestamp`` or
                    ``close`` columns.
    """
    _validate_input(df)

    if df.empty:
        result = df.copy()
        result["target"] = pd.array([], dtype="int8")
        result["max_gain_pct"] = pd.array([], dtype="float64")
        result["time_to_max_min"] = pd.array([], dtype="float64")
        return result

    if "ticker" in df.columns:
        return _generate_multi_ticker(df, threshold_pct, lookforward_minutes)

    return _generate_single_group(df, threshold_pct, lookforward_minutes)


# ---------------------------------------------------------------------------
# LabelGenerator class (config-driven wrapper)
# ---------------------------------------------------------------------------


class LabelGenerator:
    """Config-driven wrapper around ``generate_labels``.

    Reads ``threshold_pct`` and ``lookforward_minutes`` from the config dict
    so callers do not need to pass them explicitly at each call site.

    Configuration keys (all optional, fall back to defaults):
      feature_engineering.target_threshold_pct      float  default 20.0
      feature_engineering.target_lookforward_minutes int    default 120

    Usage::

        gen = LabelGenerator(config)
        labeled_df = gen.generate(option_df)
        stats = gen.validate(labeled_df)
        labeled_df = gen.generate_for_file("data/processed/features/2025-03-03_features.csv")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged config dict.
        """
        fe_cfg = config.get("feature_engineering", {})
        self.threshold_pct: float = fe_cfg.get("target_threshold_pct", 20.0)
        self.lookforward_minutes: int = fe_cfg.get("target_lookforward_minutes", 120)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ``generate_labels`` with the configured parameters.

        Args:
            df: DataFrame with ``timestamp`` and ``close`` columns.

        Returns:
            DataFrame with target, max_gain_pct, time_to_max_min columns.
        """
        return generate_labels(df, self.threshold_pct, self.lookforward_minutes)

    def generate_for_file(self, path: str | Path) -> pd.DataFrame:
        """Load a CSV or Parquet feature file, apply labels, and return.

        If the file already contains a ``target`` column it is overwritten so
        that labels are always consistent with the current configuration.

        Args:
            path: Path to a CSV (``.csv``) or Parquet (``.parquet``) file.
                  The file must have ``timestamp`` and ``close`` columns.

        Returns:
            DataFrame with labels applied (target, max_gain_pct,
            time_to_max_min).  Rows sorted by timestamp.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is missing required columns.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix!r} (expected .csv or .parquet)")

        # Drop existing label columns so they are regenerated cleanly
        for col in ("target", "max_gain_pct", "time_to_max_min"):
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.sort_values("timestamp").reset_index(drop=True)
        labeled = self.generate(df)

        logger.info(
            f"LabelGenerator: labeled {len(labeled)} rows from {path.name} "
            f"(positive rate {labeled['target'].mean():.4f})"
        )
        return labeled

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate label distribution and metadata completeness.

        Checks that ``target``, ``max_gain_pct``, and ``time_to_max_min``
        columns are present and have sane values.

        Args:
            df: Labeled DataFrame (output of ``generate`` or ``generate_for_file``).

        Returns:
            Stats dict with keys:
              n_total, n_positive, n_negative, positive_rate,
              coverage_pct (% of rows with a non-NaN max_gain_pct),
              max_gain_max, max_gain_mean, missing_columns (list)
        """
        stats: Dict[str, Any] = {
            "n_total": len(df),
            "n_positive": 0,
            "n_negative": 0,
            "positive_rate": 0.0,
            "coverage_pct": 0.0,
            "max_gain_max": float("nan"),
            "max_gain_mean": float("nan"),
            "missing_columns": [],
        }

        required = ["target", "max_gain_pct", "time_to_max_min"]
        missing = [c for c in required if c not in df.columns]
        stats["missing_columns"] = missing
        if missing:
            logger.warning(f"LabelGenerator.validate: missing columns {missing}")
            return stats

        stats["n_positive"] = int(df["target"].sum())
        stats["n_negative"] = int((df["target"] == 0).sum())
        if len(df) > 0:
            stats["positive_rate"] = stats["n_positive"] / len(df)

        non_nan = df["max_gain_pct"].notna()
        stats["coverage_pct"] = float(non_nan.mean() * 100)

        if non_nan.any():
            stats["max_gain_max"] = float(df.loc[non_nan, "max_gain_pct"].max())
            stats["max_gain_mean"] = float(df.loc[non_nan, "max_gain_pct"].mean())

        logger.info(
            f"LabelGenerator.validate: {stats['n_total']} rows, "
            f"{stats['n_positive']} positive ({stats['positive_rate']:.2%}), "
            f"coverage {stats['coverage_pct']:.1f}%"
        )
        return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are absent."""
    missing = [c for c in ("timestamp", "close") if c not in df.columns]
    if missing:
        raise ValueError(
            f"generate_labels requires columns {missing!r} — "
            f"got {df.columns.tolist()!r}"
        )


def _generate_multi_ticker(
    df: pd.DataFrame,
    threshold_pct: float,
    lookforward_minutes: int,
) -> pd.DataFrame:
    """Compute labels per-ticker, preserving the original row order.

    Grouping ensures that forward bars from ticker B are never used when
    labelling a bar belonging to ticker A.
    """
    labeled_parts = []
    for ticker, group_df in df.groupby("ticker", sort=False):
        group_sorted = group_df.sort_values("timestamp").reset_index(drop=True)
        labeled = _generate_single_group(group_sorted, threshold_pct, lookforward_minutes)
        labeled_parts.append(labeled)

    # Concatenate and restore the original row order
    combined = pd.concat(labeled_parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def _generate_single_group(
    df: pd.DataFrame,
    threshold_pct: float,
    lookforward_minutes: int,
) -> pd.DataFrame:
    """Core label computation for a single sorted DataFrame.

    Assumes ``df`` is sorted ascending by ``timestamp`` and belongs to one
    contract (or the caller does not care about ticker boundaries).

    Algorithm (O(n log n) via searchsorted):
      For each bar i:
        1. Binary-search for the index range (i+1 .. cutoff_idx) whose
           timestamps fall within the forward window.
        2. Compute gains relative to closes[i].
        3. Find the index of the maximum gain.
        4. Set target=1 if that maximum ≥ threshold_pct.

    Args:
        df:                  Single-ticker DataFrame sorted by timestamp.
        threshold_pct:       Minimum % gain for a positive label.
        lookforward_minutes: Forward window size in minutes.

    Returns:
        df with target, max_gain_pct, time_to_max_min columns appended.
    """
    df = df.copy()
    n = len(df)
    timestamps = df["timestamp"].values
    closes = df["close"].values
    lookforward_ms = lookforward_minutes * 60_000  # convert minutes → ms

    targets = np.zeros(n, dtype=np.int8)
    max_gains = np.full(n, np.nan, dtype=np.float64)
    times_to_max = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        entry_price = closes[i]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        entry_ts = timestamps[i]
        cutoff_ts = entry_ts + lookforward_ms

        # Binary search: slice [start_idx, end_idx) covers all future bars
        # within the forward window.
        start_idx = i + 1
        end_idx = int(np.searchsorted(timestamps, cutoff_ts, side="right"))

        if start_idx >= end_idx:
            # No future bars within the window (end of day or sparse data)
            continue

        future_closes = closes[start_idx:end_idx]
        future_ts = timestamps[start_idx:end_idx]

        gains = (future_closes - entry_price) / entry_price * 100

        best_idx = int(np.nanargmax(gains))
        max_gain = float(gains[best_idx])

        max_gains[i] = max_gain
        times_to_max[i] = (future_ts[best_idx] - entry_ts) / 60_000.0

        if max_gain >= threshold_pct:
            targets[i] = 1

    df["target"] = targets
    df["max_gain_pct"] = max_gains
    df["time_to_max_min"] = times_to_max

    return df
