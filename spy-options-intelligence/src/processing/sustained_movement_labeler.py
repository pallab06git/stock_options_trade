# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Sustained upward movement label generator.

Rather than asking "did the price ever spike ≥20% in 120 minutes?" (the
existing binary target), this module asks a softer, more nuanced question:

    "Did the option price rise, and then *stay* above the entry price for
    at least ``sustain_minutes`` consecutive minutes, confirmed after a
    ``confirmation_minutes`` buffer?"

This captures genuine directional moves that hold — not just momentary
spikes that immediately reverse.

Algorithm (per bar i, per ticker)
----------------------------------
  1. Entry price  P  = close[i]
  2. Look forward exactly ``confirmation_minutes`` minutes to find the
     *confirmation bar* (first bar whose timestamp ≥ entry_ts + conf_ms).
  3. If close[conf_idx] > P  →  check consecutive sustain:
       Starting at conf_idx, walk forward bar by bar.
       Count consecutive bars where close > P.
       If count ≥ sustain_minutes  →  target_sustained = 1.
  4. Record:
       gain_pct_at_confirmation : % gain at the confirmation bar (NaN if
                                   no confirmation bar exists)
       magnitude_bucket         : category of the gain at confirmation
       sustain_minutes_actual   : consecutive bars above entry price
                                   starting at the confirmation bar

Magnitude buckets
-----------------
  'below_zero'  : gain < 0  (price fell at confirmation)
  '0-1%'        : 0 ≤ gain < 1
  '1-5%'        : 1 ≤ gain < 5
  '5-10%'       : 5 ≤ gain < 10
  '10-20%'      : 10 ≤ gain < 20
  '20%+'        : gain ≥ 20

Output columns
--------------
  target_sustained          int8    — 1 = sustained move, 0 = no
  gain_pct_at_confirmation  float   — % gain at the confirmation bar
  magnitude_bucket          str     — one of the six buckets above
  sustain_minutes_actual    float   — consecutive minutes above entry
                                       at confirmation bar (0 if conf bar
                                       price ≤ entry)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Ordered bucket boundaries: (label, lower_bound, upper_bound)
# lower_bound is inclusive, upper_bound is exclusive (except last)
_BUCKETS = [
    ("below_zero", -np.inf, 0.0),
    ("0-1%",        0.0,    1.0),
    ("1-5%",        1.0,    5.0),
    ("5-10%",       5.0,   10.0),
    ("10-20%",     10.0,   20.0),
    ("20%+",       20.0,   np.inf),
]

MAGNITUDE_BUCKETS = [b[0] for b in _BUCKETS]


# ---------------------------------------------------------------------------
# Module-level function (primary API)
# ---------------------------------------------------------------------------


def generate_sustained_labels(
    df: pd.DataFrame,
    confirmation_minutes: int = 15,
    sustain_minutes: int = 5,
) -> pd.DataFrame:
    """Generate sustained-movement labels for an options price DataFrame.

    For each row at time T with option close price P:
      - Find the bar at T + confirmation_minutes
      - If that bar's close > P, count consecutive bars above P
      - If consecutive count ≥ sustain_minutes → target_sustained = 1

    Per-ticker isolation: if a ``ticker`` column is present, labels are
    computed independently per ticker to prevent cross-contract leakage.

    Args:
        df: DataFrame with at minimum ``timestamp`` (Unix ms, sorted ascending
            per ticker) and ``close`` (option close price).  Optional
            ``ticker`` column enables per-ticker label isolation.
        confirmation_minutes: Minutes after entry to check the confirmation
            bar.  Default 15.
        sustain_minutes: Minimum consecutive bars above entry price at
            confirmation required for a positive label.  Default 5.

    Returns:
        Copy of ``df`` with four new columns:
          target_sustained          (int8)
          gain_pct_at_confirmation  (float)
          magnitude_bucket          (str)
          sustain_minutes_actual    (float)

    Raises:
        ValueError: If ``df`` is missing required columns.
    """
    _validate_input(df)

    if df.empty:
        result = df.copy()
        result["target_sustained"] = pd.array([], dtype="int8")
        result["gain_pct_at_confirmation"] = pd.array([], dtype="float64")
        result["magnitude_bucket"] = pd.array([], dtype="object")
        result["sustain_minutes_actual"] = pd.array([], dtype="float64")
        return result

    if "ticker" in df.columns:
        return _generate_multi_ticker(df, confirmation_minutes, sustain_minutes)

    return _generate_single_group(df, confirmation_minutes, sustain_minutes)


def classify_magnitude(gain_pct: float) -> str:
    """Return the magnitude bucket label for a given % gain.

    Args:
        gain_pct: The percentage gain at the confirmation bar.

    Returns:
        One of: 'below_zero', '0-1%', '1-5%', '5-10%', '10-20%', '20%+'.
    """
    if np.isnan(gain_pct):
        return "below_zero"
    for label, lo, hi in _BUCKETS:
        if lo <= gain_pct < hi:
            return label
    return "20%+"  # fallback for exactly 20% (should be caught above)


# ---------------------------------------------------------------------------
# SustainedMovementLabeler class (config-driven wrapper)
# ---------------------------------------------------------------------------


class SustainedMovementLabeler:
    """Config-driven wrapper around ``generate_sustained_labels``.

    Reads ``confirmation_minutes`` and ``sustain_minutes`` from the config
    dict so callers do not need to pass them at each call site.

    Configuration keys (all optional, fall back to defaults):
      sustained_movement.confirmation_minutes  int  default 15
      sustained_movement.sustain_minutes       int  default 5

    Usage::

        labeler = SustainedMovementLabeler(config)
        labeled_df = labeler.label(df)
        stats = labeler.validate(labeled_df)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            config: Merged config dict.  Reads
                    ``sustained_movement.confirmation_minutes`` and
                    ``sustained_movement.sustain_minutes``.
                    Defaults are used if config is None or keys are absent.
        """
        cfg = (config or {}).get("sustained_movement", {})
        self.confirmation_minutes: int = int(cfg.get("confirmation_minutes", 15))
        self.sustain_minutes: int = int(cfg.get("sustain_minutes", 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ``generate_sustained_labels`` with the configured parameters.

        Args:
            df: DataFrame with ``timestamp`` and ``close`` columns.

        Returns:
            DataFrame with target_sustained, gain_pct_at_confirmation,
            magnitude_bucket, sustain_minutes_actual columns appended.
        """
        return generate_sustained_labels(
            df,
            confirmation_minutes=self.confirmation_minutes,
            sustain_minutes=self.sustain_minutes,
        )

    def label_for_file(self, path: str | Path) -> pd.DataFrame:
        """Load a CSV or Parquet feature file, apply labels, and return.

        Overwrites any existing sustained-movement label columns so labels
        are always consistent with the current configuration.

        Args:
            path: Path to a CSV or Parquet feature file with ``timestamp``
                  and ``close`` columns.

        Returns:
            DataFrame with sustained movement labels applied.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If file is missing required columns or format.
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
            raise ValueError(f"Unsupported file format: {suffix!r}")

        # Drop existing label columns for clean regeneration
        for col in (
            "target_sustained",
            "gain_pct_at_confirmation",
            "magnitude_bucket",
            "sustain_minutes_actual",
        ):
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.sort_values("timestamp").reset_index(drop=True)
        labeled = self.label(df)

        n_pos = int(labeled["target_sustained"].sum())
        logger.info(
            f"SustainedMovementLabeler: {len(labeled)} rows from {path.name} "
            f"| {n_pos} positive ({n_pos / max(len(labeled), 1):.2%}) "
            f"| conf={self.confirmation_minutes}min sustain={self.sustain_minutes}min"
        )
        return labeled

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate label distribution and magnitude breakdown.

        Args:
            df: Labeled DataFrame (output of ``label`` or ``label_for_file``).

        Returns:
            Stats dict with keys:
              n_total, n_positive, n_negative, positive_rate,
              coverage_pct (% of rows with a non-NaN gain_pct),
              missing_columns (list),
              magnitude_breakdown (dict: bucket → count)
        """
        stats: Dict[str, Any] = {
            "n_total": len(df),
            "n_positive": 0,
            "n_negative": 0,
            "positive_rate": 0.0,
            "coverage_pct": 0.0,
            "missing_columns": [],
            "magnitude_breakdown": {b: 0 for b in MAGNITUDE_BUCKETS},
        }

        required = [
            "target_sustained",
            "gain_pct_at_confirmation",
            "magnitude_bucket",
            "sustain_minutes_actual",
        ]
        missing = [c for c in required if c not in df.columns]
        stats["missing_columns"] = missing
        if missing:
            logger.warning(f"SustainedMovementLabeler.validate: missing {missing}")
            return stats

        stats["n_positive"] = int(df["target_sustained"].sum())
        stats["n_negative"] = int((df["target_sustained"] == 0).sum())
        if len(df) > 0:
            stats["positive_rate"] = stats["n_positive"] / len(df)

        non_nan = df["gain_pct_at_confirmation"].notna()
        stats["coverage_pct"] = float(non_nan.mean() * 100)

        # Magnitude breakdown
        for bucket in MAGNITUDE_BUCKETS:
            mask = df["magnitude_bucket"] == bucket
            stats["magnitude_breakdown"][bucket] = int(mask.sum())

        logger.info(
            f"SustainedMovementLabeler.validate: {stats['n_total']} rows | "
            f"{stats['n_positive']} positive ({stats['positive_rate']:.2%}) | "
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
            f"generate_sustained_labels requires columns {missing!r} — "
            f"got {df.columns.tolist()!r}"
        )


def _generate_multi_ticker(
    df: pd.DataFrame,
    confirmation_minutes: int,
    sustain_minutes: int,
) -> pd.DataFrame:
    """Compute labels per-ticker, preserving original row order."""
    labeled_parts = []
    for _ticker, group_df in df.groupby("ticker", sort=False):
        group_sorted = group_df.sort_values("timestamp").reset_index(drop=True)
        labeled = _generate_single_group(
            group_sorted, confirmation_minutes, sustain_minutes
        )
        labeled_parts.append(labeled)

    combined = pd.concat(labeled_parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def _generate_single_group(
    df: pd.DataFrame,
    confirmation_minutes: int,
    sustain_minutes: int,
) -> pd.DataFrame:
    """Core label computation for a single sorted DataFrame.

    Assumes ``df`` is sorted ascending by ``timestamp`` and belongs to one
    contract (or the caller does not care about ticker boundaries).

    Args:
        df:                   Single-ticker DataFrame sorted by timestamp.
        confirmation_minutes: Minutes to the confirmation check point.
        sustain_minutes:      Min consecutive bars above entry to be positive.

    Returns:
        df with four new label columns appended.
    """
    df = df.copy()
    n = len(df)
    timestamps = df["timestamp"].values
    closes = df["close"].values
    conf_ms = confirmation_minutes * 60_000  # minutes → milliseconds

    targets = np.zeros(n, dtype=np.int8)
    gain_at_conf = np.full(n, np.nan, dtype=np.float64)
    buckets = np.full(n, "", dtype=object)
    sustain_actual = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        entry_price = closes[i]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        entry_ts = timestamps[i]
        conf_target_ts = entry_ts + conf_ms

        # Find the confirmation bar: first bar with timestamp >= conf_target_ts
        conf_idx = int(np.searchsorted(timestamps, conf_target_ts, side="left"))
        if conf_idx >= n:
            # No confirmation bar within the data
            continue

        conf_price = closes[conf_idx]
        if np.isnan(conf_price) or conf_price <= 0:
            continue

        # Record gain at confirmation regardless of sustain
        gain_pct = (conf_price - entry_price) / entry_price * 100.0
        gain_at_conf[i] = gain_pct
        buckets[i] = classify_magnitude(gain_pct)

        if conf_price <= entry_price:
            sustain_actual[i] = 0.0
            continue

        # Count consecutive bars above entry_price starting at conf_idx
        consec = 0
        max_look = min(n, conf_idx + sustain_minutes + 60)  # don't look too far
        for j in range(conf_idx, max_look):
            if closes[j] > entry_price:
                consec += 1
                if consec >= sustain_minutes:
                    targets[i] = 1
                    break
            else:
                break  # streak broken — stop counting

        sustain_actual[i] = float(consec)

    df["target_sustained"] = targets
    df["gain_pct_at_confirmation"] = gain_at_conf
    df["magnitude_bucket"] = buckets
    df["sustain_minutes_actual"] = sustain_actual

    return df
