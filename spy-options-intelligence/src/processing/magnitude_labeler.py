# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Magnitude label generator — direction-agnostic volatility prediction.

Rather than asking "did the price go UP?" (the existing directional target),
this module asks:

    "Did the option price move at least ``min_magnitude_pct`` in EITHER
    direction within the next ``lookforward_minutes`` minutes?"

This enables straddle strategies: buy both a call and a put; one side profits
if the model is correct about the *magnitude* of the move, regardless of
direction.

Implementation note
-------------------
The feature CSVs already contain pre-computed forward-looking outcome
columns:

    ``max_gain_120m``   : best % gain seen in the next 120 min (always ≥ 0)
    ``min_loss_120m``   : worst % loss seen in the next 120 min (always ≤ 0)

These are computed during feature engineering and stored in the CSVs, so
``MagnitudeLabeler`` simply reads them — no raw-bar scanning required.

Formula
-------
    abs_max_move = max(max_gain_120m.clip(lower=0), abs(min_loss_120m))
    target_magnitude = 1  if  abs_max_move >= min_magnitude_pct  else  0

Output columns added by ``label()``
------------------------------------
    target_magnitude  int8   — 1 = big move (≥ min_magnitude_pct), 0 = no
    abs_max_move_pct  float  — larger of upward gain or downward loss (%)
    move_direction    str    — 'up', 'down', or 'flat'
    magnitude_bucket  str    — one of the five buckets below

Magnitude buckets
-----------------
    '0-5%'    : abs_max_move ∈ [0, 5)
    '5-10%'   : abs_max_move ∈ [5, 10)
    '10-20%'  : abs_max_move ∈ [10, 20)
    '20-30%'  : abs_max_move ∈ [20, 30)
    '30%+'    : abs_max_move ≥ 30
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Ordered list of magnitude bucket labels (used for display and iteration)
MAGNITUDE_BUCKETS: list[str] = ["0-5%", "5-10%", "10-20%", "20-30%", "30%+"]

# Required source columns (pre-computed during feature engineering)
_REQUIRED_COLS = {"max_gain_120m", "min_loss_120m"}


def _assign_bucket(abs_move: float) -> str:
    """Map a single absolute % move value to its bucket label."""
    if abs_move < 5.0:
        return "0-5%"
    if abs_move < 10.0:
        return "5-10%"
    if abs_move < 20.0:
        return "10-20%"
    if abs_move < 30.0:
        return "20-30%"
    return "30%+"


class MagnitudeLabeler:
    """Label bars by absolute price-move magnitude (direction-agnostic).

    Parameters
    ----------
    min_magnitude_pct:
        Minimum % move (in either direction) required for a positive label.
        Default is 20.0 (%).
    """

    def __init__(self, min_magnitude_pct: float = 20.0) -> None:
        if min_magnitude_pct <= 0:
            raise ValueError(
                f"min_magnitude_pct must be positive, got {min_magnitude_pct}"
            )
        self.min_magnitude = min_magnitude_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add magnitude label columns to *df* and return it.

        Uses pre-computed ``max_gain_120m`` and ``min_loss_120m`` columns to
        derive the absolute maximum move without any forward scanning.

        Args:
            df: Feature DataFrame containing ``max_gain_120m`` and
                ``min_loss_120m`` columns.

        Returns:
            The same DataFrame with four new columns appended:
            ``target_magnitude``, ``abs_max_move_pct``, ``move_direction``,
            ``magnitude_bucket``.

        Raises:
            KeyError: If required columns are missing from *df*.
        """
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise KeyError(
                f"MagnitudeLabeler requires columns {_REQUIRED_COLS}, "
                f"but {missing} are missing from df"
            )

        # Absolute maximum move in either direction
        max_up   = df["max_gain_120m"].clip(lower=0.0)        # best upside %
        max_down = df["min_loss_120m"].abs()                   # worst downside %
        abs_max  = pd.concat([max_up, max_down], axis=1).max(axis=1)

        df = df.copy()
        df["abs_max_move_pct"] = abs_max

        # Binary target
        df["target_magnitude"] = (abs_max >= self.min_magnitude).astype(np.int8)

        # Direction of the winning side
        # Primary direction = whichever of up/down reaches the threshold first.
        # If both exceed threshold, up wins (calls > puts tiebreak).
        up_qualifies   = df["max_gain_120m"] >= self.min_magnitude
        down_qualifies = df["min_loss_120m"]  <= -self.min_magnitude
        df["move_direction"] = np.where(
            up_qualifies,  "up",
            np.where(down_qualifies, "down", "flat")
        )

        # Magnitude bucket (vectorised)
        df["magnitude_bucket"] = pd.cut(
            abs_max,
            bins=[-np.inf, 5.0, 10.0, 20.0, 30.0, np.inf],
            labels=MAGNITUDE_BUCKETS,
            right=False,
        ).astype(str)

        logger.debug(
            "MagnitudeLabeler.label: %d rows | positive rate %.2f%% (>= %.0f%%)",
            len(df),
            df["target_magnitude"].mean() * 100,
            self.min_magnitude,
        )
        return df

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return summary statistics for a labeled DataFrame.

        Requires that ``label()`` has already been called so that
        ``target_magnitude``, ``abs_max_move_pct``, ``move_direction``, and
        ``magnitude_bucket`` columns exist.

        Returns:
            dict with keys:
                n_total, n_positive, positive_rate,
                n_up, n_down, avg_abs_magnitude,
                magnitude_breakdown (dict bucket → count)
        """
        required = {"target_magnitude", "abs_max_move_pct", "move_direction", "magnitude_bucket"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"validate() requires {required}, but {missing} are missing. "
                "Call label() first."
            )

        pos = df[df["target_magnitude"] == 1]
        breakdown = {b: int((df["magnitude_bucket"] == b).sum()) for b in MAGNITUDE_BUCKETS}

        return {
            "n_total": len(df),
            "n_positive": int(df["target_magnitude"].sum()),
            "positive_rate": float(df["target_magnitude"].mean()),
            "n_up": int((pos["move_direction"] == "up").sum()),
            "n_down": int((pos["move_direction"] == "down").sum()),
            "avg_abs_magnitude": float(pos["abs_max_move_pct"].mean()) if len(pos) else 0.0,
            "magnitude_breakdown": breakdown,
        }
