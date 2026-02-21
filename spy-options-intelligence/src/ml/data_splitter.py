# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Time-based train / validation / test splitting for ML model development.

Why time-based (not random)
----------------------------
Options price data is a time series.  Using a random split would allow the
model to see future bars during training — a form of look-ahead bias known as
"data leakage".  For example, a random split might put bar 09:45 from 2025-11-05
in the training set while bar 09:40 from the same day sits in the test set.
The model would then implicitly learn patterns that depend on knowing the future.

The correct approach is a strict chronological cut:
  train  →  earliest dates                     (default 70%)
  val    →  dates immediately after train       (default 15%)
  test   →  latest dates (never seen in train)  (default 15%)

Split granularity
-----------------
When the DataFrame has a ``date`` column the split is made at **date
boundaries** so that all bars from a given trading day end up in the same
set.  This is the preferred mode because intraday bars within one day are
highly autocorrelated and splitting them across sets would introduce subtle
leakage.

When no ``date`` column is present the module falls back to **row-level**
splitting on the timestamp-sorted DataFrame (still strictly chronological).

Module-level function
---------------------
  ``time_based_split(df, train_ratio, val_ratio)``
    → (train_df, val_df, test_df)

DataSplitter class
------------------
Config-driven wrapper with additional helpers:
  ``split(df)``                     → (train_df, val_df, test_df)
  ``split_dates(dates)``            → (train_dates, val_dates, test_dates)
  ``get_summary(train, val, test)`` → dict with row counts, date ranges,
                                      positive rates

Configuration keys (all optional):
  data_preparation.train_ratio  float  default 0.70
  data_preparation.val_ratio    float  default 0.15
  # test_ratio is inferred: 1 - train_ratio - val_ratio (default 0.15)
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Module-level function (primary API)
# ---------------------------------------------------------------------------


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train, validation, and test sets.

    The split respects time order — training data always precedes validation
    data, which always precedes test data.  No timestamp from a later period
    appears in an earlier set.

    Split granularity:
      - If ``df`` has a ``date`` column: whole trading days are kept together
        (date-level split).  All rows for a given date land in exactly one set.
      - Otherwise: rows are split in chronological order by row index after
        sorting by ``timestamp`` (row-level split).

    Args:
        df:          Feature DataFrame.  Must contain a ``timestamp`` column.
                     An optional ``date`` column (YYYY-MM-DD strings) enables
                     date-level splitting.
        train_ratio: Fraction of data to use for training.  Default 0.70.
        val_ratio:   Fraction of data to use for validation.  Default 0.15.
                     The test fraction is ``1 - train_ratio - val_ratio``
                     (default 0.15).

    Returns:
        ``(train_df, val_df, test_df)`` — three non-overlapping DataFrames
        in strict chronological order.  Each has a reset integer index.

    Raises:
        ValueError: If ``train_ratio`` or ``val_ratio`` ≤ 0.
        ValueError: If ``train_ratio + val_ratio ≥ 1.0`` (no room for test).
        ValueError: If ``df`` is missing a ``timestamp`` column.
    """
    _validate_ratios(train_ratio, val_ratio)

    if df.empty:
        empty = df.iloc[0:0].copy()
        return empty, empty, empty

    if "timestamp" not in df.columns:
        raise ValueError(
            "time_based_split requires a 'timestamp' column. "
            f"Got: {df.columns.tolist()!r}"
        )

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    if "date" in df_sorted.columns:
        train_df, val_df, test_df = _split_by_date(df_sorted, train_ratio, val_ratio)
    else:
        train_df, val_df, test_df = _split_by_rows(df_sorted, train_ratio, val_ratio)

    _log_split(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# DataSplitter class (config-driven wrapper)
# ---------------------------------------------------------------------------


class DataSplitter:
    """Config-driven time-based splitter for ML training data.

    Reads split ratios from ``config["data_preparation"]`` and exposes a
    unified ``split(df)`` entry point plus helpers for date-list splits and
    summary reporting.

    Usage::

        splitter = DataSplitter(config)
        train, val, test = splitter.split(features_df)
        print(splitter.get_summary(train, val, test))
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged config dict.  Split ratios are read from
                    ``data_preparation.train_ratio`` and
                    ``data_preparation.val_ratio``.
        """
        prep_cfg = config.get("data_preparation", {})
        self.train_ratio: float = prep_cfg.get("train_ratio", 0.70)
        self.val_ratio: float = prep_cfg.get("val_ratio", 0.15)
        _validate_ratios(self.train_ratio, self.val_ratio)

    @property
    def test_ratio(self) -> float:
        """Derived test fraction (1 − train − val)."""
        return round(1.0 - self.train_ratio - self.val_ratio, 10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply ``time_based_split`` with the configured ratios.

        Args:
            df: Feature DataFrame with at minimum a ``timestamp`` column.

        Returns:
            ``(train_df, val_df, test_df)``
        """
        return time_based_split(df, self.train_ratio, self.val_ratio)

    def split_dates(
        self, dates: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Partition a list of date strings chronologically.

        Useful for planning which dates go into each split before loading
        the full feature DataFrames from disk.

        Args:
            dates: List of date strings (any format, sorted lexicographically
                   — works correctly for YYYY-MM-DD).

        Returns:
            ``(train_dates, val_dates, test_dates)`` — three lists in
            chronological order.
        """
        sorted_dates = sorted(dates)
        n = len(sorted_dates)

        if n == 0:
            return [], [], []

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Guarantee at least 1 date per set when there are enough dates
        if n >= 3:
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)
            val_end = min(val_end, n - 1)
            train_end = min(train_end, val_end - 1)

        return (
            sorted_dates[:train_end],
            sorted_dates[train_end:val_end],
            sorted_dates[val_end:],
        )

    def get_summary(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for each split.

        Args:
            train: Training split DataFrame.
            val:   Validation split DataFrame.
            test:  Test split DataFrame.

        Returns:
            Dict with keys ``train``, ``val``, ``test`` — each holding a
            nested dict — plus ``total_rows`` and ``configured_ratios``.
        """
        total = len(train) + len(val) + len(test)

        summary = {
            "train": _describe_split(train),
            "val": _describe_split(val),
            "test": _describe_split(test),
            "total_rows": total,
            "configured_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
        }

        logger.info(
            f"DataSplitter split: train={len(train)} | val={len(val)} | "
            f"test={len(test)} | total={total}"
        )
        return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    """Raise ValueError for invalid ratio combinations."""
    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be > 0; got {train_ratio}")
    if val_ratio <= 0:
        raise ValueError(f"val_ratio must be > 0; got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0 (to leave room for the "
            f"test set); got {train_ratio} + {val_ratio} = "
            f"{train_ratio + val_ratio}"
        )


def _split_by_date(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique trading dates (preferred — keeps whole days together)."""
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    # When date count is very small, ensure each set has at least 1 date
    if n_dates >= 3:
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n_dates - 1)
        train_end = min(train_end, val_end - 1)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    val_df = df[df["date"].isin(val_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)

    return train_df, val_df, test_df


def _split_by_rows(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Row-level fallback when no 'date' column is present."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def _describe_split(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a summary dict for one split."""
    desc: Dict[str, Any] = {"n_rows": len(df)}

    if df.empty:
        return desc

    if "date" in df.columns:
        dates = df["date"].dropna()
        if not dates.empty:
            desc["date_start"] = str(dates.min())
            desc["date_end"] = str(dates.max())
            desc["n_dates"] = int(df["date"].nunique())

    if "timestamp" in df.columns:
        desc["ts_start"] = int(df["timestamp"].min())
        desc["ts_end"] = int(df["timestamp"].max())

    if "target" in df.columns:
        desc["positive_rate"] = float(df["target"].mean())
        desc["n_positive"] = int(df["target"].sum())

    return desc


def _log_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Log a compact one-line split summary."""
    parts = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        date_range = ""
        if "date" in df.columns and not df.empty:
            date_range = f" [{df['date'].min()} → {df['date'].max()}]"
        parts.append(f"{name}={len(df)}{date_range}")

    logger.info(f"time_based_split: {' | '.join(parts)}")
