# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Training data preparation module.

Reads consolidated per-option-per-minute Parquet files (produced by
the Consolidator) and adds a `target_future_prices` column by looking
up actual historical option prices at T+1..T+N minutes ahead.

This is an OFFLINE module used only for ML training data generation.
In production the ML model predicts target_future_prices; this module
populates ground-truth values from historical data so the model can
learn from them.

Output is written to data/processed/training/.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class TrainingDataPrep:
    """Prepare training datasets from consolidated historical data."""

    def __init__(self, config: Dict[str, Any]):
        sv = config.get("signal_validation", {})
        self.prediction_window_minutes = sv.get("prediction_window_minutes", 120)

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.consolidated_path = Path(
            parquet_cfg.get("consolidated_path", "data/processed/consolidated")
        )
        self.training_path = Path(
            parquet_cfg.get("training_path", "data/processed/training")
        )
        self.compression = parquet_cfg.get("compression", "snappy")

        training_cfg = config.get("training", {})
        self.min_target_coverage_pct = training_cfg.get("min_target_coverage_pct", 50.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, dates: List[str]) -> Dict[str, Any]:
        """Prepare training data for a list of trading dates.

        For each date:
          1. Load the consolidated Parquet
          2. Add target_future_prices from historical lookahead
          3. Filter rows below min target coverage
          4. Write to training output directory

        Args:
            dates: List of date strings (YYYY-MM-DD).

        Returns:
            Stats dict with totals across all dates.
        """
        total_rows_in = 0
        total_rows_out = 0
        total_options = set()
        dates_processed = 0
        dates_skipped = 0

        for date in dates:
            result = self._prepare_single_date(date)
            if result is None:
                dates_skipped += 1
                continue

            dates_processed += 1
            total_rows_in += result["rows_in"]
            total_rows_out += result["rows_out"]
            total_options.update(result["tickers"])

        stats = {
            "status": "success",
            "dates_processed": dates_processed,
            "dates_skipped": dates_skipped,
            "total_rows_in": total_rows_in,
            "total_rows_out": total_rows_out,
            "total_rows_filtered": total_rows_in - total_rows_out,
            "unique_options": len(total_options),
            "prediction_window_minutes": self.prediction_window_minutes,
            "min_target_coverage_pct": self.min_target_coverage_pct,
        }

        logger.info(
            f"Training data prep complete: {dates_processed} dates, "
            f"{total_rows_out}/{total_rows_in} rows retained, "
            f"{len(total_options)} unique options"
        )
        return stats

    # ------------------------------------------------------------------
    # Per-date processing
    # ------------------------------------------------------------------

    def _prepare_single_date(self, date: str) -> Dict[str, Any] | None:
        """Process a single date: load, add targets, filter, write."""
        df = self._load_consolidated(date)
        if df is None:
            return None

        rows_in = len(df)

        # Add target_future_prices
        df = self._compute_target_future_prices(df)

        # Filter rows with insufficient target coverage
        df = self._filter_by_target_coverage(df)

        rows_out = len(df)
        tickers = set(df["ticker"].dropna().unique()) if "ticker" in df.columns else set()

        if rows_out == 0:
            logger.warning(f"No rows survived target coverage filter for {date}")
            return {"rows_in": rows_in, "rows_out": 0, "tickers": set()}

        # Write output
        self._write_output(df, date)

        logger.info(
            f"Training data for {date}: {rows_out}/{rows_in} rows "
            f"({len(tickers)} options)"
        )
        return {"rows_in": rows_in, "rows_out": rows_out, "tickers": tickers}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_consolidated(self, date: str) -> pd.DataFrame | None:
        """Load a consolidated Parquet file for the given date."""
        path = self.consolidated_path / f"{date}.parquet"
        if not path.exists():
            logger.warning(f"Consolidated data not found: {path}")
            return None
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------

    def _compute_target_future_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target_future_prices column from historical lookahead.

        For each row at minute T with ticker X, looks up option_avg_price
        at T+60000, T+120000, ..., T+N*60000. Missing minutes get NaN.
        Rows without a ticker get None.
        """
        window = self.prediction_window_minutes

        # Build lookup: (ticker, timestamp) → option_avg_price
        has_ticker = df["ticker"].notna()
        if has_ticker.any():
            price_lookup = {}
            for _, row in df[has_ticker].iterrows():
                key = (row["ticker"], int(row["timestamp"]))
                price_lookup[key] = row["option_avg_price"]
        else:
            price_lookup = {}

        targets = []
        for _, row in df.iterrows():
            ticker = row.get("ticker")
            if ticker is None or pd.isna(ticker):
                targets.append(None)
                continue

            ts = int(row["timestamp"])
            future_prices = []
            for offset in range(1, window + 1):
                future_ts = ts + offset * 60000
                price = price_lookup.get((ticker, future_ts), np.nan)
                future_prices.append(price)
            targets.append(future_prices)

        df["target_future_prices"] = targets
        return df

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_by_target_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where target_future_prices has too many NaNs.

        Rows without a ticker (SPY-only) are always removed since they
        have no target. Rows with a ticker are kept if the percentage of
        non-NaN values in target_future_prices >= min_target_coverage_pct.
        """
        window = self.prediction_window_minutes
        threshold = self.min_target_coverage_pct / 100.0

        keep_mask = []
        for _, row in df.iterrows():
            target = row.get("target_future_prices")
            if target is None:
                keep_mask.append(False)
                continue

            target_list = list(target)
            if len(target_list) == 0:
                keep_mask.append(False)
                continue

            non_nan = sum(1 for v in target_list if not np.isnan(v))
            coverage = non_nan / window
            keep_mask.append(coverage >= threshold)

        return df[keep_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(self, df: pd.DataFrame, date: str) -> Path:
        """Write training data to Parquet."""
        self.training_path.mkdir(parents=True, exist_ok=True)
        path = self.training_path / f"{date}.parquet"
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        logger.info(f"Wrote training data to {path} ({len(df)} rows)")
        return path
