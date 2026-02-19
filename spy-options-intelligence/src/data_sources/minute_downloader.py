# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Bulk minute-bar downloader for SPY and VIX.

Wraps PolygonEquityClient + direct Parquet writes with:
- Resume support (skips dates where Parquet file already exists)
- tqdm progress bar over trading days
- Unified stats reporting

Handles both equities (SPY) and indices (I:VIX) via the same
get_aggs() call — only the filesystem path prefix differs.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from src.data_sources.polygon_client import PolygonEquityClient
from src.utils.logger import get_logger

logger = get_logger()


def _source_name(ticker: str) -> str:
    """Map ticker to filesystem-safe directory name.

    SPY  → "spy"
    I:VIX → "vix"   (preserve existing data/ convention)
    other → ticker.lower() with ":" replaced by "_"
    """
    upper = ticker.upper()
    if upper == "I:VIX":
        return "vix"
    return ticker.lower().replace(":", "_")


class MinuteDownloader:
    """Download minute-level OHLCV bars for an equity or index ticker.

    Uses PolygonEquityClient for API calls and writes date-partitioned
    Parquet files directly (bypassing ParquetSink for simplicity).

    Output path: data/raw/{source}/{YYYY-MM-DD}.parquet
    where source = _source_name(ticker).
    """

    def __init__(self, config: Dict[str, Any], connection_manager):
        """
        Args:
            config: Full merged config dict.
            connection_manager: Shared ConnectionManager (rate-limiter + REST client).
        """
        self.config = config
        self.connection_manager = connection_manager

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.base_path = Path(parquet_cfg.get("base_path", "data/raw"))
        self.compression = parquet_cfg.get("compression", "snappy")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Download minute bars for *ticker* over the given date range.

        Args:
            ticker: Equity or index ticker (e.g. "SPY", "I:VIX").
            start_date: Start date inclusive (YYYY-MM-DD).
            end_date: End date inclusive (YYYY-MM-DD).
            resume: If True, skip dates where output file already exists.

        Returns:
            Stats dict with keys:
              - dates_downloaded: number of dates written to disk
              - dates_skipped: number of dates skipped (resume)
              - total_bars: total minute bars written
        """
        source = _source_name(ticker)
        dates = self._date_range(start_date, end_date)

        client = PolygonEquityClient(self.config, self.connection_manager, ticker=ticker)
        client.connect()

        stats = {"dates_downloaded": 0, "dates_skipped": 0, "total_bars": 0}

        with tqdm(dates, desc=f"Downloading {ticker} minutes", unit="day") as pbar:
            for date in pbar:
                out_path = self.base_path / source / f"{date}.parquet"

                if resume and out_path.exists():
                    stats["dates_skipped"] += 1
                    pbar.set_postfix(status="skip", date=date)
                    continue

                # fetch_historical(date, date) yields records for that one day
                records = list(client.fetch_historical(date, date))

                if not records:
                    pbar.set_postfix(status="empty", date=date)
                    continue

                # Normalize source field to filesystem-safe name
                for r in records:
                    r["source"] = source

                out_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(records)
                df.to_parquet(
                    out_path,
                    engine="pyarrow",
                    compression=self.compression,
                    index=False,
                )

                stats["dates_downloaded"] += 1
                stats["total_bars"] += len(records)
                pbar.set_postfix(status=f"{len(records)} bars", date=date)

        client.disconnect()

        logger.info(
            f"MinuteDownloader {ticker}: "
            f"{stats['dates_downloaded']} downloaded, "
            f"{stats['dates_skipped']} skipped, "
            f"{stats['total_bars']} total bars"
        )
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        """Return a list of date strings from start_date to end_date inclusive."""
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates: List[str] = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
