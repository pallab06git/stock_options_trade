# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Generic options bar downloader using massive.RESTClient.list_aggs().

Receives a filtered contract list from ContractSelector and downloads
minute bars for each contract in parallel.  Contains zero ticker-construction
or strike-selection logic — those responsibilities belong entirely to
ContractSelector and OptionsTickerBuilder.

Execution flow
--------------
    selector = ContractSelector(config, mode="test")   # or "prod"
    downloader = MassiveOptionsDownloader(config, api_key, selector)
    downloader.run("2025-03-01", "2025-03-31")

    Internally, run() does for each trading date:
      1. Read underlying opening price from local Parquet (no API call)
      2. selector.get_contracts(date, opening_price)  → filtered contract list
      3. download_tickers(contracts, date)             → parallel list_aggs()
      4. Write one Parquet per contract per date

Output path (identical to TargetedOptionsDownloader for schema compatibility):
    data/raw/options/minute/{safe_ticker}/{date}.parquet
    where safe_ticker = ticker.replace(":", "_")

Config keys (pipeline_v2.massive_options in pipeline_v2.yaml)
-------------------------------------------------------------
    limit_per_request : bars per list_aggs call         (default 500)
    max_workers       : parallel download threads        (default 4)

API key resolution (sources.yaml / env vars)
--------------------------------------------
    1. env  MASSIVE_API_KEY
    2. config massive.api_key
    3. env  POLYGON_API_KEY        (Massive accepts Polygon keys)
    4. config polygon.api_key
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.data_sources.contract_selector import ContractSelector
from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()

try:
    from massive import RESTClient as MassiveRESTClient  # type: ignore
    _MASSIVE_AVAILABLE = True
except ImportError:
    MassiveRESTClient = None  # type: ignore
    _MASSIVE_AVAILABLE = False


class MassiveOptionsDownloader:
    """Parallel options minute-bar downloader backed by massive.RESTClient.

    This class is deliberately free of any ticker-construction or
    strike-selection logic.  It receives contract dicts from ContractSelector
    and is responsible only for:
      - Reading opening prices from local Parquet files
      - Calling list_aggs() for each contract (in parallel)
      - Persisting the results as date-partitioned Parquet files

    Args:
        config:   Full merged config dict.
        api_key:  Massive.com API key.
        selector: Initialised ContractSelector instance (TEST or PROD mode).

    Raises:
        ImportError: If the ``massive`` package is not installed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: str,
        selector: ContractSelector,
    ) -> None:
        if not _MASSIVE_AVAILABLE:
            raise ImportError(
                "The `massive` package is required for MassiveOptionsDownloader.\n"
                "Install with:  pip install massive"
            )
        if not api_key:
            raise ValueError("api_key must not be empty.")

        self._client   = MassiveRESTClient(api_key)
        self._selector = selector
        self.config    = config

        opts_cfg = config.get("pipeline_v2", {}).get("massive_options", {})
        self._limit       = opts_cfg.get("limit_per_request", 500)
        self._max_workers = opts_cfg.get("max_workers", 4)

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.base_path   = Path(parquet_cfg.get("base_path", "data/raw"))
        self.compression = parquet_cfg.get("compression", "snappy")

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], selector: ContractSelector
    ) -> "MassiveOptionsDownloader":
        """Construct from merged config, resolving API key automatically.

        Key resolution order:
          1. env  MASSIVE_API_KEY
          2. config massive.api_key
          3. env  POLYGON_API_KEY
          4. config polygon.api_key

        Args:
            config:   Full merged config dict.
            selector: Initialised ContractSelector instance.

        Raises:
            ValueError: If no API key can be resolved.
        """
        api_key = (
            os.getenv("MASSIVE_API_KEY")
            or config.get("massive", {}).get("api_key", "")
            or os.getenv("POLYGON_API_KEY")
            or config.get("polygon", {}).get("api_key", "")
        )
        if not api_key:
            raise ValueError(
                "No API key found. Set MASSIVE_API_KEY or POLYGON_API_KEY."
            )
        return cls(config, api_key, selector)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_price(self, date: str, underlying: str = "SPY") -> float:
        """Read the opening price from local Parquet — no API call.

        Uses the first bar (by timestamp) of the underlying's minute data
        for the given date.

        Args:
            date:       Trading date (YYYY-MM-DD).
            underlying: Ticker symbol whose Parquet to read (default "SPY").

        Returns:
            Opening price as float.

        Raises:
            FileNotFoundError: Parquet file not found for date.
            ValueError:        File exists but is empty.
        """
        path = self.base_path / underlying.lower() / f"{date}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{underlying} minute data not found for {date}: {path}. "
                f"Run download-minute --ticker {underlying} first."
            )
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"{underlying} minute data is empty for {date}")
        return float(df.sort_values("timestamp").iloc[0]["open"])

    def download_tickers(
        self,
        contracts: List[Dict[str, Any]],
        date: str,
        resume: bool = True,
    ) -> int:
        """Download bars for multiple contracts in parallel.

        Spawns up to max_workers threads, each calling list_aggs() for one
        contract.  Results are written to Parquet as they complete.

        Args:
            contracts: Contract dicts from ContractSelector.get_contracts().
                       Each must have at least a "ticker" key.
            date:      Trading date (YYYY-MM-DD).
            resume:    If True, skip contracts whose Parquet already exists.

        Returns:
            Total bars written across all contracts.
        """
        if not contracts:
            return 0

        total_bars = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._download_single, contract, date, resume): contract
                for contract in contracts
            }
            for future in as_completed(futures):
                contract = futures[future]
                try:
                    bars = future.result()
                    total_bars += bars
                except Exception as exc:
                    logger.warning(
                        f"[{date}] Download failed for "
                        f"{contract.get('ticker', '?')}: {exc}"
                    )

        return total_bars

    def run(
        self,
        start_date: str,
        end_date: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Download options bars for all dates in [start_date, end_date].

        Prompts for TEST mode params before the download loop starts so the
        user is asked once, up front, before any progress bars appear.

        Requires underlying minute Parquet to already exist for each date
        (reads opening price from it).  Dates without local data are skipped.

        Args:
            start_date: First date (YYYY-MM-DD), inclusive.
            end_date:   Last date  (YYYY-MM-DD), inclusive.
            resume:     If True, skip contracts whose Parquet already exists.

        Returns:
            Stats dict:
              dates_processed, dates_skipped, contracts_found, total_bars.
        """
        # Prompt once before the loop so the tqdm bar starts cleanly
        if self._selector.needs_prompt:
            self._selector.prompt_once()

        underlying = self._selector.underlying
        dates = self._date_range(start_date, end_date)

        stats: Dict[str, Any] = {
            "dates_processed": 0,
            "dates_skipped":   0,
            "contracts_found": 0,
            "total_bars":      0,
        }

        for date in tqdm(dates, desc="Massive options download", unit="day"):
            # ---- Opening price ----------------------------------------
            try:
                opening_price = self.get_opening_price(date, underlying)
            except FileNotFoundError:
                logger.warning(
                    f"[{date}] {underlying} data not found — skipping"
                )
                stats["dates_skipped"] += 1
                continue
            except Exception as exc:
                logger.warning(
                    f"[{date}] Cannot read {underlying} opening price: {exc}"
                )
                stats["dates_skipped"] += 1
                continue

            # ---- Contract selection ------------------------------------
            try:
                contracts = self._selector.get_contracts(date, opening_price)
            except Exception as exc:
                logger.warning(
                    f"[{date}] Contract selection failed — skipping: {exc}"
                )
                stats["dates_skipped"] += 1
                continue

            if not contracts:
                logger.warning(f"[{date}] No contracts selected — skipping")
                stats["dates_skipped"] += 1
                continue

            stats["contracts_found"] += len(contracts)

            # ---- Parallel bar download ---------------------------------
            try:
                bars = self.download_tickers(contracts, date, resume)
                stats["total_bars"] += bars
                stats["dates_processed"] += 1
            except Exception as exc:
                logger.warning(f"[{date}] Download failed — skipping: {exc}")
                stats["dates_skipped"] += 1

        logger.info(
            f"MassiveOptionsDownloader finished — "
            f"{stats['dates_processed']} processed, "
            f"{stats['dates_skipped']} skipped, "
            f"{stats['contracts_found']} contracts, "
            f"{stats['total_bars']} total bars"
        )
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_single(
        self,
        contract: Dict[str, Any],
        date: str,
        resume: bool = True,
    ) -> int:
        """Download and persist bars for one contract on one date.

        Args:
            contract: Dict with at least {"ticker": "O:SPY..."}.
            date:     Trading date (YYYY-MM-DD).
            resume:   Skip if the output Parquet already exists.

        Returns:
            Number of bars written (0 if no data or skipped).
        """
        ticker      = contract["ticker"]
        safe_ticker = ticker.replace(":", "_")
        out_dir     = self.base_path / "options" / "minute" / safe_ticker
        out_path    = out_dir / f"{date}.parquet"

        if resume and out_path.exists():
            existing = pd.read_parquet(out_path)
            logger.debug(f"[{date}] {ticker}: already exists ({len(existing)} bars)")
            return len(existing)

        bars = self._fetch_bars(ticker, date)
        if not bars:
            logger.debug(f"[{date}] {ticker}: no data returned")
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(bars)
        df.to_parquet(
            out_path,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        logger.info(
            f"[{date}] {ticker}: {len(bars)} bars written "
            f"(strike={contract.get('strike')}, "
            f"type={contract.get('contract_type')}, "
            f"expiry={contract.get('expiry_date')})"
        )
        return len(bars)

    def _fetch_bars(self, ticker: str, date: str) -> List[Dict[str, Any]]:
        """Call massive list_aggs() for one ticker on one date.

        Wraps the call in the project retry handler for 5xx / 429 errors.

        Args:
            ticker: Options ticker (e.g. "O:SPY250304C00601000").
            date:   Trading date (YYYY-MM-DD).

        Returns:
            List of bar dicts. Empty if the contract has no data.
        """
        @with_retry(source="polygon", config=self.config)
        def _call():
            try:
                bars = []
                for agg in self._client.list_aggs(
                    ticker,
                    1,          # multiplier
                    "minute",   # timespan
                    date,       # from
                    date,       # to
                    adjusted="true",
                    sort="asc",
                    limit=self._limit,
                ):
                    bars.append({
                        "timestamp":    getattr(agg, "timestamp",    None),
                        "open":         getattr(agg, "open",         None),
                        "high":         getattr(agg, "high",         None),
                        "low":          getattr(agg, "low",          None),
                        "close":        getattr(agg, "close",        None),
                        "volume":       getattr(agg, "volume",       None),
                        "vwap":         getattr(agg, "vwap",         None),
                        "transactions": getattr(agg, "transactions", None),
                        "ticker":       ticker,
                        "source":       "options_minute_massive",
                    })
                return bars
            except Exception as exc:
                code = getattr(exc, "status_code", getattr(exc, "status", 0))
                if isinstance(code, int) and code > 0:
                    raise RetryableError(str(exc), status_code=code) from exc
                raise

        try:
            return _call()
        except Exception as exc:
            logger.debug(f"_fetch_bars({ticker}, {date}): {exc}")
            return []

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        """Return list of date strings from start to end inclusive."""
        from datetime import datetime, timedelta
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end     = datetime.strptime(end_date,   "%Y-%m-%d")
        dates: List[str] = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
