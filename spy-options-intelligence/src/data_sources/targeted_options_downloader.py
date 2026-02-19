# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Targeted options contract downloader for 2 calls + 2 puts per day.

For each trading date:
  1. Read SPY opening price from local Parquet (no API call).
  2. Query Polygon for options in ±discovery_range_pct strike range.
  3. Select the 2 calls with lowest strike > opening and
     the 2 puts with highest strike < opening.
  4. Download minute bars for each selected contract.

Output paths:
  - contracts JSON: data/raw/options/contracts/{date}_contracts.json
  - minute bars:    data/raw/options/minute/{safe_ticker}/{date}.parquet
                    where safe_ticker = ticker.replace(":", "_")
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()


class TargetedOptionsDownloader:
    """Download 2 calls + 2 puts per trading day and their minute bars.

    Configuration is read from config["pipeline_v2"]["options"]:
      - n_calls: number of call contracts to select (default 2)
      - n_puts: number of put contracts to select (default 2)
      - discovery_range_pct: ± fraction for Polygon query (default 0.05)
      - expiration_search_days: days ahead to search for expiration (default 5)
    """

    def __init__(self, config: Dict[str, Any], connection_manager):
        """
        Args:
            config: Full merged config dict.
            connection_manager: Shared ConnectionManager.
        """
        self.config = config
        self.connection_manager = connection_manager

        v2 = config.get("pipeline_v2", {})
        opts_cfg = v2.get("options", {})
        self.n_calls = opts_cfg.get("n_calls", 2)
        self.n_puts = opts_cfg.get("n_puts", 2)
        self.discovery_range_pct = opts_cfg.get("discovery_range_pct", 0.05)
        self.expiration_search_days = opts_cfg.get("expiration_search_days", 5)

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.base_path = Path(parquet_cfg.get("base_path", "data/raw"))
        self.compression = parquet_cfg.get("compression", "snappy")

        # Multiplier for get_aggs() call
        self._multiplier = (
            config.get("polygon", {})
            .get("equities", {})
            .get("SPY", {})
            .get("multiplier", 1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_price(self, date: str) -> float:
        """Read SPY opening price from local Parquet (no API call).

        Returns the open of the first minute bar on the given date.

        Args:
            date: Trading date (YYYY-MM-DD).

        Returns:
            Opening price as float.

        Raises:
            FileNotFoundError: If SPY minute data is not available for date.
            ValueError: If file is empty.
        """
        spy_path = self.base_path / "spy" / f"{date}.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(
                f"SPY minute data not found for {date}: {spy_path}. "
                "Run download-minute --ticker SPY first."
            )
        df = pd.read_parquet(spy_path)
        if df.empty:
            raise ValueError(f"SPY minute data is empty for {date}")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return float(df.iloc[0]["open"])

    def discover_targeted(self, date: str, opening_price: float) -> List[Dict]:
        """Discover and select 2 calls + 2 puts for the given date.

        Queries Polygon for contracts within ±discovery_range_pct, then:
          - Calls: lowest n_calls strikes strictly above opening_price
          - Puts:  highest n_puts strikes strictly below opening_price

        Tries successive expiration dates (up to expiration_search_days)
        until contracts are found.

        Args:
            date: Trading date (YYYY-MM-DD).
            opening_price: SPY opening price.

        Returns:
            List of up to n_calls+n_puts contract dicts.
        """
        lower = round(opening_price * (1 - self.discovery_range_pct), 2)
        upper = round(opening_price * (1 + self.discovery_range_pct), 2)
        date_dt = datetime.strptime(date, "%Y-%m-%d")

        contracts: List[Dict] = []
        for offset in range(1, self.expiration_search_days + 1):
            expiry = (date_dt + timedelta(days=offset)).strftime("%Y-%m-%d")

            self.connection_manager.acquire_rate_limit(source="options")

            @with_retry(source="polygon", config=self.config)
            def _api_call(_expiry=expiry):
                rest = self.connection_manager.get_rest_client()
                try:
                    it = rest.list_options_contracts(
                        underlying_ticker="SPY",
                        expiration_date=_expiry,
                        strike_price_gte=lower,
                        strike_price_lte=upper,
                        limit=100,
                        sort="ticker",
                        order="asc",
                    )
                    return list(it)
                except Exception as exc:
                    code = getattr(exc, "status_code", getattr(exc, "status", 0))
                    if isinstance(code, int) and code > 0:
                        raise RetryableError(str(exc), status_code=code) from exc
                    raise

            raw = _api_call()
            if raw:
                contracts = [self._transform_contract(c) for c in raw]
                logger.info(
                    f"[{date}] Found {len(contracts)} contracts for expiry {expiry}"
                )
                break

        if not contracts:
            logger.warning(
                f"[{date}] No contracts found within ±{self.discovery_range_pct*100:.0f}% "
                f"of opening={opening_price} after {self.expiration_search_days} days"
            )
            return []

        calls = sorted(
            [c for c in contracts if c["contract_type"] == "call"
             and c["strike_price"] > opening_price],
            key=lambda c: c["strike_price"],
        )
        puts = sorted(
            [c for c in contracts if c["contract_type"] == "put"
             and c["strike_price"] < opening_price],
            key=lambda c: -c["strike_price"],
        )
        selected = calls[: self.n_calls] + puts[: self.n_puts]

        if len(selected) < self.n_calls + self.n_puts:
            logger.warning(
                f"[{date}] Only {len(selected)} contracts selected "
                f"(wanted {self.n_calls} calls + {self.n_puts} puts)"
            )
        return selected

    def download_minute(self, contract_ticker: str, date: str) -> int:
        """Download minute bars for a single options contract.

        Args:
            contract_ticker: Options ticker (e.g. "O:SPY250307C00625000").
            date: Trading date (YYYY-MM-DD).

        Returns:
            Number of bars written (0 if no data).
        """
        safe_ticker = contract_ticker.replace(":", "_")
        out_dir = self.base_path / "options" / "minute" / safe_ticker
        out_path = out_dir / f"{date}.parquet"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.connection_manager.acquire_rate_limit(source="options")

        @with_retry(source="polygon", config=self.config)
        def _api_call():
            rest = self.connection_manager.get_rest_client()
            try:
                aggs = rest.get_aggs(
                    ticker=contract_ticker,
                    multiplier=self._multiplier,
                    timespan="minute",
                    from_=date,
                    to=date,
                    limit=50000,
                    sort="asc",
                )
                return aggs if aggs else []
            except Exception as exc:
                code = getattr(exc, "status_code", getattr(exc, "status", 0))
                if isinstance(code, int) and code > 0:
                    raise RetryableError(str(exc), status_code=code) from exc
                raise

        raw = _api_call()
        if not raw:
            return 0

        records = []
        for agg in raw:
            records.append({
                "timestamp": getattr(agg, "timestamp", None),
                "open": getattr(agg, "open", None),
                "high": getattr(agg, "high", None),
                "low": getattr(agg, "low", None),
                "close": getattr(agg, "close", None),
                "volume": getattr(agg, "volume", None),
                "vwap": getattr(agg, "vwap", None),
                "transactions": getattr(agg, "transactions", None),
                "ticker": contract_ticker,
                "source": "options_minute",
            })

        df = pd.DataFrame(records)
        df.to_parquet(
            out_path,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        logger.info(
            f"Written {len(records)} minute bars for {contract_ticker} on {date}"
        )
        return len(records)

    def run(
        self,
        start_date: str,
        end_date: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Run targeted options download for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            resume: If True, skip contract download when file already exists.

        Returns:
            Stats dict with dates_processed, dates_skipped, contracts_found,
            total_bars.
        """
        dates = self._date_range(start_date, end_date)
        stats: Dict[str, Any] = {
            "dates_processed": 0,
            "dates_skipped": 0,
            "contracts_found": 0,
            "total_bars": 0,
        }

        for date in tqdm(dates, desc="Targeted options download", unit="day"):
            # Skip if SPY data not yet downloaded
            spy_path = self.base_path / "spy" / f"{date}.parquet"
            if not spy_path.exists():
                logger.warning(
                    f"[{date}] SPY data not found — skipping options download"
                )
                stats["dates_skipped"] += 1
                continue

            try:
                opening_price = self.get_opening_price(date)
            except Exception as exc:
                logger.warning(f"[{date}] Cannot read SPY opening price: {exc}")
                stats["dates_skipped"] += 1
                continue

            # Discover targeted contracts
            try:
                contracts = self.discover_targeted(date, opening_price)
            except Exception as exc:
                logger.warning(f"[{date}] Contract discovery failed — skipping: {exc}")
                stats["dates_skipped"] += 1
                continue

            if not contracts:
                stats["dates_processed"] += 1
                continue

            stats["contracts_found"] += len(contracts)

            # Save contracts JSON
            self._save_contracts(contracts, date)

            # Download minute bars for each contract
            for c in contracts:
                ticker = c["ticker"]
                safe = ticker.replace(":", "_")
                out_path = self.base_path / "options" / "minute" / safe / f"{date}.parquet"

                if resume and out_path.exists():
                    logger.debug(f"[{date}] Skipping {ticker} (file exists)")
                    continue

                try:
                    bars = self.download_minute(ticker, date)
                    stats["total_bars"] += bars
                except Exception as exc:
                    logger.warning(
                        f"[{date}] Minute download failed for {ticker} — skipping: {exc}"
                    )

            stats["dates_processed"] += 1

        logger.info(
            f"TargetedOptionsDownloader: {stats['dates_processed']} processed, "
            f"{stats['dates_skipped']} skipped, {stats['contracts_found']} contracts, "
            f"{stats['total_bars']} total bars"
        )
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transform_contract(self, c) -> Dict[str, Any]:
        """Transform a Polygon contract object to a standardized dict."""
        return {
            "ticker": getattr(c, "ticker", ""),
            "strike_price": float(getattr(c, "strike_price", 0) or 0),
            "expiration_date": getattr(c, "expiration_date", ""),
            "contract_type": (getattr(c, "contract_type", "") or "").lower(),
            "underlying_ticker": getattr(c, "underlying_ticker", "SPY"),
        }

    def _save_contracts(self, contracts: List[Dict], date: str) -> Path:
        """Persist selected contracts as JSON."""
        output_dir = self.base_path / "options" / "contracts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{date}_contracts.json"
        with open(output_path, "w") as f:
            json.dump(contracts, f, indent=2)
        logger.debug(f"Saved {len(contracts)} targeted contracts to {output_path}")
        return output_path

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        """Return list of date strings from start to end inclusive."""
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates: List[str] = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
