# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Polygon.io options contract discovery client.

Discovers SPY options contracts within a configurable strike range
(±1% by default) of the opening price. One-shot utility used before
streaming — not a BaseSource subclass.
"""

import json
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List

from polygon.websocket.models import Market

from src.utils.connection_manager import ConnectionManager
from src.utils.logger import get_logger
from src.utils.retry_handler import RetryableError, with_retry

logger = get_logger()


class PolygonOptionsClient:
    """
    Discover and persist SPY options contracts for a given trading date.

    Reads configuration from config["polygon"]["options"]:
        - underlying_ticker: equity symbol (default "SPY")
        - strike_range_pct: fraction for ± range (default 0.01)
        - max_contracts: cap on returned contracts (default 100)
        - expiration_lookahead_days: days ahead for expiry (default 1)
    """

    def __init__(self, config: Dict[str, Any], connection_manager: ConnectionManager):
        self.config = config
        self.connection_manager = connection_manager

        opts = config.get("polygon", {}).get("options", {})
        self.underlying_ticker = opts.get("underlying_ticker", "SPY")
        self.strike_range_pct = opts.get("strike_range_pct", 0.01)
        self.max_contracts = opts.get("max_contracts", 100)
        self.expiration_lookahead_days = opts.get("expiration_lookahead_days", 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_opening_price(self, date: str) -> float:
        """
        Fetch the opening price of the underlying ticker for *date*.

        Args:
            date: Trading date (YYYY-MM-DD).

        Returns:
            Opening price as a float.

        Raises:
            ValueError: If no data is returned for the date.
        """
        self.connection_manager.acquire_rate_limit(source="options")

        @with_retry(source="polygon", config=self.config)
        def _api_call():
            client = self.connection_manager.get_rest_client()
            try:
                result = client.get_daily_open_close(
                    ticker=self.underlying_ticker,
                    date=date,
                )
                return result
            except Exception as e:
                status_code = getattr(e, "status_code", getattr(e, "status", 0))
                if isinstance(status_code, int) and status_code > 0:
                    raise RetryableError(str(e), status_code=status_code) from e
                raise

        result = _api_call()
        if result is None:
            raise ValueError(
                f"No daily open/close data for {self.underlying_ticker} on {date}"
            )

        open_price = getattr(result, "open", None)
        if open_price is None:
            raise ValueError(
                f"No opening price for {self.underlying_ticker} on {date}"
            )

        logger.info(
            f"Opening price for {self.underlying_ticker} on {date}: {open_price}"
        )
        return float(open_price)

    def discover_contracts(
        self, date: str, opening_price: float
    ) -> List[Dict[str, Any]]:
        """
        Discover options contracts within the configured strike range.

        Args:
            date: Trading date (YYYY-MM-DD) used to compute expiration.
            opening_price: Underlying opening price for strike range calc.

        Returns:
            List of standardized contract dicts, capped at max_contracts.
        """
        lower = round(opening_price * (1 - self.strike_range_pct), 2)
        upper = round(opening_price * (1 + self.strike_range_pct), 2)

        expiration_date = (
            datetime.strptime(date, "%Y-%m-%d")
            + timedelta(days=self.expiration_lookahead_days)
        ).strftime("%Y-%m-%d")

        logger.info(
            f"Discovering {self.underlying_ticker} options: "
            f"strike [{lower}, {upper}], expiry {expiration_date}"
        )

        self.connection_manager.acquire_rate_limit(source="options")

        @with_retry(source="polygon", config=self.config)
        def _api_call():
            client = self.connection_manager.get_rest_client()
            try:
                contracts_iter = client.list_options_contracts(
                    underlying_ticker=self.underlying_ticker,
                    expiration_date=expiration_date,
                    strike_price_gte=lower,
                    strike_price_lte=upper,
                    limit=self.max_contracts,
                    sort="ticker",
                    order="asc",
                )
                return list(contracts_iter)
            except Exception as e:
                status_code = getattr(e, "status_code", getattr(e, "status", 0))
                if isinstance(status_code, int) and status_code > 0:
                    raise RetryableError(str(e), status_code=status_code) from e
                raise

        raw_contracts = _api_call()
        contracts = [self._transform_contract(c) for c in raw_contracts]

        # Enforce max_contracts cap
        contracts = contracts[: self.max_contracts]

        logger.info(f"Discovered {len(contracts)} contracts for {date}")
        return contracts

    def save_contracts(self, contracts: List[Dict[str, Any]], date: str) -> Path:
        """
        Persist discovered contracts as JSON.

        Args:
            contracts: List of contract dicts.
            date: Trading date (YYYY-MM-DD) used in filename.

        Returns:
            Path to the written JSON file.
        """
        output_dir = Path("data/raw/options/contracts")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{date}_contracts.json"
        with open(output_path, "w") as f:
            json.dump(contracts, f, indent=2)

        logger.info(f"Saved {len(contracts)} contracts to {output_path}")
        return output_path

    def load_contracts(self, date: str) -> List[Dict[str, Any]]:
        """
        Load previously saved contracts for a date.

        Args:
            date: Trading date (YYYY-MM-DD).

        Returns:
            List of contract dicts.

        Raises:
            FileNotFoundError: If no contract file exists for the date.
        """
        path = Path("data/raw/options/contracts") / f"{date}_contracts.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No contract file for {date}. Run discovery first."
            )

        with open(path, "r") as f:
            contracts = json.load(f)

        logger.info(f"Loaded {len(contracts)} contracts from {path}")
        return contracts

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    # Sentinel to signal the generator thread has exited
    _SENTINEL = object()

    def stream_realtime(
        self, date: str, stop_event: threading.Event = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream real-time options aggregates via Polygon WebSocket.

        Loads previously discovered contracts for *date*, subscribes to
        per-minute aggregate channels, and yields transformed records.

        Args:
            date: Trading date (YYYY-MM-DD) — used to load contracts.
            stop_event: Threading event to signal shutdown.

        Yields:
            Standardized options aggregate dicts.
        """
        stop_event = stop_event or threading.Event()
        contracts = self.load_contracts(date)
        tickers = [c["ticker"] for c in contracts][: self.max_contracts]

        if not tickers:
            logger.warning(f"No contracts to stream for {date}")
            return

        logger.info(f"Streaming {len(tickers)} options contracts for {date}")
        msg_queue: queue.Queue = queue.Queue(maxsize=10000)
        reconnect_delay = 5

        def _ws_thread():
            while not stop_event.is_set():
                try:
                    ws_client = self.connection_manager.get_ws_client(
                        market=Market.Options
                    )
                    ws_client.subscribe(*[f"AM.{t}" for t in tickers])

                    def _handle_msg(msgs):
                        for msg in msgs:
                            if stop_event.is_set():
                                return
                            record = self._transform_options_agg(msg)
                            try:
                                msg_queue.put_nowait(record)
                            except queue.Full:
                                pass  # Backpressure — drop oldest

                    ws_client.run(handle_msg=_handle_msg)
                except Exception as e:
                    if stop_event.is_set():
                        break
                    logger.warning(
                        f"Options WebSocket disconnected: {e}. "
                        f"Reconnecting in {reconnect_delay}s"
                    )
                    time.sleep(reconnect_delay)

            msg_queue.put(self._SENTINEL)

        thread = threading.Thread(target=_ws_thread, daemon=True)
        thread.start()
        logger.info("Options WebSocket streaming started")

        try:
            while not stop_event.is_set():
                try:
                    item = msg_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is self._SENTINEL:
                    break
                yield item
        finally:
            stop_event.set()
            thread.join(timeout=5.0)
            logger.info("Options WebSocket streaming stopped")

    @staticmethod
    def _transform_options_agg(agg) -> Dict[str, Any]:
        """
        Transform a Polygon options Agg message to a standardized dict.

        Args:
            agg: Polygon WebSocket Agg object.

        Returns:
            Standardized options aggregate record.
        """
        return {
            "timestamp": getattr(agg, "timestamp", None),
            "open": getattr(agg, "open", None),
            "high": getattr(agg, "high", None),
            "low": getattr(agg, "low", None),
            "close": getattr(agg, "close", None),
            "volume": getattr(agg, "volume", None),
            "vwap": getattr(agg, "vwap", None),
            "transactions": getattr(agg, "transactions", None),
            "ticker": getattr(agg, "symbol", None),
            "source": "options",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_contract(contract) -> Dict[str, Any]:
        """
        Map a Polygon OptionsContract object to a standardized dict.

        Args:
            contract: Polygon OptionsContract object.

        Returns:
            Dict with normalized field names.
        """
        return {
            "ticker": getattr(contract, "ticker", None),
            "underlying_ticker": getattr(contract, "underlying_ticker", None),
            "strike_price": getattr(contract, "strike_price", None),
            "expiration_date": getattr(contract, "expiration_date", None),
            "contract_type": getattr(contract, "contract_type", None),
            "exercise_style": getattr(contract, "exercise_style", None),
            "primary_exchange": getattr(contract, "primary_exchange", None),
            "shares_per_contract": getattr(contract, "shares_per_contract", None),
        }
