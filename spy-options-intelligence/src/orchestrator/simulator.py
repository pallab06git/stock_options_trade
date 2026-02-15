# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Feed simulator — replay historical Parquet data as a real-time stream.

Reads date-partitioned Parquet files, sorts records by timestamp, and
yields them with configurable inter-record delays to mimic live data.
Implements the ``stream_realtime()`` interface so it can be injected
into StreamingRunner as a drop-in replacement for a WebSocket client.

Supported sources: spy, vix, options, news, consolidated.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Map source name → base directory
_SOURCE_DIRS = {
    "spy": "data/raw/spy",
    "vix": "data/raw/i:vix",
    "options": "data/raw/options",
    "news": "data/raw/news",
    "consolidated": "data/processed/consolidated",
}


class FeedSimulator:
    """Replay historical data as a simulated real-time feed.

    Usage (standalone)::

        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        for record in sim.stream_realtime(stop_event=stop):
            process(record)

    Usage (injected into StreamingRunner)::

        sim = FeedSimulator(config, source="spy", date="2026-02-10")
        runner = StreamingRunner(config, ticker="SPY", client=sim)
        runner.run()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        source: str,
        date: str,
        speed: float = 1.0,
    ):
        """
        Args:
            config: Full merged configuration dict.
            source: Data source name (spy, vix, options, news, consolidated).
            date: Date of the Parquet file to replay (YYYY-MM-DD).
            speed: Playback speed multiplier. 1.0 = real-time, 10.0 = 10x
                   faster, 0 = no delay (as fast as possible).
        """
        self.config = config
        self.source = source.lower()
        self.date = date
        self.speed = max(speed, 0.0)

        # Read simulator config overrides
        sim_cfg = config.get("simulator", {})
        if speed == 1.0 and "speed_multiplier" in sim_cfg:
            self.speed = max(sim_cfg["speed_multiplier"], 0.0)

        # Resolve Parquet file path
        base_dir = sim_cfg.get("data_dir") or _SOURCE_DIRS.get(self.source, f"data/raw/{self.source}")
        self._parquet_path = Path(base_dir) / f"{self.date}.parquet"

        # Stats
        self._stats: Dict[str, Any] = {
            "records_loaded": 0,
            "records_emitted": 0,
            "total_delay_seconds": 0.0,
            "source": self.source,
            "date": self.date,
            "speed": self.speed,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_records(self) -> List[Dict[str, Any]]:
        """Load and sort records from the Parquet file.

        Returns:
            List of record dicts sorted by timestamp ascending.

        Raises:
            FileNotFoundError: If the Parquet file does not exist.
        """
        if not self._parquet_path.exists():
            raise FileNotFoundError(
                f"No Parquet file found at {self._parquet_path}"
            )

        df = pd.read_parquet(self._parquet_path)

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        records = df.to_dict(orient="records")
        self._stats["records_loaded"] = len(records)
        logger.info(
            f"Simulator loaded {len(records)} records from {self._parquet_path}"
        )
        return records

    # ------------------------------------------------------------------
    # Streaming interface (BaseSource-compatible)
    # ------------------------------------------------------------------

    def stream_realtime(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Yield records with inter-record delays simulating real-time.

        Compatible with ``StreamingRunner`` which calls
        ``client.stream_realtime(stop_event=...)``.

        Args:
            stop_event: Optional ``threading.Event`` to signal early stop.

        Yields:
            Record dicts in timestamp order.
        """
        stop_event: Optional[threading.Event] = kwargs.get("stop_event")
        records = self.load_records()

        if not records:
            logger.warning(f"No records to simulate for {self.source}/{self.date}")
            return

        prev_ts: Optional[float] = None

        for record in records:
            if stop_event and stop_event.is_set():
                logger.info("Simulator stop_event received — ending replay")
                break

            ts = record.get("timestamp")
            if ts is not None and prev_ts is not None and self.speed > 0:
                # Compute delay in seconds (timestamps are in Unix ms)
                gap_ms = ts - prev_ts
                if gap_ms > 0:
                    delay = (gap_ms / 1000.0) / self.speed
                    # Cap individual delay to avoid hanging on large gaps
                    delay = min(delay, 5.0)
                    self._stats["total_delay_seconds"] += delay
                    if stop_event:
                        stop_event.wait(timeout=delay)
                        if stop_event.is_set():
                            break
                    else:
                        time.sleep(delay)

            prev_ts = ts
            self._stats["records_emitted"] += 1
            yield record

        logger.info(
            f"Simulator finished: {self._stats['records_emitted']}/{self._stats['records_loaded']} "
            f"records emitted, {self._stats['total_delay_seconds']:.1f}s total delay"
        )

    # ------------------------------------------------------------------
    # Stubs for BaseSource compatibility (StreamingRunner doesn't call these)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """No-op — simulator reads from local files."""
        pass

    def disconnect(self) -> None:
        """No-op — simulator reads from local files."""
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return simulation statistics.

        Returns:
            Dict with records_loaded, records_emitted, total_delay_seconds,
            source, date, and speed.
        """
        return dict(self._stats)
