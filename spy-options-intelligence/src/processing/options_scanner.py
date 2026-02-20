# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Options 20%-move scanner.

Scans processed options feature files for intraday moves ≥ 20% relative
to any price seen in the prior 120-minute reference window.

For each detected event the scanner records:
  - How long the option stayed above the +20% level
  - How long it stayed above the +10% level (sustained run)

Events are written to CSV in data/reports/options_movement/.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class OptionsScanner:
    """Scan processed options features for 20%+ intraday price moves.

    Configuration is read from config["pipeline_v2"]["scanner"]:
      - reference_window_minutes: rolling lookback for reference price (default 120)
      - trigger_threshold_pct: % gain to trigger an event (default 20.0)
      - sustained_threshold_pct: % gain to track duration (default 10.0)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged config dict.
        """
        v2 = config.get("pipeline_v2", {})
        scanner_cfg = v2.get("scanner", {})
        self.ref_window = scanner_cfg.get("reference_window_minutes", 120)
        self.trigger_pct = scanner_cfg.get("trigger_threshold_pct", 20.0)
        self.sustained_pct = scanner_cfg.get("sustained_threshold_pct", 10.0)

        reporting_cfg = v2.get("reporting", {})
        self.reports_dir = Path(
            reporting_cfg.get("reports_dir", "data/reports")
        )
        self.features_path = Path(
            reporting_cfg.get("features_path", "data/processed/features")
        )
        self._last_scan_stats: Dict[str, Any] = {
            "contract_days": 0,
            "total_bars": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Scan all options feature files in [start_date, end_date] for big moves.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of event dicts (see CSV column schema below).
        """
        dates = self._date_range(start_date, end_date)
        all_events: List[Dict[str, Any]] = []
        contract_days = 0
        total_bars = 0

        options_dir = self.features_path / "options"
        if not options_dir.exists():
            logger.warning(f"Options features directory not found: {options_dir}")
            self._last_scan_stats = {"contract_days": 0, "total_bars": 0}
            return []

        for date in dates:
            for ticker_dir in sorted(options_dir.iterdir()):
                if not ticker_dir.is_dir():
                    continue
                feat_path = ticker_dir / f"{date}.parquet"
                if not feat_path.exists():
                    continue

                try:
                    df_loaded = pd.read_parquet(feat_path)
                    contract_days += 1
                    total_bars += len(df_loaded)
                    events = self._scan_single(
                        feat_path, ticker_dir.name, date, _df=df_loaded
                    )
                    all_events.extend(events)
                except Exception as exc:
                    logger.warning(
                        f"Scan failed for {ticker_dir.name}/{date}: {exc}"
                    )

        self._last_scan_stats = {
            "contract_days": contract_days,
            "total_bars": total_bars,
        }
        logger.info(
            f"Scan [{start_date} → {end_date}]: {len(all_events)} events found"
        )
        return all_events

    def generate_report(
        self,
        events: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Path:
        """Write events to CSV and print a summary to the console.

        Args:
            events: List of event dicts from scan().
            start_date: Start date used in filename.
            end_date: End date used in filename.

        Returns:
            Path to the written CSV file.
        """
        out_dir = self.reports_dir / "options_movement"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{start_date}_{end_date}_movement.csv"

        columns = [
            "date",
            "ticker",
            "trigger_time_et",
            "reference_time_et",
            "gain_pct",
            "above_20pct_duration_min",
            "above_10pct_duration_min",
            "reference_price",
            "trigger_price",
        ]

        if events:
            df = pd.DataFrame(events, columns=columns)
        else:
            df = pd.DataFrame(columns=columns)

        df.to_csv(out_path, index=False)

        # Console summary
        cdays = self._last_scan_stats["contract_days"]
        t_bars = self._last_scan_stats["total_bars"]
        n_events = len(events)

        print(f"\n--- Options Movement Report ---")
        print(f"Period:                   {start_date} → {end_date}")
        print(f"Contract-days scanned:    {cdays}")
        print(f"Total minute bars:        {t_bars}")
        print(f"Total events:             {n_events}")

        if n_events > 0:
            df_ev = pd.DataFrame(events)

            # Events per contract-day (zeros included for no-event days)
            per_cday = df_ev.groupby(["date", "ticker"]).size().values
            n_zero = max(0, cdays - len(per_cday))
            all_counts = np.concatenate(
                [per_cday, np.zeros(n_zero, dtype=int)]
            ) if cdays > 0 else per_cday
            print(
                f"Events/contract-day:      "
                f"min={int(all_counts.min())} "
                f"median={np.median(all_counts):.1f} "
                f"max={int(all_counts.max())}"
            )

            # Positive-minute stats
            total_pos_mins = int(df_ev["above_20pct_duration_min"].sum())
            pos_rate = total_pos_mins / t_bars * 100.0 if t_bars > 0 else 0.0
            med_dur = df_ev["above_20pct_duration_min"].median()
            mean_dur = df_ev["above_20pct_duration_min"].mean()
            print(f"Total >20% minutes:       {total_pos_mins}")
            print(f"Positive-minute rate:     {pos_rate:.2f}%")
            print(
                f"Duration >20% (med/mean): "
                f"{med_dur:.1f} / {mean_dur:.1f} min"
            )

            # Hour distribution
            df_ev["_hour"] = pd.to_numeric(
                df_ev["trigger_time_et"].str[:2], errors="coerce"
            ).fillna(0).astype(int)
            hour_counts = df_ev.groupby("_hour").size()
            if not hour_counts.empty:
                max_cnt = hour_counts.max()
                print(f"\nEvent distribution by trigger hour (ET):")
                for hour, cnt in sorted(hour_counts.items()):
                    bar_len = round(cnt / max_cnt * 30) if max_cnt > 0 else 0
                    print(f"  {hour:02d}:xx  {cnt:4d}  {'#' * bar_len}")
        else:
            print(f"Events/contract-day:      min=0 median=0.0 max=0")
            print(f"Total >20% minutes:       0")
            print(f"Positive-minute rate:     0.00%")
            print(f"Duration >20% (med/mean): N/A")

        print(f"\nSaved to: {out_path}")
        return out_path

    def load_reports(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load all movement CSVs, optionally filtered by date range.

        Args:
            start_date: Filter events on or after this date (YYYY-MM-DD).
            end_date: Filter events on or before this date (YYYY-MM-DD).

        Returns:
            Combined DataFrame from all matching report files.
        """
        report_dir = self.reports_dir / "options_movement"
        if not report_dir.exists():
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for csv_path in sorted(report_dir.glob("*_movement.csv")):
            try:
                df = pd.read_csv(csv_path)
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"Could not read report {csv_path}: {exc}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        if start_date and "date" in combined.columns:
            combined = combined[combined["date"] >= start_date]
        if end_date and "date" in combined.columns:
            combined = combined[combined["date"] <= end_date]

        return combined.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal scan logic
    # ------------------------------------------------------------------

    def _scan_single(
        self,
        feat_path: Path,
        safe_ticker: str,
        date: str,
        _df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Scan one options feature file for 20%+ moves.

        Args:
            feat_path: Path to the feature Parquet file.
            safe_ticker: Filesystem-safe ticker name.
            date: Trading date string (YYYY-MM-DD).

        Returns:
            List of event dicts for this ticker/date.
        """
        import pytz

        et_tz = pytz.timezone("America/New_York")

        df = _df if _df is not None else pd.read_parquet(feat_path)
        if df.empty or "close" not in df.columns:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)
        close = df["close"].values
        timestamps = df["timestamp"].values
        n = len(close)

        events: List[Dict[str, Any]] = []
        consumed = np.zeros(n, dtype=bool)  # bars already part of a prior event

        # Floor for the reference window start.  After each event ends (price drops
        # below the sustained threshold), this advances to the bar after the drop.
        # This prevents the same intraday low from triggering multiple independent
        # events: a new event must establish a fresh low AFTER the previous one ended.
        min_ref_start_bar = 0

        trigger_mult = 1.0 + self.trigger_pct / 100.0
        sustained_mult = 1.0 + self.sustained_pct / 100.0

        for t in range(n):
            if consumed[t]:
                continue

            # Reference window: [max(min_ref_start_bar, t-ref_window) : t]
            # The floor prevents looking back into a trough that already fired an event.
            ref_start = max(min_ref_start_bar, t - self.ref_window)
            ref_prices = close[ref_start:t]
            if len(ref_prices) == 0:
                continue

            # Check if current price exceeds any ref_price by trigger_pct
            ref_min = np.nanmin(ref_prices)
            if ref_min <= 0 or np.isnan(ref_min):
                continue

            if close[t] < ref_min * trigger_mult:
                continue

            # Event triggered — find the reference bar (lowest in window)
            ref_idx = ref_start + int(np.nanargmin(ref_prices))
            ref_price = close[ref_idx]
            trigger_price = close[t]
            gain_pct = (trigger_price - ref_price) / ref_price * 100.0

            # Measure durations forward from trigger bar
            above_20_dur = 0
            above_10_dur = 0
            event_end_bar = n  # default: move runs to end of data
            for fwd in range(t, n):
                consumed[fwd] = True
                if close[fwd] >= ref_price * trigger_mult:
                    above_20_dur += 1
                if close[fwd] >= ref_price * sustained_mult:
                    above_10_dur += 1
                else:
                    # Once it drops below 10% sustained level, stop tracking.
                    # Advance the reference floor to here so subsequent events
                    # cannot reuse the low that generated this one.
                    event_end_bar = fwd + 1
                    break

            # Advance reference floor regardless of whether the loop hit the break
            # (if it ran to end-of-data, event_end_bar == n and no further bars exist).
            min_ref_start_bar = event_end_bar

            # Convert timestamps to ET strings
            def _ts_to_et(ts_ms: int) -> str:
                try:
                    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=et_tz)
                    return dt.strftime("%H:%M:%S")
                except Exception:
                    return ""

            events.append({
                "date": date,
                "ticker": safe_ticker.replace("_", ":", 1),
                "trigger_time_et": _ts_to_et(int(timestamps[t])),
                "reference_time_et": _ts_to_et(int(timestamps[ref_idx])),
                "gain_pct": round(gain_pct, 2),
                "above_20pct_duration_min": above_20_dur,
                "above_10pct_duration_min": above_10_dur,
                "reference_price": round(float(ref_price), 4),
                "trigger_price": round(float(trigger_price), 4),
            })

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        """Return date strings from start to end inclusive."""
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates: List[str] = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
