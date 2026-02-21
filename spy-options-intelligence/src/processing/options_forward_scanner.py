# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Forward scanner: prospective 20%-move detector for options.

For each minute bar (the candidate entry point) this scanner looks FORWARD
up to ``forward_window_minutes`` bars and asks:

    "If I hold this option from now, will the price rise ≥20% within the
    next N minutes — and how long does that gain sustain above 10%?"

For each qualifying entry bar the scanner records:
  - entry_time_et          : the candidate entry bar (when you'd buy)
  - trigger_time_et        : when the price first crossed the +20% threshold
  - minutes_to_trigger     : bars from entry to the first 20%-hit
  - gain_pct               : % gain from entry price to trigger price
  - above_20pct_duration_min : minutes it stayed ≥20% above entry price
                               (counted from trigger bar forward)
  - above_10pct_duration_min : minutes it stayed ≥10% above entry price
                               (sustained run — stops at first bar that drops below 10%)
  - entry_price / trigger_price

Deduplication
-------------
Once a qualifying run is identified (from first_hit_bar to event_end_bar),
those bars are marked as consumed so that a later entry bar does not report
the same underlying run as a new event.  The *earliest* entry point that
discovers a given run is the one that gets recorded.

This is the *forward* variant — it identifies the best entry opportunities
in historical data and is the complement to OptionsBacktestScanner.

Events are written to CSV in data/reports/options_forward/.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class OptionsForwardScanner:
    """Forward-looking scanner: identifies entry points ahead of 20%+ moves.

    At each bar the scanner looks forward up to ``forward_window_minutes`` bars
    and checks whether the option price will reach a 20%+ gain relative to the
    current close.  If so, it records the entry bar, the first bar that crossed
    the threshold, and how long the gain was sustained above 10%.

    Configuration is read from config["pipeline_v2"]["scanner"]:
      - reference_window_minutes: forward look-ahead window in bars (default 120)
      - trigger_threshold_pct: % gain to detect a qualifying move (default 20.0)
      - sustained_threshold_pct: % gain lower bound for duration tracking (default 10.0)
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
        """Scan all options feature files in [start_date, end_date] for forward moves.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of event dicts (see CSV column schema in generate_report).
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
                        f"Forward scan failed for {ticker_dir.name}/{date}: {exc}"
                    )

        self._last_scan_stats = {
            "contract_days": contract_days,
            "total_bars": total_bars,
        }
        logger.info(
            f"Forward scan [{start_date} → {end_date}]: {len(all_events)} events found"
        )
        return all_events

    def generate_report(
        self,
        events: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Path:
        """Write events to CSV and print a forward-scan summary to the console.

        Args:
            events: List of event dicts from scan().
            start_date: Start date used in filename.
            end_date: End date used in filename.

        Returns:
            Path to the written CSV file.
        """
        out_dir = self.reports_dir / "options_forward"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{start_date}_{end_date}_forward.csv"

        columns = [
            "date",
            "ticker",
            "entry_time_et",
            "trigger_time_et",
            "minutes_to_trigger",
            "gain_pct",
            "above_20pct_duration_min",
            "above_10pct_duration_min",
            "entry_price",
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

        print(f"\n--- Options Forward Scan Report ---")
        print(f"Period:                   {start_date} → {end_date}")
        print(f"Forward window:           {self.ref_window} min")
        print(f"Contract-days scanned:    {cdays}")
        print(f"Total minute bars:        {t_bars:,}")
        print(f"Total qualifying entries: {n_events:,}")

        observations: List[str] = []

        if n_events > 0:
            df_ev = pd.DataFrame(events)

            # Events per contract-day (zeros included for no-event days)
            per_cday = df_ev.groupby(["date", "ticker"]).size().values
            n_zero = max(0, cdays - len(per_cday))
            all_counts = np.concatenate(
                [per_cday, np.zeros(n_zero, dtype=int)]
            ) if cdays > 0 else per_cday
            print(
                f"Entries/contract-day:     "
                f"min={int(all_counts.min())} "
                f"median={np.median(all_counts):.1f} "
                f"max={int(all_counts.max())}"
            )

            # Time to trigger (latency from entry to 20% hit)
            med_ttrig  = df_ev["minutes_to_trigger"].median()
            mean_ttrig = df_ev["minutes_to_trigger"].mean()
            print(
                f"Mins to trigger (med/mean): "
                f"{med_ttrig:.1f} / {mean_ttrig:.1f} min"
            )

            # Duration stats
            med_dur20  = df_ev["above_20pct_duration_min"].median()
            mean_dur20 = df_ev["above_20pct_duration_min"].mean()
            med_dur10  = df_ev["above_10pct_duration_min"].median()
            mean_dur10 = df_ev["above_10pct_duration_min"].mean()
            print(
                f"Duration ≥20% (med/mean): "
                f"{med_dur20:.1f} / {mean_dur20:.1f} min"
            )
            print(
                f"Duration ≥10% (med/mean): "
                f"{med_dur10:.1f} / {mean_dur10:.1f} min"
            )

            # Gain distribution
            med_gain = df_ev["gain_pct"].median()
            max_gain = df_ev["gain_pct"].max()
            print(f"Gain at trigger (med/max): {med_gain:.1f}% / {max_gain:.1f}%")

            # Hour distribution of entry times
            df_ev["_entry_hour"] = pd.to_numeric(
                df_ev["entry_time_et"].str[:2], errors="coerce"
            ).fillna(0).astype(int)
            hour_counts = df_ev.groupby("_entry_hour").size()
            if not hour_counts.empty:
                max_cnt = hour_counts.max()
                print(f"\nEntry distribution by hour (ET):")
                for hour, cnt in sorted(hour_counts.items()):
                    bar_len = round(cnt / max_cnt * 30) if max_cnt > 0 else 0
                    print(f"  {hour:02d}:xx  {cnt:4d}  {'#' * bar_len}")

            # Hour distribution of trigger times (when 20% threshold was first hit)
            df_ev["_trigger_hour"] = pd.to_numeric(
                df_ev["trigger_time_et"].str[:2], errors="coerce"
            ).fillna(0).astype(int)
            trig_hour_counts = df_ev.groupby("_trigger_hour").size()
            if not trig_hour_counts.empty:
                max_tcnt = trig_hour_counts.max()
                print(f"\nEvent distribution by trigger hour (ET):")
                for hour, cnt in sorted(trig_hour_counts.items()):
                    bar_len = round(cnt / max_tcnt * 30) if max_tcnt > 0 else 0
                    print(f"  {hour:02d}:xx  {cnt:4d}  {'#' * bar_len}")

            # ------------------------------------------------------------------
            # Key observations (auto-generated)
            # ------------------------------------------------------------------

            # 1. Trigger latency
            if med_ttrig <= 15:
                observations.append(
                    f"Median {med_ttrig:.0f} min to trigger — moves materialize quickly; "
                    f"entry timing is critical since the window is narrow"
                )
            elif med_ttrig <= 45:
                observations.append(
                    f"Median {med_ttrig:.0f} min to trigger (mean {mean_ttrig:.0f} min) — "
                    f"qualifying moves take ~half an hour on average to develop after entry"
                )
            else:
                observations.append(
                    f"Median {med_ttrig:.0f} min to trigger (mean {mean_ttrig:.0f} min) — "
                    f"most qualifying moves develop slowly; patience required after entry"
                )

            # 2. Duration above 20% vs 10% (run sustainability)
            if med_dur20 <= 5 and mean_dur20 > 3 * med_dur20:
                observations.append(
                    f"Runs are brief once triggered: median {med_dur20:.0f} min above "
                    f"+{self.trigger_pct:.0f}% but mean {mean_dur20:.1f} min — a small number "
                    f"of long-tail events pull the average up significantly"
                )
            elif med_dur10 >= 2 * med_dur20:
                observations.append(
                    f"Sustained moves: median {med_dur10:.0f} min above "
                    f"+{self.sustained_pct:.0f}% vs {med_dur20:.0f} min above "
                    f"+{self.trigger_pct:.0f}% — options tend to hold partial gains "
                    f"well after the peak fades"
                )
            else:
                observations.append(
                    f"Duration above +{self.trigger_pct:.0f}%: med {med_dur20:.0f} min / "
                    f"mean {mean_dur20:.1f} min — duration above +{self.sustained_pct:.0f}%: "
                    f"med {med_dur10:.0f} min / mean {mean_dur10:.1f} min"
                )

            # 3. Peak entry hour
            if not hour_counts.empty:
                peak_hour = int(hour_counts.idxmax())
                peak_cnt  = int(hour_counts.max())
                peak_pct  = peak_cnt / n_events * 100
                if peak_hour <= 9:
                    observations.append(
                        f"Open dominates: 09:xx has the most qualifying entries "
                        f"({peak_cnt}, {peak_pct:.0f}%) — buying near the open gives the "
                        f"highest frequency of catching a 20%+ run within the session"
                    )
                elif peak_hour >= 15:
                    observations.append(
                        f"Late-day entries most frequent: {peak_hour:02d}:xx "
                        f"({peak_cnt}, {peak_pct:.0f}%) — gamma-driven end-of-day moves "
                        f"provide the most entry opportunities"
                    )
                else:
                    observations.append(
                        f"{peak_hour:02d}:xx has the most qualifying entries "
                        f"({peak_cnt}, {peak_pct:.0f}%) — mid-session momentum "
                        f"creates the most entry opportunities"
                    )

            # 3b. Peak trigger hour (when does the 20% move actually materialise?)
            if not trig_hour_counts.empty:
                peak_trig_hour = int(trig_hour_counts.idxmax())
                peak_trig_cnt  = int(trig_hour_counts.max())
                peak_trig_pct  = peak_trig_cnt / n_events * 100
                if peak_trig_hour <= 9:
                    observations.append(
                        f"Most moves trigger at the open: 09:xx accounts for "
                        f"{peak_trig_cnt} triggers ({peak_trig_pct:.0f}%) — "
                        f"gap-and-go dynamics resolve rapidly near the 9:30 open"
                    )
                elif peak_trig_hour >= 15:
                    observations.append(
                        f"Triggers cluster into the close: {peak_trig_hour:02d}:xx "
                        f"has the most trigger events ({peak_trig_cnt}, {peak_trig_pct:.0f}%) — "
                        f"end-of-day acceleration drives the majority of confirmed 20%+ moves"
                    )
                else:
                    observations.append(
                        f"Moves confirm in mid-session: {peak_trig_hour:02d}:xx "
                        f"has the most trigger events ({peak_trig_cnt}, {peak_trig_pct:.0f}%) — "
                        f"the bulk of 20%+ moves materialise in mid-morning trading"
                    )

            # 4. Gain magnitude
            if max_gain > 2 * self.trigger_pct:
                observations.append(
                    f"Median gain at trigger {med_gain:.1f}%, max {max_gain:.1f}% — "
                    f"tail events can reach {max_gain/self.trigger_pct:.1f}× the trigger "
                    f"threshold, indicating occasional explosive moves"
                )
            else:
                observations.append(
                    f"Median gain at trigger {med_gain:.1f}%, max {max_gain:.1f}% — "
                    f"gains cluster just above the +{self.trigger_pct:.0f}% threshold "
                    f"with limited upside beyond it"
                )

        else:
            print(f"Entries/contract-day:     min=0 median=0.0 max=0")
            print(f"Mins to trigger (med/mean): N/A")
            print(f"Duration ≥20% (med/mean): N/A")
            print(f"Duration ≥10% (med/mean): N/A")
            observations.append("No qualifying forward entries found in the scanned period")

        # Print key observations
        print(f"\n  Key observations:")
        for obs in observations:
            print(f"  - {obs}")

        print(f"\nSaved to: {out_path}")
        return out_path

    def load_reports(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load all forward-scan CSVs, optionally filtered by date range.

        Args:
            start_date: Filter events on or after this date (YYYY-MM-DD).
            end_date: Filter events on or before this date (YYYY-MM-DD).

        Returns:
            Combined DataFrame from all matching report files.
        """
        report_dir = self.reports_dir / "options_forward"
        if not report_dir.exists():
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for csv_path in sorted(report_dir.glob("*_forward.csv")):
            try:
                df = pd.read_csv(csv_path)
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"Could not read forward report {csv_path}: {exc}")

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
        """Scan one options feature file for prospective 20%+ forward moves.

        At each bar ``t``, looks forward up to ``ref_window`` bars.  If any
        forward bar reaches ≥20% above ``close[t]``, the earliest such bar is
        the trigger.  Duration above 20% and 10% is measured forward from the
        trigger until the price first drops below the 10% sustained level.

        Bars inside an identified run are marked consumed so that subsequent
        entry bars do not re-report the same underlying move.

        Args:
            feat_path: Path to the feature Parquet file.
            safe_ticker: Filesystem-safe ticker name.
            date: Trading date string (YYYY-MM-DD).
            _df: Pre-loaded DataFrame (avoids double read when called from scan()).

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
        # Bars that belong to a run already recorded — skip as entry points.
        consumed = np.zeros(n, dtype=bool)

        trigger_mult   = 1.0 + self.trigger_pct   / 100.0
        sustained_mult = 1.0 + self.sustained_pct / 100.0

        def _ts_to_et(ts_ms: int) -> str:
            try:
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=et_tz)
                return dt.strftime("%H:%M:%S")
            except Exception:
                return ""

        for t in range(n):
            if consumed[t]:
                continue

            entry_price = close[t]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            # Forward window: bars t+1 … t+ref_window (inclusive)
            fwd_end = min(t + self.ref_window + 1, n)
            if t + 1 >= fwd_end:
                continue

            forward_prices = close[t + 1 : fwd_end]

            # Does any forward bar hit the 20% trigger?
            qualifying = forward_prices >= entry_price * trigger_mult
            if not np.any(qualifying):
                continue

            # First forward bar that crosses the 20% threshold
            first_hit_offset = int(np.argmax(qualifying))  # argmax → first True
            first_hit_bar   = t + 1 + first_hit_offset
            trigger_price   = close[first_hit_bar]
            gain_pct        = (trigger_price - entry_price) / entry_price * 100.0
            minutes_to_trigger = first_hit_bar - t

            # Measure how long the run sustains from first_hit_bar forward.
            # Stop counting when price first drops below the 10% sustained level.
            # Mark all run bars as consumed so we don't re-report this run.
            above_20_dur = 0
            above_10_dur = 0
            event_end_bar = n
            for fwd in range(first_hit_bar, n):
                consumed[fwd] = True
                if close[fwd] >= entry_price * trigger_mult:
                    above_20_dur += 1
                if close[fwd] >= entry_price * sustained_mult:
                    above_10_dur += 1
                else:
                    # Price dropped below 10% — run is over
                    event_end_bar = fwd + 1
                    break

            # Also mark the entry bar and all pre-trigger bars as consumed.
            # Without this, bars t+1 … first_hit_bar-1 would each look forward,
            # find the same run at first_hit_bar, and produce duplicate events.
            for pre in range(t, first_hit_bar):
                consumed[pre] = True

            events.append({
                "date":                    date,
                "ticker":                  safe_ticker.replace("_", ":", 1),
                "entry_time_et":           _ts_to_et(int(timestamps[t])),
                "trigger_time_et":         _ts_to_et(int(timestamps[first_hit_bar])),
                "minutes_to_trigger":      minutes_to_trigger,
                "gain_pct":                round(gain_pct, 2),
                "above_20pct_duration_min": above_20_dur,
                "above_10pct_duration_min": above_10_dur,
                "entry_price":             round(float(entry_price), 4),
                "trigger_price":           round(float(trigger_price), 4),
            })

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date_range(self, start_date: str, end_date: str) -> List[str]:
        """Return date strings from start to end inclusive."""
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end     = datetime.strptime(end_date,   "%Y-%m-%d")
        dates: List[str] = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
