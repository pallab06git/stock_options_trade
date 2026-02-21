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
        print(f"Contract-days scanned:    {cdays}")
        print(f"Total minute bars:        {t_bars:,}")
        print(f"Total events:             {n_events:,}")

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
                f"Events/contract-day:      "
                f"min={int(all_counts.min())} "
                f"median={np.median(all_counts):.1f} "
                f"max={int(all_counts.max())}"
            )

            # Positive-minute stats (same definition as backtest)
            total_pos_mins = int(df_ev["above_20pct_duration_min"].sum())
            pos_rate = total_pos_mins / t_bars * 100.0 if t_bars > 0 else 0.0
            med_dur = df_ev["above_20pct_duration_min"].median()
            mean_dur = df_ev["above_20pct_duration_min"].mean()
            print(f"Total >20% minutes:       {total_pos_mins:,}")
            print(f"Positive-minute rate:     {pos_rate:.2f}%")
            print(f"Duration >20% (med/mean): {med_dur:.1f} / {mean_dur:.1f} min")

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
            # Key observations (auto-generated, matching backtest format)
            # ------------------------------------------------------------------

            # 1. Positive-minute rate
            if pos_rate >= 50:
                observations.append(
                    f"{pos_rate:.2f}% positive-minute rate — over half of all scanned minutes "
                    f"the option was above its +{self.trigger_pct:.0f}% trigger level, which "
                    f"reflects real 0DTE behavior (options spike and stay elevated for long stretches)"
                )
            elif pos_rate >= 20:
                observations.append(
                    f"{pos_rate:.2f}% positive-minute rate — roughly 1-in-"
                    f"{round(100/pos_rate):.0f} minutes the option was above its "
                    f"+{self.trigger_pct:.0f}% trigger level"
                )
            else:
                observations.append(
                    f"{pos_rate:.2f}% positive-minute rate — moves above "
                    f"+{self.trigger_pct:.0f}% are infrequent; most contracts "
                    f"expire without a significant spike"
                )

            # 2. Peak trigger hour
            if not trig_hour_counts.empty:
                peak_hour = int(trig_hour_counts.idxmax())
                peak_cnt  = int(trig_hour_counts.max())
                peak_pct  = peak_cnt / n_events * 100
                if peak_hour >= 15:
                    observations.append(
                        f"Late-day concentration: {peak_hour:02d}:xx has the most events "
                        f"({peak_cnt}, {peak_pct:.0f}%), consistent with gamma acceleration into close"
                    )
                elif peak_hour <= 9:
                    observations.append(
                        f"Open-bell dominance: 09:xx has the most events "
                        f"({peak_cnt}, {peak_pct:.0f}%), driven by opening-range volatility"
                    )
                else:
                    observations.append(
                        f"{peak_hour:02d}:xx has the most events ({peak_cnt}, {peak_pct:.0f}%), "
                        f"mid-session momentum driving intraday moves"
                    )

            # 3. Morning distribution (09–11:xx)
            morning = {h: trig_hour_counts[h] for h in [9, 10, 11] if h in trig_hour_counts.index}
            if len(morning) >= 2:
                vals = list(morning.values())
                spread = max(vals) - min(vals)
                if spread <= max(vals) * 0.25:
                    avg_m = sum(vals) / len(vals)
                    labels = " / ".join(f"{h:02d}:xx" for h in morning)
                    observations.append(
                        f"Morning distribution even: {labels} each have "
                        f"~{avg_m:.0f} events — no single morning hour dominates"
                    )
                else:
                    peak_m = max(morning, key=morning.get)
                    observations.append(
                        f"Morning peak at {peak_m:02d}:xx ({morning[peak_m]} events) "
                        f"while other morning hours are quieter"
                    )

            # 4. Duration skew
            if mean_dur > 2 * med_dur:
                observations.append(
                    f"Median {med_dur:.0f} min above +{self.trigger_pct:.0f}% but "
                    f"mean {mean_dur:.1f} min — heavily right-skewed, most events are "
                    f"brief but a few run for hours"
                )
            elif mean_dur > 1.5 * med_dur:
                observations.append(
                    f"Median {med_dur:.0f} min above +{self.trigger_pct:.0f}% but "
                    f"mean {mean_dur:.1f} min — moderately right-skewed; some events "
                    f"sustain well beyond the typical duration"
                )
            else:
                observations.append(
                    f"Median {med_dur:.0f} min and mean {mean_dur:.1f} min above "
                    f"+{self.trigger_pct:.0f}% — relatively symmetric duration distribution"
                )

        else:
            print(f"Events/contract-day:      min=0 median=0.0 max=0")
            print(f"Total >20% minutes:       0")
            print(f"Positive-minute rate:     0.00%")
            print(f"Duration >20% (med/mean): N/A")
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
