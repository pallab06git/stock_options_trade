# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Extract and annotate all historical signals from the ensemble strategy.

For each bar where AVG(LightGBM, RF) >= threshold, captures:
    - Option / contract metadata
    - Entry conditions (price, IV, momentum, time)
    - Model confidence (per model and combined)
    - Plain-English explanation of why the signal fired
    - Actual outcome (magnitude, direction, P&L)
    - Simplified straddle P&L simulation
    - Single-option P&L (if you bought just the signalled option)
    - Direction analysis (did the option go UP or DOWN?)
    - Cumulative P&L across all signals

Column mapping
--------------
The feature CSVs do NOT contain delta/gamma/theta/vega or option_mid.
Corrections applied vs the original spec:
    option_symbol        → ticker         (O:SPY260220C00687000)
    option_mid           → close
    abs_magnitude        → abs_max_move_pct
    time_to_peak         → time_to_max_min
    rsi_14               → opt_rsi_14
    spy_return_5min      → spy_return_5m
    gain_magnitude_bucket→ magnitude_bucket
    target               → target_magnitude
    iv                   → implied_volatility
    contract_type        → 1 = CALL, 0 = PUT (int)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Option ticker pattern:  O:SPY{YYMMDD}{C|P}{8-digit-strike}
_TICKER_RE = re.compile(r"SPY(\d{6})([CP])(\d{8})")


def _extract_strike(ticker: str) -> float:
    """Return strike price from Polygon option ticker string."""
    m = _TICKER_RE.search(ticker)
    return int(m.group(3)) / 1000.0 if m else float("nan")


def _extract_expiry(ticker: str) -> str:
    """Return 'YYYY-MM-DD' expiry from Polygon option ticker string."""
    m = _TICKER_RE.search(ticker)
    if m:
        d = m.group(1)
        return f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
    return "Unknown"


def _contract_type_str(val) -> str:
    """Convert numeric contract_type (1/0) to 'CALL'/'PUT'."""
    try:
        return "CALL" if int(val) == 1 else "PUT"
    except (TypeError, ValueError):
        return str(val)


def _ts_to_et(ts_ms: float) -> str:
    """Convert millisecond epoch timestamp to 'HH:MM ET' string."""
    try:
        return (
            pd.to_datetime(ts_ms, unit="ms", utc=True)
            .tz_convert("America/New_York")
            .strftime("%H:%M ET")
        )
    except Exception:
        return str(ts_ms)


class SignalExtractor:
    """Extract complete signal history with context, straddle simulation,
    single-option P&L, and direction analysis.

    Parameters
    ----------
    feature_cols:
        Ordered list of feature column names from the saved artifact.
    """

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_signals(
        self,
        lgbm_model,
        rf_model,
        full_df: pd.DataFrame,
        threshold: float = 0.97,
    ) -> pd.DataFrame:
        """Extract all signals across the full dataset.

        Parameters
        ----------
        lgbm_model:
            Fitted LightGBM classifier.
        rf_model:
            Fitted RandomForest classifier.
        full_df:
            Feature DataFrame with ``target_magnitude``,
            ``abs_max_move_pct``, ``move_direction``, ``magnitude_bucket``
            already added by ``MagnitudeLabeler.label()``.
        threshold:
            AVG-probability threshold for declaring a signal.

        Returns
        -------
        DataFrame with one row per signal and columns for metadata,
        model outputs, explanation, straddle simulation,
        single-option P&L, and cumulative P&L.
        """
        print("\n" + "=" * 80)
        print("  EXTRACTING ALL HISTORICAL SIGNALS")
        print("=" * 80)

        X = full_df[self.feature_cols].fillna(0.0).values.astype(np.float32)
        lgbm_proba = lgbm_model.predict_proba(X)[:, 1]
        rf_proba   = rf_model.predict_proba(X)[:, 1]
        avg_proba  = (lgbm_proba + rf_proba) / 2.0

        signals_mask = avg_proba >= threshold
        n_signals    = int(signals_mask.sum())
        print(f"\n  Total signals @ AVG≥{threshold}: {n_signals:,}")
        print(f"  Running extraction…")

        signal_rows: List[Dict[str, Any]] = []

        for idx in np.where(signals_mask)[0]:
            row      = full_df.iloc[int(idx)]
            ticker   = str(row.get("ticker", "Unknown"))
            ct_str   = _contract_type_str(row.get("contract_type", "?"))
            ts_ms    = row.get("timestamp", 0)
            ts_et    = _ts_to_et(float(ts_ms)) if ts_ms else "?"
            entry    = float(row.get("close", 0.0))
            iv       = float(row.get("implied_volatility", float("nan")))
            l_proba  = float(lgbm_proba[idx])
            r_proba  = float(rf_proba[idx])
            a_proba  = float(avg_proba[idx])

            info: Dict[str, Any] = {
                "signal_id":            int(idx),
                "date":                 str(row.get("date", "")),
                "time_et":              ts_et,
                "timestamp_ms":         float(ts_ms),
                "ticker":               ticker,
                "contract_type":        ct_str,
                # entry conditions
                "entry_price":          entry,
                "spy_price":            float(row.get("spy_close", float("nan"))),
                "strike":               _extract_strike(ticker),
                "expiry":               _extract_expiry(ticker),
                "implied_volatility":   iv,
                "log_moneyness":        float(row.get("log_moneyness", float("nan"))),
                "opt_price_change_open": float(row.get("opt_price_change_open", float("nan"))),
                "hour_et":              int(row.get("hour_et", 0)),
                "minute_et":            int(row.get("minute_et", 0)),
                "opt_rsi_14":           float(row.get("opt_rsi_14", float("nan"))),
                "spy_return_5m":        float(row.get("spy_return_5m", float("nan"))),
                "time_to_expiry_days":  float(row.get("time_to_expiry_days", float("nan"))),
                "is_0dte":              bool(row.get("is_0dte", False)),
                # model outputs
                "lgbm_confidence":      l_proba,
                "rf_confidence":        r_proba,
                "avg_confidence":       a_proba,
                # actual outcome
                "target_magnitude":     int(row.get("target_magnitude", -1)),
                "abs_max_move_pct":     float(row.get("abs_max_move_pct", float("nan"))),
                "move_direction":       str(row.get("move_direction", "flat")),
                "magnitude_bucket":     str(row.get("magnitude_bucket", "?")),
                "time_to_max_min":      float(row.get("time_to_max_min", float("nan"))),
            }

            info["explanation"] = self._generate_explanation(info)
            info.update(self._simulate_straddle(info))
            info.update(self._analyze_price_movement(row))
            info.update(self._calculate_single_option_pnl(entry, info))
            signal_rows.append(info)

        signals_df = pd.DataFrame(signal_rows)

        if len(signals_df) > 0:
            signals_df      = self._add_cumulative_pnl(signals_df)
            direction_stats = self._analyze_direction_prediction(signals_df)

            tp_count  = int((signals_df["target_magnitude"] == 1).sum())
            precision = tp_count / len(signals_df)
            print(f"\n  ✅ Extracted {len(signals_df):,} signals")
            print(
                f"     Date range:     {signals_df['date'].min()} "
                f"→ {signals_df['date'].max()}"
            )
            print(f"     Avg confidence: {signals_df['avg_confidence'].mean():.1%}")
            print(f"     Precision:      {precision:.1%}  ({tp_count} TP / {len(signals_df)-tp_count} FP)")

            print("\n" + "=" * 80)
            print("  DIRECTION ANALYSIS")
            print("=" * 80)
            print(
                f"\n  Options went UP:   {direction_stats['up_count']} "
                f"({direction_stats['up_pct']:.1%})"
            )
            print(
                f"  Options went DOWN: {direction_stats['down_count']} "
                f"({direction_stats['down_pct']:.1%})"
            )
            print(f"\n  Single-option strategy (buy the signalled option):")
            print(f"    Win rate:         {direction_stats['single_win_rate']:.1%}")
            print(f"    Total profit:     ${direction_stats['single_total_profit']:,.0f}")
            print(f"    Avg profit/trade: ${direction_stats['single_avg_profit']:,.0f}")
        else:
            print("  No signals found.")
            tp_count = 0

        logger.info(
            "SignalExtractor: %d signals  precision=%.3f  threshold=%.2f",
            len(signals_df),
            tp_count / max(len(signals_df), 1) if len(signals_df) > 0 else 0.0,
            threshold,
        )
        return signals_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_explanation(self, info: Dict[str, Any]) -> str:
        """Generate plain-English explanation of why the signal fired."""
        ct          = info["contract_type"]
        lm          = info["log_moneyness"]
        iv          = info["implied_volatility"]
        pc          = info["opt_price_change_open"]
        hour        = info["hour_et"]
        minute      = info["minute_et"]
        avg_conf    = info["avg_confidence"]

        # Moneyness description (log_moneyness: positive = OTM for calls)
        if abs(lm) > 0.10:
            moneyness_desc = "Deep out-of-the-money"
        elif abs(lm) > 0.05:
            moneyness_desc = "Out-of-the-money"
        elif lm < -0.05:
            moneyness_desc = "In-the-money"
        else:
            moneyness_desc = "Near-the-money"

        # IV description
        if not np.isnan(iv):
            if iv > 0.50:
                iv_desc = f"extreme implied volatility (IV={iv*100:.0f}%)"
            elif iv > 0.30:
                iv_desc = f"very high implied volatility (IV={iv*100:.0f}%)"
            elif iv > 0.20:
                iv_desc = f"elevated implied volatility (IV={iv*100:.0f}%)"
            else:
                iv_desc = f"normal implied volatility (IV={iv*100:.0f}%)"
        else:
            iv_desc = "unknown implied volatility"

        # Price momentum description
        if not np.isnan(pc):
            if pc > 10:
                momentum_desc = f"surging up strongly (+{pc:.1f}% since open)"
            elif pc > 3:
                momentum_desc = f"moving up (+{pc:.1f}% since open)"
            elif pc < -10:
                momentum_desc = f"dropping sharply ({pc:.1f}% since open)"
            elif pc < -3:
                momentum_desc = f"moving down ({pc:.1f}% since open)"
            else:
                momentum_desc = f"relatively flat ({pc:+.1f}% since open)"
        else:
            momentum_desc = "with unknown momentum"

        ampm  = "AM" if hour < 12 else "PM"
        h12   = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
        return (
            f"{moneyness_desc} {ct} option with {iv_desc} "
            f"that is {momentum_desc}. "
            f"Signal fired at {h12}:{minute:02d} {ampm} ET "
            f"(combined model confidence: {avg_conf:.1%}). "
            f"Historical pattern: setups matching these criteria "
            f"have produced explosive 20%+ moves."
        )

    def _simulate_straddle(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate straddle P&L using the signal contract's own data.

        Simplified model (no ATM lookup — data only contains OTM contracts):
            Entry cost     = 2 × close × 100  (estimated CALL + PUT at same price)
            TP exit value  = winner × (1 + abs_move/100) + loser × 0.94
            FP exit value  = both legs × 0.92  (theta decay)
        """
        entry       = info["entry_price"]
        is_tp       = info["target_magnitude"] == 1
        abs_move    = info["abs_max_move_pct"]
        direction   = info["move_direction"]

        if entry <= 0 or np.isnan(entry):
            return {
                "straddle_entry_price":  entry,
                "straddle_cost":         float("nan"),
                "straddle_exit_value":   float("nan"),
                "straddle_profit":       float("nan"),
                "straddle_return_pct":   float("nan"),
                "straddle_winner":       "N/A",
            }

        if is_tp and not np.isnan(abs_move):
            winner_exit = entry * (1.0 + abs_move / 100.0)
            loser_exit  = entry * 0.94
            exit_per_share = winner_exit + loser_exit
            winner_side = "CALL" if direction == "up" else "PUT"
        else:
            # FP: both legs lose ~4% each
            exit_per_share = entry * 2.0 * 0.92
            winner_side    = "NONE"

        cost_per_share   = entry * 2.0
        profit_per_share = exit_per_share - cost_per_share

        # Scale to 1 contract (100 shares per leg × 2 legs)
        straddle_cost   = cost_per_share   * 100
        straddle_profit = profit_per_share * 100
        straddle_return = (straddle_profit / straddle_cost) * 100

        return {
            "straddle_entry_price": entry,
            "straddle_cost":        straddle_cost,
            "straddle_exit_value":  exit_per_share * 100,
            "straddle_profit":      straddle_profit,
            "straddle_return_pct":  straddle_return,
            "straddle_winner":      winner_side,
        }

    def _analyze_price_movement(self, row) -> Dict[str, Any]:
        """Analyze what actually happened to the option price.

        Uses pre-computed labels from MagnitudeLabeler (max_gain_120m,
        min_loss_120m, move_direction, time_to_max_min) to determine
        direction and magnitude of the actual move, avoiding the complexity
        of forward-scanning a multi-ticker interleaved dataframe.
        """
        entry_price     = float(row.get("close", 0.0))
        max_gain_pct    = float(row.get("max_gain_120m", float("nan")))
        min_loss_pct    = float(row.get("min_loss_120m", float("nan")))  # negative value
        move_direction  = str(row.get("move_direction", "unknown")).lower()
        time_to_max_min = row.get("time_to_max_min", float("nan"))

        _nan = float("nan")

        if entry_price <= 0 or np.isnan(max_gain_pct):
            return {
                "price_direction":  "unknown",
                "price_change_pct": _nan,
                "exit_price":       entry_price,
                "max_price":        entry_price,
                "min_price":        entry_price,
                "max_gain_pct":     _nan,
                "max_loss_pct":     _nan,
                "minutes_to_max":   0,
                "minutes_to_min":   0,
                "time_to_exit":     0,
            }

        max_price = entry_price * (1.0 + max_gain_pct / 100.0)
        min_price = (
            entry_price * (1.0 + min_loss_pct / 100.0)
            if not np.isnan(min_loss_pct)
            else entry_price
        )

        t = (
            int(time_to_max_min)
            if not (isinstance(time_to_max_min, float) and np.isnan(time_to_max_min))
            else 0
        )

        if move_direction == "up":
            price_direction  = "UP"
            price_change_pct = max_gain_pct
            exit_price       = max_price
        elif move_direction == "down":
            price_direction  = "DOWN"
            price_change_pct = float(min_loss_pct) if not np.isnan(min_loss_pct) else -max_gain_pct
            exit_price       = min_price
        else:
            price_direction  = "FLAT"
            price_change_pct = 0.0
            exit_price       = entry_price

        return {
            "price_direction":  price_direction,
            "price_change_pct": price_change_pct,
            "exit_price":       exit_price,
            "max_price":        max_price,
            "min_price":        min_price,
            "max_gain_pct":     max_gain_pct,
            "max_loss_pct":     float(min_loss_pct) if not np.isnan(min_loss_pct) else _nan,
            "minutes_to_max":   t,
            "minutes_to_min":   t,
            "time_to_exit":     t,
        }

    def _calculate_single_option_pnl(
        self, entry_price: float, movement_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate P&L if you bought JUST the signalled option (1 contract = 100 shares)."""
        exit_price       = movement_analysis.get("exit_price", entry_price)
        price_change_pct = movement_analysis.get("price_change_pct", float("nan"))

        if entry_price <= 0 or np.isnan(entry_price):
            return {
                "single_option_entry_cost":   float("nan"),
                "single_option_exit_value":   float("nan"),
                "single_option_pnl_dollars":  float("nan"),
                "single_option_pnl_pct":      float("nan"),
                "single_option_profitable":   False,
            }

        shares      = 100
        entry_cost  = entry_price * shares
        exit_value  = float(exit_price) * shares
        pnl_dollars = exit_value - entry_cost
        is_profit   = pnl_dollars > 0

        return {
            "single_option_entry_cost":  entry_cost,
            "single_option_exit_value":  exit_value,
            "single_option_pnl_dollars": pnl_dollars,
            "single_option_pnl_pct":     float(price_change_pct),
            "single_option_profitable":  is_profit,
        }

    def _add_cumulative_pnl(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Sort by date/timestamp and compute cumulative single-option P&L."""
        sort_cols = ["date"]
        if "timestamp_ms" in signals_df.columns:
            sort_cols.append("timestamp_ms")
        signals_df = signals_df.sort_values(sort_cols).reset_index(drop=True)
        signals_df["cumulative_pnl"] = (
            signals_df["single_option_pnl_dollars"]
            .fillna(0.0)
            .cumsum()
        )
        return signals_df

    def _analyze_direction_prediction(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse whether the model is predicting direction or just magnitude."""
        up_count   = int((signals_df["price_direction"] == "UP").sum())
        down_count = int((signals_df["price_direction"] == "DOWN").sum())
        total      = len(signals_df)

        profitable_trades = int(signals_df["single_option_profitable"].sum())
        win_rate          = profitable_trades / total if total > 0 else 0.0
        total_profit      = float(signals_df["single_option_pnl_dollars"].sum())
        avg_profit        = float(signals_df["single_option_pnl_dollars"].mean()) if total > 0 else 0.0

        return {
            "up_count":            up_count,
            "down_count":          down_count,
            "up_pct":              up_count / total if total > 0 else 0.0,
            "down_pct":            down_count / total if total > 0 else 0.0,
            "single_win_rate":     win_rate,
            "single_total_profit": total_profit,
            "single_avg_profit":   avg_profit,
        }
