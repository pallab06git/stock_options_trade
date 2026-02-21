# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""False positive severity analysis for options spike predictions.

Motivation
----------
A false positive in this context means the model predicted "this option will
spike ≥20% in the next 120 minutes" but it did not.  The question for risk
management is not just *how often* the model is wrong, but *how wrong* it is
when it fails — i.e., what is the worst drawdown seen in the forward window
on a missed trade?

The ``min_loss_120m`` column (added to feature CSVs in Step 39) captures the
minimum percentage change of the option price in the 120-minute forward window
relative to the entry bar.  A value of ``-12.0`` means the option fell as much
as 12% below entry price at some point during the next 120 minutes.

By analysing this distribution across all false positives the module produces:

  1. Loss severity distribution (0–5%, 5–10%, … buckets)
  2. Stop-loss threshold analysis — what % of losses a given stop would catch
  3. ASCII histogram for terminal display
  4. Stop-loss recommendations (conservative, moderate, aggressive)
  5. Risk-reward summary

Inputs
------
The primary input is the ``trades_path`` CSV produced by ``ModelBacktester``
(e.g. ``data/reports/backtest/xgboost_v2_trades_*.csv``).  The CSV must
contain the columns: ``is_true_positive``, ``min_loss_120m``, ``max_gain_120m``,
``predicted_proba``.

Public API
----------
  PredictionErrorAnalyzer
    .load_false_positives(trades_path)   → pd.DataFrame
    .generate_risk_report(fp_df)         → dict
    .stop_loss_impact(fp_df, stop_losses) → dict
    .plot_ascii(fp_df, bins)             → str  (ASCII histogram)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Default stop-loss levels to evaluate (% below entry; negative = loss)
_DEFAULT_STOP_LOSSES = [-5.0, -10.0, -15.0, -20.0, -25.0]


class PredictionErrorAnalyzer:
    """Analyse how wrong the model is when it produces a false positive.

    Reads the per-trade CSV from ``ModelBacktester`` and focuses exclusively
    on false positives (predicted=1, actual=0) to characterise the downside
    risk a trader would face when the model fires an incorrect signal.

    The key metric is ``min_loss_120m``: the worst percentage drawdown seen
    in the 120-minute forward window after entry.  If ``min_loss_120m = -12``
    on a false positive, a stop-loss at -10% would have exited the trade with
    a controlled loss; without a stop the trader holds until expiry.
    """

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def load_false_positives(self, trades_path: Union[str, Path]) -> pd.DataFrame:
        """Load the backtest trades CSV and return only false positive rows.

        Args:
            trades_path: Path to the per-trade CSV from ``ModelBacktester``.

        Returns:
            DataFrame of false positive rows.  Returns an empty DataFrame if
            there are no false positives.

        Raises:
            FileNotFoundError: If *trades_path* does not exist.
            ValueError: If required columns are missing from the CSV.
        """
        path = Path(trades_path)
        if not path.exists():
            raise FileNotFoundError(f"PredictionErrorAnalyzer: file not found: {path}")

        df = pd.read_csv(path)

        required = {"is_true_positive", "min_loss_120m", "max_gain_120m"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"PredictionErrorAnalyzer: missing columns in trades CSV: {missing!r}. "
                f"Re-run backtest with the updated pipeline to generate these columns."
            )

        fp_df = df[df["is_true_positive"] == False].copy()  # noqa: E712
        fp_df = fp_df.reset_index(drop=True)

        logger.info(
            f"PredictionErrorAnalyzer: loaded {len(df)} trades | "
            f"{len(fp_df)} false positives"
        )
        return fp_df

    def generate_risk_report(self, fp_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute loss severity statistics for false positive trades.

        Args:
            fp_df: DataFrame of false positive rows (from ``load_false_positives``).

        Returns:
            Dict with keys:

            General:
              ``total_false_positives``      int
              ``pct_price_never_below_entry`` float  (min_loss_120m ≥ 0)

            Worst-drawdown distribution (all FPs):
              ``mean_worst_drawdown_pct``    float
              ``median_worst_drawdown_pct``  float
              ``p25_worst_drawdown_pct``     float   (25% had BETTER drawdown)
              ``p50_worst_drawdown_pct``     float
              ``p75_worst_drawdown_pct``     float   (75% had BETTER drawdown)
              ``p90_worst_drawdown_pct``     float
              ``max_worst_drawdown_pct``     float   (worst single trade)

            Loss buckets (% of FPs whose min_loss fell in each range):
              ``pct_0_to_5pct``   float
              ``pct_5_to_10pct``  float
              ``pct_10_to_15pct`` float
              ``pct_15_to_20pct`` float
              ``pct_over_20pct``  float

            Stop-loss impact (% of losses a stop would have capped):
              ``stop_5pct_triggered_pct``  float
              ``stop_10pct_triggered_pct`` float
              ``stop_15pct_triggered_pct`` float
              ``stop_20pct_triggered_pct`` float

            Recommendations:
              ``stop_loss_conservative_pct`` float   (p75 of all losses)
              ``stop_loss_moderate_pct``     float   (p90 of all losses)
              ``stop_loss_aggressive_pct``   float   (p95 of all losses)

        Raises:
            ValueError: If fp_df is empty or missing ``min_loss_120m``.
        """
        if fp_df.empty:
            raise ValueError("generate_risk_report: no false positives to analyse")
        if "min_loss_120m" not in fp_df.columns:
            raise ValueError("generate_risk_report: 'min_loss_120m' column required")

        losses = fp_df["min_loss_120m"].dropna()
        n = len(fp_df)

        if losses.empty:
            raise ValueError("generate_risk_report: all min_loss_120m values are NaN")

        n_never_below = int((losses >= 0).sum())

        def _pct_triggered(stop: float) -> float:
            """Fraction of FPs whose min_loss_120m < stop (stop-loss triggered)."""
            return float((losses < stop).sum()) / n

        report: Dict[str, Any] = {
            "total_false_positives": n,
            "pct_price_never_below_entry": round(n_never_below / n, 4),
            # Drawdown percentiles
            "mean_worst_drawdown_pct": round(float(losses.mean()), 2),
            "median_worst_drawdown_pct": round(float(losses.median()), 2),
            "p25_worst_drawdown_pct": round(float(losses.quantile(0.25)), 2),
            "p50_worst_drawdown_pct": round(float(losses.quantile(0.50)), 2),
            "p75_worst_drawdown_pct": round(float(losses.quantile(0.75)), 2),
            "p90_worst_drawdown_pct": round(float(losses.quantile(0.90)), 2),
            "max_worst_drawdown_pct": round(float(losses.min()), 2),
            # Loss buckets
            "pct_0_to_5pct": round(float(((losses >= -5) & (losses < 0)).sum()) / n, 4),
            "pct_5_to_10pct": round(float(((losses >= -10) & (losses < -5)).sum()) / n, 4),
            "pct_10_to_15pct": round(float(((losses >= -15) & (losses < -10)).sum()) / n, 4),
            "pct_15_to_20pct": round(float(((losses >= -20) & (losses < -15)).sum()) / n, 4),
            "pct_over_20pct": round(float((losses < -20).sum()) / n, 4),
            "pct_never_below_entry": round(n_never_below / n, 4),
            # Stop-loss impact
            "stop_5pct_triggered_pct": round(_pct_triggered(-5.0), 4),
            "stop_10pct_triggered_pct": round(_pct_triggered(-10.0), 4),
            "stop_15pct_triggered_pct": round(_pct_triggered(-15.0), 4),
            "stop_20pct_triggered_pct": round(_pct_triggered(-20.0), 4),
            # Recommendations (negative = stop-loss below entry)
            "stop_loss_conservative_pct": round(float(losses.quantile(0.75)), 2),
            "stop_loss_moderate_pct": round(float(losses.quantile(0.90)), 2),
            "stop_loss_aggressive_pct": round(float(losses.quantile(0.95)), 2),
        }

        logger.info(
            f"generate_risk_report: median drawdown={report['median_worst_drawdown_pct']:.1f}% | "
            f"p90 drawdown={report['p90_worst_drawdown_pct']:.1f}% | "
            f"max loss={report['max_worst_drawdown_pct']:.1f}%"
        )
        return report

    def stop_loss_impact(
        self,
        fp_df: pd.DataFrame,
        stop_losses: Optional[List[float]] = None,
    ) -> Dict[float, Dict[str, Any]]:
        """Evaluate each stop-loss level: what % of FPs does it catch, and at what cost?

        For each candidate stop-loss level, computes:
          - ``triggered_count``: how many FPs had min_loss_120m < stop_level
          - ``triggered_pct``: fraction of all FPs where stop would have fired
          - ``avg_exit_loss_pct``: average loss if you exit exactly at the stop
            (approximated as the stop level itself)
          - ``max_alternative_loss_pct``: for FPs NOT caught by the stop, the
            worst drawdown seen (i.e. what you'd experience without the stop)

        Args:
            fp_df:       DataFrame of false positives.
            stop_losses: List of stop-loss levels (negative %, e.g. [-5, -10, -15]).
                         Defaults to ``[-5, -10, -15, -20, -25]``.

        Returns:
            Dict keyed by stop level → impact stats dict.
        """
        if stop_losses is None:
            stop_losses = _DEFAULT_STOP_LOSSES

        losses = fp_df["min_loss_120m"].dropna()
        n = len(fp_df)
        result: Dict[float, Dict[str, Any]] = {}

        for stop in sorted(stop_losses):
            triggered = losses < stop
            not_triggered = ~triggered

            result[stop] = {
                "stop_level_pct": stop,
                "triggered_count": int(triggered.sum()),
                "triggered_pct": round(float(triggered.sum()) / n, 4) if n > 0 else 0.0,
                "exit_loss_pct": stop,  # controlled loss at stop
                "uncaught_max_loss_pct": (
                    round(float(losses[not_triggered].min()), 2)
                    if not_triggered.any()
                    else 0.0
                ),
                "uncaught_count": int(not_triggered.sum()),
            }

        return result

    def plot_ascii(self, fp_df: pd.DataFrame, bins: int = 10) -> str:
        """Return a multi-line ASCII histogram of the min_loss_120m distribution.

        Args:
            fp_df: DataFrame of false positive rows.
            bins:  Number of histogram buckets.  Default 10.

        Returns:
            Formatted string suitable for ``click.echo`` or ``print``.
        """
        if fp_df.empty or "min_loss_120m" not in fp_df.columns:
            return "(no false positive data)"

        losses = fp_df["min_loss_120m"].dropna()
        if losses.empty:
            return "(all min_loss_120m values are NaN)"

        counts, edges = np.histogram(losses, bins=bins)
        max_count = max(counts) if counts.max() > 0 else 1
        bar_width = 30

        lines = [
            "False Positive Drawdown Distribution (min_loss_120m)",
            "─" * 60,
        ]
        for i, count in enumerate(counts):
            left = edges[i]
            right = edges[i + 1]
            bar_len = int(count / max_count * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            pct = count / len(losses) * 100
            lines.append(f"[{left:6.1f}%, {right:6.1f}%)  {bar}  {count:4d} ({pct:4.1f}%)")

        lines.append("─" * 60)
        lines.append(
            f"n={len(losses)} | "
            f"median={losses.median():.1f}% | "
            f"p90={losses.quantile(0.90):.1f}% | "
            f"worst={losses.min():.1f}%"
        )
        return "\n".join(lines)

    def calculate_expected_value(
        self,
        precision: float,
        avg_win_pct: float = 20.0,
        avg_loss_pct: Optional[float] = None,
        stop_loss_pct: float = -10.0,
    ) -> Dict[str, Any]:
        """Calculate expected value of the trading strategy per signal.

        Models each signal as a Bernoulli trial: with probability *precision*
        the trade wins *avg_win_pct*, and with probability ``1 - precision``
        it loses *avg_loss_pct* (or *stop_loss_pct* if avg_loss_pct is None).

        The *stop_loss_pct* should be a negative number representing the
        maximum controlled loss on a false positive (e.g. ``-10.0`` means the
        position is exited at −10%).

        Args:
            precision:      Fraction of signals that are true positives (0–1).
            avg_win_pct:    Mean % gain on winning trades.  Default 20.0.
            avg_loss_pct:   Mean % loss on losing trades.  If ``None`` the
                            stop-loss level is used as the loss (assumes perfect
                            stop execution).
            stop_loss_pct:  Stop-loss level in % (negative). Used as avg_loss
                            when *avg_loss_pct* is not provided.  Default -10.0.

        Returns:
            Dict with keys:
              ``win_rate``          float  — precision
              ``loss_rate``         float  — 1 - precision
              ``avg_win_pct``       float
              ``avg_loss_pct``      float  — the loss used in the calculation
              ``expected_value_pct`` float — EV per trade in %
              ``profitable``        bool   — True when EV > 0
              ``breakeven_win_rate`` float — minimum win rate for EV = 0

        Raises:
            ValueError: If *precision* is not in [0, 1].
        """
        if not 0.0 <= precision <= 1.0:
            raise ValueError(
                f"calculate_expected_value: precision must be in [0, 1], got {precision!r}"
            )

        if avg_loss_pct is None:
            avg_loss_pct = stop_loss_pct

        win_rate = precision
        loss_rate = 1.0 - precision
        ev = win_rate * avg_win_pct + loss_rate * avg_loss_pct

        # Breakeven win rate: w * avg_win + (1-w) * avg_loss = 0
        # w = -avg_loss / (avg_win - avg_loss)
        if avg_win_pct != avg_loss_pct:
            breakeven = -avg_loss_pct / (avg_win_pct - avg_loss_pct)
        else:
            breakeven = float("nan")

        result: Dict[str, Any] = {
            "win_rate": round(win_rate, 6),
            "loss_rate": round(loss_rate, 6),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "expected_value_pct": round(ev, 4),
            "profitable": bool(ev > 0),
            "breakeven_win_rate": round(breakeven, 6) if not (breakeven != breakeven) else None,
        }

        logger.info(
            f"calculate_expected_value: precision={win_rate:.1%} | "
            f"EV={ev:+.2f}% per trade | profitable={result['profitable']}"
        )
        return result
