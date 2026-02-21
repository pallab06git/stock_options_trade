# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Detailed trade-by-trade backtesting simulator.

Simulates actual options trading with correct contract sizing (100 shares per
contract), entry/exit logic, and comprehensive trade logging.

Design
------
Uses **label-based simulation**: the pre-computed ``max_gain_120m`` and
``min_loss_120m`` columns in the feature CSVs tell us the best gain and worst
loss achievable in the 120 minutes following each bar.  Exit logic:

* Target hit only → exit at target price; time ≈ ``time_to_max_min``
* Stop only       → exit at stop price; time ≈ estimated fraction of window
* Both triggered  → conservative: stop assumed first (risk management)
* Neither         → time limit; final price at ``max_gain_120m``

Options math:
    1 contract = 100 shares
    cost per contract = price_per_share × 100
    num_contracts = floor(position_size_usd / cost_per_contract)
    P&L (USD) = (exit_price − entry_price) × 100 × num_contracts

Usage
-----
    from src.ml.trade_simulator import TradeSimulator

    simulator = TradeSimulator(position_size_usd=12_500)
    trades = simulator.simulate_period(test_df, y_proba, threshold=0.67)
    report = simulator.generate_monthly_report(trades, "2025-07")
    simulator.print_monthly_summary(report)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARES_PER_CONTRACT = 100  # Standard US equity options multiplier


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


class Trade:
    """Full lifecycle of a single options trade.

    Parameters
    ----------
    trade_id:
        Sequential identifier within the simulation run.
    entry_time:
        Human-readable entry timestamp string (``"YYYY-MM-DD HH:MM ET"``).
    contract_symbol:
        Option ticker (e.g. ``"O:SPY250701C00580000"``).
    entry_price_per_share:
        Option mid/close price at entry (dollars per share, NOT per contract).
    position_size_usd:
        Target position size in USD.  Actual size may be slightly less due to
        rounding down to whole contracts.
    confidence:
        Model predicted probability at signal time.
    trigger_factors:
        List of dicts describing the top features that drove the signal.
    """

    SHARES_PER_CONTRACT = SHARES_PER_CONTRACT

    def __init__(
        self,
        trade_id: int,
        entry_time: str,
        contract_symbol: str,
        entry_price_per_share: float,
        position_size_usd: float,
        confidence: float,
        trigger_factors: Optional[List[Dict]] = None,
    ) -> None:
        self.trade_id = trade_id
        self.entry_time = entry_time
        self.contract_symbol = contract_symbol
        self.entry_price_per_share = float(entry_price_per_share)

        # ── Options contract sizing ──────────────────────────────────────
        self.cost_per_contract = self.entry_price_per_share * self.SHARES_PER_CONTRACT
        # Round down: never overshoot position budget
        if self.cost_per_contract > 0:
            self.num_contracts = int(position_size_usd / self.cost_per_contract)
        else:
            self.num_contracts = 0
        self.actual_position_size = self.num_contracts * self.cost_per_contract

        self.confidence = float(confidence)
        self.trigger_factors: List[Dict] = trigger_factors or []

        # ── Exit fields (populated by close_trade) ───────────────────────
        self.exit_time: Optional[str] = None
        self.exit_price_per_share: Optional[float] = None
        self.exit_cost_per_contract: Optional[float] = None
        self.exit_reason: Optional[str] = None
        self.time_in_trade_minutes: Optional[float] = None
        self.profit_loss_usd: Optional[float] = None
        self.profit_loss_pct: Optional[float] = None
        self.is_winner: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_trade(
        self,
        exit_time: str,
        exit_price_per_share: float,
        exit_reason: str,
        time_in_trade_minutes: Optional[float] = None,
    ) -> None:
        """Close the trade and compute P&L.

        Parameters
        ----------
        exit_time:
            Human-readable exit timestamp string.
        exit_price_per_share:
            Option price at exit (dollars per share).
        exit_reason:
            One of ``"Target hit"``, ``"Stop-loss"``, ``"Time limit"``, etc.
        time_in_trade_minutes:
            If provided, overrides the default time calculation (used when
            we only have label-based timing estimates).
        """
        self.exit_time = exit_time
        self.exit_price_per_share = float(exit_price_per_share)
        self.exit_cost_per_contract = self.exit_price_per_share * self.SHARES_PER_CONTRACT
        self.exit_reason = exit_reason

        if time_in_trade_minutes is not None:
            self.time_in_trade_minutes = float(time_in_trade_minutes)
        else:
            # Fallback: parse timestamps if they look like "YYYY-MM-DD HH:MM ET"
            try:
                entry_dt = pd.to_datetime(self.entry_time.replace(" ET", ""))
                exit_dt = pd.to_datetime(self.exit_time.replace(" ET", ""))
                self.time_in_trade_minutes = (exit_dt - entry_dt).total_seconds() / 60.0
            except Exception:
                self.time_in_trade_minutes = 0.0

        # ── P&L ───────────────────────────────────────────────────────────
        if self.num_contracts > 0 and self.cost_per_contract > 0:
            profit_per_contract = self.exit_cost_per_contract - self.cost_per_contract
            self.profit_loss_usd = profit_per_contract * self.num_contracts
            self.profit_loss_pct = (
                (self.exit_cost_per_contract - self.cost_per_contract)
                / self.cost_per_contract
                * 100.0
            )
        else:
            self.profit_loss_usd = 0.0
            self.profit_loss_pct = 0.0

        self.is_winner = (self.profit_loss_usd or 0.0) > 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the trade."""
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "contract_symbol": self.contract_symbol,
            # Per-share prices
            "entry_price_per_share": round(self.entry_price_per_share, 4),
            "exit_price_per_share": (
                round(self.exit_price_per_share, 4) if self.exit_price_per_share is not None else None
            ),
            # Contract details
            "cost_per_contract_entry": round(self.cost_per_contract, 2),
            "cost_per_contract_exit": (
                round(self.exit_cost_per_contract, 2) if self.exit_cost_per_contract is not None else None
            ),
            "num_contracts": self.num_contracts,
            "actual_position_size": round(self.actual_position_size, 2),
            # Results
            "confidence": round(self.confidence, 4),
            "time_in_trade_minutes": (
                round(self.time_in_trade_minutes, 1) if self.time_in_trade_minutes is not None else None
            ),
            "profit_loss_usd": (
                round(self.profit_loss_usd, 2) if self.profit_loss_usd is not None else None
            ),
            "profit_loss_pct": (
                round(self.profit_loss_pct, 2) if self.profit_loss_pct is not None else None
            ),
            "exit_reason": self.exit_reason,
            "is_winner": self.is_winner,
            "trigger_factors": self.trigger_factors,
        }


# ---------------------------------------------------------------------------
# TradeSimulator
# ---------------------------------------------------------------------------


class TradeSimulator:
    """Simulate options trades using pre-computed max/min gain labels.

    Parameters
    ----------
    position_size_usd:
        Target USD size per trade (default $12,500).
    target_gain_pct:
        Take-profit level as a percentage gain (default 30.0%).
    stop_loss_pct:
        Stop-loss level as a percentage loss (default -12.0%).
    max_time_minutes:
        Maximum holding period before time-limit exit (default 120).
    fee_per_trade_usd:
        Round-trip commission per trade in USD (default $4.00).
    """

    def __init__(
        self,
        position_size_usd: float = 12_500.0,
        target_gain_pct: float = 30.0,
        stop_loss_pct: float = -12.0,
        max_time_minutes: int = 120,
        fee_per_trade_usd: float = 4.0,
    ) -> None:
        self.position_size_usd = float(position_size_usd)
        self.target_gain_pct = float(target_gain_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.max_time_minutes = int(max_time_minutes)
        self.fee_per_trade_usd = float(fee_per_trade_usd)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate_from_label_row(
        self,
        row: pd.Series,
        trade_id: int,
        confidence: float,
        trigger_factors: Optional[List[Dict]] = None,
    ) -> Optional[Trade]:
        """Simulate one trade from a feature-CSV row using pre-computed labels.

        Uses ``close`` as entry price, ``max_gain_120m`` / ``min_loss_120m``
        (both in percent) and ``time_to_max_min`` to determine exit.

        Exit priority (conservative):
        1. Both target AND stop triggered → assume stop hit first.
        2. Target only → exit at target; time = ``time_to_max_min``.
        3. Stop only   → exit at stop; time ≈ 30 min (estimated).
        4. Neither     → time limit; price at ``max_gain_120m``.

        Returns ``None`` if the row has invalid price data (zero/NaN close).
        """
        entry_price = float(row.get("close", row.get("opt_close", 0.0)) or 0.0)
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None

        max_gain_pct = float(row.get("max_gain_120m", 0.0) or 0.0)
        min_loss_pct = float(row.get("min_loss_120m", 0.0) or 0.0)
        time_to_max = float(row.get("time_to_max_min", self.max_time_minutes) or self.max_time_minutes)

        # Build human-readable entry timestamp
        date_str = str(row.get("date", ""))
        hour = int(row.get("hour_et", 9))
        minute = int(row.get("minute_et", 30))
        entry_time = f"{date_str} {hour:02d}:{minute:02d} ET"

        ticker = str(row.get("ticker", "UNKNOWN"))

        trade = Trade(
            trade_id=trade_id,
            entry_time=entry_time,
            contract_symbol=ticker,
            entry_price_per_share=entry_price,
            position_size_usd=self.position_size_usd,
            confidence=confidence,
            trigger_factors=trigger_factors or [],
        )

        # Skip if we can't buy even 1 contract
        if trade.num_contracts < 1:
            return None

        # ── Exit logic ────────────────────────────────────────────────────
        hits_target = max_gain_pct >= self.target_gain_pct
        hits_stop = min_loss_pct <= self.stop_loss_pct

        if hits_target and hits_stop:
            # Conservative: assume stop hit first (worst case)
            exit_pct = self.stop_loss_pct / 100.0
            exit_price = entry_price * (1.0 + exit_pct)
            exit_reason = "Stop-loss (before target)"
            exit_minutes = max(5.0, time_to_max * 0.30)  # rough estimate

        elif hits_target:
            exit_pct = self.target_gain_pct / 100.0
            exit_price = entry_price * (1.0 + exit_pct)
            exit_reason = "Target hit"
            exit_minutes = max(1.0, time_to_max)

        elif hits_stop:
            exit_pct = self.stop_loss_pct / 100.0
            exit_price = entry_price * (1.0 + exit_pct)
            exit_reason = "Stop-loss"
            exit_minutes = 30.0  # estimated — we don't know exact bar

        else:
            # Time limit: exit at whatever the max gain was (partial or zero)
            exit_pct = max_gain_pct / 100.0
            exit_price = entry_price * (1.0 + exit_pct)
            exit_reason = "Time limit (120 min)"
            exit_minutes = float(self.max_time_minutes)

        # Build exit timestamp
        exit_hour = hour + int((minute + int(exit_minutes)) // 60)
        exit_min = int((minute + int(exit_minutes)) % 60)
        exit_time = f"{date_str} {exit_hour:02d}:{exit_min:02d} ET"

        trade.close_trade(
            exit_time=exit_time,
            exit_price_per_share=round(exit_price, 4),
            exit_reason=exit_reason,
            time_in_trade_minutes=round(exit_minutes, 1),
        )
        return trade

    def simulate_period(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        threshold: float,
        model: Any = None,
        feature_cols: Optional[List[str]] = None,
    ) -> List[Trade]:
        """Simulate all trades in a test period.

        Parameters
        ----------
        df:
            Feature DataFrame (as loaded by ``load_features``), containing
            ``close``, ``max_gain_120m``, ``min_loss_120m``, etc.
        predictions:
            Predicted probabilities from ``model.predict_proba``; must have
            the same length as ``df``.
        threshold:
            Minimum probability to fire a signal.
        model:
            Fitted XGBClassifier.  If provided, its ``feature_importances_``
            are used to populate ``trigger_factors`` for each trade.
        feature_cols:
            Column names used as model features (needed for trigger_factors).

        Returns
        -------
        List of completed ``Trade`` objects (excludes skipped rows with 0 contracts).
        """
        signal_mask = predictions >= threshold
        signal_indices = np.where(signal_mask)[0]

        # Pre-compute top-5 feature importances for trigger_factors
        importance_top5: List[Dict] = []
        if model is not None and feature_cols is not None:
            try:
                fi = model.feature_importances_
                top_idx = np.argsort(fi)[::-1][:5]
                importance_top5 = [
                    {"feature": feature_cols[i], "importance": round(float(fi[i]), 4)}
                    for i in top_idx
                ]
            except Exception:
                pass

        trades: List[Trade] = []
        for trade_id, idx in enumerate(signal_indices, start=1):
            row = df.iloc[idx]
            confidence = float(predictions[idx])

            # For trigger_factors, show feature importance + actual value
            trigger_factors = []
            if importance_top5 and feature_cols is not None:
                for item in importance_top5:
                    fname = item["feature"]
                    fval = float(row.get(fname, float("nan")))
                    trigger_factors.append(
                        {
                            "feature": fname,
                            "value": round(fval, 4) if np.isfinite(fval) else None,
                            "importance": item["importance"],
                        }
                    )

            trade = self.simulate_from_label_row(row, trade_id, confidence, trigger_factors)
            if trade is not None:
                trades.append(trade)

        return trades

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_monthly_report(
        self, trades: List[Trade], month_str: str
    ) -> Dict[str, Any]:
        """Generate a comprehensive monthly trading report.

        Parameters
        ----------
        trades:
            List of completed ``Trade`` objects for the month.
        month_str:
            Label string (e.g. ``"2025-07"``).

        Returns
        -------
        Dict with all summary statistics and individual trade dicts.
        """
        if not trades:
            return {
                "month": month_str,
                "total_trades": 0,
                "message": "No trades executed",
                "trades": [],
            }

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        gross_profit = sum(t.profit_loss_usd for t in winners if t.profit_loss_usd)
        gross_loss = sum(t.profit_loss_usd for t in losers if t.profit_loss_usd)
        net_profit = gross_profit + gross_loss  # gross_loss is negative
        total_fees = len(trades) * self.fee_per_trade_usd
        net_profit_after_fees = net_profit - total_fees

        times = [t.time_in_trade_minutes for t in trades if t.time_in_trade_minutes is not None]

        # Contract type: find 'C' or 'P' in ticker
        calls = [t for t in trades if _extract_contract_type(t.contract_symbol) == "C"]
        puts = [t for t in trades if _extract_contract_type(t.contract_symbol) == "P"]

        exit_reasons: Dict[str, int] = {}
        for t in trades:
            r = t.exit_reason or "Unknown"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        total_capital = sum(t.actual_position_size for t in trades)

        return {
            "month": month_str,
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(trades),
            # Profit metrics
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "net_profit_usd": round(net_profit, 2),
            "total_fees_usd": round(total_fees, 2),
            "net_profit_after_fees_usd": round(net_profit_after_fees, 2),
            # Per-trade profit metrics
            "avg_profit_per_win_usd": round(
                float(np.mean([t.profit_loss_usd for t in winners])) if winners else 0.0, 2
            ),
            "median_profit_per_win_usd": round(
                float(np.median([t.profit_loss_usd for t in winners])) if winners else 0.0, 2
            ),
            "best_trade_usd": round(
                float(max(t.profit_loss_usd for t in winners)) if winners else 0.0, 2
            ),
            "avg_loss_per_loss_usd": round(
                float(np.mean([t.profit_loss_usd for t in losers])) if losers else 0.0, 2
            ),
            "median_loss_per_loss_usd": round(
                float(np.median([t.profit_loss_usd for t in losers])) if losers else 0.0, 2
            ),
            "worst_trade_usd": round(
                float(min(t.profit_loss_usd for t in losers)) if losers else 0.0, 2
            ),
            # Return metrics
            "avg_return_per_trade_pct": round(
                float(np.mean([t.profit_loss_pct for t in trades if t.profit_loss_pct is not None])), 2
            ),
            "roi_pct": round(
                (net_profit_after_fees / total_capital * 100.0) if total_capital > 0 else 0.0, 2
            ),
            # Time metrics
            "avg_time_minutes": round(float(np.mean(times)), 1) if times else 0.0,
            "median_time_minutes": round(float(np.median(times)), 1) if times else 0.0,
            "min_time_minutes": round(float(np.min(times)), 1) if times else 0.0,
            "max_time_minutes": round(float(np.max(times)), 1) if times else 0.0,
            # Position metrics
            "total_capital_deployed_usd": round(total_capital, 2),
            "total_contracts_traded": sum(t.num_contracts for t in trades),
            "calls_traded": len(calls),
            "puts_traded": len(puts),
            # Exit reason breakdown
            "exit_reasons": exit_reasons,
            # Individual trades
            "trades": [t.to_dict() for t in trades],
        }

    def print_trade_log(self, trade: Trade, trade_num: int) -> None:
        """Print a single trade's full lifecycle to stdout."""
        print(f"\nTrade #{trade_num}")
        print("─" * 80)
        print("Entry:")
        print(f"  Date/Time:            {trade.entry_time}")
        print(f"  Contract:             {trade.contract_symbol}")
        print(f"  Entry Price:          ${trade.entry_price_per_share:.2f} per share")
        print(f"  Cost per Contract:    ${trade.cost_per_contract:,.0f}  (100 shares)")
        print(f"  Number of Contracts:  {trade.num_contracts}")
        print(f"  Actual Position:      ${trade.actual_position_size:,.0f}")
        print(f"  Model Confidence:     {trade.confidence:.1%}")
        print()
        print("Exit:")
        print(f"  Date/Time:            {trade.exit_time}")
        exit_price = trade.exit_price_per_share or 0.0
        exit_cost = trade.exit_cost_per_contract or 0.0
        print(f"  Exit Price:           ${exit_price:.2f} per share")
        print(f"  Cost per Contract:    ${exit_cost:,.0f}  (100 shares)")
        print(f"  Exit Reason:          {trade.exit_reason}")
        print(f"  Time in Trade:        {trade.time_in_trade_minutes:.1f} minutes")
        print()

        pnl_usd = trade.profit_loss_usd or 0.0
        pnl_pct = trade.profit_loss_pct or 0.0
        price_diff = exit_price - trade.entry_price_per_share
        contract_pnl = exit_cost - trade.cost_per_contract

        if trade.is_winner:
            print("Result: WIN")
            print(
                f"  Price Change:         ${trade.entry_price_per_share:.2f} → "
                f"${exit_price:.2f} (+${price_diff:.2f} per share)"
            )
            print(f"  Per Contract Profit:  +${contract_pnl:,.0f}")
            print(
                f"  Total Profit:         {trade.num_contracts} contracts × "
                f"${contract_pnl:,.0f} = +${pnl_usd:,.0f}  (+{pnl_pct:.1f}%)"
            )
        else:
            print("Result: LOSS")
            print(
                f"  Price Change:         ${trade.entry_price_per_share:.2f} → "
                f"${exit_price:.2f} (${price_diff:.2f} per share)"
            )
            print(f"  Per Contract Loss:    ${contract_pnl:,.0f}")
            print(
                f"  Total Loss:           {trade.num_contracts} contracts × "
                f"${contract_pnl:,.0f} = ${pnl_usd:,.0f}  ({pnl_pct:.1f}%)"
            )

        if trade.trigger_factors:
            print()
            print("  Top signal factors:")
            for f in trade.trigger_factors[:3]:
                val_str = f"{f['value']:.4g}" if f.get("value") is not None else "n/a"
                print(f"    {f['feature']} = {val_str}  (importance {f.get('importance', 0):.3f})")

        print("─" * 80)

    def print_monthly_summary(self, report: Dict[str, Any]) -> None:
        """Print the monthly summary report to stdout."""
        print("\n" + "=" * 70)
        print(f"MONTHLY SUMMARY — {report['month']}")
        print("=" * 70 + "\n")

        if report["total_trades"] == 0:
            print("No trades executed this month.")
            print("=" * 70)
            return

        print(f"Total Trades:      {report['total_trades']}")
        print(
            f"Winning Trades:    {report['winning_trades']}  ({report['win_rate']:.1%})"
        )
        print(f"Losing Trades:     {report['losing_trades']}")
        print()

        print("Position Sizing:")
        print(f"  Target per trade:    ${self.position_size_usd:,.0f}")
        print(f"  Total deployed:      ${report['total_capital_deployed_usd']:,.0f}")
        print(f"  Contracts:           "
              f"{report['calls_traded']} calls / {report['puts_traded']} puts "
              f"({report['total_contracts_traded']} total)")
        print()

        print("Time in Market:")
        print(f"  Average:   {report['avg_time_minutes']:.1f} min")
        print(f"  Median:    {report['median_time_minutes']:.1f} min")
        print(f"  Range:     {report['min_time_minutes']:.1f} – {report['max_time_minutes']:.1f} min")
        print()

        print("Exit Reasons:")
        for reason, count in sorted(report["exit_reasons"].items()):
            pct = count / report["total_trades"] * 100
            print(f"  {reason:<30} {count:>4}  ({pct:.0f}%)")
        print()

        if report["winning_trades"] > 0:
            print("Winning Trades:")
            print(f"  Average profit:    +${report['avg_profit_per_win_usd']:>10,.0f}")
            print(f"  Median profit:     +${report['median_profit_per_win_usd']:>10,.0f}")
            print(f"  Best trade:        +${report['best_trade_usd']:>10,.0f}")
            print(f"  Total gross:       +${report['gross_profit_usd']:>10,.0f}")
            print()

        if report["losing_trades"] > 0:
            print("Losing Trades:")
            print(f"  Average loss:       ${report['avg_loss_per_loss_usd']:>10,.0f}")
            print(f"  Median loss:        ${report['median_loss_per_loss_usd']:>10,.0f}")
            print(f"  Worst trade:        ${report['worst_trade_usd']:>10,.0f}")
            print(f"  Total gross:        ${report['gross_loss_usd']:>10,.0f}")
            print()

        print("Net Results:")
        net = report["net_profit_after_fees_usd"]
        sign = "+" if net >= 0 else ""
        print(f"  Gross P&L:         {sign}${report['net_profit_usd']:>10,.0f}")
        print(f"  Fees:               -${report['total_fees_usd']:>9,.0f}")
        print(f"  Net Profit:        {sign}${net:>10,.0f}")
        print(f"  Avg return/trade:    {report['avg_return_per_trade_pct']:>+9.1f}%")
        print(f"  Monthly ROI:         {report['roi_pct']:>+9.2f}%")
        print()

        if net >= 25_000:
            print("Target assessment: EXCEEDS TARGET (net > $25,000)")
        elif net >= 20_000:
            print("Target assessment: MEETS TARGET (net $20,000 – $25,000)")
        elif net >= 15_000:
            print("Target assessment: BELOW TARGET (net $15,000 – $20,000)")
        else:
            print("Target assessment: INSUFFICIENT (net < $15,000)")

        print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_contract_type(ticker: str) -> str:
    """Return ``'C'`` or ``'P'`` from an option ticker, or ``'X'`` if not found."""
    m = re.search(r"\d([CP])\d", ticker)
    return m.group(1) if m else "X"
