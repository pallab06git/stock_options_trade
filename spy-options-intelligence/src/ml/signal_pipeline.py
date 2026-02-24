# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""End-to-end signal pipeline orchestrating entry, exit, and simulation.

Wires together:
  - Feature loading
  - Stacked ensemble (entry scoring)
  - Exit signal model (exit decisions)
  - Real-bar simulator (trade execution)
  - Report generation (PipelineReport)

Usage
-----
    from src.ml.signal_pipeline import SignalPipeline

    pipeline = SignalPipeline(config)
    result = pipeline.run_backtest(
        ensemble_path="models/stacked_ensemble.pkl",
        exit_model_path="models/exit_model.pkl",
        start_date="2025-03-03",
        end_date="2025-11-25",
    )
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.exit_signal_model import ExitSignalModel
from src.ml.real_bar_simulator import RealBarSimulator
from src.ml.trade_simulator import Trade
from src.utils.logger import get_logger

logger = get_logger()


class PipelineReport:
    """Generate comprehensive reports from pipeline backtest results."""

    def generate_report(
        self,
        trades: List[Trade],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a full pipeline report.

        Sections:
          - Executive summary (total P&L, win rate, signal count)
          - Monthly P&L breakdown
          - Exit quality analysis
          - Trade log

        Args:
            trades: List of completed Trade objects.
            config: Pipeline configuration used.

        Returns:
            Report dict with all sections.
        """
        if not trades:
            return {
                "executive_summary": {
                    "total_trades": 0,
                    "net_pnl": 0.0,
                    "win_rate": 0.0,
                },
                "monthly_pnl": [],
                "exit_analysis": {},
                "trades": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
            }

        trade_dicts = [t.to_dict() for t in trades]

        # Executive summary
        winners = [t for t in trades if t.is_winner]
        total_pnl = sum(t.profit_loss_usd or 0 for t in trades)
        fees = len(trades) * config.get("fee_per_trade", 4.0)
        net_pnl = total_pnl - fees

        hold_times = [t.time_in_trade_minutes or 0 for t in trades]

        exec_summary = {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(trades) - len(winners),
            "win_rate": round(len(winners) / len(trades), 4),
            "gross_pnl": round(total_pnl, 2),
            "fees": round(fees, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
            "avg_hold_minutes": round(np.mean(hold_times), 1),
            "median_hold_minutes": round(np.median(hold_times), 1),
            "avg_trades_per_day": 0.0,  # Filled by caller
        }

        # Monthly P&L
        monthly = self._compute_monthly_pnl(trades, fees / len(trades))

        # Exit analysis
        exit_analysis = self._compute_exit_analysis(trades)

        return {
            "executive_summary": exec_summary,
            "monthly_pnl": monthly,
            "exit_analysis": exit_analysis,
            "trades": trade_dicts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }

    def _compute_monthly_pnl(
        self, trades: List[Trade], fee_per_trade: float
    ) -> List[Dict[str, Any]]:
        """Aggregate P&L by calendar month."""
        monthly: Dict[str, List[Trade]] = defaultdict(list)

        for t in trades:
            if t.entry_time:
                # Parse month from entry_time "YYYY-MM-DD HH:MM ET"
                month = t.entry_time[:7]
                monthly[month].append(t)

        result = []
        for month in sorted(monthly.keys()):
            month_trades = monthly[month]
            winners = [t for t in month_trades if t.is_winner]
            gross = sum(t.profit_loss_usd or 0 for t in month_trades)
            fees = len(month_trades) * fee_per_trade
            result.append({
                "month": month,
                "trades": len(month_trades),
                "winners": len(winners),
                "win_rate": round(len(winners) / len(month_trades), 4),
                "gross_pnl": round(gross, 2),
                "fees": round(fees, 2),
                "net_pnl": round(gross - fees, 2),
            })

        return result

    def _compute_exit_analysis(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze exit reasons and their P&L impact."""
        by_reason: Dict[str, List[Trade]] = defaultdict(list)
        for t in trades:
            reason = t.exit_reason or "Unknown"
            by_reason[reason].append(t)

        analysis = {}
        for reason, reason_trades in sorted(by_reason.items()):
            pnls = [t.profit_loss_usd or 0 for t in reason_trades]
            analysis[reason] = {
                "count": len(reason_trades),
                "avg_pnl": round(np.mean(pnls), 2),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(
                    sum(1 for t in reason_trades if t.is_winner) / len(reason_trades), 4
                ),
            }

        return analysis


class SignalPipeline:
    """End-to-end orchestrator for entry → exit → trade simulation.

    Parameters
    ----------
    config : dict
        Full configuration dict.  Reads from config["signal_pipeline"].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        cfg = self._config.get("signal_pipeline", {})

        self.position_size = cfg.get("position_size_usd", 10_000)
        self.max_signals_per_day = cfg.get("max_signals_per_day", 3)
        self.entry_threshold = cfg.get("entry_threshold", 0.70)
        self.exit_threshold = cfg.get("exit_threshold", 0.60)
        self.eod_close_minutes = cfg.get("eod_close_minutes_before", 10)
        self.max_hold_minutes = cfg.get("max_hold_minutes", 120)

        self._report_generator = PipelineReport()

    def run_backtest(
        self,
        ensemble_path: str | Path,
        exit_model_path: str | Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full backtest using entry ensemble + exit model.

        Orchestrates:
          1. Load models
          2. Load features per date
          3. Entry scoring with stacked ensemble
          4. Top-N selection per day
          5. Real-bar simulation with exit model
          6. Generate PipelineReport

        Args:
            ensemble_path: Path to stacked ensemble artifact.
            exit_model_path: Path to exit model artifact.
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD).
            output_dir: Optional output directory.

        Returns:
            Summary dict with all results.
        """
        features_dir = self._config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        raw_options_dir = self._config.get("feature_engineering", {}).get(
            "options_data_dir", "data/raw/options/minute"
        )

        # Use RealBarSimulator for actual trade execution
        sim_config = dict(self._config)
        sim_config.setdefault("signal_pipeline", {}).update({
            "position_size_usd": self.position_size,
            "max_signals_per_day": self.max_signals_per_day,
            "entry_threshold": self.entry_threshold,
            "eod_close_minutes_before": self.eod_close_minutes,
        })

        simulator = RealBarSimulator(sim_config)
        sim_result = simulator.simulate_period(
            entry_model_path=str(ensemble_path),
            exit_model_path=str(exit_model_path),
            features_dir=features_dir,
            raw_options_dir=raw_options_dir,
            start_date=start_date,
            end_date=end_date,
        )

        # Reconstruct Trade objects from sim_result
        trades = self._reconstruct_trades(sim_result.get("trades", []))

        # Generate full report
        pipeline_config = {
            "position_size_usd": self.position_size,
            "max_signals_per_day": self.max_signals_per_day,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "max_hold_minutes": self.max_hold_minutes,
            "fee_per_trade": simulator.fee,
        }

        report = self._report_generator.generate_report(trades, pipeline_config)

        # Update with simulation metadata
        n_days = sim_result.get("n_days", 0)
        report["executive_summary"]["avg_trades_per_day"] = round(
            len(trades) / max(n_days, 1), 1
        )
        report["start_date"] = start_date
        report["end_date"] = end_date
        report["n_days"] = n_days

        # Build result
        result = {
            "start_date": start_date,
            "end_date": end_date,
            "n_days": n_days,
            "total_trades": len(trades),
            "win_rate": report["executive_summary"]["win_rate"],
            "net_pnl": report["executive_summary"]["net_pnl"],
            "avg_trades_per_day": report["executive_summary"]["avg_trades_per_day"],
            "report": report,
        }

        # Save if output_dir specified
        if output_dir:
            self._save_report(report, output_dir)

        return result

    def generate_daily_signals(
        self,
        date: str,
        features_df: pd.DataFrame,
        ensemble_path: str | Path,
    ) -> pd.DataFrame:
        """Generate entry signals for a single day (paper trading mode).

        Args:
            date: Trading date.
            features_df: Feature DataFrame for the date.
            ensemble_path: Path to ensemble artifact.

        Returns:
            DataFrame with signal candidates ranked by probability.
        """
        import joblib
        from src.ml.stacked_ensemble import StackedEnsemble

        artifact = joblib.load(ensemble_path)

        if "base_models" in artifact:
            model = StackedEnsemble()
            model.load(ensemble_path)
            feature_cols = artifact["feature_cols"]
            proba = model.predict_proba(features_df, feature_cols)
        else:
            model = artifact["model"]
            feature_cols = artifact["feature_cols"]
            available_cols = [c for c in feature_cols if c in features_df.columns]
            X = features_df[available_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(X)[:, 1]

        # Build signals DataFrame
        signals = features_df.copy()
        signals["entry_proba"] = proba
        signals["date"] = date

        # Filter by threshold and rank
        signals = signals[signals["entry_proba"] >= self.entry_threshold]
        signals = signals.nlargest(self.max_signals_per_day, "entry_proba")

        return signals.reset_index(drop=True)

    def _reconstruct_trades(self, trade_dicts: List[Dict]) -> List[Trade]:
        """Reconstruct Trade objects from serialized dicts."""
        trades = []
        for td in trade_dicts:
            t = Trade(
                trade_id=td.get("trade_id", 0),
                entry_time=td.get("entry_time", ""),
                contract_symbol=td.get("contract_symbol", ""),
                entry_price_per_share=td.get("entry_price_per_share", 0),
                position_size_usd=self.position_size,
                confidence=td.get("confidence", 0),
            )
            if td.get("exit_time"):
                t.close_trade(
                    exit_time=td["exit_time"],
                    exit_price_per_share=td.get("exit_price_per_share", 0),
                    exit_reason=td.get("exit_reason", "Unknown"),
                    time_in_trade_minutes=td.get("time_in_trade_minutes"),
                )
            trades.append(t)
        return trades

    def _save_report(self, report: Dict[str, Any], output_dir: str) -> None:
        """Save the pipeline report to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "pipeline_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save trades CSV
        if report.get("trades"):
            trades_df = pd.DataFrame(report["trades"])
            trades_df.to_csv(out / "pipeline_trades.csv", index=False)

        logger.info(f"Pipeline report saved to {out}")
