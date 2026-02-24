# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Real-bar trade simulator using entry + exit models.

Replaces label-based simulation (max_gain_120m) with bar-by-bar
simulation on actual minute data, using the exit model to determine
when to close each trade.

Usage
-----
    from src.ml.real_bar_simulator import RealBarSimulator

    simulator = RealBarSimulator(config)
    result = simulator.simulate_period(
        entry_model_path="models/stacked_ensemble.pkl",
        exit_model_path="models/exit_model.pkl",
        features_dir="data/processed/features",
        raw_options_dir="data/raw/options/minute",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.ml.exit_signal_model import EXIT_FEATURE_COLS, ExitFeatureEngineer, ExitSignalModel
from src.ml.trade_simulator import Trade
from src.utils.logger import get_logger

logger = get_logger()


class RealBarSimulator:
    """Bar-by-bar trade simulator with model-driven exit decisions.

    Parameters
    ----------
    config : dict
        Full configuration dict.  Reads from config["signal_pipeline"].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("signal_pipeline", {})
        self.position_size = cfg.get("position_size_usd", 10_000)
        self.max_signals_per_day = cfg.get("max_signals_per_day", 3)
        self.entry_threshold = cfg.get("entry_threshold", 0.70)
        self.fee = cfg.get("fee_per_trade", 4.0)
        self.eod_close_minutes = cfg.get("eod_close_minutes_before", 10)
        self._exit_feature_eng = ExitFeatureEngineer()

    def simulate_day(
        self,
        date: str,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        entry_proba: np.ndarray,
        exit_model: ExitSignalModel,
        raw_options_dir: str | Path,
    ) -> List[Trade]:
        """Simulate trades for a single trading day.

        Steps:
          1. Rank bars by entry probability, take top-N candidates
          2. Process candidates chronologically (one at a time)
          3. For each candidate: load raw option bars, walk forward
          4. At each bar: compute exit features, call exit model
          5. When exit fires → create Trade with actual entry/exit prices
          6. Only proceed to next signal after current trade closes

        Args:
            date: Trading date (YYYY-MM-DD).
            features_df: Feature CSV data for this date.
            feature_cols: Feature columns used by entry model.
            entry_proba: Entry probabilities for each row.
            exit_model: Trained ExitSignalModel.
            raw_options_dir: Directory with raw option Parquet files.

        Returns:
            List of completed Trade objects.
        """
        raw_dir = Path(raw_options_dir)
        trades: List[Trade] = []

        if len(features_df) == 0:
            return trades

        # Find candidate signals above threshold
        signal_mask = entry_proba >= self.entry_threshold
        if not signal_mask.any():
            return trades

        signal_indices = np.where(signal_mask)[0]
        signal_probas = entry_proba[signal_indices]

        # Rank by probability, take top candidates
        rank_order = np.argsort(signal_probas)[::-1]
        candidates = []
        for rank_idx in rank_order:
            idx = signal_indices[rank_idx]
            row = features_df.iloc[idx]
            ts = int(row.get("timestamp", 0))

            # Directional alignment filter: skip signals misaligned with SPY
            contract_type = int(row.get("contract_type", 1))
            spy_ret = float(row.get("spy_return_5m", 0))
            if contract_type == 1 and spy_ret < -0.01:  # CALL but SPY falling
                continue
            if contract_type == 0 and spy_ret > 0.01:   # PUT but SPY rising
                continue

            candidates.append({
                "idx": idx,
                "timestamp": ts,
                "proba": float(signal_probas[rank_idx]),
                "ticker": str(row.get("ticker", "UNKNOWN")),
                "entry_price": float(row.get("close", row.get("opt_close", 0))),
                "contract_type": contract_type,
                "hour_et": int(row.get("hour_et", 9)),
                "minute_et": int(row.get("minute_et", 30)),
                "spy_close": float(row.get("spy_close", 0)),
                "minutes_since_open": float(row.get("minutes_since_open", 0)),
            })

        # Sort chronologically
        candidates.sort(key=lambda c: c["timestamp"])

        # Process candidates sequentially
        current_exit_time = 0
        trade_id = 1

        for cand in candidates:
            if len(trades) >= self.max_signals_per_day:
                break

            # Don't enter if previous trade hasn't closed yet
            if cand["timestamp"] < current_exit_time:
                continue

            # Don't enter in last 30 minutes
            if cand["minutes_since_open"] > 360:  # 360 min = 3:30 PM
                continue

            entry_price = cand["entry_price"]
            if entry_price <= 0:
                continue

            # Load raw option bars (layout: {ticker_dir}/{date}.parquet)
            safe_ticker = cand["ticker"].replace(":", "_")
            opt_file = raw_dir / safe_ticker / f"{date}.parquet"
            if not opt_file.exists():
                continue

            try:
                bars = pd.read_parquet(opt_file).sort_values("timestamp")
            except Exception:
                continue

            if bars.empty:
                continue

            # Find entry bar in raw data
            entry_ts = cand["timestamp"]
            entry_bar_idx = int(bars["timestamp"].searchsorted(entry_ts))
            if entry_bar_idx >= len(bars):
                entry_bar_idx = len(bars) - 1

            # Walk forward bar by bar
            trade = self._simulate_trade(
                bars=bars,
                entry_bar_idx=entry_bar_idx,
                entry_price=entry_price,
                contract_type=cand["contract_type"],
                confidence=cand["proba"],
                ticker=cand["ticker"],
                date=date,
                hour_et=cand["hour_et"],
                minute_et=cand["minute_et"],
                spy_close=cand["spy_close"],
                exit_model=exit_model,
                trade_id=trade_id,
            )

            if trade is not None:
                trades.append(trade)
                trade_id += 1
                # Update exit time to prevent overlapping trades
                if trade.exit_time:
                    current_exit_time = entry_ts + int(
                        (trade.time_in_trade_minutes or 0) * 60000
                    )

        return trades

    def simulate_period(
        self,
        entry_model_path: str | Path = "",
        exit_model_path: str | Path = "",
        features_dir: str | Path = "",
        raw_options_dir: str | Path = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simulate trades across a date range.

        Args:
            entry_model_path: Path to entry model artifact.
            exit_model_path: Path to exit model artifact.
            features_dir: Directory with feature CSVs.
            raw_options_dir: Directory with raw option Parquet files.
            start_date: Start date filter (YYYY-MM-DD).
            end_date: End date filter (YYYY-MM-DD).
            output_dir: Optional output directory for results.

        Returns:
            Summary dict with trades, P&L, and statistics.
        """
        # Load entry model
        entry_artifact = joblib.load(entry_model_path)
        if "base_models" in entry_artifact:
            from src.ml.stacked_ensemble import StackedEnsemble
            entry_model = StackedEnsemble()
            entry_model.load(entry_model_path)
            feature_cols = entry_artifact["feature_cols"]
            is_ensemble = True
        else:
            entry_model = entry_artifact["model"]
            feature_cols = entry_artifact["feature_cols"]
            is_ensemble = False

        # Load exit model
        exit_model = ExitSignalModel()
        exit_model.load(exit_model_path)

        # Find feature files
        feat_dir = Path(features_dir)
        raw_dir = Path(raw_options_dir)
        feat_files = sorted(feat_dir.glob("*_features.csv"))

        all_trades: List[Trade] = []
        dates_processed = 0

        for feat_file in feat_files:
            date = feat_file.stem.replace("_features", "")

            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            # Only process dates that have at least one raw option parquet file
            # Layout: {raw_dir}/{ticker_dir}/{date}.parquet
            if not any(raw_dir.glob(f"*/{date}.parquet")):
                continue

            try:
                df = pd.read_csv(feat_file)
            except Exception:
                continue

            if df.empty:
                continue

            # Score with entry model
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < len(feature_cols) * 0.5:
                continue

            X = df[available_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if is_ensemble:
                proba = entry_model.predict_proba(df, available_cols)
            else:
                proba = entry_model.predict_proba(X)[:, 1]

            day_trades = self.simulate_day(
                date=date,
                features_df=df,
                feature_cols=available_cols,
                entry_proba=proba,
                exit_model=exit_model,
                raw_options_dir=raw_options_dir,
            )

            all_trades.extend(day_trades)
            dates_processed += 1

            if day_trades:
                day_pnl = sum(t.profit_loss_usd or 0 for t in day_trades)
                logger.info(f"{date}: {len(day_trades)} trades, P&L=${day_pnl:+,.0f}")

        # Generate summary
        result = self._generate_summary(all_trades, dates_processed, start_date, end_date)

        # Save if output_dir specified
        if output_dir:
            self._save_results(result, all_trades, output_dir)

        return result

    def _simulate_trade(
        self,
        bars: pd.DataFrame,
        entry_bar_idx: int,
        entry_price: float,
        contract_type: int,
        confidence: float,
        ticker: str,
        date: str,
        hour_et: int,
        minute_et: int,
        spy_close: float,
        exit_model: ExitSignalModel,
        trade_id: int,
    ) -> Optional[Trade]:
        """Simulate a single trade by walking forward through bars."""
        entry_time = f"{date} {hour_et:02d}:{minute_et:02d} ET"

        trade = Trade(
            trade_id=trade_id,
            entry_time=entry_time,
            contract_symbol=ticker,
            entry_price_per_share=entry_price,
            position_size_usd=self.position_size,
            confidence=confidence,
        )

        if trade.num_contracts < 1:
            return None

        # Entry bar's minutes since 9:30 AM open
        entry_minutes_since_open = (hour_et - 9) * 60 + minute_et - 30

        # Walk forward bar by bar
        max_bars = min(exit_model.max_hold_minutes, len(bars) - entry_bar_idx)

        for offset in range(1, max_bars):
            bar_idx = entry_bar_idx + offset
            if bar_idx >= len(bars):
                break

            current_price = float(bars.iloc[bar_idx]["close"])
            unrealized_pnl = (current_price - entry_price) / entry_price * 100
            time_in_trade = float(offset)

            # Minutes to 4:00 PM ET close (market = 9:30→16:00 = 390 min)
            minutes_to_close = max(0, 390 - (entry_minutes_since_open + offset))

            # Compute exit features
            exit_feats = self._exit_feature_eng.compute_exit_features(
                bars_df=bars,
                entry_idx=entry_bar_idx,
                entry_price=entry_price,
                contract_type=contract_type,
                entry_spy_close=spy_close,
            )

            if offset < len(exit_feats):
                feat_row = exit_feats.iloc[[offset]]
            else:
                feat_row = np.zeros((1, 15), dtype=np.float32)

            should_exit, prob, reason = exit_model.should_exit(
                feat_row, unrealized_pnl, time_in_trade, minutes_to_close
            )

            if should_exit:
                exit_hour = hour_et + int((minute_et + offset) // 60)
                exit_min = int((minute_et + offset) % 60)
                exit_time = f"{date} {exit_hour:02d}:{exit_min:02d} ET"

                trade.close_trade(
                    exit_time=exit_time,
                    exit_price_per_share=round(current_price, 4),
                    exit_reason=reason,
                    time_in_trade_minutes=time_in_trade,
                )
                return trade

        # Time limit: force close at last available bar
        last_bar_idx = min(entry_bar_idx + max_bars - 1, len(bars) - 1)
        last_price = float(bars.iloc[last_bar_idx]["close"])
        exit_minutes = last_bar_idx - entry_bar_idx
        exit_hour = hour_et + int((minute_et + exit_minutes) // 60)
        exit_min = int((minute_et + exit_minutes) % 60)
        exit_time = f"{date} {exit_hour:02d}:{exit_min:02d} ET"

        trade.close_trade(
            exit_time=exit_time,
            exit_price_per_share=round(last_price, 4),
            exit_reason="Time limit",
            time_in_trade_minutes=float(exit_minutes),
        )
        return trade

    def _generate_summary(
        self,
        trades: List[Trade],
        n_days: int,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Dict[str, Any]:
        """Generate summary statistics from completed trades."""
        if not trades:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "n_days": n_days,
                "total_trades": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_trades_per_day": 0.0,
                "trades": [],
            }

        winners = [t for t in trades if t.is_winner]
        total_pnl = sum(t.profit_loss_usd or 0 for t in trades)
        total_fees = len(trades) * self.fee

        return {
            "start_date": start_date,
            "end_date": end_date,
            "n_days": n_days,
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(trades) - len(winners),
            "win_rate": len(winners) / len(trades),
            "net_pnl": round(total_pnl - total_fees, 2),
            "gross_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
            "avg_trades_per_day": round(len(trades) / max(n_days, 1), 1),
            "avg_hold_minutes": round(
                np.mean([t.time_in_trade_minutes or 0 for t in trades]), 1
            ),
            "trades": [t.to_dict() for t in trades],
            "exit_reasons": self._count_exit_reasons(trades),
        }

    def _count_exit_reasons(self, trades: List[Trade]) -> Dict[str, int]:
        """Count exit reasons across all trades."""
        reasons: Dict[str, int] = {}
        for t in trades:
            r = t.exit_reason or "Unknown"
            reasons[r] = reasons.get(r, 0) + 1
        return reasons

    def _save_results(
        self,
        result: Dict[str, Any],
        trades: List[Trade],
        output_dir: str,
    ) -> None:
        """Save simulation results to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        with open(out / "simulation_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save trades CSV
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])
            trades_df.to_csv(out / "trades.csv", index=False)

        logger.info(f"Simulation results saved to {out}")
