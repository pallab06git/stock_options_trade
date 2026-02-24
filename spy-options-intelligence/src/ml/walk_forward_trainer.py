# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Walk-forward retraining with expanding window for realistic OOS evaluation.

Instead of a single train/test split, the walk-forward trainer:
1. Starts with a minimum training window (e.g., 6 months)
2. Trains ensemble + exit model on that window
3. Simulates real-bar trades on the NEXT month (true out-of-sample)
4. Expands training window by one month and repeats

This avoids the regime-shift problem where a static model trained on
Mar-Oct 2025 fails on Dec 2025-Feb 2026 because market conditions changed.

Usage
-----
    from src.ml.walk_forward_trainer import WalkForwardTrainer

    trainer = WalkForwardTrainer(config)
    result = trainer.run(
        features_dir="data/processed/features",
        raw_options_dir="data/raw/options/minute",
        min_train_months=6,
    )

Architecture
------------
  For each fold (expanding window):
    1. TRAIN: months [0, N)
    2. VAL:   month N (last month of training window, for meta-learner)
    3. TEST:  month N+1 (true out-of-sample, never seen during training)
    4. Retrain StackedEnsemble on TRAIN+VAL
    5. Retrain ExitSignalModel on TRAIN+VAL signals
    6. Simulate real-bar trades on TEST month
    7. Collect trades (these are genuine forward-walk results)

  Model-agreement filter:
    At prediction time, require >= min_agreement (default 2/3) base models
    to score above 0.5 for the signal to pass.  This catches regime-degraded
    periods where models disagree (agreement dropped from 70-89% to 10-22%).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.ml.trade_simulator import Trade
from src.utils.logger import get_logger

logger = get_logger()


def _get_monthly_boundaries(dates: List[str]) -> List[Tuple[str, str]]:
    """Group sorted dates into (first_date, last_date) per calendar month.

    Args:
        dates: Sorted list of date strings (YYYY-MM-DD).

    Returns:
        List of (first_date, last_date) tuples per month.
    """
    if not dates:
        return []

    months: Dict[str, List[str]] = {}
    for d in sorted(dates):
        key = d[:7]  # "YYYY-MM"
        months.setdefault(key, []).append(d)

    return [
        (month_dates[0], month_dates[-1])
        for month_dates in months.values()
    ]


class WalkForwardTrainer:
    """Walk-forward retraining with expanding window.

    Parameters
    ----------
    config : dict
        Full config dict.  Reads from config["signal_pipeline"],
        config["stacked_ensemble"], config["exit_model"].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        pipeline_cfg = self._config.get("signal_pipeline", {})
        self.position_size = pipeline_cfg.get("position_size_usd", 10_000)
        self.max_signals_per_day = pipeline_cfg.get("max_signals_per_day", 3)
        self.entry_threshold = pipeline_cfg.get("entry_threshold", 0.70)
        self.min_agreement = pipeline_cfg.get("min_model_agreement", 2)

    def run(
        self,
        features_dir: str | Path = "data/processed/features",
        raw_options_dir: str | Path = "data/raw/options/minute",
        min_train_months: int = 6,
        output_dir: Optional[str | Path] = None,
        entry_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward validation across all available data.

        Args:
            features_dir: Directory with feature CSVs.
            raw_options_dir: Directory with raw option Parquet files.
            min_train_months: Minimum number of months for first training window.
            output_dir: Directory to save results.
            entry_threshold: Override entry threshold (default from config).

        Returns:
            Summary dict with per-fold metrics and aggregated out-of-sample results.
        """
        from src.ml.data_splitter import DataSplitter
        from src.ml.exit_signal_model import ExitSignalModel
        from src.ml.real_bar_simulator import RealBarSimulator
        from src.ml.stacked_ensemble import StackedEnsemble
        from src.ml.train_xgboost import _NON_FEATURE_COLS, load_features

        if entry_threshold is not None:
            self.entry_threshold = entry_threshold

        feat_dir = Path(features_dir)
        raw_dir = Path(raw_options_dir)

        # Discover all available dates
        feat_files = sorted(feat_dir.glob("*_features.csv"))
        all_dates = [f.stem.replace("_features", "") for f in feat_files]
        if not all_dates:
            raise ValueError(f"No feature files found in {feat_dir}")

        # Group into monthly boundaries
        monthly = _get_monthly_boundaries(all_dates)
        n_months = len(monthly)
        logger.info(f"Walk-forward: {n_months} months available, "
                     f"min_train={min_train_months}")

        if n_months < min_train_months + 1:
            raise ValueError(
                f"Need at least {min_train_months + 1} months for walk-forward "
                f"(have {n_months})"
            )

        # Walk-forward loop
        all_trades: List[Trade] = []
        fold_results: List[Dict[str, Any]] = []

        for fold_idx in range(min_train_months, n_months):
            # TRAIN: months [0, fold_idx - 1)
            # VAL:   month [fold_idx - 1]
            # TEST:  month [fold_idx]
            train_start = monthly[0][0]
            train_end = monthly[fold_idx - 2][1]  # end of train (excl. val)
            val_start = monthly[fold_idx - 1][0]
            val_end = monthly[fold_idx - 1][1]
            test_start = monthly[fold_idx][0]
            test_end = monthly[fold_idx][1]
            test_month = test_start[:7]

            logger.info(
                f"\n{'='*60}\n"
                f"Fold {fold_idx - min_train_months + 1}: "
                f"TRAIN {train_start}→{train_end}, "
                f"VAL {val_start}→{val_end}, "
                f"TEST {test_start}→{test_end}\n"
                f"{'='*60}"
            )

            # Load training and validation data
            train_df = load_features(feat_dir, train_start, train_end)
            val_df = load_features(feat_dir, val_start, val_end)
            test_df = load_features(feat_dir, test_start, test_end)

            if train_df.empty or val_df.empty:
                logger.warning(f"Fold {fold_idx}: empty train/val, skipping")
                continue

            # Determine feature columns (intersection of train + val columns
            # so we don't crash when some dates lack microstructure features)
            common_cols = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
            feature_cols = sorted(
                [c for c in common_cols if c not in _NON_FEATURE_COLS]
            )

            # --- Train ensemble ---
            ensemble = StackedEnsemble(self._config)
            try:
                metrics = ensemble.train(train_df, val_df, feature_cols)
            except Exception as exc:
                logger.warning(f"Fold {fold_idx}: ensemble training failed: {exc}")
                continue

            logger.info(
                f"Ensemble AUC={metrics['val_roc_auc']:.4f}, "
                f"precision={metrics['val_precision']:.4f}"
            )

            # --- Train exit model ---
            # Generate exit training data from train+val signals
            exit_model = ExitSignalModel(self._config)
            try:
                exit_train_df = self._generate_exit_data(
                    ensemble, feature_cols, feat_dir,
                    train_start, val_end, raw_dir,
                )
                if not exit_train_df.empty:
                    exit_metrics = exit_model.train(training_data=exit_train_df)
                    logger.info(f"Exit model AUC={exit_metrics.get('val_auc', 0):.4f}")
                else:
                    logger.warning("No exit training data, using rule-based exit only")
            except Exception as exc:
                logger.warning(f"Exit model training failed: {exc}")

            # --- Simulate test month ---
            simulator = RealBarSimulator(self._config)
            simulator.entry_threshold = self.entry_threshold
            fold_trades = self._simulate_test_month(
                ensemble=ensemble,
                exit_model=exit_model,
                feature_cols=feature_cols,
                feat_dir=feat_dir,
                raw_dir=raw_dir,
                test_start=test_start,
                test_end=test_end,
            )

            # Collect results
            fold_pnl = sum(t.profit_loss_usd or 0 for t in fold_trades)
            fold_wins = sum(1 for t in fold_trades if (t.profit_loss_usd or 0) > 0)
            fold_total = len(fold_trades)
            fold_win_rate = fold_wins / fold_total * 100 if fold_total > 0 else 0

            fold_result = {
                "fold": fold_idx - min_train_months + 1,
                "test_month": test_month,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train_rows": len(train_df),
                "n_val_rows": len(val_df),
                "n_test_rows": len(test_df),
                "n_trades": fold_total,
                "n_wins": fold_wins,
                "win_rate": round(fold_win_rate, 1),
                "net_pnl": round(fold_pnl, 2),
                "avg_pnl": round(fold_pnl / fold_total, 2) if fold_total > 0 else 0,
                "ensemble_auc": round(metrics["val_roc_auc"], 4),
            }
            fold_results.append(fold_result)
            all_trades.extend(fold_trades)

            logger.info(
                f"Fold {fold_result['fold']} ({test_month}): "
                f"{fold_total} trades, {fold_win_rate:.0f}% win, "
                f"P&L=${fold_pnl:+,.0f}"
            )

        # --- Aggregate results ---
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if (t.profit_loss_usd or 0) > 0)
        total_pnl = sum(t.profit_loss_usd or 0 for t in all_trades)

        result = {
            "method": "walk_forward_expanding_window",
            "min_train_months": min_train_months,
            "entry_threshold": self.entry_threshold,
            "min_agreement": self.min_agreement,
            "n_folds": len(fold_results),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            "trades_per_day": round(
                total_trades / max(sum(fr["n_trades"] > 0 or 1 for fr in fold_results), 1), 1
            ) if fold_results else 0,
            "folds": fold_results,
        }

        # Monthly summary (entry_time format: "YYYY-MM-DD HH:MM ET")
        monthly_pnl = {}
        for t in all_trades:
            et = getattr(t, "entry_time", "") or ""
            month = et[:7] if len(et) >= 7 else "unknown"
            monthly_pnl.setdefault(month, 0)
            monthly_pnl[month] += t.profit_loss_usd or 0
        result["monthly_pnl"] = {k: round(v, 2) for k, v in sorted(monthly_pnl.items())}

        # Save if output_dir specified
        if output_dir:
            self._save_results(result, all_trades, output_dir)

        return result

    def _simulate_test_month(
        self,
        ensemble,
        exit_model,
        feature_cols: List[str],
        feat_dir: Path,
        raw_dir: Path,
        test_start: str,
        test_end: str,
    ) -> List[Trade]:
        """Simulate trades for a test month using trained models.

        Applies model-agreement filter: requires >= min_agreement base models
        to score above 0.5 for a signal to pass.

        Adds regime labels to feature data so the simulator can use
        regime-adaptive exit (low-vol → model exit, high-vol → trailing stop).
        """
        from src.ml.real_bar_simulator import RealBarSimulator

        simulator = RealBarSimulator(self._config)
        simulator.entry_threshold = self.entry_threshold

        # Extract regime detector from ensemble (fitted during training)
        regime_detector = getattr(ensemble, '_regime_detector', None)

        feat_files = sorted(feat_dir.glob("*_features.csv"))
        all_trades: List[Trade] = []

        for feat_file in feat_files:
            date = feat_file.stem.replace("_features", "")
            if date < test_start or date > test_end:
                continue

            # Only dates with raw option bars
            if not any(raw_dir.glob(f"*/{date}.parquet")):
                continue

            try:
                df = pd.read_csv(feat_file)
            except Exception:
                continue

            if df.empty:
                continue

            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < len(feature_cols) * 0.5:
                continue

            # Add regime labels for regime-adaptive exit
            if regime_detector is not None and regime_detector._is_fitted:
                try:
                    df = regime_detector.predict(df)
                except Exception:
                    df["market_regime"] = -1

            # Score with ensemble (simple average)
            proba = ensemble.predict_proba(df, available_cols)

            # Model-agreement filter: require >= min_agreement base models > 0.5
            if self.min_agreement > 1 and hasattr(ensemble, '_base_models'):
                agreement_mask = self._compute_agreement_mask(
                    ensemble, df, available_cols
                )
                # Zero out probabilities where agreement is insufficient
                proba = np.where(agreement_mask, proba, 0.0)

            day_trades = simulator.simulate_day(
                date=date,
                features_df=df,
                feature_cols=available_cols,
                entry_proba=proba,
                exit_model=exit_model,
                raw_options_dir=str(raw_dir),
            )
            all_trades.extend(day_trades)

        return all_trades

    def _compute_agreement_mask(
        self,
        ensemble,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Compute boolean mask: True where >= min_agreement base models score > 0.5.

        This catches regime-degraded periods where models disagree.
        """
        fcols = ensemble._feature_cols
        aligned = df.reindex(columns=fcols, fill_value=0.0)
        X = aligned.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        votes = np.zeros(len(X), dtype=int)
        for name, model in ensemble._base_models.items():
            model_proba = model.predict_proba(X)[:, 1]
            votes += (model_proba > 0.5).astype(int)

        return votes >= self.min_agreement

    def _generate_exit_data(
        self,
        ensemble,
        feature_cols: List[str],
        feat_dir: Path,
        start_date: str,
        end_date: str,
        raw_dir: Path,
    ) -> pd.DataFrame:
        """Generate exit training data from ensemble signals over date range."""
        from src.ml.exit_signal_model import ExitSignalModel

        exit_model = ExitSignalModel(self._config)

        feat_files = sorted(feat_dir.glob("*_features.csv"))
        raw_dates = set()
        for ticker_dir in raw_dir.iterdir():
            if ticker_dir.is_dir():
                for p in ticker_dir.glob("*.parquet"):
                    raw_dates.add(p.stem)

        all_signals = []
        for feat_file in feat_files:
            date = feat_file.stem.replace("_features", "")
            if date < start_date or date > end_date:
                continue
            if date not in raw_dates:
                continue

            try:
                df = pd.read_csv(feat_file)
            except Exception:
                continue

            if df.empty:
                continue

            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < len(feature_cols) * 0.8:
                continue

            proba = ensemble.predict_proba(df, available_cols)

            # Take top-3 signals per day
            top_indices = np.argsort(proba)[::-1][:3]
            for idx in top_indices:
                if proba[idx] < 0.5:
                    continue
                row = df.iloc[idx]
                all_signals.append({
                    "date": date,
                    "ticker": str(row.get("ticker", "UNKNOWN")),
                    "timestamp": int(row.get("timestamp", 0)),
                    "entry_price": float(row.get("close", row.get("opt_close", 0))),
                    "contract_type": int(row.get("contract_type", 1)),
                })

        if not all_signals:
            return pd.DataFrame()

        signals_df = pd.DataFrame(all_signals)
        return exit_model.generate_training_data(
            feature_df=pd.DataFrame(),
            entry_signals=signals_df,
            raw_options_dir=str(raw_dir),
        )

    def _save_results(
        self,
        result: Dict[str, Any],
        trades: List[Trade],
        output_dir: str | Path,
    ) -> None:
        """Save walk-forward results to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary_path = out / "walk_forward_results.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")

        # Trade log CSV
        if trades:
            trade_rows = []
            for t in trades:
                et = getattr(t, "entry_time", "") or ""
                trade_rows.append({
                    "date": et[:10] if len(et) >= 10 else "",
                    "ticker": t.contract_symbol,
                    "entry_price": t.entry_price_per_share,
                    "exit_price": t.exit_price_per_share,
                    "pnl_pct": t.profit_loss_pct,
                    "pnl_usd": t.profit_loss_usd,
                    "exit_reason": t.exit_reason,
                    "time_in_trade": t.time_in_trade_minutes,
                    "confidence": t.confidence,
                    "regime": getattr(t, "regime", -1),
                })
            trades_df = pd.DataFrame(trade_rows)
            trades_path = out / "walk_forward_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades: {trades_path}")
