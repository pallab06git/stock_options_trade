# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Exit signal model for detecting optimal trade exit points.

Architecture
------------
  ExitFeatureEngineer: computes 15 features per bar after entry:
    - Trade-relative (5): unrealized P&L, time held, drawdown, peak P&L, bars since peak
    - Momentum exhaustion (5): short-term return, deceleration, volume decline,
      RSI reversal, MACD cross
    - Market context (5): SPY return since entry, SPY vol change, regime at entry,
      regime now, regime changed

  ExitSignalModel: LightGBM classifier that predicts whether to exit at each bar.
    - Hard rules override model: stop-loss, time limit, EOD close
    - Model captures momentum exhaustion for softer exits

Usage
-----
    from src.ml.exit_signal_model import ExitSignalModel

    exit_model = ExitSignalModel(config)
    metrics = exit_model.train(entry_model_path, features_dir, raw_options_dir)
    exit_model.save("models/exit_model.pkl")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger()

_LGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    pass


class ExitFeatureEngineer:
    """Compute exit-decision features for an active trade.

    All features are computed relative to the entry bar and the
    trade's running P&L, so they capture the trade's lifecycle.
    """

    def compute_exit_features(
        self,
        bars_df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        contract_type: int,
        entry_spy_close: float = 0.0,
        entry_regime: int = 0,
    ) -> pd.DataFrame:
        """Compute exit features for all bars after entry.

        Args:
            bars_df: Raw option minute bars (with spy_ columns merged).
            entry_idx: Index of the entry bar in bars_df.
            entry_price: Option close price at entry.
            contract_type: 1=call, 0=put.
            entry_spy_close: SPY close at entry time.
            entry_regime: Market regime at entry (0–3).

        Returns:
            DataFrame with 15 exit feature columns for bars after entry.
        """
        post_entry = bars_df.iloc[entry_idx:].copy()
        n = len(post_entry)

        if n == 0 or entry_price <= 0:
            return pd.DataFrame()

        close = post_entry["close"].values.astype(float)

        # --- Trade-relative features (5) ---
        unrealized_pnl = (close - entry_price) / entry_price * 100
        post_entry["unrealized_pnl_pct"] = unrealized_pnl

        post_entry["time_in_trade_min"] = np.arange(n, dtype=float)

        peak_pnl = np.maximum.accumulate(unrealized_pnl)
        post_entry["peak_pnl_pct"] = peak_pnl

        drawdown = unrealized_pnl - peak_pnl
        post_entry["drawdown_from_peak_pct"] = drawdown

        # Bars since peak P&L
        bars_since = np.zeros(n, dtype=float)
        peak_idx = 0
        for i in range(n):
            if unrealized_pnl[i] >= peak_pnl[i] - 1e-8:
                peak_idx = i
            bars_since[i] = i - peak_idx
        post_entry["bars_since_peak"] = bars_since

        # --- Momentum exhaustion features (5) ---
        close_series = post_entry["close"]
        ret_1m = close_series.pct_change(1) * 100
        ret_3m = close_series.pct_change(3) * 100
        ret_5m = close_series.pct_change(5) * 100

        post_entry["opt_return_3m_post_entry"] = ret_3m.fillna(0)

        post_entry["momentum_deceleration"] = (ret_1m - ret_5m).fillna(0)

        vol = post_entry.get("volume", pd.Series(0, index=post_entry.index))
        vol_ma5 = vol.rolling(5, min_periods=1).mean().replace(0, np.nan)
        post_entry["volume_decline_ratio"] = (vol / vol_ma5).fillna(1.0)

        # RSI reversal: 1 if RSI was >70 and has crossed below
        rsi_col = post_entry.get("opt_rsi_14", pd.Series(50, index=post_entry.index))
        rsi_vals = rsi_col.fillna(50).values
        was_above_70 = np.zeros(n, dtype=float)
        seen_70 = False
        for i in range(n):
            if rsi_vals[i] > 70:
                seen_70 = True
            was_above_70[i] = 1.0 if (seen_70 and rsi_vals[i] < 70) else 0.0
        post_entry["rsi_reversal"] = was_above_70

        # MACD histogram turning negative
        macd_hist = post_entry.get(
            "spy_macd_hist", pd.Series(0, index=post_entry.index)
        ).fillna(0)
        macd_prev = macd_hist.shift(1).fillna(0)
        post_entry["macd_cross"] = (
            (macd_prev > 0) & (macd_hist <= 0)
        ).astype(float)

        # --- Market context features (5) ---
        spy_close = post_entry.get(
            "spy_close", pd.Series(entry_spy_close, index=post_entry.index)
        ).fillna(entry_spy_close)
        if entry_spy_close > 0:
            post_entry["spy_return_since_entry"] = (
                (spy_close - entry_spy_close) / entry_spy_close * 100
            )
        else:
            post_entry["spy_return_since_entry"] = 0.0

        spy_vol = post_entry.get(
            "spy_vol_std_5m", pd.Series(0, index=post_entry.index)
        ).fillna(0)
        entry_vol = float(spy_vol.iloc[0]) if len(spy_vol) > 0 else 0.0
        safe_vol = max(entry_vol, 1e-8)
        post_entry["spy_vol_change"] = (spy_vol - entry_vol) / safe_vol * 100

        post_entry["regime_at_entry"] = float(entry_regime)

        regime_now = post_entry.get(
            "market_regime", pd.Series(entry_regime, index=post_entry.index)
        ).fillna(entry_regime)
        post_entry["regime_now"] = regime_now.astype(float)
        post_entry["regime_changed"] = (regime_now != entry_regime).astype(float)

        exit_feature_cols = [
            "unrealized_pnl_pct", "time_in_trade_min", "drawdown_from_peak_pct",
            "peak_pnl_pct", "bars_since_peak",
            "opt_return_3m_post_entry", "momentum_deceleration",
            "volume_decline_ratio", "rsi_reversal", "macd_cross",
            "spy_return_since_entry", "spy_vol_change",
            "regime_at_entry", "regime_now", "regime_changed",
        ]

        return post_entry[exit_feature_cols].reset_index(drop=True)


# Exit feature column names (used for training and prediction)
EXIT_FEATURE_COLS = [
    "unrealized_pnl_pct", "time_in_trade_min", "drawdown_from_peak_pct",
    "peak_pnl_pct", "bars_since_peak",
    "opt_return_3m_post_entry", "momentum_deceleration",
    "volume_decline_ratio", "rsi_reversal", "macd_cross",
    "spy_return_since_entry", "spy_vol_change",
    "regime_at_entry", "regime_now", "regime_changed",
]


class ExitSignalModel:
    """LightGBM exit signal classifier.

    Parameters
    ----------
    config : dict
        Full configuration dict.  Reads from config["exit_model"].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("exit_model", {})
        self.exit_threshold = cfg.get("exit_threshold", 0.6)
        self.max_hold_minutes = cfg.get("max_hold_minutes", 120)
        self.min_hold_minutes = cfg.get("min_hold_minutes", 5)
        self.stop_loss_pct = cfg.get("stop_loss_pct", -20.0)
        self.reversal_pct = cfg.get("reversal_pct", 15.0)
        self.random_state = cfg.get("random_state", 42)

        # Adaptive early-loss exit: lower model threshold when trade is losing early.
        # Tier 1: loss >= pct_t1 AND time <= time_t1 min → model threshold drops to thresh_t1
        # Tier 2: loss >= pct_t2 AND time <= time_t2 min → model threshold drops to thresh_t2
        self.early_loss_pct_t1    = cfg.get("early_loss_pct_tier1",   -3.0)
        self.early_loss_time_t1   = cfg.get("early_loss_time_tier1",   10.0)
        self.early_loss_thresh_t1 = cfg.get("early_loss_threshold_tier1", 0.25)
        self.early_loss_pct_t2    = cfg.get("early_loss_pct_tier2",   -6.0)
        self.early_loss_time_t2   = cfg.get("early_loss_time_tier2",   20.0)
        self.early_loss_thresh_t2 = cfg.get("early_loss_threshold_tier2", 0.15)
        # Early-loss detection can start earlier than standard min_hold_minutes.
        # Hard-stop trades typically hit -12% in 4–6 min, before the 5-min gate opens.
        self.min_hold_early_loss  = cfg.get("min_hold_early_loss", 2.0)

        self._model = None
        self._feature_engineer = ExitFeatureEngineer()
        self._is_trained = False

    def generate_training_data(
        self,
        feature_df: pd.DataFrame,
        entry_signals: pd.DataFrame,
        raw_options_dir: str | Path,
    ) -> pd.DataFrame:
        """Generate exit training data from historical entry signals.

        For each entry signal, walk forward through the option's bars.
        Label the optimal exit bar: the bar just before a reversal of
        reversal_pct from peak P&L.

        Args:
            feature_df: Feature CSV data with entry signal rows.
            entry_signals: DataFrame with columns: date, ticker, timestamp,
                          entry_price, contract_type.
            raw_options_dir: Directory containing raw option Parquet files.

        Returns:
            DataFrame with exit features + exit_label column.
        """
        raw_dir = Path(raw_options_dir)
        all_rows = []

        for _, signal in entry_signals.iterrows():
            date = str(signal["date"])
            ticker = str(signal["ticker"])
            entry_price = float(signal["entry_price"])
            contract_type = int(signal.get("contract_type", 1))
            entry_ts = int(signal.get("timestamp", 0))

            # Find raw option file (layout: {ticker_dir}/{date}.parquet)
            safe_ticker = ticker.replace(":", "_")
            opt_file = raw_dir / safe_ticker / f"{date}.parquet"
            if not opt_file.exists():
                continue

            try:
                bars = pd.read_parquet(opt_file).sort_values("timestamp")
            except Exception:
                continue

            if bars.empty:
                continue

            # Find entry bar
            if entry_ts > 0:
                entry_idx = bars["timestamp"].searchsorted(entry_ts)
                entry_idx = min(entry_idx, len(bars) - 1)
            else:
                entry_idx = 0

            # Compute exit features
            exit_feats = self._feature_engineer.compute_exit_features(
                bars_df=bars,
                entry_idx=entry_idx,
                entry_price=entry_price,
                contract_type=contract_type,
            )

            if exit_feats.empty:
                continue

            # Label: find optimal exit point
            pnl = exit_feats["unrealized_pnl_pct"].values
            peak = exit_feats["peak_pnl_pct"].values
            labels = np.zeros(len(exit_feats), dtype=int)

            # Exit label = 1 at the bar just before reversal_pct drawdown from peak
            for i in range(1, len(pnl)):
                if peak[i] > 0 and (peak[i] - pnl[i]) >= self.reversal_pct:
                    # Mark the bar before as optimal exit
                    labels[max(0, i - 1)] = 1
                    break

            # If no reversal detected, mark last bar as exit (time limit)
            if labels.sum() == 0 and len(labels) > 1:
                labels[-1] = 1

            exit_feats["exit_label"] = labels
            exit_feats["date"] = date
            exit_feats["ticker"] = ticker
            all_rows.append(exit_feats)

        if not all_rows:
            return pd.DataFrame()

        return pd.concat(all_rows, ignore_index=True)

    def train(
        self,
        entry_model_path: str | Path = "",
        features_dir: str | Path = "",
        raw_options_dir: str | Path = "",
        training_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Train the exit signal model.

        Can be called with either:
        1. Pre-built training_data DataFrame (for testing)
        2. Paths to generate training data from historical signals

        Args:
            entry_model_path: Path to entry model artifact (for signal generation).
            features_dir: Path to feature CSVs.
            raw_options_dir: Path to raw option Parquet files.
            training_data: Pre-built exit training DataFrame (overrides paths).

        Returns:
            Training metrics dict.
        """
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is required for ExitSignalModel")

        if training_data is not None:
            exit_df = training_data
        else:
            exit_df = self._generate_from_paths(
                entry_model_path, features_dir, raw_options_dir
            )

        if exit_df.empty:
            raise ValueError("No exit training data generated")

        # Prepare features
        feat_cols = [c for c in EXIT_FEATURE_COLS if c in exit_df.columns]
        X = exit_df[feat_cols].values.astype(np.float32)
        y = exit_df["exit_label"].values.astype(int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Chronological split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Class balance
        n_neg = max((y_train == 0).sum(), 1)
        n_pos = max((y_train == 1).sum(), 1)
        scale_pw = n_neg / n_pos

        # Train LightGBM
        self._model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pw,
            random_state=self.random_state,
            verbosity=-1,
        )

        if len(X_val) > 0 and len(np.unique(y_val)) >= 2:
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            self._model.fit(X_train, y_train)

        self._is_trained = True

        # Compute metrics
        metrics = {
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "exit_label_rate": float(y.mean()),
            "feature_cols": feat_cols,
        }

        if len(X_val) > 0 and len(np.unique(y_val)) >= 2:
            val_proba = self._model.predict_proba(X_val)[:, 1]
            val_preds = (val_proba >= 0.5).astype(int)
            metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
            metrics["val_precision"] = float(precision_score(y_val, val_preds, zero_division=0))
            metrics["val_recall"] = float(recall_score(y_val, val_preds, zero_division=0))
        else:
            metrics["val_auc"] = 0.0
            metrics["val_precision"] = 0.0
            metrics["val_recall"] = 0.0

        logger.info(
            f"Exit model trained: {len(X_train)} train, {len(X_val)} val, "
            f"AUC={metrics['val_auc']:.4f}"
        )

        return metrics

    def should_exit(
        self,
        exit_features: pd.DataFrame | np.ndarray,
        unrealized_pnl: float,
        time_in_trade: float,
        minutes_to_close: float = 999.0,
    ) -> Tuple[bool, float, str]:
        """Decide whether to exit the current trade.

        Hard rules (override model):
          - unrealized_pnl <= stop_loss_pct    → "Hard stop"  (absolute backstop)
          - time_in_trade >= max_hold_minutes   → "Time limit"
          - minutes_to_close <= 10              → "EOD close"

        Adaptive early-loss exit (model-driven, lower threshold when losing early):
          - Tier 1: loss >= -3% AND time <= 10 min → exit if P(exit) >= 0.25
          - Tier 2: loss >= -6% AND time <= 20 min → exit if P(exit) >= 0.15

        Standard model rule:
          - P(exit) >= exit_threshold           → "Momentum exhaustion"

        Args:
            exit_features: Exit feature vector or 1-row DataFrame.
            unrealized_pnl: Current unrealized P&L percentage.
            time_in_trade: Minutes since entry.
            minutes_to_close: Minutes until market close (4:00 PM ET).

        Returns:
            Tuple of (should_exit, probability, reason).
        """
        # 1. Hard stop (absolute backstop — model-independent)
        if unrealized_pnl <= self.stop_loss_pct:
            return True, 1.0, "Hard stop"

        # 2. Time-based hard rules
        if time_in_trade >= self.max_hold_minutes:
            return True, 1.0, "Time limit"

        if minutes_to_close <= 10:
            return True, 1.0, "EOD close"

        # 3. Absolute minimum gate (no exits in first 2 minutes regardless)
        if time_in_trade < self.min_hold_early_loss:
            return False, 0.0, "Min hold"

        # 4. Get model probability once (used for all remaining checks)
        if self._model is None or not self._is_trained:
            return False, 0.0, "No model"

        if isinstance(exit_features, pd.DataFrame):
            X = exit_features[EXIT_FEATURE_COLS].values.astype(np.float32)
        else:
            X = exit_features.reshape(1, -1).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self._model.predict_proba(X)[:, 1]
        p = float(proba[0]) if len(proba) > 0 else 0.0

        # 5. Adaptive early-loss exit: trust model sooner when trade is losing.
        # Fires from min_hold_early_loss (2 min) so it can catch fast-falling trades
        # that would otherwise slide past the standard 5-min min_hold into a hard stop.
        if (unrealized_pnl <= self.early_loss_pct_t1
                and time_in_trade <= self.early_loss_time_t1
                and p >= self.early_loss_thresh_t1):
            return True, p, "Early loss exit"

        if (unrealized_pnl <= self.early_loss_pct_t2
                and time_in_trade <= self.early_loss_time_t2
                and p >= self.early_loss_thresh_t2):
            return True, p, "Early loss exit"

        # 6. Standard min hold gate for momentum exhaustion (longer wait)
        if time_in_trade < self.min_hold_minutes:
            return False, p, "Min hold"

        # 7. Standard momentum exhaustion
        if p >= self.exit_threshold:
            return True, p, "Momentum exhaustion"

        return False, p, "Hold"

    def save(self, path: str | Path) -> None:
        """Save exit model artifact."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained exit model")
        artifact = {
            "model": self._model,
            "exit_threshold": self.exit_threshold,
            "max_hold_minutes": self.max_hold_minutes,
            "min_hold_minutes": self.min_hold_minutes,
            "stop_loss_pct": self.stop_loss_pct,
            "feature_cols": EXIT_FEATURE_COLS,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        logger.info(f"Exit model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load exit model artifact."""
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self.exit_threshold = artifact.get("exit_threshold", 0.6)
        self.max_hold_minutes = artifact.get("max_hold_minutes", 120)
        self.min_hold_minutes = artifact.get("min_hold_minutes", 5)
        self.stop_loss_pct = artifact.get("stop_loss_pct", -20.0)
        self._is_trained = True
        logger.info(f"Exit model loaded from {path}")

    def _generate_from_paths(
        self,
        entry_model_path: str | Path,
        features_dir: str | Path,
        raw_options_dir: str | Path,
    ) -> pd.DataFrame:
        """Generate exit training data from entry model + feature files."""
        feat_dir = Path(features_dir)
        raw_dir = Path(raw_options_dir)

        # Load entry model to generate signals
        artifact = joblib.load(entry_model_path)

        # Determine if it's a stacked ensemble or XGBoost
        if "base_models" in artifact:
            from src.ml.stacked_ensemble import StackedEnsemble
            entry_model = StackedEnsemble()
            entry_model.load(entry_model_path)
            feature_cols = artifact["feature_cols"]
            is_ensemble = True
        else:
            entry_model = artifact["model"]
            feature_cols = artifact["feature_cols"]
            is_ensemble = False

        # Find dates with both features and raw options
        # Layout: {raw_dir}/{ticker_dir}/{date}.parquet → collect dates from parquet filenames
        feat_files = sorted(feat_dir.glob("*_features.csv"))
        raw_dates = set()
        for ticker_dir in raw_dir.iterdir():
            if ticker_dir.is_dir():
                for p in ticker_dir.glob("*.parquet"):
                    raw_dates.add(p.stem)  # stem = "2025-03-03"

        all_signals = []
        for feat_file in feat_files:
            date = feat_file.stem.replace("_features", "")
            if date not in raw_dates:
                continue

            try:
                df = pd.read_csv(feat_file)
            except Exception:
                continue

            if df.empty:
                continue

            # Score with entry model
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < len(feature_cols) * 0.8:
                continue

            X = df[available_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if is_ensemble:
                proba = entry_model.predict_proba(df, available_cols)
            else:
                proba = entry_model.predict_proba(X)[:, 1]

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
        return self.generate_training_data(
            feature_df=pd.DataFrame(),
            entry_signals=signals_df,
            raw_options_dir=raw_options_dir,
        )
