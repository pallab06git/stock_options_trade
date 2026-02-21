# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Multi-model comparison framework.

Evaluates multiple trained models at multiple confidence thresholds using
the existing TradeSimulator, then produces side-by-side comparison reports.

Usage
-----
    from src.ml.model_comparator import ModelComparator
    from src.ml.trade_simulator import TradeSimulator

    comparator = ModelComparator(position_size_usd=12_500)

    comparator.add_model("xgboost",  xgb_model,  feature_cols=feat_cols, best_params=xgb_best)
    comparator.add_model("lightgbm", lgbm_model, feature_cols=feat_cols, best_params=lgbm_best)

    for name in comparator.model_names:
        comparator.evaluate_at_thresholds(name, test_df)

    print(comparator.generate_comparison_report())
    comparator.save_results("reports/model_comparison")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.ml.trade_simulator import TradeSimulator, _extract_contract_type
from src.ml.train_xgboost import _NON_FEATURE_COLS

logger = logging.getLogger(__name__)

# Default thresholds for multi-threshold sweep
DEFAULT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# ModelComparator
# ---------------------------------------------------------------------------


class ModelComparator:
    """Compare multiple models at multiple confidence thresholds.

    Parameters
    ----------
    position_size_usd:
        USD per trade for TradeSimulator (default $12,500).
    target_gain_pct:
        Take-profit percentage (default 30.0%).
    stop_loss_pct:
        Stop-loss percentage (default -12.0%).
    fee_per_trade_usd:
        Round-trip commission (default $4.00).
    monthly_profit_target:
        Net profit goal per month in USD used for "Meets Target" column
        (default $10,000).
    """

    def __init__(
        self,
        position_size_usd: float = 12_500.0,
        target_gain_pct: float = 30.0,
        stop_loss_pct: float = -12.0,
        fee_per_trade_usd: float = 4.0,
        monthly_profit_target: float = 10_000.0,
    ) -> None:
        self.position_size_usd = float(position_size_usd)
        self.target_gain_pct = float(target_gain_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.fee_per_trade_usd = float(fee_per_trade_usd)
        self.monthly_profit_target = float(monthly_profit_target)

        # {model_name: {'model': ..., 'feature_cols': [...], 'best_params': {...}, ...}}
        self._models: Dict[str, Dict[str, Any]] = {}

        # {model_name: {threshold: result_dict}}
        self._results: Dict[str, Dict[float, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    @property
    def model_names(self) -> List[str]:
        return list(self._models.keys())

    def add_model(
        self,
        name: str,
        model: Any,
        feature_cols: Optional[List[str]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        optimization_score: float = 0.0,
        model_type: str = "sklearn",
    ) -> None:
        """Register a trained model for comparison.

        Parameters
        ----------
        name:
            Short identifier (e.g. ``"xgboost"``).
        model:
            Trained model with ``predict_proba(X) → ndarray`` interface.
            For LSTM wrap in ``LSTMPredictor``.
        feature_cols:
            Ordered list of feature column names used during training.
            If ``None``, derived from the test DataFrame at evaluation time
            by excluding ``_NON_FEATURE_COLS``.
        best_params:
            Best hyperparameters (informational only).
        optimization_score:
            Score from hyperparameter optimization (informational).
        model_type:
            One of ``"sklearn"`` (default), ``"lgbm"``, ``"xgboost"``,
            or ``"lstm"``.
        """
        self._models[name] = {
            "model": model,
            "feature_cols": feature_cols,
            "best_params": best_params or {},
            "optimization_score": float(optimization_score),
            "model_type": model_type,
        }
        self._results[name] = {}
        logger.info("Registered model '%s' (type=%s)", name, model_type)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_at_thresholds(
        self,
        model_name: str,
        test_df: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[float, Dict[str, Any]]:
        """Evaluate a registered model at multiple confidence thresholds.

        Runs ``TradeSimulator.simulate_period()`` for each threshold and
        collects detailed win/loss statistics.

        Parameters
        ----------
        model_name:
            Key used in ``add_model()``.
        test_df:
            Feature DataFrame.  Must contain the feature columns used to
            train the model, plus the simulation columns (``close``,
            ``ticker``, ``max_gain_120m``, ``min_loss_120m``, etc.).
        thresholds:
            Confidence thresholds to sweep (default ``DEFAULT_THRESHOLDS``).

        Returns
        -------
        Dict mapping threshold → result dict.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not registered. Call add_model() first.")

        thresholds = thresholds or DEFAULT_THRESHOLDS
        entry = self._models[model_name]
        model = entry["model"]

        # Resolve feature columns
        feat_cols = entry["feature_cols"]
        if feat_cols is None:
            feat_cols = sorted(c for c in test_df.columns if c not in _NON_FEATURE_COLS)
            entry["feature_cols"] = feat_cols  # cache for later

        # Fill missing feature columns with 0
        for col in feat_cols:
            if col not in test_df.columns:
                test_df = test_df.copy()
                test_df[col] = 0.0

        X_test = test_df[feat_cols].fillna(0.0).values.astype(np.float32)
        y_proba = model.predict_proba(X_test)[:, 1]

        sim = TradeSimulator(
            position_size_usd=self.position_size_usd,
            target_gain_pct=self.target_gain_pct,
            stop_loss_pct=self.stop_loss_pct,
            fee_per_trade_usd=self.fee_per_trade_usd,
        )

        results: Dict[float, Dict[str, Any]] = {}

        for threshold in thresholds:
            trades = sim.simulate_period(test_df, y_proba, threshold, model, feat_cols)

            winners = [t for t in trades if t.is_winner]
            losers = [t for t in trades if not t.is_winner]
            calls = [t for t in trades if _extract_contract_type(t.contract_symbol) == "C"]
            puts = [t for t in trades if _extract_contract_type(t.contract_symbol) == "P"]

            gross_profit = sum(t.profit_loss_usd for t in winners if t.profit_loss_usd)
            gross_loss = sum(t.profit_loss_usd for t in losers if t.profit_loss_usd)
            monthly_profit = gross_profit + gross_loss
            fees = len(trades) * self.fee_per_trade_usd
            net_profit = monthly_profit - fees

            pnl_vals = [t.profit_loss_usd for t in trades if t.profit_loss_usd is not None]

            results[threshold] = {
                "threshold": threshold,
                "total_signals": len(trades),
                # Call / Put split
                "calls": len(calls),
                "puts": len(puts),
                "call_pct": len(calls) / len(trades) * 100 if trades else 0.0,
                "put_pct": len(puts) / len(trades) * 100 if trades else 0.0,
                # Win metrics
                "wins": len(winners),
                "win_rate": len(winners) / len(trades) if trades else 0.0,
                "win_profit_min": float(min(t.profit_loss_usd for t in winners)) if winners else 0.0,
                "win_profit_median": float(np.median([t.profit_loss_usd for t in winners])) if winners else 0.0,
                "win_profit_mean": float(np.mean([t.profit_loss_usd for t in winners])) if winners else 0.0,
                "win_profit_max": float(max(t.profit_loss_usd for t in winners)) if winners else 0.0,
                # Loss metrics
                "losses": len(losers),
                "loss_amount_min": float(min(t.profit_loss_usd for t in losers)) if losers else 0.0,
                "loss_amount_median": float(np.median([t.profit_loss_usd for t in losers])) if losers else 0.0,
                "loss_amount_mean": float(np.mean([t.profit_loss_usd for t in losers])) if losers else 0.0,
                "loss_amount_max": float(max(t.profit_loss_usd for t in losers)) if losers else 0.0,
                # Aggregate P&L
                "gross_profit_usd": round(gross_profit, 2),
                "gross_loss_usd": round(gross_loss, 2),
                "monthly_profit_usd": round(monthly_profit, 2),
                "total_fees_usd": round(fees, 2),
                "net_profit_usd": round(net_profit, 2),
                "avg_pnl_per_trade_usd": round(float(np.mean(pnl_vals)), 2) if pnl_vals else 0.0,
                # Trade list (serialised)
                "trades": [t.to_dict() for t in trades],
            }

        self._results[model_name] = results
        logger.info(
            "Evaluated '%s' at %d thresholds — best net: $%,.0f",
            model_name,
            len(thresholds),
            max((r["net_profit_usd"] for r in results.values()), default=0.0),
        )
        return results

    # ------------------------------------------------------------------
    # Signal overlap
    # ------------------------------------------------------------------

    def find_signal_overlap(
        self,
        model_results: Optional[Dict[str, Dict[float, Dict[str, Any]]]] = None,
        threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """Find signals where multiple models fire at the same bar.

        A "signal" is identified by ``(contract_symbol, entry_time)``.
        Two models "agree" when both produce a trade with the same key.

        Parameters
        ----------
        model_results:
            ``{model_name: {threshold: result_dict}}``.
            Defaults to the internally stored ``self._results``.
        threshold:
            Which threshold level to compare (default 0.80).

        Returns
        -------
        Dict with overlap statistics and breakdown.
        """
        model_results = model_results or self._results
        model_signals: Dict[str, Set[Tuple[str, str]]] = {}

        for model_name, thresh_results in model_results.items():
            if threshold in thresh_results:
                trades = thresh_results[threshold].get("trades", [])
                model_signals[model_name] = {
                    (t["contract_symbol"], t["entry_time"]) for t in trades
                }

        if not model_signals:
            return {
                "total_unique_signals": 0,
                "all_models_agree": 0,
                "majority_agree": 0,
                "overlap_breakdown": {},
                "detailed_overlaps": {},
            }

        # Collect all unique signals across models
        all_signals: Set[Tuple[str, str]] = set()
        for sigs in model_signals.values():
            all_signals.update(sigs)

        n_models = len(model_signals)
        overlap_by_count: Dict[int, List[Dict[str, Any]]] = {}

        for signal_key in all_signals:
            agreeing = [
                mname
                for mname, sigs in model_signals.items()
                if signal_key in sigs
            ]
            n_agree = len(agreeing)
            if n_agree not in overlap_by_count:
                overlap_by_count[n_agree] = []
            overlap_by_count[n_agree].append(
                {
                    "contract_symbol": signal_key[0],
                    "entry_time": signal_key[1],
                    "models": agreeing,
                }
            )

        majority_threshold = n_models // 2 + 1
        majority_agree = sum(
            len(v) for k, v in overlap_by_count.items() if k >= majority_threshold
        )

        return {
            "threshold": threshold,
            "n_models_compared": n_models,
            "model_names": list(model_signals.keys()),
            "total_unique_signals": len(all_signals),
            "all_models_agree": len(overlap_by_count.get(n_models, [])),
            "majority_agree": majority_agree,
            "overlap_breakdown": {
                f"{k}_models": len(v) for k, v in sorted(overlap_by_count.items())
            },
            "detailed_overlaps": {
                str(k): v for k, v in sorted(overlap_by_count.items())
            },
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_comparison_report(
        self, comparison_threshold: float = 0.80
    ) -> pd.DataFrame:
        """Generate a side-by-side comparison table at a fixed threshold.

        Parameters
        ----------
        comparison_threshold:
            Threshold used for the comparison columns (default 0.80).

        Returns
        -------
        DataFrame with one row per model.
        """
        rows = []
        for model_name, entry in self._models.items():
            thresh_results = self._results.get(model_name, {})
            r = thresh_results.get(comparison_threshold, {})

            net = r.get("net_profit_usd", 0.0)
            rows.append(
                {
                    "Model": model_name,
                    "Opt Score": round(entry["optimization_score"], 4),
                    f"Signals ({comparison_threshold:.0%})": r.get("total_signals", 0),
                    "Win Rate": f"{r.get('win_rate', 0.0):.1%}",
                    "Calls %": f"{r.get('call_pct', 0.0):.1f}%",
                    "Puts %": f"{r.get('put_pct', 0.0):.1f}%",
                    "Avg Win $": f"${r.get('win_profit_mean', 0.0):,.0f}",
                    "Avg Loss $": f"${r.get('loss_amount_mean', 0.0):,.0f}",
                    "Net Profit": f"${net:+,.0f}",
                    "Meets Target": (
                        "YES" if net >= self.monthly_profit_target else "NO"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def get_best_threshold_per_model(self) -> Dict[str, Dict[str, Any]]:
        """Return the threshold that maximises net_profit_usd for each model.

        Returns
        -------
        Dict mapping model name → ``{threshold, net_profit_usd, ...}``.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for model_name, thresh_results in self._results.items():
            if not thresh_results:
                continue
            best_t = max(thresh_results, key=lambda t: thresh_results[t].get("net_profit_usd", 0.0))
            r = thresh_results[best_t]
            best[model_name] = {
                "best_threshold": best_t,
                "net_profit_usd": r.get("net_profit_usd", 0.0),
                "total_signals": r.get("total_signals", 0),
                "win_rate": r.get("win_rate", 0.0),
            }
        return best

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str | Path) -> None:
        """Save all results to ``{output_dir}/{model_name}_results.json``.

        Also writes ``model_comparison.csv`` (the comparison table) and
        ``overlap_analysis.json`` at each default threshold.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Per-model JSON
        for model_name, thresh_results in self._results.items():
            path = out / f"{model_name}_results.json"
            # Convert float keys to strings for JSON
            serialisable = {str(k): v for k, v in thresh_results.items()}
            with open(path, "w") as fh:
                json.dump(serialisable, fh, indent=2, default=str)
            logger.info("Saved %s results → %s", model_name, path)

        # Comparison CSV
        df = self.generate_comparison_report()
        df.to_csv(out / "model_comparison.csv", index=False)
        logger.info("Saved model_comparison.csv")

        # Overlap analysis at each threshold
        for t in DEFAULT_THRESHOLDS:
            overlap = self.find_signal_overlap(threshold=t)
            with open(out / f"overlap_{t:.2f}.json", "w") as fh:
                json.dump(overlap, fh, indent=2, default=str)
