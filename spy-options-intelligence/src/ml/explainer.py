# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Signal explainability module using SHAP values for XGBoost.

Provides ``SignalExplainer`` which wraps a trained XGBoost model and a
``shap.TreeExplainer`` to produce human-readable explanations for buy signals.

Usage
-----
    from src.ml.explainer import SignalExplainer
    import joblib

    artifact = joblib.load("models/xgboost_v2.pkl")
    explainer = SignalExplainer.from_artifact(artifact)

    explanation = explainer.explain_signal(
        features={"hour_et": 10.0, "opt_return_1m": 0.15, ...},
        prediction_proba=0.94,
        threshold=0.90,
    )
    print(explanation)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    import shap as _shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature interpretation helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    """Format a decimal (0.05 = 5%) as a percentage string."""
    return f"{v * 100:+.2f}%"


def _vol_regime_label(v: float) -> str:
    r = round(v)
    return {0: "low", 1: "normal", 2: "high"}.get(r, f"{v:.0f}")


def _rsi_label(v: float) -> str:
    if v >= 70:
        return "overbought"
    elif v <= 30:
        return "oversold"
    return "neutral"


def _moneyness_label(m: float) -> str:
    """Describe moneyness = strike / spot."""
    if m < 0.97:
        return "deep ITM"
    elif m < 0.995:
        return "ITM"
    elif m <= 1.005:
        return "ATM"
    elif m <= 1.03:
        return "OTM"
    return "deep OTM"


# One interpretation function per feature.  Each takes (value, impact) and
# returns a plain string sentence.
_FEATURE_INTERPRETATIONS: Dict[str, Any] = {
    # ── Contract characteristics ─────────────────────────────────────────
    "contract_type": lambda v, i: (
        f"Contract type: {'Call' if round(v) == 1 else 'Put'}"
    ),
    "strike": lambda v, i: (
        f"Strike price: ${v:.2f}"
    ),
    "time_to_expiry_days": lambda v, i: (
        f"Days to expiry: {v:.1f}"
        + (" (0DTE — gamma risk elevated)" if v < 1 else "")
    ),
    "is_0dte": lambda v, i: (
        "0DTE option — same-day expiry, gamma-driven moves"
        if round(v) == 1
        else "Multi-day expiry — theta decay less aggressive"
    ),
    # ── Moneyness ────────────────────────────────────────────────────────
    "moneyness": lambda v, i: (
        f"Moneyness {v:.4f} ({_moneyness_label(v)}) — "
        + ("high delta sensitivity" if abs(v - 1) < 0.03 else "lower delta")
    ),
    "log_moneyness": lambda v, i: (
        f"Log-moneyness {v:+.4f} ({'ITM' if v < 0 else 'OTM'})"
    ),
    # ── Implied volatility ───────────────────────────────────────────────
    "implied_volatility": lambda v, i: (
        f"IV {v:.1%} — "
        + ("elevated, options expensive" if v > 0.40
           else "compressed, options cheap" if v < 0.15
           else "moderate IV")
    ),
    "iv_change_1m": lambda v, i: (
        f"IV change last 1 min: {v:+.4f} — "
        + ("rising, increasing fear/demand" if v > 0 else "falling, calmer market")
    ),
    "iv_change_5m": lambda v, i: (
        f"IV change last 5 min: {v:+.4f} — "
        + ("sustained IV expansion" if v > 0 else "IV contraction")
    ),
    "iv_change_open": lambda v, i: (
        f"IV vs open: {v:+.4f} — "
        + ("IV has risen since open" if v > 0 else "IV has compressed since open")
    ),
    # ── Option price action ──────────────────────────────────────────────
    "opt_return_1m": lambda v, i: (
        f"Option return last 1 min: {_fmt_pct(v)} — "
        + ("sharp upward move" if v > 0.10 else "down" if v < -0.05 else "quiet")
    ),
    "opt_return_5m": lambda v, i: (
        f"Option return last 5 min: {_fmt_pct(v)} — "
        + ("sustained rally" if v > 0.15 else "declining" if v < -0.05 else "flat")
    ),
    "opt_return_15m": lambda v, i: (
        f"Option return last 15 min: {_fmt_pct(v)}"
    ),
    "opt_return_30m": lambda v, i: (
        f"Option return last 30 min: {_fmt_pct(v)}"
    ),
    "opt_return_60m": lambda v, i: (
        f"Option return last 60 min: {_fmt_pct(v)}"
    ),
    "opt_price_change_open": lambda v, i: (
        f"Option vs open price: {_fmt_pct(v)} — "
        + ("strong intraday gain" if v > 0.20 else "below open" if v < -0.10 else "near open")
    ),
    "opt_rsi_14": lambda v, i: (
        f"Option RSI-14: {v:.0f} — {_rsi_label(v)}"
    ),
    "opt_hl_range_5m": lambda v, i: (
        f"Option HL range last 5 min: {_fmt_pct(v)} — "
        + ("volatile" if v > 0.10 else "tight range")
    ),
    "opt_vwap_dist_pct": lambda v, i: (
        f"Option price vs VWAP: {_fmt_pct(v)} — "
        + ("above VWAP (buyers in control)" if v > 0 else "below VWAP (sellers)")
    ),
    "opt_vs_spy_return_1m": lambda v, i: (
        f"Option vs SPY return (1 min): {v:+.3f} — "
        + ("option outpacing SPY (leverage amplifying)" if v > 0 else "option lagging SPY")
    ),
    # ── Option volume ────────────────────────────────────────────────────
    "opt_vol_ratio_5m": lambda v, i: (
        f"Option volume {v:.1f}x its 5-min MA — "
        + ("volume spike, unusual interest" if v > 2 else "normal volume")
    ),
    "opt_vol_pct_cumday": lambda v, i: (
        f"Option volume: {v:.1%} of today's cumulative total"
    ),
    "opt_vol_ma5": lambda v, i: (
        f"Option 5-min volume MA: {v:,.0f} contracts"
    ),
    "opt_bar_count": lambda v, i: (
        f"Option bars today: {v:.0f} — "
        + ("well-established contract" if v > 50 else "early in the session")
    ),
    "transactions_ratio": lambda v, i: (
        f"Option/SPY transaction ratio: {v:.4f} — "
        + ("unusual option activity relative to SPY" if v > 0.01 else "normal ratio")
    ),
    # ── SPY price action ─────────────────────────────────────────────────
    "spy_close": lambda v, i: (
        f"SPY current price: ${v:.2f}"
    ),
    "spy_open": lambda v, i: (
        f"SPY opening price: ${v:.2f}"
    ),
    "spy_high": lambda v, i: (
        f"SPY session high: ${v:.2f}"
    ),
    "spy_low": lambda v, i: (
        f"SPY session low: ${v:.2f}"
    ),
    "spy_return_1m": lambda v, i: (
        f"SPY return last 1 min: {_fmt_pct(v)} — "
        + ("strong upward bar" if v > 0.003 else "down" if v < -0.003 else "flat")
    ),
    "spy_return_5m": lambda v, i: (
        f"SPY return last 5 min: {_fmt_pct(v)} — "
        + ("momentum building" if v > 0.005 else "declining" if v < -0.005 else "range-bound")
    ),
    "spy_return_15m": lambda v, i: (
        f"SPY return last 15 min: {_fmt_pct(v)}"
    ),
    "spy_return_30m": lambda v, i: (
        f"SPY return last 30 min: {_fmt_pct(v)}"
    ),
    "spy_return_60m": lambda v, i: (
        f"SPY return last 60 min: {_fmt_pct(v)}"
    ),
    "spy_hl_range_5m": lambda v, i: (
        f"SPY 5-min HL range: {_fmt_pct(v)} — "
        + ("high intrabar volatility" if v > 0.005 else "tight bars")
    ),
    "spy_hl_range_30m": lambda v, i: (
        f"SPY 30-min HL range: {_fmt_pct(v)} — "
        + ("wide swing session" if v > 0.01 else "low volatility")
    ),
    # ── SPY technicals ───────────────────────────────────────────────────
    "spy_rsi_14": lambda v, i: (
        f"SPY RSI-14: {v:.0f} — {_rsi_label(v)}"
    ),
    "spy_macd": lambda v, i: (
        f"SPY MACD line: {v:+.4f} — "
        + ("above signal, bullish bias" if v > 0 else "below signal, bearish bias")
    ),
    "spy_macd_hist": lambda v, i: (
        f"SPY MACD histogram: {v:+.4f} — "
        + ("expanding bullish momentum" if v > 0 else "bearish divergence")
    ),
    "spy_macd_signal": lambda v, i: (
        f"SPY MACD signal: {v:+.4f}"
    ),
    "spy_ema_9": lambda v, i: (
        f"SPY EMA-9: ${v:.2f}"
    ),
    "spy_ema_21": lambda v, i: (
        f"SPY EMA-21: ${v:.2f}"
    ),
    "spy_ema_diff": lambda v, i: (
        f"SPY EMA9 − EMA21: {v:+.4f} — "
        + ("short MA above long MA, bullish trend" if v > 0 else "short MA below long, bearish trend")
    ),
    "spy_bb_pct_b": lambda v, i: (
        f"SPY Bollinger %B: {v:.2f} — "
        + ("near upper band, overbought" if v > 0.8 else "near lower band, oversold" if v < 0.2 else "mid-band")
    ),
    "spy_bb_upper": lambda v, i: (
        f"SPY Bollinger upper band: ${v:.2f}"
    ),
    "spy_bb_lower": lambda v, i: (
        f"SPY Bollinger lower band: ${v:.2f}"
    ),
    "spy_vwap": lambda v, i: (
        f"SPY VWAP: ${v:.2f}"
    ),
    "spy_vwap_dist_pct": lambda v, i: (
        f"SPY price vs VWAP: {_fmt_pct(v)} — "
        + ("trading above VWAP, buyers dominant" if v > 0 else "below VWAP, selling pressure")
    ),
    # ── SPY volume ───────────────────────────────────────────────────────
    "spy_volume": lambda v, i: (
        f"SPY bar volume: {v:,.0f} shares"
    ),
    "spy_vol_ratio_5m": lambda v, i: (
        f"SPY volume {v:.1f}x its 5-min MA — "
        + ("volume surge" if v > 2 else "below-average volume" if v < 0.5 else "average")
    ),
    "spy_vol_ratio_30m": lambda v, i: (
        f"SPY volume {v:.1f}x its 30-min MA"
    ),
    "spy_vol_zscore": lambda v, i: (
        f"SPY volume z-score: {v:+.2f} — "
        + ("statistically abnormal volume" if abs(v) > 2 else "normal distribution")
    ),
    "spy_vol_regime": lambda v, i: (
        f"SPY volume regime: {_vol_regime_label(v)} — "
        + ("institutional-level participation" if round(v) == 2 else "retail/normal flow")
    ),
    "spy_vol_ma5": lambda v, i: (
        f"SPY 5-min volume MA: {v:,.0f} shares"
    ),
    "spy_vol_ma30": lambda v, i: (
        f"SPY 30-min volume MA: {v:,.0f} shares"
    ),
    "spy_vol_std_5m": lambda v, i: (
        f"SPY 5-min volume std: {v:,.0f}"
    ),
    "spy_vol_std_30m": lambda v, i: (
        f"SPY 30-min volume std: {v:,.0f}"
    ),
    "spy_transactions": lambda v, i: (
        f"SPY transactions (last bar): {v:,.0f}"
    ),
    "spy_bar_count": lambda v, i: (
        f"SPY bars today: {v:.0f}"
    ),
    # ── Time features ────────────────────────────────────────────────────
    "hour_et": lambda v, i: (
        f"Current hour (ET): {int(v):02d}:xx — "
        + (
            "prime spike window (10-11 AM)" if 10 <= v < 12
            else "opening volatility (9-10 AM)" if v < 10
            else "power hour (3-4 PM)" if v >= 15
            else "midday lull"
        )
    ),
    "minute_et": lambda v, i: (
        f"Current minute: :{int(v):02d}"
    ),
    "minute_of_day": lambda v, i: (
        f"Minute of day: {v:.0f} (of 390 trading minutes)"
    ),
    "minutes_since_open": lambda v, i: (
        f"Minutes since 9:30 AM open: {v:.0f}"
        + (" (opening hour)" if v < 60 else " (afternoon)" if v > 240 else "")
    ),
    "is_morning": lambda v, i: (
        "Morning session (<11:30 AM) — higher volatility window"
        if round(v) == 1
        else "Afternoon session — typically lower volatility"
    ),
    "is_last_hour": lambda v, i: (
        "Last trading hour (3-4 PM) — increased volume and directional moves"
        if round(v) == 1
        else "Not the last hour"
    ),
    "pct_day_elapsed": lambda v, i: (
        f"Day elapsed: {v:.1%} — "
        + ("early session" if v < 0.25 else "late session" if v > 0.75 else "midday")
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SignalExplainer:
    """Generate human-readable explanations for XGBoost buy signals.

    Uses ``shap.TreeExplainer`` to compute per-feature SHAP values, then maps
    each feature to a plain-English sentence describing its contribution.

    Parameters
    ----------
    model:
        A fitted XGBoost ``Booster`` or sklearn-compatible classifier.
    feature_names:
        Ordered list of feature column names (must match model training order).
    """

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "shap is required for signal explainability. "
                "Install it with: pip install shap"
            )
        self.model = model
        self.feature_names = list(feature_names)
        self.shap_explainer = _shap.TreeExplainer(model)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "SignalExplainer":
        """Construct from a joblib model artifact dict.

        Parameters
        ----------
        artifact:
            Dict with at least ``"model"`` and ``"feature_cols"`` keys,
            as saved by ``XGBoostTrainer``.
        """
        return cls(artifact["model"], artifact["feature_cols"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_signal(
        self,
        features: Dict[str, float],
        prediction_proba: float,
        threshold: float = 0.90,
    ) -> str:
        """Generate a detailed explanation for a buy signal.

        Parameters
        ----------
        features:
            Mapping of feature name → value for the bar being explained.
            Missing features default to 0.0.
        prediction_proba:
            Model output probability (0–1).
        threshold:
            Decision threshold used to fire the signal.

        Returns
        -------
        str
            Multi-line console-ready explanation with SHAP impacts,
            human-readable interpretations, and risk factor callouts.
        """
        # Build feature array in model's expected column order
        feature_array = np.array(
            [features.get(name, 0.0) for name in self.feature_names],
            dtype=np.float32,
        )

        # SHAP values: shape (n_features,) for a single sample
        shap_values = self.shap_explainer.shap_values([feature_array])[0]

        # Build sorted impact list
        impacts: List[Dict] = []
        for name, value, shap_val in zip(
            self.feature_names,
            feature_array.tolist(),
            shap_values.tolist() if hasattr(shap_values, "tolist") else list(shap_values),
        ):
            impacts.append(
                {
                    "feature": name,
                    "value": float(value),
                    "impact": float(shap_val),
                    "interpretation": self._interpret_feature(name, float(value), float(shap_val)),
                }
            )

        # Sort by absolute SHAP value descending — biggest drivers first
        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return self._format_explanation(
            prediction_proba=prediction_proba,
            threshold=threshold,
            top_impacts=impacts[:10],
            features=features,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpret_feature(self, name: str, value: float, impact: float) -> str:
        """Convert a feature name + value into a plain-English sentence.

        Parameters
        ----------
        name:
            Feature column name.
        value:
            Numeric feature value at the bar.
        impact:
            SHAP value (positive = pushes toward buy signal).

        Returns
        -------
        str
            Human-readable description of what this feature indicates.
        """
        fn = _FEATURE_INTERPRETATIONS.get(name)
        if fn is not None:
            try:
                return fn(value, impact)
            except Exception:
                pass  # fall through to generic
        # Generic fallback for unknown features
        return f"{name} = {value:.4g}"

    def _format_explanation(
        self,
        prediction_proba: float,
        threshold: float,
        top_impacts: List[Dict],
        features: Dict[str, float],
    ) -> str:
        """Render the explanation as a formatted console string.

        Parameters
        ----------
        prediction_proba:
            Model confidence (0–1).
        threshold:
            Decision threshold for the signal.
        top_impacts:
            List of top-10 feature impact dicts (feature, value, impact,
            interpretation).
        features:
            Original feature dict (used for contextual header information).

        Returns
        -------
        str
            Multi-line explanation string ready for ``print()`` or file output.
        """
        margin = prediction_proba - threshold
        is_marginal = abs(margin) < 0.02

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append("=" * 70)
        title = "🎯 SIGNAL DETECTED"
        if is_marginal:
            title += " (CAUTION: Near Threshold)"
        lines.append(title)
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Confidence:  {prediction_proba:.1%}")
        lines.append(f"Threshold:   {threshold:.1%}")

        margin_str = f"{margin:+.1%} above threshold"
        if is_marginal:
            margin_str += "  ⚠️  CLOSE CALL"
        lines.append(f"Margin:      {margin_str}")

        # ── Contributing factors ──────────────────────────────────────────
        lines.append("")
        lines.append("─" * 70)
        lines.append("📊 TOP CONTRIBUTING FACTORS")
        lines.append("─" * 70)
        lines.append("")

        for i, item in enumerate(top_impacts, 1):
            # Strength indicator: 1–5 dots scaled by absolute impact
            n_dots = min(5, max(1, int(abs(item["impact"]) * 20)))
            indicator = "🔴" * n_dots
            direction = "BULLISH" if item["impact"] > 0 else "BEARISH"

            lines.append(
                f"{i:2}. {indicator} {item['feature']}  "
                f"(Impact: {item['impact']:+.3f})"
            )
            lines.append(f"    └─ {item['interpretation']}")
            lines.append(f"    └─ Signal: {direction}")
            lines.append("")

        # ── Risk factors ──────────────────────────────────────────────────
        risk_factors = [x for x in top_impacts if x["impact"] < 0]
        if risk_factors:
            lines.append("─" * 70)
            lines.append("⚠️  RISK FACTORS (Caution)")
            lines.append("─" * 70)
            lines.append("")
            for item in risk_factors:
                lines.append(f"  • {item['feature']}: {item['interpretation']}")
            lines.append("")
            if len(risk_factors) >= 2:
                lines.append(
                    "💡 RECOMMENDATION: Proceed with caution or skip this signal"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
