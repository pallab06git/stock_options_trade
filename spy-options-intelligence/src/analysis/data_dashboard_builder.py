# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Build interactive HTML dashboard showing raw data → features → outcomes.

Pipeline layers visualised
--------------------------
1. Raw SPY minute bars        (data/raw/spy/{date}.parquet)
2. Raw option minute bars     (data/raw/options/minute/{ticker}/{date}.parquet)
3. Processed features         (data/processed/features/{date}_features.csv)
4. Model predictions          (optional – supply a model artifact path)
5. Actual outcomes            (target, max_gain_120m, min_loss_120m columns)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Feature columns used in the heatmap (row 4)
# Names match the ACTUAL column names in the feature CSVs.
# ---------------------------------------------------------------------------

_HEATMAP_FEATURES: List[str] = [
    "spy_return_1m",
    "spy_return_5m",
    "spy_return_15m",
    "opt_return_1m",
    "opt_return_5m",
    "opt_return_15m",
    "spy_rsi_14",
    "spy_macd",
    "spy_macd_hist",
    "spy_vol_ratio_5m",
    "spy_vol_zscore",
    "implied_volatility",
    "iv_change_1m",
    "moneyness",
    "opt_vs_spy_return_1m",
    "spy_bb_pct_b",
    "spy_vwap_dist_pct",
    "opt_vwap_dist_pct",
    "pct_day_elapsed",
    "opt_hl_range_5m",
]

# Colours for call / put contract traces
_CALL_COLORS = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728"]
_PUT_COLORS = ["#ff7f0e", "#ffbb78", "#9467bd", "#c5b0d5", "#8c564b"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _contract_type_from_ticker(ticker: str) -> int:
    """Return 1 (CALL) or 0 (PUT) by parsing the ticker string."""
    m = re.search(r"\d([CP])\d", ticker)
    if m:
        return 1 if m.group(1) == "C" else 0
    return -1


# ---------------------------------------------------------------------------
# DataDashboardBuilder
# ---------------------------------------------------------------------------


class DataDashboardBuilder:
    """Create an interactive Plotly HTML dashboard of the full data pipeline.

    Usage
    -----
    ::

        builder = DataDashboardBuilder(model_path="models/xgboost_v3_clean.pkl")
        data = builder.extract_sample_day("2025-06-25", n_contracts=5)
        builder.build_dashboard(data, "reports/dashboard/2025-06-25.html")
        builder.show_feature_derivation(data["features"], minute_idx=5)
    """

    def __init__(
        self,
        spy_raw_dir: str = "data/raw/spy",
        options_raw_dir: str = "data/raw/options/minute",
        features_dir: str = "data/processed/features",
        model_path: Optional[str] = None,
    ):
        self.spy_raw_dir = Path(spy_raw_dir)
        self.options_raw_dir = Path(options_raw_dir)
        self.features_dir = Path(features_dir)
        self._model = None
        self._feature_cols: List[str] = []

        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        import pickle

        try:
            with open(model_path, "rb") as fh:
                artifact = pickle.load(fh)
        except Exception:
            try:
                import joblib
                artifact = joblib.load(model_path)
            except Exception as exc:
                print(f"  WARNING: could not load model ({exc})")
                return

        self._model = artifact.get("model")
        self._feature_cols = artifact.get("feature_cols", [])
        print(f"  Model loaded: {len(self._feature_cols)} features")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def extract_sample_day(
        self,
        date_str: str,
        n_contracts: int = 5,
    ) -> Dict:
        """Extract all pipeline data for a specific trading day.

        Args:
            date_str: Date to visualise (``YYYY-MM-DD``, e.g. ``'2025-06-25'``).
            n_contracts: Number of option contracts to include (default 5).

        Returns:
            Dict with keys:
              ``date``, ``spy_raw``, ``options_raw``, ``features``,
              ``predictions``, ``outcomes``.
        """
        print(f"\nExtracting data for {date_str}...")

        features = self._load_features(date_str)
        spy_raw = self._load_spy_raw(date_str, features)
        options_raw = self._load_options_raw(date_str, n_contracts, features)
        predictions = self._get_predictions(features)
        outcomes = self._get_outcomes(features)

        return {
            "date": date_str,
            "spy_raw": spy_raw,
            "options_raw": options_raw,
            "features": features,
            "predictions": predictions,
            "outcomes": outcomes,
        }

    def _load_features(self, date_str: str) -> pd.DataFrame:
        """Load processed feature CSV for the given date."""
        path = self.features_dir / f"{date_str}_features.csv"
        if not path.exists():
            print(f"  Features: not found at {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True
            ).dt.tz_convert("America/New_York")
        print(f"  Features: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _load_spy_raw(
        self, date_str: str, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Load raw SPY minute bars from Parquet.

        Falls back to reconstructing from features if the Parquet file is absent.
        """
        parquet = self.spy_raw_dir / f"{date_str}.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True
            ).dt.tz_convert("America/New_York")
            df = df.sort_values("timestamp").reset_index(drop=True)
            print(f"  SPY raw: {len(df)} bars (from Parquet)")
            return df

        # Fall back: reconstruct from features
        if not features_df.empty:
            spy_cols = [
                "timestamp", "spy_open", "spy_high", "spy_low",
                "spy_close", "spy_volume",
            ]
            avail = [c for c in spy_cols if c in features_df.columns]
            spy_df = (
                features_df[avail]
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            spy_df = spy_df.rename(columns={
                "spy_open": "open",
                "spy_high": "high",
                "spy_low": "low",
                "spy_close": "close",
                "spy_volume": "volume",
            })
            print(f"  SPY raw: {len(spy_df)} bars (reconstructed from features)")
            return spy_df

        print("  SPY raw: no data found")
        return pd.DataFrame()

    def _load_options_raw(
        self,
        date_str: str,
        n_contracts: int,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Load raw option minute bars for a sample of contracts.

        Selection strategy: half CALLs, half PUTs (rounded up).
        Falls back to option OHLCV columns from the feature CSV.
        """
        # Pick representative tickers from features
        sample_tickers: List[str] = []
        if not features_df.empty and "ticker" in features_df.columns:
            calls = features_df[features_df["contract_type"] == 1]["ticker"].unique()
            puts = features_df[features_df["contract_type"] == 0]["ticker"].unique()
            half = max(n_contracts // 2, 1)
            sample_tickers = list(calls[:half]) + list(puts[:half])
            sample_tickers = sample_tickers[:n_contracts]

        dfs: List[pd.DataFrame] = []
        for ticker in sample_tickers:
            # Raw options path: data/raw/options/minute/O_SPY.../{date}.parquet
            dir_name = ticker.replace("O:", "O_")
            opt_file = self.options_raw_dir / dir_name / f"{date_str}.parquet"

            if opt_file.exists():
                df = pd.read_parquet(opt_file)
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                ).dt.tz_convert("America/New_York")
                df["ticker"] = ticker
                df["contract_type"] = _contract_type_from_ticker(ticker)
                dfs.append(df.sort_values("timestamp"))
            elif not features_df.empty and "ticker" in features_df.columns:
                # Fall back: extract OHLCV from features CSV
                sub = features_df[features_df["ticker"] == ticker][
                    [
                        "timestamp", "ticker", "contract_type",
                        "open", "high", "low", "close", "volume",
                        "implied_volatility",
                    ]
                ].copy()
                if not sub.empty:
                    dfs.append(sub)

        if not dfs:
            print("  Options raw: no data found")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        print(
            f"  Options raw: {len(result)} bars across "
            f"{result['ticker'].nunique()} contracts "
            f"(from {'Parquet' if opt_file.exists() else 'features fallback'})"
        )
        return result

    def _get_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run the loaded model on features and return per-row confidence."""
        if self._model is None or features_df.empty or not self._feature_cols:
            return pd.DataFrame()

        # Build feature matrix; missing columns → 0
        cols = []
        missing = []
        for c in self._feature_cols:
            if c in features_df.columns:
                cols.append(features_df[c].values)
            else:
                cols.append(np.zeros(len(features_df), dtype=np.float32))
                missing.append(c)

        X = np.column_stack(cols).astype(np.float32)
        proba = self._model.predict_proba(X)[:, 1]

        preds = pd.DataFrame({
            "timestamp": features_df["timestamp"],
            "ticker": features_df.get(
                "ticker", pd.Series([""] * len(features_df), index=features_df.index)
            ),
            "confidence": proba,
        })
        note = f" ({len(missing)} features 0-filled)" if missing else ""
        print(f"  Predictions: {len(preds)} rows{note}")
        return preds

    def _get_outcomes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract outcome labels from the feature DataFrame."""
        if features_df.empty:
            return pd.DataFrame()

        wanted = [
            "timestamp", "ticker", "contract_type",
            "target", "max_gain_120m", "min_loss_120m", "time_to_max_min",
        ]
        avail = [c for c in wanted if c in features_df.columns]
        outcomes = features_df[avail].copy()

        if "target" in outcomes.columns:
            n_pos = int(outcomes["target"].sum())
            n_neg = len(outcomes) - n_pos
            print(f"  Outcomes: {n_pos} positive, {n_neg} negative")
        return outcomes

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def build_dashboard(
        self,
        data: Dict,
        output_path: str = "dashboard.html",
    ) -> go.Figure:
        """Create a 5-row interactive HTML dashboard with Plotly.

        Rows
        ----
        1. SPY candlestick + volume
        2. CALL option contracts (close price)
        3. PUT option contracts (close price)
        4. Feature heatmap (time × 20 features, z-scored)
        5. Model confidence line + actual win-rate bars

        Args:
            data: Dict returned by :meth:`extract_sample_day`.
            output_path: Where to write the ``*.html`` file.

        Returns:
            The Plotly :class:`~plotly.graph_objects.Figure`.
        """
        date_str = data["date"]
        spy_raw = data["spy_raw"]
        options_raw = data["options_raw"]
        features = data["features"]
        predictions = data["predictions"]
        outcomes = data["outcomes"]

        fig = make_subplots(
            rows=5,
            cols=1,
            row_heights=[0.25, 0.18, 0.18, 0.22, 0.17],
            subplot_titles=(
                f"SPY Price & Volume — {date_str}",
                "CALL Contracts (close price)",
                "PUT Contracts (close price)",
                "Feature Heatmap — top 20 (z-scored ±3σ)",
                "Model Confidence & Actual Outcomes",
            ),
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"type": "heatmap"}],
                [{"secondary_y": True}],
            ],
            vertical_spacing=0.06,
        )

        # ── Row 1: SPY ──────────────────────────────────────────────────
        if not spy_raw.empty:
            if {"open", "high", "low", "close"}.issubset(spy_raw.columns):
                fig.add_trace(
                    go.Candlestick(
                        x=spy_raw["timestamp"],
                        open=spy_raw["open"],
                        high=spy_raw["high"],
                        low=spy_raw["low"],
                        close=spy_raw["close"],
                        name="SPY",
                        increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350",
                    ),
                    row=1, col=1,
                )
            elif "close" in spy_raw.columns:
                fig.add_trace(
                    go.Scatter(
                        x=spy_raw["timestamp"],
                        y=spy_raw["close"],
                        name="SPY Close",
                        line=dict(color="#26a69a"),
                    ),
                    row=1, col=1,
                )
            if "volume" in spy_raw.columns:
                fig.add_trace(
                    go.Bar(
                        x=spy_raw["timestamp"],
                        y=spy_raw["volume"],
                        name="SPY Volume",
                        marker_color="rgba(120, 120, 200, 0.3)",
                    ),
                    row=1, col=1, secondary_y=True,
                )

        # ── Rows 2 & 3: Option contracts ────────────────────────────────
        if not options_raw.empty and "ticker" in options_raw.columns:
            if "contract_type" in options_raw.columns:
                calls = options_raw[options_raw["contract_type"] == 1]
                puts = options_raw[options_raw["contract_type"] == 0]
            else:
                calls = options_raw[options_raw["ticker"].str.contains(r"\dC\d", regex=True)]
                puts = options_raw[options_raw["ticker"].str.contains(r"\dP\d", regex=True)]

            for i, ticker in enumerate(calls["ticker"].unique()[:3]):
                sub = calls[calls["ticker"] == ticker].sort_values("timestamp")
                # Extract a readable label from ticker (e.g. "C608.0 0dte")
                m = re.search(r"([CP])(\d{8})$", ticker)
                label = f"C{int(m.group(2))/1000:.0f}" if m else ticker[-12:]
                fig.add_trace(
                    go.Scatter(
                        x=sub["timestamp"],
                        y=sub["close"],
                        name=label,
                        mode="lines",
                        line=dict(color=_CALL_COLORS[i % len(_CALL_COLORS)]),
                    ),
                    row=2, col=1,
                )

            for i, ticker in enumerate(puts["ticker"].unique()[:3]):
                sub = puts[puts["ticker"] == ticker].sort_values("timestamp")
                m = re.search(r"([CP])(\d{8})$", ticker)
                label = f"P{int(m.group(2))/1000:.0f}" if m else ticker[-12:]
                fig.add_trace(
                    go.Scatter(
                        x=sub["timestamp"],
                        y=sub["close"],
                        name=label,
                        mode="lines",
                        line=dict(color=_PUT_COLORS[i % len(_PUT_COLORS)]),
                    ),
                    row=3, col=1,
                )

        # ── Row 4: Feature heatmap ───────────────────────────────────────
        if not features.empty:
            avail_feats = [c for c in _HEATMAP_FEATURES if c in features.columns]
            if avail_feats:
                # Average across tickers per timestamp
                feat_agg = (
                    features.groupby("timestamp")[avail_feats]
                    .mean()
                    .sort_index()
                )
                # Z-score and clamp to ±3σ
                mean_ = feat_agg.mean()
                std_ = feat_agg.std().replace(0, 1)
                feat_z = ((feat_agg - mean_) / std_).clip(-3, 3)

                fig.add_trace(
                    go.Heatmap(
                        z=feat_z.T.values,
                        x=feat_z.index,
                        y=feat_z.columns.tolist(),
                        colorscale="RdBu",
                        zmid=0,
                        zmin=-3,
                        zmax=3,
                        name="Features",
                        colorbar=dict(
                            title="σ",
                            len=0.2,
                            y=0.25,
                            tickfont=dict(size=9),
                        ),
                    ),
                    row=4, col=1,
                )

        # ── Row 5: Predictions & Outcomes ───────────────────────────────
        if not predictions.empty and "confidence" in predictions.columns:
            pred_agg = (
                predictions.groupby("timestamp")["confidence"]
                .mean()
                .sort_index()
                .reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_agg["timestamp"],
                    y=pred_agg["confidence"],
                    name="Avg Model Confidence",
                    mode="lines",
                    line=dict(color="#ff7f0e", width=2),
                ),
                row=5, col=1,
            )
            # 0.70 decision threshold line
            fig.add_hline(
                y=0.70,
                line_dash="dash",
                line_color="rgba(255,80,80,0.7)",
                annotation_text="0.70 threshold",
                annotation_font_size=10,
                row=5, col=1,
            )

        if not outcomes.empty and "target" in outcomes.columns:
            out_agg = (
                outcomes.groupby("timestamp")["target"]
                .mean()
                .sort_index()
                .reset_index()
            )
            bar_colors = [
                "rgba(38,166,154,0.55)" if v > 0 else "rgba(239,83,80,0.25)"
                for v in out_agg["target"]
            ]
            fig.add_trace(
                go.Bar(
                    x=out_agg["timestamp"],
                    y=out_agg["target"],
                    name="Actual Win Rate",
                    marker_color=bar_colors,
                ),
                row=5, col=1, secondary_y=True,
            )

        # ── Layout ──────────────────────────────────────────────────────
        fig.update_layout(
            height=1900,
            title_text=f"Data Pipeline Dashboard — {date_str}",
            showlegend=True,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(
                orientation="v",
                x=1.01,
                y=1.0,
                xanchor="left",
                font=dict(size=9),
            ),
            margin=dict(r=160),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

        # Axis labels
        fig.update_yaxes(title_text="SPY Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)
        fig.update_yaxes(title_text="Confidence", row=5, col=1,
                         secondary_y=False, range=[0, 1])
        fig.update_yaxes(title_text="Win Rate", row=5, col=1,
                         secondary_y=True, range=[0, 1])
        fig.update_xaxes(title_text="Time (ET)", row=5, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        print(f"\n  Dashboard saved: {out_path}")
        return fig

    # ------------------------------------------------------------------
    # Feature derivation console viewer
    # ------------------------------------------------------------------

    def show_feature_derivation(
        self,
        features_df: pd.DataFrame,
        minute_idx: int = 0,
    ) -> None:
        """Print how features were calculated for a specific minute bar.

        Shows raw OHLCV values, derived returns, technical indicators,
        time features, and the actual outcome label.

        Args:
            features_df: Feature DataFrame for one trading day.
            minute_idx:  Row index to inspect (0-indexed, default 0).
        """
        if features_df.empty or minute_idx >= len(features_df):
            print("No data or index out of range.")
            return

        row = features_df.iloc[minute_idx]
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  FEATURE DERIVATION — Row {minute_idx}")
        print(sep)

        ts = row.get("timestamp", "?")
        print(f"\n  Timestamp        : {ts}")
        print(f"  Ticker           : {row.get('ticker', '?')}")
        ctype_int = int(row.get("contract_type", -1))
        ctype_str = "CALL" if ctype_int == 1 else ("PUT" if ctype_int == 0 else "?")
        print(f"  Type             : {ctype_str}  (contract_type={ctype_int})")
        print(f"  Strike           : ${row.get('strike', float('nan')):.2f}")

        print(f"\n  --- SPY at this bar ---")
        for col, lbl in (
            ("spy_open",   "spy_open  "),
            ("spy_high",   "spy_high  "),
            ("spy_low",    "spy_low   "),
            ("spy_close",  "spy_close "),
        ):
            v = row.get(col, float("nan"))
            print(f"  {lbl}        : ${v:.4f}")
        print(f"  spy_volume       : {int(row.get('spy_volume', 0)):,}")

        print(f"\n  --- Option at this bar ---")
        for col, lbl in (
            ("open",   "open   "),
            ("high",   "high   "),
            ("low",    "low    "),
            ("close",  "close  "),
        ):
            v = row.get(col, float("nan"))
            print(f"  {lbl}            : ${v:.4f}")
        print(f"  volume           : {int(row.get('volume', 0)):,}")
        iv = row.get("implied_volatility", float("nan"))
        print(f"  implied_vol      : {iv:.4f}  ({iv * 100:.2f}%)")
        print(f"  moneyness        : {row.get('moneyness', float('nan')):.4f}")

        print(f"\n  --- Derived Returns ---")
        for feat in (
            "spy_return_1m", "spy_return_5m", "spy_return_15m",
            "opt_return_1m", "opt_return_5m", "opt_return_15m",
            "opt_vs_spy_return_1m",
        ):
            if feat in row.index:
                print(f"  {feat:<26}: {row[feat]:+.4f}%")

        print(f"\n  --- Technical Indicators ---")
        for feat in (
            "spy_rsi_14", "spy_macd", "spy_macd_hist",
            "spy_bb_pct_b", "spy_ema_diff",
            "spy_vol_zscore", "spy_vol_ratio_5m",
            "opt_rsi_14", "opt_hl_range_5m",
        ):
            if feat in row.index:
                print(f"  {feat:<26}: {row[feat]:.4f}")

        print(f"\n  --- Time Features ---")
        print(f"  hour_et          : {int(row.get('hour_et', 0)):02d}:xx ET")
        print(f"  minute_of_day    : {int(row.get('minute_of_day', 0))}")
        print(f"  minutes_since_open: {int(row.get('minutes_since_open', 0))}")
        print(f"  pct_day_elapsed  : {row.get('pct_day_elapsed', float('nan')):.3f}")
        print(f"  is_morning       : {int(row.get('is_morning', 0))}")
        print(f"  is_last_hour     : {int(row.get('is_last_hour', 0))}")

        print(f"\n  --- Actual Outcome ---")
        target = row.get("target", float("nan"))
        outcome_str = "WIN" if target == 1 else ("LOSS" if target == 0 else "?")
        print(f"  target           : {int(target) if not np.isnan(target) else '?'}  ({outcome_str})")
        print(f"  max_gain_120m    : {row.get('max_gain_120m', float('nan')):+.2f}%")
        print(f"  min_loss_120m    : {row.get('min_loss_120m', float('nan')):+.2f}%")
        print(f"  time_to_max_min  : {row.get('time_to_max_min', float('nan')):.0f} min")
        print(f"\n{sep}\n")
