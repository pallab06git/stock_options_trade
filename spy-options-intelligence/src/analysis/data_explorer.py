# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Streamlit data explorer — tabular view of the full data pipeline.

Launch
------
    streamlit run src/analysis/data_explorer.py
    streamlit run src/analysis/data_explorer.py -- --features-dir data/processed/features
"""

from __future__ import annotations

import re
import sys
from datetime import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path defaults (can be overridden via CLI args passed after --)
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES_DIR = Path("data/processed/features")
_DEFAULT_SPY_RAW_DIR = Path("data/raw/spy")
_DEFAULT_OPTIONS_RAW_DIR = Path("data/raw/options/minute")
_DEFAULT_MODEL_PATH = "models/xgboost_v3_clean.pkl"

# Parse optional --features-dir argument passed after `--`
_features_dir = _DEFAULT_FEATURES_DIR
for i, arg in enumerate(sys.argv):
    if arg == "--features-dir" and i + 1 < len(sys.argv):
        _features_dir = Path(sys.argv[i + 1])

# ---------------------------------------------------------------------------
# Column groups for the Features tab
# ---------------------------------------------------------------------------

_COL_GROUPS = {
    "Returns (SPY)": [
        "spy_return_1m", "spy_return_5m", "spy_return_15m",
        "spy_return_30m", "spy_return_60m",
    ],
    "Returns (Option)": [
        "opt_return_1m", "opt_return_5m", "opt_return_15m",
        "opt_return_30m", "opt_return_60m",
        "opt_price_change_open", "opt_vs_spy_return_1m",
    ],
    "SPY Technicals": [
        "spy_rsi_14", "spy_macd", "spy_macd_signal", "spy_macd_hist",
        "spy_bb_pct_b", "spy_bb_upper", "spy_bb_lower",
        "spy_ema_9", "spy_ema_21", "spy_ema_diff",
        "spy_vwap_dist_pct", "spy_hl_range_5m", "spy_hl_range_30m",
    ],
    "Option Technicals": [
        "opt_rsi_14", "opt_hl_range_5m", "opt_vwap_dist_pct",
        "implied_volatility", "iv_change_1m", "iv_change_5m", "iv_change_open",
    ],
    "Volume": [
        "spy_vol_ma5", "spy_vol_ma30", "spy_vol_ratio_5m", "spy_vol_ratio_30m",
        "spy_vol_zscore", "spy_vol_std_5m", "spy_vol_std_30m", "spy_vol_regime",
        "opt_vol_ma5", "opt_vol_ratio_5m", "transactions_ratio", "opt_bar_count",
    ],
    "Strike / Moneyness": [
        "strike", "contract_type", "moneyness", "log_moneyness",
        "time_to_expiry_days", "is_0dte",
    ],
    "Time": [
        "hour_et", "minute_et", "minute_of_day", "minutes_since_open",
        "pct_day_elapsed", "is_morning", "is_last_hour",
    ],
}

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _get_available_dates(features_dir: str) -> List[str]:
    files = sorted(Path(features_dir).glob("*_features.csv"))
    return [f.stem.replace("_features", "") for f in files]


@st.cache_data(show_spinner=False)
def _load_features(date_str: str, features_dir: str) -> pd.DataFrame:
    path = Path(features_dir) / f"{date_str}_features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("America/New_York")
        )
    return df


@st.cache_data(show_spinner=False)
def _load_spy_raw(date_str: str, spy_raw_dir: str) -> pd.DataFrame:
    path = Path(spy_raw_dir) / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("America/New_York")
        )
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str):
    """Returns (model, feature_cols) or (None, [])."""
    import pickle
    if not model_path or not Path(model_path).exists():
        return None, []
    try:
        with open(model_path, "rb") as fh:
            artifact = pickle.load(fh)
    except Exception:
        try:
            import joblib
            artifact = joblib.load(model_path)
        except Exception:
            return None, []
    return artifact.get("model"), artifact.get("feature_cols", [])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"{val:+.2f}%"


def _color_return(val):
    """Green for positive returns, red for negative."""
    if pd.isna(val) or val == 0:
        return ""
    return "color: #26a69a" if val > 0 else "color: #ef5350"


def _color_target(val):
    if val == 1:
        return "background-color: rgba(38,166,154,0.25); color: #26a69a; font-weight: bold"
    if val == 0:
        return "background-color: rgba(239,83,80,0.18); color: #ef5350"
    return ""


def _color_confidence(val):
    if pd.isna(val):
        return ""
    if val >= 0.70:
        return "background-color: rgba(38,166,154,0.3); font-weight: bold"
    if val >= 0.50:
        return "background-color: rgba(255,193,7,0.2)"
    return ""


def _format_ts(ts_series: pd.Series) -> pd.Series:
    """Format timezone-aware timestamp to HH:MM string."""
    try:
        return ts_series.dt.strftime("%H:%M")
    except Exception:
        return ts_series.astype(str)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="SPY Options Data Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 SPY Options Data Explorer")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        # Date
        dates = _get_available_dates(str(_features_dir))
        if not dates:
            st.error(f"No feature files found in {_features_dir}")
            st.stop()

        # Default to most recent date
        default_idx = len(dates) - 1
        date_str = st.selectbox("Date", dates, index=default_idx)

        # Time range
        st.subheader("Time Range (ET)")
        col_a, col_b = st.columns(2)
        with col_a:
            t_start = st.time_input("From", value=time(9, 30), step=60)
        with col_b:
            t_end = st.time_input("To", value=time(16, 0), step=60)

        st.subheader("Contract Filter")
        contract_type = st.radio(
            "Type", ["All", "CALL", "PUT"], horizontal=True
        )

        # Model
        st.subheader("Model (optional)")
        model_path_input = st.text_input(
            "Path", value=_DEFAULT_MODEL_PATH, help="Leave blank to hide predictions"
        )

    # ── Load data ────────────────────────────────────────────────────────
    with st.spinner(f"Loading {date_str}…"):
        features_all = _load_features(date_str, str(_features_dir))
        spy_raw_all = _load_spy_raw(date_str, str(_DEFAULT_SPY_RAW_DIR))

    if features_all.empty:
        st.error(f"No feature data for {date_str}. Check --features-dir.")
        st.stop()

    # ── Apply time filter ────────────────────────────────────────────────
    def _filter_time(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "timestamp" not in df.columns:
            return df
        ts = df["timestamp"].dt.time
        return df[(ts >= t_start) & (ts <= t_end)].reset_index(drop=True)

    features = _filter_time(features_all)
    spy_raw = _filter_time(spy_raw_all)

    # ── Apply contract type filter ───────────────────────────────────────
    if contract_type != "All" and "contract_type" in features.columns:
        ct_val = 1 if contract_type == "CALL" else 0
        features = features[features["contract_type"] == ct_val].reset_index(drop=True)

    # ── Ticker filter in sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.subheader("Tickers")
        all_tickers = sorted(features["ticker"].unique().tolist()) if "ticker" in features.columns else []
        if all_tickers:
            selected_tickers = st.multiselect(
                "Show tickers",
                options=all_tickers,
                default=all_tickers[:5],
            )
            if selected_tickers:
                features = features[
                    features["ticker"].isin(selected_tickers)
                ].reset_index(drop=True)

    # ── Summary strip ────────────────────────────────────────────────────
    n_rows = len(features)
    n_tickers = features["ticker"].nunique() if "ticker" in features.columns else 0
    n_pos = int(features["target"].sum()) if "target" in features.columns else 0
    n_total = len(features)
    pos_rate = n_pos / n_total if n_total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", date_str)
    c2.metric("Rows", f"{n_rows:,}")
    c3.metric("Contracts", n_tickers)
    c4.metric("Win rate", f"{pos_rate:.1%}  ({n_pos}/{n_total})")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_spy, tab_opts, tab_feats, tab_preds = st.tabs(
        ["SPY Raw", "Options", "Features", "Predictions & Outcomes"]
    )

    # ── Tab 1: SPY Raw ───────────────────────────────────────────────────
    with tab_spy:
        st.subheader(f"SPY Minute Bars — {date_str}")

        if spy_raw.empty:
            # Fall back: reconstruct from features (deduplicated per timestamp)
            spy_cols = ["timestamp", "spy_open", "spy_high", "spy_low",
                        "spy_close", "spy_volume", "spy_vwap"]
            avail = [c for c in spy_cols if c in features.columns]
            spy_raw = (
                features[avail]
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            st.caption("Source: reconstructed from features CSV (raw Parquet not found)")
        else:
            st.caption(f"Source: data/raw/spy/{date_str}.parquet")

        display_cols = [c for c in
                        ["timestamp", "open", "high", "low", "close", "volume", "vwap",
                         "spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "spy_vwap"]
                        if c in spy_raw.columns]
        spy_display = spy_raw[display_cols].copy()

        if "timestamp" in spy_display.columns:
            spy_display["time"] = _format_ts(spy_display["timestamp"])
            spy_display = spy_display.drop(columns=["timestamp"])
            cols_order = ["time"] + [c for c in spy_display.columns if c != "time"]
            spy_display = spy_display[cols_order]

        # Rename spy_* columns for cleaner display
        spy_display = spy_display.rename(columns={
            "spy_open": "open", "spy_high": "high", "spy_low": "low",
            "spy_close": "close", "spy_volume": "volume", "spy_vwap": "vwap",
        })

        price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in spy_display.columns]
        vol_cols = [c for c in ["volume"] if c in spy_display.columns]

        styler = spy_display.style
        if price_cols:
            styler = styler.format({c: "{:.2f}" for c in price_cols})
        if vol_cols:
            styler = styler.format({c: "{:,.0f}" for c in vol_cols})

        st.dataframe(styler, use_container_width=True, height=500)
        st.caption(f"{len(spy_display)} bars  |  {t_start.strftime('%H:%M')} – {t_end.strftime('%H:%M')} ET")

    # ── Tab 2: Options ───────────────────────────────────────────────────
    with tab_opts:
        st.subheader(f"Option Minute Bars — {date_str}")

        opt_cols = [
            "timestamp", "ticker", "contract_type",
            "strike", "time_to_expiry_days", "is_0dte",
            "open", "high", "low", "close", "volume",
            "implied_volatility",
            "opt_return_1m", "opt_return_5m",
        ]
        avail = [c for c in opt_cols if c in features.columns]
        opt_display = features[avail].copy()

        # Readable contract type
        if "contract_type" in opt_display.columns:
            opt_display["type"] = opt_display["contract_type"].map({1: "CALL", 0: "PUT"})
            opt_display = opt_display.drop(columns=["contract_type"])

        # Short ticker label (last 12 chars)
        if "ticker" in opt_display.columns:
            opt_display["ticker_short"] = opt_display["ticker"].str[-14:]

        if "timestamp" in opt_display.columns:
            opt_display["time"] = _format_ts(opt_display["timestamp"])
            opt_display = opt_display.drop(columns=["timestamp"])

        # Reorder: time, ticker_short, type, strike, ...
        lead = [c for c in ["time", "ticker_short", "type", "strike",
                             "time_to_expiry_days", "is_0dte"]
                if c in opt_display.columns]
        rest = [c for c in opt_display.columns if c not in lead and c != "ticker"]
        opt_display = opt_display[lead + rest]

        fmt = {}
        for c in ["open", "high", "low", "close"]:
            if c in opt_display.columns:
                fmt[c] = "{:.3f}"
        if "strike" in opt_display.columns:
            fmt["strike"] = "{:.1f}"
        if "implied_volatility" in opt_display.columns:
            fmt["implied_volatility"] = "{:.2%}"
        if "volume" in opt_display.columns:
            fmt["volume"] = "{:,.0f}"
        for c in ["opt_return_1m", "opt_return_5m"]:
            if c in opt_display.columns:
                fmt[c] = "{:+.2f}%"

        ret_cols = [c for c in ["opt_return_1m", "opt_return_5m"] if c in opt_display.columns]
        styler = opt_display.style.format(fmt)
        if ret_cols:
            styler = styler.applymap(_color_return, subset=ret_cols)

        st.dataframe(styler, use_container_width=True, height=500)
        st.caption(f"{len(opt_display):,} rows  |  {n_tickers} contract(s)")

    # ── Tab 3: Features ──────────────────────────────────────────────────
    with tab_feats:
        st.subheader(f"Calculated Features — {date_str}")

        # Column group selector
        all_groups = list(_COL_GROUPS.keys())
        selected_groups = st.multiselect(
            "Feature groups",
            options=all_groups,
            default=["Returns (SPY)", "Returns (Option)", "SPY Technicals"],
            help="Choose which feature categories to display",
        )

        selected_feat_cols: List[str] = []
        for g in selected_groups:
            selected_feat_cols += [c for c in _COL_GROUPS[g] if c in features.columns]

        if not selected_feat_cols:
            st.info("Select at least one feature group above.")
        else:
            id_cols = [c for c in ["timestamp", "ticker", "contract_type"]
                       if c in features.columns]
            feat_display = features[id_cols + selected_feat_cols].copy()

            if "contract_type" in feat_display.columns:
                feat_display["type"] = feat_display["contract_type"].map({1: "CALL", 0: "PUT"})
                feat_display = feat_display.drop(columns=["contract_type"])

            if "timestamp" in feat_display.columns:
                feat_display["time"] = _format_ts(feat_display["timestamp"])
                feat_display = feat_display.drop(columns=["timestamp"])

            lead = [c for c in ["time", "ticker", "type"] if c in feat_display.columns]
            data_cols = [c for c in feat_display.columns if c not in lead]
            feat_display = feat_display[lead + data_cols]

            # Format numeric columns
            fmt = {c: "{:+.4f}" for c in data_cols if feat_display[c].dtype in (float, np.float64)}
            # Return cols get % suffix
            ret_cols_all = (
                _COL_GROUPS["Returns (SPY)"]
                + _COL_GROUPS["Returns (Option)"]
            )
            ret_cols_present = [c for c in ret_cols_all if c in feat_display.columns]
            for c in ret_cols_present:
                fmt[c] = "{:+.3f}%"

            styler = feat_display.style.format(fmt, na_rep="—")
            if ret_cols_present:
                styler = styler.applymap(_color_return, subset=ret_cols_present)

            st.dataframe(styler, use_container_width=True, height=500)
            st.caption(
                f"{len(feat_display):,} rows  |  {len(data_cols)} feature columns shown"
            )

    # ── Tab 4: Predictions & Outcomes ────────────────────────────────────
    with tab_preds:
        st.subheader(f"Predictions & Outcomes — {date_str}")

        model, feature_cols = _load_model(model_path_input)

        # Build base outcome table
        outcome_cols = [c for c in
                        ["timestamp", "ticker", "contract_type",
                         "target", "max_gain_120m", "min_loss_120m", "time_to_max_min"]
                        if c in features.columns]
        pred_display = features[outcome_cols].copy()

        # Model predictions
        if model is not None and feature_cols:
            cols = []
            for c in feature_cols:
                if c in features.columns:
                    cols.append(features[c].values)
                else:
                    cols.append(np.zeros(len(features), dtype=np.float32))
            X = np.column_stack(cols).astype(np.float32)
            proba = model.predict_proba(X)[:, 1]
            pred_display["confidence"] = proba
        else:
            if model_path_input and not Path(model_path_input).exists():
                st.warning(f"Model not found: {model_path_input}")

        # Readable labels
        if "contract_type" in pred_display.columns:
            pred_display["type"] = pred_display["contract_type"].map({1: "CALL", 0: "PUT"})
            pred_display = pred_display.drop(columns=["contract_type"])

        if "target" in pred_display.columns:
            pred_display["outcome"] = pred_display["target"].map({1: "WIN", 0: "LOSS"})

        if "timestamp" in pred_display.columns:
            pred_display["time"] = _format_ts(pred_display["timestamp"])
            pred_display = pred_display.drop(columns=["timestamp"])

        # Reorder columns
        lead = [c for c in ["time", "ticker", "type", "confidence", "outcome", "target"]
                if c in pred_display.columns]
        rest = [c for c in pred_display.columns if c not in lead]
        pred_display = pred_display[lead + rest]

        fmt = {}
        if "confidence" in pred_display.columns:
            fmt["confidence"] = "{:.3f}"
        if "target" in pred_display.columns:
            fmt["target"] = "{:.0f}"
        for c in ["max_gain_120m", "min_loss_120m"]:
            if c in pred_display.columns:
                fmt[c] = "{:+.2f}%"
        if "time_to_max_min" in pred_display.columns:
            fmt["time_to_max_min"] = "{:.0f} min"

        styler = pred_display.style.format(fmt, na_rep="—")
        if "target" in pred_display.columns:
            styler = styler.applymap(_color_target, subset=["target"])
        if "outcome" in pred_display.columns:
            styler = styler.applymap(_color_target, subset=["outcome"])
        if "confidence" in pred_display.columns:
            styler = styler.applymap(_color_confidence, subset=["confidence"])

        st.dataframe(styler, use_container_width=True, height=500)

        # Quick stats
        if "target" in pred_display.columns and len(pred_display) > 0:
            n_p = int(pred_display["target"].sum())
            n_n = len(pred_display) - n_p
            st.caption(
                f"{len(pred_display):,} rows  |  "
                f"WIN: {n_p}  LOSS: {n_n}  ({n_p/len(pred_display):.1%} win rate)"
            )
            if "confidence" in pred_display.columns:
                signals = (pred_display["confidence"] >= 0.70).sum()
                st.caption(f"Signals @ ≥0.70 threshold: {signals}")


if __name__ == "__main__":
    main()
