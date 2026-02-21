# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Standalone Streamlit dashboard for ML model comparison results.

Reads the JSON / CSV output produced by ``ModelComparator.save_results()``
and renders four interactive tabs:

  1. **Model Comparison**  — side-by-side metrics table + net-profit bar chart
  2. **Threshold Sweep**   — per-model, per-threshold line charts (net profit,
                              win rate, signal count)
  3. **Signal Overlap**    — Venn-style breakdown of how many signals each
                              combination of models agree on
  4. **Trade Explorer**    — filterable table of individual trades for a chosen
                              model + threshold

Launch
------
    streamlit run src/ml/dashboard.py -- --results-dir reports/model_comparison

The ``--results-dir`` argument defaults to ``data/reports/model_comparison``
if not provided.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Streamlit import guard
# ---------------------------------------------------------------------------

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "streamlit is required for this dashboard. "
        "Install it with: pip install streamlit>=1.30.0"
    ) from exc

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = Path("data/reports/model_comparison")

# Threshold labels shown in the UI
_THRESHOLD_LABELS = {
    0.70: "70%",
    0.75: "75%",
    0.80: "80%",
    0.85: "85%",
    0.90: "90%",
    0.95: "95%",
}

# Colour palette for models (cycles if more than 6 models)
_MODEL_COLOURS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_results_dir(results_dir: str) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Load all ``*_results.json`` files from ``results_dir``.

    Returns
    -------
    Dict mapping ``model_name → {threshold → result_dict}``.
    """
    path = Path(results_dir)
    if not path.exists():
        return {}

    all_results: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for json_path in sorted(path.glob("*_results.json")):
        model_name = json_path.stem.replace("_results", "")
        try:
            with open(json_path) as fh:
                raw = json.load(fh)
            # Keys are stored as strings (JSON limitation) — convert back to float
            all_results[model_name] = {float(k): v for k, v in raw.items()}
        except Exception as exc:
            st.warning(f"Could not load {json_path.name}: {exc}")

    return all_results


@st.cache_data(show_spinner=False)
def _load_comparison_csv(results_dir: str) -> pd.DataFrame:
    """Load ``model_comparison.csv`` from the results directory."""
    path = Path(results_dir) / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_overlap_json(results_dir: str, threshold: float) -> Dict[str, Any]:
    """Load ``overlap_{threshold:.2f}.json`` from the results directory."""
    path = Path(results_dir) / f"overlap_{threshold:.2f}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Parse CLI args (streamlit passes extra args after --)
# ---------------------------------------------------------------------------


def _get_results_dir() -> str:
    """Return the results directory from CLI args or the default."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--results-dir", "--results_dir") and i + 1 < len(args):
            return args[i + 1]
    return str(_DEFAULT_RESULTS_DIR)


# ---------------------------------------------------------------------------
# Tab 1: Model Comparison
# ---------------------------------------------------------------------------


def _tab_model_comparison(
    results_dir: str,
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
) -> None:
    st.header("Model Comparison")

    comp_df = _load_comparison_csv(results_dir)

    if comp_df.empty and not all_results:
        st.info(
            "No model comparison data found. "
            "Run `ModelComparator.save_results()` first."
        )
        return

    # --- Comparison table ---
    if not comp_df.empty:
        st.subheader("Side-by-Side Metrics (80% threshold)")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(
            "Signals, Win Rate, Avg Win/Loss, and Net Profit are all measured "
            "at the 80% confidence threshold."
        )

    # --- Net profit bar chart ---
    if all_results:
        st.subheader("Net Profit vs Threshold")

        model_names = list(all_results.keys())
        thresholds = sorted(
            {t for res in all_results.values() for t in res.keys()}
        )

        if _PLOTLY_AVAILABLE:
            fig = go.Figure()
            for i, model_name in enumerate(model_names):
                colour = _MODEL_COLOURS[i % len(_MODEL_COLOURS)]
                res = all_results[model_name]
                xs = [t for t in thresholds if t in res]
                ys = [res[t].get("net_profit_usd", 0.0) for t in xs]
                fig.add_trace(
                    go.Scatter(
                        x=[f"{t:.0%}" for t in xs],
                        y=ys,
                        mode="lines+markers",
                        name=model_name,
                        line=dict(color=colour, width=2),
                        marker=dict(size=8),
                        hovertemplate=(
                            f"<b>{model_name}</b><br>"
                            "Threshold: %{x}<br>"
                            "Net Profit: $%{y:,.0f}<extra></extra>"
                        ),
                    )
                )

            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Break-even",
            )
            fig.update_layout(
                xaxis_title="Confidence Threshold",
                yaxis_title="Net Profit (USD)",
                legend_title="Model",
                hovermode="x unified",
                height=420,
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: plain DataFrame
            rows = []
            for model_name, res in all_results.items():
                for t in thresholds:
                    if t in res:
                        rows.append(
                            {
                                "Model": model_name,
                                "Threshold": f"{t:.0%}",
                                "Net Profit": f"${res[t].get('net_profit_usd', 0):+,.0f}",
                            }
                        )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Win rate bar chart ---
    if all_results and _PLOTLY_AVAILABLE:
        st.subheader("Win Rate vs Threshold")

        fig2 = go.Figure()
        for i, model_name in enumerate(model_names):
            colour = _MODEL_COLOURS[i % len(_MODEL_COLOURS)]
            res = all_results[model_name]
            xs = [t for t in thresholds if t in res]
            ys = [res[t].get("win_rate", 0.0) * 100 for t in xs]
            fig2.add_trace(
                go.Scatter(
                    x=[f"{t:.0%}" for t in xs],
                    y=ys,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=colour, width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        "Threshold: %{x}<br>"
                        "Win Rate: %{y:.1f}%<extra></extra>"
                    ),
                )
            )
        fig2.update_layout(
            xaxis_title="Confidence Threshold",
            yaxis_title="Win Rate (%)",
            legend_title="Model",
            hovermode="x unified",
            height=380,
            margin=dict(t=20),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Threshold Sweep
# ---------------------------------------------------------------------------


def _tab_threshold_sweep(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
) -> None:
    st.header("Threshold Sweep")

    if not all_results:
        st.info("No results loaded.")
        return

    model_names = sorted(all_results.keys())
    selected_model = st.selectbox("Select model", model_names, key="sweep_model")
    res = all_results.get(selected_model, {})

    if not res:
        st.warning(f"No threshold results found for model '{selected_model}'.")
        return

    thresholds = sorted(res.keys())
    rows = []
    for t in thresholds:
        r = res[t]
        rows.append(
            {
                "Threshold": f"{t:.0%}",
                "Signals": r.get("total_signals", 0),
                "Calls": r.get("calls", 0),
                "Puts": r.get("puts", 0),
                "Win Rate (%)": round(r.get("win_rate", 0.0) * 100, 1),
                "Gross Profit": r.get("gross_profit_usd", 0.0),
                "Gross Loss": r.get("gross_loss_usd", 0.0),
                "Fees": r.get("total_fees_usd", 0.0),
                "Net Profit": r.get("net_profit_usd", 0.0),
                "Avg P&L / Trade": r.get("avg_pnl_per_trade_usd", 0.0),
            }
        )

    df = pd.DataFrame(rows)

    # --- Metrics table ---
    st.subheader(f"{selected_model} — Metrics by Threshold")
    st.dataframe(
        df.style.format(
            {
                "Gross Profit": "${:,.0f}",
                "Gross Loss": "${:,.0f}",
                "Fees": "${:,.0f}",
                "Net Profit": "${:+,.0f}",
                "Avg P&L / Trade": "${:+,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not _PLOTLY_AVAILABLE:
        return

    # --- Signal count bar chart ---
    st.subheader("Signal Count by Threshold")
    fig_sig = px.bar(
        df,
        x="Threshold",
        y="Signals",
        color_discrete_sequence=["#1f77b4"],
        labels={"Signals": "Total Signals"},
        height=320,
    )
    fig_sig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_sig, use_container_width=True)

    # --- Net profit line chart ---
    st.subheader("Net Profit by Threshold")
    fig_net = go.Figure()
    fig_net.add_trace(
        go.Bar(
            x=df["Threshold"],
            y=df["Net Profit"],
            marker_color=[
                "#2ca02c" if v >= 0 else "#d62728" for v in df["Net Profit"]
            ],
            hovertemplate="Threshold: %{x}<br>Net Profit: $%{y:+,.0f}<extra></extra>",
        )
    )
    fig_net.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_net.update_layout(
        yaxis_title="Net Profit (USD)",
        height=320,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # --- Calls vs Puts stacked bar ---
    st.subheader("Calls vs Puts by Threshold")
    fig_cp = go.Figure()
    fig_cp.add_trace(
        go.Bar(
            x=df["Threshold"],
            y=df["Calls"],
            name="Calls",
            marker_color="#2ca02c",
        )
    )
    fig_cp.add_trace(
        go.Bar(
            x=df["Threshold"],
            y=df["Puts"],
            name="Puts",
            marker_color="#d62728",
        )
    )
    fig_cp.update_layout(barmode="stack", height=320, margin=dict(t=20))
    st.plotly_chart(fig_cp, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Signal Overlap
# ---------------------------------------------------------------------------


def _tab_signal_overlap(results_dir: str) -> None:
    st.header("Signal Overlap")

    threshold_options = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    selected_t = st.selectbox(
        "Threshold",
        threshold_options,
        index=2,  # default 0.80
        format_func=lambda t: f"{t:.0%}",
        key="overlap_threshold",
    )

    overlap = _load_overlap_json(results_dir, selected_t)

    if not overlap:
        st.info(
            f"No overlap data found for threshold {selected_t:.0%}. "
            "Run `ModelComparator.save_results()` first."
        )
        return

    # --- Summary KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Signals", overlap.get("total_unique_signals", 0))
    col2.metric("Models Compared", overlap.get("n_models_compared", 0))
    col3.metric("All Models Agree", overlap.get("all_models_agree", 0))
    col4.metric("Majority Agree", overlap.get("majority_agree", 0))

    st.caption(
        f"Models: {', '.join(overlap.get('model_names', []))}"
    )

    # --- Overlap breakdown bar chart ---
    breakdown = overlap.get("overlap_breakdown", {})
    if breakdown and _PLOTLY_AVAILABLE:
        st.subheader("Signal Agreement Breakdown")
        labels = [k.replace("_models", " model(s)") for k in breakdown.keys()]
        values = list(breakdown.values())

        fig = px.bar(
            x=labels,
            y=values,
            labels={"x": "Agreement Level", "y": "Number of Signals"},
            color=values,
            color_continuous_scale=["#d62728", "#ff7f0e", "#2ca02c"],
            height=360,
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Detailed overlap table ---
    detailed = overlap.get("detailed_overlaps", {})
    if detailed:
        st.subheader("Signals Where All Models Agree")
        n_models_str = str(overlap.get("n_models_compared", ""))
        all_agree = detailed.get(n_models_str, [])
        if all_agree:
            agree_df = pd.DataFrame(all_agree)
            if "models" in agree_df.columns:
                agree_df["models"] = agree_df["models"].apply(
                    lambda m: ", ".join(m) if isinstance(m, list) else m
                )
            st.dataframe(agree_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals where all models agree at this threshold.")


# ---------------------------------------------------------------------------
# Tab 4: Trade Explorer
# ---------------------------------------------------------------------------


def _tab_trade_explorer(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
) -> None:
    st.header("Trade Explorer")

    if not all_results:
        st.info("No results loaded.")
        return

    model_names = sorted(all_results.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Model", model_names, key="explorer_model")
    with col2:
        thresholds = sorted(all_results.get(selected_model, {}).keys())
        if not thresholds:
            st.warning(f"No threshold results for '{selected_model}'.")
            return
        selected_t = st.selectbox(
            "Threshold",
            thresholds,
            index=min(2, len(thresholds) - 1),
            format_func=lambda t: f"{t:.0%}",
            key="explorer_threshold",
        )

    res = all_results.get(selected_model, {}).get(selected_t, {})
    raw_trades = res.get("trades", [])

    if not raw_trades:
        st.info("No trades at this threshold.")
        return

    trades_df = pd.DataFrame(raw_trades)

    # --- Summary metrics ---
    n_trades = len(trades_df)
    n_wins = trades_df.get("is_winner", pd.Series(dtype=bool)).sum() if "is_winner" in trades_df.columns else 0
    net = res.get("net_profit_usd", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", n_trades)
    col2.metric("Winners", int(n_wins))
    col3.metric(
        "Win Rate",
        f"{n_wins / n_trades:.1%}" if n_trades else "—",
    )
    col4.metric("Net Profit", f"${net:+,.0f}")

    # --- Filters ---
    st.subheader("Filters")
    filter_cols = st.columns(3)

    with filter_cols[0]:
        outcome_filter = st.selectbox(
            "Outcome",
            ["All", "Winners", "Losers"],
            key="explorer_outcome",
        )
    with filter_cols[1]:
        exit_reasons = sorted(trades_df["exit_reason"].unique()) if "exit_reason" in trades_df.columns else []
        exit_filter = st.multiselect(
            "Exit Reason",
            exit_reasons,
            default=exit_reasons,
            key="explorer_exit",
        )
    with filter_cols[2]:
        contract_types = sorted(trades_df["contract_type"].unique()) if "contract_type" in trades_df.columns else []
        type_filter = st.multiselect(
            "Contract Type",
            contract_types,
            default=contract_types,
            key="explorer_ctype",
        )

    # Apply filters
    mask = pd.Series(True, index=trades_df.index)
    if outcome_filter == "Winners" and "is_winner" in trades_df.columns:
        mask &= trades_df["is_winner"].astype(bool)
    elif outcome_filter == "Losers" and "is_winner" in trades_df.columns:
        mask &= ~trades_df["is_winner"].astype(bool)
    if exit_filter and "exit_reason" in trades_df.columns:
        mask &= trades_df["exit_reason"].isin(exit_filter)
    if type_filter and "contract_type" in trades_df.columns:
        mask &= trades_df["contract_type"].isin(type_filter)

    display_df = trades_df[mask].copy()

    # --- Display columns ---
    show_cols = [
        c for c in [
            "trade_id", "contract_symbol", "contract_type",
            "entry_time", "exit_time", "time_in_trade_minutes",
            "entry_price_per_share", "exit_price_per_share",
            "num_contracts", "profit_loss_usd", "profit_loss_pct",
            "exit_reason", "confidence",
        ]
        if c in display_df.columns
    ]

    if "profit_loss_usd" in display_df.columns:
        display_df = display_df.sort_values("profit_loss_usd", ascending=False)

    st.dataframe(
        display_df[show_cols] if show_cols else display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {len(display_df)} of {n_trades} trades")

    # --- P&L distribution histogram ---
    if "profit_loss_usd" in display_df.columns and len(display_df) > 1 and _PLOTLY_AVAILABLE:
        st.subheader("P&L Distribution")
        fig = px.histogram(
            display_df,
            x="profit_loss_usd",
            nbins=30,
            labels={"profit_loss_usd": "P&L (USD)"},
            color_discrete_sequence=["#1f77b4"],
            height=340,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ML Model Comparison",
        page_icon="📊",
        layout="wide",
    )

    st.title("ML Model Comparison Dashboard")

    results_dir = _get_results_dir()
    dir_path = Path(results_dir)

    # Sidebar: data directory status
    with st.sidebar:
        st.header("Data Directory")
        st.code(str(dir_path.resolve()))
        if dir_path.exists():
            json_files = list(dir_path.glob("*_results.json"))
            st.success(f"{len(json_files)} model result(s) found")
            for f in sorted(json_files):
                st.caption(f"• {f.name}")
        else:
            st.warning("Directory not found")
            st.info(
                "Generate results with:\n"
                "```python\n"
                "comparator.save_results(\n"
                f'    "{results_dir}"\n'
                ")\n"
                "```"
            )

        st.divider()
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

        if not _PLOTLY_AVAILABLE:
            st.warning(
                "Plotly not installed — charts disabled. "
                "Install with: `pip install plotly`"
            )

    # Load data
    all_results = _load_results_dir(results_dir)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Model Comparison",
            "Threshold Sweep",
            "Signal Overlap",
            "Trade Explorer",
        ]
    )

    with tab1:
        _tab_model_comparison(results_dir, all_results)

    with tab2:
        _tab_threshold_sweep(all_results)

    with tab3:
        _tab_signal_overlap(results_dir)

    with tab4:
        _tab_trade_explorer(all_results)


if __name__ == "__main__":
    main()
