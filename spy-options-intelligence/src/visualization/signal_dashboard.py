# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Interactive HTML signal analysis dashboard.

Renders a single self-contained HTML file with five sections:
  1. Header  — run metadata (date range, threshold, model names)
  2. Summary — KPIs: total signals, precision, avg confidence, straddle P&L stats
  3. Charts  — confidence distribution, magnitude breakdown, P&L timeline, daily count
  4. Signal table — sortable/filterable table of all signals
  5. Signal details — expandable per-signal card with explanation + straddle simulation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger()

# ── Colour palette (dark theme) ────────────────────────────────────────────────
_CALL_COLOUR   = "#00d4a3"   # teal-green
_PUT_COLOUR    = "#ff6b6b"   # coral-red
_TP_COLOUR     = "#4ecdc4"   # cyan
_FP_COLOUR     = "#ff6b6b"   # red
_NEUTRAL       = "#a8a8b3"   # light grey
_BG_DARK       = "#0d1117"
_PANEL_DARK    = "#161b22"
_BORDER        = "#30363d"


class SignalDashboard:
    """Build an interactive Plotly dashboard from extracted signal data.

    Parameters
    ----------
    lgbm_model_name:
        Display name for the LightGBM model (shown in header).
    rf_model_name:
        Display name for the RandomForest model (shown in header).
    threshold:
        AVG-probability threshold that was used to fire signals.
    """

    def __init__(
        self,
        lgbm_model_name: str = "lgbm_deep_opt",
        rf_model_name: str   = "rf_deep_opt",
        threshold: float     = 0.97,
    ) -> None:
        self.lgbm_model_name = lgbm_model_name
        self.rf_model_name   = rf_model_name
        self.threshold       = threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build_dashboard(
        self,
        signals_df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """Build and write a standalone HTML dashboard.

        Parameters
        ----------
        signals_df:
            Output of ``SignalExtractor.extract_all_signals()``.
        output_path:
            File path for the generated HTML.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("  BUILDING SIGNAL ANALYSIS DASHBOARD")
        print("=" * 80)

        if len(signals_df) == 0:
            print("  No signals — writing empty dashboard.")
            html = self._empty_html()
            output_path.write_text(html, encoding="utf-8")
            return

        # Pre-compute summary metrics
        summary = self._compute_summary(signals_df)
        print(f"  Signals : {summary['n_signals']:,}")
        print(f"  Precision: {summary['precision']:.1%}  "
              f"({summary['n_tp']} TP / {summary['n_fp']} FP)")
        print(f"  Date range: {summary['date_min']} → {summary['date_max']}")

        # Build each section as HTML string
        header_html  = self._build_header(summary)
        summary_html = self._build_summary(summary)
        charts_html  = self._build_charts(signals_df, summary)
        table_html   = self._build_signal_table(signals_df)
        details_html = self._build_signal_details(signals_df)

        full_html = self._assemble_html(
            header_html, summary_html, charts_html, table_html, details_html
        )

        output_path.write_text(full_html, encoding="utf-8")
        size_kb = output_path.stat().st_size / 1024
        print(f"\n  Dashboard saved → {output_path}  ({size_kb:.0f} KB)")
        logger.info(
            "SignalDashboard: saved %d signals to %s (%.0f KB)",
            len(signals_df), output_path, size_kb,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Section builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_header(self, summary: Dict[str, Any]) -> str:
        return f"""
<div class="section header-section">
  <h1 class="dashboard-title">SPY Options — Signal Analysis Dashboard</h1>
  <div class="header-meta">
    <span class="badge">LightGBM: {self.lgbm_model_name}</span>
    <span class="badge">RF: {self.rf_model_name}</span>
    <span class="badge threshold-badge">AVG threshold ≥ {self.threshold:.0%}</span>
    <span class="badge date-badge">{summary['date_min']} → {summary['date_max']}</span>
    <span class="badge">{summary['n_signals']:,} signals</span>
  </div>
</div>"""

    def _build_summary(self, summary: Dict[str, Any]) -> str:
        avg_straddle = summary.get("avg_straddle_return", float("nan"))
        total_profit = summary.get("total_straddle_profit", 0.0)
        straddle_str = (
            f"{avg_straddle:+.1f}%" if not np.isnan(avg_straddle) else "N/A"
        )
        profit_str   = f"${total_profit:,.0f}"
        win_rate_str = f"{summary['win_rate']:.1%}" if summary['n_signals'] > 0 else "—"

        kpis = [
            ("Total Signals",   f"{summary['n_signals']:,}",       "kpi-neutral"),
            ("Precision",       f"{summary['precision']:.1%}",     "kpi-good" if summary['precision'] >= 0.95 else "kpi-warn"),
            ("Avg Confidence",  f"{summary['avg_confidence']:.1%}", "kpi-neutral"),
            ("Avg Magnitude",   f"{summary['avg_magnitude']:.1f}%", "kpi-good"),
            ("Straddle Return", straddle_str,                       "kpi-good" if not np.isnan(avg_straddle) and avg_straddle > 0 else "kpi-bad"),
            ("Total P&L (sim)", profit_str,                        "kpi-good" if total_profit > 0 else "kpi-bad"),
            ("Win Rate",        win_rate_str,                       "kpi-neutral"),
            ("CALLs / PUTs",    f"{summary['n_calls']} / {summary['n_puts']}", "kpi-neutral"),
        ]

        cards = "".join(
            f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>'
            for lbl, val, cls in kpis
        )
        return f'<div class="section"><div class="kpi-row">{cards}</div></div>'

    def _build_charts(self, df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        plots_html = []

        # ── Row 1: Confidence distribution + Magnitude distribution ───────────
        plots_html.append(self._chart_confidence_vs_precision(df))
        plots_html.append(self._chart_magnitude_breakdown(df))

        # ── Row 2: Daily signal timeline + Straddle P&L ──────────────────────
        plots_html.append(self._chart_daily_timeline(df))
        plots_html.append(self._chart_straddle_pnl(df))

        grid = "".join(
            f'<div class="chart-cell">{h}</div>' for h in plots_html
        )
        return f'<div class="section"><h2>Performance Charts</h2><div class="chart-grid">{grid}</div></div>'

    def _build_signal_table(self, df: pd.DataFrame) -> str:
        """Sortable HTML table of all signals."""
        cols = [
            "signal_id", "date", "time_et", "ticker", "contract_type",
            "strike", "expiry", "entry_price", "implied_volatility",
            "lgbm_confidence", "rf_confidence", "avg_confidence",
            "target_magnitude", "abs_max_move_pct", "move_direction",
            "magnitude_bucket", "straddle_return_pct",
        ]
        # Only keep columns that exist
        cols = [c for c in cols if c in df.columns]

        headers = "".join(f"<th onclick=\"sortTable(this)\">{c}</th>" for c in cols)

        def _fmt(val, col: str) -> str:
            if pd.isna(val):
                return "<td>—</td>"
            if col in ("lgbm_confidence", "rf_confidence", "avg_confidence", "implied_volatility"):
                return f"<td>{float(val):.1%}</td>"
            if col in ("entry_price", "abs_max_move_pct", "straddle_return_pct"):
                return f"<td>{float(val):.2f}</td>"
            if col == "target_magnitude":
                cls = "tp" if int(val) == 1 else "fp"
                label = "TP" if int(val) == 1 else "FP"
                return f'<td class="{cls}">{label}</td>'
            return f"<td>{val}</td>"

        rows = []
        for _, row in df[cols].iterrows():
            cells = "".join(_fmt(row[c], c) for c in cols)
            rows.append(f"<tr>{cells}</tr>")

        rows_html = "\n".join(rows)
        return f"""
<div class="section">
  <h2>All Signals ({len(df):,})</h2>
  <div class="table-wrapper">
    <table id="signal-table" class="signal-table">
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    def _build_signal_details(self, df: pd.DataFrame) -> str:
        """Collapsible cards for up to 100 signals (top by confidence)."""
        top = df.nlargest(100, "avg_confidence")
        cards = []
        for _, row in top.iterrows():
            is_tp  = int(row.get("target_magnitude", -1)) == 1
            outcome_cls  = "outcome-tp" if is_tp else "outcome-fp"
            outcome_text = "✅ TRUE POSITIVE" if is_tp else "❌ FALSE POSITIVE"
            abs_move  = row.get("abs_max_move_pct", float("nan"))
            move_str  = f"{abs_move:.1f}%" if not pd.isna(abs_move) else "N/A"
            direction = str(row.get("move_direction", "flat")).upper()
            straddle_ret = row.get("straddle_return_pct", float("nan"))
            straddle_str = f"{straddle_ret:+.1f}%" if not pd.isna(straddle_ret) else "N/A"
            straddle_profit = row.get("straddle_profit", float("nan"))
            profit_str = f"${straddle_profit:+,.0f}" if not pd.isna(straddle_profit) else "N/A"
            explanation = str(row.get("explanation", ""))

            card_id = f"signal_{int(row.get('signal_id', 0))}"
            cards.append(f"""
<details class="signal-card" id="{card_id}">
  <summary class="signal-card-summary">
    <span class="sig-date">{row.get('date','')} {row.get('time_et','')}</span>
    <span class="sig-ticker">{row.get('ticker','')}</span>
    <span class="sig-type type-{str(row.get('contract_type','')).lower()}">{row.get('contract_type','')}</span>
    <span class="sig-conf">avg={float(row.get('avg_confidence',0)):.1%}</span>
    <span class="{outcome_cls}">{outcome_text}</span>
  </summary>
  <div class="signal-card-body">
    <div class="signal-meta-grid">
      <div><strong>Strike</strong><br>{row.get('strike','N/A')}</div>
      <div><strong>Expiry</strong><br>{row.get('expiry','N/A')}</div>
      <div><strong>Entry Price</strong><br>${float(row.get('entry_price',0)):.2f}</div>
      <div><strong>IV</strong><br>{float(row.get('implied_volatility', float('nan'))):.1%}</div>
      <div><strong>Abs Move</strong><br>{move_str}</div>
      <div><strong>Direction</strong><br>{direction}</div>
      <div><strong>Magnitude Bucket</strong><br>{row.get('magnitude_bucket','?')}</div>
      <div><strong>Straddle Return</strong><br>{straddle_str}</div>
      <div><strong>Straddle Profit</strong><br>{profit_str}</div>
      <div><strong>LightGBM Conf</strong><br>{float(row.get('lgbm_confidence',0)):.1%}</div>
      <div><strong>RF Conf</strong><br>{float(row.get('rf_confidence',0)):.1%}</div>
      <div><strong>Time to Max</strong><br>{row.get('time_to_max_min','?')} min</div>
    </div>
    <div class="signal-explanation">{explanation}</div>
  </div>
</details>""")

        cards_html = "\n".join(cards)
        return f"""
<div class="section">
  <h2>Signal Details (top 100 by confidence)</h2>
  <div class="details-container">{cards_html}</div>
</div>"""

    # ──────────────────────────────────────────────────────────────────────────
    # Chart helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _chart_confidence_vs_precision(self, df: pd.DataFrame) -> str:
        """Scatter: avg_confidence vs abs_max_move_pct, coloured by TP/FP."""
        tp_mask = df["target_magnitude"] == 1
        fp_mask = ~tp_mask

        fig = go.Figure()
        if tp_mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[tp_mask, "avg_confidence"],
                y=df.loc[tp_mask, "abs_max_move_pct"],
                mode="markers",
                name="TP",
                marker=dict(color=_TP_COLOUR, size=6, opacity=0.7),
                text=df.loc[tp_mask, "ticker"],
            ))
        if fp_mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[fp_mask, "avg_confidence"],
                y=df.loc[fp_mask, "abs_max_move_pct"],
                mode="markers",
                name="FP",
                marker=dict(color=_FP_COLOUR, size=6, opacity=0.7),
                text=df.loc[fp_mask, "ticker"],
            ))

        fig.update_layout(
            title="Confidence vs. Magnitude",
            xaxis_title="Avg Confidence",
            yaxis_title="Abs Max Move (%)",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _chart_magnitude_breakdown(self, df: pd.DataFrame) -> str:
        """Bar chart: signal count by magnitude_bucket, split TP/FP."""
        if "magnitude_bucket" not in df.columns:
            return "<p>No magnitude_bucket column.</p>"

        bucket_order = ["0-5%", "5-10%", "10-20%", "20-30%", "30%+", "unknown"]
        buckets = [b for b in bucket_order if b in df["magnitude_bucket"].unique()]

        tp_counts = [
            int((df.loc[df["magnitude_bucket"] == b, "target_magnitude"] == 1).sum())
            for b in buckets
        ]
        fp_counts = [
            int((df.loc[df["magnitude_bucket"] == b, "target_magnitude"] != 1).sum())
            for b in buckets
        ]

        fig = go.Figure(data=[
            go.Bar(name="TP", x=buckets, y=tp_counts, marker_color=_TP_COLOUR),
            go.Bar(name="FP", x=buckets, y=fp_counts, marker_color=_FP_COLOUR),
        ])
        fig.update_layout(
            barmode="stack",
            title="Signals by Magnitude Bucket",
            xaxis_title="Magnitude Bucket",
            yaxis_title="Signal Count",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _chart_daily_timeline(self, df: pd.DataFrame) -> str:
        """Line chart: daily signal count with TP/FP stacked areas."""
        if "date" not in df.columns:
            return "<p>No date column.</p>"

        daily = (
            df.groupby(["date", "target_magnitude"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        daily.columns = [str(c) for c in daily.columns]

        fig = go.Figure()
        if "1" in daily.columns:
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["1"],
                mode="lines+markers", name="TP",
                line=dict(color=_TP_COLOUR, width=2),
                fill="tozeroy", fillcolor="rgba(78,205,196,0.15)",
            ))
        if "0" in daily.columns:
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["0"],
                mode="lines+markers", name="FP",
                line=dict(color=_FP_COLOUR, width=2),
                fill="tozeroy", fillcolor="rgba(255,107,107,0.15)",
            ))

        fig.update_layout(
            title="Daily Signal Count",
            xaxis_title="Date",
            yaxis_title="Signals",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _chart_straddle_pnl(self, df: pd.DataFrame) -> str:
        """Histogram: straddle_return_pct distribution."""
        if "straddle_return_pct" not in df.columns:
            return "<p>No straddle_return_pct column.</p>"

        valid = df["straddle_return_pct"].dropna()
        if len(valid) == 0:
            return "<p>No straddle return data.</p>"

        tp_mask = df.loc[df["straddle_return_pct"].notna(), "target_magnitude"] == 1
        tp_returns = df.loc[df["target_magnitude"] == 1, "straddle_return_pct"].dropna()
        fp_returns = df.loc[df["target_magnitude"] != 1, "straddle_return_pct"].dropna()

        fig = go.Figure()
        if len(tp_returns):
            fig.add_trace(go.Histogram(
                x=tp_returns, name="TP",
                marker_color=_TP_COLOUR, opacity=0.75,
                nbinsx=40,
            ))
        if len(fp_returns):
            fig.add_trace(go.Histogram(
                x=fp_returns, name="FP",
                marker_color=_FP_COLOUR, opacity=0.75,
                nbinsx=20,
            ))

        fig.update_layout(
            barmode="overlay",
            title="Straddle Return Distribution (%)",
            xaxis_title="Return (%)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_signals   = len(df)
        n_tp        = int((df["target_magnitude"] == 1).sum())
        n_fp        = n_signals - n_tp
        precision   = n_tp / n_signals if n_signals else 0.0
        avg_conf    = float(df["avg_confidence"].mean()) if "avg_confidence" in df else float("nan")
        avg_mag     = float(df["abs_max_move_pct"].mean()) if "abs_max_move_pct" in df else float("nan")
        date_min    = str(df["date"].min()) if "date" in df.columns else "?"
        date_max    = str(df["date"].max()) if "date" in df.columns else "?"
        n_calls     = int((df["contract_type"] == "CALL").sum()) if "contract_type" in df else 0
        n_puts      = int((df["contract_type"] == "PUT").sum())  if "contract_type" in df else 0

        avg_straddle    = float(df["straddle_return_pct"].mean())   if "straddle_return_pct" in df else float("nan")
        total_profit    = float(df["straddle_profit"].sum())         if "straddle_profit" in df else 0.0
        win_rate        = float((df["straddle_return_pct"] > 0).mean()) if "straddle_return_pct" in df else float("nan")

        return dict(
            n_signals=n_signals, n_tp=n_tp, n_fp=n_fp,
            precision=precision, avg_confidence=avg_conf,
            avg_magnitude=avg_mag, date_min=date_min, date_max=date_max,
            n_calls=n_calls, n_puts=n_puts,
            avg_straddle_return=avg_straddle,
            total_straddle_profit=total_profit,
            win_rate=win_rate,
        )

    def _assemble_html(
        self,
        header: str,
        summary: str,
        charts: str,
        table: str,
        details: str,
    ) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPY Options Signal Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {_BG_DARK}; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }}
  .section {{ padding: 20px 24px; border-bottom: 1px solid {_BORDER}; }}
  .header-section {{ background: {_PANEL_DARK}; padding: 28px 24px 20px; }}
  .dashboard-title {{ font-size: 22px; font-weight: 700; color: #e6edf3; margin-bottom: 12px; }}
  .header-meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .badge {{ background: #21262d; border: 1px solid {_BORDER}; border-radius: 4px; padding: 4px 10px; font-size: 12px; color: {_NEUTRAL}; }}
  .threshold-badge {{ color: {_CALL_COLOUR}; border-color: {_CALL_COLOUR}; }}
  .date-badge {{ color: #79c0ff; border-color: #79c0ff; }}

  /* KPI row */
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .kpi-card {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 16px 20px; min-width: 120px; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
  .kpi-lbl {{ font-size: 11px; color: {_NEUTRAL}; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-good .kpi-val {{ color: {_CALL_COLOUR}; }}
  .kpi-bad  .kpi-val {{ color: {_FP_COLOUR}; }}
  .kpi-warn .kpi-val {{ color: #f0a500; }}
  .kpi-neutral .kpi-val {{ color: #e6edf3; }}

  /* Charts */
  h2 {{ font-size: 16px; font-weight: 600; color: #e6edf3; margin-bottom: 16px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart-cell {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 12px; }}

  /* Table */
  .table-wrapper {{ overflow-x: auto; }}
  .signal-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .signal-table th, .signal-table td {{ padding: 6px 10px; border: 1px solid {_BORDER}; white-space: nowrap; }}
  .signal-table th {{ background: {_PANEL_DARK}; cursor: pointer; user-select: none; font-weight: 600; color: {_NEUTRAL}; }}
  .signal-table th:hover {{ color: #e6edf3; }}
  .signal-table tr:nth-child(even) {{ background: #0d1117; }}
  .signal-table tr:hover {{ background: #21262d; }}
  .tp {{ color: {_TP_COLOUR}; font-weight: 600; }}
  .fp {{ color: {_FP_COLOUR}; font-weight: 600; }}

  /* Signal detail cards */
  .details-container {{ display: flex; flex-direction: column; gap: 10px; }}
  .signal-card {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; }}
  .signal-card-summary {{ padding: 12px 16px; cursor: pointer; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; list-style: none; }}
  .signal-card-summary::-webkit-details-marker {{ display: none; }}
  .signal-card[open] .signal-card-summary {{ border-bottom: 1px solid {_BORDER}; }}
  .signal-card-body {{ padding: 16px; }}
  .sig-date  {{ color: {_NEUTRAL}; font-size: 12px; min-width: 140px; }}
  .sig-ticker {{ font-family: monospace; font-size: 12px; color: #79c0ff; }}
  .sig-type  {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
  .type-call {{ background: rgba(0,212,163,0.2); color: {_CALL_COLOUR}; }}
  .type-put  {{ background: rgba(255,107,107,0.2); color: {_PUT_COLOUR}; }}
  .sig-conf  {{ font-size: 12px; color: {_NEUTRAL}; }}
  .outcome-tp {{ padding: 2px 8px; border-radius: 3px; background: rgba(78,205,196,0.15); color: {_TP_COLOUR}; font-size: 11px; }}
  .outcome-fp {{ padding: 2px 8px; border-radius: 3px; background: rgba(255,107,107,0.15); color: {_FP_COLOUR}; font-size: 11px; }}
  .signal-meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-bottom: 14px; }}
  .signal-meta-grid div {{ font-size: 12px; }}
  .signal-meta-grid strong {{ display: block; color: {_NEUTRAL}; font-size: 11px; margin-bottom: 2px; }}
  .signal-explanation {{ font-size: 13px; color: #a8b8d8; line-height: 1.6; padding: 12px; background: #0d1117; border-radius: 4px; border-left: 3px solid {_CALL_COLOUR}; }}
</style>
</head>
<body>
{header}
{summary}
{charts}
{table}
{details}
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const colIdx = Array.from(th.parentRow ? th.parentRow.cells : th.parentElement.cells).indexOf(th);
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.rows);
  const asc = th.dataset.sort !== 'asc';
  rows.sort((a, b) => {{
    const va = a.cells[colIdx]?.textContent.trim() ?? '';
    const vb = b.cells[colIdx]?.textContent.trim() ?? '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
  th.dataset.sort = asc ? 'asc' : 'desc';
}}
</script>
</body>
</html>"""

    def _empty_html(self) -> str:
        return (
            "<!DOCTYPE html><html><body style='background:#0d1117;color:#e6edf3;"
            "font-family:sans-serif;padding:40px'>"
            "<h1>SPY Options Signal Dashboard</h1>"
            "<p>No signals found at the configured threshold.</p></body></html>"
        )
