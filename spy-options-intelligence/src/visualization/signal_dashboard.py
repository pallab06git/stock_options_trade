# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Interactive HTML signal analysis dashboard with money flow analysis.

Renders a single self-contained HTML file with six sections:
  1. Header      — run metadata (date range, threshold, model names)
  2. Summary     — KPIs: signals, precision, direction breakdown, single-option P&L
  3. Insight box — plain-English interpretation of direction vs magnitude
  4. Charts      — confidence, magnitude, daily timeline, straddle P&L, cumulative P&L
  5. Signal table — sortable/filterable table of all signals
  6. Signal details — expandable per-signal card with money flow + straddle simulation
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
_UP_COLOUR     = "#00d4a3"   # green for UP
_DOWN_COLOUR   = "#ff6b6b"   # red for DOWN
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
        print(f"  Options went UP: {summary['up_count']} ({summary['up_pct']:.1%})  "
              f"DOWN: {summary['down_count']} ({summary['down_pct']:.1%})")
        print(f"  Single-option win rate: {summary['single_win_rate']:.1%}  "
              f"total P&L: ${summary['single_total_profit']:,.0f}")
        print(f"  Date range: {summary['date_min']} → {summary['date_max']}")

        # Build each section as HTML string
        header_html  = self._build_header(summary)
        summary_html = self._build_summary(summary, signals_df)
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

    def _build_summary(self, summary: Dict[str, Any], df: pd.DataFrame) -> str:
        avg_straddle = summary.get("avg_straddle_return", float("nan"))
        total_straddle = summary.get("total_straddle_profit", 0.0)
        straddle_str = f"{avg_straddle:+.1f}%" if not np.isnan(avg_straddle) else "N/A"

        single_win   = summary.get("single_win_rate", float("nan"))
        single_pnl   = summary.get("single_total_profit", 0.0)
        single_avg   = summary.get("single_avg_profit", float("nan"))
        up_count     = summary.get("up_count", 0)
        down_count   = summary.get("down_count", 0)
        n            = summary["n_signals"]
        up_pct       = summary.get("up_pct", 0.0)
        down_pct     = summary.get("down_pct", 0.0)

        single_win_str  = f"{single_win:.1%}"  if not np.isnan(single_win)  else "—"
        single_avg_str  = f"${single_avg:,.0f}" if not np.isnan(single_avg) else "—"

        kpis = [
            # Row 1: Core model stats
            ("Total Signals",         f"{n:,}",                                  "kpi-neutral"),
            ("Precision",             f"{summary['precision']:.1%}",             "kpi-good" if summary['precision'] >= 0.95 else "kpi-warn"),
            ("Avg Confidence",        f"{summary['avg_confidence']:.1%}",        "kpi-neutral"),
            ("Avg Magnitude",         f"{summary['avg_magnitude']:.1f}%",        "kpi-good"),
            # Row 2: Direction breakdown
            ("Options Went UP ↑",     f"{up_count} ({up_pct:.1%})",             "kpi-good"),
            ("Options Went DOWN ↓",   f"{down_count} ({down_pct:.1%})",          "kpi-bad" if down_pct > 0.3 else "kpi-neutral"),
            # Row 3: Single-option strategy
            ("Single-Option Win Rate", single_win_str,                           "kpi-good" if not np.isnan(single_win) and single_win >= 0.7 else "kpi-warn"),
            ("Single-Option Avg P&L",  single_avg_str,                           "kpi-good" if not np.isnan(single_avg) and single_avg > 0 else "kpi-bad"),
            ("Single-Option Total",    f"${single_pnl:,.0f}",                   "kpi-good" if single_pnl > 0 else "kpi-bad"),
            # Row 4: Straddle strategy
            ("Straddle Avg Return",    straddle_str,                             "kpi-good" if not np.isnan(avg_straddle) and avg_straddle > 0 else "kpi-bad"),
            ("Straddle Total P&L",     f"${total_straddle:,.0f}",               "kpi-good" if total_straddle > 0 else "kpi-bad"),
            ("CALLs / PUTs",           f"{summary['n_calls']} / {summary['n_puts']}", "kpi-neutral"),
        ]

        cards = "".join(
            f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>'
            for lbl, val, cls in kpis
        )
        insight_html = self._generate_direction_insight(up_count, down_count, n, single_win, single_pnl)

        return f"""
<div class="section">
  <div class="kpi-row">{cards}</div>
  {insight_html}
</div>"""

    def _generate_direction_insight(
        self,
        up_count: int,
        down_count: int,
        total: int,
        single_win_rate: float,
        single_total_profit: float,
    ) -> str:
        """Generate the insight box explaining what the direction split means."""
        if total == 0:
            return ""

        up_pct   = up_count / total * 100
        win_str  = f"{single_win_rate:.1%}" if not np.isnan(single_win_rate) else "N/A"
        pnl_str  = f"${single_total_profit:,.0f}"

        if up_pct > 90:
            direction_verdict = (
                f'<p class="insight-positive">✅ <strong>Model IS predicting direction (UP)!</strong> '
                f'{up_pct:.0f}% of signalled options went UP. This is NOT merely magnitude prediction — '
                f'it is directional prediction with high accuracy. Buying just the CALL/PUT that fired '
                f'(not a straddle) captures this directional edge directly.</p>'
            )
        elif up_pct < 10:
            direction_verdict = (
                f'<p class="insight-warn">⚠️ <strong>Model IS predicting direction (DOWN)!</strong> '
                f'Only {up_pct:.0f}% went UP ({100-up_pct:.0f}% went DOWN). '
                f'Strong downward directional bias.</p>'
            )
        elif 40 <= up_pct <= 60:
            direction_verdict = (
                f'<p class="insight-neutral">⚪ <strong>Model is NOT predicting direction.</strong> '
                f'{up_pct:.0f}% UP / {100-up_pct:.0f}% DOWN — near-50/50 split. '
                f'The model is predicting magnitude (20%+ move) but not direction. '
                f'A straddle is the appropriate strategy.</p>'
            )
        else:
            direction_verdict = (
                f'<p class="insight-warn">⚠️ <strong>Model has a directional bias.</strong> '
                f'{up_pct:.0f}% went UP suggests partial direction prediction — '
                f'not pure magnitude. Consider directional sizing.</p>'
            )

        return f"""
<div class="insight-box">
  <h3>🔍 Key Insight: Is the Model Predicting Direction?</h3>
  <p>Out of <strong>{total}</strong> signals at this threshold:</p>
  <ul>
    <li><strong style="color:{_UP_COLOUR}">{up_count} ({up_pct:.1f}%)</strong> options went <strong>UP</strong></li>
    <li><strong style="color:{_DOWN_COLOUR}">{down_count} ({100-up_pct:.1f}%)</strong> options went <strong>DOWN</strong></li>
  </ul>
  {direction_verdict}
  <p style="margin-top:10px">
    <strong>Single-option strategy</strong> (buy just the option the model signalled, 1 contract = 100 shares):
    win rate = <strong>{win_str}</strong> | total profit = <strong>{pnl_str}</strong>
  </p>
  <p style="margin-top:6px; font-size:12px; color:{_NEUTRAL}">
    <em>Note: "100% precision" means every signal bar had a ≥20% max-move within 120 minutes.
    It does not predict which direction. The UP/DOWN split above reveals whether the model
    has a hidden directional bias.</em>
  </p>
</div>"""

    def _build_charts(self, df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        plots_html = []

        # ── Row 1: Confidence vs magnitude + Magnitude bucket breakdown ────────
        plots_html.append(self._chart_confidence_vs_precision(df))
        plots_html.append(self._chart_magnitude_breakdown(df))

        # ── Row 2: Daily signal timeline + Straddle P&L distribution ─────────
        plots_html.append(self._chart_daily_timeline(df))
        plots_html.append(self._chart_straddle_pnl(df))

        # ── Row 3: Direction breakdown + Single-option P&L distribution ───────
        plots_html.append(self._chart_direction_breakdown(df))
        plots_html.append(self._chart_single_option_pnl(df))

        grid = "".join(
            f'<div class="chart-cell">{h}</div>' for h in plots_html
        )
        # Full-width cumulative P&L chart below the grid
        cumulative_html = self._build_cumulative_chart(df)

        return (
            f'<div class="section">'
            f'<h2>Performance Charts</h2>'
            f'<div class="chart-grid">{grid}</div>'
            f'<div class="chart-full">{cumulative_html}</div>'
            f'</div>'
        )

    def _build_signal_table(self, df: pd.DataFrame) -> str:
        """Sortable HTML table of all signals."""
        cols = [
            "signal_id", "date", "time_et", "ticker", "contract_type",
            "strike", "expiry", "entry_price", "implied_volatility",
            "lgbm_confidence", "rf_confidence", "avg_confidence",
            "target_magnitude", "price_direction", "abs_max_move_pct",
            "move_direction", "magnitude_bucket",
            "single_option_pnl_dollars", "single_option_pnl_pct",
            "straddle_return_pct", "cumulative_pnl",
        ]
        cols = [c for c in cols if c in df.columns]

        headers = "".join(f"<th onclick=\"sortTable(this)\">{c}</th>" for c in cols)

        def _fmt(val, col: str) -> str:
            if pd.isna(val):
                return "<td>—</td>"
            if col in ("lgbm_confidence", "rf_confidence", "avg_confidence", "implied_volatility"):
                return f"<td>{float(val):.1%}</td>"
            if col in ("entry_price", "abs_max_move_pct", "straddle_return_pct", "single_option_pnl_pct"):
                return f"<td>{float(val):.2f}</td>"
            if col in ("single_option_pnl_dollars", "cumulative_pnl"):
                v = float(val)
                cls = "tp" if v >= 0 else "fp"
                return f'<td class="{cls}">${v:+,.0f}</td>'
            if col == "target_magnitude":
                cls = "tp" if int(val) == 1 else "fp"
                label = "TP" if int(val) == 1 else "FP"
                return f'<td class="{cls}">{label}</td>'
            if col == "price_direction":
                cls = "tp" if str(val) == "UP" else ("fp" if str(val) == "DOWN" else "")
                return f'<td class="{cls}">{val}</td>'
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

            abs_move      = row.get("abs_max_move_pct", float("nan"))
            move_str      = f"{abs_move:.1f}%" if not pd.isna(abs_move) else "N/A"
            direction     = str(row.get("move_direction", "flat")).upper()
            straddle_ret  = row.get("straddle_return_pct", float("nan"))
            straddle_str  = f"{straddle_ret:+.1f}%" if not pd.isna(straddle_ret) else "N/A"
            straddle_profit = row.get("straddle_profit", float("nan"))
            s_profit_str  = f"${straddle_profit:+,.0f}" if not pd.isna(straddle_profit) else "N/A"
            explanation   = str(row.get("explanation", ""))

            # ── Money flow fields ──────────────────────────────────────────────
            price_dir     = str(row.get("price_direction", "unknown"))
            price_chg     = row.get("price_change_pct", float("nan"))
            exit_price    = row.get("exit_price", float("nan"))
            max_gain      = row.get("max_gain_pct", float("nan"))
            max_loss      = row.get("max_loss_pct", float("nan"))
            t_exit        = row.get("time_to_exit", 0)
            so_cost       = row.get("single_option_entry_cost", float("nan"))
            so_exit       = row.get("single_option_exit_value", float("nan"))
            so_pnl        = row.get("single_option_pnl_dollars", float("nan"))
            so_pct        = row.get("single_option_pnl_pct", float("nan"))
            so_win        = bool(row.get("single_option_profitable", False))
            cum_pnl       = row.get("cumulative_pnl", float("nan"))

            price_dir_cls = "money-up" if price_dir == "UP" else ("money-down" if price_dir == "DOWN" else "")
            so_pnl_cls    = "money-up" if so_win else "money-down"

            price_chg_str = f"{price_chg:+.1f}%" if not pd.isna(price_chg) else "N/A"
            exit_str      = f"${exit_price:.2f}" if not pd.isna(exit_price) else "N/A"
            max_gain_str  = f"+{max_gain:.1f}%" if not pd.isna(max_gain) else "N/A"
            max_loss_str  = f"{max_loss:.1f}%" if not pd.isna(max_loss) else "N/A"
            so_cost_str   = f"${so_cost:,.0f}" if not pd.isna(so_cost) else "N/A"
            so_exit_str   = f"${so_exit:,.0f}" if not pd.isna(so_exit) else "N/A"
            so_pnl_str    = f"${so_pnl:+,.0f} ({so_pct:+.1f}%)" if not pd.isna(so_pnl) else "N/A"
            cum_pnl_str   = f"${cum_pnl:+,.0f}" if not pd.isna(cum_pnl) else "N/A"

            explain_html  = self._explain_signal_meaning(row)

            card_id = f"signal_{int(row.get('signal_id', 0))}"
            cards.append(f"""
<details class="signal-card" id="{card_id}">
  <summary class="signal-card-summary">
    <span class="sig-date">{row.get('date','')} {row.get('time_et','')}</span>
    <span class="sig-ticker">{row.get('ticker','')}</span>
    <span class="sig-type type-{str(row.get('contract_type','')).lower()}">{row.get('contract_type','')}</span>
    <span class="sig-conf">avg={float(row.get('avg_confidence',0)):.1%}</span>
    <span class="{price_dir_cls} sig-direction">{price_dir}</span>
    <span class="{outcome_cls}">{outcome_text}</span>
  </summary>
  <div class="signal-card-body">

    <div class="money-section">
      <h4>💵 Single-Option Analysis (Buy This Option, 1 Contract)</h4>
      <div class="money-grid">
        <div><strong>Entry Cost</strong><br>{so_cost_str}</div>
        <div><strong>Exit Value</strong><br>{so_exit_str}</div>
        <div><strong>Direction</strong><br><span class="{price_dir_cls}">{price_dir} ({price_chg_str})</span></div>
        <div><strong>P&amp;L</strong><br><span class="{so_pnl_cls}">{so_pnl_str}</span></div>
        <div><strong>Cumulative P&amp;L</strong><br>{cum_pnl_str}</div>
        <div><strong>Exit at {t_exit} min</strong><br>{exit_str}</div>
      </div>
    </div>

    <div class="money-section">
      <h4>📊 Price Movement (Next 120 Min)</h4>
      <div class="money-grid">
        <div><strong>Max Gain</strong><br><span class="money-up">{max_gain_str}</span></div>
        <div><strong>Max Loss</strong><br><span class="money-down">{max_loss_str}</span></div>
        <div><strong>Straddle Return</strong><br>{straddle_str}</div>
        <div><strong>Straddle Profit</strong><br>{s_profit_str}</div>
        <div><strong>Abs Move</strong><br>{move_str}</div>
        <div><strong>Magnitude Bucket</strong><br>{row.get('magnitude_bucket','?')}</div>
      </div>
    </div>

    <div class="signal-meta-grid">
      <div><strong>Strike</strong><br>{row.get('strike','N/A')}</div>
      <div><strong>Expiry</strong><br>{row.get('expiry','N/A')}</div>
      <div><strong>Entry Price</strong><br>${float(row.get('entry_price',0)):.2f}</div>
      <div><strong>IV</strong><br>{float(row.get('implied_volatility', float('nan'))):.1%}</div>
      <div><strong>LightGBM Conf</strong><br>{float(row.get('lgbm_confidence',0)):.1%}</div>
      <div><strong>RF Conf</strong><br>{float(row.get('rf_confidence',0)):.1%}</div>
    </div>

    <div class="signal-explanation">{explanation}</div>

    <div class="signal-meaning">{explain_html}</div>

  </div>
</details>""")

        cards_html = "\n".join(cards)
        return f"""
<div class="section">
  <h2>Signal Details (top 100 by confidence)</h2>
  <div class="details-container">{cards_html}</div>
</div>"""

    def _explain_signal_meaning(self, row) -> str:
        """Plain-English explanation of what the signal outcome means."""
        target    = int(row.get("target_magnitude", -1))
        direction = str(row.get("price_direction", "unknown"))
        so_pnl    = row.get("single_option_pnl_dollars", float("nan"))
        so_pct    = row.get("single_option_pnl_pct", float("nan"))

        pnl_str = (
            f"${so_pnl:+,.0f} ({so_pct:+.1f}%)" if not pd.isna(so_pnl) else "unknown P&L"
        )

        if target == 1 and direction == "UP":
            return (
                f'<p class="insight-positive">✅ <strong>Correct directional prediction:</strong> '
                f'Model predicted ≥20% move, and option went <strong>UP</strong>. '
                f'Single-option P&amp;L: <strong>{pnl_str}</strong>. '
                f'The model effectively predicted direction (UP) here.</p>'
            )
        elif target == 1 and direction == "DOWN":
            return (
                f'<p class="insight-positive">✅ <strong>Correct magnitude prediction, opposite direction:</strong> '
                f'Model predicted ≥20% move ✓, but the option went <strong>DOWN</strong>. '
                f'Single-option P&amp;L: <strong>{pnl_str}</strong>. '
                f'A straddle would have captured this move; a directional bet would have lost.</p>'
            )
        elif target == 1 and direction == "FLAT":
            return (
                f'<p class="insight-warn">⚠️ <strong>Target=1 but option stayed flat:</strong> '
                f'The maximum move was within the 120-min window but the directional movement '
                f'appears minimal. Single-option P&amp;L: <strong>{pnl_str}</strong>.</p>'
            )
        elif target == 0:
            return (
                f'<p class="insight-negative">❌ <strong>False positive:</strong> '
                f'Model fired but no ≥20% move occurred. '
                f'Single-option P&amp;L: <strong>{pnl_str}</strong>. '
                f'Direction: {direction}.</p>'
            )
        else:
            return (
                f'<p>Target={target}, Direction={direction}, P&amp;L={pnl_str}</p>'
            )

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
        """Line chart: daily signal count."""
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

        tp_returns = df.loc[df["target_magnitude"] == 1, "straddle_return_pct"].dropna()
        fp_returns = df.loc[df["target_magnitude"] != 1, "straddle_return_pct"].dropna()

        fig = go.Figure()
        if len(tp_returns):
            fig.add_trace(go.Histogram(
                x=tp_returns, name="TP", marker_color=_TP_COLOUR, opacity=0.75, nbinsx=40,
            ))
        if len(fp_returns):
            fig.add_trace(go.Histogram(
                x=fp_returns, name="FP", marker_color=_FP_COLOUR, opacity=0.75, nbinsx=20,
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

    def _chart_direction_breakdown(self, df: pd.DataFrame) -> str:
        """Pie chart: UP vs DOWN vs FLAT price_direction."""
        if "price_direction" not in df.columns:
            return "<p>No price_direction column.</p>"

        counts = df["price_direction"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        colours = [
            _UP_COLOUR if l == "UP" else (_DOWN_COLOUR if l == "DOWN" else _NEUTRAL)
            for l in labels
        ]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colours),
            textinfo="label+percent",
            hole=0.4,
        )])
        fig.update_layout(
            title="Option Direction (UP / DOWN / FLAT)",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _chart_single_option_pnl(self, df: pd.DataFrame) -> str:
        """Histogram: single_option_pnl_dollars distribution."""
        if "single_option_pnl_dollars" not in df.columns:
            return "<p>No single_option_pnl_dollars column.</p>"

        pnl = df["single_option_pnl_dollars"].dropna()
        if len(pnl) == 0:
            return "<p>No single-option P&L data.</p>"

        colours = [_UP_COLOUR if v >= 0 else _DOWN_COLOUR for v in pnl]

        fig = go.Figure(data=[go.Histogram(
            x=pnl,
            nbinsx=50,
            marker=dict(
                color=pnl.values,
                colorscale=[[0, _DOWN_COLOUR], [0.5, "#aaaaaa"], [1, _UP_COLOUR]],
                line=dict(color=_BORDER, width=0.5),
            ),
            name="Single-Option P&L",
        )])
        fig.add_vline(x=0, line_dash="dash", line_color=_NEUTRAL, line_width=1)
        fig.update_layout(
            title="Single-Option P&L Distribution ($)",
            xaxis_title="P&L per Trade ($)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_cumulative_chart(self, df: pd.DataFrame) -> str:
        """Full-width chart: cumulative single-option P&L vs straddle P&L over time."""
        if "cumulative_pnl" not in df.columns:
            return ""

        sorted_df = df.sort_values(["date", "timestamp_ms"] if "timestamp_ms" in df.columns else ["date"])
        x_axis    = list(range(len(sorted_df)))
        dates     = sorted_df["date"].tolist()

        fig = go.Figure()

        # Single-option cumulative P&L
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=sorted_df["cumulative_pnl"].values,
            mode="lines",
            name="Single-Option Strategy",
            line=dict(color=_CALL_COLOUR, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,163,0.08)",
            text=dates,
            hovertemplate="Signal #%{x}<br>Date: %{text}<br>Cumulative P&L: $%{y:,.0f}<extra></extra>",
        ))

        # Straddle cumulative P&L (if available)
        if "straddle_profit" in sorted_df.columns:
            straddle_cum = sorted_df["straddle_profit"].fillna(0.0).cumsum().values
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=straddle_cum,
                mode="lines",
                name="Straddle Strategy",
                line=dict(color="#79c0ff", width=2, dash="dash"),
                text=dates,
                hovertemplate="Signal #%{x}<br>Date: %{text}<br>Straddle Cumulative: $%{y:,.0f}<extra></extra>",
            ))

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color=_NEUTRAL, line_width=1)

        fig.update_layout(
            title="Cumulative P&L Over Time (Signal Number)",
            xaxis_title="Signal Number (chronological)",
            yaxis_title="Cumulative Profit ($)",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            hovermode="x unified",
            height=420,
            margin=dict(l=60, r=20, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_signals = len(df)
        n_tp      = int((df["target_magnitude"] == 1).sum())
        n_fp      = n_signals - n_tp
        precision = n_tp / n_signals if n_signals else 0.0
        avg_conf  = float(df["avg_confidence"].mean()) if "avg_confidence" in df.columns else float("nan")
        avg_mag   = float(df["abs_max_move_pct"].mean()) if "abs_max_move_pct" in df.columns else float("nan")
        date_min  = str(df["date"].min()) if "date" in df.columns else "?"
        date_max  = str(df["date"].max()) if "date" in df.columns else "?"
        n_calls   = int((df["contract_type"] == "CALL").sum()) if "contract_type" in df.columns else 0
        n_puts    = int((df["contract_type"] == "PUT").sum())  if "contract_type" in df.columns else 0

        avg_straddle   = float(df["straddle_return_pct"].mean()) if "straddle_return_pct" in df.columns else float("nan")
        total_straddle = float(df["straddle_profit"].sum())       if "straddle_profit" in df.columns else 0.0
        win_rate       = float((df["straddle_return_pct"] > 0).mean()) if "straddle_return_pct" in df.columns else float("nan")

        # Direction and single-option stats
        up_count   = int((df["price_direction"] == "UP").sum())   if "price_direction" in df.columns else 0
        down_count = int((df["price_direction"] == "DOWN").sum()) if "price_direction" in df.columns else 0
        up_pct     = up_count / n_signals   if n_signals else 0.0
        down_pct   = down_count / n_signals if n_signals else 0.0

        single_win     = float((df["single_option_profitable"] == True).mean()) if "single_option_profitable" in df.columns else float("nan")
        single_pnl     = float(df["single_option_pnl_dollars"].sum())           if "single_option_pnl_dollars" in df.columns else 0.0
        single_avg     = float(df["single_option_pnl_dollars"].mean())          if "single_option_pnl_dollars" in df.columns else float("nan")

        return dict(
            n_signals=n_signals, n_tp=n_tp, n_fp=n_fp,
            precision=precision, avg_confidence=avg_conf,
            avg_magnitude=avg_mag, date_min=date_min, date_max=date_max,
            n_calls=n_calls, n_puts=n_puts,
            avg_straddle_return=avg_straddle,
            total_straddle_profit=total_straddle,
            win_rate=win_rate,
            up_count=up_count, down_count=down_count,
            up_pct=up_pct, down_pct=down_pct,
            single_win_rate=single_win,
            single_total_profit=single_pnl,
            single_avg_profit=single_avg,
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
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .kpi-card {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 16px 20px; min-width: 130px; }}
  .kpi-val {{ font-size: 20px; font-weight: 700; line-height: 1.2; }}
  .kpi-lbl {{ font-size: 11px; color: {_NEUTRAL}; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-good .kpi-val    {{ color: {_CALL_COLOUR}; }}
  .kpi-bad  .kpi-val    {{ color: {_FP_COLOUR}; }}
  .kpi-warn .kpi-val    {{ color: #f0a500; }}
  .kpi-neutral .kpi-val {{ color: #e6edf3; }}

  /* Insight box */
  .insight-box {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-left: 4px solid {_CALL_COLOUR}; border-radius: 6px; padding: 16px 20px; margin-top: 8px; }}
  .insight-box h3 {{ font-size: 14px; font-weight: 600; color: #e6edf3; margin-bottom: 10px; }}
  .insight-box ul {{ margin: 8px 0 10px 20px; font-size: 13px; }}
  .insight-box li {{ margin-bottom: 4px; }}
  .insight-positive {{ color: {_CALL_COLOUR}; font-size: 13px; line-height: 1.5; margin-top: 8px; }}
  .insight-negative  {{ color: {_FP_COLOUR};  font-size: 13px; line-height: 1.5; margin-top: 8px; }}
  .insight-warn      {{ color: #f0a500;       font-size: 13px; line-height: 1.5; margin-top: 8px; }}
  .insight-neutral   {{ color: {_NEUTRAL};    font-size: 13px; line-height: 1.5; margin-top: 8px; }}

  /* Charts */
  h2 {{ font-size: 16px; font-weight: 600; color: #e6edf3; margin-bottom: 16px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart-cell {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 12px; }}
  .chart-full {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 12px; margin-top: 16px; }}

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
  .signal-card-body {{ padding: 16px; display: flex; flex-direction: column; gap: 14px; }}
  .sig-date  {{ color: {_NEUTRAL}; font-size: 12px; min-width: 140px; }}
  .sig-ticker {{ font-family: monospace; font-size: 12px; color: #79c0ff; }}
  .sig-type  {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
  .type-call {{ background: rgba(0,212,163,0.2); color: {_CALL_COLOUR}; }}
  .type-put  {{ background: rgba(255,107,107,0.2); color: {_PUT_COLOUR}; }}
  .sig-conf  {{ font-size: 12px; color: {_NEUTRAL}; }}
  .sig-direction {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; }}
  .outcome-tp {{ padding: 2px 8px; border-radius: 3px; background: rgba(78,205,196,0.15); color: {_TP_COLOUR}; font-size: 11px; }}
  .outcome-fp {{ padding: 2px 8px; border-radius: 3px; background: rgba(255,107,107,0.15); color: {_FP_COLOUR}; font-size: 11px; }}

  /* Money flow */
  .money-section {{ background: #0d1117; border-radius: 4px; padding: 12px 14px; }}
  .money-section h4 {{ font-size: 12px; font-weight: 600; color: {_NEUTRAL}; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .money-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }}
  .money-grid div {{ font-size: 12px; }}
  .money-grid strong {{ display: block; color: {_NEUTRAL}; font-size: 10px; margin-bottom: 2px; text-transform: uppercase; }}
  .money-up   {{ color: {_UP_COLOUR}; font-weight: 600; }}
  .money-down {{ color: {_DOWN_COLOUR}; font-weight: 600; }}

  /* Meta grid */
  .signal-meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }}
  .signal-meta-grid div {{ font-size: 12px; }}
  .signal-meta-grid strong {{ display: block; color: {_NEUTRAL}; font-size: 11px; margin-bottom: 2px; }}

  /* Explanation + meaning */
  .signal-explanation {{ font-size: 13px; color: #a8b8d8; line-height: 1.6; padding: 12px; background: #0d1117; border-radius: 4px; border-left: 3px solid {_CALL_COLOUR}; }}
  .signal-meaning {{ font-size: 13px; line-height: 1.6; }}
  .signal-meaning p {{ padding: 10px 12px; background: #0d1117; border-radius: 4px; }}
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
  const colIdx = Array.from(th.parentElement.cells).indexOf(th);
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.rows);
  const asc = th.dataset.sort !== 'asc';
  rows.sort((a, b) => {{
    const va = a.cells[colIdx]?.textContent.trim() ?? '';
    const vb = b.cells[colIdx]?.textContent.trim() ?? '';
    const na = parseFloat(va.replace(/[$,+%]/g, '')),
          nb = parseFloat(vb.replace(/[$,+%]/g, ''));
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
