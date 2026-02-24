# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Interactive HTML trade dashboard for pipeline backtest results.

Generates a standalone dark-theme Plotly dashboard with:
  1. KPI cards (Total P&L, Win Rate, Avg Hold, Signals/Day)
  2. Equity curve with drawdown overlay
  3. Monthly P&L bar chart (with $10K target line)
  4. Trade scatter (date vs P&L, color=win/loss, size=confidence)
  5. Exit reason distribution + P&L by exit type
  6. Sortable trade table

Usage
-----
    from src.visualization.trade_dashboard import TradeDashboard

    dashboard = TradeDashboard()
    dashboard.build_dashboard(report_dict, "reports/trade_dashboard.html")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

# Dark theme colors (matching signal_dashboard.py)
_BG_DARK = "#0d1117"
_PANEL_DARK = "#161b22"
_BORDER = "#30363d"
_TEXT = "#e6edf3"
_NEUTRAL = "#8b949e"
_GREEN = "#00d4a3"
_RED = "#f85149"
_BLUE = "#79c0ff"
_YELLOW = "#d29922"


class TradeDashboard:
    """Build a standalone HTML trade dashboard from pipeline results."""

    def build_dashboard(
        self,
        report: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Build and save the HTML dashboard.

        Args:
            report: Pipeline report dict (from SignalPipeline or JSON file).
            output_path: Path to write the HTML file.
        """
        try:
            import plotly.graph_objects  # noqa: F401
        except ImportError:
            raise ImportError(
                "plotly is required for TradeDashboard. "
                "Install with: pip install plotly"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trades = report.get("trades", [])
        exec_summary = report.get("executive_summary", {})
        monthly_pnl = report.get("monthly_pnl", [])
        exit_analysis = report.get("exit_analysis", {})

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Build chart sections
        kpi_html = self._build_kpi_cards(exec_summary)
        equity_html = self._build_equity_curve(trades_df) if len(trades_df) > 0 else ""
        monthly_html = self._build_monthly_pnl(monthly_pnl) if monthly_pnl else ""
        scatter_html = self._build_trade_scatter(trades_df) if len(trades_df) > 0 else ""
        exit_html = self._build_exit_analysis(exit_analysis) if exit_analysis else ""
        table_html = self._build_trade_table(trades_df) if len(trades_df) > 0 else ""

        html = self._assemble_html(
            kpi_html, equity_html, monthly_html,
            scatter_html, exit_html, table_html,
            exec_summary, report,
        )

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Trade dashboard saved to {output_path}")

    def _build_kpi_cards(self, summary: Dict[str, Any]) -> str:
        """Build KPI cards section."""
        net_pnl = summary.get("net_pnl", 0)
        pnl_color = _GREEN if net_pnl >= 0 else _RED
        pnl_sign = "+" if net_pnl >= 0 else ""

        cards = [
            ("Net P&L", f"{pnl_sign}${net_pnl:,.0f}", pnl_color),
            ("Win Rate", f"{summary.get('win_rate', 0):.0%}", _GREEN if summary.get("win_rate", 0) >= 0.6 else _YELLOW),
            ("Total Trades", f"{summary.get('total_trades', 0)}", _BLUE),
            ("Avg Hold", f"{summary.get('avg_hold_minutes', 0):.0f} min", _NEUTRAL),
            ("Trades/Day", f"{summary.get('avg_trades_per_day', 0):.1f}", _NEUTRAL),
            ("Avg P&L/Trade", f"${summary.get('avg_pnl_per_trade', 0):,.0f}", _BLUE),
        ]

        html = '<div class="kpi-grid">'
        for label, value, color in cards:
            html += f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{value}</div>
            </div>"""
        html += "</div>"
        return html

    def _build_equity_curve(self, df: pd.DataFrame) -> str:
        """Build equity curve with drawdown overlay."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if "profit_loss_usd" not in df.columns:
            return ""

        pnl = df["profit_loss_usd"].fillna(0).values
        cumulative = np.cumsum(pnl)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak

        x = list(range(1, len(cumulative) + 1))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )

        fig.add_trace(go.Scatter(
            x=x, y=cumulative,
            mode="lines", name="Cumulative P&L",
            line=dict(color=_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,163,0.08)",
            hovertemplate="Trade #%{x}<br>Cumulative: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=drawdown,
            mode="lines", name="Drawdown",
            line=dict(color=_RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.1)",
            hovertemplate="Trade #%{x}<br>Drawdown: $%{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            title="Equity Curve & Drawdown",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=450,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_monthly_pnl(self, monthly: List[Dict[str, Any]]) -> str:
        """Build monthly P&L bar chart."""
        import plotly.graph_objects as go

        if not monthly:
            return ""

        months = [m["month"] for m in monthly]
        net_pnls = [m.get("net_pnl", 0) for m in monthly]
        colors = [_GREEN if p >= 0 else _RED for p in net_pnls]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=months, y=net_pnls,
            marker_color=colors,
            name="Net P&L",
            hovertemplate="%{x}<br>Net P&L: $%{y:,.0f}<extra></extra>",
        ))

        # $10K target line
        fig.add_hline(
            y=10000, line=dict(color=_YELLOW, width=1, dash="dash"),
            annotation_text="$10K Target",
            annotation_position="top right",
        )
        fig.add_hline(y=0, line=dict(color=_NEUTRAL, width=0.5))

        fig.update_layout(
            title="Monthly P&L",
            xaxis_title="Month",
            yaxis_title="Net P&L ($)",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_trade_scatter(self, df: pd.DataFrame) -> str:
        """Build trade scatter: P&L vs trade number, colored by win/loss."""
        import plotly.graph_objects as go

        if "profit_loss_usd" not in df.columns:
            return ""

        pnl = df["profit_loss_usd"].fillna(0)
        is_winner = df.get("is_winner", pnl > 0).fillna(False)
        confidence = df.get("confidence", pd.Series(0.5, index=df.index))

        colors = [_GREEN if w else _RED for w in is_winner]
        sizes = (confidence * 15 + 5).clip(5, 20)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1, len(pnl) + 1)),
            y=pnl,
            mode="markers",
            marker=dict(
                color=colors,
                size=sizes,
                opacity=0.7,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate=(
                "Trade #%{x}<br>P&L: $%{y:,.0f}<br>"
                "Confidence: %{customdata:.0%}<extra></extra>"
            ),
            customdata=confidence,
        ))

        fig.add_hline(y=0, line=dict(color=_NEUTRAL, width=0.5))

        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="Trade Number",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_exit_analysis(self, analysis: Dict[str, Any]) -> str:
        """Build exit reason distribution + P&L by exit type."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not analysis:
            return ""

        reasons = list(analysis.keys())
        counts = [analysis[r]["count"] for r in reasons]
        avg_pnls = [analysis[r]["avg_pnl"] for r in reasons]
        win_rates = [analysis[r]["win_rate"] for r in reasons]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Exit Reason Distribution", "Avg P&L by Exit Reason"),
        )

        colors_count = [_GREEN if analysis[r]["avg_pnl"] >= 0 else _RED for r in reasons]

        fig.add_trace(go.Bar(
            x=reasons, y=counts,
            marker_color=colors_count,
            name="Count",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        ), row=1, col=1)

        colors_pnl = [_GREEN if p >= 0 else _RED for p in avg_pnls]
        fig.add_trace(go.Bar(
            x=reasons, y=avg_pnls,
            marker_color=colors_pnl,
            name="Avg P&L",
            hovertemplate="%{x}<br>Avg P&L: $%{y:,.0f}<extra></extra>",
        ), row=1, col=2)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_PANEL_DARK,
            plot_bgcolor=_PANEL_DARK,
            height=350,
            margin=dict(l=50, r=20, t=60, b=40),
            showlegend=False,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_trade_table(self, df: pd.DataFrame) -> str:
        """Build an HTML table of trades."""
        if df.empty:
            return "<p>No trades to display.</p>"

        cols = [
            ("trade_id", "#"),
            ("entry_time", "Entry"),
            ("exit_time", "Exit"),
            ("contract_symbol", "Contract"),
            ("confidence", "Conf"),
            ("time_in_trade_minutes", "Hold (min)"),
            ("profit_loss_usd", "P&L ($)"),
            ("profit_loss_pct", "P&L (%)"),
            ("exit_reason", "Exit Reason"),
        ]

        html = '<div class="table-wrap"><table class="trade-table"><thead><tr>'
        for _, header in cols:
            html += f"<th>{header}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            pnl = row.get("profit_loss_usd", 0) or 0
            row_class = "win-row" if pnl > 0 else "loss-row" if pnl < 0 else ""
            html += f'<tr class="{row_class}">'
            for col_name, _ in cols:
                val = row.get(col_name, "")
                if col_name == "profit_loss_usd" and val is not None:
                    color = _GREEN if val > 0 else _RED if val < 0 else _NEUTRAL
                    html += f'<td style="color:{color}">${val:,.0f}</td>'
                elif col_name == "profit_loss_pct" and val is not None:
                    color = _GREEN if val > 0 else _RED if val < 0 else _NEUTRAL
                    html += f'<td style="color:{color}">{val:+.1f}%</td>'
                elif col_name == "confidence" and val is not None:
                    html += f"<td>{val:.0%}</td>"
                elif col_name == "time_in_trade_minutes" and val is not None:
                    html += f"<td>{val:.0f}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"

        html += "</tbody></table></div>"
        return html

    def _assemble_html(
        self,
        kpi_html: str,
        equity_html: str,
        monthly_html: str,
        scatter_html: str,
        exit_html: str,
        table_html: str,
        summary: Dict[str, Any],
        report: Dict[str, Any],
    ) -> str:
        """Assemble all sections into a complete HTML page."""
        start_date = report.get("start_date", "N/A")
        end_date = report.get("end_date", "N/A")
        n_days = report.get("n_days", 0)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Dashboard — {start_date} to {end_date}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {_BG_DARK}; color: {_TEXT}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px; padding: 20px; }}
  h1 {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 600; color: {_TEXT}; margin: 20px 0 12px; }}
  .subtitle {{ color: {_NEUTRAL}; font-size: 12px; margin-bottom: 16px; }}

  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 20px; }}
  .kpi-card {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 14px; text-align: center; }}
  .kpi-label {{ font-size: 11px; color: {_NEUTRAL}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; }}

  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart-cell {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 12px; }}
  .chart-full {{ background: {_PANEL_DARK}; border: 1px solid {_BORDER}; border-radius: 6px; padding: 12px; margin-top: 16px; }}

  .table-wrap {{ overflow-x: auto; margin-top: 16px; }}
  .trade-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .trade-table th {{ background: {_PANEL_DARK}; color: {_NEUTRAL}; padding: 8px 10px; text-align: left; border-bottom: 1px solid {_BORDER}; position: sticky; top: 0; }}
  .trade-table td {{ padding: 6px 10px; border-bottom: 1px solid {_BORDER}; }}
  .trade-table tr:hover {{ background: rgba(255,255,255,0.03); }}
  .win-row {{ background: rgba(0,212,163,0.03); }}
  .loss-row {{ background: rgba(248,81,73,0.03); }}

  @media (max-width: 900px) {{
    .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
    .chart-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<h1>Trade Dashboard</h1>
<div class="subtitle">{start_date} to {end_date} &middot; {n_days} trading days &middot; {summary.get('total_trades', 0)} trades</div>

{kpi_html}

<div class="chart-full">{equity_html}</div>

<div class="chart-grid">
  <div class="chart-cell">{monthly_html}</div>
  <div class="chart-cell">{scatter_html}</div>
</div>

<div class="chart-full">{exit_html}</div>

<h2>Trade Log</h2>
{table_html}

</body>
</html>"""
