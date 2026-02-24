# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for TradeDashboard."""

import json

import numpy as np
import pandas as pd
import pytest

from src.visualization.trade_dashboard import TradeDashboard


def _make_report(n_trades=5):
    """Create a synthetic pipeline report dict."""
    trades = []
    for i in range(n_trades):
        pnl = 500.0 if i % 2 == 0 else -300.0
        trades.append({
            "trade_id": i + 1,
            "entry_time": f"2025-03-{3 + i:02d} 10:00 ET",
            "exit_time": f"2025-03-{3 + i:02d} 10:30 ET",
            "contract_symbol": "O:SPY250307C00580000",
            "entry_price_per_share": 4.0,
            "exit_price_per_share": 4.5 if pnl > 0 else 3.7,
            "num_contracts": 25,
            "actual_position_size": 10000.0,
            "confidence": 0.75 + i * 0.03,
            "time_in_trade_minutes": 30.0,
            "profit_loss_usd": pnl,
            "profit_loss_pct": pnl / 100,
            "exit_reason": "Target hit" if pnl > 0 else "Hard stop",
            "is_winner": pnl > 0,
        })

    return {
        "executive_summary": {
            "total_trades": n_trades,
            "winning_trades": sum(1 for t in trades if t["is_winner"]),
            "losing_trades": sum(1 for t in trades if not t["is_winner"]),
            "win_rate": 0.6,
            "gross_pnl": 900.0,
            "fees": 20.0,
            "net_pnl": 880.0,
            "avg_pnl_per_trade": 176.0,
            "avg_hold_minutes": 30.0,
            "median_hold_minutes": 30.0,
            "avg_trades_per_day": 2.5,
        },
        "monthly_pnl": [
            {"month": "2025-03", "trades": 3, "winners": 2, "win_rate": 0.67,
             "gross_pnl": 700, "fees": 12, "net_pnl": 688},
            {"month": "2025-04", "trades": 2, "winners": 1, "win_rate": 0.50,
             "gross_pnl": 200, "fees": 8, "net_pnl": 192},
        ],
        "exit_analysis": {
            "Target hit": {"count": 3, "avg_pnl": 500, "total_pnl": 1500, "win_rate": 1.0},
            "Hard stop": {"count": 2, "avg_pnl": -300, "total_pnl": -600, "win_rate": 0.0},
        },
        "trades": trades,
        "start_date": "2025-03-03",
        "end_date": "2025-04-30",
        "n_days": 42,
    }


class TestTradeDashboardInit:
    def test_creates_instance(self):
        dashboard = TradeDashboard()
        assert dashboard is not None


class TestBuildDashboard:
    def test_creates_html_file(self, tmp_path):
        dashboard = TradeDashboard()
        report = _make_report()
        output = tmp_path / "dashboard.html"
        dashboard.build_dashboard(report, output)
        assert output.exists()

    def test_html_contains_plotly(self, tmp_path):
        dashboard = TradeDashboard()
        report = _make_report()
        output = tmp_path / "dashboard.html"
        dashboard.build_dashboard(report, output)
        html = output.read_text()
        assert "plotly" in html.lower()

    def test_html_contains_kpis(self, tmp_path):
        dashboard = TradeDashboard()
        report = _make_report()
        output = tmp_path / "dashboard.html"
        dashboard.build_dashboard(report, output)
        html = output.read_text()
        assert "Net P&amp;L" in html or "Net P&L" in html
        assert "Win Rate" in html

    def test_html_contains_dates(self, tmp_path):
        dashboard = TradeDashboard()
        report = _make_report()
        output = tmp_path / "dashboard.html"
        dashboard.build_dashboard(report, output)
        html = output.read_text()
        assert "2025-03-03" in html
        assert "2025-04-30" in html

    def test_empty_report(self, tmp_path):
        dashboard = TradeDashboard()
        report = {
            "executive_summary": {"total_trades": 0, "net_pnl": 0, "win_rate": 0},
            "monthly_pnl": [],
            "exit_analysis": {},
            "trades": [],
        }
        output = tmp_path / "empty_dashboard.html"
        dashboard.build_dashboard(report, output)
        assert output.exists()

    def test_creates_parent_dirs(self, tmp_path):
        dashboard = TradeDashboard()
        report = _make_report()
        output = tmp_path / "nested" / "dir" / "dashboard.html"
        dashboard.build_dashboard(report, output)
        assert output.exists()


class TestBuildKPICards:
    def test_returns_html(self):
        dashboard = TradeDashboard()
        html = dashboard._build_kpi_cards({"net_pnl": 5000, "win_rate": 0.65, "total_trades": 10})
        assert "kpi-grid" in html
        assert "$5,000" in html

    def test_negative_pnl(self):
        dashboard = TradeDashboard()
        html = dashboard._build_kpi_cards({"net_pnl": -3000, "win_rate": 0.4, "total_trades": 5})
        assert "$-3,000" in html or "-$3,000" in html


class TestBuildEquityCurve:
    def test_returns_chart_html(self):
        dashboard = TradeDashboard()
        df = pd.DataFrame({
            "profit_loss_usd": [500, -200, 300, 100, -50],
        })
        html = dashboard._build_equity_curve(df)
        assert "Cumulative" in html

    def test_empty_df(self):
        dashboard = TradeDashboard()
        html = dashboard._build_equity_curve(pd.DataFrame())
        assert html == ""


class TestBuildTradeTable:
    def test_returns_table(self):
        dashboard = TradeDashboard()
        df = pd.DataFrame([{
            "trade_id": 1,
            "entry_time": "2025-03-03 10:00 ET",
            "exit_time": "2025-03-03 10:30 ET",
            "contract_symbol": "O:SPY250307C00580000",
            "confidence": 0.85,
            "time_in_trade_minutes": 30,
            "profit_loss_usd": 500,
            "profit_loss_pct": 12.5,
            "exit_reason": "Target hit",
        }])
        html = dashboard._build_trade_table(df)
        assert "<table" in html
        assert "Target hit" in html

    def test_empty_df(self):
        dashboard = TradeDashboard()
        html = dashboard._build_trade_table(pd.DataFrame())
        assert "No trades" in html
