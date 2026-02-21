# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for src/ml/trade_simulator.py.

Coverage
--------
- Trade construction and P&L math (100-shares-per-contract)
- Trade.close_trade() — winner, loser, time calculation
- Trade.to_dict() serialisation
- TradeSimulator.simulate_from_label_row() — all 4 exit paths
- TradeSimulator.simulate_period()
- TradeSimulator.generate_monthly_report()
- _extract_contract_type() helper
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.trade_simulator import (
    SHARES_PER_CONTRACT,
    Trade,
    TradeSimulator,
    _extract_contract_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_row(
    close: float = 5.00,
    max_gain_120m: float = 40.0,
    min_loss_120m: float = -5.0,
    time_to_max_min: float = 60.0,
    date: str = "2025-09-15",
    hour_et: int = 10,
    minute_et: int = 30,
    ticker: str = "O:SPY250915C00580000",
) -> pd.Series:
    """Build a minimal feature-CSV-style row for simulation tests."""
    return pd.Series(
        {
            "close": close,
            "max_gain_120m": max_gain_120m,
            "min_loss_120m": min_loss_120m,
            "time_to_max_min": time_to_max_min,
            "date": date,
            "hour_et": hour_et,
            "minute_et": minute_et,
            "ticker": ticker,
        }
    )


def default_simulator(**kwargs) -> TradeSimulator:
    """Return a TradeSimulator with sensible test defaults."""
    params = {
        "position_size_usd": 12_500.0,
        "target_gain_pct": 30.0,
        "stop_loss_pct": -12.0,
        "max_time_minutes": 120,
        "fee_per_trade_usd": 4.0,
    }
    params.update(kwargs)
    return TradeSimulator(**params)


# ---------------------------------------------------------------------------
# SHARES_PER_CONTRACT constant
# ---------------------------------------------------------------------------


class TestSharesPerContract:
    def test_constant_is_100(self):
        assert SHARES_PER_CONTRACT == 100

    def test_class_attribute_matches_module(self):
        assert Trade.SHARES_PER_CONTRACT == SHARES_PER_CONTRACT


# ---------------------------------------------------------------------------
# Trade construction
# ---------------------------------------------------------------------------


class TestTradeConstruction:
    def test_basic_construction(self):
        t = Trade(
            trade_id=1,
            entry_time="2025-09-15 10:30 ET",
            contract_symbol="O:SPY250915C00580000",
            entry_price_per_share=5.00,
            position_size_usd=12_500.0,
            confidence=0.75,
        )
        assert t.trade_id == 1
        assert t.entry_time == "2025-09-15 10:30 ET"
        assert t.entry_price_per_share == 5.00
        assert t.confidence == 0.75
        assert t.trigger_factors == []

    def test_cost_per_contract(self):
        """cost_per_contract = entry_price × 100."""
        t = Trade(1, "t", "SYM", 4.20, 12_500.0, 0.70)
        assert t.cost_per_contract == pytest.approx(420.0)

    def test_num_contracts_floors_to_integer(self):
        """$12,500 / $420 per contract = 29.76… → 29 contracts."""
        t = Trade(1, "t", "SYM", 4.20, 12_500.0, 0.70)
        assert t.num_contracts == 29

    def test_actual_position_size(self):
        """actual_position = num_contracts × cost_per_contract."""
        t = Trade(1, "t", "SYM", 4.20, 12_500.0, 0.70)
        assert t.actual_position_size == pytest.approx(29 * 420.0)

    def test_zero_entry_price_gives_zero_contracts(self):
        t = Trade(1, "t", "SYM", 0.0, 12_500.0, 0.70)
        assert t.num_contracts == 0
        assert t.actual_position_size == 0.0

    def test_trigger_factors_stored(self):
        factors = [{"feature": "rsi_14", "value": 28.0, "importance": 0.12}]
        t = Trade(1, "t", "SYM", 5.0, 12_500.0, 0.70, trigger_factors=factors)
        assert t.trigger_factors == factors

    def test_exit_fields_initially_none(self):
        t = Trade(1, "t", "SYM", 5.0, 12_500.0, 0.70)
        assert t.exit_time is None
        assert t.profit_loss_usd is None
        assert t.is_winner is None


# ---------------------------------------------------------------------------
# Trade.close_trade()
# ---------------------------------------------------------------------------


class TestTradeCloseTrade:
    def test_winner_pnl_math(self):
        """entry=$5.00, exit=$6.50 (+30%), 29 contracts → P&L correct."""
        t = Trade(1, "t", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("2025-09-15 11:30 ET", 6.50, "Target hit", 60.0)

        # cost_per_contract = 5.00 × 100 = 500
        # exit_cost_per_contract = 6.50 × 100 = 650
        # profit_per_contract = 650 - 500 = 150
        # num_contracts = int(12500 / 500) = 25
        # total_pnl = 25 × 150 = 3750
        assert t.num_contracts == 25
        assert t.profit_loss_usd == pytest.approx(3_750.0)
        assert t.profit_loss_pct == pytest.approx(30.0)
        assert t.is_winner is True

    def test_loser_pnl_math(self):
        """entry=$5.00, exit=$4.40 (-12%), 25 contracts → negative P&L."""
        t = Trade(1, "t", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("2025-09-15 11:00 ET", 4.40, "Stop-loss", 30.0)

        assert t.profit_loss_usd == pytest.approx(-1_500.0)
        assert t.profit_loss_pct == pytest.approx(-12.0)
        assert t.is_winner is False

    def test_time_in_trade_explicit(self):
        t = Trade(1, "t", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("t2", 5.50, "Target hit", time_in_trade_minutes=45.0)
        assert t.time_in_trade_minutes == 45.0

    def test_time_in_trade_parsed_from_timestamps(self):
        """If time_in_trade_minutes=None, parse from ET strings."""
        t = Trade(1, "2025-09-15 10:00 ET", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("2025-09-15 11:00 ET", 5.50, "Target hit")
        assert t.time_in_trade_minutes == pytest.approx(60.0)

    def test_zero_contracts_gives_zero_pnl(self):
        t = Trade(1, "t", "SYM", 0.0, 12_500.0, 0.70)  # entry_price=0 → 0 contracts
        t.close_trade("t2", 5.00, "Time limit", 120.0)
        assert t.profit_loss_usd == pytest.approx(0.0)
        assert t.is_winner is False

    def test_exit_cost_per_contract(self):
        t = Trade(1, "t", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("t2", 7.00, "Target hit", 45.0)
        assert t.exit_cost_per_contract == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# Trade.to_dict()
# ---------------------------------------------------------------------------


class TestTradeToDict:
    def test_required_keys_present(self):
        t = Trade(1, "entry", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("exit", 6.50, "Target hit", 60.0)
        d = t.to_dict()
        required = {
            "trade_id", "entry_time", "exit_time", "contract_symbol",
            "entry_price_per_share", "exit_price_per_share",
            "cost_per_contract_entry", "cost_per_contract_exit",
            "num_contracts", "actual_position_size",
            "confidence", "time_in_trade_minutes",
            "profit_loss_usd", "profit_loss_pct",
            "exit_reason", "is_winner", "trigger_factors",
        }
        assert required.issubset(d.keys())

    def test_is_json_serialisable(self):
        import json
        t = Trade(1, "entry", "SYM", 5.00, 12_500.0, 0.70)
        t.close_trade("exit", 6.50, "Target hit", 60.0)
        json.dumps(t.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# _extract_contract_type()
# ---------------------------------------------------------------------------


class TestExtractContractType:
    def test_call_ticker(self):
        assert _extract_contract_type("O:SPY250915C00580000") == "C"

    def test_put_ticker(self):
        assert _extract_contract_type("O:SPY250915P00560000") == "P"

    def test_unknown_ticker(self):
        assert _extract_contract_type("UNKNOWN") == "X"

    def test_short_ticker(self):
        assert _extract_contract_type("SPY") == "X"


# ---------------------------------------------------------------------------
# TradeSimulator construction
# ---------------------------------------------------------------------------


class TestTradeSimulatorConstruction:
    def test_defaults(self):
        sim = TradeSimulator()
        assert sim.position_size_usd == 12_500.0
        assert sim.target_gain_pct == 30.0
        assert sim.stop_loss_pct == -12.0
        assert sim.max_time_minutes == 120
        assert sim.fee_per_trade_usd == 4.0

    def test_custom_params(self):
        sim = TradeSimulator(position_size_usd=5_000.0, target_gain_pct=50.0)
        assert sim.position_size_usd == 5_000.0
        assert sim.target_gain_pct == 50.0


# ---------------------------------------------------------------------------
# TradeSimulator.simulate_from_label_row()
# ---------------------------------------------------------------------------


class TestSimulateFromLabelRow:
    def test_target_hit_only(self):
        """max_gain >= target → 'Target hit'."""
        sim = default_simulator(target_gain_pct=30.0, stop_loss_pct=-12.0)
        row = make_row(close=5.00, max_gain_120m=40.0, min_loss_120m=-5.0, time_to_max_min=50.0)
        trade = sim.simulate_from_label_row(row, trade_id=1, confidence=0.75)

        assert trade is not None
        assert trade.exit_reason == "Target hit"
        assert trade.exit_price_per_share == pytest.approx(5.00 * 1.30)
        assert trade.is_winner is True

    def test_stop_hit_only(self):
        """min_loss <= stop → 'Stop-loss'."""
        sim = default_simulator(target_gain_pct=30.0, stop_loss_pct=-12.0)
        row = make_row(close=5.00, max_gain_120m=10.0, min_loss_120m=-20.0)
        trade = sim.simulate_from_label_row(row, 1, 0.70)

        assert trade is not None
        assert trade.exit_reason == "Stop-loss"
        assert trade.exit_price_per_share == pytest.approx(5.00 * 0.88)
        assert trade.is_winner is False

    def test_both_triggered_stop_first(self):
        """Both target AND stop triggered → conservative: stop assumed first."""
        sim = default_simulator(target_gain_pct=30.0, stop_loss_pct=-12.0)
        row = make_row(close=5.00, max_gain_120m=40.0, min_loss_120m=-15.0)
        trade = sim.simulate_from_label_row(row, 1, 0.70)

        assert trade is not None
        assert "Stop-loss" in trade.exit_reason
        assert trade.is_winner is False

    def test_time_limit_neither_triggered(self):
        """Neither target nor stop → 'Time limit (120 min)'."""
        sim = default_simulator(target_gain_pct=30.0, stop_loss_pct=-12.0)
        row = make_row(close=5.00, max_gain_120m=10.0, min_loss_120m=-5.0)
        trade = sim.simulate_from_label_row(row, 1, 0.70)

        assert trade is not None
        assert trade.exit_reason == "Time limit (120 min)"
        assert trade.time_in_trade_minutes == pytest.approx(120.0)

    def test_zero_price_returns_none(self):
        row = make_row(close=0.0)
        sim = default_simulator()
        assert sim.simulate_from_label_row(row, 1, 0.70) is None

    def test_negative_price_returns_none(self):
        row = make_row(close=-1.0)
        sim = default_simulator()
        assert sim.simulate_from_label_row(row, 1, 0.70) is None

    def test_too_expensive_returns_none(self):
        """If entry price so high that no full contract fits, return None."""
        sim = TradeSimulator(position_size_usd=100.0)  # tiny position
        row = make_row(close=20.00)  # $2000 per contract > $100 budget
        assert sim.simulate_from_label_row(row, 1, 0.70) is None

    def test_entry_time_format(self):
        row = make_row(date="2025-10-01", hour_et=9, minute_et=35)
        sim = default_simulator()
        trade = sim.simulate_from_label_row(row, 1, 0.70)
        assert trade is not None
        assert "2025-10-01" in trade.entry_time
        assert "09:35" in trade.entry_time

    def test_exit_time_constructed(self):
        sim = default_simulator(target_gain_pct=30.0)
        row = make_row(close=5.0, max_gain_120m=40.0, min_loss_120m=-5.0,
                       time_to_max_min=45.0, date="2025-10-01", hour_et=10, minute_et=30)
        trade = sim.simulate_from_label_row(row, 1, 0.70)
        assert trade is not None
        assert trade.exit_time is not None
        assert "2025-10-01" in trade.exit_time

    def test_nan_close_returns_none(self):
        row = make_row(close=float("nan"))
        sim = default_simulator()
        assert sim.simulate_from_label_row(row, 1, 0.70) is None


# ---------------------------------------------------------------------------
# TradeSimulator.simulate_period()
# ---------------------------------------------------------------------------


class TestSimulatePeriod:
    def _make_df(self, n: int = 5, close: float = 5.0) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "close": close,
                "max_gain_120m": 40.0,
                "min_loss_120m": -5.0,
                "time_to_max_min": 60.0,
                "date": "2025-09-15",
                "hour_et": 10,
                "minute_et": 30 + i,
                "ticker": "O:SPY250915C00580000",
            })
        return pd.DataFrame(rows)

    def test_no_signals_empty_list(self):
        sim = default_simulator()
        df = self._make_df(5)
        preds = np.array([0.50, 0.55, 0.60, 0.65, 0.66])  # all below 0.67
        trades = sim.simulate_period(df, preds, threshold=0.67)
        assert trades == []

    def test_signals_above_threshold(self):
        sim = default_simulator()
        df = self._make_df(3)
        preds = np.array([0.80, 0.90, 0.70])  # all above 0.67
        trades = sim.simulate_period(df, preds, threshold=0.67)
        assert len(trades) == 3

    def test_partial_signals(self):
        sim = default_simulator()
        df = self._make_df(4)
        preds = np.array([0.80, 0.50, 0.90, 0.60])
        trades = sim.simulate_period(df, preds, threshold=0.67)
        assert len(trades) == 2

    def test_skips_zero_price_rows(self):
        sim = default_simulator()
        df = self._make_df(3, close=0.0)
        preds = np.array([0.80, 0.90, 0.70])
        trades = sim.simulate_period(df, preds, threshold=0.67)
        assert trades == []

    def test_with_model_populates_trigger_factors(self):
        sim = default_simulator()
        df = self._make_df(1)
        preds = np.array([0.80])
        feature_cols = ["close", "max_gain_120m"]

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])

        trades = sim.simulate_period(df, preds, 0.67, model=mock_model, feature_cols=feature_cols)
        assert len(trades) == 1
        assert len(trades[0].trigger_factors) > 0

    def test_returns_trade_objects(self):
        sim = default_simulator()
        df = self._make_df(2)
        preds = np.array([0.80, 0.90])
        trades = sim.simulate_period(df, preds, threshold=0.67)
        for t in trades:
            assert isinstance(t, Trade)


# ---------------------------------------------------------------------------
# TradeSimulator.generate_monthly_report()
# ---------------------------------------------------------------------------


class TestGenerateMonthlyReport:
    def _make_winning_trade(self, trade_id: int = 1) -> Trade:
        t = Trade(trade_id, "entry", "O:SPY250915C00580000", 5.00, 12_500.0, 0.75)
        t.close_trade("exit", 6.50, "Target hit", 60.0)
        return t

    def _make_losing_trade(self, trade_id: int = 2) -> Trade:
        t = Trade(trade_id, "entry", "O:SPY250915P00560000", 5.00, 12_500.0, 0.70)
        t.close_trade("exit", 4.40, "Stop-loss", 30.0)
        return t

    def test_empty_trades(self):
        sim = default_simulator()
        report = sim.generate_monthly_report([], "2025-09")
        assert report["total_trades"] == 0
        assert report["month"] == "2025-09"
        assert "No trades executed" in report["message"]

    def test_single_winner_report(self):
        sim = default_simulator()
        t = self._make_winning_trade()
        report = sim.generate_monthly_report([t], "2025-09")
        assert report["total_trades"] == 1
        assert report["winning_trades"] == 1
        assert report["losing_trades"] == 0
        assert report["win_rate"] == pytest.approx(1.0)
        assert report["gross_profit_usd"] > 0
        assert report["gross_loss_usd"] == 0.0

    def test_single_loser_report(self):
        sim = default_simulator()
        t = self._make_losing_trade()
        report = sim.generate_monthly_report([t], "2025-09")
        assert report["winning_trades"] == 0
        assert report["losing_trades"] == 1
        assert report["win_rate"] == pytest.approx(0.0)
        assert report["gross_loss_usd"] < 0

    def test_mixed_trades_win_rate(self):
        sim = default_simulator()
        trades = [self._make_winning_trade(1), self._make_losing_trade(2),
                  self._make_winning_trade(3)]
        report = sim.generate_monthly_report(trades, "2025-09")
        assert report["total_trades"] == 3
        assert report["winning_trades"] == 2
        assert report["losing_trades"] == 1
        assert report["win_rate"] == pytest.approx(2 / 3)

    def test_required_keys(self):
        sim = default_simulator()
        trades = [self._make_winning_trade()]
        report = sim.generate_monthly_report(trades, "2025-09")
        required = {
            "month", "total_trades", "winning_trades", "losing_trades",
            "win_rate", "gross_profit_usd", "gross_loss_usd",
            "net_profit_usd", "total_fees_usd", "net_profit_after_fees_usd",
            "avg_return_per_trade_pct", "roi_pct",
            "avg_time_minutes", "median_time_minutes",
            "total_capital_deployed_usd", "total_contracts_traded",
            "calls_traded", "puts_traded",
            "exit_reasons", "trades",
        }
        assert required.issubset(report.keys())

    def test_net_after_fees(self):
        """net_after_fees = gross_pnl - n_trades × fee."""
        sim = TradeSimulator(fee_per_trade_usd=10.0)
        trades = [self._make_winning_trade(i) for i in range(3)]
        report = sim.generate_monthly_report(trades, "2025-09")
        expected_fees = 3 * 10.0
        assert report["total_fees_usd"] == pytest.approx(expected_fees)
        assert report["net_profit_after_fees_usd"] == pytest.approx(
            report["net_profit_usd"] - expected_fees
        )

    def test_calls_puts_counted_correctly(self):
        sim = default_simulator()
        # call: O:SPY250915C00580000, put: O:SPY250915P00560000
        call_trade = self._make_winning_trade(1)   # C in ticker
        put_trade = self._make_losing_trade(2)     # P in ticker
        report = sim.generate_monthly_report([call_trade, put_trade], "2025-09")
        assert report["calls_traded"] == 1
        assert report["puts_traded"] == 1

    def test_exit_reasons_counted(self):
        sim = default_simulator()
        trades = [self._make_winning_trade(1), self._make_losing_trade(2)]
        report = sim.generate_monthly_report(trades, "2025-09")
        assert "Target hit" in report["exit_reasons"]
        assert "Stop-loss" in report["exit_reasons"]

    def test_trades_list_in_report(self):
        sim = default_simulator()
        trades = [self._make_winning_trade()]
        report = sim.generate_monthly_report(trades, "2025-09")
        assert len(report["trades"]) == 1
        assert isinstance(report["trades"][0], dict)
