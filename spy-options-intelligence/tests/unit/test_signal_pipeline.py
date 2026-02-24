# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for SignalPipeline and PipelineReport."""

import numpy as np
import pandas as pd
import pytest

from src.ml.signal_pipeline import PipelineReport, SignalPipeline
from src.ml.trade_simulator import Trade


class TestPipelineReportInit:
    def test_creates_instance(self):
        report = PipelineReport()
        assert report is not None


class TestPipelineReportGenerate:
    def test_empty_trades(self):
        report = PipelineReport()
        result = report.generate_report([], {})
        assert result["executive_summary"]["total_trades"] == 0
        assert result["executive_summary"]["net_pnl"] == 0.0

    def test_with_trades(self):
        report = PipelineReport()
        t1 = Trade(1, "2025-03-03 10:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.85)
        t1.close_trade("2025-03-03 10:30 ET", 5.0, "Target hit", 30.0)
        t2 = Trade(2, "2025-03-03 11:00 ET", "O:SPY250307P00580000", 3.0, 10000, 0.75)
        t2.close_trade("2025-03-03 11:15 ET", 2.5, "Hard stop", 15.0)

        result = report.generate_report([t1, t2], {"fee_per_trade": 4.0})

        assert result["executive_summary"]["total_trades"] == 2
        assert result["executive_summary"]["winning_trades"] == 1
        assert result["executive_summary"]["losing_trades"] == 1
        assert result["executive_summary"]["win_rate"] == 0.5

    def test_monthly_breakdown(self):
        report = PipelineReport()
        t1 = Trade(1, "2025-03-03 10:00 ET", "O:SPY", 4.0, 10000, 0.85)
        t1.close_trade("2025-03-03 10:30 ET", 5.0, "Target", 30.0)
        t2 = Trade(2, "2025-04-01 10:00 ET", "O:SPY", 3.0, 10000, 0.75)
        t2.close_trade("2025-04-01 10:30 ET", 3.5, "Target", 30.0)

        result = report.generate_report([t1, t2], {"fee_per_trade": 4.0})
        assert len(result["monthly_pnl"]) == 2
        months = [m["month"] for m in result["monthly_pnl"]]
        assert "2025-03" in months
        assert "2025-04" in months

    def test_exit_analysis(self):
        report = PipelineReport()
        t1 = Trade(1, "2025-03-03 10:00 ET", "O:SPY", 4.0, 10000, 0.85)
        t1.close_trade("2025-03-03 10:30 ET", 5.0, "Target hit", 30.0)
        t2 = Trade(2, "2025-03-03 11:00 ET", "O:SPY", 4.0, 10000, 0.75)
        t2.close_trade("2025-03-03 11:30 ET", 3.0, "Hard stop", 30.0)

        result = report.generate_report([t1, t2], {"fee_per_trade": 4.0})
        assert "Target hit" in result["exit_analysis"]
        assert "Hard stop" in result["exit_analysis"]
        assert result["exit_analysis"]["Target hit"]["count"] == 1

    def test_report_has_generated_at(self):
        report = PipelineReport()
        result = report.generate_report([], {})
        assert "generated_at" in result

    def test_trade_dicts_included(self):
        report = PipelineReport()
        t = Trade(1, "2025-03-03 10:00 ET", "O:SPY", 4.0, 10000, 0.85)
        t.close_trade("2025-03-03 10:30 ET", 5.0, "Target", 30.0)

        result = report.generate_report([t], {"fee_per_trade": 4.0})
        assert len(result["trades"]) == 1
        assert result["trades"][0]["trade_id"] == 1


class TestSignalPipelineInit:
    def test_defaults(self):
        pipeline = SignalPipeline()
        assert pipeline.position_size == 10_000
        assert pipeline.max_signals_per_day == 3
        assert pipeline.entry_threshold == 0.70
        assert pipeline.exit_threshold == 0.60

    def test_custom_config(self):
        config = {"signal_pipeline": {
            "position_size_usd": 5000,
            "max_signals_per_day": 2,
            "entry_threshold": 0.80,
        }}
        pipeline = SignalPipeline(config)
        assert pipeline.position_size == 5000
        assert pipeline.max_signals_per_day == 2
        assert pipeline.entry_threshold == 0.80


class TestReconstructTrades:
    def test_roundtrip(self):
        pipeline = SignalPipeline()

        t = Trade(1, "2025-03-03 10:00 ET", "O:SPY250307C00580000", 4.0, 10000, 0.85)
        t.close_trade("2025-03-03 10:30 ET", 5.0, "Target hit", 30.0)

        trade_dicts = [t.to_dict()]
        reconstructed = pipeline._reconstruct_trades(trade_dicts)

        assert len(reconstructed) == 1
        rt = reconstructed[0]
        assert rt.trade_id == 1
        assert rt.exit_reason == "Target hit"
        assert rt.profit_loss_usd is not None

    def test_empty_list(self):
        pipeline = SignalPipeline()
        result = pipeline._reconstruct_trades([])
        assert result == []


class TestSaveReport:
    def test_saves_json(self, tmp_path):
        pipeline = SignalPipeline()
        report = {
            "executive_summary": {"total_trades": 1},
            "trades": [{"trade_id": 1, "entry_time": "2025-03-03 10:00 ET"}],
        }
        pipeline._save_report(report, str(tmp_path / "output"))
        assert (tmp_path / "output" / "pipeline_report.json").exists()
        assert (tmp_path / "output" / "pipeline_trades.csv").exists()

    def test_saves_without_trades(self, tmp_path):
        pipeline = SignalPipeline()
        report = {"executive_summary": {"total_trades": 0}, "trades": []}
        pipeline._save_report(report, str(tmp_path / "output"))
        assert (tmp_path / "output" / "pipeline_report.json").exists()
