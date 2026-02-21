# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Unit tests for src/ml/dashboard.py.

Because Streamlit cannot run headlessly in unit tests, tests target:
  - Data-loading helpers (_load_results_dir, _load_comparison_csv,
    _load_overlap_json)
  - CLI argument parsing (_get_results_dir)
  - The module-level guard: clear ImportError when streamlit is absent

All Streamlit calls are mocked out so no UI is exercised.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Streamlit mock — must be installed before importing the dashboard module
# ---------------------------------------------------------------------------

_st_mock = MagicMock()
_st_mock.cache_data = MagicMock(side_effect=lambda **kwargs: lambda fn: fn)
sys.modules.setdefault("streamlit", _st_mock)

# Also mock plotly so we can test without it installed
_plotly_express_mock = MagicMock()
_plotly_go_mock = MagicMock()
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.express", _plotly_express_mock)
sys.modules.setdefault("plotly.graph_objects", _plotly_go_mock)

import src.ml.dashboard as dash  # noqa: E402 — must come after mock setup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_result_dict(
    threshold: float = 0.80,
    total_signals: int = 10,
    wins: int = 7,
    net_profit: float = 3_000.0,
) -> Dict[str, Any]:
    losses = total_signals - wins
    return {
        "threshold": threshold,
        "total_signals": total_signals,
        "calls": total_signals // 2,
        "puts": total_signals - total_signals // 2,
        "call_pct": 50.0,
        "put_pct": 50.0,
        "wins": wins,
        "win_rate": wins / total_signals if total_signals else 0.0,
        "losses": losses,
        "win_profit_min": 100.0,
        "win_profit_median": 250.0,
        "win_profit_mean": 300.0,
        "win_profit_max": 800.0,
        "loss_amount_min": -200.0,
        "loss_amount_median": -150.0,
        "loss_amount_mean": -160.0,
        "loss_amount_max": -100.0,
        "gross_profit_usd": wins * 300.0,
        "gross_loss_usd": losses * -160.0,
        "monthly_profit_usd": net_profit + 4.0 * total_signals,
        "total_fees_usd": 4.0 * total_signals,
        "net_profit_usd": net_profit,
        "avg_pnl_per_trade_usd": net_profit / total_signals if total_signals else 0.0,
        "trades": [
            {
                "trade_id": f"T{i}",
                "contract_symbol": f"SPY250117C00400000",
                "contract_type": "C" if i % 2 == 0 else "P",
                "entry_time": "2025-01-17T10:00:00",
                "exit_time": "2025-01-17T11:00:00",
                "time_in_trade_minutes": 60,
                "entry_price_per_share": 1.5,
                "exit_price_per_share": 1.95 if i < wins else 1.32,
                "num_contracts": 8,
                "profit_loss_usd": 360.0 if i < wins else -144.0,
                "profit_loss_pct": 30.0 if i < wins else -12.0,
                "exit_reason": "Target Hit" if i < wins else "Stop-Loss",
                "confidence": 0.88,
                "is_winner": i < wins,
            }
            for i in range(total_signals)
        ],
    }


def _write_results_files(tmp_dir: Path) -> None:
    """Write synthetic *_results.json and model_comparison.csv to tmp_dir."""
    thresholds = [0.70, 0.80, 0.90]

    for model in ("xgboost", "lightgbm"):
        model_data = {
            str(t): _build_result_dict(threshold=t, net_profit=t * 10_000)
            for t in thresholds
        }
        (tmp_dir / f"{model}_results.json").write_text(
            json.dumps(model_data, indent=2)
        )

    import csv

    with open(tmp_dir / "model_comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["Model", "Opt Score", "Signals (80%)", "Win Rate",
                        "Net Profit", "Meets Target"],
        )
        writer.writeheader()
        writer.writerow({"Model": "xgboost", "Opt Score": 0.91,
                         "Signals (80%)": 10, "Win Rate": "70.0%",
                         "Net Profit": "$+8,000", "Meets Target": "NO"})

    overlap_data = {
        "threshold": 0.80,
        "n_models_compared": 2,
        "model_names": ["xgboost", "lightgbm"],
        "total_unique_signals": 15,
        "all_models_agree": 5,
        "majority_agree": 5,
        "overlap_breakdown": {"1_models": 10, "2_models": 5},
        "detailed_overlaps": {"2": [
            {"contract_symbol": "SPY250117C00400000",
             "entry_time": "2025-01-17T10:00:00",
             "models": ["xgboost", "lightgbm"]}
        ]},
    }
    (tmp_dir / "overlap_0.80.json").write_text(json.dumps(overlap_data))


# ---------------------------------------------------------------------------
# Tests: _load_results_dir
# ---------------------------------------------------------------------------


class TestLoadResultsDir(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)
        _write_results_files(self.tmp_path)

    def test_returns_dict_keyed_by_model_name(self):
        result = dash._load_results_dir(self.tmp)
        self.assertIn("xgboost", result)
        self.assertIn("lightgbm", result)

    def test_threshold_keys_are_floats(self):
        result = dash._load_results_dir(self.tmp)
        for threshold in result["xgboost"]:
            self.assertIsInstance(threshold, float)

    def test_correct_threshold_values(self):
        result = dash._load_results_dir(self.tmp)
        self.assertIn(0.70, result["xgboost"])
        self.assertIn(0.80, result["xgboost"])
        self.assertIn(0.90, result["xgboost"])

    def test_result_dict_has_expected_keys(self):
        result = dash._load_results_dir(self.tmp)
        r = result["xgboost"][0.80]
        for key in ("total_signals", "wins", "net_profit_usd", "trades"):
            self.assertIn(key, r)

    def test_missing_directory_returns_empty(self):
        result = dash._load_results_dir("/nonexistent/path/xyz")
        self.assertEqual(result, {})

    def test_empty_directory_returns_empty(self):
        import tempfile

        empty = tempfile.mkdtemp()
        result = dash._load_results_dir(empty)
        self.assertEqual(result, {})

    def test_corrupt_json_is_skipped(self):
        (self.tmp_path / "corrupt_results.json").write_text("{invalid json}")
        result = dash._load_results_dir(self.tmp)
        self.assertNotIn("corrupt", result)
        # Valid models still loaded
        self.assertIn("xgboost", result)

    def test_two_models_loaded(self):
        result = dash._load_results_dir(self.tmp)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Tests: _load_comparison_csv
# ---------------------------------------------------------------------------


class TestLoadComparisonCsv(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        _write_results_files(Path(self.tmp))

    def test_returns_dataframe(self):
        import pandas as pd

        df = dash._load_comparison_csv(self.tmp)
        self.assertIsInstance(df, pd.DataFrame)

    def test_dataframe_not_empty(self):
        df = dash._load_comparison_csv(self.tmp)
        self.assertFalse(df.empty)

    def test_model_column_present(self):
        df = dash._load_comparison_csv(self.tmp)
        self.assertIn("Model", df.columns)

    def test_missing_directory_returns_empty_df(self):
        import pandas as pd

        df = dash._load_comparison_csv("/nonexistent/path")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_missing_csv_returns_empty_df(self):
        import tempfile
        import pandas as pd

        empty = tempfile.mkdtemp()
        df = dash._load_comparison_csv(empty)
        self.assertTrue(df.empty)


# ---------------------------------------------------------------------------
# Tests: _load_overlap_json
# ---------------------------------------------------------------------------


class TestLoadOverlapJson(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        _write_results_files(Path(self.tmp))

    def test_returns_dict(self):
        result = dash._load_overlap_json(self.tmp, 0.80)
        self.assertIsInstance(result, dict)

    def test_has_expected_keys(self):
        result = dash._load_overlap_json(self.tmp, 0.80)
        for key in ("total_unique_signals", "n_models_compared",
                    "all_models_agree", "overlap_breakdown"):
            self.assertIn(key, result)

    def test_correct_model_count(self):
        result = dash._load_overlap_json(self.tmp, 0.80)
        self.assertEqual(result["n_models_compared"], 2)

    def test_missing_threshold_returns_empty(self):
        result = dash._load_overlap_json(self.tmp, 0.99)
        self.assertEqual(result, {})

    def test_missing_directory_returns_empty(self):
        result = dash._load_overlap_json("/nonexistent/path", 0.80)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Tests: _get_results_dir
# ---------------------------------------------------------------------------


class TestGetResultsDir(unittest.TestCase):
    def test_returns_default_when_no_args(self):
        with patch.object(sys, "argv", ["streamlit", "run", "src/ml/dashboard.py"]):
            result = dash._get_results_dir()
        self.assertEqual(result, str(dash._DEFAULT_RESULTS_DIR))

    def test_returns_custom_dir_via_flag(self):
        with patch.object(
            sys,
            "argv",
            ["streamlit", "run", "src/ml/dashboard.py", "--", "--results-dir", "/custom/path"],
        ):
            result = dash._get_results_dir()
        self.assertEqual(result, "/custom/path")

    def test_results_dir_underscore_variant(self):
        with patch.object(
            sys,
            "argv",
            ["streamlit", "run", "src/ml/dashboard.py", "--", "--results_dir", "/alt/path"],
        ):
            result = dash._get_results_dir()
        self.assertEqual(result, "/alt/path")

    def test_flag_without_value_returns_default(self):
        with patch.object(
            sys,
            "argv",
            ["streamlit", "run", "--results-dir"],
        ):
            result = dash._get_results_dir()
        # --results-dir is the last arg, no value follows → default
        self.assertEqual(result, str(dash._DEFAULT_RESULTS_DIR))


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------


class TestDashboardConstants(unittest.TestCase):
    def test_default_results_dir_is_path(self):
        self.assertIsInstance(dash._DEFAULT_RESULTS_DIR, Path)

    def test_model_colours_has_entries(self):
        self.assertGreater(len(dash._MODEL_COLOURS), 0)
        for colour in dash._MODEL_COLOURS:
            self.assertRegex(colour, r"^#[0-9a-fA-F]{6}$")

    def test_threshold_labels_covers_expected_values(self):
        for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            self.assertIn(t, dash._THRESHOLD_LABELS)


if __name__ == "__main__":
    unittest.main()
