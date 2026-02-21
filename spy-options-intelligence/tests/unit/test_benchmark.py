# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for src/ml/benchmark.py — prediction speed benchmarking."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ml.benchmark import benchmark_prediction_speed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_model() -> MagicMock:
    """Mock model that returns immediately with a fixed probability output."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    return m


# ---------------------------------------------------------------------------
# TestBenchmarkReturnShape
# ---------------------------------------------------------------------------


class TestBenchmarkReturnShape:
    def test_returns_dict(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2, 0.3], n_iterations=10)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2, 0.3], n_iterations=10)
        required = {
            "n_iterations",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "max_latency_ms",
            "meets_100ms_requirement",
        }
        assert set(result.keys()) == required

    def test_n_iterations_matches_input(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2, 0.3], n_iterations=50)
        assert result["n_iterations"] == 50

    def test_meets_requirement_is_bool(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1], n_iterations=5)
        assert isinstance(result["meets_100ms_requirement"], bool)


# ---------------------------------------------------------------------------
# TestBenchmarkLatencyValues
# ---------------------------------------------------------------------------


class TestBenchmarkLatencyValues:
    def test_all_latencies_are_positive(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2], n_iterations=20)
        for key in ("mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
                    "p99_latency_ms", "max_latency_ms"):
            assert result[key] > 0.0, f"{key} should be positive"

    def test_percentile_ordering(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1], n_iterations=100)
        assert result["p50_latency_ms"] <= result["p95_latency_ms"]
        assert result["p95_latency_ms"] <= result["p99_latency_ms"]
        assert result["p99_latency_ms"] <= result["max_latency_ms"]

    def test_mean_between_min_and_max(self):
        result = benchmark_prediction_speed(_fast_model(), [0.5], n_iterations=50)
        assert result["mean_latency_ms"] <= result["max_latency_ms"]

    def test_mock_model_meets_100ms(self):
        """A mock model (no real computation) must easily beat 100 ms."""
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2], n_iterations=100)
        assert result["meets_100ms_requirement"] is True

    def test_meets_requirement_consistent_with_p99(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1], n_iterations=50)
        expected = result["p99_latency_ms"] < 100.0
        assert result["meets_100ms_requirement"] == expected


# ---------------------------------------------------------------------------
# TestBenchmarkInputHandling
# ---------------------------------------------------------------------------


class TestBenchmarkInputHandling:
    def test_list_input_accepted(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2, 0.3], n_iterations=5)
        assert "mean_latency_ms" in result

    def test_numpy_array_input_accepted(self):
        sample = np.array([0.1, 0.2, 0.3])
        result = benchmark_prediction_speed(_fast_model(), sample, n_iterations=5)
        assert "mean_latency_ms" in result

    def test_model_called_with_2d_array(self):
        model = _fast_model()
        benchmark_prediction_speed(model, [0.1, 0.2], n_iterations=5)
        call_arg = model.predict_proba.call_args[0][0]
        assert call_arg.ndim == 2
        assert call_arg.shape[0] == 1

    def test_model_called_n_plus_warmup_times(self):
        """predict_proba is called for both warmup and timed iterations."""
        model = _fast_model()
        n_iter = 10
        benchmark_prediction_speed(model, [0.1], n_iterations=n_iter)
        total_calls = model.predict_proba.call_count
        # warmup (≤20) + n_iter timed calls
        assert total_calls >= n_iter


# ---------------------------------------------------------------------------
# TestBenchmarkValidation
# ---------------------------------------------------------------------------


class TestBenchmarkValidation:
    def test_raises_on_empty_features(self):
        model = _fast_model()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        with pytest.raises(ValueError, match="empty"):
            benchmark_prediction_speed(model, [], n_iterations=5)

    def test_raises_on_zero_iterations(self):
        with pytest.raises(ValueError, match="n_iterations"):
            benchmark_prediction_speed(_fast_model(), [0.1], n_iterations=0)

    def test_raises_on_negative_iterations(self):
        with pytest.raises(ValueError, match="n_iterations"):
            benchmark_prediction_speed(_fast_model(), [0.1], n_iterations=-1)

    def test_single_iteration_works(self):
        result = benchmark_prediction_speed(_fast_model(), [0.1, 0.2], n_iterations=1)
        assert result["n_iterations"] == 1
        assert result["mean_latency_ms"] > 0
