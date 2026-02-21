# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Prediction-speed benchmarking for real-time trading requirements.

Real-time use case
------------------
In live trading the model must produce a buy/no-buy decision within a single
minute bar (~60 seconds).  The hard latency requirement is **< 100 ms** per
prediction including feature lookup and model inference.

This module measures **model inference latency only** (the XGBClassifier
``predict_proba`` call on a single sample).  Feature construction latency is
measured separately in the ingestion pipeline.

Benchmark methodology
---------------------
1. Warm-up: run N_WARMUP calls before measuring (avoids JIT / cache cold-start).
2. Timed loop: ``time.perf_counter`` around each individual prediction.
3. Report: mean, p50, p95, p99, max latency in milliseconds.

Public API
----------
  benchmark_prediction_speed(model, sample_features, n_iterations=1000)
      → dict with latency statistics and a boolean ``meets_100ms_requirement``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()

_N_WARMUP = 20  # calls before timing begins


def benchmark_prediction_speed(
    model,
    sample_features: Union[List[float], np.ndarray],
    n_iterations: int = 1000,
) -> Dict[str, Any]:
    """Measure single-sample prediction latency for a fitted model.

    Runs *n_iterations* ``predict_proba`` calls on a single feature vector and
    reports latency percentiles.  A warm-up phase of up to 20 calls is run
    first to amortise any first-call overhead (XGBoost model compilation, CPU
    branch-predictor warm-up).

    Args:
        model:           Fitted classifier with a ``predict_proba`` method.
        sample_features: A single feature vector as a list or 1-D array.
                         Shape (n_features,).  Values are cast to float32.
        n_iterations:    Number of timed prediction calls.  Default 1000.

    Returns:
        Dict with keys:

        ``n_iterations``         int   — number of timed calls.
        ``mean_latency_ms``      float — arithmetic mean latency (ms).
        ``p50_latency_ms``       float — 50th percentile (median) latency (ms).
        ``p95_latency_ms``       float — 95th percentile latency (ms).
        ``p99_latency_ms``       float — 99th percentile latency (ms).
        ``max_latency_ms``       float — maximum observed latency (ms).
        ``meets_100ms_requirement`` bool — True if p99 < 100 ms.

    Raises:
        ValueError: If sample_features is empty or n_iterations < 1.
    """
    sample_array = np.asarray(sample_features, dtype=np.float32)
    if sample_array.ndim == 1:
        sample_array = sample_array.reshape(1, -1)

    if sample_array.size == 0:
        raise ValueError("benchmark_prediction_speed: sample_features is empty")
    if n_iterations < 1:
        raise ValueError(
            f"benchmark_prediction_speed: n_iterations must be ≥ 1, got {n_iterations}"
        )

    # Warm-up: avoid JIT / cache cold-start effects
    n_warmup = min(_N_WARMUP, n_iterations)
    for _ in range(n_warmup):
        model.predict_proba(sample_array)

    # Timed loop
    times_ms: List[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        model.predict_proba(sample_array)
        times_ms.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times_ms)
    p99 = float(np.percentile(arr, 99))

    result: Dict[str, Any] = {
        "n_iterations": n_iterations,
        "mean_latency_ms": float(np.mean(arr)),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": p99,
        "max_latency_ms": float(np.max(arr)),
        "meets_100ms_requirement": p99 < 100.0,
    }

    logger.info(
        f"benchmark_prediction_speed: {n_iterations} iterations | "
        f"mean={result['mean_latency_ms']:.3f}ms | "
        f"p99={result['p99_latency_ms']:.3f}ms | "
        f"{'✅ <100ms' if result['meets_100ms_requirement'] else '❌ >100ms'}"
    )

    return result
