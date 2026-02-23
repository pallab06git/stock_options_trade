#!/usr/bin/env python
# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
"""Final optimisation pipeline: maximum precision at 2–3 signals/day (Step 67).

Invokes the ``ml final-optimization`` CLI command with production defaults.

Steps executed
--------------
1.  Load full feature dataset (all available dates)
2.  Apply MagnitudeLabeler  (target = |move| ≥ 20%)
3.  70 / 30 chronological split  →  80 / 20 sub-split within train
4.  DeepHyperparameterOptimizer on LightGBM   (200 Optuna trials)
5.  DeepHyperparameterOptimizer on RandomForest (200 Optuna trials)
6.  Retrain best models on full training set
7.  EnsembleStrategy: AND / OR / AVG combinations at multiple thresholds
8.  MonteCarloSimulator: 1 000-iteration monthly P&L distribution
9.  Save  reports/final_optimization/final_optimization_results.json

Run with:
    python run_final_optimization.py > logs/final_optimization.log 2>&1 &

Expected wall time: 4–8 hours (200 trials × 2 models + ensemble + MC).
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — console + timestamped file
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"final_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("final_opt")

# ---------------------------------------------------------------------------
# Production constants
# ---------------------------------------------------------------------------

FEATURES_DIR          = "data/processed/features"
OUTPUT_DIR            = "reports/final_optimization"
N_TRIALS              = 200
TARGET_SIGNALS_PER_DAY = 2.5
MIN_MAGNITUDE         = 20.0
N_MC_SIMULATIONS      = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    wall_start = time.time()
    _banner("Final Optimisation Pipeline (Step 67) — starting")
    log.info("Log file:  %s", LOG_PATH)
    log.info("PID:       %d", os.getpid())
    log.info("")
    log.info("Configuration:")
    log.info("  features_dir:           %s", FEATURES_DIR)
    log.info("  output_dir:             %s", OUTPUT_DIR)
    log.info("  n_trials (per model):   %d", N_TRIALS)
    log.info("  target_signals_per_day: %.1f", TARGET_SIGNALS_PER_DAY)
    log.info("  min_magnitude:          %.1f%%", MIN_MAGNITUDE)
    log.info("  n_mc_simulations:       %d", N_MC_SIMULATIONS)

    cmd = [
        sys.executable, "-m", "src.cli", "ml", "final-optimization",
        "--features-dir",           FEATURES_DIR,
        "--n-trials",               str(N_TRIALS),
        "--target-signals-per-day", str(TARGET_SIGNALS_PER_DAY),
        "--min-magnitude",          str(MIN_MAGNITUDE),
        "--n-mc-simulations",       str(N_MC_SIMULATIONS),
        "--output",                 OUTPUT_DIR,
    ]

    log.info("")
    log.info("Launching:  %s", " ".join(cmd))
    log.info("")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        log.error("Pipeline exited with code %d", result.returncode)
        sys.exit(1)

    _banner("Pipeline complete")
    log.info("Total wall time: %s", _elapsed(wall_start))
    log.info("")
    log.info("Results:  %s/final_optimization_results.json", OUTPUT_DIR)
    log.info("Models:   %s/lgbm_deep_opt.pkl", OUTPUT_DIR)
    log.info("          %s/rf_deep_opt.pkl",   OUTPUT_DIR)


if __name__ == "__main__":
    main()
