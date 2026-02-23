#!/usr/bin/env python3
# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Master script: Volatility/Magnitude Prediction Experiment (Step 63).

Runs the complete magnitude prediction pipeline autonomously:

    Phase 1  Generate magnitude label preview (fast diagnostic)
    Phase 2  Train XGBoost + LightGBM + RandomForest with Optuna (50 trials)
    Phase 3  Print final comparison vs directional baseline

Usage
-----
    # Run in background, log to file:
    python run_magnitude_experiment.py > magnitude_experiment.log 2>&1 &

    # Smoke-test (5 trials):
    python run_magnitude_experiment.py --n-trials 5

    # Check progress:
    tail -f magnitude_experiment.log

Expected runtime: 6-9 hours (50 trials × 3 models × Optuna overhead)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES_DIR   = "data/processed/features"
START_DATE     = "2025-03-03"
END_DATE       = "2026-02-19"
MIN_MAGNITUDE  = 20.0        # %: minimum abs move to label as positive
N_TRIALS       = 50          # Optuna trials per model type
CV_SPLITS      = 3
POSITION_SIZE  = 12500       # USD per straddle
THRESHOLDS     = "0.50,0.60,0.70,0.75,0.80,0.85,0.90"
OUTPUT_DIR     = "reports/magnitude_experiment"

# Directional baseline (from sustained-movement-experiment + leakage audit)
DIRECTIONAL_BASELINE = {
    "best_model": "xgboost_v3_clean",
    "best_threshold": 0.70,
    "precision": 0.519,      # 51.9% @ 0.70 on test holdout (106 signals)
    "signals_per_day": 1.5,
    "monthly_pnl_directional": None,  # directional model: 33% ceiling
}


def _banner(text: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def _run_phase(label: str, cmd: list[str]) -> int:
    """Run a CLI command, stream its output, and return exit code."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {label}")
    print(f"  CMD: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, check=False)
    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {label}: {status}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Magnitude prediction experiment")
    parser.add_argument("--n-trials",       type=int,   default=N_TRIALS)
    parser.add_argument("--min-magnitude",  type=float, default=MIN_MAGNITUDE)
    parser.add_argument("--features-dir",   default=FEATURES_DIR)
    parser.add_argument("--start-date",     default=START_DATE)
    parser.add_argument("--end-date",       default=END_DATE)
    parser.add_argument("--output",         default=OUTPUT_DIR)
    args = parser.parse_args()

    t_start = time.time()

    _banner("MAGNITUDE PREDICTION EXPERIMENT  (Step 63)")
    print(f"  Start time:    {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Features dir:  {args.features_dir}")
    print(f"  Date range:    {args.start_date} → {args.end_date}")
    print(f"  Min magnitude: {args.min_magnitude}%")
    print(f"  Optuna trials: {args.n_trials} per model")
    print(f"  Output dir:    {args.output}")

    # ── Phase 1: Label preview ─────────────────────────────────────────────
    _banner("PHASE 1 / 2  —  Magnitude Label Preview")
    rc = _run_phase(
        "generate-magnitude-labels",
        [
            sys.executable, "-m", "src.cli", "ml", "generate-magnitude-labels",
            "--features-dir",  args.features_dir,
            "--start-date",    args.start_date,
            "--end-date",      args.end_date,
            "--min-magnitude", str(args.min_magnitude),
        ],
    )
    if rc != 0:
        print("\n  ERROR: Label preview failed.  Aborting.")
        return rc

    # ── Phase 2: Train models ──────────────────────────────────────────────
    _banner("PHASE 2 / 2  —  Train Magnitude Models")
    print(
        f"  Expected runtime: {args.n_trials * 3 // 60 + 1}–"
        f"{args.n_trials * 3 // 30 + 2} hours (estimate)"
    )
    rc = _run_phase(
        "train-magnitude-models",
        [
            sys.executable, "-m", "src.cli", "ml", "train-magnitude-models",
            "--features-dir",   args.features_dir,
            "--start-date",     args.start_date,
            "--end-date",       args.end_date,
            "--min-magnitude",  str(args.min_magnitude),
            "--n-trials",       str(args.n_trials),
            "--cv-splits",      str(CV_SPLITS),
            "--thresholds",     THRESHOLDS,
            "--position-size",  str(POSITION_SIZE),
            "--output",         args.output,
        ],
    )
    if rc != 0:
        print("\n  ERROR: Model training failed.")
        return rc

    # ── Phase 3: Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins        = rem // 60

    _banner("EXPERIMENT COMPLETE")
    print(f"  Finished:    {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Elapsed:     {hours}h {mins}m")
    print()
    print("  Directional baseline (xgboost_v3_clean @ 0.70):")
    print(f"    Precision:     {DIRECTIONAL_BASELINE['precision']:.1%}")
    print(f"    Signals/day:   {DIRECTIONAL_BASELINE['signals_per_day']}")
    print(f"    Problem:       33% direction ceiling (random on up/down)")
    print()
    print("  Magnitude experiment results → see above output or:")
    results_path = Path(args.output) / "magnitude_results.json"
    print(f"    {results_path}")
    print()
    print("  Scenarios:")
    print("    >= 60% precision → MAGNITUDE MODEL WORKS  → trade straddles")
    print("    50-60% precision → MARGINAL IMPROVEMENT   → consider high threshold")
    print("    < 50% precision  → NO IMPROVEMENT          → fundamental limit")
    print()
    print("  Models saved to:")
    print(f"    {Path(args.output) / 'models'}/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
