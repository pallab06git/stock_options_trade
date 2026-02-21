# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""ML pipeline CLI — registered as the ``ml`` subgroup of the main CLI.

Usage
-----
    python -m src.cli ml generate-features --start-date 2025-03-03 --end-date 2026-01-31
    python -m src.cli ml train --start-date 2025-03-03 --end-date 2026-01-31
    python -m src.cli ml train --start-date 2025-03-03 --end-date 2026-01-31 --model-version v2
    python -m src.cli ml feature-importance --model-path models/xgboost_v1.pkl
    python -m src.cli ml feature-importance --model-path models/xgboost_v1.pkl --top-n 15
    python -m src.cli ml backtest --model-path models/xgboost_v1.pkl
    python -m src.cli ml backtest --model-path models/xgboost_v1.pkl --start-date 2025-03-03

Config loading
--------------
All commands load the full merged config via ``ConfigLoader(config_dir)``.
``config/ml_settings.yaml`` is automatically included.  CLI flags override
individual config values where supported.
"""

import sys

import click

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logger


@click.group("ml")
def ml_cli():
    """ML pipeline: feature engineering, training, evaluation."""
    pass


# ---------------------------------------------------------------------------
# generate-features
# ---------------------------------------------------------------------------


@ml_cli.command("generate-features")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--start-date",
    default=None,
    help="Override feature_engineering.start_date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Override feature_engineering.end_date (YYYY-MM-DD).",
)
def generate_features(config_dir, start_date, end_date):
    """Engineer ML features from options + SPY minute bars.

    Reads downloaded minute Parquet files, computes 66+ features per bar,
    attaches forward-looking binary target labels, and writes one
    ``{date}_features.csv`` per trading day to the features directory.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        # Apply CLI overrides into config
        fe_cfg = config.setdefault("feature_engineering", {})
        if start_date:
            fe_cfg["start_date"] = start_date
        if end_date:
            fe_cfg["end_date"] = end_date

        effective_start = fe_cfg.get("start_date")
        effective_end = fe_cfg.get("end_date")

        from src.processing.ml_feature_engineer import MLFeatureEngineer

        engineer = MLFeatureEngineer(config)
        stats = engineer.run(
            start_date=effective_start,
            end_date=effective_end,
        )

        click.echo("\n--- ML Feature Engineering Summary ---")
        click.echo(f"Date range:       {effective_start} → {effective_end}")
        click.echo(f"Dates processed:  {stats.get('dates_processed', 0)}")
        click.echo(f"Dates skipped:    {stats.get('dates_skipped', 0)}")
        click.echo(f"Dates failed:     {stats.get('dates_failed', 0)}")
        click.echo(f"Total rows:       {stats.get('total_rows', 0)}")
        click.echo(f"Total features:   {stats.get('n_features', 0)}")
        click.echo(f"Positive rate:    {stats.get('positive_rate', 0):.2%}")
        click.echo(f"Output dir:       {fe_cfg.get('features_dir', 'data/processed/features')}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@ml_cli.command("train")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD). No lower bound if omitted.",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD). No upper bound if omitted.",
)
@click.option(
    "--model-version",
    default=None,
    help="Override ml_training.xgboost.model_version (e.g. v2).",
)
def train_model(config_dir, start_date, end_date, model_version):
    """Train an XGBoost classifier on engineered feature CSVs.

    Pipeline: load CSVs → chronological split → undersample training set →
    fit XGBoost → evaluate on validation set → save model artifact →
    log metrics JSON.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        # Apply CLI overrides
        if model_version:
            config.setdefault("ml_training", {}).setdefault("xgboost", {})[
                "model_version"
            ] = model_version

        features_dir = config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        models_dir = config.get("ml_paths", {}).get("models_dir", "models")
        logs_dir = config.get("ml_paths", {}).get(
            "training_logs_dir", "data/logs/training"
        )

        from src.ml.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(config)
        metrics = trainer.train(
            features_dir=features_dir,
            start_date=start_date,
            end_date=end_date,
            models_dir=models_dir,
            logs_dir=logs_dir,
        )

        click.echo("\n--- XGBoost Training Summary ---")
        click.echo(f"Date range:       {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"Train rows:       {metrics['train_rows']}")
        click.echo(f"Val rows:         {metrics['val_rows']}")
        click.echo(f"Test rows:        {metrics['test_rows']}")
        click.echo(f"Features:         {metrics['n_features']}")
        click.echo(f"Best iteration:   {metrics['best_iteration']}")
        click.echo(f"Val accuracy:     {metrics['val_accuracy']:.4f}")
        click.echo(f"Val precision:    {metrics['val_precision']:.4f}")
        click.echo(f"Val recall:       {metrics['val_recall']:.4f}")
        click.echo(f"Val F1:           {metrics['val_f1']:.4f}")
        click.echo(f"Val ROC-AUC:      {metrics['val_roc_auc']:.4f}")
        click.echo(f"Model saved:      {metrics['model_path']}")
        click.echo(f"Metrics log:      {metrics['log_path']}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# feature-importance
# ---------------------------------------------------------------------------


@ml_cli.command("feature-importance")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the .pkl model artifact (e.g. models/xgboost_v1.pkl).",
)
@click.option(
    "--importance-type",
    default=None,
    type=click.Choice(["weight", "gain", "cover", "total_gain", "total_cover"]),
    help="Override ml_training.feature_importance.importance_type.",
)
@click.option(
    "--top-n",
    default=None,
    type=int,
    help="Override ml_training.feature_importance.top_n.",
)
def feature_importance(config_dir, model_path, importance_type, top_n):
    """Analyze and display feature importances from a trained model artifact.

    Loads the .pkl artifact, extracts XGBoost feature scores, saves a CSV
    report, and prints an ASCII bar chart of the top-N features.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        # Apply CLI overrides
        fi_cfg = config.setdefault("ml_training", {}).setdefault(
            "feature_importance", {}
        )
        if importance_type:
            fi_cfg["importance_type"] = importance_type
        if top_n is not None:
            fi_cfg["top_n"] = top_n

        output_dir = fi_cfg.get(
            "output_dir", "data/reports/feature_importance"
        )

        from src.ml.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer(config)
        df = analyzer.analyze(model_path, output_dir=output_dir)

        click.echo(f"\n{analyzer.plot_summary(df)}")
        click.echo(f"\nFull report saved to: {output_dir}/")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@ml_cli.command("backtest")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the .pkl model artifact (e.g. models/xgboost_v1.pkl).",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD). No lower bound if omitted.",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD). No upper bound if omitted.",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Override the model artifact's prediction threshold (0.0–1.0). "
    "Useful for high-precision trading (e.g. 0.90). "
    "Omit to use the value stored in the artifact.",
)
def backtest(config_dir, model_path, start_date, end_date, threshold):
    """Run ML model backtest on the chronological test split.

    Loads the model artifact, splits features chronologically, predicts on
    the test set (never seen during training), and reports precision / recall /
    lift over the random baseline.  Saves a per-trade CSV and a JSON metrics
    report.

    Use --threshold to override the model's default threshold (stored at
    training time) for high-precision signal filtering, e.g. --threshold 0.90.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        features_dir = config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        output_dir = (
            config.get("ml_training", {})
            .get("backtest", {})
            .get("output_dir", "data/reports/backtest")
        )

        from src.ml.backtest import ModelBacktester

        backtester = ModelBacktester(config)
        result = backtester.run(
            model_path=model_path,
            features_dir=features_dir,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            threshold=threshold,
        )

        m = result["metrics"]
        lift_str = f"{m['lift']:.2f}x" if m["lift"] is not None else "n/a"
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"

        click.echo("\n--- ML Backtest Summary (test split only) ---")
        click.echo(f"Date range:       {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"Test rows:        {m['n_test_rows']}")
        click.echo(f"Signals fired:    {m['n_signals']}  ({m['signal_rate']:.2%} of bars)")
        click.echo(f"True positives:   {m['n_true_positives']}")
        click.echo(f"False positives:  {m['n_false_positives']}")
        click.echo(f"Precision:        {m['precision']:.4f}")
        click.echo(f"Recall:           {m['recall']:.4f}")
        click.echo(f"F1:               {m['f1']:.4f}")
        click.echo(f"ROC-AUC:          {auc_str}")
        click.echo(f"Positive rate:    {m['positive_rate_test']:.2%}  (baseline)")
        if m["avg_gain_all_bars"] is not None:
            click.echo(f"Avg gain (all):   {m['avg_gain_all_bars']:.4f}  (random baseline)")
        if m["avg_gain_signals"] is not None:
            click.echo(f"Avg gain (signals): {m['avg_gain_signals']:.4f}")
        click.echo(f"Lift:             {lift_str}")
        click.echo(f"Trades CSV:       {result['trades_path']}")
        click.echo(f"Report JSON:      {result['report_path']}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# find-threshold
# ---------------------------------------------------------------------------


@ml_cli.command("find-threshold")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the .pkl model artifact (e.g. models/xgboost_v2.pkl).",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD). No lower bound if omitted.",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD). No upper bound if omitted.",
)
@click.option(
    "--min-precision",
    default=0.90,
    type=float,
    show_default=True,
    help="Minimum acceptable precision (0.0–1.0).",
)
@click.option(
    "--output",
    default=None,
    help="Optional path to save the full threshold sweep as a CSV file.",
)
def find_threshold(config_dir, model_path, start_date, end_date, min_precision, output):
    """Find the probability threshold that achieves minimum precision.

    Sweeps thresholds 0.50–0.99 on the VALIDATION split (never the test set)
    and finds the lowest threshold satisfying --min-precision while keeping
    recall as high as possible.  Use the reported threshold with ``backtest
    --threshold`` to measure its effect on the unseen test set.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        features_dir = config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )

        import numpy as np
        import joblib

        from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
        from src.ml.data_splitter import DataSplitter
        from src.ml.evaluate import find_optimal_threshold_for_precision

        # Load artifact
        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]

        # Load and split features — use VALIDATION set only
        df = load_features(features_dir, start_date, end_date)
        if df.empty:
            raise ValueError(f"No feature data found in {features_dir}")

        splitter = DataSplitter(config)
        _, val_df, _ = splitter.split(df)
        if val_df.empty:
            raise ValueError("Validation split is empty — not enough dates")

        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["target"].values.astype(np.int8)

        result = find_optimal_threshold_for_precision(
            model, X_val, y_val, min_precision=min_precision
        )

        click.echo(f"\n--- Threshold Analysis (validation set) ---")
        click.echo(f"Model:            {model_path}")
        click.echo(f"Val rows:         {len(val_df)}")
        click.echo(f"Min precision:    {min_precision:.0%}")

        if result["achievable"]:
            click.echo(f"\n✅ {min_precision:.0%} precision IS achievable")
            click.echo(f"Optimal threshold:  {result['optimal_threshold']:.2f}")
            click.echo(f"Achieved precision: {result['achieved_precision']:.2%}")
            click.echo(f"Achieved recall:    {result['achieved_recall']:.2%}")
            click.echo(f"Signals on val:     {result['n_signals']}  "
                       f"({result['signal_rate']:.2%} of bars)")
            click.echo(
                f"\n→ Run backtest with:  ml backtest --model-path {model_path} "
                f"--threshold {result['optimal_threshold']:.2f}"
            )
        else:
            analysis = result["analysis_df"]
            best_prec = analysis["precision"].max(skipna=True)
            click.echo(f"\n❌ {min_precision:.0%} precision is NOT achievable on this model")
            click.echo(f"Best precision found: {best_prec:.2%}")
            click.echo(
                "Consider: lower --min-precision, retrain with more data, "
                "or try a higher-capacity model."
            )

        # Print summary table: every 5th threshold
        click.echo("\nThreshold sweep (every 5th step):")
        click.echo(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'Signals':>8}")
        click.echo("-" * 42)
        df_sweep = result["analysis_df"]
        for _, row in df_sweep[df_sweep.index % 5 == 0].iterrows():
            prec_str = (
                f"{row['precision']:.2%}"
                if row["precision"] == row["precision"]  # NaN check
                else "   n/a  "
            )
            click.echo(
                f"{row['threshold']:>10.2f}  {prec_str:>10}  "
                f"{row['recall']:>7.2%}  {int(row['n_signals']):>8}"
            )

        if output:
            result["analysis_df"].to_csv(output, index=False)
            click.echo(f"\nFull sweep saved to: {output}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# benchmark-speed
# ---------------------------------------------------------------------------


@ml_cli.command("benchmark-speed")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the .pkl model artifact (e.g. models/xgboost_v2.pkl).",
)
@click.option(
    "--n-iterations",
    default=1000,
    type=int,
    show_default=True,
    help="Number of timed prediction calls.",
)
def benchmark_speed(config_dir, model_path, n_iterations):
    """Benchmark single-sample prediction latency.

    Measures how fast the model can produce a buy/no-buy decision on a single
    feature vector.  Requirement for real-time trading: p99 < 100 ms.

    Runs a warm-up phase before timing to avoid cold-start bias.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        import numpy as np
        import joblib

        from src.ml.benchmark import benchmark_prediction_speed

        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]

        # Use a zero-vector sample — latency is independent of feature values
        sample = np.zeros(len(feature_cols), dtype=np.float32)

        result = benchmark_prediction_speed(model, sample, n_iterations=n_iterations)

        req_str = "✅ Meets <100ms" if result["meets_100ms_requirement"] else "❌ Exceeds 100ms"

        click.echo(f"\n--- Prediction Speed Benchmark ---")
        click.echo(f"Model:          {model_path}")
        click.echo(f"Features:       {len(feature_cols)}")
        click.echo(f"Iterations:     {result['n_iterations']}")
        click.echo(f"Mean latency:   {result['mean_latency_ms']:.3f} ms")
        click.echo(f"P50 latency:    {result['p50_latency_ms']:.3f} ms")
        click.echo(f"P95 latency:    {result['p95_latency_ms']:.3f} ms")
        click.echo(f"P99 latency:    {result['p99_latency_ms']:.3f} ms")
        click.echo(f"Max latency:    {result['max_latency_ms']:.3f} ms")
        click.echo(f"Requirement:    {req_str}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# analyze-errors
# ---------------------------------------------------------------------------


@ml_cli.command("analyze-errors")
@click.option(
    "--trades",
    "trades_path",
    required=True,
    help="Path to the per-trade CSV produced by 'ml backtest' "
    "(e.g. data/reports/backtest/xgboost_v2_trades_*.csv).",
)
@click.option(
    "--output",
    default=None,
    help="Optional path to save the risk report as a JSON file.",
)
def analyze_errors(trades_path, output):
    """Analyse false positive severity and recommend stop-loss levels.

    Reads the per-trade CSV from 'ml backtest', isolates false positive signals
    (predicted buy but price did NOT spike ≥20%), and characterises the
    downside risk using the min_loss_120m column (worst drawdown in the
    120-minute forward window from entry).

    Outputs:
      - ASCII histogram of the drawdown distribution
      - Loss severity bucket breakdown (0–5%, 5–10%, …, >20%)
      - Stop-loss trigger rates for 5/10/15/20% stops
      - Conservative / moderate / aggressive stop-loss recommendations
    """
    try:
        import json

        from src.ml.error_analyzer import PredictionErrorAnalyzer

        analyzer = PredictionErrorAnalyzer()

        # ── Load false positives ──────────────────────────────────────────
        fp_df = analyzer.load_false_positives(trades_path)

        if fp_df.empty:
            click.echo("No false positives found in trades CSV — nothing to analyse.")
            return

        # ── ASCII histogram ───────────────────────────────────────────────
        click.echo("\n" + analyzer.plot_ascii(fp_df))

        # ── Risk report ───────────────────────────────────────────────────
        report = analyzer.generate_risk_report(fp_df)

        click.echo("\n--- False Positive Risk Report ---")
        click.echo(f"Total false positives:    {report['total_false_positives']}")
        click.echo(
            f"Price never below entry:  "
            f"{report['pct_price_never_below_entry']:.1%}"
        )
        click.echo("")
        click.echo("Worst-drawdown distribution (min_loss_120m):")
        click.echo(f"  Mean:    {report['mean_worst_drawdown_pct']:>7.2f}%")
        click.echo(f"  Median:  {report['median_worst_drawdown_pct']:>7.2f}%")
        click.echo(f"  P75:     {report['p75_worst_drawdown_pct']:>7.2f}%")
        click.echo(f"  P90:     {report['p90_worst_drawdown_pct']:>7.2f}%")
        click.echo(f"  Worst:   {report['max_worst_drawdown_pct']:>7.2f}%")
        click.echo("")
        click.echo("Loss bucket breakdown (% of FPs in each range):")
        click.echo(f"  Price >= entry (no loss):  {report['pct_never_below_entry']:.1%}")
        click.echo(f"  Loss   0-5%:   {report['pct_0_to_5pct']:.1%}")
        click.echo(f"  Loss   5-10%:  {report['pct_5_to_10pct']:.1%}")
        click.echo(f"  Loss  10-15%:  {report['pct_10_to_15pct']:.1%}")
        click.echo(f"  Loss  15-20%:  {report['pct_15_to_20pct']:.1%}")
        click.echo(f"  Loss  >20%:    {report['pct_over_20pct']:.1%}")

        # ── Stop-loss impact ──────────────────────────────────────────────
        impact = analyzer.stop_loss_impact(fp_df)

        click.echo("")
        click.echo("Stop-loss trigger analysis:")
        click.echo(
            f"  {'Stop level':>12}  {'Triggered':>10}  {'Triggered%':>11}  "
            f"{'Uncaught':>9}  {'Uncaught max loss':>18}"
        )
        click.echo("  " + "-" * 68)
        for stop, stats in sorted(impact.items()):
            uncaught_str = (
                f"{stats['uncaught_max_loss_pct']:>+.1f}%"
                if stats["uncaught_count"] > 0
                else "     n/a"
            )
            click.echo(
                f"  {stop:>+11.0f}%  "
                f"{stats['triggered_count']:>10}  "
                f"{stats['triggered_pct']:>10.1%}  "
                f"{stats['uncaught_count']:>9}  "
                f"{uncaught_str:>18}"
            )

        # ── Recommendations ───────────────────────────────────────────────
        click.echo("")
        click.echo("Stop-loss recommendations:")
        click.echo(
            f"  Conservative (p75): {report['stop_loss_conservative_pct']:>+.1f}%"
            f"  (protects against 25% of worst FP drawdowns)"
        )
        click.echo(
            f"  Moderate     (p90): {report['stop_loss_moderate_pct']:>+.1f}%"
            f"  (protects against 10% of worst FP drawdowns)"
        )
        click.echo(
            f"  Aggressive   (p95): {report['stop_loss_aggressive_pct']:>+.1f}%"
            f"  (protects against 5% of worst FP drawdowns)"
        )

        # ── Expected value analysis ───────────────────────────────────────
        n_total = len(fp_df) + int(
            # True positives = total trades - false positives
            # We can compute precision from fp_df metadata if available
            0
        )
        # Load total trade count from the CSV header to compute precision
        import pandas as _pd

        all_trades = _pd.read_csv(trades_path)
        n_signals = len(all_trades)
        n_fp = len(fp_df)
        n_tp = n_signals - n_fp
        precision = n_tp / n_signals if n_signals > 0 else 0.0

        conservative_sl = report["stop_loss_conservative_pct"]
        ev = analyzer.calculate_expected_value(
            precision=precision,
            avg_win_pct=20.0,  # model's target threshold
            stop_loss_pct=conservative_sl,
        )

        click.echo("")
        click.echo("Expected value analysis (conservative stop-loss):")
        click.echo(f"  Win rate:         {ev['win_rate']:.1%}")
        click.echo(f"  Avg win:          +{ev['avg_win_pct']:.1f}%")
        click.echo(f"  Avg loss (stop):  {ev['avg_loss_pct']:+.1f}%")
        click.echo(f"  EV per trade:     {ev['expected_value_pct']:+.2f}%")
        click.echo(
            f"  Breakeven rate:   {ev['breakeven_win_rate']:.1%}"
            if ev["breakeven_win_rate"] is not None
            else "  Breakeven rate:   n/a"
        )
        click.echo(
            f"  Strategy:         {'Profitable' if ev['profitable'] else 'Unprofitable'}"
        )

        # ── Optional JSON save ────────────────────────────────────────────
        if output:
            report["expected_value"] = ev
            with open(output, "w") as fh:
                json.dump(report, fh, indent=2)
            click.echo(f"\nRisk report saved to: {output}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
