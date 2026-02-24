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


# ---------------------------------------------------------------------------
# threshold-analysis
# ---------------------------------------------------------------------------


@ml_cli.command("threshold-analysis")
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
    help="Earliest feature date (YYYY-MM-DD).  No lower bound if omitted.",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date (YYYY-MM-DD).  No upper bound if omitted.",
)
@click.option(
    "--min-threshold",
    default=0.70,
    type=float,
    show_default=True,
    help="Lowest threshold to evaluate.",
)
@click.option(
    "--max-threshold",
    default=0.95,
    type=float,
    show_default=True,
    help="Highest threshold to evaluate.",
)
@click.option(
    "--step",
    default=0.01,
    type=float,
    show_default=True,
    help="Step size between thresholds.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for CSV/JSON reports.  "
    "Defaults to data/reports/threshold_analysis.",
)
def threshold_analysis(
    config_dir,
    model_path,
    start_date,
    end_date,
    min_threshold,
    max_threshold,
    step,
    output,
):
    """Comprehensive threshold sensitivity analysis with monthly breakdown.

    Sweeps probability thresholds across the full feature dataset and reports
    signal counts, precision, recall, TP profit / FP loss / FN missed-gain
    distributions, and expected value — broken down by full-year aggregate,
    calendar month, and trading day.

    NOTE: Analysis spans the entire date range including training data.
    Precision on training dates will be optimistic.  Use 'ml backtest
    --threshold X' for unbiased held-out test-set evaluation.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        import joblib
        import numpy as _np

        from src.ml.threshold_analyzer import ThresholdAnalyzer

        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        features_dir = config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("data/reports/threshold_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build threshold list (round to 2dp to avoid floating-point drift)
        thresholds = [
            round(t, 2)
            for t in _np.arange(min_threshold, max_threshold + step * 0.5, step)
        ]

        # Load artifact
        artifact = joblib.load(model_path)

        click.echo(f"\n--- Threshold Sensitivity Analysis ---")
        click.echo(f"Model:          {model_path}")
        click.echo(f"Features dir:   {features_dir}")
        click.echo(f"Date range:     {start_date or 'all'} → {end_date or 'all'}")
        click.echo(
            f"Thresholds:     {thresholds[0]:.2f} – {thresholds[-1]:.2f} "
            f"(step {step:.2f}, n={len(thresholds)})"
        )
        click.echo(
            "NOTE: includes training data — use 'ml backtest' for test-only metrics\n"
        )

        analyzer = ThresholdAnalyzer()
        results = analyzer.analyze_full_year(
            artifact=artifact,
            features_dir=features_dir,
            thresholds=thresholds,
            start_date=start_date,
            end_date=end_date,
        )

        aggregate_df = results["aggregate"]
        monthly_df = results["monthly"]
        daily_df = results["daily"]

        click.echo(
            f"Loaded:  {results['total_samples']:,} rows | "
            f"{results['n_dates']} dates | {results['n_months']} months"
        )
        click.echo(
            f"Range:   {results['date_range'][0]}  →  {results['date_range'][1]}\n"
        )

        # ── Save CSVs ─────────────────────────────────────────────────
        aggregate_df.to_csv(out_dir / "aggregate_analysis.csv", index=False)
        monthly_df.to_csv(out_dir / "monthly_breakdown.csv", index=False)
        daily_df.to_csv(out_dir / "daily_breakdown.csv", index=False)

        # ── Monthly summary pivot ─────────────────────────────────────
        monthly_summary = analyzer.generate_monthly_summary(monthly_df)
        monthly_summary.to_csv(out_dir / "monthly_summary.csv", index=False)

        # ── ASCII bar chart ───────────────────────────────────────────
        chart_str = analyzer.plot_monthly_signals(monthly_summary)
        (out_dir / "monthly_signals_chart.txt").write_text(chart_str)
        click.echo(chart_str)

        # ── Aggregate comparison at key thresholds ────────────────────
        key_ts = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        key_mask = aggregate_df["threshold"].round(2).isin(key_ts)
        comparison = aggregate_df[key_mask].copy()
        display_cols = [
            c
            for c in [
                "threshold",
                "total_signals",
                "signal_rate",
                "precision",
                "recall",
                "tp_profit_pct_avg",
                "tp_profit_pct_median",
                "fp_loss_pct_avg",
                "fp_loss_pct_median",
                "fn_missed_pct_avg",
                "fn_missed_pct_median",
                "expected_value_pct",
            ]
            if c in comparison.columns
        ]
        click.echo("\n--- Aggregate: Key Thresholds ---")
        with _np.printoptions(precision=3):
            import pandas as _pd

            with _pd.option_context("display.float_format", "{:.3f}".format):
                click.echo(comparison[display_cols].to_string(index=False))

        comparison.to_csv(out_dir / "aggregate_key_thresholds.csv", index=False)

        # ── Monthly summary table ─────────────────────────────────────
        click.echo("\n--- Monthly Summary (signals | precision | EV) ---")
        click.echo(monthly_summary.to_string(index=False))

        # ── Optimal threshold search ──────────────────────────────────
        click.echo("\n--- Optimal Threshold Recommendations ---")

        # min_signals: require at least n_dates / 10 total signals across full dataset
        min_sig_ev = max(10, results["n_dates"] // 10)
        min_sig_safe = max(5, results["n_dates"] // 20)

        opt_ev = analyzer.find_optimal_threshold(
            aggregate_df,
            optimization_metric="expected_value_pct",
            min_precision=0.90,
            min_signals=min_sig_ev,
        )
        if opt_ev["status"] == "SUCCESS":
            m = opt_ev["metrics"]
            nd = results["n_dates"]
            click.echo(
                f"\n1. Max expected value (precision >= 90%):\n"
                f"   Threshold:      {opt_ev['optimal_threshold']:.2f}\n"
                f"   Precision:      {m['precision']:.1%}\n"
                f"   Recall:         {m['recall']:.1%}\n"
                f"   Signals total:  {m['total_signals']}\n"
                f"   Signals/day:    {m['total_signals'] / nd:.1f}\n"
                f"   EV/trade:       {m['expected_value_pct']:+.2f}%\n"
                f"   TP avg profit:  {m.get('tp_profit_pct_avg') or 'n/a'}%\n"
                f"   FP avg loss:    {m.get('fp_loss_pct_avg') or 'n/a'}%\n"
                f"   FN avg missed:  {m.get('fn_missed_pct_avg') or 'n/a'}%"
            )
        else:
            click.echo(f"\n1. {opt_ev['message']}")

        opt_safe = analyzer.find_optimal_threshold(
            aggregate_df,
            optimization_metric="precision",
            min_precision=0.93,
            min_signals=min_sig_safe,
        )
        if opt_safe["status"] == "SUCCESS":
            m2 = opt_safe["metrics"]
            click.echo(
                f"\n2. Max precision (>= 93%):\n"
                f"   Threshold:      {opt_safe['optimal_threshold']:.2f}\n"
                f"   Precision:      {m2['precision']:.1%}\n"
                f"   Signals/day:    {m2['total_signals'] / results['n_dates']:.1f}\n"
                f"   EV/trade:       {m2['expected_value_pct']:+.2f}%"
            )
        else:
            click.echo(f"\n2. {opt_safe['message']}")

        # ── Save recommendations ──────────────────────────────────────
        recommendations = {
            "max_expected_value": opt_ev,
            "max_precision": opt_safe,
            "metadata": {
                "model_path": str(model_path),
                "date_range": list(results["date_range"]),
                "total_samples": results["total_samples"],
                "n_dates": results["n_dates"],
                "n_months": results["n_months"],
                "thresholds_swept": thresholds,
            },
        }
        with open(out_dir / "recommendations.json", "w") as fh:
            _json.dump(recommendations, fh, indent=2, default=str)

        click.echo(f"\nReports saved to: {out_dir}/")
        click.echo(
            f"  aggregate_analysis.csv    ({len(aggregate_df)} rows)\n"
            f"  monthly_breakdown.csv     ({len(monthly_df)} rows)\n"
            f"  daily_breakdown.csv       ({len(daily_df)} rows)\n"
            f"  monthly_summary.csv\n"
            f"  monthly_signals_chart.txt\n"
            f"  aggregate_key_thresholds.csv\n"
            f"  recommendations.json"
        )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# explain-signal
# ---------------------------------------------------------------------------


@ml_cli.command("explain-signal")
@click.option(
    "--model-path",
    required=True,
    help="Path to the .pkl model artifact (e.g. models/xgboost_v2.pkl).",
)
@click.option(
    "--features-file",
    required=True,
    help="Path to a features CSV file (one of the data/processed/features/*.csv files).",
)
@click.option(
    "--ticker",
    default=None,
    help="Filter to a specific option ticker (e.g. O:SPY250321C00580000). "
    "If omitted, the row with the highest model probability is selected.",
)
@click.option(
    "--row-index",
    default=None,
    type=int,
    help="Explain a specific 0-based row index instead of highest probability.",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Override the model artifact's decision threshold for the explanation "
    "header (0.0–1.0). Defaults to threshold stored in the artifact.",
)
def explain_signal(model_path, features_file, ticker, row_index, threshold):
    """Explain why the model fired (or would fire) a buy signal.

    Loads a features CSV, selects the row with the highest model confidence
    (or a specific --ticker / --row-index), computes SHAP values, and prints
    a detailed explanation showing:

    \\b
    - Top 10 contributing features with SHAP values
    - Human-readable interpretation of each feature
    - Risk factors (features pushing toward no-signal)
    - Confidence vs threshold margin
    """
    try:
        import joblib
        import numpy as np
        import pandas as pd

        from src.ml.explainer import SignalExplainer

        # ── Load model artifact ───────────────────────────────────────────
        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        model_threshold = float(artifact.get("threshold", 0.90))
        effective_threshold = threshold if threshold is not None else model_threshold

        # ── Load features ─────────────────────────────────────────────────
        df = pd.read_csv(features_file)
        if df.empty:
            raise ValueError(f"Features file is empty: {features_file}")

        # Apply ticker filter
        if ticker is not None:
            mask = df.get("ticker", pd.Series(dtype=str)) == ticker
            if not mask.any():
                raise ValueError(f"Ticker '{ticker}' not found in {features_file}")
            df = df[mask].reset_index(drop=True)

        # Select row
        if row_index is not None:
            if row_index >= len(df) or row_index < 0:
                raise ValueError(
                    f"--row-index {row_index} out of range (0–{len(df) - 1})"
                )
            chosen_df = df.iloc[[row_index]]
        else:
            # Select row with highest predicted probability
            X_all = df[feature_cols].fillna(0).values.astype(np.float32)
            probas = model.predict_proba(X_all)[:, 1]
            best_idx = int(np.argmax(probas))
            chosen_df = df.iloc[[best_idx]]
            click.echo(
                f"Selected row {best_idx} with highest confidence "
                f"({probas[best_idx]:.1%}) from {len(df)} rows."
            )

        row = chosen_df.iloc[0]
        features_dict = {
            col: float(row[col]) for col in feature_cols if col in row.index
        }

        # Predict probability for chosen row
        X_row = np.array(
            [features_dict.get(c, 0.0) for c in feature_cols], dtype=np.float32
        )
        pred_proba = float(model.predict_proba([X_row])[0][1])

        # Contextual header
        ticker_label = str(row.get("ticker", "unknown"))
        date_label = str(row.get("date", "unknown"))
        time_label = (
            f"{int(row.get('hour_et', 0)):02d}:{int(row.get('minute_et', 0)):02d} ET"
            if "hour_et" in row.index
            else ""
        )

        click.echo(f"\nFile:       {features_file}")
        click.echo(f"Ticker:     {ticker_label}")
        click.echo(f"Date/Time:  {date_label}  {time_label}")
        click.echo(f"Model:      {model_path}")
        click.echo("")

        # ── Build explainer and explain ───────────────────────────────────
        explainer = SignalExplainer(model, feature_cols)
        explanation = explainer.explain_signal(
            features=features_dict,
            prediction_proba=pred_proba,
            threshold=effective_threshold,
        )
        click.echo(explanation)

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


@ml_cli.command("walk-forward")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--threshold",
    default=0.67,
    type=float,
    show_default=True,
    help="Probability threshold applied to every test split (default: 0.67, same as backtest).",
)
@click.option(
    "--train-months",
    default=3,
    type=int,
    show_default=True,
    help="Training window size in calendar months.",
)
@click.option(
    "--test-months",
    default=1,
    type=int,
    show_default=True,
    help="Test window size in calendar months.  Also the slide step.",
)
@click.option(
    "--position-size",
    default=12_500.0,
    type=float,
    show_default=True,
    help="USD position size per trade for simulation (default $12,500).",
)
@click.option(
    "--target-gain",
    default=30.0,
    type=float,
    show_default=True,
    help="Take-profit percentage for simulation (default 30%).",
)
@click.option(
    "--stop-loss",
    default=-12.0,
    type=float,
    show_default=True,
    help="Stop-loss percentage for simulation (default -12%).",
)
@click.option(
    "--show-trades",
    is_flag=True,
    default=False,
    help="Print individual trade details for each test month.",
)
@click.option(
    "--output",
    default=None,
    help="Directory to save the JSON results (default: data/reports/walk_forward).",
)
def walk_forward(
    config_dir, threshold, train_months, test_months,
    position_size, target_gain, stop_loss, show_trades, output,
):
    """Walk-forward validation to assess model stability across time.

    Re-trains XGBoost from scratch on each rolling window and evaluates on
    the following unseen month.  Shows whether the 91.9% backtest precision
    is typical or an outlier across different market regimes.

    \\b
    Split scheme (default: 3-month train, 1-month test):
      Split 1: Train [Mar Apr May] → Test [Jun]
      Split 2: Train [Apr May Jun] → Test [Jul]
      ...
      Split 9: Train [Nov Dec Jan] → Test [Feb]

    NOTE: each fold re-trains a fresh model — results are slower than
    'ml backtest' but give a true out-of-sample estimate for every month.
    """
    import json as _json
    from pathlib import Path as _Path

    from src.ml.trade_simulator import TradeSimulator
    from src.ml.walk_forward_validator import WalkForwardValidator

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        features_dir = config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("data/reports/walk_forward")
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n--- Walk-Forward Validation ---")
        click.echo(f"Features dir:   {features_dir}")
        click.echo(f"Threshold:      {threshold:.2f}")
        click.echo(f"Train window:   {train_months} month(s)")
        click.echo(f"Test window:    {test_months} month(s)")
        click.echo(f"Position size:  ${position_size:,.0f} per trade")
        click.echo(f"Target gain:    +{target_gain:.0f}%  |  Stop loss: {stop_loss:.0f}%")
        click.echo(
            "NOTE: re-trains XGBoost for each split — this may take several minutes\n"
        )

        simulator = TradeSimulator(
            position_size_usd=position_size,
            target_gain_pct=target_gain,
            stop_loss_pct=stop_loss,
        )

        validator = WalkForwardValidator(
            features_dir=features_dir,
            train_window_months=train_months,
            test_window_months=test_months,
            simulator=simulator,
        )

        # Show date splits preview
        splits = validator.get_date_splits()
        click.echo(f"Generated {len(splits)} train-test split(s):")
        for i, (ts, te, vs, ve) in enumerate(splits, 1):
            click.echo(f"  Split {i:2d}: Train {ts} → {te}  |  Test {vs} → {ve}")
        click.echo("")

        summary = validator.run_validation(threshold=threshold)

        if summary["status"] != "SUCCESS":
            click.echo(f"Validation failed: {summary.get('message', summary['status'])}")
            for r in summary.get("splits", []):
                if r["status"] != "SUCCESS":
                    click.echo(
                        f"  Split {r.get('split_index','?')}: {r['status']} — {r.get('reason','')}"
                    )
            sys.exit(1)

        # ── Per-split results table ────────────────────────────────────
        click.echo("--- Per-Split Results ---")
        click.echo(
            f"{'Split':>6}  {'Test Period':>22}  {'Signals':>8}  {'Prec':>7}  "
            f"{'TP':>5}  {'FP':>5}  {'EV%':>7}"
        )
        click.echo("  " + "-" * 70)
        for r in summary["splits"]:
            idx = r.get("split_index", "?")
            if r["status"] == "SUCCESS":
                click.echo(
                    f"  {idx:>4}  {r['test_period']:>22}  "
                    f"{r['total_signals']:>8}  {r['precision']:>6.1%}  "
                    f"{r['true_positives']:>5}  {r['false_positives']:>5}  "
                    f"{r['expected_value_pct']:>+7.1f}"
                )
            else:
                click.echo(
                    f"  {idx:>4}  {r['test_period']:>22}  "
                    f"{'[SKIP]':>8}  {'n/a':>7}  {'--':>5}  {'--':>5}  {'n/a':>7}"
                    f"  ({r.get('reason','')})"
                )

        # ── ASCII bar chart ────────────────────────────────────────────
        click.echo("")
        click.echo(validator.plot_results(summary))

        # ── Summary statistics ─────────────────────────────────────────
        click.echo("--- Summary Statistics ---")
        click.echo(f"Splits evaluated:  {summary['successful_splits']} / {summary['total_splits']}")
        click.echo(f"Threshold:         {summary['threshold']:.2f}")
        click.echo("")
        click.echo("Precision across test months:")
        click.echo(f"  Mean:    {summary['precision_mean']:.1%}")
        click.echo(f"  Median:  {summary['precision_median']:.1%}")
        click.echo(f"  Std dev: {summary['precision_std']:.1%}")
        click.echo(f"  Range:   {summary['precision_min']:.1%} – {summary['precision_max']:.1%}")
        click.echo("")
        click.echo("Signals per test month:")
        click.echo(f"  Mean:    {summary['signals_mean']:.0f}")
        click.echo(f"  Median:  {summary['signals_median']:.0f}")
        click.echo(f"  Range:   {summary['signals_min']} – {summary['signals_max']}")
        click.echo("")
        click.echo("Expected value per trade (%):")
        click.echo(f"  Mean:    {summary['ev_mean']:+.2f}%")
        click.echo(f"  Median:  {summary['ev_median']:+.2f}%")
        click.echo(f"  Std dev: {summary['ev_std']:.2f}%")

        # ── Interpretation ─────────────────────────────────────────────
        click.echo("")
        click.echo("--- Interpretation ---")

        mean_p = summary["precision_mean"]
        if mean_p >= 0.93:
            click.echo("Strong POC: model consistently achieves >93% precision")
            click.echo("  → Ready for production with minor tuning")
        elif mean_p >= 0.90:
            click.echo("Adequate POC: model averages 90–93% precision")
            click.echo("  → Consider improvements (feature engineering, LSTM, ensemble)")
        else:
            click.echo("Weak POC: model averages <90% precision")
            click.echo("  → Significant improvements needed")

        click.echo("")
        std_p = summary["precision_std"]
        if std_p < 0.03:
            click.echo(f"Stability: GOOD (std={std_p:.1%}) — very consistent across time")
        elif std_p < 0.06:
            click.echo(f"Stability: MODERATE (std={std_p:.1%}) — acceptable variance")
        else:
            click.echo(f"Stability: POOR (std={std_p:.1%}) — high month-to-month variance")

        # Compare against the existing backtest result
        click.echo("")
        backtest_prec = 0.919
        diff = backtest_prec - mean_p
        if abs(diff) < 0.02:
            click.echo(
                f"Existing backtest (91.9%) vs walk-forward mean ({mean_p:.1%}): "
                "consistent — backtest is representative"
            )
        elif diff > 0:
            click.echo(
                f"Existing backtest (91.9%) is {diff:.1%} above walk-forward mean "
                f"({mean_p:.1%}) — backtest period was slightly favorable"
            )
        else:
            click.echo(
                f"Existing backtest (91.9%) is {-diff:.1%} below walk-forward mean "
                f"({mean_p:.1%}) — model may perform better on average"
            )

        # ── Trade Simulation Summary ────────────────────────────────────
        click.echo("")
        click.echo("--- Trade Simulation ---")
        click.echo(
            f"Position size:  ${position_size:,.0f} per trade  |  "
            f"Target: +{target_gain:.0f}%  |  Stop: {stop_loss:.0f}%"
        )
        click.echo("")

        sim_agg = summary.get("simulation")
        if sim_agg and sim_agg.get("total_trades", 0) > 0:
            net = sim_agg["total_net_profit_usd"]
            sign = "+" if net >= 0 else ""
            click.echo(f"  Months with trades:  {sim_agg['months_simulated']}")
            click.echo(f"  Total trades:        {sim_agg['total_trades']}")
            click.echo(
                f"  Win rate:            {sim_agg['overall_win_rate']:.1%}  "
                f"({sim_agg['total_wins']} wins / {sim_agg['total_losses']} losses)"
            )
            click.echo(f"  Total net profit:    {sign}${net:,.0f}")
        else:
            click.echo("  No trades executed (all months below threshold or zero-price entries)")

        # Per-month table
        click.echo("")
        click.echo(
            f"  {'Month':>8}  {'Trades':>7}  {'Win%':>7}  {'Net P&L':>12}  "
            f"{'ROI':>7}  {'Calls':>5}  {'Puts':>5}"
        )
        click.echo("  " + "-" * 62)
        for r in summary["splits"]:
            if r["status"] != "SUCCESS":
                continue
            rep = r.get("trade_report")
            month = r["test_month"]
            if rep and rep["total_trades"] > 0:
                n = rep["total_trades"]
                wr = rep["win_rate"]
                net_m = rep["net_profit_after_fees_usd"]
                roi = rep["roi_pct"]
                calls = rep["calls_traded"]
                puts = rep["puts_traded"]
                sign_m = "+" if net_m >= 0 else ""
                click.echo(
                    f"  {month:>8}  {n:>7}  {wr:>6.1%}  "
                    f"{sign_m}${net_m:>10,.0f}  {roi:>+6.2f}%  "
                    f"{calls:>5}  {puts:>5}"
                )
            else:
                click.echo(
                    f"  {month:>8}  {'0':>7}  {'n/a':>7}  {'$0':>12}  {'n/a':>7}  "
                    f"{'--':>5}  {'--':>5}"
                )

        # Individual trade logs (--show-trades)
        if show_trades:
            click.echo("")
            click.echo("--- Individual Trades ---")
            for r in summary["splits"]:
                if r["status"] != "SUCCESS":
                    continue
                rep = r.get("trade_report")
                if not rep or rep["total_trades"] == 0:
                    continue
                month_label = r["test_month"]
                click.echo(f"\n{'=' * 70}")
                click.echo(f"  {month_label}  —  {rep['total_trades']} trades")
                click.echo(f"{'=' * 70}")
                for t in rep["trades"]:
                    click.echo(f"\nTrade #{t['trade_id']}")
                    click.echo("─" * 70)
                    click.echo("  Entry:")
                    click.echo(f"    Date/Time:          {t['entry_time']}")
                    click.echo(f"    Contract:           {t['contract_symbol']}")
                    click.echo(
                        f"    Entry price:        ${t['entry_price_per_share']:.2f}/share"
                    )
                    click.echo(
                        f"    Cost per contract:  ${t['cost_per_contract_entry']:,.0f}  "
                        f"(100 shares)"
                    )
                    click.echo(f"    Contracts:          {t['num_contracts']}")
                    click.echo(
                        f"    Actual position:    ${t['actual_position_size']:,.0f}"
                    )
                    click.echo(f"    Confidence:         {t['confidence']:.1%}")
                    click.echo("  Exit:")
                    click.echo(f"    Date/Time:          {t['exit_time']}")
                    click.echo(
                        f"    Exit price:         ${t['exit_price_per_share']:.2f}/share"
                    )
                    click.echo(
                        f"    Cost per contract:  ${t['cost_per_contract_exit']:,.0f}  "
                        f"(100 shares)"
                    )
                    click.echo(f"    Reason:             {t['exit_reason']}")
                    click.echo(f"    Time in trade:      {t['time_in_trade_minutes']:.1f} min")
                    pnl = t["profit_loss_usd"]
                    pct = t["profit_loss_pct"]
                    sign_t = "+" if (pnl or 0) >= 0 else ""
                    result_label = "WIN" if t["is_winner"] else "LOSS"
                    click.echo(
                        f"  Result: {result_label}  "
                        f"{sign_t}${pnl:,.0f}  ({sign_t}{pct:.1f}%)"
                    )
                    click.echo("─" * 70)

        # ── Save results ───────────────────────────────────────────────
        out_path = out_dir / "walk_forward_results.json"
        with open(out_path, "w") as fh:
            _json.dump(summary, fh, indent=2, default=str)
        click.echo(f"\nResults saved to: {out_path}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# full-comparison
# ---------------------------------------------------------------------------


@ml_cli.command("full-comparison")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--model-path",
    "model_paths",
    multiple=True,
    required=True,
    help=(
        "Model to include in format NAME=PATH "
        "(e.g. --model-path xgboost=models/xgboost_v2.pkl). "
        "Repeat for each model."
    ),
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing feature CSVs. Defaults to config value.",
)
@click.option(
    "--test-start-date",
    required=True,
    help="First date of the test window (YYYY-MM-DD).",
)
@click.option(
    "--test-end-date",
    required=True,
    help="Last date of the test window (YYYY-MM-DD).",
)
@click.option(
    "--thresholds",
    default="0.70,0.75,0.80,0.85,0.90,0.95",
    show_default=True,
    help="Comma-separated confidence thresholds to sweep.",
)
@click.option(
    "--position-size",
    default=12_500.0,
    type=float,
    show_default=True,
    help="USD position size per trade.",
)
@click.option(
    "--target-gain",
    default=30.0,
    type=float,
    show_default=True,
    help="Take-profit percentage.",
)
@click.option(
    "--stop-loss",
    default=-12.0,
    type=float,
    show_default=True,
    help="Stop-loss percentage (negative).",
)
@click.option(
    "--monthly-profit-target",
    default=10_000.0,
    type=float,
    show_default=True,
    help="Monthly net-profit goal for 'Meets Target' column (USD).",
)
@click.option(
    "--overlap-threshold",
    default=0.80,
    type=float,
    show_default=True,
    help="Threshold used for signal-overlap analysis.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for all results (default: data/reports/model_comparison).",
)
def full_comparison(
    config_dir,
    model_paths,
    features_dir,
    test_start_date,
    test_end_date,
    thresholds,
    position_size,
    target_gain,
    stop_loss,
    monthly_profit_target,
    overlap_threshold,
    output,
):
    """Compare multiple trained models at multiple confidence thresholds.

    For each registered model the command:

    \\b
    1. Loads the .pkl artifact from the provided path.
    2. Loads feature CSVs filtered to --test-start-date / --test-end-date.
    3. Runs TradeSimulator at every --thresholds value via ModelComparator.
    4. Prints a side-by-side comparison table.
    5. Reports signal overlap between models at --overlap-threshold.
    6. Saves JSON + CSV results to --output.
    7. Prints the command to launch the ML dashboard for the saved results.

    \\b
    Example usage:

        python -m src.cli ml full-comparison \\
            --model-path xgboost=models/xgboost_v2.pkl \\
            --model-path lightgbm=models/lightgbm_v1.pkl \\
            --test-start-date 2025-12-23 \\
            --test-end-date 2026-02-19

    Each --model-path argument must be in NAME=PATH format.
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib
    import numpy as _np
    import pandas as _pd

    from src.ml.model_comparator import ModelComparator, DEFAULT_THRESHOLDS
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        # ── Resolve paths ──────────────────────────────────────────────
        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("data/reports/model_comparison")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Parse --thresholds ─────────────────────────────────────────
        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats "
                f"(e.g. '0.70,0.80,0.90'), got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        # ── Parse --model-path NAME=PATH pairs ────────────────────────
        model_entries = []
        for spec in model_paths:
            if "=" not in spec:
                click.echo(
                    f"Error: --model-path must be NAME=PATH, got: {spec!r}",
                    err=True,
                )
                sys.exit(1)
            name, path = spec.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name or not path:
                click.echo(
                    f"Error: both name and path required in NAME=PATH, got: {spec!r}",
                    err=True,
                )
                sys.exit(1)
            model_entries.append((name, path))

        click.echo(f"\n--- Full Model Comparison ---")
        click.echo(f"Features dir:       {feat_dir}")
        click.echo(f"Test window:        {test_start_date} → {test_end_date}")
        click.echo(
            f"Thresholds:         {', '.join(f'{t:.0%}' for t in threshold_list)}"
        )
        click.echo(f"Position size:      ${position_size:,.0f} per trade")
        click.echo(f"Target / Stop:      +{target_gain:.0f}% / {stop_loss:.0f}%")
        click.echo(f"Output dir:         {out_dir}")
        click.echo("")

        # ── Load test features ─────────────────────────────────────────
        click.echo("Loading test features…")
        test_df = load_features(feat_dir, test_start_date, test_end_date)
        if test_df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{test_start_date} → {test_end_date}",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"  Loaded {len(test_df):,} rows across "
            f"{test_df['date'].nunique() if 'date' in test_df.columns else '?'} dates\n"
        )

        # ── Build ModelComparator ──────────────────────────────────────
        comparator = ModelComparator(
            position_size_usd=position_size,
            target_gain_pct=target_gain,
            stop_loss_pct=stop_loss,
            monthly_profit_target=monthly_profit_target,
        )

        # ── Register models ────────────────────────────────────────────
        for model_name, model_path in model_entries:
            click.echo(f"Loading model '{model_name}' from {model_path}…")
            try:
                artifact = joblib.load(model_path)
            except FileNotFoundError:
                click.echo(f"  ERROR: file not found: {model_path}", err=True)
                sys.exit(1)
            except Exception as exc:
                click.echo(f"  ERROR loading {model_path}: {exc}", err=True)
                sys.exit(1)

            model = artifact.get("model") or artifact
            feature_cols = artifact.get("feature_cols")
            best_params = artifact.get("params") or artifact.get("best_params") or {}
            opt_score = float(artifact.get("optimization_score", 0.0))
            model_type = artifact.get("model_type", "xgboost")

            comparator.add_model(
                name=model_name,
                model=model,
                feature_cols=feature_cols,
                best_params=best_params,
                optimization_score=opt_score,
                model_type=model_type,
            )
            click.echo(
                f"  Registered '{model_name}' "
                f"(type={model_type}, "
                f"features={len(feature_cols) if feature_cols else 'auto'})\n"
            )

        # ── Evaluate all models at all thresholds ──────────────────────
        for model_name in comparator.model_names:
            click.echo(
                f"Evaluating '{model_name}' at "
                f"{len(threshold_list)} threshold(s)…"
            )
            results = comparator.evaluate_at_thresholds(
                model_name, test_df, thresholds=threshold_list
            )

            # Quick per-threshold summary
            click.echo(
                f"  {'Threshold':>10}  {'Signals':>8}  {'Win%':>7}  {'Net Profit':>12}"
            )
            click.echo(f"  {'-' * 44}")
            for t in sorted(results.keys()):
                r = results[t]
                net = r.get("net_profit_usd", 0.0)
                sign = "+" if net >= 0 else ""
                click.echo(
                    f"  {t:>10.0%}  {r.get('total_signals', 0):>8}  "
                    f"{r.get('win_rate', 0.0):>6.1%}  "
                    f"{sign}${net:>10,.0f}"
                )
            click.echo("")

        # ── Best threshold per model ───────────────────────────────────
        click.echo("--- Best Threshold Per Model ---")
        best_per_model = comparator.get_best_threshold_per_model()
        for model_name, info in best_per_model.items():
            net = info["net_profit_usd"]
            sign = "+" if net >= 0 else ""
            click.echo(
                f"  {model_name:<20}  threshold={info['best_threshold']:.0%}  "
                f"signals={info['total_signals']}  "
                f"win_rate={info['win_rate']:.1%}  "
                f"net={sign}${net:,.0f}"
            )
        click.echo("")

        # ── Side-by-side comparison table ─────────────────────────────
        click.echo("--- Side-by-Side Comparison (80% threshold) ---")
        comp_df = comparator.generate_comparison_report(comparison_threshold=0.80)
        # Align columns for terminal output
        click.echo(comp_df.to_string(index=False))
        click.echo("")

        # ── Signal overlap ─────────────────────────────────────────────
        if len(comparator.model_names) >= 2:
            click.echo(
                f"--- Signal Overlap at {overlap_threshold:.0%} Threshold ---"
            )
            overlap = comparator.find_signal_overlap(threshold=overlap_threshold)
            click.echo(
                f"  Unique signals:    {overlap.get('total_unique_signals', 0)}"
            )
            click.echo(
                f"  All models agree:  {overlap.get('all_models_agree', 0)}"
            )
            click.echo(
                f"  Majority agree:    {overlap.get('majority_agree', 0)}"
            )
            breakdown = overlap.get("overlap_breakdown", {})
            for k, v in sorted(breakdown.items()):
                click.echo(f"  {k.replace('_', ' ')}:  {v}")
            click.echo("")

        # ── Save results ───────────────────────────────────────────────
        click.echo(f"Saving results to {out_dir}/…")
        comparator.save_results(out_dir)

        # Print file list
        saved_files = sorted(out_dir.iterdir())
        for f in saved_files:
            size_kb = f.stat().st_size / 1024
            click.echo(f"  {f.name:<45}  {size_kb:>6.1f} KB")

        # ── Dashboard launch hint ──────────────────────────────────────
        click.echo(
            f"\nTo explore results interactively, launch the dashboard:\n"
            f"  streamlit run src/ml/dashboard.py -- "
            f"--results-dir {out_dir}"
        )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# detect-leakage
# ---------------------------------------------------------------------------


@ml_cli.command("detect-leakage")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model artifact (.pkl), e.g. models/xgboost_v2.pkl",
)
@click.option(
    "--features-dir",
    default="data/processed/features",
    show_default=True,
    type=click.Path(exists=True),
    help="Directory containing *_features.csv files.",
)
@click.option(
    "--test-size",
    default=1_000,
    show_default=True,
    type=int,
    help="Number of random-noise samples for the random-data test.",
)
@click.option(
    "--output",
    default="data/reports/leakage_detection",
    show_default=True,
    type=click.Path(),
    help="Directory where the leakage report JSON is written.",
)
def detect_leakage(model_path, features_dir, test_size, output):
    """Comprehensive data leakage detection for a trained model.

    Runs six tests:

    \b
    1. Random-data test    — model confidence on pure Gaussian noise
    2. Source-code audit   — scan feature engineering for lookahead patterns
    3. Lookahead features  — known future-leaking columns in model feature set
    4. Target-in-features  — target/label columns must not appear as inputs
    5. Temporal ordering   — feature DataFrame must be sorted by timestamp
    6. Train/test contamination — zero date overlap between splits

    Example:

    \b
        python -m src.cli ml detect-leakage \\
            --model-path models/xgboost_v2.pkl \\
            --features-dir data/processed/features/
    """
    import joblib
    from pathlib import Path

    from src.ml.leakage_detector import LeakageDetector
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.ml.data_splitter import DataSplitter

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 72

    click.echo(f"\n{SEP}")
    click.echo("  COMPREHENSIVE DATA LEAKAGE DETECTION")
    click.echo(SEP)
    click.echo(f"  Model   : {model_path}")
    click.echo(f"  Features: {features_dir}")
    click.echo(f"  Output  : {out_dir}")
    click.echo(SEP)

    # ── Load model ────────────────────────────────────────────────────
    click.echo("\nLoading model artifact…")
    artifact     = joblib.load(model_path)
    model        = artifact["model"]
    feature_cols = artifact.get("feature_cols", [])
    click.echo(
        f"  Model type   : {type(model).__name__}\n"
        f"  Feature cols : {len(feature_cols)}"
    )

    detector = LeakageDetector()

    # ── Test 1: Random data ───────────────────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 1/6 — Random-Data Test  (STRONGEST INDICATOR)")
    click.echo(SEP)
    click.echo(f"  Generating {test_size:,} samples of Gaussian noise…")
    rdt = detector.test_on_random_data(model, feature_cols, n_samples=test_size)
    click.echo(f"  Average confidence  : {rdt['avg_confidence']:.4f}")
    click.echo(f"  Max confidence      : {rdt['max_confidence']:.4f}")
    click.echo(f"  95th percentile     : {rdt['percentile_95']:.4f}")
    click.echo(
        f"  High-conf signals   : {rdt['high_confidence_count']} / {test_size}"
        f"  (threshold ≥{rdt['high_confidence_threshold']:.0%})"
    )
    click.echo(
        f"\n  VERDICT: {'🚨 ' if rdt['leakage_detected'] else '✅ '}{rdt['verdict']}"
    )
    click.echo(f"  {rdt['explanation']}")

    # ── Test 2: Feature audit ─────────────────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 2/6 — Feature-Engineering Source Audit")
    click.echo(SEP)
    fa = detector.audit_feature_definitions()
    if fa.get("error"):
        click.echo(f"  WARNING: {fa['error']}")
    else:
        click.echo(f"  Suspicious patterns found: {fa['pattern_count']}")
        for p in fa.get("suspicious_patterns", []):
            click.echo(f"\n  [{p['severity']}] {p['pattern']}")
            click.echo(f"    → {p['description']}")
            click.echo(f"    Fix: {p['recommendation']}")
        verdict_str = "🚨 PATTERNS FOUND" if fa["leakage_likely"] else "✅ PASS"
        click.echo(f"\n  VERDICT: {verdict_str}")

    # ── Test 3: Known lookahead features ─────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 3/6 — Known Lookahead Features in Model")
    click.echo(SEP)
    kl = detector.check_known_lookahead_features(feature_cols)
    click.echo(f"  Features checked: {kl['features_checked']}")
    if kl["lookahead_features"]:
        for f in kl["lookahead_features"]:
            click.echo(f"\n  🚨 '{f['feature']}'")
            click.echo(f"     {f['reason']}")
    verdict_str = "🚨 LOOKAHEAD FEATURES PRESENT" if kl["leakage_detected"] else "✅ PASS"
    click.echo(f"\n  VERDICT: {verdict_str}")

    # ── Test 4: Target not in features ───────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 4/6 — Target Columns Absent from Feature Set")
    click.echo(SEP)
    tf = detector.verify_target_not_in_features(feature_cols)
    click.echo(f"  Features inspected : {tf['features_inspected']}")
    click.echo(f"  Target cols checked: {', '.join(sorted(tf['target_cols_checked']))}")
    if tf["contaminated_cols"]:
        click.echo(f"  🚨 Contaminated    : {tf['contaminated_cols']}")
    verdict_str = "🚨 TARGET COLS IN FEATURES" if tf["leakage_detected"] else "✅ PASS"
    click.echo(f"\n  VERDICT: {verdict_str}")

    # ── Test 5: Temporal ordering ─────────────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 5/6 — Temporal Ordering of Feature Data")
    click.echo(SEP)
    click.echo("  Loading all feature CSVs…")
    try:
        df = load_features(features_dir)
        to = detector.verify_temporal_ordering(df)
        click.echo(f"  Total rows  : {to['total_rows']:,}")
        click.echo(f"  Violations  : {to['violations']}")
        verdict_str = (
            "🚨 ORDERING VIOLATIONS FOUND"
            if not to.get("ordering_valid", True)
            else "✅ PASS"
        )
        click.echo(f"\n  VERDICT: {verdict_str}")
    except Exception as exc:
        click.echo(f"  WARNING: Could not run temporal check — {exc}")

    # ── Test 6: Train/test contamination ─────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  TEST 6/6 — Train / Test Date Contamination")
    click.echo(SEP)
    try:
        feat_files  = sorted(Path(features_dir).glob("*_features.csv"))
        all_dates   = sorted(p.stem.split("_features")[0] for p in feat_files)
        n           = len(all_dates)
        train_end   = int(n * 0.70)
        val_end     = int(n * (0.70 + 0.15))
        train_dates = all_dates[:train_end]
        test_dates  = all_dates[val_end:]

        cc = detector.detect_train_test_contamination(train_dates, test_dates)
        click.echo(f"  Train: {cc['train_start']} → {cc['train_end']}  ({len(train_dates)} dates)")
        click.echo(f"  Val  : {all_dates[train_end]} → {all_dates[val_end - 1]}  ({val_end - train_end} dates)")
        click.echo(f"  Test : {cc['test_start']} → {cc['test_end']}  ({len(test_dates)} dates)")
        click.echo(f"  Gap (train end → test start): {cc['gap_days']} days")
        if cc["overlap_dates"]:
            click.echo(f"  🚨 Overlapping dates: {cc['overlap_dates'][:10]}")
        verdict_str = "🚨 CONTAMINATION" if cc["contamination_detected"] else "✅ PASS"
        click.echo(f"\n  VERDICT: {verdict_str}")
    except Exception as exc:
        click.echo(f"  WARNING: Could not run contamination check — {exc}")

    # ── Generate report ───────────────────────────────────────────────
    report_path = out_dir / "leakage_report.json"
    report      = detector.generate_report(str(report_path))

    # ── Final summary ─────────────────────────────────────────────────
    click.echo(f"\n{SEP}")
    click.echo("  FINAL VERDICT")
    click.echo(SEP)
    safe = report["safe_to_proceed"]
    icon = "✅" if safe else "🚨"
    click.echo(f"\n  {icon}  {report['overall_verdict']}")
    click.echo(f"  Safe to proceed: {'YES' if safe else 'NO'}")

    if report["critical_issues"]:
        click.echo("\n  Critical issues:")
        for issue in report["critical_issues"]:
            click.echo(f"    🚨 {issue}")

    if report["warnings"]:
        click.echo("\n  Warnings (non-blocking):")
        for w in report["warnings"]:
            click.echo(f"    ⚠️  {w}")

    click.echo(f"\n  Report saved → {report_path}")

    if safe:
        click.echo(
            "\n  The 100% win rate at 0.80 threshold appears GENUINE.\n"
            "  You may proceed with:\n"
            "    1. Full-year walk-forward validation\n"
            "    2. Threshold optimisation (0.80–0.85 range)\n"
            "    3. Pattern analysis of winning trades\n"
            "    4. Paper trading preparation"
        )
    else:
        click.echo(
            "\n  DO NOT PROCEED until leakage is resolved.\n"
            "  Fix the critical issues above, then re-run:\n"
            "    1. Feature engineering\n"
            "    2. Model training\n"
            "    3. This leakage detection"
        )

    click.echo(f"\n{SEP}\n")

    if not safe:
        sys.exit(1)


# ---------------------------------------------------------------------------
# sustained-movement-experiment
# ---------------------------------------------------------------------------


@ml_cli.command("sustained-movement-experiment")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--confirmation-minutes",
    default=15,
    type=int,
    show_default=True,
    help="Minutes after entry to check the confirmation bar.",
)
@click.option(
    "--sustain-minutes",
    default=5,
    type=int,
    show_default=True,
    help="Min consecutive bars above entry price at confirmation for a positive label.",
)
@click.option(
    "--n-trials",
    default=30,
    type=int,
    show_default=True,
    help="Optuna trials per model type.",
)
@click.option(
    "--cv-splits",
    default=3,
    type=int,
    show_default=True,
    help="TimeSeriesSplit folds inside each Optuna trial.",
)
@click.option(
    "--thresholds",
    default="0.50,0.60,0.70,0.80",
    show_default=True,
    help="Comma-separated evaluation thresholds.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for models and reports. "
    "Defaults to data/reports/sustained_movement.",
)
def sustained_movement_experiment(
    config_dir,
    features_dir,
    start_date,
    end_date,
    confirmation_minutes,
    sustain_minutes,
    n_trials,
    cv_splits,
    thresholds,
    output,
):
    """Sustained-movement prediction experiment with three model types.

    This command:

    \b
    1. Loads feature CSVs from ``--features-dir`` (or config default).
    2. Applies SustainedMovementLabeler (confirmation_minutes / sustain_minutes).
    3. Shows label distribution and magnitude breakdown.
    4. Chronologically splits data (70/30 train/test).
    5. Trains XGBoost + LightGBM + RandomForest with Optuna optimisation.
    6. Evaluates all models with SustainedMovementEvaluator.
    7. Saves models, comparison report, and precision-by-magnitude breakdown.

    NOTE: Trains from scratch — may take several minutes depending on
    --n-trials.  Use --n-trials 5 for a quick smoke-test run.

    \b
    Example:
        python -m src.cli ml sustained-movement-experiment \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --n-trials 30 --confirmation-minutes 15 --sustain-minutes 5
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.multi_model_trainer import MultiModelTrainer
    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.sustained_movement_labeler import (
        SustainedMovementLabeler,
        MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("data/reports/sustained_movement")
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # ── Parse thresholds ───────────────────────────────────────────
        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats, "
                f"got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        click.echo("\n" + "=" * 70)
        click.echo("  SUSTAINED MOVEMENT PREDICTION EXPERIMENT")
        click.echo("=" * 70)
        click.echo(f"  Features dir:         {feat_dir}")
        click.echo(f"  Date range:           {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"  Confirmation window:  {confirmation_minutes} min")
        click.echo(f"  Sustain requirement:  {sustain_minutes} consecutive min above entry")
        click.echo(f"  Optuna trials/model:  {n_trials}")
        click.echo(f"  CV splits:            {cv_splits}")
        click.echo(f"  Eval thresholds:      {', '.join(f'{t:.0%}' for t in threshold_list)}")
        click.echo(f"  Output dir:           {out_dir}")
        click.echo("=" * 70)

        # ── Step 1: Load feature CSVs ──────────────────────────────────
        click.echo("\n[1/5] Loading feature CSVs…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} → {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # ── Step 2: Apply SustainedMovementLabeler ─────────────────────
        click.echo(
            f"\n[2/5] Applying SustainedMovementLabeler "
            f"(conf={confirmation_minutes}min, sustain={sustain_minutes}min)…"
        )
        labeler_cfg = {
            "sustained_movement": {
                "confirmation_minutes": confirmation_minutes,
                "sustain_minutes": sustain_minutes,
            }
        }
        labeler = SustainedMovementLabeler(labeler_cfg)
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows:      {stats['n_total']:,}")
        click.echo(f"  Positive labels: {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
        click.echo(f"  Coverage:        {stats['coverage_pct']:.1f}% rows have confirmation bar")
        click.echo("\n  Magnitude breakdown (at confirmation bar):")
        for bucket in MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct   = count / max(stats["n_total"], 1) * 100
            bar   = "█" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

        if stats["n_positive"] < 20:
            click.echo(
                "\n  WARNING: Very few positive labels. "
                "Consider lowering --confirmation-minutes or --sustain-minutes.",
                err=True,
            )

        # ── Step 3: Chronological train/test split ─────────────────────
        click.echo("\n[3/5] Splitting data (70% train / 30% test, chronological)…")
        n_total = len(df)
        split_idx = int(n_total * 0.70)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)

        click.echo(
            f"  Train: {len(train_df):,} rows  "
            f"({int(train_df['target_sustained'].sum())} positives)"
        )
        click.echo(
            f"  Test:  {len(test_df):,} rows  "
            f"({int(test_df['target_sustained'].sum())} positives)"
        )

        # Determine feature columns
        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and c not in {
                "target_sustained", "gain_pct_at_confirmation",
                "magnitude_bucket", "sustain_minutes_actual",
            }
        ]
        feature_cols = sorted(feature_cols)
        click.echo(f"  Feature columns: {len(feature_cols)}")

        # ── Feature list verification ───────────────────────────────────
        # Task 1: Print full feature list and check for forbidden columns.
        # Forbidden columns directly encode future outcome info and would
        # constitute data leakage if present in the feature set.
        _FORBIDDEN_FEATURE_COLS = {
            "max_gain_intraday",
            "gain_magnitude_bucket",
            "gain_at_confirmation",
            "max_gain_during_sustain",
            "peak_gain_time",
            "reversion_time",
            "sustain_duration",
        }
        click.echo("\n  --- Feature List Verification ---")
        click.echo(f"  Checking {len(feature_cols)} feature(s) against "
                   f"{len(_FORBIDDEN_FEATURE_COLS)} forbidden names:")
        forbidden_found = [c for c in feature_cols if c in _FORBIDDEN_FEATURE_COLS]
        if forbidden_found:
            click.echo(
                f"\n  🚨 LEAKAGE ALERT: {len(forbidden_found)} forbidden column(s) "
                f"found in feature set:",
                err=True,
            )
            for col in forbidden_found:
                click.echo(f"    ❌ {col}", err=True)
        else:
            click.echo("  ✅ No forbidden columns detected in feature set.")

        click.echo(f"\n  All {len(feature_cols)} features (alphabetical):")
        for i, col in enumerate(feature_cols, start=1):
            marker = "  ❌ FORBIDDEN" if col in _FORBIDDEN_FEATURE_COLS else ""
            click.echo(f"    {i:>3}. {col}{marker}")

        # Task 4: Magnitude bucketing data lineage note.
        # The `magnitude_bucket` column is produced by SustainedMovementLabeler
        # from ACTUAL future bar prices (gain_pct_at_confirmation = the real
        # price % change at the confirmation bar).  It is NOT derived from any
        # model prediction or feature — so precision-by-magnitude measures how
        # well the model identifies REAL large-move events, not a circular stat.
        click.echo(
            "\n  NOTE (magnitude bucketing): 'magnitude_bucket' in test_df comes "
            "from SustainedMovementLabeler — it reflects the ACTUAL price change "
            "at the confirmation bar, computed from raw bar data AFTER training. "
            "It is excluded from the feature set and is used only for outcome "
            "stratification, so precision-by-magnitude is free of leakage."
        )

        # ── Step 4: Train models with Optuna ───────────────────────────
        click.echo(
            f"\n[4/5] Training XGBoost + LightGBM + RandomForest "
            f"({n_trials} Optuna trials each)…"
        )
        click.echo("  (This may take several minutes)")

        trainer = MultiModelTrainer(
            n_trials=n_trials,
            cv_splits=cv_splits,
        )
        artifacts = trainer.train(
            df=train_df,
            target_col="target_sustained",
            feature_cols=feature_cols,
        )

        click.echo("\n  Training complete:")
        for model_name, artifact in artifacts.items():
            opt_score = artifact.get("optimization_score", 0.0)
            val_prec  = artifact.get("val_precision_at_0_70", 0.0)
            click.echo(
                f"    {model_name:<15}: "
                f"Optuna score={opt_score:.4f}  "
                f"val_precision@0.70={val_prec:.4f}"
            )

        # Save model artifacts
        saved_models = trainer.save_artifacts(artifacts, models_dir)
        click.echo(f"\n  Models saved to: {models_dir}/")
        for name, path in saved_models.items():
            size_kb = path.stat().st_size / 1024
            click.echo(f"    {path.name:<40}  {size_kb:>6.1f} KB")

        # ── Step 5: Evaluate ───────────────────────────────────────────
        click.echo(f"\n[5/5] Evaluating models on test set ({len(test_df):,} rows)…")

        evaluator = SustainedMovementEvaluator(
            thresholds=threshold_list,
            target_col="target_sustained",
        )
        eval_results = evaluator.evaluate(artifacts, test_df)

        # Print per-model summary at each threshold
        click.echo(
            f"\n  {'Model':<18}  {'Threshold':>10}  {'Signals':>8}  "
            f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}"
        )
        click.echo("  " + "-" * 68)
        for model_name, mdata in eval_results["models"].items():
            for t, r in sorted(mdata["threshold_results"].items()):
                click.echo(
                    f"  {model_name:<18}  {t:>10.0%}  "
                    f"{r['n_signals']:>8}  "
                    f"{r['precision']:>10.3f}  "
                    f"{r['recall']:>8.3f}  "
                    f"{r['f1']:>6.3f}"
                )

        # Comparison report at primary threshold
        primary_t = threshold_list[min(2, len(threshold_list) - 1)]
        report_df = evaluator.generate_report(eval_results, comparison_threshold=primary_t)
        if not report_df.empty:
            click.echo(f"\n  Side-by-side at {primary_t:.0%} threshold:")
            click.echo("  " + report_df.to_string(index=False).replace("\n", "\n  "))

        # Precision-by-magnitude (at primary threshold)
        click.echo(f"\n  Precision by magnitude bucket (threshold={primary_t:.0%}):")
        for model_name, mdata in eval_results["models"].items():
            pbm = mdata.get("precision_by_magnitude", {})
            by_t = pbm.get("by_threshold", {})
            click.echo(f"\n    {model_name}:")
            click.echo(
                f"      {'Bucket':<12}  {'Signals':>8}  {'TP':>6}  {'Precision':>10}"
            )
            click.echo("      " + "-" * 42)
            bucket_data = by_t.get(primary_t, {})
            for bucket in MAGNITUDE_BUCKETS:
                bd = bucket_data.get(bucket, {})
                n_sig = bd.get("n_signals", 0)
                n_tp  = bd.get("n_tp", 0)
                prec  = bd.get("precision", 0.0)
                click.echo(
                    f"      {bucket:<12}  {n_sig:>8}  {n_tp:>6}  {prec:>10.3f}"
                )

        # Model agreement
        if len(artifacts) >= 2:
            click.echo(f"\n  Model agreement at {primary_t:.0%} threshold:")
            agr_t = eval_results["model_agreement"].get(primary_t, {})
            click.echo(
                f"    Unique signals:       {agr_t.get('total_unique_signals', 0)}"
            )
            click.echo(
                f"    All models agree:     {agr_t.get('all_agree_count', 0)}"
            )
            click.echo(
                f"    Majority agree:       {agr_t.get('majority_agree_count', 0)}"
            )
            for k, v in sorted(agr_t.get("agreement_breakdown", {}).items()):
                click.echo(f"    {k.replace('_', ' ')}: {v}")

        # ── Leakage verification ───────────────────────────────────────
        click.echo("\n" + "=" * 70)
        click.echo("  LEAKAGE VERIFICATION")
        click.echo("=" * 70)

        for model_name, artifact in artifacts.items():
            model_obj   = artifact["model"]
            model_fcols = artifact.get("feature_cols") or feature_cols

            # Task 2: Random data test
            click.echo(f"\n  [{model_name}] Random-noise test ({10_000:,} rows):")
            rand_result = evaluator.test_on_random_data(
                model=model_obj,
                feature_cols=model_fcols,
                n_samples=10_000,
            )
            prefix = "  🚨" if rand_result["leakage_suspected"] else "  ✅"
            click.echo(f"    {prefix} {rand_result['verdict']}")

            # Task 3: Feature importance
            click.echo(f"\n  [{model_name}] Top-20 feature importances:")
            fi_df = evaluator.print_feature_importance(
                model=model_obj,
                feature_cols=model_fcols,
                top_n=20,
            )
            if fi_df.empty:
                click.echo("    (no feature_importances_ available)")
            else:
                click.echo(
                    "    " + fi_df.to_string(index=False).replace("\n", "\n    ")
                )

        click.echo("\n" + "=" * 70)

        # Save all results
        saved_files = evaluator.save_results(eval_results, report_df, out_dir)
        click.echo(f"\n  Results saved to: {out_dir}/")
        for stem, path in saved_files.items():
            size_kb = path.stat().st_size / 1024
            click.echo(f"    {path.name:<45}  {size_kb:>6.1f} KB")

        click.echo("\n" + "=" * 70)
        click.echo("  EXPERIMENT COMPLETE")
        click.echo("=" * 70)
        click.echo(
            f"\n  To explore results with the ML dashboard:\n"
            f"    streamlit run src/ml/dashboard.py -- --results-dir {out_dir}"
        )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@ml_cli.command("analyze-direction-split")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a .pkl model artifact (e.g. reports/sustained_with_directional/models/xgboost_sustained.pkl).",
)
@click.option(
    "--test-data",
    "features_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to features directory or a single features CSV.",
)
@click.option(
    "--test-start-date",
    default=None,
    help="Start date for test window (YYYY-MM-DD). Defaults to last 30% of data.",
)
@click.option(
    "--test-end-date",
    default=None,
    help="End date for test window (YYYY-MM-DD). Defaults to last 30% of data.",
)
@click.option(
    "--threshold",
    default=0.70,
    type=float,
    show_default=True,
    help="Probability threshold for firing a signal.",
)
def analyze_direction_split(model_path, features_dir, test_start_date, test_end_date, threshold):
    """Analyze precision when price is currently rising vs falling.

    Loads a trained model artifact and a features directory, then splits
    all signals at the given threshold into two groups:

    \b
      RISING  — opt_return_1m > 0  AND opt_return_5m > 0
      FALLING — opt_return_1m < 0  OR  opt_return_5m < 0

    Prints precision, TP/FP counts, and magnitude-bucket breakdown for each
    group so you can measure whether filtering by current direction helps.

    \b
    Example:
        python -m src.cli ml analyze-direction-split \\
            --model reports/sustained_with_directional/models/xgboost_sustained.pkl \\
            --test-data data/processed/features/ \\
            --threshold 0.70
    """
    import joblib
    from pathlib import Path

    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features

    try:
        # --- Load artifact -------------------------------------------------------
        click.echo(f"\nLoading model from: {model_path}")
        artifact   = joblib.load(model_path)
        model      = artifact["model"]
        feat_cols  = artifact.get("feature_cols") or []
        model_name = artifact.get("model_name", Path(model_path).stem)
        click.echo(f"  Model    : {model_name}")
        click.echo(f"  Features : {len(feat_cols)}")

        # --- Load features -------------------------------------------------------
        click.echo(f"\nLoading features from: {features_dir}")
        df = load_features(features_dir, test_start_date, test_end_date)
        click.echo(f"  Rows loaded: {len(df):,}")

        if len(df) == 0:
            click.echo("No data found for the specified date range.", err=True)
            sys.exit(1)

        # --- Chronological test split (last 30%) if no explicit dates given ------
        if test_start_date is None and test_end_date is None:
            dates = sorted(df["date"].unique())
            split_idx = int(len(dates) * 0.70)
            test_dates = set(dates[split_idx:])
            test_df = df[df["date"].isin(test_dates)].copy()
            click.echo(
                f"  Test window: {min(test_dates)} → {max(test_dates)}"
                f"  ({len(test_dates)} dates, {len(test_df):,} rows)"
            )
        else:
            test_df = df.copy()
            click.echo(f"  Test rows  : {len(test_df):,}")

        # --- Resolve target column -----------------------------------------------
        target_col = "target_sustained"
        if target_col not in test_df.columns:
            if "target" in test_df.columns:
                click.echo(
                    "\n  Note: 'target_sustained' not found; using 'target' as label proxy."
                )
                test_df[target_col] = test_df["target"]
            else:
                click.echo(
                    "Error: neither 'target_sustained' nor 'target' column found.",
                    err=True,
                )
                sys.exit(1)

        # --- Run direction-split analysis ----------------------------------------
        evaluator = SustainedMovementEvaluator(target_col=target_col)
        evaluator.analyze_precision_by_current_direction(
            model_name=model_name,
            model=model,
            feature_cols=feat_cols,
            test_df=test_df,
            threshold=threshold,
        )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@ml_cli.command("test-consolidation")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a .pkl model artifact.",
)
@click.option(
    "--test-data",
    "features_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to features directory.",
)
@click.option(
    "--test-start-date",
    default=None,
    help="Start date for test window (YYYY-MM-DD). Defaults to last 30% of data.",
)
@click.option(
    "--test-end-date",
    default=None,
    help="End date for test window (YYYY-MM-DD). Defaults to last 30% of data.",
)
@click.option(
    "--threshold",
    default=0.70,
    type=float,
    show_default=True,
    help="Probability threshold for firing a signal.",
)
@click.option(
    "--consol-5m-pct",
    default=1.0,
    type=float,
    show_default=True,
    help="Max |opt_return_5m| to classify as consolidating (%).",
)
@click.option(
    "--consol-15m-pct",
    default=2.0,
    type=float,
    show_default=True,
    help="Max |opt_return_15m| to classify as consolidating (%).",
)
@click.option(
    "--output",
    default="reports/consolidation_hypothesis/",
    show_default=True,
    type=click.Path(),
    help="Directory to save JSON results.",
)
def test_consolidation(
    model_path,
    features_dir,
    test_start_date,
    test_end_date,
    threshold,
    consol_5m_pct,
    consol_15m_pct,
    output,
):
    """Test the consolidation -> breakout hypothesis.

    Splits signals into two groups based on whether the option was in a tight
    trading range (consolidating) or already moving (volatile) at signal time.

    \b
      CONSOLIDATING   : |opt_return_5m| < consol_5m_pct  AND
                        |opt_return_15m| < consol_15m_pct
      NON-CONSOLIDATING: everything else

    If consolidating signals show substantially higher precision (>10 pp),
    the hypothesis is confirmed and consolidation features should be added.

    \b
    Example:
        python -m src.cli ml test-consolidation \\
            --model reports/sustained_with_directional/models/xgboost.pkl \\
            --test-data data/processed/features/ \\
            --threshold 0.70 \\
            --output reports/consolidation_hypothesis/
    """
    import json
    import joblib
    from pathlib import Path

    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features

    try:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Load artifact -------------------------------------------------------
        click.echo(f"\nLoading model from: {model_path}")
        artifact   = joblib.load(model_path)
        model      = artifact["model"]
        feat_cols  = artifact.get("feature_cols") or []
        model_name = artifact.get("model_name", Path(model_path).stem)
        click.echo(f"  Model    : {model_name}")
        click.echo(f"  Features : {len(feat_cols)}")

        # --- Load features -------------------------------------------------------
        click.echo(f"\nLoading features from: {features_dir}")
        df = load_features(features_dir, test_start_date, test_end_date)
        click.echo(f"  Rows loaded: {len(df):,}")

        if len(df) == 0:
            click.echo("No data found for the specified date range.", err=True)
            sys.exit(1)

        # --- Chronological test split (last 30%) if no explicit dates given ------
        if test_start_date is None and test_end_date is None:
            dates = sorted(df["date"].unique())
            split_idx = int(len(dates) * 0.70)
            test_dates = set(dates[split_idx:])
            test_df = df[df["date"].isin(test_dates)].copy()
            click.echo(
                f"  Test window: {min(test_dates)} → {max(test_dates)}"
                f"  ({len(test_dates)} dates, {len(test_df):,} rows)"
            )
        else:
            test_df = df.copy()
            click.echo(f"  Test rows  : {len(test_df):,}")

        # --- Resolve target column -----------------------------------------------
        target_col = "target_sustained"
        if target_col not in test_df.columns:
            if "target" in test_df.columns:
                click.echo(
                    "\n  Note: 'target_sustained' not found; using 'target' as label proxy."
                )
                test_df[target_col] = test_df["target"]
            else:
                click.echo(
                    "Error: neither 'target_sustained' nor 'target' column found.",
                    err=True,
                )
                sys.exit(1)

        # --- Run hypothesis test -------------------------------------------------
        evaluator = SustainedMovementEvaluator(target_col=target_col)
        results = evaluator.analyze_precision_by_consolidation(
            model_name=model_name,
            model=model,
            feature_cols=feat_cols,
            test_df=test_df,
            threshold=threshold,
            consol_5m_pct=consol_5m_pct,
            consol_15m_pct=consol_15m_pct,
        )

        # --- Save results --------------------------------------------------------
        def _serial(obj):
            if isinstance(obj, (bool, int, float, str)):
                return obj
            return str(obj)

        out_path = out_dir / "consolidation_test_results.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_serial)
        size_kb = out_path.stat().st_size / 1024
        click.echo(f"  Results saved: {out_path}  ({size_kb:.1f} KB)")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@ml_cli.command("consolidation-sweep")
@click.option(
    "--model", "model_path", required=True, type=click.Path(exists=True),
    help="Path to a .pkl model artifact.",
)
@click.option(
    "--test-data", "features_dir", required=True, type=click.Path(exists=True),
    help="Path to features directory.",
)
@click.option(
    "--test-start-date", default=None,
    help="Test window start (YYYY-MM-DD). Defaults to last 30% of dates.",
)
@click.option(
    "--test-end-date", default=None,
    help="Test window end (YYYY-MM-DD). Defaults to last 30% of dates.",
)
@click.option(
    "--threshold", default=0.70, type=float, show_default=True,
    help="Signal confidence threshold.",
)
@click.option(
    "--min-signals", default=10, type=int, show_default=True,
    help="Skip combinations with fewer signals than this.",
)
@click.option(
    "--output", default="reports/consolidation_sweep/", show_default=True,
    type=click.Path(), help="Directory to save JSON results.",
)
def consolidation_sweep(
    model_path, features_dir, test_start_date, test_end_date,
    threshold, min_signals, output,
):
    """Sweep consolidation window × tightness-threshold combinations.

    Evaluates all combinations of rolling-range windows (3m–20m) against
    multiple tightness thresholds (0.5%–3.0%) and ranks them by precision.
    Identifies the optimal consolidation definition for use as a signal filter.

    \b
    Example:
        python -m src.cli ml consolidation-sweep \\
            --model reports/sustained_with_consolidation/models/xgboost.pkl \\
            --test-data data/processed/features/ \\
            --threshold 0.70
    """
    import json
    import joblib
    from pathlib import Path

    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features

    try:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nLoading model: {model_path}")
        artifact   = joblib.load(model_path)
        model      = artifact["model"]
        feat_cols  = artifact.get("feature_cols") or []
        model_name = artifact.get("model_name", Path(model_path).stem)
        click.echo(f"  Model    : {model_name}")
        click.echo(f"  Features : {len(feat_cols)}")

        click.echo(f"\nLoading features from: {features_dir}")
        df = load_features(features_dir, test_start_date, test_end_date)
        click.echo(f"  Rows loaded: {len(df):,}")
        if len(df) == 0:
            click.echo("No data found.", err=True); sys.exit(1)

        if test_start_date is None and test_end_date is None:
            dates     = sorted(df["date"].unique())
            split_idx = int(len(dates) * 0.70)
            test_dates = set(dates[split_idx:])
            test_df   = df[df["date"].isin(test_dates)].copy()
            click.echo(
                f"  Test window: {min(test_dates)} → {max(test_dates)}"
                f"  ({len(test_dates)} dates, {len(test_df):,} rows)"
            )
        else:
            test_df = df.copy()

        target_col = "target_sustained"
        if target_col not in test_df.columns:
            if "target" in test_df.columns:
                click.echo("\n  Note: using 'target' as label proxy.")
                test_df[target_col] = test_df["target"]
            else:
                click.echo("Error: no target column found.", err=True); sys.exit(1)

        evaluator = SustainedMovementEvaluator(target_col=target_col)
        results = evaluator.analyze_consolidation_parameter_sweep(
            model_name=model_name,
            model=model,
            feature_cols=feat_cols,
            test_df=test_df,
            threshold=threshold,
            min_signals=min_signals,
        )

        def _ser(o):
            if isinstance(o, (bool, int, float, str)): return o
            return str(o)

        out_path = out_dir / "sweep_results.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_ser)
        click.echo(f"  Results saved: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@ml_cli.command("analyze-call-put-split")
@click.option(
    "--model", "model_path", required=True, type=click.Path(exists=True),
    help="Path to a .pkl model artifact.",
)
@click.option(
    "--test-data", "features_dir", required=True, type=click.Path(exists=True),
    help="Path to features directory.",
)
@click.option(
    "--test-start-date", default=None,
    help="Test window start date (YYYY-MM-DD).",
)
@click.option(
    "--test-end-date", default=None,
    help="Test window end date (YYYY-MM-DD).",
)
@click.option(
    "--threshold", default=0.70, type=float, show_default=True,
    help="Signal confidence threshold.",
)
@click.option(
    "--output", default="reports/call_put_analysis/", show_default=True,
    type=click.Path(), help="Directory to save JSON results.",
)
def analyze_call_put_split(
    model_path, features_dir, test_start_date, test_end_date, threshold, output,
):
    """Analyze whether model signals equally on calls vs puts.

    If the model fires on calls and puts at similar rates (~50/50) while the
    dataset contains both types, then ~50% of signals will fail by construction
    (the wrong option type relative to the underlying's direction).  This command
    tests that hypothesis and, if confirmed, identifies the easy fix: adding an
    is_call feature.

    \b
    Example:
        python -m src.cli ml analyze-call-put-split \\
            --model reports/sustained_with_consolidation_v2/models/xgboost.pkl \\
            --test-data data/processed/features/ \\
            --threshold 0.70
    """
    import json
    import joblib
    from pathlib import Path

    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features

    try:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nLoading model: {model_path}")
        artifact   = joblib.load(model_path)
        model      = artifact["model"]
        feat_cols  = artifact.get("feature_cols") or []
        model_name = artifact.get("model_name", Path(model_path).stem)
        target_col = artifact.get("target_col", "target")
        click.echo(f"  Model    : {model_name}")
        click.echo(f"  Features : {len(feat_cols)}")

        click.echo(f"\nLoading features from: {features_dir}")
        df = load_features(features_dir, test_start_date, test_end_date)
        click.echo(f"  Rows loaded: {len(df):,}")
        if len(df) == 0:
            click.echo("No data found.", err=True); sys.exit(1)

        # Resolve target column
        if target_col not in df.columns:
            for fallback in ("target_sustained", "target"):
                if fallback in df.columns:
                    target_col = fallback
                    break
        click.echo(f"  Target col : {target_col}")

        evaluator = SustainedMovementEvaluator(target_col=target_col)
        results = evaluator.analyze_signals_by_option_type(
            model_name=model_name,
            model=model,
            feature_cols=feat_cols,
            test_df=df,
            threshold=threshold,
        )

        out_path = out_dir / f"{model_name}_call_put_split.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        click.echo(f"  Results saved: {out_path}")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-split-models
# ---------------------------------------------------------------------------


@ml_cli.command("train-split-models")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--confirmation-minutes",
    default=15,
    type=int,
    show_default=True,
    help="Minutes after entry to check the confirmation bar.",
)
@click.option(
    "--sustain-minutes",
    default=5,
    type=int,
    show_default=True,
    help="Min consecutive bars above entry price at confirmation.",
)
@click.option(
    "--n-trials",
    default=30,
    type=int,
    show_default=True,
    help="Optuna trials per model (per option type).",
)
@click.option(
    "--cv-splits",
    default=3,
    type=int,
    show_default=True,
    help="TimeSeriesSplit folds inside each Optuna trial.",
)
@click.option(
    "--thresholds",
    default="0.50,0.60,0.70,0.80",
    show_default=True,
    help="Comma-separated evaluation thresholds.",
)
@click.option(
    "--output",
    default="reports/split_call_put_models/",
    show_default=True,
    type=click.Path(),
    help="Output directory for models and evaluation results.",
)
def train_split_models(
    config_dir,
    features_dir,
    start_date,
    end_date,
    confirmation_minutes,
    sustain_minutes,
    n_trials,
    cv_splits,
    thresholds,
    output,
):
    """Train separate models for CALLs and PUTs to eliminate directional confusion.

    When a single model trains on both call and put options simultaneously,
    directional features (SPY return, option momentum) become noise because the
    same direction helps calls and hurts puts.  Training type-specific models
    allows these features to work as intended.

    Pipeline (identical to sustained-movement-experiment):

    \b
    1. Load feature CSVs.
    2. Apply SustainedMovementLabeler (confirmation / sustain windows).
    3. Chronological 70/30 train/test split.
    4. Train XGBoost + LightGBM separately for CALLs and PUTs (4 models total).
    5. Evaluate each model on its type's test subset; combine results.
    6. Save models and evaluation report.

    \b
    Example:
        python -m src.cli ml train-split-models \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --n-trials 30 --output reports/split_call_put_models/
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.multi_model_trainer import MultiModelTrainer
    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.sustained_movement_labeler import (
        SustainedMovementLabeler,
        MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats, got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        click.echo("\n" + "=" * 70)
        click.echo("  SEPARATE CALL / PUT MODEL TRAINING  (Step 57)")
        click.echo("=" * 70)
        click.echo(f"  Features dir         : {feat_dir}")
        click.echo(f"  Date range           : {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"  Confirmation window  : {confirmation_minutes} min")
        click.echo(f"  Sustain requirement  : {sustain_minutes} consecutive min")
        click.echo(f"  Optuna trials/model  : {n_trials}  (4 models × {n_trials} = "
                   f"{4*n_trials} total trials)")
        click.echo(f"  CV splits            : {cv_splits}")
        click.echo(f"  Eval thresholds      : {', '.join(f'{t:.0%}' for t in threshold_list)}")
        click.echo(f"  Output dir           : {out_dir}")
        click.echo("=" * 70)

        # ── Step 1: Load feature CSVs ──────────────────────────────────
        click.echo("\n[1/5] Loading feature CSVs…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} → {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        if "contract_type" not in df.columns:
            click.echo("Error: 'contract_type' column not found in feature data.", err=True)
            sys.exit(1)

        calls_total = int((df["contract_type"] == 1).sum())
        puts_total  = int((df["contract_type"] == 0).sum())
        click.echo(f"  CALLs: {calls_total:,}  ({calls_total/len(df):.1%})")
        click.echo(f"  PUTs : {puts_total:,}  ({puts_total/len(df):.1%})")

        # ── Step 2: Apply SustainedMovementLabeler ─────────────────────
        click.echo(
            f"\n[2/5] Applying SustainedMovementLabeler "
            f"(conf={confirmation_minutes}min, sustain={sustain_minutes}min)…"
        )
        labeler_cfg = {
            "sustained_movement": {
                "confirmation_minutes": confirmation_minutes,
                "sustain_minutes": sustain_minutes,
            }
        }
        labeler = SustainedMovementLabeler(labeler_cfg)
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows      : {stats['n_total']:,}")
        click.echo(f"  Positive labels : {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
        click.echo(f"  Coverage        : {stats['coverage_pct']:.1f}% rows have confirmation bar")
        click.echo("\n  Magnitude breakdown:")
        for bucket in MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct   = count / max(stats["n_total"], 1) * 100
            bar   = "█" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

        # ── Step 3: Chronological train/test split ─────────────────────
        click.echo("\n[3/5] Splitting data (70% train / 30% test, chronological)…")
        n_total   = len(df)
        split_idx = int(n_total * 0.70)
        train_df  = df.iloc[:split_idx].reset_index(drop=True)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)

        click.echo(
            f"  Train: {len(train_df):,} rows  "
            f"({int(train_df['target_sustained'].sum())} positives)  "
            f"calls={int((train_df['contract_type']==1).sum()):,}  "
            f"puts={int((train_df['contract_type']==0).sum()):,}"
        )
        click.echo(
            f"  Test : {len(test_df):,} rows  "
            f"({int(test_df['target_sustained'].sum())} positives)  "
            f"calls={int((test_df['contract_type']==1).sum()):,}  "
            f"puts={int((test_df['contract_type']==0).sum()):,}"
        )

        # Determine feature columns (same exclusions as sustained-movement-experiment)
        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and c not in {
                "target_sustained", "gain_pct_at_confirmation",
                "magnitude_bucket", "sustain_minutes_actual",
            }
        ]
        feature_cols = sorted(feature_cols)
        click.echo(f"  Feature columns: {len(feature_cols)}")
        click.echo(f"  (contract_type will be dropped within each type's model)")

        # ── Step 4: Train split models ─────────────────────────────────
        click.echo(
            f"\n[4/5] Training XGBoost + LightGBM  ×  CALL / PUT  "
            f"({n_trials} Optuna trials each, 4 total models)…"
        )
        click.echo("  (This will take several minutes per model)")

        trainer = MultiModelTrainer(n_trials=n_trials, cv_splits=cv_splits)
        artifacts = trainer.train_call_put_models_separately(
            df=train_df,
            target_col="target_sustained",
            feature_cols=feature_cols,
        )

        click.echo("\n  Training complete:")
        for name, art in artifacts.items():
            click.echo(
                f"    {name:<22}: "
                f"Optuna={art.get('optimization_score',0):.4f}  "
                f"val_prec@0.7={art.get('val_precision_at_0_70',0):.4f}"
            )

        # Save artifacts
        saved_models = trainer.save_artifacts(
            {k: v for k, v in artifacts.items()}, models_dir
        )
        click.echo(f"\n  Models saved to: {models_dir}/")
        for name, path in saved_models.items():
            click.echo(f"    {path.name:<40}  {path.stat().st_size/1024:>6.1f} KB")

        # ── Step 5: Evaluate split models ──────────────────────────────
        click.echo(f"\n[5/5] Evaluating split models on test set ({len(test_df):,} rows)…")

        evaluator = SustainedMovementEvaluator(
            thresholds=threshold_list,
            target_col="target_sustained",
        )

        all_eval: dict = {}
        for thresh in threshold_list:
            click.echo(f"\n  --- Threshold {thresh:.0%} ---")
            call_art = artifacts["xgboost_call"]
            put_art  = artifacts["xgboost_put"]
            xgb_res  = evaluator.evaluate_split_models(
                call_artifact=call_art,
                put_artifact=put_art,
                test_df=test_df,
                threshold=thresh,
            )
            xgb_res["model"] = "xgboost"

            lgbm_call_art = artifacts["lightgbm_call"]
            lgbm_put_art  = artifacts["lightgbm_put"]
            lgbm_res = evaluator.evaluate_split_models(
                call_artifact=lgbm_call_art,
                put_artifact=lgbm_put_art,
                test_df=test_df,
                threshold=thresh,
            )
            lgbm_res["model"] = "lightgbm"

            all_eval[str(thresh)] = {
                "xgboost":  xgb_res,
                "lightgbm": lgbm_res,
            }

        # Save evaluation results
        eval_path = out_dir / "split_model_eval.json"
        with open(eval_path, "w") as fh:
            _json.dump(all_eval, fh, indent=2)
        click.echo(f"\n  Evaluation saved: {eval_path}")

        # Summary table
        click.echo(f"\n{'='*70}")
        click.echo("  RESULTS SUMMARY")
        click.echo(f"{'='*70}")
        click.echo(f"  {'Model':<12}  {'Threshold':>10}  {'Signals':>8}  "
                   f"{'Precision':>10}  {'vs baseline':>12}")
        click.echo("  " + "-" * 60)
        for t_str, t_res in all_eval.items():
            for mname, mres in t_res.items():
                click.echo(
                    f"  {mname:<12}  {float(t_str):>9.0%}  "
                    f"{mres['combined_signals']:>8,}  "
                    f"{mres['combined_precision']:>9.1%}  "
                    f"{mres['improvement_vs_mixed']:>+11.1%}"
                )
        click.echo(f"{'='*70}")
        click.echo("\n  Baseline (mixed XGBoost @ 70%): 34.3% precision")
        click.echo(f"  Output dir: {out_dir}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-baseline-split  (Step 58)
# ---------------------------------------------------------------------------


@ml_cli.command("train-baseline-split")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files (65-feature baseline).",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--confirmation-minutes",
    default=15,
    type=int,
    show_default=True,
    help="Minutes after entry to check the confirmation bar.",
)
@click.option(
    "--sustain-minutes",
    default=5,
    type=int,
    show_default=True,
    help="Min consecutive bars above entry price at confirmation.",
)
@click.option(
    "--n-trials",
    default=50,
    type=int,
    show_default=True,
    help="Optuna trials per model (per option type).",
)
@click.option(
    "--cv-splits",
    default=3,
    type=int,
    show_default=True,
    help="TimeSeriesSplit folds inside each Optuna trial.",
)
@click.option(
    "--thresholds",
    default="0.50,0.60,0.70,0.80",
    show_default=True,
    help="Comma-separated evaluation thresholds.",
)
@click.option(
    "--output",
    default="reports/baseline_split_models/",
    show_default=True,
    type=click.Path(),
    help="Output directory for models and evaluation results.",
)
def train_baseline_split(
    config_dir,
    features_dir,
    start_date,
    end_date,
    confirmation_minutes,
    sustain_minutes,
    n_trials,
    cv_splits,
    thresholds,
    output,
):
    """Train separate CALL/PUT models on the original 65-feature baseline (Step 58).

    This is the definitive test of whether the call/put split architecture
    improves precision when using the CLEAN 65-feature baseline — before
    directional or consolidation features were added.

    The 65-feature set contains no directional features that become noisy when
    trained on mixed call/put data, so any improvement from splitting is a pure
    architectural benefit, not a feature artefact.

    Pipeline (identical to train-split-models):

    \b
    1. Load 65-feature CSVs.
    2. Apply SustainedMovementLabeler (confirmation / sustain windows).
    3. Chronological 70/30 train/test split.
    4. Train XGBoost + LightGBM for CALLs and PUTs separately (4 models).
    5. Evaluate each model on its type's test subset; combine.
    6. Save models and evaluation report.

    \b
    Baseline reference (mixed models, 65 features):
      XGBoost @ 0.70 → 33.7% precision  (Step 53 first run)

    \b
    Example:
        python -m src.cli ml train-baseline-split \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --n-trials 50 --output reports/baseline_split_models/
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.multi_model_trainer import MultiModelTrainer
    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.sustained_movement_labeler import (
        SustainedMovementLabeler,
        MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats, got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        click.echo("\n" + "=" * 70)
        click.echo("  BASELINE SPLIT MODELS: 65 Features, Separate CALL/PUT  (Step 58)")
        click.echo("=" * 70)
        click.echo(f"  Features dir         : {feat_dir}")
        click.echo(f"  Date range           : {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"  Confirmation window  : {confirmation_minutes} min")
        click.echo(f"  Sustain requirement  : {sustain_minutes} consecutive min")
        click.echo(f"  Optuna trials/model  : {n_trials}  (4 models total)")
        click.echo(f"  CV splits            : {cv_splits}")
        click.echo(f"  Eval thresholds      : {', '.join(f'{t:.0%}' for t in threshold_list)}")
        click.echo(f"  Output dir           : {out_dir}")
        click.echo("=" * 70)

        # ── Step 1: Load feature CSVs ──────────────────────────────────
        click.echo("\n[1/5] Loading feature CSVs…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} → {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        n_cols  = len(df.columns)
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates  ({n_cols} columns)")

        if "contract_type" not in df.columns:
            click.echo("Error: 'contract_type' column not found in feature data.", err=True)
            sys.exit(1)

        calls_total = int((df["contract_type"] == 1).sum())
        puts_total  = int((df["contract_type"] == 0).sum())
        click.echo(f"  CALLs: {calls_total:,}  ({calls_total/len(df):.1%})")
        click.echo(f"  PUTs : {puts_total:,}  ({puts_total/len(df):.1%})")

        # Sanity-check: confirm this is the 65-feature baseline
        expected_max_cols = 90  # 65 features + ~16 metadata/target cols
        if n_cols > expected_max_cols:
            click.echo(
                f"\n  WARNING: {n_cols} columns found — expected ≤{expected_max_cols} "
                f"for the 65-feature baseline.\n"
                f"  Run 'ml generate-features' after removing directional/consolidation "
                f"methods from ml_feature_engineer.py, then retry.",
                err=True,
            )
            sys.exit(1)

        # ── Step 2: Apply SustainedMovementLabeler ─────────────────────
        click.echo(
            f"\n[2/5] Applying SustainedMovementLabeler "
            f"(conf={confirmation_minutes}min, sustain={sustain_minutes}min)…"
        )
        labeler_cfg = {
            "sustained_movement": {
                "confirmation_minutes": confirmation_minutes,
                "sustain_minutes": sustain_minutes,
            }
        }
        labeler = SustainedMovementLabeler(labeler_cfg)
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows      : {stats['n_total']:,}")
        click.echo(f"  Positive labels : {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
        click.echo(f"  Coverage        : {stats['coverage_pct']:.1f}% rows have confirmation bar")
        click.echo("\n  Magnitude breakdown:")
        for bucket in MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct   = count / max(stats["n_total"], 1) * 100
            bar   = "█" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

        # ── Step 3: Chronological train/test split ─────────────────────
        click.echo("\n[3/5] Splitting data (70% train / 30% test, chronological)…")
        n_total   = len(df)
        split_idx = int(n_total * 0.70)
        train_df  = df.iloc[:split_idx].reset_index(drop=True)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)

        click.echo(
            f"  Train: {len(train_df):,} rows  "
            f"({int(train_df['target_sustained'].sum())} positives)  "
            f"calls={int((train_df['contract_type']==1).sum()):,}  "
            f"puts={int((train_df['contract_type']==0).sum()):,}"
        )
        click.echo(
            f"  Test : {len(test_df):,} rows  "
            f"({int(test_df['target_sustained'].sum())} positives)  "
            f"calls={int((test_df['contract_type']==1).sum()):,}  "
            f"puts={int((test_df['contract_type']==0).sum()):,}"
        )

        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and c not in {
                "target_sustained", "gain_pct_at_confirmation",
                "magnitude_bucket", "sustain_minutes_actual",
            }
        ]
        feature_cols = sorted(feature_cols)
        click.echo(f"  Feature columns: {len(feature_cols)}  "
                   f"(contract_type dropped within each type's model)")

        # ── Step 4: Train split models ─────────────────────────────────
        click.echo(
            f"\n[4/5] Training XGBoost + LightGBM  ×  CALL / PUT  "
            f"({n_trials} Optuna trials each, 4 total models)…"
        )
        click.echo("  (This will take several minutes per model)")

        trainer = MultiModelTrainer(n_trials=n_trials, cv_splits=cv_splits)
        artifacts = trainer.train_call_put_models_separately(
            df=train_df,
            target_col="target_sustained",
            feature_cols=feature_cols,
        )

        click.echo("\n  Training complete:")
        for name, art in artifacts.items():
            click.echo(
                f"    {name:<22}: "
                f"Optuna={art.get('optimization_score',0):.4f}  "
                f"val_prec@0.7={art.get('val_precision_at_0_70',0):.4f}"
            )

        saved_models = trainer.save_artifacts(
            {k: v for k, v in artifacts.items()}, models_dir
        )
        click.echo(f"\n  Models saved to: {models_dir}/")
        for name, path in saved_models.items():
            click.echo(f"    {path.name:<40}  {path.stat().st_size/1024:>6.1f} KB")

        # ── Step 5: Evaluate split models ──────────────────────────────
        click.echo(f"\n[5/5] Evaluating split models on test set ({len(test_df):,} rows)…")

        evaluator = SustainedMovementEvaluator(
            thresholds=threshold_list,
            target_col="target_sustained",
        )

        all_eval: dict = {}
        for thresh in threshold_list:
            click.echo(f"\n  --- Threshold {thresh:.0%} ---")
            xgb_res = evaluator.evaluate_split_models(
                call_artifact=artifacts["xgboost_call"],
                put_artifact=artifacts["xgboost_put"],
                test_df=test_df,
                threshold=thresh,
            )
            xgb_res["model"] = "xgboost"

            lgbm_res = evaluator.evaluate_split_models(
                call_artifact=artifacts["lightgbm_call"],
                put_artifact=artifacts["lightgbm_put"],
                test_df=test_df,
                threshold=thresh,
            )
            lgbm_res["model"] = "lightgbm"

            all_eval[str(thresh)] = {
                "xgboost":  xgb_res,
                "lightgbm": lgbm_res,
            }

        # Save evaluation results
        eval_path = out_dir / "baseline_split_eval.json"
        with open(eval_path, "w") as fh:
            _json.dump(all_eval, fh, indent=2)
        click.echo(f"\n  Evaluation saved: {eval_path}")

        # Summary table
        click.echo(f"\n{'='*70}")
        click.echo("  RESULTS SUMMARY  (baseline = 33.7% mixed XGBoost @ 70%)")
        click.echo(f"{'='*70}")
        click.echo(f"  {'Model':<12}  {'Threshold':>10}  {'Signals':>8}  "
                   f"{'Precision':>10}  {'vs baseline':>12}")
        click.echo("  " + "-" * 60)
        for t_str, t_res in all_eval.items():
            for mname, mres in t_res.items():
                click.echo(
                    f"  {mname:<12}  {float(t_str):>9.0%}  "
                    f"{mres['combined_signals']:>8,}  "
                    f"{mres['combined_precision']:>9.1%}  "
                    f"{mres['improvement_vs_mixed']:>+11.1%}"
                )
        click.echo(f"{'='*70}")
        click.echo("\n  Reference: mixed 65-feature XGBoost @ 70% → 33.7% precision (Step 53)")
        click.echo(f"  Output dir: {out_dir}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# generate-spy-labels
# ---------------------------------------------------------------------------


@ml_cli.command("generate-spy-labels")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--lookforward-minutes",
    default=20,
    type=int,
    show_default=True,
    help="Minutes ahead to check for the SPY confirmation bar.",
)
@click.option(
    "--min-move-pct",
    default=0.2,
    type=float,
    show_default=True,
    help="Minimum SPY % gain required at the confirmation bar.",
)
@click.option(
    "--sustain-minutes",
    default=5,
    type=int,
    show_default=True,
    help="Consecutive bars after confirmation that must stay above entry.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory. Defaults to data/processed/spy_labels.",
)
def generate_spy_labels(
    config_dir,
    features_dir,
    start_date,
    end_date,
    lookforward_minutes,
    min_move_pct,
    sustain_minutes,
    output,
):
    """Label each SPY minute bar by whether it makes a sustained upward move.

    Loads feature CSVs, applies SPYMovementLabeler, deduplicates to one row
    per (date, minutes_since_open), and saves the labeled per-minute dataset
    as a single CSV for inspection or downstream use.

    \b
    Adds four columns:
      spy_target           : 1 if SPY moves up >= min-move-pct% and sustains
      spy_gain_at_conf     : actual % gain at the confirmation bar
      spy_max_gain_window  : max % gain in the full forward window
      spy_magnitude_bucket : gain magnitude bucket label

    \b
    Example:
        python -m src.cli ml generate-spy-labels \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --lookforward-minutes 20 --min-move-pct 0.2 --sustain-minutes 5
    """
    import json as _json
    from pathlib import Path as _Path

    import pandas as _pd

    from src.ml.train_xgboost import load_features
    from src.processing.spy_movement_labeler import (
        SPYMovementLabeler,
        SPY_MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("data/processed/spy_labels")
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo("\n" + "=" * 70)
        click.echo("  GENERATE SPY MOVEMENT LABELS")
        click.echo("=" * 70)
        click.echo(f"  Features dir:          {feat_dir}")
        click.echo(f"  Date range:            {start_date or 'all'} -> {end_date or 'all'}")
        click.echo(f"  Lookforward window:    {lookforward_minutes} min")
        click.echo(f"  Min SPY move:          {min_move_pct}%")
        click.echo(f"  Sustain requirement:   {sustain_minutes} consecutive bars")
        click.echo(f"  Output dir:            {out_dir}")
        click.echo("=" * 70)

        # -- Step 1: Load feature CSVs --
        click.echo("\n[1/3] Loading feature CSVs...")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} -> {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # -- Step 2: Apply SPYMovementLabeler --
        click.echo(
            f"\n[2/3] Applying SPYMovementLabeler "
            f"(lookforward={lookforward_minutes}min, "
            f"min_move={min_move_pct}%, sustain={sustain_minutes}min)..."
        )
        labeler = SPYMovementLabeler(
            lookforward_minutes=lookforward_minutes,
            min_move_pct=min_move_pct,
            sustained_minutes=sustain_minutes,
        )
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows:      {stats['n_total']:,}")
        click.echo(f"  Positive labels: {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
        click.echo(f"  Coverage:        {stats['coverage_pct']:.1f}% rows have confirmation bar")
        click.echo(f"  Avg SPY gain:    {stats['avg_spy_gain_pct']:.3f}% (when positive)")
        click.echo("\n  Magnitude breakdown:")
        for bucket in SPY_MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct   = count / max(stats["n_total"], 1) * 100
            bar   = "X" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

        # -- Step 3: Deduplicate to per-minute and save --
        click.echo("\n[3/3] Deduplicating to per-(date, minutes_since_open) and saving...")

        _TIME_COLS_KEEP = {
            "date", "minutes_since_open", "hour_et", "minute_et",
            "minute_of_day", "pct_day_elapsed", "is_morning", "is_last_hour",
            "day_of_week",
        }
        keep_cols = [
            c for c in df.columns
            if c.startswith("spy_") or c in _TIME_COLS_KEEP
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]

        per_min_df = (
            df[keep_cols]
            .drop_duplicates(subset=["date", "minutes_since_open"])
            .sort_values(["date", "minutes_since_open"])
            .reset_index(drop=True)
        )

        out_csv = out_dir / "spy_labeled_per_minute.csv"
        per_min_df.to_csv(out_csv, index=False)

        out_json = out_dir / "spy_label_stats.json"
        with open(out_json, "w") as fh:
            _json.dump(
                {
                    **stats,
                    "lookforward_minutes": lookforward_minutes,
                    "min_move_pct": min_move_pct,
                    "sustained_minutes": sustain_minutes,
                    "start_date": start_date,
                    "end_date": end_date,
                    "n_per_minute_rows": len(per_min_df),
                    "n_dates": int(per_min_df["date"].nunique()),
                    "feature_cols": [
                        c for c in per_min_df.columns
                        if c not in {
                            "date", "minutes_since_open",
                            "spy_target", "spy_gain_at_conf",
                            "spy_max_gain_window", "spy_magnitude_bucket",
                        }
                    ],
                },
                fh,
                indent=2,
                default=str,
            )

        click.echo(f"  Per-minute rows: {len(per_min_df):,}")
        click.echo(f"  Dates:           {per_min_df['date'].nunique()}")
        click.echo(f"  Columns:         {len(per_min_df.columns)}")
        click.echo(f"\n  Saved: {out_csv}")
        click.echo(f"  Saved: {out_json}")
        click.echo("\nDone.")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-spy-model
# ---------------------------------------------------------------------------


@ml_cli.command("train-spy-model")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--lookforward-minutes",
    default=20,
    type=int,
    show_default=True,
    help="Minutes ahead to check for the SPY confirmation bar.",
)
@click.option(
    "--min-move-pct",
    default=0.2,
    type=float,
    show_default=True,
    help="Minimum SPY % gain at the confirmation bar for a positive label.",
)
@click.option(
    "--sustain-minutes",
    default=5,
    type=int,
    show_default=True,
    help="Consecutive bars after confirmation that must stay above entry.",
)
@click.option(
    "--n-trials",
    default=30,
    type=int,
    show_default=True,
    help="Optuna trials per model type.",
)
@click.option(
    "--cv-splits",
    default=3,
    type=int,
    show_default=True,
    help="TimeSeriesSplit folds inside each Optuna trial.",
)
@click.option(
    "--thresholds",
    default="0.50,0.60,0.70,0.80",
    show_default=True,
    help="Comma-separated evaluation thresholds.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory. Defaults to reports/spy_movement_model.",
)
def train_spy_model(
    config_dir,
    features_dir,
    start_date,
    end_date,
    lookforward_minutes,
    min_move_pct,
    sustain_minutes,
    n_trials,
    cv_splits,
    thresholds,
    output,
):
    """Train models to predict SPY directional movement.

    Unlike the sustained-movement experiment (which predicts individual
    OPTION price direction), this command predicts whether SPY itself will
    rise at least min-move-pct% within lookforward-minutes bars and sustain
    that gain for sustain-minutes consecutive bars.

    \b
    Key differences vs sustained-movement-experiment:
      - Labels are shared across contracts (one label per SPY minute)
      - Training rows = unique (date, minutes_since_open) -- no per-contract noise
      - Features = SPY technical indicators + calendar/time features only
      - Target column: spy_target (1 = SPY up-move confirmed + sustained)

    \b
    Pipeline:
    1. Load feature CSVs
    2. Apply SPYMovementLabeler (adds spy_target + metadata)
    3. Deduplicate to one row per (date, minutes_since_open)
    4. Select SPY technical + time/calendar features
    5. Chronological 70/30 train/test split
    6. Train XGBoost + LightGBM + RandomForest with Optuna
    7. Evaluate and report precision-by-magnitude

    \b
    Example:
        python -m src.cli ml train-spy-model \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --lookforward-minutes 20 --min-move-pct 0.2 --sustain-minutes 5 \\
            --n-trials 50 --output reports/spy_movement_model/
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.multi_model_trainer import MultiModelTrainer
    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features
    from src.processing.spy_movement_labeler import (
        SPYMovementLabeler,
        SPY_MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("reports/spy_movement_model")
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats, "
                f"got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        click.echo("\n" + "=" * 70)
        click.echo("  SPY MOVEMENT MODEL -- STEP 59")
        click.echo("=" * 70)
        click.echo(f"  Features dir:          {feat_dir}")
        click.echo(f"  Date range:            {start_date or 'all'} -> {end_date or 'all'}")
        click.echo(f"  Lookforward window:    {lookforward_minutes} min")
        click.echo(f"  Min SPY move:          {min_move_pct}%")
        click.echo(f"  Sustain requirement:   {sustain_minutes} consecutive bars")
        click.echo(f"  Optuna trials/model:   {n_trials}")
        click.echo(f"  CV splits:             {cv_splits}")
        click.echo(f"  Eval thresholds:       {', '.join(f'{t:.0%}' for t in threshold_list)}")
        click.echo(f"  Output dir:            {out_dir}")
        click.echo("=" * 70)

        # -- Step 1: Load feature CSVs --
        click.echo("\n[1/6] Loading feature CSVs...")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} -> {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # -- Step 2: Apply SPYMovementLabeler --
        click.echo(
            f"\n[2/6] Applying SPYMovementLabeler "
            f"(lookforward={lookforward_minutes}min, "
            f"min_move={min_move_pct}%, sustain={sustain_minutes}min)..."
        )
        labeler = SPYMovementLabeler(
            lookforward_minutes=lookforward_minutes,
            min_move_pct=min_move_pct,
            sustained_minutes=sustain_minutes,
        )
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows (all contracts): {stats['n_total']:,}")
        click.echo(f"  Positive labels:            {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
        click.echo(f"  Coverage:                   {stats['coverage_pct']:.1f}% rows have confirmation bar")
        click.echo("\n  SPY magnitude breakdown:")
        for bucket in SPY_MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct   = count / max(stats["n_total"], 1) * 100
            bar   = "X" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

        # -- Step 3: Deduplicate to per-minute rows --
        click.echo("\n[3/6] Deduplicating to one row per (date, minutes_since_open)...")

        _SPY_LABEL_COLS = {
            "spy_target", "spy_gain_at_conf",
            "spy_max_gain_window", "spy_magnitude_bucket",
        }
        _TIME_COLS = {
            "minutes_since_open", "hour_et", "minute_et", "minute_of_day",
            "pct_day_elapsed", "is_morning", "is_last_hour",
            "day_of_week", "is_monday", "is_friday",
        }
        _SPY_RAW_COLS = {"spy_open", "spy_high", "spy_low"}

        keep_cols = [
            c for c in df.columns
            if (
                (c.startswith("spy_") and c not in _SPY_RAW_COLS)
                or c in _TIME_COLS
                or c == "date"
            )
        ]

        spy_df = (
            df[keep_cols]
            .drop_duplicates(subset=["date", "minutes_since_open"])
            .sort_values(["date", "minutes_since_open"])
            .reset_index(drop=True)
        )

        click.echo(f"  Per-minute rows: {len(spy_df):,}")
        click.echo(f"  Dates:           {spy_df['date'].nunique()}")
        click.echo(
            f"  Positive rate after dedup: "
            f"{spy_df['spy_target'].mean():.2%} "
            f"({int(spy_df['spy_target'].sum())} positives)"
        )

        # -- Step 4: Select feature columns --
        click.echo("\n[4/6] Selecting feature columns...")

        feature_cols = sorted([
            c for c in spy_df.columns
            if c not in _SPY_LABEL_COLS
            and c not in {"date", "minutes_since_open"}
            and c not in _SPY_RAW_COLS
        ])
        click.echo(f"  Feature columns: {len(feature_cols)}")
        click.echo("  Features (alphabetical):")
        for i, col in enumerate(feature_cols, start=1):
            click.echo(f"    {i:>3}. {col}")

        if len(feature_cols) < 5:
            click.echo(
                "\n  ERROR: Too few feature columns. "
                "Check that spy_* columns are present in feature CSVs.",
                err=True,
            )
            sys.exit(1)

        # -- Step 5: Chronological train/test split --
        click.echo("\n[5/6] Splitting data (70% train / 30% test, chronological)...")
        n_total   = len(spy_df)
        split_idx = int(n_total * 0.70)
        train_df  = spy_df.iloc[:split_idx].reset_index(drop=True)
        test_df   = spy_df.iloc[split_idx:].reset_index(drop=True)

        click.echo(
            f"  Train: {len(train_df):,} rows  "
            f"({int(train_df['spy_target'].sum())} positives | "
            f"{train_df['spy_target'].mean():.2%} rate)"
        )
        click.echo(
            f"  Test:  {len(test_df):,} rows  "
            f"({int(test_df['spy_target'].sum())} positives | "
            f"{test_df['spy_target'].mean():.2%} rate)"
        )

        if int(train_df["spy_target"].sum()) < 20:
            click.echo(
                "\n  WARNING: Very few positive labels in training set. "
                "Consider adjusting --lookforward-minutes or --min-move-pct.",
                err=True,
            )

        # -- Step 6: Train + Evaluate --
        click.echo(
            f"\n[6/6] Training XGBoost + LightGBM + RandomForest "
            f"({n_trials} Optuna trials each)..."
        )
        click.echo("  (This may take several minutes)")

        trainer = MultiModelTrainer(n_trials=n_trials, cv_splits=cv_splits)
        artifacts = trainer.train(
            df=train_df,
            target_col="spy_target",
            feature_cols=feature_cols,
        )

        click.echo("\n  Training complete:")
        for model_name, artifact in artifacts.items():
            opt_score = artifact.get("optimization_score", 0.0)
            val_prec  = artifact.get("val_precision_at_0_70", 0.0)
            click.echo(
                f"    {model_name:<15}: "
                f"Optuna score={opt_score:.4f}  "
                f"val_precision@0.70={val_prec:.4f}"
            )

        saved_models = trainer.save_artifacts(artifacts, models_dir)
        click.echo(f"\n  Models saved to: {models_dir}/")
        for name, path in saved_models.items():
            click.echo(f"    {path.name:<40}  {path.stat().st_size/1024:>6.1f} KB")

        # Rename spy_magnitude_bucket -> magnitude_bucket for the evaluator
        eval_df = test_df.rename(columns={"spy_magnitude_bucket": "magnitude_bucket"})

        evaluator = SustainedMovementEvaluator(
            thresholds=threshold_list,
            target_col="spy_target",
            magnitude_buckets=list(SPY_MAGNITUDE_BUCKETS),
        )
        eval_results = evaluator.evaluate(artifacts, eval_df)

        click.echo(
            f"\n  {'Model':<18}  {'Threshold':>10}  {'Signals':>8}  "
            f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}"
        )
        click.echo("  " + "-" * 68)
        for model_name, mdata in eval_results["models"].items():
            for t, r in sorted(mdata["threshold_results"].items()):
                click.echo(
                    f"  {model_name:<18}  {t:>10.0%}  "
                    f"{r['n_signals']:>8}  "
                    f"{r['precision']:>10.3f}  "
                    f"{r['recall']:>8.3f}  "
                    f"{r['f1']:>6.3f}"
                )

        click.echo("\n  Precision by SPY magnitude bucket (threshold=70%):")
        click.echo(f"  {'Bucket':<14}  {'Signals':>8}  {'TP':>6}  {'Prec':>8}")
        click.echo("  " + "-" * 42)
        for model_name, mdata in eval_results["models"].items():
            click.echo(f"\n  [{model_name}]")
            # precision_by_magnitude is model-level, structure:
            # {"by_threshold": {t: {bucket: {n_signals, n_tp, precision}}}, ...}
            mag_by_thresh = mdata.get("precision_by_magnitude", {}).get(
                "by_threshold", {}
            )
            # threshold key is float 0.7 in memory
            mag_data = mag_by_thresh.get(0.70, mag_by_thresh.get(0.7, {}))
            if not mag_data:
                click.echo("    (no magnitude data at 70%)")
                continue
            for bucket in SPY_MAGNITUDE_BUCKETS:
                bd    = mag_data.get(bucket, {})
                n_sig = bd.get("n_signals", 0)
                n_tp  = bd.get("n_tp", 0)
                prec  = bd.get("precision", 0.0)
                click.echo(
                    f"    {bucket:<14}: {n_sig:>6}  {n_tp:>5}  {prec:>7.1%}"
                )

        eval_path = out_dir / "spy_model_eval.json"
        with open(eval_path, "w") as fh:
            _json.dump(eval_results, fh, indent=2, default=str)
        click.echo(f"\n  Evaluation saved: {eval_path}")

        stats_path = out_dir / "spy_label_stats.json"
        with open(stats_path, "w") as fh:
            _json.dump(
                {
                    **stats,
                    "n_per_minute_rows": len(spy_df),
                    "n_train_rows": len(train_df),
                    "n_test_rows": len(test_df),
                    "n_feature_cols": len(feature_cols),
                    "feature_cols": feature_cols,
                },
                fh,
                indent=2,
                default=str,
            )

        click.echo(f"\n{'='*70}")
        click.echo("  RESULTS SUMMARY -- SPY Movement Model (Step 59)")
        click.echo(f"{'='*70}")
        click.echo("  Reference (sustained-movement @ 70%): 33.7% precision")
        click.echo(f"  SPY label: +{min_move_pct}% in {lookforward_minutes}m, "
                   f"sustained {sustain_minutes}m")
        click.echo(f"  Training rows:   {len(train_df):,} unique SPY minutes")
        click.echo(f"  Test rows:       {len(test_df):,} unique SPY minutes")
        click.echo(f"  Positive rate:   {spy_df['spy_target'].mean():.2%}")
        click.echo()
        click.echo(f"  {'Model':<18}  {'70% Thresh':>12}  {'Signals':>8}  {'Precision':>10}")
        click.echo("  " + "-" * 55)
        for model_name, mdata in eval_results["models"].items():
            r70 = mdata["threshold_results"].get(0.70, {})
            click.echo(
                f"  {model_name:<18}  {'70%':>12}  "
                f"{r70.get('n_signals', 0):>8}  "
                f"{r70.get('precision', 0.0):>9.1%}"
            )
        click.echo(f"{'='*70}")
        click.echo(f"\n  Output dir: {out_dir}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# test-consolidation-filter
# ---------------------------------------------------------------------------


@ml_cli.command("test-consolidation-filter")
@click.option(
    "--model",
    required=True,
    help="Path to trained model pickle (e.g. reports/spy_movement_model/models/xgboost.pkl).",
)
@click.option(
    "--model-type",
    type=click.Choice(["spy", "option"]),
    required=True,
    help="'spy' uses SPYMovementLabeler + per-minute dedup; 'option' uses SustainedMovementLabeler.",
)
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--threshold",
    default=0.70,
    type=float,
    show_default=True,
    help="Probability threshold for model predictions.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for results JSON. Defaults to reports/consolidation_filter_test.",
)
def test_consolidation_filter(
    model,
    model_type,
    config_dir,
    features_dir,
    start_date,
    end_date,
    threshold,
    output,
):
    """Discover optimal consolidation parameters and test as post-filter.

    Workflow:

    \b
    1. Load feature CSVs and apply appropriate labeler.
    2. Chronological 70/30 train/test split.
    3. Run ConsolidationAnalyzer on TRAINING data (no test leakage).
    4. Load model artifact; predict on test data.
    5. Apply ConsolidationFilter with optimal parameters.
    6. Compare precision / signal count before vs after filter.

    \b
    Model types:
      spy    -- SPYMovementLabeler labels, per-minute dedup, spy_high/low range
      option -- SustainedMovementLabeler labels, per-contract range (high/low)

    \b
    Examples:
        python -m src.cli ml test-consolidation-filter \\
            --model reports/spy_movement_model/models/xgboost.pkl \\
            --model-type spy \\
            --start-date 2025-03-03 --end-date 2026-02-19

        python -m src.cli ml test-consolidation-filter \\
            --model reports/sustained_with_directional/models/xgboost.pkl \\
            --model-type option \\
            --start-date 2025-03-03 --end-date 2026-02-19
    """
    import json as _json
    import pickle as _pickle
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.analysis.consolidation_analyzer import ConsolidationAnalyzer
    from src.ml.consolidation_filter import ConsolidationFilter
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.spy_movement_labeler import SPYMovementLabeler
    from src.processing.sustained_movement_labeler import SustainedMovementLabeler

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("reports/consolidation_filter_test")
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo("\n" + "=" * 72)
        click.echo("  CONSOLIDATION FILTER TEST -- STEP 60")
        click.echo("=" * 72)
        click.echo(f"  Model:           {model}")
        click.echo(f"  Model type:      {model_type}")
        click.echo(f"  Features dir:    {feat_dir}")
        click.echo(f"  Date range:      {start_date or 'all'} -> {end_date or 'all'}")
        click.echo(f"  Pred threshold:  {threshold:.0%}")
        click.echo(f"  Output dir:      {out_dir}")
        click.echo("=" * 72)

        # ── Step 1: Load feature CSVs ──────────────────────────────────
        click.echo("\n[1/5] Loading feature CSVs...")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo("Error: no feature data found.", err=True)
            sys.exit(1)
        click.echo(f"  Loaded {len(df):,} rows across {df['date'].nunique()} dates")

        # ── Step 2: Apply labeler ──────────────────────────────────────
        click.echo(f"\n[2/5] Applying {'SPY' if model_type == 'spy' else 'Sustained'} labeler...")

        if model_type == "spy":
            spy_labeler = SPYMovementLabeler(
                lookforward_minutes=20, min_move_pct=0.2, sustained_minutes=5
            )
            df = spy_labeler.label(df)
            target_col = "spy_target"

            # Deduplicate to per-minute (keep OHLC for consolidation analysis)
            _SPY_LABEL_COLS = {
                "spy_target", "spy_gain_at_conf",
                "spy_max_gain_window", "spy_magnitude_bucket",
            }
            keep_cols = [
                c for c in df.columns
                if c.startswith("spy_") or c in {
                    "date", "minutes_since_open", "hour_et", "minute_et",
                    "minute_of_day", "pct_day_elapsed", "is_morning",
                    "is_last_hour", "day_of_week",
                }
            ]
            df = (
                df[keep_cols]
                .drop_duplicates(subset=["date", "minutes_since_open"])
                .sort_values(["date", "minutes_since_open"])
                .reset_index(drop=True)
            )
            click.echo(f"  After dedup: {len(df):,} per-minute rows")
            click.echo(
                f"  Positive rate: {df[target_col].mean():.2%} "
                f"({int(df[target_col].sum())} positives)"
            )

        else:  # option
            opt_labeler_cfg = {
                "sustained_movement": {
                    "confirmation_minutes": 15,
                    "sustain_minutes": 5,
                }
            }
            opt_labeler = SustainedMovementLabeler(opt_labeler_cfg)
            df = opt_labeler.label(df)
            target_col = "target_sustained"
            click.echo(
                f"  Positive rate: {df[target_col].mean():.2%} "
                f"({int(df[target_col].sum())} positives)"
            )

        # ── Step 3: Chronological 70/30 split ─────────────────────────
        click.echo("\n[3/5] Splitting data (70% train / 30% test, chronological)...")
        n_total   = len(df)
        split_idx = int(n_total * 0.70)
        train_df  = df.iloc[:split_idx].reset_index(drop=True)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)
        click.echo(
            f"  Train: {len(train_df):,} rows | "
            f"{int(train_df[target_col].sum())} positives"
        )
        click.echo(
            f"  Test:  {len(test_df):,} rows | "
            f"{int(test_df[target_col].sum())} positives"
        )

        # ── Step 4: Discover optimal parameters (training set only) ───
        click.echo("\n[4/5] Running ConsolidationAnalyzer on training data...")
        analyzer = ConsolidationAnalyzer(min_samples=50)

        if model_type == "spy":
            analysis = analyzer.analyze_spy_consolidation(train_df)
        else:
            analysis = analyzer.analyze_option_consolidation(train_df)

        optimal = analysis["optimal"]
        if not optimal:
            click.echo("  ERROR: no valid parameter combinations found.", err=True)
            sys.exit(1)

        opt_window    = int(optimal["window"])
        opt_threshold = float(optimal["threshold"])
        click.echo(
            f"\n  Optimal parameters: window={opt_window}m, "
            f"threshold={opt_threshold:.2f}%"
        )
        click.echo(
            f"  Training precision at optimal: {optimal['precision']:.1%} "
            f"(baseline {analysis['baseline_precision']:.1%}, "
            f"lift {optimal['lift']:.2f}x)"
        )

        # ── Step 5: Load model, predict, filter, compare ──────────────
        click.echo("\n[5/5] Loading model, predicting, and applying filter...")

        model_path = _Path(model)
        if not model_path.exists():
            click.echo(f"  ERROR: model file not found: {model_path}", err=True)
            sys.exit(1)

        with open(model_path, "rb") as fh:
            artifact = _pickle.load(fh)

        clf         = artifact["model"]
        feature_cols = artifact.get("feature_cols") or []
        click.echo(
            f"  Model: {artifact.get('model_type', 'unknown')}  "
            f"|  {len(feature_cols)} features"
        )

        # Build X_test — only use feature columns that exist in test_df
        available = [c for c in feature_cols if c in test_df.columns]
        if len(available) < len(feature_cols):
            missing_fc = set(feature_cols) - set(available)
            click.echo(
                f"  WARNING: {len(missing_fc)} model feature(s) missing from test_df "
                f"— filling with 0.0",
                err=True,
            )
        X_test = (
            test_df[available]
            .reindex(columns=feature_cols, fill_value=0.0)
            .fillna(0.0)
            .values.astype(_np.float32)
        )

        probas = clf.predict_proba(X_test)[:, 1]
        preds  = (probas >= threshold).astype(bool)
        y_true = test_df[target_col].values.astype(_np.int8)

        # Before-filter precision
        n_sig_before = int(preds.sum())
        tp_before     = int(((preds == 1) & (y_true == 1)).sum())
        prec_before   = tp_before / max(n_sig_before, 1)

        # Apply consolidation filter
        cf = ConsolidationFilter()
        if model_type == "spy":
            filtered_preds = cf.apply_spy_filter(
                preds, test_df, window=opt_window, threshold=opt_threshold
            )
        else:
            filtered_preds = cf.apply_option_filter(
                preds, test_df, window=opt_window, threshold=opt_threshold
            )

        # After-filter precision
        n_sig_after = int(filtered_preds.sum())
        tp_after     = int(((filtered_preds == 1) & (y_true == 1)).sum())
        prec_after   = tp_after / max(n_sig_after, 1)

        delta_prec    = prec_after - prec_before
        pct_signals   = n_sig_after / max(n_sig_before, 1) * 100

        # ── Save and print results ─────────────────────────────────────
        results = {
            "model":            str(model_path),
            "model_type":       model_type,
            "threshold":        threshold,
            "optimal_window":   opt_window,
            "optimal_threshold": opt_threshold,
            "training_precision_at_optimal": float(optimal["precision"]),
            "baseline_precision": float(analysis["baseline_precision"]),
            "before_filter": {
                "n_signals":  n_sig_before,
                "true_positives": tp_before,
                "precision":  prec_before,
            },
            "after_filter": {
                "n_signals":  n_sig_after,
                "true_positives": tp_after,
                "precision":  prec_after,
            },
            "delta_precision": delta_prec,
            "signal_retention_pct": pct_signals,
            "consolidation_analysis": analysis["results"],
        }

        out_json = out_dir / "consolidation_filter_results.json"
        with open(out_json, "w") as fh:
            _json.dump(results, fh, indent=2, default=str)

        click.echo(f"\n{'='*72}")
        click.echo("  RESULTS SUMMARY")
        click.echo(f"{'='*72}")
        click.echo(f"  Model type:            {model_type}")
        click.echo(f"  Optimal parameters:    window={opt_window}m, threshold={opt_threshold:.2f}%")
        click.echo(f"  Prediction threshold:  {threshold:.0%}")
        click.echo()
        click.echo(f"  {'':30}  {'Signals':>8}  {'TPs':>6}  {'Precision':>10}")
        click.echo("  " + "-" * 58)
        click.echo(
            f"  {'BEFORE filter':<30}  {n_sig_before:>8,}  {tp_before:>6}  {prec_before:>9.1%}"
        )
        click.echo(
            f"  {'AFTER filter':<30}  {n_sig_after:>8,}  {tp_after:>6}  {prec_after:>9.1%}"
        )
        click.echo()
        click.echo(f"  Precision change:      {delta_prec:>+.1%}")
        click.echo(f"  Signal retention:      {pct_signals:.1f}%")
        click.echo(f"  Training lift:         {optimal['lift']:.2f}x")
        click.echo(f"{'='*72}")
        click.echo(f"\n  Results saved: {out_json}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# threshold-sweep
# ---------------------------------------------------------------------------


@ml_cli.command("threshold-sweep")
@click.option(
    "--model",
    required=True,
    help="Path to trained model pickle (artifact with model, feature_cols, target_col).",
)
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--min-threshold",
    default=0.70,
    type=float,
    show_default=True,
    help="Lowest threshold to test.",
)
@click.option(
    "--max-threshold",
    default=0.95,
    type=float,
    show_default=True,
    help="Highest threshold to test.",
)
@click.option(
    "--step",
    default=0.05,
    type=float,
    show_default=True,
    help="Step size between thresholds.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory. Defaults to reports/threshold_sweep.",
)
def threshold_sweep(
    model,
    config_dir,
    features_dir,
    start_date,
    end_date,
    min_threshold,
    max_threshold,
    step,
    output,
):
    """Test a model at multiple confidence thresholds to find the precision sweet spot.

    Applies SustainedMovementLabeler, does 70/30 chronological split, then
    sweeps thresholds from min to max looking for thresholds that achieve
    >= 50% precision with 10-50 signals per day.

    \b
    For split CALL/PUT models, automatically filters the test set to the
    appropriate contract type (option_type stored in the artifact).

    \b
    Examples:
        python -m src.cli ml threshold-sweep \\
            --model reports/baseline_split_models/models/xgboost_call.pkl \\
            --start-date 2025-03-03 --end-date 2026-02-19

        python -m src.cli ml threshold-sweep \\
            --model reports/baseline_split_models/models/xgboost_put.pkl \\
            --start-date 2025-03-03 --end-date 2026-02-19
    """
    import json as _json
    import pickle as _pickle
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.sustained_movement_evaluator import SustainedMovementEvaluator
    from src.ml.train_xgboost import load_features
    from src.processing.sustained_movement_labeler import (
        SustainedMovementLabeler,
        MAGNITUDE_BUCKETS,
    )

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("reports/threshold_sweep")
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = _Path(model)
        if not model_path.exists():
            click.echo(f"Error: model not found: {model_path}", err=True)
            sys.exit(1)

        # Build threshold list
        thresholds = []
        t = min_threshold
        while t <= max_threshold + 1e-9:
            thresholds.append(round(t, 4))
            t += step

        click.echo("\n" + "=" * 72)
        click.echo("  THRESHOLD SWEEP -- STEP 61")
        click.echo("=" * 72)
        click.echo(f"  Model:           {model_path}")
        click.echo(f"  Features dir:    {feat_dir}")
        click.echo(f"  Date range:      {start_date or 'all'} -> {end_date or 'all'}")
        click.echo(
            f"  Thresholds:      {', '.join(f'{t:.2f}' for t in thresholds)}"
        )
        click.echo(f"  Output dir:      {out_dir}")
        click.echo("=" * 72)

        # ── Step 1: Load artifact (pickle or joblib) ───────────────────
        click.echo("\n[1/4] Loading model artifact...")
        try:
            with open(model_path, "rb") as fh:
                artifact = _pickle.load(fh)
        except Exception:
            import joblib as _joblib
            artifact = _joblib.load(model_path)

        model_name  = artifact.get("model_name") or artifact.get("model_type", "model")
        option_type = artifact.get("option_type", "mixed")
        target_col  = artifact.get("target_col", "target_sustained")
        click.echo(
            f"  Model: {model_name}  |  option_type={option_type}  "
            f"|  target={target_col}  |  {len(artifact.get('feature_cols', []))} features"
        )

        # ── Step 2: Load features + label ──────────────────────────────
        click.echo("\n[2/4] Loading features and applying SustainedMovementLabeler...")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo("Error: no feature data found.", err=True)
            sys.exit(1)
        click.echo(f"  Loaded {len(df):,} rows across {df['date'].nunique()} dates")

        labeler_cfg = {
            "sustained_movement": {
                "confirmation_minutes": 15,
                "sustain_minutes":      5,
            }
        }
        labeler = SustainedMovementLabeler(labeler_cfg)
        df = labeler.label(df)
        click.echo(
            f"  Positive rate: {df[target_col].mean():.2%} "
            f"({int(df[target_col].sum())} positives)"
        )

        # ── Step 3: Chronological 70/30 split + optional type filter ───
        click.echo("\n[3/4] Splitting data...")
        n_total   = len(df)
        split_idx = int(n_total * 0.70)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)

        # For split CALL/PUT models, filter test set to matching type
        if option_type == "call":
            test_df = test_df[test_df["contract_type"] == 1].reset_index(drop=True)
            click.echo(f"  Filtered to CALL contracts: {len(test_df):,} rows")
        elif option_type == "put":
            test_df = test_df[test_df["contract_type"] == 0].reset_index(drop=True)
            click.echo(f"  Filtered to PUT contracts: {len(test_df):,} rows")
        else:
            click.echo(f"  Test set (all contracts): {len(test_df):,} rows")

        click.echo(
            f"  Test positives: {int(test_df[target_col].sum())} "
            f"({test_df[target_col].mean():.2%})"
        )

        if int(test_df[target_col].sum()) < 10:
            click.echo(
                "\n  WARNING: very few positive labels in test set. "
                "Results may be unreliable.",
                err=True,
            )

        # ── Step 4: Threshold sweep ────────────────────────────────────
        click.echo("\n[4/4] Running threshold sweep...")

        evaluator = SustainedMovementEvaluator(
            thresholds=thresholds,
            target_col=target_col,
            magnitude_buckets=list(MAGNITUDE_BUCKETS),
        )
        sweep_results = evaluator.comprehensive_threshold_sweep(
            artifact=artifact,
            test_df=test_df,
            thresholds=thresholds,
        )

        # Save results
        stem = model_path.stem  # e.g. "xgboost_call"
        out_json = out_dir / f"{stem}_threshold_sweep.json"
        with open(out_json, "w") as fh:
            _json.dump(sweep_results, fh, indent=2, default=str)
        click.echo(f"\n  Results saved: {out_json}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# leakage-audit
# ---------------------------------------------------------------------------


@ml_cli.command("leakage-audit")
@click.option(
    "--model-path", "model_path_str",
    default="models/xgboost_v2.pkl",
    show_default=True,
    help="Model artifact to audit (dict with 'model' and 'feature_cols' keys).",
)
@click.option(
    "--compare-path", "compare_path_str",
    default=None,
    help="Optional clean model for comparison (e.g. models/xgboost_v3_clean.pkl).",
)
@click.option(
    "--features-dir", default=None,
    help="Feature CSV directory (overrides config).",
)
@click.option(
    "--start-date", default="2025-03-03", show_default=True,
    help="Feature date range start.",
)
@click.option(
    "--end-date", default="2026-02-19", show_default=True,
    help="Feature date range end.",
)
@click.option(
    "--output", default="reports/leakage_audit", show_default=True,
    help="Output directory for the audit JSON report.",
)
@click.option(
    "--config-dir", default="config", show_default=True,
    help="Directory containing YAML config files.",
)
def leakage_audit(
    model_path_str, compare_path_str, features_dir,
    start_date, end_date, output, config_dir,
):
    """Comprehensive data leakage audit for a model artifact -- Step 62.

    \b
    Runs 8 leakage detection tests:
      1. Random data test (strongest indicator)
      2. Source code audit (ml_feature_engineer.py)
      3. Known lookahead features check
      4. Target columns not in feature set
      5. Temporal ordering validation
      6. Train / test contamination detection
      7. 120-minute correlation analysis (new)
      8. Feature importance red-flag analysis (new)

    \b
    Also reports:
      - Precision at multiple thresholds on 30% holdout
      - Comparison model precision (if --compare-path provided)
      - Fresh 2026 performance (dates >= 2026-01-01)
      - Overall verdict: CLEAN or LEAKED

    \b
    Examples:
        python -m src.cli ml leakage-audit \\
            --model-path models/xgboost_v2.pkl \\
            --compare-path models/xgboost_v3_clean.pkl

        python -m src.cli ml leakage-audit \\
            --model-path models/xgboost_v2.pkl \\
            --output reports/leakage_audit/v2_audit
    """
    import json as _json
    import pickle as _pickle
    from pathlib import Path as _Path

    import numpy as _np

    from src.ml.leakage_detector import LeakageDetector
    from src.ml.train_xgboost import load_features
    from src.processing.sustained_movement_labeler import SustainedMovementLabeler

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Banner ────────────────────────────────────────────────────────
        click.echo("\n" + "=" * 72)
        click.echo("  LEAKAGE AUDIT -- STEP 62")
        click.echo("=" * 72)
        click.echo(f"  Audit model:     {model_path_str}")
        if compare_path_str:
            click.echo(f"  Compare model:   {compare_path_str}")
        click.echo(f"  Features dir:    {feat_dir}")
        click.echo(f"  Date range:      {start_date} -> {end_date}")
        click.echo(f"  Output dir:      {out_dir}")
        click.echo("=" * 72)

        # ── Step 1: Load primary artifact ─────────────────────────────────
        click.echo("\n[1/7] Loading model artifact...")
        model_path = _Path(model_path_str)
        if not model_path.exists():
            click.echo(f"Error: model not found: {model_path}", err=True)
            sys.exit(1)

        try:
            with open(model_path, "rb") as fh:
                artifact = _pickle.load(fh)
        except Exception:
            import joblib as _joblib
            artifact = _joblib.load(model_path)

        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        model_name = artifact.get("model_name") or model_path.stem
        click.echo(
            f"  Loaded: {model_name}  |  {len(feature_cols)} features  "
            f"|  saved: {artifact.get('saved_at', 'unknown')}"
        )

        # ── Step 2: Load features ─────────────────────────────────────────
        click.echo("\n[2/7] Loading feature data...")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo("Error: no feature data found.", err=True)
            sys.exit(1)
        click.echo(f"  Loaded {len(df):,} rows across {df['date'].nunique()} dates")

        labeler_cfg = {
            "sustained_movement": {
                "confirmation_minutes": 15,
                "sustain_minutes": 5,
            }
        }
        labeler = SustainedMovementLabeler(labeler_cfg)
        df = labeler.label(df)
        click.echo(
            f"  Positive rate: {df['target_sustained'].mean():.2%} "
            f"({int(df['target_sustained'].sum())} positives)"
        )

        # ── Step 3: Train / test split ────────────────────────────────────
        click.echo("\n[3/7] Splitting data (70% train / 30% test)...")
        n_total = len(df)
        split_idx = int(n_total * 0.70)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        train_dates = sorted(train_df["date"].astype(str).unique().tolist())
        test_dates = sorted(test_df["date"].astype(str).unique().tolist())
        click.echo(
            f"  Train: {len(train_df):,} rows  ({train_dates[0]} -> {train_dates[-1]})"
        )
        click.echo(
            f"  Test : {len(test_df):,} rows  ({test_dates[0]} -> {test_dates[-1]})"
        )

        fresh_df = test_df[
            test_df["date"].astype(str) >= "2026-01-01"
        ].reset_index(drop=True)
        click.echo(f"  Fresh 2026 subset: {len(fresh_df):,} rows")

        # ── Step 4: Run all 8 leakage tests ──────────────────────────────
        click.echo("\n[4/7] Running leakage detection tests...")
        detector = LeakageDetector()

        # Test 1: Random data
        click.echo("  Test 1/8: Random data test...")
        t1 = detector.test_on_random_data(model, feature_cols)
        v1 = "PASS" if not t1.get("leakage_detected") else "FAIL"
        click.echo(
            f"    -> {v1}  |  {t1.get('high_confidence_count', 0)} high-conf signals "
            f"on {t1.get('n_samples', 1000)} random rows  "
            f"|  avg_p={t1.get('avg_confidence', 0):.3f}  "
            f"max_p={t1.get('max_confidence', 0):.3f}"
        )

        # Test 2: Source code audit
        click.echo("  Test 2/8: Source code audit...")
        t2 = detector.audit_feature_definitions()
        v2_verdict = "PASS" if not t2.get("leakage_likely") else "FAIL"
        click.echo(
            f"    -> {v2_verdict}  |  {t2.get('pattern_count', 0)} suspicious pattern(s)"
        )
        for p in t2.get("suspicious_patterns", []):
            click.echo(f"       [{p.get('severity', '?')}] {p.get('pattern')}")

        # Test 3: Known lookahead features
        click.echo("  Test 3/8: Known lookahead features check...")
        t3 = detector.check_known_lookahead_features(feature_cols)
        v3 = "PASS" if not t3.get("leakage_detected") else "FAIL"
        click.echo(
            f"    -> {v3}  |  checked {t3.get('features_checked', 0)} features"
        )
        for item in t3.get("lookahead_features", []):
            click.echo(f"       DETECTED: '{item['feature']}'")

        # Test 4: Target in features
        click.echo("  Test 4/8: Target columns not in feature set...")
        t4 = detector.verify_target_not_in_features(feature_cols)
        v4 = "PASS" if not t4.get("leakage_detected") else "FAIL"
        click.echo(
            f"    -> {v4}  |  contaminated: {t4.get('contaminated_cols', [])}"
        )

        # Test 5: Temporal ordering
        click.echo("  Test 5/8: Temporal ordering...")
        t5 = detector.verify_temporal_ordering(df)
        v5 = "PASS" if t5.get("ordering_valid") else "WARN"
        click.echo(
            f"    -> {v5}  |  {t5.get('violations', 0)} violation(s) "
            f"in {t5.get('total_rows', 0):,} rows"
        )

        # Test 6: Train/test contamination
        click.echo("  Test 6/8: Train/test contamination...")
        t6 = detector.detect_train_test_contamination(train_dates, test_dates)
        v6 = "PASS" if not t6.get("contamination_detected") else "FAIL"
        click.echo(
            f"    -> {v6}  |  {t6.get('overlap_count', 0)} overlapping dates  "
            f"|  gap={t6.get('gap_days', '?')} days"
        )

        # Test 7: Correlation analysis (new)
        click.echo("  Test 7/8: 120-minute correlation analysis...")
        t7 = detector.check_120min_specific_leaks(df, feature_cols)
        v7 = "PASS" if not t7.get("leakage_suspected") else "WARN"
        click.echo(
            f"    -> {v7}  |  outcome col: {t7.get('outcome_column_used', 'none')}  "
            f"|  {len(t7.get('suspicious_by_name', []))} name pattern(s)  "
            f"|  {len(t7.get('suspicious_by_correlation', []))} suspicious corr(s)"
        )
        for item in t7.get("suspicious_by_name", []):
            click.echo(f"       NAME FLAG: '{item['feature']}'")
        top_corr = t7.get("correlation_table_top20", [])[:5]
        if top_corr:
            click.echo("       Top 5 feature correlations with outcome:")
            sev_map = {
                r["feature"]: r.get("severity", "OK")
                for r in t7.get("suspicious_by_correlation", [])
            }
            for item in top_corr:
                sev = sev_map.get(item["feature"], "OK")
                marker = f" [{sev}]" if sev != "OK" else ""
                click.echo(
                    f"         {item['feature']:<35} {item['correlation']:+.4f}{marker}"
                )

        # Test 8: Feature importance (new)
        click.echo("  Test 8/8: Feature importance red-flag analysis...")
        t8 = detector.analyze_feature_importance(model, feature_cols, top_n=20)
        v8 = "PASS" if not t8.get("leakage_suspected") else "FAIL"
        click.echo(
            f"    -> {v8}  |  top feature: {t8.get('top_feature', 'none')}"
        )
        for flag in t8.get("red_flags", []):
            click.echo(f"       {flag}")

        click.echo("\n  Top 10 features by importance:")
        for entry in t8.get("ranked_features", [])[:10]:
            marker = (
                " [CRITICAL]" if "CRITICAL" in entry.get("flag", "")
                else " [WARN]" if "WARNING" in entry.get("flag", "")
                else ""
            )
            click.echo(
                f"    {entry['rank']:2d}. {entry['feature']:<35} "
                f"{entry['importance_pct']:>8}{marker}"
            )

        # ── Step 5: Precision at thresholds on test holdout ───────────────
        click.echo("\n[5/7] Precision at thresholds on test holdout...")
        audit_thresholds = [0.50, 0.60, 0.67, 0.70, 0.80, 0.85, 0.90]

        def _eval_precision(eval_df, eval_model, eval_features, thresh_list):
            missing = [f for f in eval_features if f not in eval_df.columns]
            # Build feature matrix; fill missing columns with 0 (NaN substitute)
            # so models with removed/leaked features can still be evaluated
            cols = []
            for f in eval_features:
                if f in eval_df.columns:
                    cols.append(eval_df[f].values)
                else:
                    cols.append(_np.zeros(len(eval_df), dtype=_np.float32))
            X = _np.column_stack(cols).astype(_np.float32)
            if missing:
                # Only show warning; still proceed with 0-filled features
                click.echo(
                    f"       NOTE: {len(missing)} feature(s) not in data, "
                    f"filled with 0: {missing}",
                    err=True,
                )
            proba = eval_model.predict_proba(X)[:, 1]
            y_true = eval_df["target_sustained"].values
            n_days = max(eval_df["date"].nunique(), 1)
            rows = []
            for t in thresh_list:
                pred = (proba >= t).astype(int)
                signals = int(pred.sum())
                tp = int((pred & y_true).sum())
                prec = tp / signals if signals > 0 else 0.0
                rows.append({
                    "threshold": t,
                    "signals": signals,
                    "per_day": round(signals / n_days, 1),
                    "precision": round(prec, 4),
                    "tp": tp,
                    "fp": signals - tp,
                })
            return rows, None

        v2_rows, err = _eval_precision(
            test_df, model, feature_cols, audit_thresholds
        )
        if err:
            click.echo(
                f"  WARNING: could not evaluate precision -- {err}", err=True
            )
            v2_rows = []
        else:
            n_days_test = test_df["date"].nunique()
            click.echo(
                f"  Model: {model_name}  "
                f"[{len(test_df):,} rows, {n_days_test} days]\n"
            )
            click.echo(
                f"   {'Thresh':>6}  {'Signals':>8}  {'Per Day':>7}  "
                f"{'Precision':>9}  {'TP':>6}  {'FP':>6}"
            )
            click.echo("  " + "-" * 56)
            for row in v2_rows:
                click.echo(
                    f"   {row['threshold']:.2f}   {row['signals']:>8,}  "
                    f"{row['per_day']:>7.1f}  "
                    f"{row['precision']:>9.1%}  {row['tp']:>6}  {row['fp']:>6}"
                )

        # ── Step 6: Comparison model ──────────────────────────────────────
        compare_rows = None
        compare_name = None
        if compare_path_str:
            click.echo(f"\n[6/7] Comparison model: {compare_path_str}")
            comp_path = _Path(compare_path_str)
            if comp_path.exists():
                try:
                    with open(comp_path, "rb") as fh:
                        comp_artifact = _pickle.load(fh)
                except Exception:
                    import joblib as _joblib2
                    comp_artifact = _joblib2.load(comp_path)

                comp_model = comp_artifact["model"]
                comp_features = comp_artifact["feature_cols"]
                compare_name = comp_artifact.get("model_name") or comp_path.stem
                click.echo(
                    f"  Loaded: {compare_name}  |  {len(comp_features)} features"
                )

                compare_rows, comp_err = _eval_precision(
                    test_df, comp_model, comp_features, audit_thresholds
                )
                if comp_err:
                    click.echo(f"  WARNING: {comp_err}", err=True)
                    compare_rows = None
                else:
                    click.echo(
                        f"\n  {'Thresh':>6}  {'V2 Prec':>8}  "
                        f"{'V3 Prec':>8}  {'Delta':>7}  "
                        f"{'V2 Sigs':>8}  {'V3 Sigs':>8}"
                    )
                    click.echo("  " + "-" * 62)
                    for v2r, v3r in zip(v2_rows or [], compare_rows):
                        delta = v3r["precision"] - v2r["precision"]
                        click.echo(
                            f"   {v2r['threshold']:.2f}   "
                            f"{v2r['precision']:>8.1%}  {v3r['precision']:>8.1%}  "
                            f"{delta:>+7.1%}  "
                            f"{v2r['signals']:>8,}  {v3r['signals']:>8,}"
                        )
            else:
                click.echo(
                    f"  WARNING: compare model not found: {comp_path}", err=True
                )
        else:
            click.echo("\n[6/7] Comparison model: (not provided)")

        # ── Step 7: Fresh 2026 data ───────────────────────────────────────
        click.echo(f"\n[7/7] Fresh 2026 performance ({len(fresh_df):,} rows)...")
        fresh_rows = None
        if len(fresh_df) >= 10:
            fresh_rows, fresh_err = _eval_precision(
                fresh_df, model, feature_cols, audit_thresholds
            )
            if fresh_err:
                click.echo(f"  WARNING: {fresh_err}", err=True)
                fresh_rows = None
            else:
                click.echo(
                    f"  Dates: {fresh_df['date'].min()} -> {fresh_df['date'].max()}, "
                    f"{fresh_df['date'].nunique()} trading days\n"
                )
                click.echo(
                    f"   {'Thresh':>6}  {'Signals':>8}  {'Per Day':>7}  "
                    f"{'Precision':>9}  {'TP':>6}  {'FP':>6}"
                )
                click.echo("  " + "-" * 56)
                for row in fresh_rows:
                    click.echo(
                        f"   {row['threshold']:.2f}   {row['signals']:>8,}  "
                        f"{row['per_day']:>7.1f}  "
                        f"{row['precision']:>9.1%}  {row['tp']:>6}  {row['fp']:>6}"
                    )
        else:
            click.echo(
                "  No rows (or too few) with date >= 2026-01-01 in test split."
            )

        # ── Generate full report ──────────────────────────────────────────
        report_path = str(out_dir / f"{model_path.stem}_leakage_audit.json")
        report = detector.generate_report(output_path=report_path)

        report["precision_on_test"] = v2_rows
        report["precision_on_fresh_2026"] = fresh_rows
        if compare_rows is not None:
            report["precision_comparison"] = {
                model_name: v2_rows,
                compare_name: compare_rows,
            }

        with open(report_path, "w") as fh:
            _json.dump(report, fh, indent=2, default=str)

        # ── Final verdict ─────────────────────────────────────────────────
        click.echo("\n" + "=" * 72)
        click.echo("  LEAKAGE AUDIT VERDICT")
        click.echo("=" * 72)
        overall = report.get("overall_verdict", "UNKNOWN")
        safe = report.get("safe_to_proceed", False)
        critical = report.get("critical_issues", [])
        warnings_list = report.get("warnings", [])

        click.echo(f"\n  Model:          {model_name}")
        click.echo(f"  Features:       {len(feature_cols)} columns")
        click.echo(f"  Overall:        {overall}")
        click.echo(f"  Safe to use:    {'YES' if safe else 'NO'}")

        if critical:
            click.echo(f"\n  Critical issues ({len(critical)}):")
            for issue in critical:
                click.echo(f"    x {issue}")
        else:
            click.echo("\n  No critical issues detected.")

        if warnings_list:
            click.echo(f"\n  Warnings ({len(warnings_list)}):")
            for w in warnings_list:
                click.echo(f"    ! {w}")

        click.echo("\n  Interpretation:")
        if "opt_vol_pct_cumday" in feature_cols:
            click.echo(
                "    x CONFIRMED LEAKED: opt_vol_pct_cumday is present "
                "(total-day volume denominator -- unavailable at bar time T)"
            )
            click.echo(
                "    -> Use xgboost_v3_clean (65 features, leak removed) "
                "for live trading."
            )
        else:
            click.echo(
                "    + opt_vol_pct_cumday NOT present -- primary known leak is absent."
            )

        if v2_rows:
            best = max(v2_rows, key=lambda r: r["precision"])
            click.echo(
                f"    Best precision: {best['precision']:.1%} "
                f"@ threshold={best['threshold']} "
                f"({best['signals']:,} signals, {best['per_day']}/day)"
            )

        click.echo(f"\n  Report saved: {report_path}")
        click.echo("=" * 72 + "\n")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# data-dashboard
# ---------------------------------------------------------------------------


@ml_cli.command("data-dashboard")
@click.option(
    "--date", "date_str",
    required=True,
    help="Trading day to visualise (YYYY-MM-DD, e.g. 2025-06-25).",
)
@click.option(
    "--n-contracts", default=5, show_default=True,
    help="Number of option contracts to display (CALLs + PUTs).",
)
@click.option(
    "--model-path", default=None,
    help="Optional model artifact (.pkl) for the predictions layer.",
)
@click.option(
    "--features-dir", default=None,
    help="Feature CSV directory (overrides config).",
)
@click.option(
    "--spy-raw-dir", default="data/raw/spy", show_default=True,
    help="Directory containing raw SPY Parquet files.",
)
@click.option(
    "--options-raw-dir", default="data/raw/options/minute", show_default=True,
    help="Directory containing raw option Parquet files.",
)
@click.option(
    "--output", default=None,
    help="Output HTML path (default: reports/dashboard/{date}_dashboard.html).",
)
@click.option(
    "--show-derivation", is_flag=True, default=False,
    help="Print feature derivation console trace for the first 3 minute bars.",
)
@click.option(
    "--config-dir", default="config", show_default=True,
    help="Directory containing YAML config files.",
)
def data_dashboard(
    date_str, n_contracts, model_path, features_dir,
    spy_raw_dir, options_raw_dir, output, show_derivation, config_dir,
):
    """Build an interactive HTML dashboard for a single trading day.

    \b
    Dashboard rows:
      1. SPY candlestick + volume
      2. CALL option contracts (close price)
      3. PUT option contracts (close price)
      4. Feature heatmap (20 features, z-scored ±3sigma)
      5. Model confidence + actual outcomes

    \b
    Examples:
        python -m src.cli ml data-dashboard --date 2025-06-25

        python -m src.cli ml data-dashboard \\
            --date 2025-09-15 \\
            --model-path models/xgboost_v3_clean.pkl \\
            --n-contracts 6 \\
            --show-derivation
    """
    from pathlib import Path as _Path

    from src.analysis.data_dashboard_builder import DataDashboardBuilder

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )

        out_html = output or f"reports/dashboard/{date_str}_dashboard.html"
        _Path(out_html).parent.mkdir(parents=True, exist_ok=True)

        click.echo("\n" + "=" * 72)
        click.echo("  DATA PIPELINE DASHBOARD")
        click.echo("=" * 72)
        click.echo(f"  Date:          {date_str}")
        click.echo(f"  Contracts:     {n_contracts}")
        click.echo(f"  Features dir:  {feat_dir}")
        if model_path:
            click.echo(f"  Model:         {model_path}")
        click.echo(f"  Output:        {out_html}")
        click.echo("=" * 72)

        builder = DataDashboardBuilder(
            spy_raw_dir=spy_raw_dir,
            options_raw_dir=options_raw_dir,
            features_dir=feat_dir,
            model_path=model_path,
        )

        data = builder.extract_sample_day(date_str, n_contracts=n_contracts)

        if data["features"].empty:
            click.echo(
                f"\nError: no feature data found for {date_str}. "
                "Check --features-dir and --date.",
                err=True,
            )
            sys.exit(1)

        builder.build_dashboard(data, output_path=out_html)

        if show_derivation:
            click.echo("\n--- Feature Derivation (first 3 bars) ---")
            for idx in range(min(3, len(data["features"]))):
                builder.show_feature_derivation(data["features"], minute_idx=idx)

        click.echo(f"\n  Dashboard ready: {out_html}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# generate-magnitude-labels
# ---------------------------------------------------------------------------


@ml_cli.command("generate-magnitude-labels")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Minimum absolute % move (up or down) to label as positive.",
)
def generate_magnitude_labels(config_dir, features_dir, start_date, end_date, min_magnitude):
    """Preview magnitude label statistics across all feature CSVs.

    Loads feature CSVs, applies MagnitudeLabeler using pre-computed
    ``max_gain_120m`` / ``min_loss_120m`` columns, and prints a distribution
    summary.  No new files are written; labels are computed in-memory.

    \b
    Example:
        python -m src.cli ml generate-magnitude-labels \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --min-magnitude 20.0
    """
    from src.ml.train_xgboost import load_features
    from src.processing.magnitude_labeler import MagnitudeLabeler, MAGNITUDE_BUCKETS

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )

        click.echo("\n" + "=" * 70)
        click.echo("  MAGNITUDE LABEL PREVIEW")
        click.echo("=" * 70)
        click.echo(f"  Features dir:  {feat_dir}")
        click.echo(f"  Date range:    {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"  Min magnitude: {min_magnitude}%  (|move| ≥ this → positive)")
        click.echo("=" * 70)

        click.echo("\n[1/2] Loading feature CSVs…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir}", err=True
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        click.echo(f"\n[2/2] Applying MagnitudeLabeler (min_magnitude={min_magnitude}%)…")
        labeler = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"\n  Total rows:       {stats['n_total']:,}")
        click.echo(
            f"  Positive labels:  {stats['n_positive']:,}  "
            f"({stats['positive_rate']:.2%})  [|move| ≥ {min_magnitude}%]"
        )
        if stats["n_positive"] > 0:
            click.echo(
                f"  Avg abs move:     {stats['avg_abs_magnitude']:.1f}%  "
                f"(among positives)"
            )
            click.echo(
                f"  Up moves:         {stats['n_up']:,}  "
                f"({stats['n_up'] / stats['n_positive']:.1%} of positives)"
            )
            click.echo(
                f"  Down moves:       {stats['n_down']:,}  "
                f"({stats['n_down'] / stats['n_positive']:.1%} of positives)"
            )

        click.echo("\n  Magnitude bucket breakdown (all rows):")
        for bucket in MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct = count / max(stats["n_total"], 1) * 100
            bar = "█" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<8}: {count:>7,}  ({pct:5.1f}%)  {bar}")

        click.echo(
            f"\n  Straddle opportunity: {stats['n_positive']:,} bars where "
            f"|move| ≥ {min_magnitude}%\n"
            "  → Run 'ml train-magnitude-models' to train a magnitude predictor."
        )

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-magnitude-models
# ---------------------------------------------------------------------------


@ml_cli.command("train-magnitude-models")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date to include (YYYY-MM-DD).",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Minimum absolute % move required for a positive label.",
)
@click.option(
    "--n-trials",
    default=50,
    type=int,
    show_default=True,
    help="Optuna trials per model type.",
)
@click.option(
    "--cv-splits",
    default=3,
    type=int,
    show_default=True,
    help="TimeSeriesSplit folds inside each Optuna trial.",
)
@click.option(
    "--thresholds",
    default="0.50,0.60,0.70,0.75,0.80,0.85,0.90",
    show_default=True,
    help="Comma-separated evaluation thresholds.",
)
@click.option(
    "--position-size",
    default=12500.0,
    type=float,
    show_default=True,
    help="Straddle position size in USD per trade.",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for models and reports. "
    "Defaults to reports/magnitude_experiment.",
)
def train_magnitude_models(
    config_dir,
    features_dir,
    start_date,
    end_date,
    min_magnitude,
    n_trials,
    cv_splits,
    thresholds,
    position_size,
    output,
):
    """Train magnitude prediction models (direction-agnostic straddle strategy).

    Full pipeline:

    \b
    1. Load feature CSVs from --features-dir.
    2. Apply MagnitudeLabeler: target = |move| >= min_magnitude% in 120 min.
    3. Chronological 70/30 train/test split.
    4. Train XGBoost + LightGBM + RandomForest with Optuna (n_trials each).
    5. Evaluate at multiple thresholds with straddle P&L estimate.
    6. Save models and JSON report.

    \b
    Straddle profit model:
      TP (model predicts big move, move >= min_magnitude%):
          PnL = (avg_magnitude - 5) / 100 * position_size
          (winning side gains avg_magnitude%; losing side loses ~5% to theta)
      FP (model predicts big move, no big move):
          PnL = -8 / 100 * position_size
          (both sides decay from theta, ~8% total loss)

    \b
    Example:
        python -m src.cli ml train-magnitude-models \\
            --start-date 2025-03-03 --end-date 2026-02-19 \\
            --n-trials 50 --min-magnitude 20.0
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np
    import pandas as _pd

    from src.ml.multi_model_trainer import MultiModelTrainer
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.magnitude_labeler import MagnitudeLabeler, MAGNITUDE_BUCKETS

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("reports/magnitude_experiment")
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Parse thresholds
        try:
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        except ValueError:
            click.echo(
                f"Error: --thresholds must be comma-separated floats, "
                f"got: {thresholds!r}",
                err=True,
            )
            sys.exit(1)

        click.echo("\n" + "=" * 72)
        click.echo("  MAGNITUDE PREDICTION EXPERIMENT  (Step 63)")
        click.echo("=" * 72)
        click.echo(f"  Features dir:   {feat_dir}")
        click.echo(f"  Date range:     {start_date or 'all'} → {end_date or 'all'}")
        click.echo(f"  Min magnitude:  {min_magnitude}%  (|move| threshold)")
        click.echo(f"  Optuna trials:  {n_trials} per model type")
        click.echo(f"  CV splits:      {cv_splits}")
        click.echo(f"  Position size:  ${position_size:,.0f} per straddle")
        click.echo(f"  Output dir:     {out_dir}")
        click.echo("=" * 72)

        # ── Step 1: Load features ──────────────────────────────────────
        click.echo("\n[1/5] Loading feature CSVs…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            click.echo(
                f"Error: no feature data found in {feat_dir} for "
                f"{start_date} → {end_date}",
                err=True,
            )
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # ── Step 2: Apply MagnitudeLabeler ─────────────────────────────
        click.echo(f"\n[2/5] Applying MagnitudeLabeler (min_magnitude={min_magnitude}%)…")
        labeler = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        df = labeler.label(df)
        stats = labeler.validate(df)

        click.echo(f"  Total rows:      {stats['n_total']:,}")
        click.echo(
            f"  Positive labels: {stats['n_positive']:,}  ({stats['positive_rate']:.2%})"
        )
        if stats["n_positive"] > 0:
            click.echo(f"  Avg abs move:    {stats['avg_abs_magnitude']:.1f}%  (among positives)")
            click.echo(
                f"  UP / DOWN:       {stats['n_up']:,} / {stats['n_down']:,}  "
                f"({stats['n_up'] / stats['n_positive']:.1%} / "
                f"{stats['n_down'] / stats['n_positive']:.1%})"
            )
        click.echo("\n  Magnitude buckets (all rows):")
        for bucket in MAGNITUDE_BUCKETS:
            count = stats["magnitude_breakdown"].get(bucket, 0)
            pct = count / max(stats["n_total"], 1) * 100
            bar = "█" * max(1, int(pct / 2))
            click.echo(f"    {bucket:<8}: {count:>7,}  ({pct:5.1f}%)  {bar}")

        if stats["n_positive"] < 50:
            click.echo(
                "\n  WARNING: Very few positive labels — model may not generalise.",
                err=True,
            )

        # ── Step 3: Chronological train/test split ─────────────────────
        click.echo("\n[3/5] Splitting data (70% train / 30% test, chronological)…")
        n_total = len(df)
        split_idx = int(n_total * 0.70)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)

        click.echo(
            f"  Train: {len(train_df):,} rows  "
            f"({int(train_df['target_magnitude'].sum())} positives)"
        )
        click.echo(
            f"  Test:  {len(test_df):,} rows  "
            f"({int(test_df['target_magnitude'].sum())} positives)"
        )

        # Determine feature columns — exclude metadata, raw prices, labels
        _MAGNITUDE_EXTRA_COLS = {
            "target_magnitude", "abs_max_move_pct", "move_direction", "magnitude_bucket",
        }
        feature_cols = sorted([
            c for c in df.columns
            if c not in _NON_FEATURE_COLS and c not in _MAGNITUDE_EXTRA_COLS
        ])
        click.echo(f"  Feature columns: {len(feature_cols)}")

        # Leakage check — ensure no forward-looking columns slipped through
        _FORBIDDEN = {
            "max_gain_120m", "min_loss_120m", "target", "target_sustained",
            "abs_max_move_pct", "move_direction", "magnitude_bucket",
            "gain_pct_at_confirmation", "sustain_minutes_actual",
        }
        forbidden_found = [c for c in feature_cols if c in _FORBIDDEN]
        if forbidden_found:
            click.echo(
                f"\n  LEAKAGE ALERT: {len(forbidden_found)} forbidden column(s) "
                f"found in feature set: {forbidden_found}",
                err=True,
            )
            sys.exit(1)
        else:
            click.echo("  Leakage check: OK — no forbidden columns in feature set.")

        # ── Step 4: Train models ────────────────────────────────────────
        click.echo(
            f"\n[4/5] Training XGBoost + LightGBM + RandomForest "
            f"({n_trials} Optuna trials each)…"
        )
        click.echo("  (This may take several hours with n_trials=50)")

        trainer = MultiModelTrainer(n_trials=n_trials, cv_splits=cv_splits)
        artifacts = trainer.train(
            df=train_df,
            target_col="target_magnitude",
            feature_cols=feature_cols,
        )

        click.echo("\n  Training complete:")
        for model_name, artifact in artifacts.items():
            opt_score = artifact.get("optimization_score", 0.0)
            val_prec  = artifact.get("val_precision_at_0_70", 0.0)
            click.echo(
                f"    {model_name:<15}: "
                f"Optuna score={opt_score:.4f}  "
                f"val_precision@0.70={val_prec:.4f}"
            )

        # Save model artifacts
        saved_models = trainer.save_artifacts(artifacts, models_dir)
        click.echo(f"\n  Models saved to: {models_dir}/")
        for name, path in saved_models.items():
            size_kb = path.stat().st_size / 1024
            click.echo(f"    {path.name:<45}  {size_kb:>6.1f} KB")

        # ── Step 5: Evaluate with straddle P&L ─────────────────────────
        click.echo(f"\n[5/5] Evaluating on test set ({len(test_df):,} rows)…")

        # Straddle P&L constants
        _THETA_DECAY_WIN_SIDE_PCT = 5.0   # losing leg loses ~5% to theta
        _THETA_FP_TOTAL_PCT       = 8.0   # total theta on failed straddle (~8%)

        y_test = test_df["target_magnitude"].values
        test_dates_n = test_df["date"].nunique() if "date" in test_df.columns else 1

        eval_records = []

        for model_name, artifact in artifacts.items():
            model = artifact["model"]
            feature_cols_art = artifact.get("feature_cols", feature_cols)

            # Align feature columns
            X_test_cols = []
            for fc in feature_cols_art:
                if fc in test_df.columns:
                    X_test_cols.append(test_df[fc].fillna(0.0).values)
                else:
                    X_test_cols.append(_np.zeros(len(test_df), dtype=_np.float32))
            X_test = _np.column_stack(X_test_cols).astype(_np.float32)

            y_proba = model.predict_proba(X_test)[:, 1]

            click.echo(f"\n  {'=' * 68}")
            click.echo(f"  {model_name.upper()}")
            click.echo(f"  {'=' * 68}")
            click.echo(
                f"  {'Thresh':>7}  {'Signals':>8}  {'Per Day':>7}  "
                f"{'Prec':>7}  {'TP':>6}  {'FP':>6}  "
                f"{'AvgMag':>8}  {'Est Mo P&L':>12}"
            )
            click.echo("  " + "-" * 75)

            for thresh in threshold_list:
                y_pred = (y_proba >= thresh).astype(int)
                tp_mask = (y_pred == 1) & (y_test == 1)
                fp_mask = (y_pred == 1) & (y_test == 0)
                fn_mask = (y_pred == 0) & (y_test == 1)

                n_tp = int(tp_mask.sum())
                n_fp = int(fp_mask.sum())
                n_fn = int(fn_mask.sum())
                n_signals = n_tp + n_fp
                precision = n_tp / n_signals if n_signals else 0.0
                per_day   = n_signals / max(test_dates_n, 1)

                # Average magnitude of true positives
                tp_abs_moves = test_df.loc[tp_mask, "abs_max_move_pct"]
                avg_magnitude = float(tp_abs_moves.mean()) if n_tp else 0.0

                # Straddle P&L estimate
                if n_tp > 0:
                    straddle_win  = (avg_magnitude - _THETA_DECAY_WIN_SIDE_PCT) / 100.0 * position_size
                else:
                    straddle_win  = 0.0
                straddle_loss = -_THETA_FP_TOTAL_PCT / 100.0 * position_size

                # Scale to monthly (22 trading days)
                trading_days_test = max(test_dates_n, 1)
                monthly_tp  = n_tp / trading_days_test * 22
                monthly_fp  = n_fp / trading_days_test * 22
                monthly_pnl = monthly_tp * straddle_win + monthly_fp * straddle_loss

                pnl_str = f"${monthly_pnl:>+10,.0f}"
                click.echo(
                    f"  {thresh:>7.0%}  {n_signals:>8,}  {per_day:>7.1f}  "
                    f"{precision:>7.1%}  {n_tp:>6}  {n_fp:>6}  "
                    f"{avg_magnitude:>7.1f}%  {pnl_str}"
                )

                eval_records.append({
                    "model": model_name,
                    "threshold": thresh,
                    "n_signals": n_signals,
                    "signals_per_day": round(per_day, 2),
                    "precision": round(precision, 4),
                    "n_tp": n_tp,
                    "n_fp": n_fp,
                    "n_fn": n_fn,
                    "avg_magnitude_tp": round(avg_magnitude, 2),
                    "monthly_pnl_usd": round(monthly_pnl, 2),
                })

            # Direction breakdown at primary threshold (0.70)
            primary_thresh = 0.70
            y_pred_primary = (y_proba >= primary_thresh).astype(int)
            tp_primary = test_df[(y_pred_primary == 1) & (y_test == 1)]
            if len(tp_primary) > 0:
                up_count   = int((tp_primary["move_direction"] == "up").sum())
                down_count = int((tp_primary["move_direction"] == "down").sum())
                click.echo(
                    f"\n  TP direction @ {primary_thresh:.0%}: "
                    f"{up_count} UP  /  {down_count} DOWN"
                    f"  ({up_count / len(tp_primary):.1%} / {down_count / len(tp_primary):.1%})"
                )

            # Precision by magnitude bucket at primary threshold
            signals_primary = test_df[y_pred_primary == 1]
            if len(signals_primary) > 0:
                click.echo(f"\n  Precision by magnitude bucket @ {primary_thresh:.0%}:")
                click.echo(
                    f"    {'Bucket':<10}  {'Signals':>8}  {'TP':>6}  {'Precision':>10}"
                )
                click.echo("    " + "-" * 40)
                for bucket in MAGNITUDE_BUCKETS:
                    b_mask = signals_primary["magnitude_bucket"] == bucket
                    b_count = int(b_mask.sum())
                    if b_count == 0:
                        continue
                    b_tp = int((signals_primary[b_mask]["target_magnitude"] == 1).sum())
                    b_prec = b_tp / b_count
                    click.echo(
                        f"    {bucket:<10}  {b_count:>8,}  {b_tp:>6}  {b_prec:>10.1%}"
                    )

        # ── Save results ───────────────────────────────────────────────
        results = {
            "experiment": "magnitude_prediction",
            "step": 63,
            "min_magnitude_pct": min_magnitude,
            "date_range": [start_date, end_date],
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_test_dates": test_dates_n,
            "positive_rate_train": float(train_df["target_magnitude"].mean()),
            "positive_rate_test": float(test_df["target_magnitude"].mean()),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "position_size_usd": position_size,
            "straddle_tp_formula": "(avg_magnitude - 5) / 100 * position_size",
            "straddle_fp_formula": "-8 / 100 * position_size",
            "threshold_results": eval_records,
        }

        results_path = out_dir / "magnitude_results.json"
        results_path.write_text(_json.dumps(results, indent=2))
        click.echo(f"\n  Results saved: {results_path}")

        # ── Verdict ────────────────────────────────────────────────────
        click.echo("\n" + "=" * 72)
        click.echo("  MAGNITUDE EXPERIMENT VERDICT")
        click.echo("=" * 72)

        # Find best precision at 0.70 threshold across all models
        best_at_70 = max(
            (r for r in eval_records if abs(r["threshold"] - 0.70) < 0.001),
            key=lambda r: r["precision"],
            default=None,
        )
        if best_at_70:
            best_prec = best_at_70["precision"]
            best_model = best_at_70["model"]
            best_pnl   = best_at_70["monthly_pnl_usd"]
            click.echo(
                f"\n  Best @ 0.70:  {best_model}  →  {best_prec:.1%} precision  "
                f"(est. monthly P&L: ${best_pnl:+,.0f})"
            )
            if best_prec >= 0.60:
                click.echo(
                    "\n  VERDICT: MAGNITUDE MODEL WORKS\n"
                    "  Strategy: Trade straddles on high-confidence signals.\n"
                    f"  Edge: {best_prec - 0.50:.1%} above random."
                )
            elif best_prec >= 0.50:
                click.echo(
                    "\n  VERDICT: MARGINAL IMPROVEMENT\n"
                    "  Magnitude model is slightly better than directional.\n"
                    "  Consider using with strict threshold (>= 0.80)."
                )
            else:
                click.echo(
                    "\n  VERDICT: NO IMPROVEMENT\n"
                    "  Even magnitude prediction fails to beat random.\n"
                    "  Problem may be fundamental at this timeframe."
                )

        click.echo(f"\n  Output dir:  {out_dir}")
        click.echo("=" * 72)

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# audit-magnitude-model
# ---------------------------------------------------------------------------


@ml_cli.command("audit-magnitude-model")
@click.option(
    "--model-path",
    required=True,
    help="Path to magnitude model .pkl artifact (e.g. reports/magnitude_experiment/models/lightgbm.pkl).",
)
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Directory containing *_features.csv files. Defaults to config value.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date (YYYY-MM-DD). Defaults to full dataset.",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date (YYYY-MM-DD). Defaults to full dataset.",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Min absolute % move used during training (to re-derive labels).",
)
@click.option(
    "--output",
    default=None,
    help="Output directory for audit report. Defaults to reports/leakage_audit/.",
)
@click.option(
    "--random-samples",
    default=10_000,
    type=int,
    show_default=True,
    help="Number of random-noise samples for Test 1.",
)
def audit_magnitude_model(
    model_path,
    config_dir,
    features_dir,
    start_date,
    end_date,
    min_magnitude,
    output,
    random_samples,
):
    """Comprehensive leakage audit for magnitude prediction models.

    Runs 9 tests:

    \b
    1. Random-data test  — fires model on pure Gaussian noise
    2. Source-code audit — scans ml_feature_engineer.py for lookahead
    3. Known lookahead   — checks for opt_vol_pct_cumday etc.
    4. Target-in-features— ensures no label column in feature set
    5. Temporal ordering — verifies DataFrame is time-sorted
    6. Train/test contamination — zero date overlap required
    7. 120-min correlation analysis — Pearson vs max_gain_120m
    8. Feature importance red-flags — dominant / cumday / known-leak
    9. Magnitude-specific checks — forbidden label cols, future-name
       patterns, correlation with abs_max_move_pct

    \b
    Example:
        python -m src.cli ml audit-magnitude-model \\
            --model-path reports/magnitude_experiment/models/lightgbm.pkl \\
            --start-date 2025-03-03 --end-date 2026-02-19
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib
    import numpy as _np

    from src.ml.leakage_detector import LeakageDetector
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.magnitude_labeler import MagnitudeLabeler, MAGNITUDE_BUCKETS

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        out_dir = _Path(output) if output else _Path("reports/leakage_audit")
        out_dir.mkdir(parents=True, exist_ok=True)

        model_p = _Path(model_path)
        model_stem = model_p.stem

        log_path = out_dir / f"{model_stem}_magnitude_audit.log"
        import sys as _sys

        old_stdout = _sys.stdout

        class _Tee:
            """Write to both the original stdout and a log file simultaneously."""
            def __init__(self, orig, f):
                self._orig = orig   # captured BEFORE sys.stdout is replaced
                self._f    = f
            def write(self, msg): self._orig.write(msg); self._f.write(msg)
            def flush(self):      self._orig.flush();    self._f.flush()

        log_fh = open(log_path, "w")
        _sys.stdout = _Tee(old_stdout, log_fh)

        width = 72
        print()
        print("=" * width)
        print("  MAGNITUDE MODEL LEAKAGE AUDIT  (Step 65)")
        print("=" * width)
        print(f"  Model:        {model_path}")
        print(f"  Features dir: {feat_dir}")
        print(f"  Date range:   {start_date or 'all'} → {end_date or 'all'}")
        print(f"  Min magnitude:{min_magnitude}%")
        print(f"  Output dir:   {out_dir}")
        print("=" * width)

        # ── Load artifact ──────────────────────────────────────────────
        print("\n[1/7] Loading model artifact…")
        artifact = joblib.load(model_path)
        model_obj   = artifact["model"]
        feature_cols = artifact.get("feature_cols", [])
        model_type   = artifact.get("model_type", "unknown")
        target_col   = artifact.get("target_col", "target_magnitude")
        saved_at     = artifact.get("saved_at", "?")
        print(
            f"  Loaded: {model_stem}  |  {len(feature_cols)} features  "
            f"|  type: {model_type}  |  saved: {saved_at}"
        )
        print(f"  Target column: {target_col}")

        # ── Load features ──────────────────────────────────────────────
        print("\n[2/7] Loading feature data…")
        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            print(f"Error: no data found in {feat_dir}", file=_sys.stderr)
            _sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        print(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # Apply MagnitudeLabeler to get magnitude target columns
        print(f"\n[3/7] Applying MagnitudeLabeler (min_magnitude={min_magnitude}%)…")
        labeler = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        df = labeler.label(df)
        stats = labeler.validate(df)
        print(
            f"  Positive rate: {stats['positive_rate']:.2%}  "
            f"({stats['n_positive']:,} positives)"
        )

        # 70/30 chronological split (mirrors training)
        print("\n[4/7] Splitting data (70% train / 30% test)…")
        n_total   = len(df)
        split_idx = int(n_total * 0.70)
        train_df  = df.iloc[:split_idx].reset_index(drop=True)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)
        print(
            f"  Train: {len(train_df):,} rows  "
            f"({train_df['date'].min()} → {train_df['date'].max()})"
        )
        print(
            f"  Test : {len(test_df):,} rows  "
            f"({test_df['date'].min()} → {test_df['date'].max()})"
        )
        fresh_2026 = test_df[test_df["date"] >= "2026-01-01"] if "date" in test_df.columns else test_df.iloc[0:0]
        print(f"  Fresh 2026 subset: {len(fresh_2026):,} rows")

        train_dates = set(train_df["date"].unique()) if "date" in train_df.columns else set()
        test_dates  = set(test_df["date"].unique())  if "date" in test_df.columns else set()

        # ── Run all 9 leakage tests ─────────────────────────────────────
        print("\n[5/7] Running leakage detection tests…")
        detector = LeakageDetector()

        # Test 1: Random data
        print(f"  Test 1/9: Random data test ({random_samples:,} samples)…")
        r1 = detector.test_on_random_data(
            model_obj,
            feature_cols=feature_cols,
            n_samples=random_samples,
            high_confidence_threshold=0.80,
            max_acceptable_signals=50,   # 50/10000 = 0.5% — generous for high base-rate task
        )
        status1 = "FAIL" if r1["leakage_detected"] else "PASS"
        print(
            f"    -> {status1}  |  {r1['high_confidence_count']} high-conf signals on "
            f"{random_samples:,} random rows  |  "
            f"avg_p={r1['avg_confidence']:.3f}  max_p={r1['max_confidence']:.3f}"
        )

        # Test 2: Source-code audit
        print("  Test 2/9: Source code audit…")
        r2 = detector.audit_feature_definitions()
        status2 = "FAIL" if r2["leakage_likely"] else "PASS"
        print(f"    -> {status2}  |  {r2['pattern_count']} suspicious pattern(s)")

        # Test 3: Known lookahead features
        print("  Test 3/9: Known lookahead features check…")
        r3 = detector.check_known_lookahead_features(feature_cols)
        status3 = "FAIL" if r3["leakage_detected"] else "PASS"
        n_la = len(r3.get("lookahead_features", []))
        print(f"    -> {status3}  |  checked {r3['features_checked']} features")
        for item in r3.get("lookahead_features", []):
            print(f"       DETECTED: '{item['feature']}'")

        # Test 4: Target columns not in feature set
        print("  Test 4/9: Target columns not in feature set…")
        r4 = detector.verify_target_not_in_features(feature_cols)
        status4 = "FAIL" if r4["leakage_detected"] else "PASS"
        print(f"    -> {status4}  |  contaminated: {r4.get('contaminated_cols', [])}")

        # Test 5: Temporal ordering
        print("  Test 5/9: Temporal ordering…")
        r5 = detector.verify_temporal_ordering(df)
        status5 = "PASS" if r5.get("ordering_valid") else "WARN"
        print(
            f"    -> {status5}  |  {r5.get('violations', 0)} violation(s) "
            f"in {len(df):,} rows"
        )

        # Test 6: Train/test contamination
        print("  Test 6/9: Train/test contamination…")
        r6 = detector.detect_train_test_contamination(
            list(train_dates), list(test_dates)
        )
        status6 = "FAIL" if r6.get("contamination_detected") else "PASS"
        print(
            f"    -> {status6}  |  "
            f"{r6.get('overlap_count', 0)} overlapping dates  |  "
            f"gap={r6.get('gap_days', '?')} days"
        )

        # Test 7: 120-min correlation analysis
        print("  Test 7/9: 120-minute correlation analysis…")
        r7 = detector.check_120min_specific_leaks(df, feature_cols)
        n_name  = len(r7.get("suspicious_by_name", []))
        n_corr  = len([x for x in r7.get("suspicious_by_correlation", []) if x.get("severity") == "HIGH"])
        status7 = "FAIL" if r7.get("leakage_suspected") else (
            "WARN" if n_name > 0 else "PASS"
        )
        print(
            f"    -> {status7}  |  outcome col: {r7.get('outcome_column_used')}  "
            f"|  {n_name} name pattern(s)  |  {n_corr} suspicious corr(s)"
        )
        for item in r7.get("suspicious_by_name", []):
            print(f"       NAME FLAG: '{item['feature']}'")
        top5_corr = sorted(
            r7.get("correlation_table_top20", []),
            key=lambda x: x.get("abs_corr", 0),
            reverse=True,
        )[:5]
        if top5_corr:
            print("       Top 5 feature correlations with outcome:")
            for item in top5_corr:
                print(f"         {item['feature']:<40} {item['correlation']:>8.4f}")

        # Test 8: Feature importance red-flag analysis
        print("  Test 8/9: Feature importance red-flag analysis…")
        r8 = detector.analyze_feature_importance(model_obj, feature_cols, top_n=10)
        status8 = "FAIL" if r8.get("leakage_suspected") else "PASS"
        print(f"    -> {status8}  |  top feature: {r8.get('top_feature')}")
        for flag in r8.get("red_flags", []):
            print(f"       {'WARNING' if 'WARNING' in flag else 'CRITICAL'}: {flag}")
        print("\n  Top 10 features by importance:")
        for entry in r8.get("ranked_features", [])[:10]:
            rank_str = f"{entry['rank']:>2}. {entry['feature']:<42} {entry['importance_pct']}"
            flag = ""
            if "CRITICAL" in entry.get("flag", ""):
                flag = " [CRITICAL]"
            elif "WARNING" in entry.get("flag", ""):
                flag = " [WARN]"
            print(f"     {rank_str}{flag}")

        # Test 9: Magnitude-specific leakage
        print("  Test 9/9: Magnitude-specific leakage checks…")
        r9 = detector.check_magnitude_specific_leaks(df, feature_cols)
        status9 = "FAIL" if r9["leakage_detected"] else "PASS"
        n_forbidden = len(r9.get("forbidden_in_features", []))
        n_future    = len(r9.get("suspicious_by_name", []))
        n_high_corr = len([x for x in r9.get("suspicious_by_correlation", []) if x.get("severity") == "HIGH"])
        print(
            f"    -> {status9}  |  outcome col: {r9.get('outcome_column_used')}  "
            f"|  {n_forbidden} forbidden col(s)  |  "
            f"{n_future} future-name(s)  |  {n_high_corr} high-corr(s)"
        )
        for item in r9.get("forbidden_in_features", []):
            print(f"       FORBIDDEN: '{item['feature']}' — {item['reason']}")
        for item in r9.get("suspicious_by_name", []):
            print(f"       FUTURE NAME: '{item['feature']}'")
        mag_top5 = sorted(
            r9.get("correlation_table_top20", []),
            key=lambda x: x.get("abs_corr", 0),
            reverse=True,
        )[:5]
        if mag_top5:
            print(f"       Top 5 feature correlations with {r9.get('outcome_column_used')}:")
            for item in mag_top5:
                sev = next(
                    (x["severity"] for x in r9.get("suspicious_by_correlation", []) if x["feature"] == item["feature"]),
                    "",
                )
                marker = "  [HIGH]" if sev == "HIGH" else ("  [MOD]" if sev == "MODERATE" else "")
                print(f"         {item['feature']:<40} {item['correlation']:>8.4f}{marker}")

        # ── Precision at thresholds on test holdout ────────────────────
        print(f"\n[6/7] Precision at thresholds on test holdout…")
        X_test_cols = []
        for fc in feature_cols:
            if fc in test_df.columns:
                X_test_cols.append(test_df[fc].fillna(0.0).values)
            else:
                X_test_cols.append(_np.zeros(len(test_df), dtype=_np.float32))
        X_test = _np.column_stack(X_test_cols).astype(_np.float32)
        y_test = test_df[target_col].values if target_col in test_df.columns else test_df["target_magnitude"].values

        print(f"  Model: {model_stem}  [{len(test_df):,} rows, {len(test_dates)} days]")
        test_precision_rows = []
        print()
        print(f"   {'Thresh':>6}   {'Signals':>8}  {'Per Day':>7}  {'Precision':>10}      TP      FP")
        print("  " + "-" * 60)
        y_proba = model_obj.predict_proba(X_test)[:, 1]
        for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            y_pred = (y_proba >= thresh).astype(int)
            tp = int(((y_pred == 1) & (y_test == 1)).sum())
            fp = int(((y_pred == 1) & (y_test == 0)).sum())
            n_sig = tp + fp
            prec = tp / n_sig if n_sig else 0.0
            per_day = n_sig / max(len(test_dates), 1)
            print(
                f"   {thresh:>6.2f}     {n_sig:>8,}  {per_day:>7.1f}  "
                f"{prec:>10.1%}    {tp:>6}  {fp:>6}"
            )
            test_precision_rows.append({
                "threshold": thresh,
                "n_signals": n_sig,
                "per_day": round(per_day, 2),
                "precision": round(prec, 4),
                "n_tp": tp,
                "n_fp": fp,
            })

        # Fresh 2026 evaluation
        if len(fresh_2026) > 0:
            print(f"\n[6b/7] Fresh 2026 performance ({len(fresh_2026):,} rows)…")
            fresh_dates = set(fresh_2026["date"].unique()) if "date" in fresh_2026.columns else set()
            X_fresh_cols = []
            for fc in feature_cols:
                if fc in fresh_2026.columns:
                    X_fresh_cols.append(fresh_2026[fc].fillna(0.0).values)
                else:
                    X_fresh_cols.append(_np.zeros(len(fresh_2026), dtype=_np.float32))
            X_fresh = _np.column_stack(X_fresh_cols).astype(_np.float32)
            y_fresh = fresh_2026[target_col].values if target_col in fresh_2026.columns else fresh_2026["target_magnitude"].values
            y_fresh_proba = model_obj.predict_proba(X_fresh)[:, 1]
            fresh_dates_range = (
                f"{fresh_2026['date'].min()} → {fresh_2026['date'].max()}"
                if "date" in fresh_2026.columns else "unknown range"
            )
            print(f"  Dates: {fresh_dates_range}, {len(fresh_dates)} trading days")
            print()
            print(f"   {'Thresh':>6}   {'Signals':>8}  {'Per Day':>7}  {'Precision':>10}      TP      FP")
            print("  " + "-" * 60)
            for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
                y_pred_f = (y_fresh_proba >= thresh).astype(int)
                tp_f = int(((y_pred_f == 1) & (y_fresh == 1)).sum())
                fp_f = int(((y_pred_f == 1) & (y_fresh == 0)).sum())
                n_f  = tp_f + fp_f
                prec_f = tp_f / n_f if n_f else 0.0
                pd_f   = n_f / max(len(fresh_dates), 1)
                print(
                    f"   {thresh:>6.2f}     {n_f:>8,}  {pd_f:>7.1f}  "
                    f"{prec_f:>10.1%}    {tp_f:>6}  {fp_f:>6}"
                )

        # ── Save report ─────────────────────────────────────────────────
        print(f"\n[7/7] Generating audit report…")
        report = detector.generate_report(
            str(out_dir / f"{model_stem}_magnitude_audit.json")
        )
        report["model_path"]             = str(model_path)
        report["model_type"]             = model_type
        report["n_features"]             = len(feature_cols)
        report["min_magnitude_pct"]      = min_magnitude
        report["test_precision_results"] = test_precision_rows

        # Re-save enriched report
        with open(out_dir / f"{model_stem}_magnitude_audit.json", "w") as fh:
            _json.dump(report, fh, indent=2, default=str)

        # ── Verdict ─────────────────────────────────────────────────────
        all_statuses = [status1, status2, status3, status4, status6, status8, status9]
        any_fail     = any(s == "FAIL" for s in all_statuses)

        print()
        print("=" * width)
        print("  MAGNITUDE AUDIT VERDICT")
        print("=" * width)
        print()
        print(f"  Model:          {model_stem}")
        print(f"  Model type:     {model_type}")
        print(f"  Features:       {len(feature_cols)} columns")
        print(f"  Overall:        {'LEAKAGE DETECTED' if any_fail else 'NO LEAKAGE DETECTED'}")
        print(f"  Safe to use:    {'NO' if any_fail else 'YES'}")
        print()

        if report["critical_issues"]:
            print(f"  Critical issues ({len(report['critical_issues'])}):")
            for issue in report["critical_issues"]:
                print(f"    x {issue}")
        else:
            print("  No critical issues found.")
            print()

        # Interpretation
        if not any_fail:
            # Check random data specifics
            rand_high = r1["high_confidence_count"]
            rand_max  = r1["max_confidence"]
            base_rate = stats["positive_rate"]
            print("  Interpretation:")
            print(
                f"    Model fires {rand_high} signal(s) on {random_samples:,} random rows "
                f"(max p={rand_max:.3f})."
            )
            if rand_max < 0.60:
                print("    -> Random test: model is uncertain on noise. CLEAN.")
            elif rand_max < 0.80:
                print(
                    "    -> Random test: moderate confidence on noise (p < 0.80). "
                    "Possible but not alarming."
                )
            else:
                print(
                    "    -> Random test: WARNING — model shows high confidence on noise. "
                    "Investigate further."
                )
            print(
                f"\n    Base rate: {base_rate:.1%} positive. Precision above "
                f"{base_rate:.0%} is genuine model lift."
            )
            best_prec = max(r["precision"] for r in test_precision_rows if r["n_signals"] > 0) if test_precision_rows else 0
            print(
                f"    Best test precision: {best_prec:.1%}. "
                f"Lift above base rate: {best_prec - base_rate:+.1%}."
            )
            if best_prec - base_rate > 0.10:
                print("    -> GENUINE model lift confirmed. No leakage detected.")
                print(
                    "    -> Proceed to fresh out-of-sample validation before "
                    "live deployment."
                )
            else:
                print(
                    "    -> Model barely exceeds base rate — "
                    "precision may be driven by base rate alone."
                )
        else:
            print("  -> LEAKAGE CONFIRMED — do NOT use this model for trading.")

        print(
            f"\n  Report saved: {out_dir / f'{model_stem}_magnitude_audit.json'}"
        )
        print(f"  Log saved:    {log_path}")
        print("=" * width)

        _sys.stdout = old_stdout
        log_fh.close()

    except Exception as exc:
        import traceback
        try:
            _sys.stdout = old_stdout
            log_fh.close()
        except Exception:
            pass
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# final-optimization
# ---------------------------------------------------------------------------


@ml_cli.command("final-optimization")
@click.option(
    "--features-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing *_features.csv files.",
)
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--n-trials",
    default=200,
    type=int,
    show_default=True,
    help="Optuna trials per model type (LightGBM and RandomForest).",
)
@click.option(
    "--target-signals-per-day",
    default=2.5,
    type=float,
    show_default=True,
    help="Desired average number of signals per trading day.",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Minimum absolute % move required for a positive label.",
)
@click.option(
    "--n-mc-simulations",
    default=1000,
    type=int,
    show_default=True,
    help="Number of Monte Carlo iterations for P&L simulation.",
)
@click.option(
    "--output",
    default="reports/final_optimization",
    show_default=True,
    help="Output directory for models and JSON results.",
)
def final_optimization(
    features_dir,
    config_dir,
    n_trials,
    target_signals_per_day,
    min_magnitude,
    n_mc_simulations,
    output,
):
    """Maximum-precision exhaustive search (Step 67).

    Targets ≤3 signals/day at near-100% precision using deep Optuna
    hyperparameter search on LightGBM and RandomForest, ensemble
    combination testing, and Monte Carlo P&L simulation.

    \b
    Pipeline:
    1. Load all feature CSVs from --features-dir.
    2. Apply MagnitudeLabeler (direction-agnostic ±20% target).
    3. Derive feature column list (exclude metadata & label cols).
    4. Chronological 70/30 split; 80/20 sub-split within train.
    5. Build numpy arrays for each split.
    6. Run DeepHyperparameterOptimizer: LightGBM + RandomForest.
    7. Retrain best models on full training set.
    8. Test ensemble combinations (AND/OR/AVG strategies).
    9. Run 1 000-iteration Monte Carlo P&L simulation.
    10. Save final_optimization_results.json.

    \b
    Example:
        python -m src.cli ml final-optimization \\
            --features-dir data/processed/features \\
            --n-trials 200 --target-signals-per-day 2.5
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib as _joblib
    import numpy as _np

    from src.ml.deep_hyperparameter_optimizer import (
        DeepHyperparameterOptimizer,
        EnsembleStrategy,
        MonteCarloSimulator,
    )
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.magnitude_labeler import MagnitudeLabeler

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        out_dir = _Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = out_dir / "final_optimization.log"
        import sys as _sys

        old_stdout = _sys.stdout

        class _Tee:
            """Write to both the original stdout and a log file simultaneously."""
            def __init__(self, orig, f):
                self._orig = orig
                self._f    = f
            def write(self, msg): self._orig.write(msg); self._f.write(msg)
            def flush(self):      self._orig.flush();    self._f.flush()

        log_fh = open(log_path, "w")
        _sys.stdout = _Tee(old_stdout, log_fh)

        width = 72
        print()
        print("=" * width)
        print("  FINAL OPTIMIZATION — Maximum Precision / Minimum Signals  (Step 67)")
        print("=" * width)
        print(f"  Features dir:          {features_dir}")
        print(f"  Min magnitude:         {min_magnitude}%")
        print(f"  Optuna trials/model:   {n_trials}")
        print(f"  Target signals/day:    {target_signals_per_day}")
        print(f"  Monte Carlo sims:      {n_mc_simulations:,}")
        print(f"  Output dir:            {out_dir}")
        print("=" * width)

        # ── [1/7] Load features ───────────────────────────────────────
        print("\n[1/7] Loading feature CSVs…")
        df = load_features(features_dir, None, None)
        if df.empty:
            print(f"Error: no feature data found in {features_dir}", file=_sys.stderr)
            _sys.stdout = old_stdout
            log_fh.close()
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        print(f"  Loaded {len(df):,} rows across {n_dates} dates")

        # ── [2/7] Apply MagnitudeLabeler ──────────────────────────────
        print(f"\n[2/7] Applying MagnitudeLabeler (min_magnitude={min_magnitude}%)…")
        labeler    = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        labeled_df = labeler.label(df)
        stats      = labeler.validate(labeled_df)
        print(
            f"  Total rows:     {stats['n_total']:,}"
            f"  |  Positive rate: {stats['positive_rate']:.2%}"
            f"  ({stats['n_positive']:,} positives)"
        )

        # ── [3/7] Feature columns ─────────────────────────────────────
        print("\n[3/7] Deriving feature column list…")
        _MAGNITUDE_EXTRA_COLS = {
            "target_magnitude", "abs_max_move_pct", "move_direction", "magnitude_bucket",
        }
        feature_cols = sorted([
            c for c in labeled_df.columns
            if c not in _NON_FEATURE_COLS and c not in _MAGNITUDE_EXTRA_COLS
        ])
        print(f"  Feature columns: {len(feature_cols)}")

        # ── [4/7] Chronological split ─────────────────────────────────
        print("\n[4/7] Splitting data (70% train / 30% test, then 80/20 sub-split)…")
        split_idx   = int(len(labeled_df) * 0.70)
        train_df    = labeled_df.iloc[:split_idx].reset_index(drop=True)
        test_df_lbl = labeled_df.iloc[split_idx:].reset_index(drop=True)

        sub_idx   = int(len(train_df) * 0.80)
        train_sub = train_df.iloc[:sub_idx].reset_index(drop=True)
        val_sub   = train_df.iloc[sub_idx:].reset_index(drop=True)

        val_days  = (
            val_sub["date"].nunique() if "date" in val_sub.columns
            else max(len(val_sub) // 390, 1)
        )
        test_days = (
            test_df_lbl["date"].nunique() if "date" in test_df_lbl.columns
            else max(len(test_df_lbl) // 390, 1)
        )

        print(
            f"  Train (full):  {len(train_df):,} rows  "
            f"({train_df['date'].min()} → {train_df['date'].max()})"
        )
        print(f"  Train sub:     {len(train_sub):,} rows")
        print(f"  Val sub:       {len(val_sub):,} rows  ({val_days} trading days)")
        print(f"  Test:          {len(test_df_lbl):,} rows  ({test_days} trading days)")

        # ── [5/7] Build numpy arrays ──────────────────────────────────
        print("\n[5/7] Building numpy arrays…")
        X_train_sub  = train_sub[feature_cols].fillna(0.0).values.astype(_np.float32)
        y_train_sub  = train_sub["target_magnitude"].values.astype(_np.int8)
        X_val_sub    = val_sub[feature_cols].fillna(0.0).values.astype(_np.float32)
        y_val_sub    = val_sub["target_magnitude"].values.astype(_np.int8)

        X_full_train = train_df[feature_cols].fillna(0.0).values.astype(_np.float32)
        y_full_train = train_df["target_magnitude"].values.astype(_np.int8)
        X_test       = test_df_lbl[feature_cols].fillna(0.0).values.astype(_np.float32)
        y_test       = test_df_lbl["target_magnitude"].values.astype(_np.int8)

        print(
            f"  X_train_sub={X_train_sub.shape}  X_val={X_val_sub.shape}  "
            f"X_full_train={X_full_train.shape}  X_test={X_test.shape}"
        )

        # ── [6/7] Hyperparameter optimisation ────────────────────────
        print(
            f"\n[6/7] Optimising LightGBM + RandomForest "
            f"({n_trials} Optuna trials each)…"
        )
        print("  (This will take several hours for n_trials=200)")

        opt = DeepHyperparameterOptimizer(
            X_train=X_train_sub,
            y_train=y_train_sub,
            X_val=X_val_sub,
            y_val=y_val_sub,
            val_days=val_days,
            target_signals_per_day=target_signals_per_day,
        )

        print("\n  → LightGBM optimisation…")
        lgbm_result = opt.optimize_lightgbm_precision(n_trials=n_trials)
        print(
            f"  LightGBM best score: {lgbm_result['best_value']:.4f}"
            f"  |  params: {lgbm_result['best_params']}"
        )

        print("\n  → RandomForest optimisation…")
        rf_result = opt.optimize_randomforest_precision(n_trials=n_trials)
        print(
            f"  RandomForest best score: {rf_result['best_value']:.4f}"
            f"  |  params: {rf_result['best_params']}"
        )

        # Retrain best models on the full training set
        print("\n  Retraining best LightGBM on full training set…")
        import warnings as _warnings
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier

        lgbm_fit_params = {
            **lgbm_result["best_params"],
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            lgbm_best = LGBMClassifier(**lgbm_fit_params)
            lgbm_best.fit(X_full_train, y_full_train)

        print("  Retraining best RandomForest on full training set…")
        rf_fit_params = {
            **rf_result["best_params"],
            "random_state": 42,
            "n_jobs": -1,
        }
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            rf_best = RandomForestClassifier(**rf_fit_params)
            rf_best.fit(X_full_train, y_full_train)

        # Save retrained artifacts
        _joblib.dump(
            {
                "model":        lgbm_best,
                "feature_cols": feature_cols,
                "best_params":  lgbm_result["best_params"],
                "model_type":   "lgbm",
                "best_value":   lgbm_result["best_value"],
            },
            out_dir / "lgbm_deep_opt.pkl",
        )
        _joblib.dump(
            {
                "model":        rf_best,
                "feature_cols": feature_cols,
                "best_params":  rf_result["best_params"],
                "model_type":   "rf",
                "best_value":   rf_result["best_value"],
            },
            out_dir / "rf_deep_opt.pkl",
        )
        print(
            f"  Artifacts saved → {out_dir}/lgbm_deep_opt.pkl"
            f"  |  {out_dir}/rf_deep_opt.pkl"
        )

        # ── [7/7] Ensemble + Monte Carlo ──────────────────────────────
        print("\n[7/7] Testing ensemble combinations + Monte Carlo simulation…")

        ensemble_result = EnsembleStrategy().test_ensemble_combinations(
            lgbm_model=lgbm_best,
            rf_model=rf_best,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df_lbl.reset_index(drop=True),
            n_test_days=test_days,
            target_signals_per_day=target_signals_per_day,
        )

        best_combo = ensemble_result["best"]
        if best_combo is not None:
            print(
                f"\n  Best ensemble:  strategy={best_combo['strategy']}"
                f"  threshold={best_combo['threshold']:.2f}"
                f"  precision={best_combo['precision']:.3f}"
                f"  spd={best_combo['signals_per_day']:.2f}"
                f"  avg_mag={best_combo['avg_magnitude']:.1f}%"
            )
            signals_mask = best_combo["signals_mask"]
        else:
            print("\n  WARNING: No viable ensemble combination found.")
            signals_mask = _np.zeros(len(test_df_lbl), dtype=bool)

        # Monte Carlo simulation
        n_test_months = test_days / 22.0
        print(f"\n  Running Monte Carlo ({n_mc_simulations:,} simulations)…")
        mc_result = MonteCarloSimulator().simulate_monthly_pnl(
            test_df=test_df_lbl.reset_index(drop=True),
            signals_mask=signals_mask,
            n_simulations=n_mc_simulations,
            n_test_months=n_test_months,
        )
        print(
            f"  Monthly P&L:  mean=${mc_result['mean']:,.0f}"
            f"  median=${mc_result['median']:,.0f}"
            f"  std=${mc_result['std']:,.0f}"
            f"  win_rate={mc_result['win_rate']:.1%}"
        )
        print(
            f"  Percentiles:  p5=${mc_result['percentiles']['p5']:,.0f}"
            f"  p25=${mc_result['percentiles']['p25']:,.0f}"
            f"  p75=${mc_result['percentiles']['p75']:,.0f}"
            f"  p95=${mc_result['percentiles']['p95']:,.0f}"
        )

        # ── Save final JSON ───────────────────────────────────────────
        serialisable_results = [
            {k: v for k, v in r.items() if k != "signals_mask"}
            for r in ensemble_result["results"]
        ]
        serialisable_best = (
            {k: v for k, v in best_combo.items() if k != "signals_mask"}
            if best_combo is not None
            else None
        )

        final_results = {
            "lgbm": {
                "best_params": lgbm_result["best_params"],
                "best_value":  lgbm_result["best_value"],
            },
            "rf": {
                "best_params": rf_result["best_params"],
                "best_value":  rf_result["best_value"],
            },
            "ensemble": {
                "results": serialisable_results,
                "best":    serialisable_best,
            },
            "monte_carlo": mc_result,
        }

        results_path = out_dir / "final_optimization_results.json"
        results_path.write_text(_json.dumps(final_results, indent=2))

        print()
        print("=" * width)
        print("  FINAL OPTIMIZATION COMPLETE")
        print("=" * width)
        print(f"\n  Results saved: {results_path}")
        print(f"  Log saved:     {log_path}")
        print("=" * width)

        _sys.stdout = old_stdout
        log_fh.close()

    except Exception as exc:
        import traceback
        try:
            _sys.stdout = old_stdout
            log_fh.close()
        except Exception:
            pass
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# walk-forward-validation
# ---------------------------------------------------------------------------


@ml_cli.command("walk-forward-validation")
@click.option(
    "--lgbm-model",
    required=True,
    type=click.Path(exists=True),
    help="Path to LightGBM artifact .pkl (e.g. reports/final_optimization/lgbm_deep_opt.pkl).",
)
@click.option(
    "--rf-model",
    required=True,
    type=click.Path(exists=True),
    help="Path to RandomForest artifact .pkl (e.g. reports/final_optimization/rf_deep_opt.pkl).",
)
@click.option(
    "--features-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing *_features.csv files.",
)
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--n-splits",
    default=5,
    type=int,
    show_default=True,
    help="Number of TimeSeriesSplit walk-forward folds.",
)
@click.option(
    "--threshold",
    default=0.97,
    type=float,
    show_default=True,
    help="AVG-probability threshold for firing a signal.",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Min absolute % move used to derive target_magnitude labels.",
)
@click.option(
    "--output",
    default="reports/walk_forward_validation",
    show_default=True,
    help="Output directory for JSON results.",
)
def walk_forward_validation(
    lgbm_model,
    rf_model,
    features_dir,
    config_dir,
    n_splits,
    threshold,
    min_magnitude,
    output,
):
    """Walk-forward stability test for the pre-trained magnitude ensemble.

    Tests the AVG@threshold ensemble (LightGBM + RandomForest from
    final-optimization) across n_splits time-series windows to verify
    that the single-period precision holds across market regimes.

    \b
    No retraining occurs — the same frozen models are evaluated on each
    chronological test window.

    \b
    Verdicts:
        STABLE          : mean_precision >= 95% AND std < 5%
        GOOD_BUT_VARIABLE: mean_precision >= 90%
        UNSTABLE        : mean_precision < 90%

    \b
    Example:
        python -m src.cli ml walk-forward-validation \\
            --lgbm-model reports/final_optimization/lgbm_deep_opt.pkl \\
            --rf-model   reports/final_optimization/rf_deep_opt.pkl \\
            --features-dir data/processed/features \\
            --n-splits 5 --threshold 0.97
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib as _joblib

    from src.ml.ensemble_walk_forward import EnsembleWalkForwardValidator
    from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
    from src.processing.magnitude_labeler import MagnitudeLabeler

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        out_dir = _Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load artifacts ────────────────────────────────────────────
        click.echo("\n[1/4] Loading model artifacts…")
        lgbm_artifact = _joblib.load(lgbm_model)
        rf_artifact   = _joblib.load(rf_model)
        lgbm_obj      = lgbm_artifact["model"]
        rf_obj        = rf_artifact["model"]
        feature_cols  = lgbm_artifact.get("feature_cols", rf_artifact.get("feature_cols", []))

        click.echo(f"  LightGBM : {lgbm_model}  ({len(feature_cols)} features)")
        click.echo(f"  RF       : {rf_model}")

        if not feature_cols:
            click.echo(
                "Error: feature_cols not found in artifact — "
                "run final-optimization first.",
                err=True,
            )
            sys.exit(1)

        # ── Load + label features ─────────────────────────────────────
        click.echo("\n[2/4] Loading and labelling feature CSVs…")
        df = load_features(features_dir, None, None)
        if df.empty:
            click.echo(f"Error: no data in {features_dir}", err=True)
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        labeler = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        df      = labeler.label(df)
        stats   = labeler.validate(df)
        click.echo(
            f"  Positive rate: {stats['positive_rate']:.2%}"
            f"  ({stats['n_positive']:,} positives  |  min_magnitude={min_magnitude}%)"
        )

        # Verify feature_cols exist in df
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            click.echo(
                f"  WARNING: {len(missing_cols)} feature col(s) missing from data "
                f"(will fill with 0): {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}",
                err=True,
            )
            for c in missing_cols:
                df[c] = 0.0

        # ── Run walk-forward validation ───────────────────────────────
        click.echo(
            f"\n[3/4] Running walk-forward validation "
            f"({n_splits} folds, threshold={threshold})…"
        )
        validator = EnsembleWalkForwardValidator(feature_cols=feature_cols)
        wf_result = validator.validate_ensemble_strategy(
            lgbm_model=lgbm_obj,
            rf_model=rf_obj,
            full_df=df,
            threshold=threshold,
            n_splits=n_splits,
        )

        # ── Save results ──────────────────────────────────────────────
        click.echo("\n[4/4] Saving results…")
        results_path = out_dir / "walk_forward_validation_results.json"
        results_path.write_text(_json.dumps(wf_result, indent=2))
        click.echo(f"  Saved → {results_path}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ── build-signal-dashboard ────────────────────────────────────────────────────
@ml_cli.command("build-signal-dashboard")
@click.option(
    "--lgbm-model",
    required=True,
    type=click.Path(exists=True),
    help="Path to LightGBM artifact (.pkl) from final-optimization.",
)
@click.option(
    "--rf-model",
    required=True,
    type=click.Path(exists=True),
    help="Path to RandomForest artifact (.pkl) from final-optimization.",
)
@click.option(
    "--features-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing *_features.csv files.",
)
@click.option("--config-dir", default="config", show_default=True)
@click.option(
    "--threshold",
    default=0.97,
    type=float,
    show_default=True,
    help="AVG-probability threshold for declaring a signal.",
)
@click.option(
    "--min-magnitude",
    default=20.0,
    type=float,
    show_default=True,
    help="Minimum abs move (%) used to define target_magnitude=1.",
)
@click.option(
    "--output",
    default="reports/signal_dashboard.html",
    show_default=True,
    help="Output path for the standalone HTML dashboard.",
)
def build_signal_dashboard(
    lgbm_model,
    rf_model,
    features_dir,
    config_dir,
    threshold,
    min_magnitude,
    output,
):
    """Extract all historical signals and build an interactive HTML dashboard.

    \b
    Example:
        python -m src.cli ml build-signal-dashboard \\
            --lgbm-model reports/final_optimization/lgbm_deep_opt.pkl \\
            --rf-model   reports/final_optimization/rf_deep_opt.pkl \\
            --features-dir data/processed/features/ \\
            --threshold 0.97 \\
            --output reports/signal_dashboard.html
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib as _joblib

    from src.ml.train_xgboost import load_features
    from src.processing.magnitude_labeler import MagnitudeLabeler
    from src.analysis.signal_extractor import SignalExtractor
    from src.visualization.signal_dashboard import SignalDashboard
    from src.utils.config_loader import ConfigLoader
    from src.utils.logger import setup_logger

    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        out_path = _Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ── [1/5] Load model artifacts ────────────────────────────────────────
        click.echo("\n[1/5] Loading model artifacts…")
        lgbm_artifact = _joblib.load(lgbm_model)
        rf_artifact   = _joblib.load(rf_model)
        lgbm_obj      = lgbm_artifact["model"]
        rf_obj        = rf_artifact["model"]
        feature_cols  = lgbm_artifact.get("feature_cols", rf_artifact.get("feature_cols", []))

        lgbm_name = _Path(lgbm_model).stem
        rf_name   = _Path(rf_model).stem
        click.echo(f"  LightGBM : {lgbm_name}  ({len(feature_cols)} features)")
        click.echo(f"  RF       : {rf_name}")

        if not feature_cols:
            click.echo(
                "Error: feature_cols not found in artifact — "
                "run final-optimization first.",
                err=True,
            )
            sys.exit(1)

        # ── [2/5] Load + label features ───────────────────────────────────────
        click.echo("\n[2/5] Loading and labelling feature CSVs…")
        df = load_features(features_dir, None, None)
        if df.empty:
            click.echo(f"Error: no data in {features_dir}", err=True)
            sys.exit(1)
        n_dates = df["date"].nunique() if "date" in df.columns else "?"
        click.echo(f"  Loaded {len(df):,} rows across {n_dates} dates")

        labeler  = MagnitudeLabeler(min_magnitude_pct=min_magnitude)
        df       = labeler.label(df)
        stats    = labeler.validate(df)
        click.echo(
            f"  Positive rate: {stats['positive_rate']:.2%}"
            f"  ({stats['n_positive']:,} positives  |  min_magnitude={min_magnitude}%)"
        )

        # Backfill missing feature columns with 0
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            click.echo(
                f"  WARNING: {len(missing_cols)} feature col(s) missing (will fill 0): "
                f"{missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}",
                err=True,
            )
            for c in missing_cols:
                df[c] = 0.0

        # ── [3/5] Extract signals ─────────────────────────────────────────────
        click.echo(f"\n[3/5] Extracting signals (threshold={threshold})…")
        extractor   = SignalExtractor(feature_cols=feature_cols)
        signals_df  = extractor.extract_all_signals(
            lgbm_model=lgbm_obj,
            rf_model=rf_obj,
            full_df=df,
            threshold=threshold,
        )

        # ── [4/5] Save signal CSV alongside the HTML ──────────────────────────
        click.echo("\n[4/5] Saving signal CSV…")
        csv_path = out_path.with_suffix(".csv")
        signals_df.to_csv(csv_path, index=False)
        click.echo(f"  Saved → {csv_path}")

        # ── [5/5] Build dashboard ─────────────────────────────────────────────
        click.echo("\n[5/5] Building dashboard…")
        dashboard = SignalDashboard(
            lgbm_model_name=lgbm_name,
            rf_model_name=rf_name,
            threshold=threshold,
        )
        dashboard.build_dashboard(signals_df=signals_df, output_path=out_path)

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-stacked-ensemble  (Step 74)
# ---------------------------------------------------------------------------


@ml_cli.command("train-stacked-ensemble")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--features-dir",
    default=None,
    help="Override feature_engineering.features_dir.",
)
@click.option(
    "--start-date",
    default=None,
    help="Earliest feature date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Latest feature date (YYYY-MM-DD).",
)
@click.option(
    "--n-regimes",
    default=4,
    type=int,
    show_default=True,
    help="Number of market regime clusters.",
)
@click.option(
    "--contamination",
    default=0.05,
    type=float,
    show_default=True,
    help="Anomaly filter contamination rate.",
)
@click.option(
    "--output",
    default="models/stacked_ensemble.pkl",
    show_default=True,
    help="Path to save ensemble artifact.",
)
def train_stacked_ensemble(
    config_dir, features_dir, start_date, end_date, n_regimes, contamination, output
):
    """Train a stacked ensemble (XGBoost + LightGBM + RF + meta-learner).

    Pipeline: load features → chronological split → train base models on
    train set → generate meta-features on val set → train meta-learner →
    calibrate → fit anomaly filter → save artifact.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        import numpy as np

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )

        # Inject ensemble config
        config.setdefault("stacked_ensemble", {})
        config["stacked_ensemble"]["n_regimes"] = n_regimes
        config["stacked_ensemble"]["contamination"] = contamination

        from src.ml.train_xgboost import load_features, _NON_FEATURE_COLS
        from src.ml.data_splitter import DataSplitter
        from src.ml.stacked_ensemble import StackedEnsemble

        df = load_features(feat_dir, start_date, end_date)
        if df.empty:
            raise ValueError(f"No feature data found in {feat_dir}")

        splitter = DataSplitter(config)
        train_df, val_df, test_df = splitter.split(df)

        click.echo(f"\n--- Stacked Ensemble Training ---")
        click.echo(f"Features dir: {feat_dir}")
        click.echo(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        feature_cols = sorted(
            [c for c in df.columns if c not in _NON_FEATURE_COLS]
        )
        click.echo(f"Features: {len(feature_cols)}")

        ensemble = StackedEnsemble(config)
        metrics = ensemble.train(train_df, val_df, feature_cols)
        ensemble.save(output)

        click.echo(f"\nVal precision:  {metrics['val_precision']:.4f}")
        click.echo(f"Val recall:     {metrics['val_recall']:.4f}")
        click.echo(f"Val F1:         {metrics['val_f1']:.4f}")
        click.echo(f"Val ROC-AUC:    {metrics['val_roc_auc']:.4f}")
        click.echo(f"Base models:    {metrics['base_model_names']}")
        click.echo(f"Saved to:       {output}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train-exit-model  (Step 75)
# ---------------------------------------------------------------------------


@ml_cli.command("train-exit-model")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--entry-model-path",
    required=True,
    help="Path to the entry model artifact (ensemble or XGBoost .pkl).",
)
@click.option(
    "--features-dir",
    default=None,
    help="Override feature_engineering.features_dir.",
)
@click.option(
    "--raw-options-dir",
    default=None,
    help="Directory with raw option minute Parquet files.",
)
@click.option(
    "--output",
    default="models/exit_model.pkl",
    show_default=True,
    help="Path to save exit model artifact.",
)
def train_exit_model(config_dir, entry_model_path, features_dir, raw_options_dir, output):
    """Train a LightGBM exit signal model from historical entry signals.

    Generates exit training data by walking forward from historical entry
    signals through raw option bars, then trains a classifier to predict
    optimal exit points.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        raw_opt_dir = raw_options_dir or config.get("feature_engineering", {}).get(
            "options_data_dir", "data/raw/options/minute"
        )

        from src.ml.exit_signal_model import ExitSignalModel

        exit_model = ExitSignalModel(config)
        metrics = exit_model.train(
            entry_model_path=entry_model_path,
            features_dir=feat_dir,
            raw_options_dir=raw_opt_dir,
        )
        exit_model.save(output)

        click.echo(f"\n--- Exit Model Training ---")
        click.echo(f"Training samples:  {metrics.get('n_train_samples', 0)}")
        click.echo(f"Exit label rate:   {metrics.get('exit_label_rate', 0):.2%}")
        click.echo(f"Val AUC:           {metrics.get('val_auc', 0):.4f}")
        click.echo(f"Val precision:     {metrics.get('val_precision', 0):.4f}")
        click.echo(f"Saved to:          {output}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# simulate-real-bars  (Step 76)
# ---------------------------------------------------------------------------


@ml_cli.command("simulate-real-bars")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--entry-model-path",
    required=True,
    help="Path to entry model artifact (ensemble .pkl).",
)
@click.option(
    "--exit-model-path",
    required=True,
    help="Path to exit model artifact (.pkl).",
)
@click.option(
    "--features-dir",
    default=None,
    help="Override feature_engineering.features_dir.",
)
@click.option(
    "--raw-options-dir",
    default=None,
    help="Directory with raw option minute Parquet files.",
)
@click.option(
    "--start-date",
    default=None,
    help="Start date for simulation (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="End date for simulation (YYYY-MM-DD).",
)
@click.option(
    "--output",
    default="reports/real_bar_simulation",
    show_default=True,
    help="Output directory for simulation results.",
)
def simulate_real_bars(
    config_dir, entry_model_path, exit_model_path, features_dir,
    raw_options_dir, start_date, end_date, output
):
    """Run bar-by-bar trade simulation using entry + exit models.

    Walks through each day chronologically, scores entry candidates,
    then simulates actual trades by walking forward through raw
    option minute bars with exit model decision-making.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        feat_dir = features_dir or config.get("feature_engineering", {}).get(
            "features_dir", "data/processed/features"
        )
        raw_opt_dir = raw_options_dir or config.get("feature_engineering", {}).get(
            "options_data_dir", "data/raw/options/minute"
        )

        from src.ml.real_bar_simulator import RealBarSimulator

        simulator = RealBarSimulator(config)
        result = simulator.simulate_period(
            entry_model_path=entry_model_path,
            exit_model_path=exit_model_path,
            features_dir=feat_dir,
            raw_options_dir=raw_opt_dir,
            start_date=start_date,
            end_date=end_date,
            output_dir=output,
        )

        click.echo(f"\n--- Real-Bar Simulation Results ---")
        click.echo(f"Period: {result.get('start_date')} → {result.get('end_date')}")
        click.echo(f"Trading days: {result.get('n_days', 0)}")
        click.echo(f"Total trades: {result.get('total_trades', 0)}")
        click.echo(f"Win rate: {result.get('win_rate', 0):.1%}")
        click.echo(f"Net P&L: ${result.get('net_pnl', 0):,.0f}")
        click.echo(f"Output: {output}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# run-signal-pipeline  (Step 77)
# ---------------------------------------------------------------------------


@ml_cli.command("run-signal-pipeline")
@click.option(
    "--config-dir",
    default="config",
    show_default=True,
    help="Directory containing YAML config files.",
)
@click.option(
    "--ensemble-path",
    required=True,
    help="Path to stacked ensemble artifact (.pkl).",
)
@click.option(
    "--exit-model-path",
    required=True,
    help="Path to exit model artifact (.pkl).",
)
@click.option(
    "--start-date",
    default=None,
    help="Backtest start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Backtest end date (YYYY-MM-DD).",
)
@click.option(
    "--output",
    default="reports/signal_pipeline",
    show_default=True,
    help="Output directory for pipeline results.",
)
def run_signal_pipeline(
    config_dir, ensemble_path, exit_model_path, start_date, end_date, output
):
    """Run end-to-end signal pipeline backtest.

    Orchestrates: feature loading → regime detection → entry scoring →
    top-N selection → real-bar simulation with exit model → report.
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()
        setup_logger(config)

        from src.ml.signal_pipeline import SignalPipeline

        pipeline = SignalPipeline(config)
        result = pipeline.run_backtest(
            ensemble_path=ensemble_path,
            exit_model_path=exit_model_path,
            start_date=start_date,
            end_date=end_date,
            output_dir=output,
        )

        click.echo(f"\n--- Signal Pipeline Results ---")
        click.echo(f"Period: {result.get('start_date')} → {result.get('end_date')}")
        click.echo(f"Trading days: {result.get('n_days', 0)}")
        click.echo(f"Total trades: {result.get('total_trades', 0)}")
        click.echo(f"Win rate: {result.get('win_rate', 0):.1%}")
        click.echo(f"Net P&L: ${result.get('net_pnl', 0):,.0f}")
        click.echo(f"Avg trades/day: {result.get('avg_trades_per_day', 0):.1f}")
        click.echo(f"Report: {output}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# build-trade-dashboard  (Step 78)
# ---------------------------------------------------------------------------


@ml_cli.command("build-trade-dashboard")
@click.option(
    "--report-path",
    required=True,
    help="Path to pipeline report JSON file.",
)
@click.option(
    "--output",
    default="reports/trade_dashboard.html",
    show_default=True,
    help="Output path for the HTML dashboard.",
)
def build_trade_dashboard(report_path, output):
    """Build an interactive HTML trade dashboard from pipeline results.

    Generates a standalone Plotly-based dark-theme dashboard with
    equity curves, monthly P&L, regime analysis, and trade details.
    """
    try:
        import json

        from src.visualization.trade_dashboard import TradeDashboard

        with open(report_path) as f:
            report = json.load(f)

        dashboard = TradeDashboard()
        dashboard.build_dashboard(report, output)

        click.echo(f"\n--- Trade Dashboard ---")
        click.echo(f"Report: {report_path}")
        click.echo(f"Dashboard: {output}")

    except Exception as exc:
        import traceback
        click.echo(f"Error: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
