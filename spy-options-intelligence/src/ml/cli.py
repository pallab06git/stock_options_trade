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
