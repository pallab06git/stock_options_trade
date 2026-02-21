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
def backtest(config_dir, model_path, start_date, end_date):
    """Run ML model backtest on the chronological test split.

    Loads the model artifact, splits features chronologically, predicts on
    the test set (never seen during training), and reports precision / recall /
    lift over the random baseline.  Saves a per-trade CSV and a JSON metrics
    report.
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
