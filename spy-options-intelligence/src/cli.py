# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""CLI entry point for SPY Options Intelligence data ingestion.

Usage:
    python -m src.cli backfill --start-date 2025-01-27 --end-date 2025-01-28
    python -m src.cli backfill --resume
    python -m src.cli backfill --help
"""

import sys

import click

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger


@click.group()
def cli():
    """SPY Options Intelligence — Data Ingestion CLI."""
    pass


@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    default=None,
    help="Override start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default=None,
    help="Override end date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from last checkpoint, skipping completed dates.",
)
def backfill(config_dir, start_date, end_date, resume):
    """Run historical SPY data backfill."""
    try:
        # Load configuration
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        # Override dates if provided via CLI
        if start_date:
            config.setdefault("historical", {}).setdefault("backfill", {})[
                "start_date"
            ] = start_date
        if end_date:
            config.setdefault("historical", {}).setdefault("backfill", {})[
                "end_date"
            ] = end_date

        # Setup logging
        setup_logger(config)
        logger = get_logger()

        # Run pipeline
        from src.orchestrator.historical_runner import HistoricalRunner

        runner = HistoricalRunner(config)
        stats = runner.run(resume=resume)

        # Print summary
        click.echo("\n--- Backfill Summary ---")
        click.echo(f"Date range:      {stats['start_date']} → {stats['end_date']}")
        click.echo(f"Dates processed: {stats['dates_processed']}")
        click.echo(f"Dates skipped:   {stats['dates_skipped']}")
        click.echo(f"Total fetched:   {stats['total_fetched']}")
        click.echo(f"Total written:   {stats['total_written']}")
        click.echo(f"Invalid:         {stats['total_invalid']}")
        click.echo(f"Duplicates:      {stats['total_duplicates']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
