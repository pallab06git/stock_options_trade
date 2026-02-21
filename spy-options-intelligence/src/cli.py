# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""CLI entry point for SPY Options Intelligence data ingestion.

Usage:
    python -m src.cli backfill --start-date 2025-01-27 --end-date 2025-01-28
    python -m src.cli backfill --ticker TSLA --start-date 2025-01-27
    python -m src.cli backfill-all --start-date 2025-01-27 --end-date 2025-01-27
    python -m src.cli backfill-news --start-date 2026-02-01 --end-date 2026-02-09
    python -m src.cli stream-news
    python -m src.cli discover --date 2026-02-09
    python -m src.cli simulate --source spy --date 2026-02-10 --speed 10
    python -m src.cli workers list
    python -m src.cli workers stop --ticker SPY
    python -m src.cli workers stop --all
    python -m src.cli health
    python -m src.cli health --ticker SPY
    python -m src.cli health --json
"""

import json as json_mod
import sys
from pathlib import Path

import click

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger


@click.group()
def cli():
    """SPY Options Intelligence — Data Ingestion CLI."""
    pass


# Register ML subgroup (generate-features, train, feature-importance, backtest)
from src.ml.cli import ml_cli  # noqa: E402
cli.add_command(ml_cli)


@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--ticker",
    default="SPY",
    help="Equity ticker symbol (e.g. SPY, TSLA).",
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
@click.option(
    "--rate-limit",
    default=None,
    type=float,
    help="Override total_requests_per_minute rate limit.",
)
def backfill(config_dir, ticker, start_date, end_date, resume, rate_limit):
    """Run historical equity data backfill."""
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

        # Override rate limit if provided
        if rate_limit is not None:
            config.setdefault("polygon", {}).setdefault("rate_limiting", {})[
                "total_requests_per_minute"
            ] = rate_limit

        # Setup logging
        setup_logger(config)
        logger = get_logger()

        # Run pipeline
        from src.orchestrator.historical_runner import HistoricalRunner

        runner = HistoricalRunner(config, ticker=ticker)
        stats = runner.run(resume=resume)

        # Print summary
        click.echo(f"\n--- Backfill Summary ({ticker}) ---")
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


@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--ticker",
    default="SPY",
    help="Equity ticker symbol to stream (e.g. SPY, TSLA).",
)
def stream(config_dir, ticker):
    """Stream real-time equity data via WebSocket."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)
        logger = get_logger()

        from src.orchestrator.streaming_runner import StreamingRunner

        runner = StreamingRunner(config, ticker=ticker)
        click.echo(f"Starting real-time stream for {ticker}...")
        stats = runner.run()

        click.echo(f"\n--- Streaming Summary ({ticker}) ---")
        click.echo(f"Status:          {stats.get('status', 'unknown')}")
        click.echo(f"Messages received: {stats['messages_received']}")
        click.echo(f"Messages written:  {stats['messages_written']}")
        click.echo(f"Invalid:           {stats['messages_invalid']}")
        click.echo(f"Duplicates:        {stats['messages_duplicates']}")
        click.echo(f"Batches flushed:   {stats['batches_flushed']}")

    except KeyboardInterrupt:
        click.echo("\nStream interrupted by user.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("stream-options")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--date",
    required=True,
    help="Trading date for contract loading (YYYY-MM-DD).",
)
def stream_options(config_dir, date):
    """Stream real-time options data via WebSocket."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.orchestrator.options_streaming_runner import OptionsStreamingRunner

        runner = OptionsStreamingRunner(config, date=date)
        click.echo(f"Starting real-time options stream for {date}...")
        stats = runner.run()

        click.echo(f"\n--- Options Streaming Summary ---")
        click.echo(f"Status:            {stats.get('status', 'unknown')}")
        click.echo(f"Messages received: {stats['messages_received']}")
        click.echo(f"Messages written:  {stats['messages_written']}")
        click.echo(f"Invalid:           {stats['messages_invalid']}")
        click.echo(f"Duplicates:        {stats['messages_duplicates']}")
        click.echo(f"Batches flushed:   {stats['batches_flushed']}")

    except KeyboardInterrupt:
        click.echo("\nStream interrupted by user.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("backfill-vix")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from last checkpoint, skipping completed dates.",
)
def backfill_vix(config_dir, start_date, end_date, resume):
    """Run historical VIX data backfill."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        # Override dates
        config.setdefault("historical", {}).setdefault("backfill", {})[
            "start_date"
        ] = start_date
        config.setdefault("historical", {}).setdefault("backfill", {})[
            "end_date"
        ] = end_date

        setup_logger(config)

        from src.data_sources.polygon_vix_client import PolygonVIXClient
        from src.orchestrator.historical_runner import HistoricalRunner
        from src.processing.validator import RecordValidator
        from src.utils.connection_manager import ConnectionManager

        cm = ConnectionManager(config)
        vix_client = PolygonVIXClient(config, cm)
        validator = RecordValidator("vix")

        runner = HistoricalRunner(
            config,
            ticker="I:VIX",
            connection_manager=cm,
            client=vix_client,
            validator=validator,
        )
        stats = runner.run(resume=resume)

        click.echo(f"\n--- VIX Backfill Summary ---")
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


@cli.command("stream-vix")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
def stream_vix(config_dir):
    """Stream real-time VIX data via WebSocket."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.polygon_vix_client import PolygonVIXClient
        from src.orchestrator.streaming_runner import StreamingRunner
        from src.processing.validator import RecordValidator
        from src.utils.connection_manager import ConnectionManager

        cm = ConnectionManager(config)
        vix_client = PolygonVIXClient(config, cm)
        validator = RecordValidator("vix")

        runner = StreamingRunner(
            config,
            ticker="I:VIX",
            connection_manager=cm,
            client=vix_client,
            validator=validator,
        )
        click.echo("Starting real-time VIX stream...")
        stats = runner.run()

        click.echo(f"\n--- VIX Streaming Summary ---")
        click.echo(f"Status:            {stats.get('status', 'unknown')}")
        click.echo(f"Messages received: {stats['messages_received']}")
        click.echo(f"Messages written:  {stats['messages_written']}")
        click.echo(f"Invalid:           {stats['messages_invalid']}")
        click.echo(f"Duplicates:        {stats['messages_duplicates']}")
        click.echo(f"Batches flushed:   {stats['batches_flushed']}")

    except KeyboardInterrupt:
        click.echo("\nStream interrupted by user.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("backfill-news")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from last checkpoint, skipping completed dates.",
)
def backfill_news(config_dir, start_date, end_date, resume):
    """Run historical news data backfill."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        # Override dates
        config.setdefault("historical", {}).setdefault("backfill", {})[
            "start_date"
        ] = start_date
        config.setdefault("historical", {}).setdefault("backfill", {})[
            "end_date"
        ] = end_date

        setup_logger(config)

        from src.data_sources.news_client import PolygonNewsClient
        from src.orchestrator.historical_runner import HistoricalRunner
        from src.processing.deduplicator import Deduplicator
        from src.processing.validator import RecordValidator
        from src.utils.connection_manager import ConnectionManager

        cm = ConnectionManager(config)
        news_client = PolygonNewsClient(config, cm)
        validator = RecordValidator("news")
        deduplicator = Deduplicator(key_field="article_id")

        runner = HistoricalRunner(
            config,
            ticker="news",
            connection_manager=cm,
            client=news_client,
            validator=validator,
            deduplicator=deduplicator,
        )
        stats = runner.run(resume=resume)

        click.echo(f"\n--- News Backfill Summary ---")
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


@cli.command("stream-news")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
def stream_news(config_dir):
    """Stream news articles via polling (no WebSocket)."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.news_client import PolygonNewsClient
        from src.orchestrator.streaming_runner import StreamingRunner
        from src.processing.deduplicator import Deduplicator
        from src.processing.validator import RecordValidator
        from src.utils.connection_manager import ConnectionManager

        cm = ConnectionManager(config)
        news_client = PolygonNewsClient(config, cm)
        validator = RecordValidator("news")
        deduplicator = Deduplicator(key_field="article_id")

        runner = StreamingRunner(
            config,
            ticker="news",
            connection_manager=cm,
            client=news_client,
            validator=validator,
            deduplicator=deduplicator,
        )
        click.echo("Starting news polling stream...")
        stats = runner.run()

        click.echo(f"\n--- News Streaming Summary ---")
        click.echo(f"Status:            {stats.get('status', 'unknown')}")
        click.echo(f"Messages received: {stats['messages_received']}")
        click.echo(f"Messages written:  {stats['messages_written']}")
        click.echo(f"Invalid:           {stats['messages_invalid']}")
        click.echo(f"Duplicates:        {stats['messages_duplicates']}")
        click.echo(f"Batches flushed:   {stats['batches_flushed']}")

    except KeyboardInterrupt:
        click.echo("\nStream interrupted by user.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--date",
    required=True,
    help="Trading date to consolidate (YYYY-MM-DD).",
)
def consolidate(config_dir, date):
    """Consolidate SPY, VIX, options, and news data for a trading day."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.processing.consolidator import Consolidator

        consolidator = Consolidator(config)
        stats = consolidator.consolidate(date)

        click.echo(f"\n--- Consolidation Summary ---")
        click.echo(f"Date:               {stats.get('date')}")
        click.echo(f"Status:             {stats.get('status')}")
        if stats.get("status") == "success":
            click.echo(f"Total rows:         {stats.get('total_rows')}")
            click.echo(f"Minutes:            {stats.get('minutes')}")
            click.echo(f"Unique options:     {stats.get('unique_options')}")
            click.echo(f"Options contracts:  {stats.get('options_contracts_processed')}")
            click.echo(f"VIX available:      {stats.get('vix_available')}")
            click.echo(f"News available:     {stats.get('news_available')}")
        else:
            click.echo(f"Reason:             {stats.get('reason', 'unknown')}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("prepare-training")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
def prepare_training(config_dir, start_date, end_date):
    """Prepare ML training data from consolidated historical files."""
    try:
        from datetime import datetime, timedelta

        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.processing.training_data_prep import TrainingDataPrep

        # Build date list
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        prep = TrainingDataPrep(config)
        stats = prep.prepare(dates)

        click.echo(f"\n--- Training Data Prep Summary ---")
        click.echo(f"Dates processed:    {stats['dates_processed']}")
        click.echo(f"Dates skipped:      {stats['dates_skipped']}")
        click.echo(f"Total rows in:      {stats['total_rows_in']}")
        click.echo(f"Total rows out:     {stats['total_rows_out']}")
        click.echo(f"Rows filtered:      {stats['total_rows_filtered']}")
        click.echo(f"Unique options:     {stats['unique_options']}")
        click.echo(f"Prediction window:  {stats['prediction_window_minutes']} min")
        click.echo(f"Min coverage:       {stats['min_target_coverage_pct']}%")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Feed simulator command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--source",
    required=True,
    help="Data source to replay (spy, vix, options, news, consolidated).",
)
@click.option(
    "--date",
    required=True,
    help="Date of the Parquet file to replay (YYYY-MM-DD).",
)
@click.option(
    "--speed",
    default=1.0,
    type=float,
    help="Playback speed multiplier (1.0 = real-time, 10.0 = 10x, 0 = no delay).",
)
def simulate(config_dir, source, date, speed):
    """Replay historical data as a simulated real-time stream."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.orchestrator.simulator import FeedSimulator

        sim = FeedSimulator(config, source=source, date=date, speed=speed)
        click.echo(
            f"Starting feed simulation: {source}/{date} at {speed}x speed..."
        )

        count = 0
        try:
            for record in sim.stream_realtime():
                count += 1
        except KeyboardInterrupt:
            pass

        stats = sim.get_stats()
        click.echo(f"\n--- Simulation Summary ---")
        click.echo(f"Source:           {stats['source']}")
        click.echo(f"Date:             {stats['date']}")
        click.echo(f"Speed:            {stats['speed']}x")
        click.echo(f"Records loaded:   {stats['records_loaded']}")
        click.echo(f"Records emitted:  {stats['records_emitted']}")
        click.echo(f"Total delay:      {stats['total_delay_seconds']:.1f}s")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Schema drift commands
# ---------------------------------------------------------------------------

@cli.command("schema-check")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--source",
    required=True,
    help="Data source name (spy, vix, options, news, consolidated).",
)
@click.option(
    "--date",
    required=True,
    help="Date of the Parquet file to check (YYYY-MM-DD).",
)
def schema_check(config_dir, source, date):
    """Check for schema drift against stored baseline."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.monitoring.schema_monitor import SchemaMonitor

        # Resolve Parquet path
        if source == "consolidated":
            parquet_path = f"data/processed/consolidated/{date}.parquet"
        else:
            parquet_path = f"data/raw/{source}/{date}.parquet"

        if not Path(parquet_path).exists():
            click.echo(f"File not found: {parquet_path}", err=True)
            sys.exit(1)

        monitor = SchemaMonitor(config)
        alerts = monitor.check_drift(source, parquet_path)

        if alerts:
            click.echo(f"Schema drift detected for '{source}' on {date}:")
            for alert in alerts:
                click.echo(f"  {alert}")
        else:
            click.echo(f"No schema drift detected for '{source}' on {date}.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("schema-baseline")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--source",
    required=True,
    help="Data source name (spy, vix, options, news, consolidated).",
)
@click.option(
    "--date",
    required=True,
    help="Date of the Parquet file to use as baseline (YYYY-MM-DD).",
)
def schema_baseline(config_dir, source, date):
    """Capture or recapture a schema baseline from a Parquet file."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.monitoring.schema_monitor import SchemaMonitor

        # Resolve Parquet path
        if source == "consolidated":
            parquet_path = f"data/processed/consolidated/{date}.parquet"
        else:
            parquet_path = f"data/raw/{source}/{date}.parquet"

        if not Path(parquet_path).exists():
            click.echo(f"File not found: {parquet_path}", err=True)
            sys.exit(1)

        monitor = SchemaMonitor(config)
        baseline = monitor.capture_baseline(source, parquet_path)
        path = monitor.save_baseline(source, baseline)

        click.echo(f"Baseline captured for '{source}' from {date}:")
        click.echo(f"  Columns: {baseline['column_count']}")
        click.echo(f"  Saved to: {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("backfill-all")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from last checkpoint for each ticker.",
)
def backfill_all(config_dir, start_date, end_date, resume):
    """Run parallel backfill for all configured tickers."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.orchestrator.parallel_runner import ParallelRunner

        runner = ParallelRunner(config)
        click.echo(
            f"Starting parallel backfill for {runner.tickers} "
            f"({start_date} → {end_date})"
        )
        results = runner.run(
            start_date=start_date,
            end_date=end_date,
            resume=resume,
            config_dir=config_dir,
        )

        click.echo("\n--- Parallel Backfill Results ---")
        for ticker, result in results.items():
            status = "OK" if result["exit_code"] == 0 else f"FAILED ({result['exit_code']})"
            click.echo(f"  {ticker}: {status}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Options discovery command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--date",
    required=True,
    help="Trading date for contract discovery (YYYY-MM-DD).",
)
def discover(config_dir, date):
    """Discover options contracts within strike range of opening price."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.polygon_options_client import PolygonOptionsClient
        from src.utils.connection_manager import ConnectionManager

        cm = ConnectionManager(config)
        client = PolygonOptionsClient(config, cm)

        # Fetch opening price
        opening_price = client.fetch_opening_price(date)
        click.echo(f"Opening price for {client.underlying_ticker} on {date}: {opening_price}")

        # Discover contracts
        contracts = client.discover_contracts(date, opening_price)

        if not contracts:
            click.echo("No contracts found in strike range.")
            return

        # Save to disk
        path = client.save_contracts(contracts, date)

        # Print summary
        click.echo(f"\n--- Options Discovery Summary ---")
        click.echo(f"Date:            {date}")
        click.echo(f"Opening price:   {opening_price}")
        lower = round(opening_price * (1 - client.strike_range_pct), 2)
        upper = round(opening_price * (1 + client.strike_range_pct), 2)
        click.echo(f"Strike range:    [{lower}, {upper}]")
        click.echo(f"Contracts found: {len(contracts)}")
        click.echo(f"Saved to:        {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Workers subgroup
# ---------------------------------------------------------------------------

@cli.group()
def workers():
    """Manage worker processes."""
    pass


@workers.command("list")
def workers_list():
    """Show all workers with PID, ticker, and status."""
    try:
        from src.orchestrator.process_manager import ProcessManager

        manager = ProcessManager({})
        worker_list = manager.list_workers()

        if not worker_list:
            click.echo("No workers registered.")
            return

        click.echo(f"{'Ticker':<8} {'PID':<8} {'Status':<12} {'Alive':<6} {'Started'}")
        click.echo("-" * 60)
        for w in worker_list:
            alive = "yes" if w["alive"] else "no"
            click.echo(
                f"{w['ticker']:<8} {w['pid']:<8} {w['status']:<12} "
                f"{alive:<6} {w['started_at']}"
            )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@workers.command("stop")
@click.option("--ticker", default=None, help="Stop a specific worker by ticker.")
@click.option("--all", "stop_all", is_flag=True, help="Stop all workers.")
def workers_stop(ticker, stop_all):
    """Stop one or all workers."""
    try:
        from src.orchestrator.process_manager import ProcessManager

        manager = ProcessManager({})

        if stop_all:
            results = manager.stop_all()
            for t, success in results.items():
                status = "stopped" if success else "failed/not running"
                click.echo(f"  {t}: {status}")
        elif ticker:
            success = manager.stop_worker(ticker)
            if success:
                click.echo(f"Stopped worker {ticker}")
            else:
                click.echo(f"Could not stop worker {ticker} (not found or not running)")
        else:
            click.echo("Specify --ticker <TICKER> or --all", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Health command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--ticker", default=None, help="Show detailed metrics for one session.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON for scripting.")
def health(ticker, as_json):
    """Show health status and metrics for all sessions."""
    try:
        from src.monitoring.health_dashboard import HealthDashboard

        dashboard = HealthDashboard()

        if ticker:
            detail = dashboard.get_session_detail(ticker)
            if detail:
                click.echo(json_mod.dumps(detail, indent=2))
            else:
                click.echo(f"No metrics found for {ticker}")
        elif as_json:
            summary = dashboard.get_health_summary()
            click.echo(json_mod.dumps(summary, indent=2))
        else:
            summary = dashboard.get_health_summary()
            click.echo(dashboard.format_table(summary))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Purge command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--category",
    default=None,
    help="Purge a single category (raw_data, processed_data, performance_metrics, "
         "schema_drift, checkpoints, heartbeat). Omit to purge all.",
)
@click.option(
    "--retention-days",
    default=None,
    type=int,
    help="Override retention days for this run.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Preview deletions without removing files (default: --dry-run).",
)
def purge(config_dir, category, retention_days, dry_run):
    """Purge old data files based on retention policy."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.utils.purge_manager import PurgeManager

        pm = PurgeManager(config)

        if category:
            result = pm.purge_category(
                category,
                retention_days_override=retention_days,
                dry_run=dry_run,
            )
            click.echo(f"\n--- Purge Summary ({category}) ---")
            click.echo(f"Mode:           {'DRY RUN' if dry_run else 'LIVE'}")
            click.echo(f"Retention:      {result['retention_days']} days")
            click.echo(f"Files scanned:  {result['files_scanned']}")
            click.echo(f"Files purged:   {result['files_purged']}")
            click.echo(f"Bytes freed:    {result['bytes_freed']}")
            click.echo(f"Files failed:   {result['files_failed']}")
        else:
            result = pm.purge_all(dry_run=dry_run)
            click.echo(f"\n--- Purge Summary (all categories) ---")
            click.echo(f"Mode:           {'DRY RUN' if dry_run else 'LIVE'}")
            click.echo(f"Files scanned:  {result['files_scanned']}")
            click.echo(f"Files purged:   {result['files_purged']}")
            click.echo(f"Bytes freed:    {result['bytes_freed']}")
            click.echo(f"Files failed:   {result['files_failed']}")
            click.echo("")
            for cat, cat_result in result["categories"].items():
                if cat_result.get("skipped"):
                    click.echo(f"  {cat:<22} (disabled)")
                else:
                    click.echo(
                        f"  {cat:<22} {cat_result['files_purged']:>4} purged "
                        f"/ {cat_result['files_scanned']:>4} scanned "
                        f"({cat_result['retention_days']}d retention)"
                    )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline v2 commands
# ---------------------------------------------------------------------------


@cli.command("download-minute")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--ticker",
    default="SPY",
    help="Ticker to download (e.g. SPY, I:VIX).",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip dates where output file already exists (default: --resume).",
)
def download_minute(config_dir, ticker, start_date, end_date, resume):
    """Download minute-level OHLCV bars for an equity or index ticker."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.minute_downloader import MinuteDownloader
        from src.utils.connection_manager import ConnectionManager
        from src.utils.hardware_monitor import HardwareMonitor

        cm = ConnectionManager(config)
        monitor = HardwareMonitor(config)
        monitor.start("download-minute")

        try:
            dl = MinuteDownloader(config, cm)
            stats = dl.download(ticker, start_date, end_date, resume=resume)
        finally:
            monitor.stop()

        click.echo(f"\n--- Download Minute Summary ({ticker}) ---")
        click.echo(f"Date range:       {start_date} → {end_date}")
        click.echo(f"Dates downloaded: {stats['dates_downloaded']}")
        click.echo(f"Dates skipped:    {stats['dates_skipped']}")
        click.echo(f"Total bars:       {stats['total_bars']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("download-options-targeted")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip dates/contracts where output file already exists (default: --resume).",
)
def download_options_targeted(config_dir, start_date, end_date, resume):
    """Download targeted options (2 calls + 2 puts per day) and their minute bars."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.targeted_options_downloader import TargetedOptionsDownloader
        from src.utils.connection_manager import ConnectionManager
        from src.utils.hardware_monitor import HardwareMonitor

        cm = ConnectionManager(config)
        monitor = HardwareMonitor(config)
        monitor.start("download-options-targeted")

        try:
            dl = TargetedOptionsDownloader(config, cm)
            stats = dl.run(start_date, end_date, resume=resume)
        finally:
            monitor.stop()

        click.echo(f"\n--- Targeted Options Download Summary ---")
        click.echo(f"Date range:        {start_date} → {end_date}")
        click.echo(f"Dates processed:   {stats['dates_processed']}")
        click.echo(f"Dates skipped:     {stats['dates_skipped']}")
        click.echo(f"Contracts found:   {stats['contracts_found']}")
        click.echo(f"Total bars:        {stats['total_bars']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("download-massive-options")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["test", "prod"]),
    help="Override contract_selector.mode from config (test or prod).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip contracts whose Parquet file already exists (default: --resume).",
)
def download_massive_options(config_dir, start_date, end_date, mode, resume):
    """Download options minute bars via Massive.com list_aggs().

    TEST mode (free tier): asks questions once per run (underlying, increment,
    n_calls, n_puts, expiry convention) then constructs tickers mathematically.

    PROD mode (paid tier): calls list_options_contracts() to discover contracts,
    filters to nearest n_calls calls + n_puts puts, then downloads bars.

    Requires underlying minute Parquet files to already exist for each date
    (produced by download-minute --ticker SPY).
    """
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.data_sources.contract_selector import ContractSelector
        from src.data_sources.massive_options_downloader import MassiveOptionsDownloader
        from src.utils.hardware_monitor import HardwareMonitor

        # Resolve selector mode: CLI flag > config > default "test"
        selector_mode = mode or (
            config.get("pipeline_v2", {})
                  .get("contract_selector", {})
                  .get("mode", "test")
        )

        # API key resolution mirrors MassiveOptionsDownloader.from_config()
        import os
        api_key = (
            os.getenv("MASSIVE_API_KEY")
            or config.get("massive", {}).get("api_key", "")
            or os.getenv("POLYGON_API_KEY")
            or config.get("polygon", {}).get("api_key", "")
        )

        selector = ContractSelector(config, mode=selector_mode, api_key=api_key or None)
        downloader = MassiveOptionsDownloader.from_config(config, selector)

        monitor = HardwareMonitor(config)
        monitor.start("download-massive-options")

        try:
            stats = downloader.run(start_date, end_date, resume=resume)
        finally:
            monitor.stop()

        click.echo(f"\n--- Massive Options Download Summary ---")
        click.echo(f"Mode:              {selector_mode}")
        click.echo(f"Date range:        {start_date} → {end_date}")
        click.echo(f"Dates processed:   {stats['dates_processed']}")
        click.echo(f"Dates skipped:     {stats['dates_skipped']}")
        click.echo(f"Contracts found:   {stats['contracts_found']}")
        click.echo(f"Total bars:        {stats['total_bars']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("engineer-features")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--source",
    default="all",
    type=click.Choice(["spy", "vix", "options", "all"]),
    help="Which source to engineer features for (default: all).",
)
def engineer_features(config_dir, start_date, end_date, source):
    """Compute lagged % change features for minute-level data."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.processing.feature_engineer import FeatureEngineer
        from src.utils.hardware_monitor import HardwareMonitor

        monitor = HardwareMonitor(config)
        monitor.start("engineer-features")

        try:
            fe = FeatureEngineer(config)
            stats = fe.run(start_date, end_date, source=source)
        finally:
            monitor.stop()

        click.echo(f"\n--- Feature Engineering Summary ---")
        click.echo(f"Date range:       {start_date} → {end_date}")
        click.echo(f"Source:           {source}")
        click.echo(f"Dates processed:  {stats['dates_processed']}")
        click.echo(f"Equity files:     {stats['equity_files']}")
        click.echo(f"Options files:    {stats['options_files']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("scan-options")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
def scan_options(config_dir, start_date, end_date):
    """Backtest scan: detect completed 20%+ moves using backward-looking logic."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.processing.options_backtest_scanner import OptionsBacktestScanner

        scanner = OptionsBacktestScanner(config)
        events = scanner.scan(start_date, end_date)
        path = scanner.generate_report(events, start_date, end_date)

        click.echo(f"\nReport saved to: {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("scan-options-forward")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--start-date",
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    help="End date (YYYY-MM-DD).",
)
def scan_options_forward(config_dir, start_date, end_date):
    """Forward scan: find entry points where price rose 20%+ in the next 120 min."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.processing.options_forward_scanner import OptionsForwardScanner

        scanner = OptionsForwardScanner(config)
        events = scanner.scan(start_date, end_date)
        path = scanner.generate_report(events, start_date, end_date)

        click.echo(f"\nReport saved to: {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("report-space")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
def report_space(config_dir):
    """Generate storage space utilization report."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.utils.space_reporter import SpaceReporter

        reporter = SpaceReporter(config)
        path = reporter.generate_report()

        click.echo(f"\nSpace report saved to: {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("report-hardware")
@click.option(
    "--config-dir",
    default="config",
    help="Path to config directory containing YAML files.",
)
@click.option(
    "--date",
    default=None,
    help="Date to summarize (YYYY-MM-DD). Defaults to today.",
)
def report_hardware(config_dir, date):
    """Show hardware usage summary for a given date."""
    try:
        loader = ConfigLoader(config_dir=config_dir)
        config = loader.load()

        setup_logger(config)

        from src.utils.hardware_monitor import HardwareMonitor

        monitor = HardwareMonitor(config)
        df = monitor.daily_summary(date)

        if df.empty:
            click.echo(f"No hardware metrics found for {date or 'today'}.")
            return

        click.echo(f"\n--- Hardware Usage Summary ({date or 'today'}) ---")
        click.echo(df.to_string(index=False))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("dashboard")
@click.option(
    "--port",
    default=8501,
    type=int,
    help="Port for the Streamlit dashboard (default: 8501).",
)
def dashboard(port):
    """Launch the Streamlit analytics dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent / "reporting" / "dashboard.py"
    if not dashboard_path.exists():
        click.echo(f"Dashboard not found: {dashboard_path}", err=True)
        sys.exit(1)

    click.echo(f"Starting dashboard on http://localhost:{port} ...")
    try:
        subprocess.run(
            ["streamlit", "run", str(dashboard_path), "--server.port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")
    except FileNotFoundError:
        click.echo(
            "streamlit not found. Install with: pip install streamlit>=1.30.0",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
