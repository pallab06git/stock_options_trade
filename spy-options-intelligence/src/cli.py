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
    python -m src.cli workers list
    python -m src.cli workers stop --ticker SPY
    python -m src.cli workers stop --all
    python -m src.cli health
    python -m src.cli health --ticker SPY
    python -m src.cli health --json
"""

import json as json_mod
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


if __name__ == "__main__":
    cli()
