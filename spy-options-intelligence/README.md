# SPY Options Intelligence Platform

Multi-source data ingestion platform for ML-based options trading research. Collects SPY equity, options, VIX, and news data from Polygon.io with dual-mode execution (historical backfill + real-time streaming), consolidates into an enriched per-option-per-minute dataset with Greeks, technical indicators, and sentiment.

---

## Features

- **4 Data Sources**: SPY per-second aggregates, options contracts (±1% strike range), VIX index, news with sentiment
- **Dual-Mode Execution**: Historical REST backfill and real-time WebSocket streaming
- **Data Consolidation**: Per-option-per-minute flat schema with Greeks (delta, gamma, theta, vega, rho, IV), RSI, MACD, Bollinger Bands, momentum, and news sentiment
- **ML Training Prep**: Forward-looking target prices with configurable prediction windows
- **Feed Simulator**: Replay historical data as simulated real-time streams at configurable speed
- **Monitoring**: Performance metrics, schema drift detection, heartbeat monitoring, health dashboard
- **Parallel Execution**: Multi-ticker backfill with subprocess isolation
- **Checkpoint/Resume**: JSON-based checkpoints for interrupted fetches
- **Config-Driven**: YAML configuration with environment variable substitution

---

## Installation

### Prerequisites

- Python 3.10+
- Polygon.io API key ([free tier available](https://polygon.io))

### Setup

```bash
cd spy-options-intelligence/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY
```

---

## Quick Start

```bash
# 1. Backfill SPY historical data
python -m src.cli backfill --start-date 2025-01-02 --end-date 2025-01-31

# 2. Backfill VIX data
python -m src.cli backfill-vix --start-date 2025-01-02 --end-date 2025-01-31

# 3. Discover options contracts for a trading day
python -m src.cli discover --date 2025-01-15

# 4. Consolidate all sources into enriched dataset
python -m src.cli consolidate --date 2025-01-15
```

---

## CLI Reference

### Data Ingestion

| Command | Description | Example |
|---------|-------------|---------|
| `backfill` | Historical equity data backfill | `python -m src.cli backfill --ticker SPY --start-date 2025-01-02 --end-date 2025-01-31 --resume` |
| `backfill-vix` | Historical VIX data backfill | `python -m src.cli backfill-vix --start-date 2025-01-02 --end-date 2025-01-31` |
| `backfill-news` | Historical news data backfill | `python -m src.cli backfill-news --start-date 2025-01-02 --end-date 2025-01-31` |
| `backfill-all` | Parallel backfill for all configured tickers | `python -m src.cli backfill-all --start-date 2025-01-02 --end-date 2025-01-31` |
| `discover` | Discover options contracts within strike range | `python -m src.cli discover --date 2025-01-15` |

### Real-Time Streaming

| Command | Description | Example |
|---------|-------------|---------|
| `stream` | Stream real-time equity data via WebSocket | `python -m src.cli stream --ticker SPY` |
| `stream-vix` | Stream real-time VIX data via WebSocket | `python -m src.cli stream-vix` |
| `stream-options` | Stream real-time options data via WebSocket | `python -m src.cli stream-options --date 2025-01-15` |
| `stream-news` | Stream news articles via polling | `python -m src.cli stream-news` |

### Processing

| Command | Description | Example |
|---------|-------------|---------|
| `consolidate` | Consolidate all sources for a trading day | `python -m src.cli consolidate --date 2025-01-15` |
| `prepare-training` | Generate ML training data with target prices | `python -m src.cli prepare-training --start-date 2025-01-02 --end-date 2025-01-31` |
| `simulate` | Replay historical data as simulated real-time | `python -m src.cli simulate --source spy --date 2025-01-15 --speed 10` |

### Operations

| Command | Description | Example |
|---------|-------------|---------|
| `health` | Show health status for all sessions | `python -m src.cli health --json` |
| `workers list` | Show all worker processes | `python -m src.cli workers list` |
| `workers stop` | Stop one or all workers | `python -m src.cli workers stop --ticker SPY` |
| `schema-check` | Check for schema drift against baseline | `python -m src.cli schema-check --source spy --date 2025-01-15` |
| `schema-baseline` | Capture schema baseline from a Parquet file | `python -m src.cli schema-baseline --source spy --date 2025-01-15` |

All commands accept `--config-dir` to override the config directory (default: `config`).

---

## Data Schemas

### SPY Equity Aggregates

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int64 | Unix timestamp (ms) |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | int64 | Trading volume |
| `vwap` | float64 | Volume-weighted average price |
| `ticker` | string | Ticker symbol (e.g. "SPY") |
| `date` | string | YYYY-MM-DD partition key |

Storage: `data/raw/spy/{date}.parquet`

### Options Aggregates

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int64 | Unix timestamp (ms) |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | int64 | Trading volume |
| `vwap` | float64 | Volume-weighted average price |
| `ticker` | string | Options contract ticker |
| `date` | string | YYYY-MM-DD partition key |

Storage: `data/raw/options/{date}.parquet`

### VIX Aggregates

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int64 | Unix timestamp (ms) |
| `open` | float64 | VIX open |
| `high` | float64 | VIX high |
| `low` | float64 | VIX low |
| `close` | float64 | VIX close |
| `ticker` | string | "I:VIX" |
| `source` | string | "vix" |
| `date` | string | YYYY-MM-DD partition key |

Storage: `data/raw/vix/{date}.parquet`

### News Articles

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int64 | Unix timestamp (ms) of published_utc |
| `article_id` | string | Unique article identifier |
| `title` | string | Article headline |
| `description` | string | Article body/summary |
| `sentiment` | float64 | Sentiment score (-1.0 to 1.0) |
| `keywords` | string | Comma-separated keywords |
| `tickers` | string | Related ticker symbols |
| `source` | string | "news" |
| `date` | string | YYYY-MM-DD partition key |

Storage: `data/raw/news/{date}.parquet`

### Consolidated (Per-Option-Per-Minute)

One row per option contract per minute. Includes:

| Group | Columns |
|-------|---------|
| **Time** | `minute_ts` |
| **SPY** | `spy_open`, `spy_high`, `spy_low`, `spy_close`, `spy_volume`, `spy_vwap` |
| **VIX** | `vix_open`, `vix_high`, `vix_low`, `vix_close` |
| **Option** | `option_ticker`, `option_open`, `option_high`, `option_low`, `option_close`, `option_volume`, `option_vwap_avg` |
| **Indicators** | `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_middle`, `bb_lower` |
| **Momentum** | `price_change`, `roc_5`, `roc_30`, `roc_60` |
| **Greeks** | `delta`, `gamma`, `theta`, `vega`, `rho`, `implied_volatility` |
| **Sentiment** | `news_sentiment`, `news_title` |

Storage: `data/processed/consolidated/{date}.parquet`

---

## Project Structure

```
spy-options-intelligence/
├── src/
│   ├── cli.py                         # 16 CLI commands (click)
│   ├── data_sources/
│   │   ├── base_source.py             # Abstract source interface
│   │   ├── polygon_client.py          # SPY equity (REST + WebSocket)
│   │   ├── polygon_options_client.py  # Options discovery + streaming
│   │   ├── polygon_vix_client.py      # VIX index (REST + WebSocket)
│   │   └── news_client.py             # News articles (REST + polling)
│   ├── sinks/
│   │   ├── base_sink.py               # Abstract sink interface
│   │   └── parquet_sink.py            # Date-partitioned Parquet
│   ├── orchestrator/
│   │   ├── historical_runner.py       # Batch backfill pipeline
│   │   ├── streaming_runner.py        # Real-time streaming pipeline
│   │   ├── options_streaming_runner.py# Options-specific streaming
│   │   ├── simulator.py               # Feed replay from historical data
│   │   ├── parallel_runner.py         # Multi-ticker subprocess runner
│   │   ├── process_manager.py         # Worker start/stop/list
│   │   └── task_manager.py            # Task tracking
│   ├── processing/
│   │   ├── validator.py               # Per-source schema validation
│   │   ├── deduplicator.py            # In-memory deduplication
│   │   ├── consolidator.py            # Multi-source join + enrichment
│   │   └── training_data_prep.py      # ML training data generation
│   ├── monitoring/
│   │   ├── performance_monitor.py     # Latency, throughput, memory alerts
│   │   ├── schema_monitor.py          # Schema drift detection
│   │   ├── error_aggregator.py        # Error rate tracking
│   │   ├── health_dashboard.py        # Unified health view
│   │   └── heartbeat_monitor.py       # Stream liveness monitoring
│   └── utils/
│       ├── config_loader.py           # YAML + env var config
│       ├── logger.py                  # Loguru + credential redaction
│       ├── retry_handler.py           # Exponential backoff + jitter
│       ├── connection_manager.py      # Polygon SDK + rate limiting
│       └── market_hours.py            # NYSE calendar + hours check
├── config/
│   ├── settings.yaml                  # Master configuration
│   ├── sources.yaml                   # API keys, endpoints, rate limits
│   ├── sinks.yaml                     # Storage destinations
│   ├── retry_policy.yaml              # Retry profiles (default, polygon)
│   └── examples/                      # Annotated example configs
├── data/
│   ├── raw/                           # Per-source Parquet (spy, options, vix, news)
│   ├── processed/                     # Consolidated + training data
│   └── logs/                          # Execution, error, performance, schema logs
├── docs/
│   ├── API_REFERENCE.md               # Module-by-module class reference
│   └── WORK_LOG.md                    # Implementation history
└── tests/
    ├── unit/                          # Unit tests (519 tests)
    └── integration/                   # Integration tests (24 tests)
```

---

## Configuration

The platform uses 4 YAML files in `config/` plus a `.env` file:

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Streaming hours, monitoring thresholds, consolidation params, simulator settings |
| `config/sources.yaml` | Polygon API key (via `${POLYGON_API_KEY}`), rate limits, per-ticker/source config |
| `config/sinks.yaml` | Parquet base path, compression, row group size |
| `config/retry_policy.yaml` | Retry profiles — max attempts, backoff, jitter, status codes |
| `.env` | Secrets (`POLYGON_API_KEY`) — git-ignored |

See `config/examples/` for annotated example configurations.

---

## Monitoring

### Health Dashboard

```bash
# Text table view
python -m src.cli health

# JSON output for scripting
python -m src.cli health --json

# Detailed metrics for one session
python -m src.cli health --ticker SPY
```

### Schema Drift Detection

```bash
# Capture baseline from known-good data
python -m src.cli schema-baseline --source spy --date 2025-01-15

# Check for drift against baseline
python -m src.cli schema-check --source spy --date 2025-01-16
```

Configurable alerts for new columns, missing columns, and type changes. Drift events are logged to `data/logs/schema/drift/`.

### Performance Monitoring

- Per-operation latency tracking (p50, p95, p99)
- Throughput monitoring (records/second)
- Memory usage alerts (RSS threshold)
- Stale/hung operation detection
- Error rate sliding window with configurable threshold
- Metrics dumped to `data/logs/performance/` at configurable intervals

### Heartbeat Monitoring

Real-time streams emit heartbeat files every 5 minutes. If no heartbeat for 15 minutes, a stall alert is logged. Heartbeat files are stored in `data/logs/heartbeat/`.

---

## Testing

```bash
# Run all tests
pytest tests/ --tb=short

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/ --tb=short

# Integration tests only
pytest tests/integration/ --tb=short
```

**543 tests passing** (519 unit + 24 integration) + 7 live tests (skipped outside market hours).

---

## Security

- All API keys loaded from `.env` (git-ignored, never committed)
- Credentials automatically redacted in logs (shows only last 4 characters)
- No hardcoded secrets in source code
- Error messages sanitized before logging
- See `SECURITY.md` for complete guidelines

---

## License

Copyright 2026 Pallab Basu Roy. All rights reserved. Proprietary and confidential. See `LICENSE` for details.
