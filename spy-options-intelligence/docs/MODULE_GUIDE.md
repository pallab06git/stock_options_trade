# Module Guide — Scripts, Descriptions & Call Hierarchy

Complete reference of every module in the platform, what it does, and how modules depend on each other.

---

## Table of Contents

1. [Module Index](#module-index)
2. [Program Hierarchy — By CLI Command](#program-hierarchy--by-cli-command)
3. [Dependency Graph](#dependency-graph)
4. [Architectural Patterns](#architectural-patterns)

---

## Module Index

### Entry Point

| Module | Description |
|--------|-------------|
| `src/cli.py` | Click-based CLI with 16 commands. The only user-facing entry point — all execution flows start here. |

### Data Sources (`src/data_sources/`)

| Module | Class | Description |
|--------|-------|-------------|
| `base_source.py` | `BaseSource` | Abstract interface that all data source clients implement. Defines `connect()`, `disconnect()`, `fetch_historical()`, `stream_realtime()`, `validate_record()`. |
| `polygon_client.py` | `PolygonEquityClient` | Fetches equity (SPY, QQQ, etc.) per-minute aggregates via Polygon REST `get_aggs()` for historical, and Polygon WebSocket (`Feed.Delayed`, `Market.Stocks`) for real-time. Handles pagination, record transformation, and rate-limited API calls. |
| `polygon_options_client.py` | `PolygonOptionsClient` | Discovers options contracts within ±1% of SPY opening price via `list_options_contracts()`, persists them as JSON, and streams real-time options aggregates via WebSocket (`Market.Options`). |
| `polygon_vix_client.py` | `PolygonVIXClient` | Fetches VIX index (`I:VIX`) aggregates — historical via REST `get_aggs()` and real-time via WebSocket (`Market.Indices`). |
| `news_client.py` | `PolygonNewsClient` | Fetches news articles via REST `list_ticker_news()` for historical, and polls at a configurable interval for "real-time". Extracts ticker-matched sentiment from Polygon insights. |

### Sinks (`src/sinks/`)

| Module | Class | Description |
|--------|-------|-------------|
| `base_sink.py` | `BaseSink` | Abstract interface for storage backends. Defines `write_batch()`, `write_single()`, `check_duplicate()`, `overwrite()`. |
| `parquet_sink.py` | `ParquetSink` | Writes records to date-partitioned Parquet files (`data/raw/{source}/{date}.parquet`) with Snappy compression. Supports compound deduplication keys for options (ticker + timestamp). |

### Orchestration (`src/orchestrator/`)

| Module | Class | Description |
|--------|-------|-------------|
| `historical_runner.py` | `HistoricalRunner` | Batch pipeline: iterates dates, fetches data from a source client, validates, deduplicates, writes to Parquet. Supports checkpoint/resume via JSON files. Generic — works with any `BaseSource` via dependency injection. |
| `streaming_runner.py` | `StreamingRunner` | Real-time pipeline: consumes a WebSocket generator, buffers records, flushes batches through validator → deduplicator → Parquet. Enforces market hours, runs heartbeat monitoring, handles graceful shutdown via SIGTERM/SIGINT. Generic — works with any `BaseSource` via dependency injection. |
| `options_streaming_runner.py` | `OptionsStreamingRunner` | Specialized streaming runner for options. Loads discovered contracts, streams via `Market.Options` WebSocket, applies compound deduplication (ticker + timestamp). |
| `simulator.py` | `FeedSimulator` | Reads a historical Parquet file and replays records as a simulated real-time stream with configurable playback speed. Implements the `BaseSource` interface so it can be injected into `StreamingRunner` for offline testing. |
| `parallel_runner.py` | `ParallelRunner` | Spawns one subprocess per ticker for parallel backfill. Each subprocess runs `python -m src.cli backfill --ticker <T>`. Maintains a process registry JSON for tracking PIDs. |
| `process_manager.py` | `ProcessManager` | Reads the process registry to list/stop worker processes. Sends SIGTERM to running workers. |
| `task_manager.py` | `TaskManager` | Internal task tracking (not directly exposed via CLI). |

### Processing (`src/processing/`)

| Module | Class | Description |
|--------|-------|-------------|
| `validator.py` | `RecordValidator` | Validates records against per-source schemas (equity, vix, options, news). Splits batches into valid/invalid lists. |
| `deduplicator.py` | `Deduplicator` | Tracks seen keys in an in-memory set to filter duplicate records. Configurable key field (`timestamp` for equity/VIX, `article_id` for news). |
| `consolidator.py` | `Consolidator` | The heaviest processing module. Loads SPY, VIX, Options, and News Parquet files for a date, aggregates to 1-minute bars, aligns VIX via `merge_asof`, computes technical indicators (RSI, MACD, Bollinger Bands), momentum (ROC), attaches news sentiment, flattens to per-option-per-minute rows, and computes Black-Scholes Greeks. |
| `training_data_prep.py` | `TrainingDataPrep` | Reads consolidated Parquet files, adds forward-looking target future prices (120-min lookahead), filters by minimum coverage, and writes to `data/processed/training/`. |

### Monitoring (`src/monitoring/`)

| Module | Class | Description |
|--------|-------|-------------|
| `performance_monitor.py` | `PerformanceMonitor` | Tracks per-operation latency (p50/p95/p99), throughput (records/sec), and memory usage. Alerts when thresholds are breached. Dumps metrics JSON to `data/logs/performance/`. |
| `error_aggregator.py` | `ErrorAggregator` | Counts errors and successes over a sliding window. Alerts when error rate exceeds configured threshold (default 1%). |
| `schema_monitor.py` | `SchemaMonitor` | Reads Parquet file metadata (no data loading) to extract schema, compares against stored baselines, and alerts on new/missing columns or type changes. |
| `health_dashboard.py` | `HealthDashboard` | Aggregates process registry + per-session metric files into a unified health view. Formats as text table or JSON. |
| `heartbeat_monitor.py` | `HeartbeatMonitor` | During streaming, writes a heartbeat JSON file every 5 minutes with message counts and connection status. Alerts if no heartbeat for 15 minutes (stalled stream). |

### Utilities (`src/utils/`)

| Module | Class/Function | Description |
|--------|----------------|-------------|
| `config_loader.py` | `ConfigLoader` | Loads `.env`, parses 4 YAML files (`settings.yaml`, `sources.yaml`, `sinks.yaml`, `retry_policy.yaml`), substitutes `${ENV_VARS}`, merges into a single config dict. |
| `logger.py` | `setup_logger()`, `get_logger()`, `redact_sensitive()` | Configures loguru with console + file handlers. Automatically redacts API keys, passwords, and connection strings in all log output. |
| `retry_handler.py` | `RetryHandler` | Wraps API calls with configurable exponential backoff + jitter. Profiles loaded from `retry_policy.yaml` (default: 3 attempts, polygon: 5 attempts). |
| `connection_manager.py` | `ConnectionManager` | Manages Polygon SDK clients (REST + WebSocket). Implements unified token-bucket rate limiting across all sources. Handles 429 responses by pausing the bucket. |
| `market_hours.py` | `MarketHours` | Checks if NYSE is open (Mon–Fri, 9:30 AM–4:00 PM ET). Accounts for holidays via Polygon market calendar API. Used by streaming runners to start/stop streams. |

---

## Program Hierarchy — By CLI Command

Every command starts with `python -m src.cli <command>`. Below is the exact call chain for each.

### 1. `backfill` — Historical Equity Backfill

```
cli.py → backfill()
├── ConfigLoader.load()
├── setup_logger()
└── HistoricalRunner(config, ticker)
    ├── creates ConnectionManager
    │   └── creates RESTClient (lazy), TokenBucket rate limiter
    ├── creates PolygonEquityClient(config, conn_mgr, ticker)
    ├── creates RecordValidator.for_equity(ticker)
    ├── creates Deduplicator(key_field="timestamp")
    └── creates ParquetSink(config)

    runner.run(resume):
    ├── client.connect()
    ├── sink.connect()
    ├── load checkpoint (JSON)
    └── FOR each date in range:
        ├── client.fetch_historical(date, date)  ← REST API call
        │   ├── conn_mgr.acquire_rate_limit()
        │   └── rest_client.get_aggs()  [Polygon SDK]
        ├── validator.validate_batch(buffer)
        ├── deduplicator.deduplicate_batch(valid)
        ├── sink.write_batch(deduped, date)  ← writes Parquet
        └── save checkpoint (JSON)
```

### 2. `backfill-vix` — Historical VIX Backfill

```
cli.py → backfill_vix()
├── ConfigLoader.load()
├── setup_logger()
├── ConnectionManager(config)
├── PolygonVIXClient(config, conn_mgr)        ← VIX-specific client
├── RecordValidator("vix")                     ← VIX schema
└── HistoricalRunner(config, "I:VIX", conn_mgr, vix_client, validator)

    runner.run(resume):
    └── [same flow as backfill, using VIX client]
```

### 3. `backfill-news` — Historical News Backfill

```
cli.py → backfill_news()
├── ConfigLoader.load()
├── setup_logger()
├── ConnectionManager(config)
├── PolygonNewsClient(config, conn_mgr)        ← News-specific client
├── RecordValidator("news")                    ← News schema
├── Deduplicator(key_field="article_id")       ← Dedup by article_id
└── HistoricalRunner(config, "news", conn_mgr, news_client, validator, deduplicator)

    runner.run(resume):
    └── [same flow as backfill, using news client + article_id dedup]
```

### 4. `backfill-all` — Parallel Multi-Ticker Backfill

```
cli.py → backfill_all()
├── ConfigLoader.load()
├── setup_logger()
└── ParallelRunner(config)
    └── reads tickers from config["orchestrator"]["tickers"]

    runner.run(start_date, end_date, resume, config_dir):
    └── FOR each ticker:
        ├── subprocess.Popen(["python", "-m", "src.cli", "backfill", "--ticker", ticker, ...])
        │   └── [each subprocess runs a full backfill command independently]
        └── registers PID in data/logs/process_registry.json
    └── waits for all subprocesses, collects exit codes
```

### 5. `stream` — Real-Time Equity Streaming

```
cli.py → stream()
├── ConfigLoader.load()
├── setup_logger()
└── StreamingRunner(config, ticker)
    ├── creates ConnectionManager
    ├── creates PolygonEquityClient(config, conn_mgr, ticker)
    ├── creates RecordValidator.for_equity(ticker)
    ├── creates Deduplicator(key_field="timestamp")
    ├── creates ParquetSink(config)
    ├── creates MarketHours(config, api_key)
    ├── creates HeartbeatMonitor(config, session_label)
    ├── creates PerformanceMonitor(config, session_label)
    └── creates ErrorAggregator(config, session_label)

    runner.run():
    ├── market_hours.is_market_open()  ← gate check
    ├── register SIGTERM/SIGINT handlers
    ├── sink.connect()
    └── LOOP over client.stream_realtime(stop_event):
        │   ← WebSocket: conn_mgr.get_ws_client() → ws.subscribe("A.{ticker}")
        ├── heartbeat.record_message()
        ├── IF heartbeat.should_send_heartbeat():
        │   └── heartbeat.send_heartbeat()  ← writes JSON
        ├── IF heartbeat.check_stalled_stream():
        │   └── log alert
        ├── buffer.append(record)
        └── IF buffer full:
            └── flush_buffer():
                ├── perf_monitor.start_operation("flush")
                ├── validator.validate_batch(buffer)
                ├── error_aggregator.record_error() / record_success()
                ├── deduplicator.deduplicate_batch(valid)
                ├── sink.write_batch(deduped, date)
                └── perf_monitor.end_operation("flush", count)
```

### 6. `stream-vix` — Real-Time VIX Streaming

```
cli.py → stream_vix()
├── ConfigLoader.load()
├── setup_logger()
├── ConnectionManager(config)
├── PolygonVIXClient(config, conn_mgr)        ← VIX client
├── RecordValidator("vix")
└── StreamingRunner(config, "I:VIX", conn_mgr, vix_client, validator)

    runner.run():
    └── [same flow as stream, using VIX WebSocket (Market.Indices)]
```

### 7. `stream-news` — Real-Time News Polling

```
cli.py → stream_news()
├── ConfigLoader.load()
├── setup_logger()
├── ConnectionManager(config)
├── PolygonNewsClient(config, conn_mgr)
├── RecordValidator("news")
├── Deduplicator(key_field="article_id")
└── StreamingRunner(config, "news", conn_mgr, news_client, validator, deduplicator)

    runner.run():
    └── [same flow as stream, but client.stream_realtime() polls REST instead of WebSocket]
```

### 8. `stream-options` — Real-Time Options Streaming

```
cli.py → stream_options()
├── ConfigLoader.load()
├── setup_logger()
└── OptionsStreamingRunner(config, date)
    ├── creates ConnectionManager
    ├── creates PolygonOptionsClient(config, conn_mgr)
    │   └── load_contracts(date)  ← reads contracts JSON from discover step
    ├── creates RecordValidator("options")
    ├── creates Deduplicator(key_field="dedup_key")  ← compound: ticker_timestamp
    ├── creates ParquetSink(config, dedup_subset=["ticker", "timestamp"])
    ├── creates MarketHours, HeartbeatMonitor, PerformanceMonitor, ErrorAggregator

    runner.run():
    └── [same flow as stream, using Market.Options WebSocket]
        └── flush adds temporary dedup_key = f"{ticker}_{timestamp}", removes before write
```

### 9. `discover` — Options Contract Discovery

```
cli.py → discover()
├── ConfigLoader.load()
├── setup_logger()
├── ConnectionManager(config)
└── PolygonOptionsClient(config, conn_mgr)

    client.fetch_opening_price(date):
    ├── conn_mgr.acquire_rate_limit()
    └── rest_client.get_daily_open_close(ticker, date)

    client.discover_contracts(date, opening_price):
    ├── compute strike range: [price * 0.99, price * 1.01]
    ├── conn_mgr.acquire_rate_limit()
    ├── rest_client.list_options_contracts(...)
    └── filter + limit to max_contracts

    client.save_contracts(contracts, date):
    └── writes data/raw/options/contracts/{date}_contracts.json
```

### 10. `consolidate` — Multi-Source Data Consolidation

```
cli.py → consolidate()
├── ConfigLoader.load()
├── setup_logger()
└── Consolidator(config)

    consolidator.consolidate(date):
    ├── load_spy(date)       ← pd.read_parquet(data/raw/spy/{date}.parquet)
    ├── load_vix(date)       ← pd.read_parquet(data/raw/vix/{date}.parquet)
    ├── load_options(date)   ← pd.read_parquet(data/raw/options/{date}.parquet)
    ├── load_contracts(date) ← json.load(contracts JSON)
    ├── load_news(date)      ← pd.read_parquet(data/raw/news/{date}.parquet)
    │
    ├── aggregate_spy_per_minute()      ← groupby minute → OHLCV
    ├── aggregate_vix_per_minute()      ← groupby minute → OHLC
    ├── aggregate_options_per_minute()  ← groupby (ticker, minute) → OHLCV
    │
    ├── align_vix(spy_1m, vix_1m)      ← pd.merge_asof(direction="backward")
    │
    ├── compute_indicators(df)          ← ta library
    │   ├── RSIIndicator(close, period=14)
    │   ├── MACD(close, fast=12, slow=26, signal=9)
    │   └── BollingerBands(close, period=20, std_dev=2.0)
    │
    ├── compute_momentum(df)
    │   └── ROC for windows [5, 30, 60]
    │
    ├── attach_news(df, news)           ← pd.merge_asof(tolerance=24h)
    │
    ├── flatten_to_per_option(spy_enriched, options_1m, contracts)
    │   └── inner join options × SPY on minute_ts
    │
    ├── compute_greeks_flat(df)         ← py_vollib
    │   ├── delta(), gamma(), theta(), vega(), rho()
    │   └── implied_volatility()
    │
    └── write output → data/processed/consolidated/{date}.parquet
```

### 11. `prepare-training` — ML Training Data Generation

```
cli.py → prepare_training()
├── ConfigLoader.load()
├── setup_logger()
└── TrainingDataPrep(config)

    prep.prepare(dates):
    └── FOR each date:
        ├── pd.read_parquet(data/processed/consolidated/{date}.parquet)
        ├── compute_target_future_prices(df)
        │   └── for each row at time T, look up price at T+1m, T+2m, ..., T+120m
        ├── filter_by_target_coverage(df)
        │   └── drop rows with <50% non-NaN targets
        └── write → data/processed/training/{date}.parquet
```

### 12. `simulate` — Feed Replay

```
cli.py → simulate()
├── ConfigLoader.load()
├── setup_logger()
└── FeedSimulator(config, source, date, speed)
    └── resolves parquet_path from source name

    LOOP over sim.stream_realtime():
    ├── load_records()
    │   └── pd.read_parquet(path) → sort by timestamp → to_dict()
    └── FOR each record:
        ├── compute delay = (ts_gap / 1000) / speed
        ├── time.sleep(delay)
        └── yield record
```

### 13. `schema-check` — Schema Drift Detection

```
cli.py → schema_check()
├── ConfigLoader.load()
├── setup_logger()
└── SchemaMonitor(config)

    monitor.check_drift(source, parquet_path):
    ├── load_baseline(source) ← data/logs/schema/{source}_baseline.json
    ├── capture schema from file ← pyarrow.parquet.read_schema() [metadata only]
    ├── detect_schema_changes(baseline, current)
    │   └── set diff: new columns, missing columns, type changes
    ├── format_alerts(source, changes)
    ├── log_drift() ← data/logs/schema/drift/{source}_{date}_{ts}.json
    └── if auto_update_baseline: save_baseline()
```

### 14. `schema-baseline` — Capture Schema Baseline

```
cli.py → schema_baseline()
├── ConfigLoader.load()
├── setup_logger()
└── SchemaMonitor(config)

    monitor.capture_baseline(source, parquet_path):
    └── pyarrow.parquet.read_schema() → {column: dtype} map

    monitor.save_baseline(source, baseline):
    └── writes data/logs/schema/{source}_baseline.json
```

### 15. `workers list` / `workers stop` — Process Management

```
cli.py → workers list/stop
└── ProcessManager({})

    list_workers():
    ├── ParallelRunner.load_registry() ← data/logs/process_registry.json
    └── FOR each ticker: psutil.pid_exists(pid) → alive/dead

    stop_worker(ticker):
    ├── ParallelRunner.load_registry()
    └── os.kill(pid, SIGTERM)
```

### 16. `health` — Health Dashboard

```
cli.py → health()
└── HealthDashboard(metrics_dir="data/logs/performance")

    get_health_summary():
    ├── ParallelRunner.load_registry()  ← process PIDs + status
    ├── get_all_sessions()              ← scan metrics_dir for JSON files
    └── merge: process info + latest metrics + psutil.pid_exists()

    format_table(summary):
    └── text table with ticker, PID, status, latency, throughput, errors
```

---

## Dependency Graph

Shows which modules depend on which. Arrows mean "uses / imports from".

```
                              ┌──────────────┐
                              │   cli.py     │  ← user entry point
                              └──────┬───────┘
                                     │
          ┌──────────────────────────┼──────────────────────────────┐
          │                          │                              │
          ▼                          ▼                              ▼
  ┌───────────────┐    ┌──────────────────────┐    ┌───────────────────────┐
  │  ConfigLoader │    │    Orchestrators      │    │  Monitoring (direct)  │
  │  setup_logger │    │                      │    │  SchemaMonitor        │
  └───────────────┘    │  HistoricalRunner    │    │  HealthDashboard      │
                       │  StreamingRunner     │    │  ProcessManager       │
                       │  OptionsStreamingRnr │    └───────────────────────┘
                       │  ParallelRunner      │
                       │  FeedSimulator       │
                       └──────────┬───────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
    │ Data Sources │    │  Processing  │    │    Monitoring     │
    │              │    │              │    │                  │
    │ EquityClient │    │ Validator    │    │ PerformanceMonitor│
    │ OptionsClient│    │ Deduplicator │    │ ErrorAggregator  │
    │ VIXClient    │    │ Consolidator │    │ HeartbeatMonitor │
    │ NewsClient   │    │ TrainingPrep │    └────────┬─────────┘
    └──────┬───────┘    └──────┬───────┘             │
           │                   │                     │
           ▼                   ▼                     │
    ┌──────────────┐    ┌──────────────┐             │
    │    Sinks     │    │  External    │             │
    │              │    │  Libraries   │             │
    │ ParquetSink  │    │  ta, vollib  │             │
    └──────┬───────┘    │  pandas, np  │             │
           │            └──────────────┘             │
           ▼                                         │
    ┌──────────────────────────────────────────────┐ │
    │               Utilities                       │◄┘
    │                                              │
    │  ConnectionManager ← rate limiting, SDK mgmt │
    │  RetryHandler      ← exponential backoff     │
    │  MarketHours       ← NYSE calendar           │
    │  ConfigLoader      ← YAML + .env             │
    │  Logger            ← loguru + redaction       │
    └──────────────────────────────────────────────┘
```

### Typical Execution Order (Full Pipeline)

```
1. backfill          ── SPY historical data       ── writes data/raw/spy/
2. backfill-vix      ── VIX historical data       ── writes data/raw/vix/
3. discover          ── options contracts          ── writes data/raw/options/contracts/
4. backfill-news     ── news articles             ── writes data/raw/news/
5. consolidate       ── merge all sources          ── reads raw/, writes data/processed/consolidated/
6. prepare-training  ── add target prices          ── reads consolidated/, writes data/processed/training/
```

For real-time, the order is:

```
1. discover          ── must run first (options need contract list)
2. stream            ── SPY real-time             ── writes data/raw/spy/
3. stream-vix        ── VIX real-time             ── writes data/raw/vix/     (parallel)
4. stream-options    ── options real-time          ── writes data/raw/options/ (parallel)
5. stream-news       ── news polling              ── writes data/raw/news/    (parallel)
6. consolidate       ── run after market close     ── writes data/processed/consolidated/
7. prepare-training  ── run after consolidation    ── writes data/processed/training/
```

### Data Flow Between Modules

```
Polygon REST API ──► PolygonEquityClient ──► HistoricalRunner ──► ParquetSink
Polygon REST API ──► PolygonVIXClient    ──► HistoricalRunner ──► ParquetSink
Polygon REST API ──► PolygonNewsClient   ──► HistoricalRunner ──► ParquetSink
Polygon REST API ──► PolygonOptionsClient ──► save_contracts() ──► JSON file

Polygon WebSocket ──► PolygonEquityClient ──► StreamingRunner ──────────► ParquetSink
Polygon WebSocket ──► PolygonVIXClient    ──► StreamingRunner ──────────► ParquetSink
Polygon WebSocket ──► PolygonOptionsClient ──► OptionsStreamingRunner ──► ParquetSink
Polygon REST poll ──► PolygonNewsClient   ──► StreamingRunner ──────────► ParquetSink

data/raw/spy/*.parquet ────┐
data/raw/vix/*.parquet ────┤
data/raw/options/*.parquet ┼──► Consolidator ──► data/processed/consolidated/*.parquet
data/raw/news/*.parquet ───┤
contracts JSON ────────────┘

data/processed/consolidated/*.parquet ──► TrainingDataPrep ──► data/processed/training/*.parquet
```

---

## Architectural Patterns

### 1. Dependency Injection in Runners

`HistoricalRunner` and `StreamingRunner` accept optional pre-built components (`client`, `validator`, `deduplicator`, `connection_manager`). This is how VIX and News reuse the same runner logic:

```
HistoricalRunner + PolygonEquityClient → SPY backfill
HistoricalRunner + PolygonVIXClient    → VIX backfill
HistoricalRunner + PolygonNewsClient   → News backfill

StreamingRunner + PolygonEquityClient → SPY streaming
StreamingRunner + PolygonVIXClient    → VIX streaming
StreamingRunner + PolygonNewsClient   → News streaming
```

### 2. Generator-Based Data Flow

All `fetch_historical()` and `stream_realtime()` methods are Python generators. Records flow through the pipeline one batch at a time without loading the entire dataset into memory.

### 3. Shared Rate Limiting

All API calls go through `ConnectionManager.acquire_rate_limit()`, which uses a single token bucket. This prevents any source from starving others when running concurrently.

### 4. Subprocess Isolation for Parallelism

`ParallelRunner` spawns separate Python processes (not threads). Each subprocess has its own memory, connection, and rate limiter. The parent process communicates only through the process registry JSON and exit codes.

### 5. Monitor Triad for Streaming

Every streaming command instantiates three monitors that work together:
- **HeartbeatMonitor** — "is the stream alive?" (5-min heartbeat file)
- **PerformanceMonitor** — "is it performing well?" (latency, throughput, memory)
- **ErrorAggregator** — "is it failing too much?" (sliding-window error rate)
