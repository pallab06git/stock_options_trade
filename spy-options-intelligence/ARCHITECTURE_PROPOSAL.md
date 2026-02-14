# SPY Options Intelligence - Technical Architecture Proposal

**Date**: 2026-02-13
**Status**: APPROVED
**Author**: Claude Code (Architecture Phase)
**Revision**: 6 (Single target variable, signal validator, ML model architecture, storage optimization)

---

## 1. System Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION LAYER                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ settings.yaml│ │ sources.yaml │ │  sinks.yaml  │ │retry_policy  │       │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
│         └────────────────┴────────────────┴────────────────┘               │
│                              │                                              │
│                    ┌─────────▼──────────┐                                   │
│                    │   ConfigLoader     │                                   │
│                    │   (YAML + .env)    │                                   │
│                    └─────────┬──────────┘                                   │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                                     │
│                               │                                             │
│  ┌────────────────────────────▼────────────────────────────┐                │
│  │              Task Manager (Central Coordinator)         │                │
│  │         Tracks runs, checkpoints, resumability          │                │
│  └──┬──────────────────┬──────────────────┬────────────┘   │                │
│     │                  │                  │                                  │
│  ┌──▼───────────┐ ┌────▼──────────┐ ┌────▼──────────┐                      │
│  │  Historical  │ │   Streaming   │ │   Simulator   │                      │
│  │   Runner     │ │    Runner     │ │  (Hist→RT)    │                      │
│  │ (Batch/REST) │ │ (WebSocket)   │ │               │                      │
│  └──┬───────────┘ └────┬──────────┘ └────┬──────────┘                      │
└─────┼──────────────────┼─────────────────┼──────────────────────────────────┘
      │                  │                 │
┌─────┼──────────────────┼─────────────────┼──────────────────────────────────┐
│     │         DATA SOURCES LAYER         │                                  │
│     │      (ALL via Polygon.io API)      │                                  │
│  ┌──▼──────────────────▼─────────────────▼───────────┐                      │
│  │              BaseSource (ABC)                     │                      │
│  └───┬──────────┬──────────┬──────────┬──────────────┘                      │
│      │          │          │          │                                      │
│  ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐                               │
│  │Polygon │ │Polygon │ │Polygon │ │Polygon │                               │
│  │  SPY   │ │Options │ │  VIX   │ │ News   │                               │
│  │Client  │ │Client  │ │Client  │ │Client  │                               │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘                               │
│      │          │          │          │                                      │
│      ▼          ▼          ▼          ▼                                      │
│          Polygon.io API (Single API Key)                                    │
│          REST + WebSocket (SPY, Options, VIX, News)                         │
└─────────────────────────────────────────────────────────────────────────────┘
      │          │          │          │
┌─────▼──────────▼──────────▼──────────▼──────────────────────────────────────┐
│                      PROCESSING LAYER                                       │
│                                                                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │  Validator  │→│ Deduplicator │→│ Late Data    │                        │
│  │ (Schema)   │  │              │  │  Handler     │                        │
│  └────────────┘  └──────────────┘  └──────────────┘                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Consolidator (On-Demand)                      │       │
│  │  Raw Parquet → Time-Align → Greeks → Features → Targets         │       │
│  │  Per-contract records: ~702K rows/day, ~115+ features           │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                STORAGE LAYER (Parquet Only - Phase 1)                       │
│                               │                                             │
│  ┌────────────────────────────▼──────────────────────────────────────┐      │
│  │                    BaseSink (ABC)                                 │      │
│  └───┬──────────────────┬───────────────────────────────────────────┘      │
│      │                  │                                                  │
│  ┌───▼──────────┐  ┌────▼──────────────┐                                   │
│  │ Parquet Sink │  │  Parquet Sink     │                                   │
│  │  (Raw Feed)  │  │  (Consolidated)  │                                   │
│  │ data/raw/**  │  │ data/processed/** │                                   │
│  └──────────────┘  └──────────────────┘                                    │
│                                                                            │
│  ┌──────────────────────────────────────┐                                   │
│  │ SQL Sink (STUB - NotImplementedError)│                                   │
│  │ Reserved for post-Phase 6 BI needs  │                                   │
│  └──────────────────────────────────────┘                                   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                      CROSS-CUTTING CONCERNS                                │
│                                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │  Logger  │ │  Retry   │ │ Market   │ │  Connection  │ │  Security   │  │
│  │ (Loguru) │ │ Handler  │ │  Hours   │ │   Manager    │ │ (Redaction) │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ └─────────────┘  │
│                                                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Performance  │ │   Schema     │ │    Error     │ │  Heartbeat   │      │
│  │  Monitor     │ │   Monitor    │ │  Aggregator  │ │   Monitor    │      │
│  │(all config.) │ │              │ │              │ │              │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key Simplification**: All four data sources use the Polygon.io API with a single API key. No Massive API dependency. This means unified authentication, unified rate limiting, and a single SDK (`polygon-api-client`).

---

## 1.5 Architecture Coupling Strategy

### Phase 1 (Current): TIGHTLY COUPLED — Intentional Design Choice

```
Polygon API → Python Client → Validator → Deduplicator → Parquet Writer
                (synchronous, single process, direct function calls)
```

**Rationale**:
- Data volume (~750K records/day) fits comfortably in a single Python process
- Simplicity enables fast development and iteration
- Validate data quality and ML approach before adding infrastructure complexity
- No distributed infrastructure overhead
- Easier debugging (single process, sequential execution)
- Faster time-to-insights for ML research

### Kafka/Spark: INTENTIONALLY EXCLUDED from Phase 1

Per SCOPE.md requirements, Kafka is explicitly excluded from Phase 1.

**Reasons**:
1. Data volume does NOT require distributed processing
2. Want to validate data pipeline works before adding infrastructure
3. Single Python process sufficient for current scale
4. Focus on ML research, not infrastructure engineering

### Phase 3: ADD KAFKA — For OPERATIONAL RESILIENCE (Not Volume)

When moving to production real-time streaming (Phase 3), add Kafka for operational benefits:

```
Production Architecture (Phase 3):

Polygon WebSocket → Kafka Topic (spy-raw) → Consumer Group:
                                              ├─ Validator → Kafka (spy-validated)
                                              └─ Raw Backup Consumer

Kafka Topic (spy-validated) → Consumer Group:
                               ├─ Parquet Writer → S3/Local Storage
                               ├─ Real-time Dashboard Consumer
                               ├─ Alerting Service Consumer
                               └─ ML Inference Consumer (optional)
```

**Kafka Benefits (Operational, NOT volume-driven)**:

1. **Fault Isolation**: Component failures don't lose data
   - Validator crashes? Raw data safe in Kafka, replay when fixed
   - Parquet writer slow/failed? Other consumers unaffected
   - Storage full? Backpressure managed via Kafka consumer lag

2. **Replay Capability**: Event sourcing for recovery
   - Reprocess data after bug fix by replaying Kafka topic
   - Recover from corrupted storage by replaying events
   - Test new processing logic on historical events

3. **Easier Maintenance**: Independent component deployment
   - Upgrade Polygon client without touching storage layer
   - Roll back individual components independently

4. **Better Debugging**: Inspect event streams
   - View raw events in Kafka topics
   - Replay specific time windows for investigation

5. **Operational Flexibility**: Add consumers without changing producers
   - Add real-time dashboard by creating new consumer
   - A/B test new processing logic with parallel consumer

6. **Data Durability**: Configurable retention
   - Keep raw events for 7 days (configurable)
   - Replay capability for disaster recovery

**Kafka Cluster Sizing (Phase 3)**:
- Cluster: 3 brokers (small, sufficient for 750K records/day)
- Replication factor: 3 (fault tolerance)
- Retention: 7 days (replay window, configurable)
- Topics: spy-raw, spy-validated, options-raw, options-validated, vix-raw, vix-validated, news-raw, consolidated
- Partitions: 3-5 per topic (parallelism, not required for volume)

### Spark: NOT NEEDED (Any Phase)

**Volume Analysis**:
- ~750K records/day = ~8.7 records/second average
- Peak during market hours: ~30 records/second (still trivial)
- Single Python process handles 1000s of records/second easily

**Decision**:
- Spark adds unnecessary complexity for this volume
- No complex distributed transformations required
- Kafka handles decoupling; Python consumers sufficient for processing
- Save Spark for if/when volume grows 100x (unlikely given ~30 option contracts/day limit)

### Migration Path

| Phase | Architecture | Focus |
|-------|-------------|-------|
| Phase 1 (Now) | Tightly coupled, single process | Validate data quality, build ML models |
| Phase 2 (ML Dev) | Still tightly coupled | Feature engineering, model training, backtesting |
| Phase 3 (Prod RT) | Add Kafka for operational resilience | Fault tolerance, replay, independent deployment |
| Phase 4+ (If needed) | Evaluate Kafka performance | Only add Spark/Flink if Kafka consumers can't keep up |

---

## 2. Technology Stack

### Python Version
- **Python 3.10+** — required for `match` statements, improved type hints, and `|` union syntax

### 2.1 Data Sources Layer

**ALL data sources use Polygon.io API** — single API key, single SDK, unified rate limiting.

| Component | Library | Version | Justification |
|-----------|---------|---------|---------------|
| Polygon SPY (REST) | `polygon-api-client` | >=1.12.0 | Official SDK; `get_aggs()` with built-in pagination |
| Polygon SPY (WebSocket) | `polygon-api-client` | >=1.12.0 | `WebSocketClient` with `Market.Stocks`, subscribe `"A.SPY"` |
| Polygon Options (REST) | `polygon-api-client` | >=1.12.0 | `list_options_contracts()` for discovery |
| Polygon Options (WebSocket) | `polygon-api-client` | >=1.12.0 | `WebSocketClient` with `Market.Options`, subscribe `"AM.{ticker}"` |
| Polygon VIX (REST) | `polygon-api-client` | >=1.12.0 | `get_aggs(ticker="I:VIX", ...)` — same SDK, indices market |
| Polygon VIX (WebSocket) | `polygon-api-client` | >=1.12.0 | `WebSocketClient` with `Market.Indices`, subscribe `"A.I:VIX"` |
| Polygon News (REST) | `polygon-api-client` | >=1.12.0 | `list_ticker_news()` — REST only, no WebSocket |
| Response Validation | `pydantic` | >=2.0.0 | Fast validation, good error messages, type coercion |

**Removed**: `requests`, `urllib3`, `websocket-client` as standalone dependencies for data sources. The `polygon-api-client` SDK handles HTTP and WebSocket internally.

**Rate Limiting — UNIFIED (Critical)**:

All four sources share a single Polygon API key, therefore a single rate limit budget:
- **Free tier**: 5 REST requests/minute (shared across SPY, Options, VIX, News)
- **Paid tier**: Higher limits (configurable)
- **Implementation**: Single token bucket in `ConnectionManager`, shared across all source clients
- **WebSocket**: Not rate-limited per message (persistent connection), but limited to 1 connection per market type
- **Allocation strategy** (configurable via YAML):

```yaml
polygon:
  rate_limiting:
    total_requests_per_minute: 5   # Free tier total budget
    allocation:                     # How to split the budget
      spy: 2                        # 2 req/min for SPY historical
      vix: 1                        # 1 req/min for VIX historical
      options: 1                    # 1 req/min for options discovery
      news: 1                       # 1 req/min for news polling
    # Historical mode: sources run sequentially, so full budget available to active source
    # Streaming mode: only news polling consumes REST budget (others use WebSocket)
```

**Connection Pooling**: Single `requests.Session` shared via the Polygon SDK. WebSocket connections: one per market type (Stocks, Options, Indices).

### 2.2 Processing Layer

| Component | Library/Approach | Justification |
|-----------|-----------------|---------------|
| Schema Validation | `pydantic` v2 models | Declarative schemas, fast validation (~10x faster than v1), clear error messages |
| Deduplication | In-memory set (timestamp + source key) | Per-second data has natural unique key (source + timestamp). Set fits in memory for one trading day (~23,400 seconds per source) |
| Late Data Handler | Timestamp comparison | Simple: reject records where `record_ts < current_day_start` in real-time mode. Configurable grace period. |
| Consolidation | `pandas` DataFrame merge → Parquet | Time-aligned asof merge of SPY + VIX + Options + News into per-contract consolidated Parquet |

**Feature Engineering Libraries** (used by Consolidator):

| Component | Library | Version | Justification |
|-----------|---------|---------|---------------|
| Options Greeks | `py_vollib` | >=1.0.1 | Black-Scholes delta, gamma, theta, vega, rho calculation |
| Technical Indicators | `pandas-ta` | >=0.3.14b | RSI, EMA, MACD — pure pandas, no C dependencies |
| News Embeddings | `sentence-transformers` | >=2.2.0 | 384-dim topic vectors (optional, can be disabled) |
| Statistical Functions | `scipy` | >=1.10.0 | Skewness, kurtosis, rolling statistical moments |
| ML Preprocessing | `scikit-learn` | >=1.3.0 | Scaling, normalization (used during consolidation) |
| Array Operations | `numpy` | >=1.24.0 | Momentum arrays, vectorized computations |

**Memory Management**: Generator-based processing for raw ingestion. Consolidation loads one full trading day into memory (~702K records x ~100 columns ~ 500MB peak). Batch size configurable (default: 10,000 records for raw ingestion).

### 2.3 Storage Layer (Parquet Only — Phase 1)

| Component | Library | Justification |
|-----------|---------|---------------|
| Parquet Writer (all storage) | `pyarrow` | Fastest Parquet implementation in Python. Native columnar compression. Schema evolution support. |
| Compression | Snappy (default) | Best read speed; reasonable compression ratio. Configurable to zstd/gzip via YAML. |
| DataFrame Construction | `pandas` | Build DataFrames from record batches before Parquet write. Direct ML pipeline integration. |
| SQL Sink | STUB (`NotImplementedError`) | File exists in project structure. Reserved for post-Phase 6 BI/analysis needs. |
| Cloud Storage | STUB (`NotImplementedError`) | Reserved for future cloud deployment. |

**Why Parquet for Everything — No SQL in Phase 1**:
- **Data volume**: ~750K records/day — does not justify database infrastructure
- **Access pattern**: Time-series, append-only, time-range queries — Parquet's sweet spot
- **Use case**: ML research with direct Pandas/Polars consumption — 10-100x faster than SQL for analytics
- **Operational simplicity**: No database to provision, manage, backup, or pay for
- **Interim SQL**: DuckDB can query Parquet files directly (`pip install duckdb`) — zero infrastructure

**Storage Structure**:
```
data/raw/                                    # Individual feeds (per-source)
├── spy/
│   └── date=YYYY-MM-DD/
│       └── spy_{batch}_{timestamp}.parquet
├── options/
│   ├── contracts/
│   │   └── {date}_contracts.json            # Daily discovered contracts
│   └── date=YYYY-MM-DD/
│       └── options_{batch}_{timestamp}.parquet
├── vix/
│   └── date=YYYY-MM-DD/
│       └── vix_{batch}_{timestamp}.parquet
└── news/
    └── date=YYYY-MM-DD/
        └── news_{batch}_{timestamp}.parquet

data/processed/                              # Multi-source consolidated
└── consolidated/
    └── date=YYYY-MM-DD/
        └── consolidated.parquet             # Per-contract, per-second, ~115+ features
```

**Estimated Storage Per Day** (~750K records):

| Source | Records/Day | Size/Day (Snappy) | Size/Month |
|--------|------------|-------------------|------------|
| SPY Aggregates | ~23,400 | ~2 MB | ~60 MB |
| Options (~30 contracts) | ~702,000 | ~50 MB | ~1.5 GB |
| VIX Aggregates | ~23,400 | ~2 MB | ~60 MB |
| News | ~500 | <1 MB | ~10 MB |
| **Raw Total** | **~750,000** | **~55 MB** | **~1.6 GB** |
| Consolidated (~115+ cols) | ~702,000 | ~180 MB | ~5.4 GB |
| **Grand Total** | — | **~235 MB** | **~7.0 GB** |

**Write Batching**: Configurable batch size (default 10,000). Records buffered in memory until batch threshold, then flushed as a single Parquet file.

**Duplicate Handling**: Check if partition directory exists. If overwrite mode: replace entire partition. Otherwise: append with incremented batch number.

### 2.4 Orchestration Layer

| Component | Approach | Justification |
|-----------|----------|---------------|
| Historical Runner | Sequential date iteration with checkpoint | Simple, debuggable. Fetches one date at a time. Checkpoint file tracks last completed date for resumability. |
| Streaming Runner | Event loop with `WebSocketClient.run()` | Polygon SDK's built-in event loop. Callbacks process messages in batches. Heartbeat thread monitors liveness. |
| Simulator | Time-delayed Parquet reader | Reads historical Parquet files, replays records with configurable delay to simulate real-time feed. |
| Task Manager | JSON-based task tracking | Lightweight. Tracks run_id, status, checkpoint, start/end times. No external dependency needed. |

**Checkpoint Mechanism (Historical)**:
```json
{
  "run_id": "uuid",
  "source": "spy",
  "last_completed_date": "2025-10-15",
  "target_end_date": "2025-10-28",
  "status": "in_progress",
  "records_processed": 234000,
  "last_updated": "2026-02-13T10:30:00Z"
}
```
On resume: read checkpoint, continue from `last_completed_date + 1 day`.

**Graceful Shutdown**: Signal handlers (SIGINT, SIGTERM) set a shutdown flag. Active batch completes, checkpoint saves, connections close. No data loss.

**Parallelization**: None in Phase 1. Sources run sequentially within a single process. This simplifies debugging and avoids rate limit conflicts.

### 2.5 Monitoring Layer

| Component | Approach | Justification |
|-----------|----------|---------------|
| Performance Monitor | `time.perf_counter()` + JSON metrics file | No external dependency. All thresholds configurable via `settings.yaml`. Tracks commit latency, throughput, memory, error rate. |
| Schema Monitor | Pydantic `model_validate()` + field diff | Compares incoming fields against expected schema. Logs drift once per day. Continues processing. |
| Error Aggregator | Counter dict + periodic JSON dump | Categorizes errors by type. Dumps summary every 15 min. Enables pattern analysis. |
| Heartbeat Monitor | JSON metadata file updated every 5 min | Per CLAUDE.md requirements. `data/logs/heartbeat/stream_status.json`. Alert if stale >15 min. |

**No per-message logging** in streaming mode. Only heartbeat updates and error/alert-level events.

### 2.6 Utilities Layer

| Component | Library | Justification |
|-----------|---------|---------------|
| Config Loader | `pyyaml` + `python-dotenv` | YAML for structure, .env for secrets. Environment variable substitution in YAML via `${VAR}` pattern. |
| Logger | `loguru` | Zero-config, structured logging, automatic rotation, easy filtering. |
| Retry Handler | `tenacity` | Declarative retry with exponential backoff + jitter. Configurable per-source. |
| Connection Manager | Custom class | Manages Polygon SDK sessions, WebSocket connections, unified rate limiting, health checks. |
| Market Hours | `pytz` + Polygon market status API | Timezone-aware market hour checks. Caches holiday calendar. |

---

## 2.7 Daily Reset Schedule

**All daily resets occur at market open (9:30 AM ET)** unless noted otherwise:

| Component | What Resets | When | How |
|-----------|-----------|------|-----|
| Deduplicator | In-memory set of seen `(source, timestamp_ms)` keys | 9:30 AM ET (market open) | `dedup_set.clear()` — fresh set for new trading day |
| Options Contract List | Discovered contract tickers | 9:30 AM ET (market open) | Clear previous day's contracts, run contract discovery for today |
| Heartbeat Monitor | `total_messages_today` counter, daily stats | 9:30 AM ET (market open) | Reset counters, new heartbeat file for the day |
| Error Aggregator | Daily error counters | Midnight ET (00:00) | Write final daily summary, reset counters |
| Performance Monitor | Daily metrics accumulators | Midnight ET (00:00) | Write final daily metrics, start new metrics file |
| Schema Monitor | "Already reported drift today" flag | Midnight ET (00:00) | Allow one new drift report per source per day |

**Implementation**: `MarketHours.on_market_open()` callback triggers market-open resets. Midnight resets via timestamp comparison (current date != last reset date).

---

## 3. Data Flow

### 3.1 Historical Mode Data Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│ Config  │────▶│  Historical  │────▶│  Data Source  │────▶│ Polygon   │
│ Loader  │     │   Runner     │     │   Client     │     │ REST API  │
└─────────┘     └──────┬───────┘     └──────┬───────┘     └───────────┘
                       │                    │
                       │                    │ Generator<Dict>
                       │                    ▼
                       │              ┌──────────────┐
                       │              │  Validator   │ ── invalid → quarantine/
                       │              │  (Pydantic)  │
                       │              └──────┬───────┘
                       │                     │ valid records
                       │                     ▼
                       │              ┌──────────────┐
                       │              │ Deduplicator │ ── duplicate → skip
                       │              └──────┬───────┘
                       │                     │ unique records
                       │                     ▼
                       │              ┌──────────────┐
                       │              │  Batch Buffer│ (10,000 records)
                       │              └──────┬───────┘
                       │                     │ full batch
                       │                     ▼
                       │              ┌──────────────┐
                       │              │  Parquet     │──▶ data/raw/{source}/
                       │              │  Sink        │    date=YYYY-MM-DD/
                       │              └──────────────┘
                       │
                       │ checkpoint after each date
                       ▼
                 data/logs/checkpoints/
```

### 3.2 Streaming Mode Data Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│ Config  │────▶│  Streaming   │────▶│  Data Source  │────▶│ Polygon   │
│ Loader  │     │   Runner     │     │   Client     │     │ WebSocket │
└─────────┘     └──────┬───────┘     └──────┬───────┘     └───────────┘
                       │                    │
                       │                    │ Callback(List[msg])
                Market Hours               │
                  Check                    ▼
                (9:30-4:00 ET)     ┌──────────────┐
                       │          │  Validator   │
                       │          └──────┬───────┘
                       │                 │
                       │                 ▼
                       │          ┌──────────────┐
                       │          │ Late Data    │ ── too old → skip + log
                       │          │  Handler     │
                       │          └──────┬───────┘
                       │                 │
                       │                 ▼
                       │          ┌──────────────┐
                       │          │ Batch Buffer │ (time or count trigger)
                       │          └──────┬───────┘
                       │                 │
                       ├─────────────────┤
                       │                 ▼
                ┌──────▼──────┐   ┌──────────────┐
                │  Heartbeat  │   │  Parquet     │──▶ data/raw/{source}/
                │  Monitor    │   │  Sink        │    date=YYYY-MM-DD/
                │ (every 5m)  │   └──────────────┘
                └─────────────┘
                       │
                 stream_status.json
```

### 3.3 Consolidation Data Flow (On-Demand)

```
  Triggered manually: python -m src.processing.consolidator --date 2025-01-15

  ┌─────────────────────────────────────────────────────────────┐
  │                    LOAD RAW DATA                            │
  │                                                             │
  │  data/raw/spy/date=YYYY-MM-DD/*.parquet      → SPY df      │
  │  data/raw/options/date=YYYY-MM-DD/*.parquet   → Options df  │
  │  data/raw/vix/date=YYYY-MM-DD/*.parquet       → VIX df      │
  │  data/raw/news/date=YYYY-MM-DD/*.parquet      → News df     │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 1: TIME-ALIGN                                         │
  │  • SPY + VIX: merge_asof on timestamp (same-second match)   │
  │  • Options: merge_asof on timestamp (per contract)           │
  │  • News: fuzzy match ±2min, aggregate sentiment              │
  │  Result: One row per second per option contract              │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 2: COMPUTE GREEKS (py_vollib)                         │
  │  • delta, gamma, theta, vega, rho per contract per second   │
  │  • Smoothed with 2-minute rolling average                   │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 3: COMPUTE MOMENTUM ARRAYS                            │
  │  • spy_vwap_momentum_120min: array of 120 % changes         │
  │  • spy_close_momentum_120min                                 │
  │  • option_mid_momentum_120min                                │
  │  • option_volume_momentum_120min                             │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 4: COMPUTE VOLATILITY (rolling std of returns)        │
  │  • 30-second, 1-min, 5-min, 30-min windows                  │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 5: COMPUTE TECHNICAL INDICATORS (pandas-ta)           │
  │  • RSI-14, EMA-9, EMA-21, MACD (12,26,9)                   │
  │  • Volume spike ratios                                       │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 6: COMPUTE CROSS-ASSET & STATISTICAL FEATURES         │
  │  • spy_option_spread_ratio, option_stock_lag_corr            │
  │  • Rolling mean, std, skewness, kurtosis, z-score (30min)   │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 7: AGGREGATE NEWS                                     │
  │  • sentiment_score, count_last_15min, avg_sentiment_15min    │
  │  • time_since_last_seconds, publisher_weight                 │
  │  • topic_embedding (optional 384-dim vector)                 │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 8: COMPUTE TARGET VARIABLE                             │
  │  • target_future_prices: array[120] of avg_P per minute     │
  │    for next 120 minutes (average option mid price)           │
  │  • Window: min(120, minutes_until_market_close)              │
  │  • Padding: remaining slots filled with last known avg price │
  │  • Derived metrics computed on-the-fly by Signal Validator   │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  WRITE CONSOLIDATED PARQUET                                  │
  │  → data/processed/consolidated/date=YYYY-MM-DD/              │
  │    consolidated.parquet                                      │
  │  ~702K rows × ~115+ columns ≈ 130-220 MB/day compressed     │
  └──────────────────────────────────────────────────────────────┘
```

### 3.4 Record Processing Pipeline (Raw Ingestion Detail)

```
Raw API Response
      │
      ▼
┌─────────────────┐
│ Pydantic Model  │  Validate schema, coerce types, extract fields
│ .model_validate()│  Reject: missing required fields, wrong types
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deduplicator    │  Key: (source, timestamp_ms)
│ .is_duplicate() │  In-memory set, reset at 9:30 AM ET daily
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Late Data Check │  Historical: accept all (backfill)
│                 │  Streaming: reject if ts < today 00:00 ET
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Buffer    │  Accumulate until batch_size OR flush_interval
│                 │  Default: 10,000 records or 60 seconds
└────────┬────────┘
         │
         ▼
   Write to Parquet
   (data/raw/{source}/date=YYYY-MM-DD/)
```

---

## 4. Component Details

### 4.1 Polygon SPY Client (`src/data_sources/polygon_client.py`)

**Purpose**: Fetch SPY per-second aggregate bars via REST (historical) and WebSocket (real-time).

**Technology Stack**:
- Primary: `polygon-api-client` (official SDK)
- Supporting: `pydantic` for response models, `tenacity` for retries

**Configuration**:
```yaml
# config/sources.yaml
polygon:
  api_key: ${POLYGON_API_KEY}
  base_url: https://api.polygon.io
  feed: delayed          # "delayed" or "realtime"
  spy:
    ticker: SPY
    multiplier: 1
    timespan: second
    limit_per_request: 50000
```

**Implementation Approach**:
- Historical: `client.get_aggs()` with date-by-date iteration. Generator yields batches.
- Streaming: `WebSocketClient(market=Market.Stocks)` subscribes to `"A.SPY"`. Callback buffers messages, flushes on batch threshold.
- Pagination: SDK handles `next_url` automatically via iterator protocol.
- Rate limiting: Coordinated via shared `ConnectionManager` token bucket.

**Error Handling**:
- Catch: `polygon.exceptions.BadResponse`, `requests.exceptions.ConnectionError`, `requests.exceptions.Timeout`
- Retry: Via `tenacity` — 3 attempts, exponential backoff (1s, 2s, 4s) + jitter, only for 5xx and timeouts
- 429 (rate limit): Respect `Retry-After` header, sleep and retry
- 403 (auth): Log redacted key suffix, raise immediately (no retry)
- Fallback: Log error, skip date, continue to next date (historical) or reconnect (streaming)

**Performance Characteristics**:
- Throughput: ~23,400 records/trading day (1 per second, 6.5 hours)
- Memory: ~50MB peak (generator-based, 10K record buffer)
- Latency: ~200ms per REST request + rate limit sleep

**Testing Strategy**:
- Unit: Mock `RESTClient.get_aggs()` return value. Test pagination, error handling, record transformation.
- Integration: Single-date fetch against production with test key.
- Fixtures: `tests/mocks/polygon_spy_response.json` with sample aggregate data.

---

### 4.2 Polygon Options Client (within `src/data_sources/polygon_client.py`)

**Purpose**: Discover next-day expiring options contracts at market open, then stream per-second data for contracts within ±1% of SPY opening price.

**Technology Stack**: Same as SPY client (shared `polygon-api-client`).

**Options Expiration Policy — NEXT-DAY ONLY**:
- If today is Monday → track options expiring Tuesday
- If today is Friday → track options expiring Monday (next trading day)
- Do NOT track same-day (0DTE) options
- Do NOT track multiple expiration dates
- Use `MarketHours.next_trading_day()` to compute expiration date (skips weekends and holidays)
- Re-discover contracts each trading day at market open (9:30 AM ET)

**Implementation Approach**:
- **Contract Discovery** (at 9:30 AM ET): Fetch SPY opening price → calculate ±1% range → compute next trading day → call `list_options_contracts(expiration_date=next_trading_day, strike_price_gte=lower, strike_price_lte=upper)`
- **Contract Storage**: Save discovered contracts as JSON in `data/raw/options/contracts/{date}_contracts.json`.
- **Streaming**: `WebSocketClient(market=Market.Options)` subscribes to `"AM.{ticker}"` for each discovered contract (max 100 per Polygon recommendation).
- **Daily Reset**: At 9:30 AM ET, clear previous day's contract list, run fresh discovery.
- **Expected Volume**: ~30 contracts within ±1% range (~14-15 calls + ~14-15 puts) x 23,400 seconds = ~702,000 records/day.

**Configuration**:
```yaml
polygon:
  options:
    underlying_ticker: SPY
    strike_range_pct: 0.01    # ±1%
    max_contracts: 100
    expiration_lookahead_days: 1  # Next-day expiry ONLY
```

**Error Handling**:
- Contract discovery failure: Retry 3x, then skip options for the day (log error)
- WebSocket disconnect mid-stream: Auto-reconnect via `ConnectionManager`, re-subscribe to all tickers
- >100 contracts discovered: Take first 100 sorted by strike proximity to opening price

---

### 4.3 Polygon VIX Client (`src/data_sources/polygon_client.py`)

**Purpose**: Fetch VIX per-second data via Polygon Indices API (REST + WebSocket).

**CORRECTION**: VIX data comes from Polygon.io (NOT Massive API). The `massive_client.py` module is no longer needed for VIX. VIX is available via Polygon's indices market using ticker `I:VIX`.

**Technology Stack**: Same `polygon-api-client` SDK used for SPY and Options.

**Configuration**:
```yaml
# config/sources.yaml
polygon:
  vix:
    ticker: "I:VIX"
    market: indices
    multiplier: 1
    timespan: second
    limit_per_request: 50000
```

**Implementation Approach**:
- **Historical REST**: `client.get_aggs(ticker="I:VIX", multiplier=1, timespan="second", from_=date, to=date, limit=50000)` — identical pattern to SPY, different ticker.
- **Real-time WebSocket**: `WebSocketClient(market=Market.Indices)` subscribes to `"A.I:VIX"` for per-second aggregate bars.
- **Response Schema**: Same aggregate bar format as SPY (o, c, h, l, s, e fields). No volume/vwap for indices.
- **Shared SDK instance**: Can reuse the same `RESTClient` as SPY (single API key).

**Performance Characteristics**:
- Throughput: ~23,400 records/trading day (1 per second)
- Memory: ~30MB peak

**Error Handling**: Same retry pattern as SPY. Rate limiting shared via unified token bucket.

---

### 4.4 Massive Client — REMOVED (`src/data_sources/massive_client.py`)

**Status**: **DEPRECATED / STUB**

The Massive API is not used. VIX data comes from Polygon (see Section 4.3). The `massive_client.py` file will be converted to a stub:

```python
class MassiveClient(BaseSource):
    """DEPRECATED: VIX data now sourced from Polygon API (ticker I:VIX).

    This module is not used in Phase 1. All data sources use Polygon.io.
    Retained as stub in case a Massive API integration is needed in future phases.
    """

    def connect(self) -> None:
        raise NotImplementedError("Massive API not used. VIX via Polygon (I:VIX).")

    # ... all methods raise NotImplementedError
```

**Implication**: Remove `MASSIVE_API_KEY` from `.env` template. Remove `massive:` section from `sources.yaml`. Remove `websocket-client` from `requirements.txt` (Polygon SDK handles WebSocket internally).

---

### 4.5 News Client (`src/data_sources/news_client.py`)

**Purpose**: Fetch market news with sentiment from Polygon News API.

**Technology Stack**: `polygon-api-client` — `list_ticker_news()` method.

**Configuration**:
```yaml
polygon:
  news:
    tickers: [SPY]             # Filter by relevant tickers
    sort: published_utc
    order: asc
    limit_per_request: 100
    poll_interval_seconds: 300  # 5 min (configurable)
```

**Implementation Approach**:
- Historical: Paginate through all news for date range
- Real-time: Poll every 5 minutes (configurable). News has no WebSocket — polling loop in `stream_realtime()`.
- Fields extracted: title, description, sentiment, sentiment_reasoning, keywords, tickers, published_utc

**Performance Characteristics**:
- Volume: ~500 articles/day (varies with market activity)
- Memory: Negligible
- REST budget consumption: 1 request per poll interval (every 5 min = 12 req/hour in streaming mode)

---

### 4.6 Parquet Sink (`src/sinks/parquet_sink.py`)

**Purpose**: Write date-partitioned Parquet files for ALL storage — both individual raw feeds and consolidated data.

**Technology Stack**:
- Primary: `pyarrow` for Parquet writes
- Supporting: `pandas` for DataFrame construction before write

**Configuration**:
```yaml
sinks:
  parquet:
    enabled: true
    base_path: data/raw
    consolidated_path: data/processed/consolidated
    compression: snappy      # snappy | zstd | gzip | none
    row_group_size: 10000
    max_file_size_mb: 256
```

**Implementation Approach**:
- Build `pandas.DataFrame` from record batch
- Convert to `pyarrow.Table`
- Write with `pq.write_table()` to partitioned path
- Raw file naming: `{source}_{batch_number}_{timestamp}.parquet`
- Consolidated file naming: `consolidated.parquet` (one per date partition)
- Overwrite mode: Delete existing partition directory, rewrite

---

### 4.7 SQL Sink — STUB (`src/sinks/sql_sink.py`)

**Status**: STUB — `NotImplementedError` on all methods. Reserved for post-Phase 6 BI/analysis needs.

---

### 4.8 Consolidator (`src/processing/consolidator.py`)

**Purpose**: Time-align all four raw data sources into per-contract, per-second consolidated Parquet files with ~115+ ML features and a single target variable (`target_future_prices[120]`).

**Technology Stack**:
- Primary: `pandas` for DataFrame operations and `merge_asof` joins
- Greeks: `py_vollib` for Black-Scholes calculations
- Technical indicators: `pandas-ta` for RSI, EMA, MACD
- Statistics: `scipy` for skewness, kurtosis
- News embeddings: `sentence-transformers` (optional)
- Output: `pyarrow` Parquet via the Parquet Sink

**Trigger**: On-demand in Phase 1 (manual CLI invocation):
```bash
python -m src.processing.consolidator --date 2025-01-15
python -m src.processing.consolidator --date-range 2025-01-01 2025-01-31
```

**Consolidated Record Schema** (per second, per option contract):

```
# === Time & Identity ===
timestamp              INT64      # Unix timestamp (ms) - PRIMARY KEY component
datetime_et            STRING     # Human-readable ET: "2025-01-15 10:30:45"
date                   STRING     # YYYY-MM-DD (partition key)
time_since_open_minutes INT32     # Minutes since 9:30 AM (0-390)
normalized_time_of_day FLOAT64    # time_since_open / 390 (intraday cycle)

# === Option Contract Identity ===
option_symbol          STRING     # e.g., "SPY250116C00700000"
contract_type          STRING     # "CALL" or "PUT"
strike                 FLOAT64    # Strike price
expiry_date            STRING     # Next-day expiration (YYYY-MM-DD)
time_to_expiry_hours   FLOAT64    # Hours until expiration

# === SPY Underlying Data (1-second aggregates) ===
spy_open               FLOAT64
spy_high               FLOAT64
spy_low                FLOAT64
spy_close              FLOAT64
spy_volume             INT64
spy_vwap               FLOAT64    # Volume-weighted average price
spy_average_trade_size INT32      # Field 'z' from Polygon

# === VIX Data (1-second aggregates) ===
vix_open               FLOAT64
vix_high               FLOAT64
vix_low                FLOAT64
vix_close              FLOAT64

# === Options Quote Data (1-second) ===
option_bid             FLOAT64    # Best bid price
option_ask             FLOAT64    # Best ask price
option_mid             FLOAT64    # (bid + ask) / 2
option_last            FLOAT64    # Last trade price
option_volume          INT64      # Trade volume
option_open_interest   INT64      # Open interest (if available)
option_implied_volatility FLOAT64 # IV (if available from Polygon)

# === Computed Greeks (Black-Scholes via py_vollib) ===
delta                  FLOAT64    # Option sensitivity to SPY price
gamma                  FLOAT64    # Rate of change of Delta
theta                  FLOAT64    # Time decay per day
vega                   FLOAT64    # Sensitivity to volatility
rho                    FLOAT64    # Sensitivity to interest rates
# Greeks smoothed with 2-minute rolling average to reduce noise

# === News/Sentiment Data (Time-Aligned ±2 minutes) ===
news_sentiment_score       FLOAT64  # +1 (positive), 0 (neutral), -1 (negative)
news_count_last_15min      INT32    # Number of news items in last 15 minutes
news_avg_sentiment_15min   FLOAT64  # Rolling average sentiment
news_time_since_last_seconds INT32  # Seconds since most recent news
news_publisher_weight      FLOAT64  # Weighted importance (Reuters > blogs)
news_topic_embedding       FLOAT64[384]  # sentence-transformer vector (optional)

# === Derived Features: Price Momentum (120-min arrays) ===
spy_vwap_momentum_120min          FLOAT64[120]  # % change vs t-n min, n=1..120
spy_close_momentum_120min         FLOAT64[120]
option_mid_momentum_120min        FLOAT64[120]
option_volume_momentum_120min     FLOAT64[120]

# === Derived Features: Volatility (Rolling Windows) ===
spy_volatility_30sec       FLOAT64  # Rolling σ of returns (30-second window)
spy_volatility_1min        FLOAT64
spy_volatility_5min        FLOAT64
spy_volatility_30min       FLOAT64

# === Derived Features: Volume Analysis ===
spy_volume_spike_ratio     FLOAT64  # current_vol / mean(vol_last_30min)
option_volume_spike_ratio  FLOAT64

# === Derived Features: Technical Indicators ===
spy_rsi_14                 FLOAT64  # 14-period RSI
spy_ema_9                  FLOAT64  # 9-period EMA
spy_ema_21                 FLOAT64  # 21-period EMA
spy_macd                   FLOAT64  # MACD (12, 26, 9)
spy_macd_signal            FLOAT64
spy_macd_hist              FLOAT64  # MACD - Signal

# === Derived Features: Cross-Asset Relationships ===
spy_option_spread_ratio    FLOAT64  # (option_mid - spy_close) / spy_close
option_stock_lag_corr      FLOAT64  # Pearson corr(option_returns, spy_returns_lagged)
relative_volatility_shift  FLOAT64  # (σ_option / σ_spy) - 1

# === Derived Features: Statistical Moments (30-min rolling) ===
spy_returns_mean_30min     FLOAT64
spy_returns_std_30min      FLOAT64
spy_returns_skewness_30min FLOAT64
spy_returns_kurtosis_30min FLOAT64
spy_returns_zscore         FLOAT64  # (current_return - mean) / std

# === Target Variable (For ML Training) ===
target_future_prices       FLOAT64[120]  # [avg_P[t+1min], avg_P[t+2min], ..., avg_P[t+120min]]
                                         # Average option mid price per minute for next 120 minutes
                                         # Window: min(120, minutes_until_market_close)
                                         # Padding: remaining slots filled with last known average price
# Training: Computed from actual future prices during consolidation
# Real-time (Phase 3): Predicted by ML model (LSTM/Transformer)
# Derived metrics (growth_pct, max_growth, sustained) computed on-the-fly by Signal Validator
```

**Output**: ~702,000 rows/day (23,400 seconds x ~30 contracts), ~115+ columns, ~130-220 MB/day compressed (reduced by removing 5 derived target fields).

**Data Alignment Rules**:
1. **SPY + VIX + Options**: Direct timestamp match (same second, `merge_asof`)
2. **News**: Fuzzy time match — for each record at time T, aggregate news from [T-120sec, T]
3. **Missing Data**:
   - VIX missing for a second: forward-fill from last available
   - Option quote missing: mark as NULL, flag with `is_imputed=1`
   - No news in last 15 min: sentiment=0, count=0

**Derived Field Calculation Order**:
1. Load raw data (SPY, Options, VIX, News) for date
2. Time-align (join on timestamp, fuzzy match news)
3. Compute Greeks (Black-Scholes: delta, gamma, theta, vega, rho)
4. Compute momentum arrays (120-min % change sequences)
5. Compute volatility (rolling σ for multiple windows)
6. Compute technical indicators (RSI, EMA, MACD via pandas-ta)
7. Compute cross-asset features (correlations, spreads)
8. Compute statistical moments (mean, std, skewness, kurtosis, z-score)
9. Aggregate news (sentiment scores, counts, publisher weights)
10. Compute target variable (future average prices array with market-close padding)
11. Write consolidated.parquet

**Performance**:
- Runs after raw data ingestion completes for a date
- Loads ~750K raw records, produces ~702K consolidated rows
- Expected time: <2 minutes per day (dominated by Greeks computation)
- Memory peak: ~500MB (full trading day in memory)

**Error Handling**:
- Missing source for a date: Fill with NaN, log warning, continue
- Greeks calculation failure (e.g., division by zero for deep ITM/OTM): Set Greeks to NaN, log
- Schema mismatch: Pydantic validation before merge

---

### 4.9 Validator (`src/processing/validator.py`)

**Purpose**: Validate incoming records against expected schemas.

**Technology Stack**: `pydantic` v2 BaseModel classes.

**Implementation Approach**:
- Define one Pydantic model per source:
  - `SPYAggregateRecord` — o, c, h, l, v, vw, s, e (all required)
  - `OptionsRecord` — contract fields + price data
  - `VIXRecord` — o, c, h, l, s, e
  - `NewsRecord` — title, published_utc, sentiment (optional fields nullable)
- `model_validate()` raises `ValidationError` on bad records
- Invalid records: log error context, write to `data/raw/{source}/quarantine/` for inspection
- Schema drift detection: Compare incoming field set against model fields, log additions/removals once per day

---

### 4.10 Retry Handler (`src/utils/retry_handler.py`)

**Purpose**: Configurable retry logic with exponential backoff.

**Technology Stack**: `tenacity` library.

**Configuration**:
```yaml
# config/retry_policy.yaml
default:
  max_attempts: 3
  initial_wait_seconds: 1.0
  max_wait_seconds: 30.0
  exponential_base: 2
  jitter: true
  retry_on_status_codes: [500, 502, 503, 504, 429]

polygon:
  max_attempts: 5
  rate_limit_wait_seconds: 12  # 5 req/min = 12s between requests
```

**Implementation Approach**:
- Provide a decorator factory: `@with_retry(source="polygon")`
- Reads retry config for the source, falls back to default
- Integrates with logger: logs each retry attempt with attempt number and wait time
- Never retries on 4xx (except 429)

---

### 4.11 Logger (`src/utils/logger.py`)

**Purpose**: Centralized logging with automatic credential redaction.

**Technology Stack**: `loguru`

**Implementation Approach**:
- Console sink: INFO level, colorized
- File sink: DEBUG level, rotated daily, retained 30 days
  - `data/logs/execution/{date}.log`
  - `data/logs/errors/{date}.log` (ERROR+ only)
- Redaction filter: Applied to all sinks. Masks API keys, passwords, connection strings.
- Format: `{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}`

---

### 4.12 Config Loader (`src/utils/config_loader.py`)

**Purpose**: Load and merge YAML configuration with environment variable substitution. Support runtime reload.

**Technology Stack**: `pyyaml` + `python-dotenv`

**Implementation Approach**:
- Load `.env` file via `dotenv`
- Parse YAML files, resolve `${VAR}` patterns against `os.environ`
- Merge configs: `settings.yaml` (master) + `sources.yaml` + `sinks.yaml` + `retry_policy.yaml`
- Validate required keys exist (fail fast on missing `POLYGON_API_KEY`)
- Provide `reload()` method for runtime config updates (monitoring thresholds)

---

### 4.13 Market Hours (`src/utils/market_hours.py`)

**Purpose**: Determine if market is open. Prevent streaming outside trading hours. Calculate next trading day for options expiration.

**Technology Stack**: `pytz` for timezone, `python-dateutil` for date math, Polygon API for holiday calendar.

**Implementation Approach**:
- `is_market_open() -> bool`: Check day-of-week (Mon-Fri), time (9:30-16:00 ET), holidays
- `seconds_until_market_close() -> int`: For graceful pre-close shutdown
- `next_trading_day(from_date) -> date`: Skip weekends and holidays — used for options expiration date calculation
- `on_market_open() -> callback`: Trigger daily resets (deduplicator, contracts, heartbeat)
- Holiday cache: Fetch from Polygon `/v1/marketstatus/upcoming` once per day, cache as JSON. Fallback: empty list if API fails.

**Configuration**:
```yaml
streaming:
  market_hours:
    timezone: America/New_York
    active_days: [0, 1, 2, 3, 4]  # Mon-Fri
    start_time: "09:30"
    end_time: "16:00"
```

---

### 4.14 Connection Manager (`src/utils/connection_manager.py`)

**Purpose**: Manage Polygon SDK sessions, WebSocket connections, and unified rate limiting across all sources.

**Implementation Approach**:
- **Single `RESTClient` instance** shared across SPY, VIX, Options, News (single API key)
- **Unified token bucket**: All REST requests draw from one shared rate limit budget
  - Free tier: 5 requests/minute total across all sources
  - Historical mode: sources run sequentially → full budget available to active source
  - Streaming mode: only news polling consumes REST budget (others use WebSocket)
- **WebSocket connections**: Up to 3 concurrent (Stocks, Options, Indices)
- **Auto-reconnect**: On WebSocket disconnect, wait with backoff, reconnect, re-subscribe
- **Health check**: Ping endpoint before critical operations

**Rate Limit Reset**: The Polygon rate limit window resets every minute. The token bucket refills automatically. If a 429 response is received, `Retry-After` header is respected (overrides the token bucket timer).

---

### 4.15 Performance Monitor (`src/monitoring/performance_monitor.py`)

**Purpose**: Track commit latency, throughput, memory usage, and error rate SLAs. ALL thresholds configurable via `settings.yaml` — zero hardcoded values.

**Configuration** (all thresholds in `config/settings.yaml`, with sensible defaults):
```yaml
monitoring:
  performance:
    commit_latency_seconds: 300           # Alert if batch write exceeds (default: 5 min)
    throughput_min_records_per_sec: 100    # Alert if throughput drops below (default: 100)
    memory_usage_mb_threshold: 1000       # Alert if process memory exceeds (default: 1 GB)
    error_rate_percent: 1.0               # Alert if error rate exceeds (default: 1%)
    error_window_minutes: 15              # Sliding window for error rate calculation (default: 15 min)
    heartbeat_check_interval_seconds: 60  # How often to check heartbeat staleness (default: 60s)
    metrics_dump_interval_minutes: 15     # How often to write metrics JSON (default: 15 min)
    latency_window_size: 100              # Rolling window size for p50/p95/p99 (default: 100)
```

**Implementation Requirements**:
- All thresholds loaded from config at startup via `ConfigLoader`
- If a threshold key is missing from config: use the default value and log a warning
- **No hardcoded thresholds anywhere in the module**
- **Runtime threshold updates**: Support config reload without process restart via `ConfigLoader.reload()`

**Alert Triggers** (all configurable):

| Metric | Config Key | Default | Check Frequency |
|--------|-----------|---------|-----------------|
| Commit latency | `commit_latency_seconds` | 300 (5 min) | Per batch write |
| Throughput | `throughput_min_records_per_sec` | 100 | Rolling average per batch |
| Memory usage | `memory_usage_mb_threshold` | 1000 (1 GB) | Every `heartbeat_check_interval_seconds` |
| Error rate | `error_rate_percent` | 1.0% | Over sliding `error_window_minutes` |

**Implementation Approach**:
- Time each batch write with `time.perf_counter()`
- Track memory via `psutil.Process().memory_info().rss`
- Maintain rolling window of latencies for percentile calculation (p50, p95, p99)
- Dump metrics to `data/logs/performance/metrics_{date}.json` at configurable interval
- Metrics file reset at midnight ET (see Section 2.7)

---

### 4.16 Heartbeat Monitor (`src/monitoring/heartbeat_monitor.py`)

**Purpose**: Track streaming connection health via periodic metadata updates.

- Background thread updates `data/logs/heartbeat/stream_status.json` every 5 minutes (configurable)
- Tracks: `last_heartbeat`, `messages_received_last_5min`, `total_messages_today`, `connection_status`, `source`, `memory_usage_mb`
- Alert: If last heartbeat stale beyond configured threshold (default 900s), write alert to `data/logs/errors/`
- `total_messages_today` reset at 9:30 AM ET (see Section 2.7)

---

### 4.17 Schema Monitor (`src/monitoring/schema_monitor.py`)

**Purpose**: Detect schema drift in API responses. Log additions/removals once per day per source. "Already reported" flag resets at midnight ET (see Section 2.7).

---

### 4.18 Error Aggregator (`src/monitoring/error_aggregator.py`)

**Purpose**: Categorize and summarize errors. Counter dict keyed by `(source, error_type)`. Dumps summary every 15 min. Counters reset at midnight ET (see Section 2.7).

---

### 4.19 Signal Validator (`src/processing/signal_validator.py`)

**Purpose**: Compute derived metrics from `target_future_prices` on-the-fly and generate alerts when configurable thresholds are met. Eliminates need to store derived target fields in consolidated Parquet.

**Dual-Mode Operation**:

| Phase | target_future_prices Source | Derived Metrics |
|-------|----------------------------|-----------------|
| Training (Historical) | Computed from actual future prices | Used for model evaluation |
| Real-time (Production) | Predicted by ML model | Computed by validator for alerts |

**Derived Metrics (computed on-the-fly, NOT stored)**:
- `growth_pct[120]` — % change from current price at each future minute
- `max_growth_pct` — Maximum % growth in the prediction window
- `time_to_max_minutes` — Minutes until max growth reached
- `sustained_above_threshold` — Whether growth exceeds threshold for N consecutive minutes

**Alert Conditions** (all configurable via `settings.yaml`):
- Growth threshold: ≥20% (default)
- Confidence threshold: ≥90% of predicted minutes show growth above threshold
- Sustained window: 5 minutes (default)

**Alert Output Schema**:
```json
{
    "alert_id": "uuid",
    "timestamp": "2025-01-15T10:30:45-05:00",
    "option_symbol": "SPY250116C00700000",
    "contract_type": "CALL",
    "strike": 700.0,
    "current_price": 5.50,
    "predicted_max_growth_pct": 25.3,
    "time_to_max_minutes": 30,
    "confidence_pct": 94.2,
    "sustained_above_threshold": true,
    "prediction_window_minutes": 120,
    "alert_level": "HIGH"
}
```

**Alert Storage**: `data/logs/alerts/{date}/alerts.jsonl` (append-only, one JSON object per line)

**Alert Levels** (configurable):
```yaml
signal_validation:
  prediction_window_minutes: 120
  padding_strategy: "last_price"
  alerts:
    growth_threshold_pct: 20.0
    confidence_threshold_pct: 90.0
    sustained_window_minutes: 5
    alert_levels:
      HIGH: { growth_pct: 20.0, confidence_pct: 90.0 }
      MEDIUM: { growth_pct: 15.0, confidence_pct: 80.0 }
      LOW: { growth_pct: 10.0, confidence_pct: 70.0 }
```

**Implementation Approach**:
- Pure computation module — no storage dependency
- Accepts `target_future_prices[120]` array + current price as input
- Returns derived metrics dict + optional alert object
- Used by both historical evaluation pipeline and real-time inference pipeline
- All thresholds configurable via `settings.yaml` (zero hardcoded values)

**ML Model Architecture** (Phase 2 — sequence-to-sequence forecasting):
- Model type: LSTM or Transformer
- Input: ~115+ features at time T (consolidated record)
- Output: `target_future_prices[120]` — per-minute average price predictions
- Training data: Historical consolidated Parquet with actual future prices
- Inference: Model outputs predictions → Signal Validator computes derived metrics → alerts generated

---

## 5. Cross-Cutting Concerns

### 5.1 Security

**Credential Management**:
- Single API key: `POLYGON_API_KEY` in `.env` file (git-ignored)
- No `MASSIVE_API_KEY` needed (VIX from Polygon)
- Loaded via `python-dotenv` → `os.getenv()`
- YAML references: `${POLYGON_API_KEY}` resolved at config load time

**Secrets Redaction**:
- `RedactingFilter` class in `logger.py`
- Regex patterns match: `api_key`, `password`, `token`, `secret`, connection strings
- Applied to ALL log sinks (console + file)
- Error messages sanitized before logging

**Testing**: `test_security.py` — assert no raw API key appears in any log output.

### 5.2 Observability

**Logging Strategy**:
- Historical mode: Log per-date completion ("Fetched 23,400 SPY bars for 2025-10-28")
- Streaming mode: NO per-message logging. Heartbeat every 5 min only.
- Errors: Always logged with full context

**Alerting Thresholds** (all configurable via `settings.yaml` — see Section 4.15):
- Commit latency exceeds configured threshold
- Throughput drops below configured minimum
- Memory usage exceeds configured threshold
- Error rate exceeds configured percentage over configured window
- Heartbeat stale beyond configured threshold
- Schema drift detected (daily)

### 5.3 Maintainability

**Code Organization**: Follows existing project structure. `massive_client.py` converted to stub.

**Dependency Changes**: Remove `websocket-client`, `requests`, `urllib3` as direct dependencies for data sources (Polygon SDK handles internally). Remove SQL libraries. Add feature engineering libraries.

---

## 6. Trade-offs and Decisions

### Decision 1: All Data Sources via Polygon (No Massive API)

**Options Considered**:
1. **Polygon for SPY/Options/News + Massive for VIX (original)**: Two API providers, two keys, two rate limits
2. **Polygon for everything (selected)**: Single provider, single key, unified rate limiting

**Rationale**: Polygon supports VIX via `I:VIX` ticker in the indices market. Using a single provider eliminates a separate API dependency, simplifies authentication (one key), and enables unified rate limiting. No Massive SDK to build/maintain.

**Implications**:
- `massive_client.py` → stub
- Remove `MASSIVE_API_KEY` from `.env`
- Remove `websocket-client` from requirements (Polygon SDK handles WebSocket)
- VIX client is a thin wrapper using the same `polygon-api-client` SDK
- All rate limits are shared — addressed by unified token bucket in `ConnectionManager`

---

### Decision 2: Parquet-Only Storage (SQL Deferred to Post-Phase 6)

**Rationale**: At ~750K records/day, Parquet provides superior read performance for ML research. No database infrastructure needed. DuckDB available as interim SQL layer.

---

### Decision 3: Per-Contract Consolidated Records (vs Aggregated)

**Rationale**: ML model predicts future option prices per contract over 120 minutes — requires per-contract features (strike, delta, gamma, IV). Signal Validator then evaluates growth thresholds. Aggregating destroys signal.

---

### Decision 4: On-Demand Consolidation (Phase 1)

**Rationale**: Research phase needs flexibility to iterate on schema. Auto post-close consolidation in Phase 3.

---

### Decision 5: Synchronous Processing

**Rationale**: ~8.7 records/sec average. Synchronous with connection pooling is sufficient.

---

### Decision 6: Next-Day Options Only (No 0DTE)

**Options Considered**:
1. **Next-day only (selected)**: Single expiration cohort, cleaner data, lower volume
2. **Same-day (0DTE)**: Higher volume, complex theta dynamics
3. **Multiple expirations**: Even more contracts, rate limit pressure

**Rationale**: Focus on short-term movements with a single, clean cohort. ~30 contracts is manageable. 0DTE adds complexity (extreme theta decay near close) without clear Phase 1 benefit.

---

### Decision 7: Configurable Monitoring Thresholds

**Rationale**: Monitoring thresholds are inherently tunable. Runtime reload enables adjustment without restart.

---

### Decision 8: Unified Rate Limiting (Single Token Bucket)

**Options Considered**:
1. **Per-source rate limits (original)**: Each source gets independent budget — but they share one API key
2. **Unified token bucket (selected)**: Single shared budget reflecting actual Polygon API limits

**Rationale**: All four sources share one Polygon API key = one rate limit budget. A unified token bucket prevents one source from starving another. In historical mode (sequential), the full budget goes to the active source. In streaming mode, only news polling consumes REST budget.

---

### Decision 9: Single Target Variable (Storage Optimization)

**Removed fields** (5 fields, now computed on-the-fly by Signal Validator):
- `target_future_growth_pct[120]` → computed from `target_future_prices`
- `target_spike_30pct_120min` → computed by `signal_validator`
- `target_sustained_20pct` → computed by `signal_validator`
- `target_max_growth_120min` → computed by `signal_validator`
- `target_time_to_max_min` → computed by `signal_validator`

**Retained**: `target_future_prices[120]` — per-minute average option mid prices for next 120 minutes.

**Rationale**:
- Reduces storage per row (~1KB savings from removed arrays + scalar fields)
- Single source of truth (prices) — derived metrics always consistent
- Flexible thresholds (change 20% → 25% without re-consolidating historical data)
- Array size unchanged (120 elements) — values represent per-minute average prices
- Derived metrics are simple arithmetic (< 1ms per record at read time)

---

## 7. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Polygon rate limits throttle ingestion (shared key) | Medium | Medium | Unified token bucket; sequential source processing; paid tier upgrade |
| Polygon VIX data quality differs from Massive | Low | Low | Validate against known VIX values; same SDK handles parsing |
| Schema changes in Polygon API responses | Low | Medium | Schema drift detection + daily alerts; Pydantic permissive mode |
| WebSocket disconnects during trading hours | Medium | High | Auto-reconnect; heartbeat alerts; checkpoint on disconnect |
| Large historical backfill exceeds memory | Low | High | Generator-based processing; configurable batch size; per-date iteration |
| Parquet file corruption on crash | Low | Medium | Write to temp file, atomic rename; checkpoint tracks last write |
| Greeks calculation errors (deep ITM/OTM) | Medium | Low | Catch exceptions, set to NaN, log warning |
| Consolidation memory spike (~500MB) | Low | Medium | One date at a time; configurable; monitor via thresholds |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| API key exposed in logs | Low | Critical | Automatic redaction filter; security tests |
| Disk space exhaustion | Medium | Medium | ~255MB/day = ~7.6GB/month. Monitor; retention; compression. |
| Daily resets not triggering | Low | Medium | MarketHours callback + timestamp fallback; log reset events |
| Stale holiday calendar | Low | Low | Weekday + time check still applies as fallback |

---

## 8. Development Phases

### Phase A: Foundation (Steps 1-3)
1. **Config Loader** — YAML + .env, validation, env var substitution, reload
2. **Logger** — Loguru, redaction filter, file rotation
3. **Retry Handler** — Tenacity wrapper with YAML config

### Phase B: Data Sources (Steps 4-7)
4. **Polygon SPY Client** — Historical REST + streaming WebSocket
5. **Polygon Options Client** — Contract discovery (next-day only) + streaming
6. **Polygon VIX Client** — REST + WebSocket via `I:VIX` ticker
7. **News Client** — REST polling (5-min configurable interval)

### Phase C: Processing Pipeline (Steps 8-10)
8. **Validator** — Pydantic models for all 4 sources
9. **Deduplicator** — In-memory set, daily reset at 9:30 AM ET
10. **Late Data Handler** — Timestamp rejection logic

### Phase D: Storage (Step 11)
11. **Parquet Sink** — Date-partitioned writes for raw + consolidated

### Phase E: Orchestration (Steps 12-14)
12. **Historical Runner** — Batch orchestration with checkpoints
13. **Streaming Runner** — WebSocket orchestration with market hours
14. **Task Manager** — Run tracking, status, resume

### Phase F: Monitoring (Steps 15-18)
15. **Market Hours** — NYSE calendar, timezone, next trading day
16. **Heartbeat Monitor** — 5-min streaming health, daily reset
17. **Performance Monitor** — Configurable SLA tracking
18. **Schema Monitor + Error Aggregator** — Drift detection, error patterns, daily resets

### Phase G: Consolidation & Integration (Steps 19-22)
19. **Consolidator** — Multi-source join → per-contract Parquet with ~115+ features
20. **Signal Validator** — On-the-fly derived metrics + configurable alert generation
20. **Feature Engineering** — Greeks, momentum, volatility, technical indicators, targets
21. **Simulator** — Historical → real-time replay
22. **Connection Manager** — Unified rate limiting, pooling, reconnect

---

## 9. Testing Strategy

### Unit Testing
- **Framework**: `pytest`
- **Coverage Target**: >80%
- **Key Tests**:
  - Config loading with missing/invalid values + reload
  - Pydantic validation for each source schema
  - Deduplication (insert, duplicate, daily reset trigger)
  - Retry behavior (mock failing then succeeding calls)
  - Market hours boundaries (weekday/weekend, holidays, next trading day calculation)
  - Credential redaction (assert no raw key in logs)
  - Parquet write/read round-trip
  - Performance monitor with various config values + missing config defaults
  - Greeks edge cases (ATM, deep ITM, deep OTM)
  - Momentum arrays at market open boundary
  - Target variable end-of-day boundary
  - Unified rate limiter (token bucket exhaustion, refill)
  - Options expiration: Mon→Tue, Fri→Mon, holiday skip

### Integration Testing
- **Key Tests**:
  - Config → Source → Validate → Deduplicate → Parquet write → Read back
  - Consolidation: 4 raw sources → merge → features → consolidated Parquet → verify schema
  - Checkpoint save/restore
  - Config reload: change threshold, verify monitor picks up new value
  - VIX via Polygon: verify `I:VIX` ticker returns expected aggregate format

### Performance Testing
- **Target**: <60 seconds for single-day historical fetch + process + write
- **Consolidation**: <2 minutes per day
- **Memory**: Peak <500MB during consolidation

---

## 10. Deployment Considerations (Future — Not Phase 1)

- **Containerization**: Dockerfile (deferred)
- **SQL Queries on Parquet**: DuckDB embedded engine (interim, no code change)
- **Full SQL Sink**: Post-Phase 6 for BI requirements
- **Kafka**: Phase 3 for operational resilience (see Section 1.5)
- **Auto-consolidation**: Phase 3 — run within 10 min of market close

---

## 11. Dependency Changes (Phase 1)

### Remove from `requirements.txt`:
```
sqlalchemy>=2.0.0              # Deferred to post-Phase 6
psycopg2-binary>=2.9.0         # Deferred to post-Phase 6
pymysql>=1.1.0                 # Deferred to post-Phase 6
websocket-client>=1.6.0        # Not needed (Polygon SDK handles WebSocket)
aiohttp>=3.9.0                 # Not needed (synchronous processing)
```

### Add to `requirements.txt`:
```
# Feature Engineering & ML
py_vollib>=1.0.1               # Options Greeks (Black-Scholes)
pandas-ta>=0.3.14b             # Technical indicators (RSI, EMA, MACD)
sentence-transformers>=2.2.0   # News topic embeddings (optional)
scipy>=1.10.0                  # Statistical functions (skewness, kurtosis)
scikit-learn>=1.3.0            # Preprocessing, scaling
```

### Remove from `.env`:
```
MASSIVE_API_KEY=...            # Not needed (VIX from Polygon)
```

### Module Status:
- `src/data_sources/polygon_client.py` — **ACTIVE**: SPY + Options + VIX (all via Polygon)
- `src/data_sources/massive_client.py` — **STUB**: `NotImplementedError`, deprecated
- `src/data_sources/news_client.py` — **ACTIVE**: Polygon News API
- `src/sinks/parquet_sink.py` — **ACTIVE**: Raw + consolidated writes
- `src/sinks/sql_sink.py` — **STUB**: `NotImplementedError`, post-Phase 6
- `src/sinks/cloud_sink.py` — **STUB**: `NotImplementedError`, future
- `src/processing/consolidator.py` — **ACTIVE**: ~115+ feature consolidated Parquet (single target variable)
- `src/processing/signal_validator.py` — **ACTIVE**: On-the-fly derived metric computation + alert generation

---

## 12. Open Questions — RESOLVED

| # | Question | Resolution |
|---|----------|------------|
| 1 | VIX data source? | **Polygon API** (`I:VIX` ticker). Not Massive. SCOPE.md had incorrect API docs. |
| 2 | Options expiration? | **Next-day only**. No 0DTE. Use `MarketHours.next_trading_day()` for date calc. |
| 3 | News polling frequency? | **5 minutes** (configurable via `polygon.news.poll_interval_seconds`). |
| 4 | Historical backfill range? | **Configurable** via settings as number of trading days. Default: configurable in YAML. |
| 5 | Consolidation timing? | **On-demand** (Phase 1 CLI). Auto post-close in Phase 3. |

**Configuration for historical backfill**:
```yaml
historical:
  backfill:
    trading_days: 30           # Number of trading days to backfill (configurable)
    # Alternatively, explicit date range:
    # start_date: "2025-01-01"
    # end_date: "2025-01-31"
```

---

## 13. Approval Required

**This architecture must be reviewed and approved before proceeding to implementation.**

### Summary of Key Choices:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API Provider | Polygon.io for ALL sources (SPY, Options, VIX, News) | Single key, unified rate limiting, no Massive dependency |
| Rate Limiting | Unified token bucket (shared across sources) | All sources share one API key = one rate limit budget |
| Storage | Parquet-only (SQL stub for post-Phase 6) | ~750K records/day; 10-100x faster analytics; no DB infra |
| Consolidated schema | Per-second, per-contract (702K rows/day, ~115+ features) | ML needs contract-level features; single target variable |
| Target variable | Single `target_future_prices[120]` (per-minute avg prices) | Derived metrics computed on-the-fly by Signal Validator |
| Signal validation | Configurable alerts (growth %, confidence, sustained window) | Thresholds tunable without re-consolidating historical data |
| Options expiration | Next-day only (no 0DTE) | Single clean cohort, lower volume, simpler analysis |
| Consolidation trigger | On-demand CLI (Phase 1), auto post-close (Phase 3) | Research flexibility now, automation later |
| Architecture coupling | Tightly coupled, single process | Sufficient for volume; validate before adding infra |
| Future decoupling | Kafka in Phase 3 for operational resilience | Fault isolation, replay — NOT for volume |
| Distributed processing | No Spark (any phase) | ~8.7 records/sec; Python sufficient |
| Processing model | Synchronous | Sufficient; simpler debugging |
| Parquet engine | PyArrow with Snappy | Fastest, most feature-complete |
| Validation | Pydantic v2 | Fast (Rust core), declarative, schema drift detection |
| Monitoring thresholds | All configurable via settings.yaml | Runtime tunable, sensible defaults, zero hardcoded |
| Daily resets | 9:30 AM ET (market open) for operational; midnight for metrics | Explicit schedule, callback + timestamp fallback |
| News polling | 5-min interval (configurable) | Balance between freshness and rate limit budget |
| Historical backfill | Configurable as trading days | Flexible for different backfill needs |

---

**APPROVED**: Architecture approved on 2026-02-13. Proceeding to implementation.
