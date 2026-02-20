# Work Log

Implementation history for SPY Options Intelligence Platform — Phase 1.

---

## Timeline

| Step | Date | Commit | Description | Tests |
|------|------|--------|-------------|-------|
| 0 | 2026-02-13 | `c94717d` | Architecture proposal and approval (Rev 6) | — |
| 1 | 2026-02-13 | `77ba01d` | Configuration system — YAML loading, env var substitution, validation | 15 |
| 2 | 2026-02-13 | `5842203` | Logging infrastructure — loguru, credential redaction, heartbeat integration | 14 |
| 3 | 2026-02-13 | `65bbe31` | Retry handler and connection manager — token-bucket rate limiting | 32 |
| 4 | 2026-02-13 | `c1bff24` | Polygon SPY client — historical minute-level data via REST | 18 |
| 5 | 2026-02-13 | `2b1bdc3` | Parquet storage sink — date partitioning, Snappy compression | 14 |
| 6 | 2026-02-13 | `6cc40da` | Data validation and deduplication modules | 23 |
| 7 | 2026-02-14 | `581470c` | Historical runner — CLI, checkpoint/resume, integration test | 28 |
| 8 | 2026-02-14 | `fcde8a0` | Performance monitoring and error aggregation | 100 |
| 9 | 2026-02-14 | `0316195` | Multi-ticker parallel execution — ParallelRunner, ProcessManager, HealthDashboard | 103 |
| 10 | — | — | Unit test suite (accumulated across Steps 1–9) | — |
| 11 | 2026-02-14 | `7a40bc4` | Real-time WebSocket streaming with market hours enforcement | 16 |
| 12 | 2026-02-14 | `c5c3ef7` | Options contract discovery — ±1% strike range, JSON persistence | 14 |
| 13 | 2026-02-14 | `1a82188` | Options streaming — compound deduplication (ticker + timestamp) | 13 |
| 14 | 2026-02-14 | `d31e9eb` | VIX data ingestion — historical REST + real-time WebSocket | 17 |
| 15 | 2026-02-14 | `4c5ebff` | News data ingestion — backfill + polling-based streaming | 20 |
| 16 | 2026-02-14 | `80c6d5c` | Data consolidation — per-option-per-minute flat schema, Greeks, indicators | 59 |
| 17 | 2026-02-14 | `64b88b9` | Schema drift detection — baseline persistence, configurable alerts | 20 |
| 18 | — | — | Late data handler — deferred to Phase 2 (Kafka + Spark watermarking) | — |
| 19 | 2026-02-14 | `88a7326` | Feed simulator — replay historical data as real-time stream | 24 |
| 20 | 2026-02-14 | `9d21cc8` | Integration tests — historical, real-time, and full pipeline flows | 24 |
| 21 | 2026-02-14 | — | Documentation — README, API reference, example configs, work log | — |
| 22 | 2026-02-18 | `f065057` | Data purge manager and memory leak fixes — LRU dedup, throughput pruning, vectorized Greeks | 21 |
| 23 | 2026-02-18 | `12dbc17` | Feature engineering and analysis rebuild — MinuteDownloader, FeatureEngineer, OptionsScanner, Streamlit dashboard | 70 |
| 24 | 2026-02-19 | `d98f679` | Retry handler refinements — exponential backoff (5xx + 429), auth log+skip, SkippableError | 9 |
| 25 | 2026-02-19 | `12dbc17` | Options strike selection fix — `_compute_strikes()` with math.ceil/floor, $0.50 increment, 6 new tests | 6 |

**Total**: 647 tests passing + 7 live tests (skipped outside market hours)

---

## Key Decisions

### Architecture (Step 0)
- Generator-based data flow for memory efficiency
- Dependency injection in runners for source/sink flexibility
- Token-bucket rate limiter shared across all sources
- Checkpoint/resume via JSON files (not database)

### Multi-Ticker (Step 9)
- Renamed `PolygonSPYClient` → `PolygonEquityClient` with ticker parameter
- Per-session monitoring (`session_label`) for parallel workers
- Subprocess-based parallelism (not threading) for isolation

### Consolidation (Step 16)
- Per-option-per-minute flat schema (no nested lists in Parquet)
- Greeks computed per-row as scalars (delta, gamma, theta, vega, rho, IV)
- `TrainingDataPrep` separated from `Consolidator` — target prices are offline ML prep only
- `merge_asof` for VIX and news time alignment

### Options Strike Selection (Step 25)
- SPY options use $0.50 strike increments — a ±5% discovery range was far too wide and returned irrelevant contracts
- `_compute_strikes(opening)` uses `math.ceil(opening / increment) * increment` for the first call strike and `math.floor(opening / increment) * increment` for the first put strike
- Edge case: when opening lands exactly on a strike boundary, calls start one increment above (so calls are always strictly above opening)
- Returned contracts matched with 1-cent tolerance (`abs(strike - target) < 0.01`) for floating-point safety
- Live test confirmed: opening=593.88 → calls [594.0, 594.5], puts [593.5, 593.0] — logic correct; API returns empty due to free-tier limitation

### Retry Behaviour (Step 24)
- All retried errors (5xx + 429) use exponential backoff: `initial_wait * base^(attempt-1)`, capped at `max_wait`
- Auth failures (401, 403): log WARNING + return `None` immediately — never retry to prevent account lockout
- `SkippableError`: new exception for data quality issues and schema drift — log WARNING + return `None`, no retry
- `with_retry` restructured from raw tenacity decorator to an outer wrapper that intercepts these two skip categories

### Deferred (Step 18)
- Late data handling deferred to Phase 2 — will use Kafka + Spark watermarking
- Phase 1 rejects data older than current day during streaming

---

## Data Sources Implemented

| Source | Historical | Real-time | CLI Commands |
|--------|-----------|-----------|--------------|
| SPY Equity | REST (`get_aggs`) | WebSocket (`Feed.Delayed`) | `backfill`, `stream` |
| Options | — | WebSocket (`Market.Options`) | `discover`, `stream-options` |
| VIX Index | REST (`I:VIX`) | WebSocket (`Market.Indices`) | `backfill-vix`, `stream-vix` |
| News | REST (`list_ticker_news`) | Polling (configurable interval) | `backfill-news`, `stream-news` |

---

## Test Coverage by Module

| Module | Tests | Key Areas |
|--------|-------|-----------|
| Config loader | 15 | YAML merge, env vars, validation |
| Logger | 14 | Redaction, file rotation, heartbeat |
| Retry + Connection | 41 | Exponential backoff (5xx+429), auth log+skip, SkippableError, jitter, rate limit |
| Polygon client | 19 | REST fetch, pagination, transform |
| Parquet sink | 14 | Write, dedup, partition, compression |
| Validator | 21 | Schema validation, batch split, error messages |
| Deduplicator | 2 | Key tracking, batch dedup |
| Historical runner | 32 | Pipeline, checkpoint/resume, date iteration |
| Performance monitor | 70 | Latency, throughput, stale ops, memory, alerts |
| Error aggregator | 30 | Rate calc, windowing, alert threshold |
| Parallel runner | 12 | Subprocess spawn, registry, results |
| Process manager | 8 | List/stop workers, SIGTERM |
| Health dashboard | 11 | Session aggregation, table format |
| Streaming runner | 10 | Buffer, flush, heartbeat, market close |
| Options client | 20 | Discovery, exact strike computation (math.ceil/floor), save/load |
| Options streaming | 7 | Compound dedup, contract loading |
| VIX client | 13 | Historical, streaming, transform |
| News client | 18 | Backfill, polling, sentiment extraction |
| Consolidator | 39 | Aggregation, indicators, Greeks, alignment |
| Training data prep | 20 | Target prices, coverage filter, date iteration |
| Schema monitor | 20 | Baseline, drift detection, alerts, auto-update |
| Feed simulator | 24 | Load, replay, speed, delay cap, stop event |
| Integration tests | 24 | Historical flow, real-time flow, full pipeline |
