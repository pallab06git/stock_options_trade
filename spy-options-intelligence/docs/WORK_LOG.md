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
| 26 | 2026-02-20 | `6eeb5e5`, `2d6fde0` | Massive.com options download pipeline — OptionsTickerBuilder, ContractSelector, MassiveOptionsDownloader rewrite, `download-massive-options` CLI, massive package, sources.yaml fix | 116 |
| 27 | 2026-02-20 | `d7d2b28` | OptionsScanner summary metrics — 8 printed metrics (contract-days, bars, events/cday, >20% mins, rate, duration, hour dist), `_last_scan_stats`, single Parquet read via `_df` param | 9 |
| 28 | 2026-02-20 | — | Full-year data collection — 241 SPY days (189,742 bars), 68 options dates (Massive ~3-month history limit), per-date parallel watcher; full-year scan: 544 events / 125 cdays / 55.43% rate | — |

**Total**: 774 tests passing + 7 live tests (skipped outside market hours)

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

### Full-Year Data Collection (Step 28)
- **Massive free tier history limit**: ~3 months. Apr–Nov 2025 options data returns empty; Mar 2025 (downloaded fresh) and Dec 2025–Feb 2026 are available. Full 12-month coverage requires a Massive plan upgrade.
- **Per-date parallel watcher**: The sequential "download all SPY → download all options" design is suboptimal. Options only needs the SPY open price (first bar) to compute ATM strikes. Implemented a file watcher that triggers options download the moment each SPY date lands on disk, giving effective per-day parallelism within Polygon's rate-limit wait windows.
- **Architectural note**: Right design is a unified `download-day` command: SPY API call for day N (1–2s), then options for day N (during the 10s rate-limit backoff), then SPY day N+1. This eliminates two full sequential passes.
- **Full-year scan results**: 125 contract-days / 44,971 bars / 544 events / 55.43% positive-minute rate / median 8.5 min / mean 45.8 min above +20%. Stats consistent with March-only sample (4.0 median events/cday, ~55% rate).

### OptionsScanner Metrics (Step 27)
- `_last_scan_stats` populated by `scan()` — passed to `generate_report()` without re-reading files. Avoids double Parquet reads by forwarding the already-loaded DataFrame via optional `_df` parameter to `_scan_single`.
- Events-per-contract-day includes zero-event days in the min/median/max calculation: `n_zero = contract_days - len(per_cday_series)` zeros appended before `np.median`.
- Hour distribution uses `pd.to_numeric(..., errors='coerce')` for robustness against empty `trigger_time_et` strings.

### Massive.com Options Pipeline (Step 26)
- `OptionsTickerBuilder` is pure math — no config, no I/O, all `@staticmethod`. Strike computation uses `math.ceil / math.floor` with an edge-case guard when opening lands exactly on a boundary (calls must always be strictly above opening).
- `ContractSelector` uses an injectable `_input_fn` (replaces `input()`) so tests never need stdin. `prompt_once()` is guarded against being called twice; `get_contracts()` auto-calls it if `_test_params` is None.
- `MassiveOptionsDownloader` has zero ticker-construction logic — all strike/expiry decisions are delegated to `ContractSelector`. The `ThreadPoolExecutor` isolates per-contract failures (one bad contract doesn't block others).
- `sources.yaml` had `api_key: ${MASSIVE_API_KEY}` which caused `ConfigError` at load time if the env var wasn't set. Fixed to `api_key: ""` — the downloader's own four-source fallback chain then picks up `POLYGON_API_KEY` from `.env`.
- Live result: 21 trading days, 42 contracts, 3,541 minute bars downloaded in 26 seconds (free tier, 1 USD increment).

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
| Options (Polygon) | — | WebSocket (`Market.Options`) | `discover`, `stream-options` |
| Options (Massive) | `list_aggs()` — works on free tier | — | `download-massive-options` |
| VIX Index | REST (`I:VIX`) | WebSocket (`Market.Indices`) | `backfill-vix`, `stream-vix` |
| News | REST (`list_ticker_news`) | Polling (configurable interval) | `backfill-news`, `stream-news` |

---

## Test Coverage by Module

| Module | Tests | Key Areas |
|--------|-------|-----------|
| OptionsTickerBuilder | 43 | build_ticker format, compute_strikes (boundary, fractional), next_trading_day, next_friday |
| ContractSelector | 38 | TEST/PROD modes, prompt_once, _resolve_expiry conventions, get_contracts schema, auto-prompt |
| MassiveOptionsDownloader | 35 | constructor, get_opening_price, _fetch_bars, _download_single, download_tickers, run, from_config |
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
