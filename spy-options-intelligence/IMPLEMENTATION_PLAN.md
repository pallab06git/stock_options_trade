# Implementation Plan - Phase 1

## Completed
✅ Architecture approved (Rev 6)
✅ Project structure created
✅ Base classes (BaseSource, BaseSink)
✅ Utilities (market_hours.py, heartbeat_monitor.py)

## Step 1: Configuration System ✅
- [x] Implement src/utils/config_loader.py
- [x] Create config/settings.yaml
- [x] Create config/sources.yaml
- [x] Create config/sinks.yaml
- [x] Create config/retry_policy.yaml
- [x] Unit tests (15 tests)

## Step 2: Logging Infrastructure ✅
- [x] Implement src/utils/logger.py
- [x] Security: credential redaction
- [x] Heartbeat integration
- [x] Unit tests (14 tests)

## Step 3: Retry & Connection Management ✅
- [x] Implement src/utils/retry_handler.py
- [x] Implement src/utils/connection_manager.py
- [x] Unit tests (32 tests)

## Step 4: Polygon Client - SPY Historical ✅
- [x] Implement src/data_sources/polygon_client.py
- [x] fetch_historical() for SPY
- [x] Pagination handling
- [x] Unit tests with mocks (18 tests)

## Step 5: Parquet Storage ✅
- [x] Implement src/sinks/parquet_sink.py
- [x] Date partitioning
- [x] Compression (Snappy)
- [x] Unit tests (14 tests)

## Step 6: Data Validation ✅
- [x] Implement src/processing/validator.py
- [x] Implement src/processing/deduplicator.py
- [x] Schema validation
- [x] Unit tests (23 tests)

## Step 7: Historical Runner ✅
- [x] Implement src/orchestrator/historical_runner.py
- [x] Batch pipeline: Polygon → Validator → Deduplicator → Parquet
- [x] Date-by-date iteration with configurable batch_size
- [x] Checkpoint/resume (JSON-based, skips completed dates)
- [x] CLI with click (src/cli.py — backfill command)
- [x] Unit tests (28 tests)
- [x] Integration test (1 day SPY data — 930 records, checkpoint/resume verified)

## Step 8: Performance Monitoring ✅
- [x] Implement src/monitoring/performance_monitor.py
- [x] Implement src/monitoring/error_aggregator.py
- [x] Configurable thresholds (from config/settings.yaml monitoring.performance)
- [x] Stale/hung operation detection (check_stale_operations)
- [x] Unit tests (100 tests — overrun, stale, no-data, degradation, burst/recovery, read+write, session_label)

## Step 9: Multi-Ticker & Parallel Execution ✅
- [x] Sub-Step 1: Config — polygon.equities ticker list + orchestrator config
- [x] Sub-Step 2: Rename PolygonSPYClient → PolygonEquityClient (ticker param, config fallback, alias)
- [x] Sub-Step 3: Generalize Validator — "equity" schema + for_equity() factory
- [x] Sub-Step 4: Refactor HistoricalRunner — ticker param, ticker-scoped checkpoints
- [x] Sub-Step 5: CLI — --ticker and --rate-limit options for backfill
- [x] Sub-Step 6: Per-session monitoring — session_label in PerformanceMonitor & ErrorAggregator
- [x] Sub-Step 7: ParallelRunner — subprocess spawning, process registry, backfill-all CLI
- [x] Sub-Step 8: ProcessManager — workers list/stop CLI commands
- [x] Sub-Step 9: HealthDashboard — unified health/metrics view, health CLI command
- [x] Sub-Steps 10-12: Backward compat verification, integration test, docs update
- [x] Unit tests (19 polygon + 21 validator + 32 runner + 12 parallel + 8 process_mgr + 11 health = 103 new/updated)
- [x] Integration test: multi-ticker parallel (5 tests)

## Step 10: Unit Test Suite
- [x] tests/unit/test_config_loader.py
- [x] tests/unit/test_polygon_client.py
- [x] tests/unit/test_parquet_sink.py
- [x] tests/unit/test_parallel_runner.py
- [x] tests/unit/test_process_manager.py
- [x] tests/unit/test_health_dashboard.py
- [ ] Coverage >80%

## Step 11: Real-time SPY Streaming ✅
- [x] Fix MarketHours timedelta import bug
- [x] Add get_ws_client() to ConnectionManager (WebSocket factory)
- [x] Implement stream_realtime() in PolygonEquityClient (Thread+Queue bridge)
- [x] Implement StreamingRunner (WebSocket → buffer → validate → dedup → Parquet)
- [x] Add stream CLI command (--ticker, --config-dir)
- [x] Market hours enforcement (before + during streaming)
- [x] Heartbeat monitoring integration (5-min heartbeat, 15-min stall alert)
- [x] Signal handlers for graceful shutdown (SIGTERM/SIGINT)
- [x] Unit tests: 2 connection_manager + 4 polygon_client + 10 streaming_runner = 16 new
- [x] Integration tests: 4 streaming flow tests (pipeline, heartbeat, stats, market close)

## Step 12: Options Discovery ✅
- [x] Implement src/data_sources/polygon_options_client.py (PolygonOptionsClient)
- [x] fetch_opening_price() — get_daily_open_close() with rate limiting and retry
- [x] discover_contracts() — list_options_contracts() with ±1% strike range, expiration lookahead
- [x] save_contracts() / load_contracts() — JSON persistence in data/raw/options/contracts/
- [x] _transform_contract() — standardized field mapping
- [x] CLI discover command (--date, --config-dir)
- [x] Unit tests (14 tests)

## Step 13: Options Streaming ✅
- [x] Add stream_realtime() and _transform_options_agg() to PolygonOptionsClient
- [x] Add dedup_subset param to ParquetSink for compound dedup (ticker + timestamp)
- [x] Implement OptionsStreamingRunner (WebSocket → buffer → validate → compound dedup → Parquet)
- [x] Add stream-options CLI command (--date, --config-dir)
- [x] Unit tests: 4 client streaming + 7 runner + 2 parquet dedup = 13 new

## Step 14: VIX Data ✅
- [x] Implement src/data_sources/polygon_vix_client.py (PolygonVIXClient)
- [x] Historical fetch via REST (I:VIX, date-by-date, transform with source="vix")
- [x] Real-time streaming via WebSocket (Market.Indices, "A.I:VIX" subscription)
- [x] Dependency injection in HistoricalRunner (connection_manager, client, validator)
- [x] Dependency injection in StreamingRunner (connection_manager, client, validator)
- [x] CLI: backfill-vix command (--start-date, --end-date, --resume)
- [x] CLI: stream-vix command (--config-dir)
- [x] Unit tests: 13 VIX client + 2 historical DI + 2 streaming DI = 17 new

## Step 15: News Data ✅
- [x] Implement src/data_sources/news_client.py (PolygonNewsClient)
- [x] Historical backfill via list_ticker_news() REST API (date-by-date)
- [x] Polling-based "streaming" (configurable interval, queue-based bridge)
- [x] Sentiment extraction from Polygon insights (ticker-matched)
- [x] ISO 8601 published_utc → Unix ms timestamp conversion
- [x] Deduplicator DI in HistoricalRunner (optional deduplicator param)
- [x] Deduplicator DI in StreamingRunner (optional deduplicator param)
- [x] CLI: backfill-news command (--start-date, --end-date, --resume)
- [x] CLI: stream-news command (--config-dir)
- [x] Unit tests: 18 news client + 1 historical DI + 1 streaming DI = 20 new
- [x] Integration tests: 11 live Polygon tests (connection, schema, validator, dedup, full pipeline)

## Step 16: Data Consolidation ✅ (Restructured: per-option-per-minute flat schema)
- [x] Install dependencies (py_vollib, ta, scipy)
- [x] Add consolidation + signal_validation config to settings.yaml
- [x] Restructure src/processing/consolidator.py — per-option-per-minute flat schema
- [x] Per-minute aggregation (SPY OHLCV+VWAP, VIX OHLC, Options per-ticker OHLCV+avg)
- [x] Time alignment (VIX → SPY minute grid via merge_asof forward-fill)
- [x] Compute technical indicators on 1-min SPY (RSI, MACD, Bollinger Bands)
- [x] Compute momentum on 1-min SPY (price_change + ROC for windows [5, 30, 60])
- [x] Attach news sentiment (merge_asof with lookback tolerance)
- [x] Flatten to one row per option per minute (inner join options × SPY)
- [x] Compute Greeks per-row as flat scalars (delta, gamma, theta, vega, rho, IV)
- [x] Separate target_future_prices into TrainingDataPrep module (offline ML training only)
- [x] CLI consolidate command with stats (minutes, unique_options)
- [x] CLI prepare-training command (--start-date, --end-date)
- [x] src/processing/training_data_prep.py — offline training data generation
  - Reads consolidated Parquet, adds target_future_prices (120-min lookahead)
  - Filters by min_target_coverage_pct (configurable, default 50%)
  - Writes to data/processed/training/
- [x] Unit tests: 39 consolidator + 20 training_data_prep = 59 tests
- [x] Integration test (1 full pipeline — flat schema, no list columns, no target in consolidator)

## Step 17: Schema Drift Detection ✅
- [x] Implement src/monitoring/schema_monitor.py (SchemaMonitor class)
- [x] Schema extraction via pyarrow.parquet.read_schema() (metadata-only, no data loading)
- [x] Baseline persistence (JSON in data/logs/schema/)
- [x] Drift detection: new columns, missing columns, type changes
- [x] Configurable alert toggles (alert_on_new_columns, alert_on_missing_columns, alert_on_type_changes)
- [x] Auto-update baseline option (auto_update_baseline)
- [x] Drift event logging (data/logs/schema/drift/)
- [x] Config: monitoring.schema section in settings.yaml
- [x] CLI: schema-check command (--source, --date)
- [x] CLI: schema-baseline command (--source, --date)
- [x] Export SchemaMonitor from src/monitoring/__init__.py
- [x] Unit tests (20 tests — init, capture, diff, alerts, check_drift, save/load, log, auto-update)

## Step 18: Late Data Handler — DEFERRED to Phase 2
- [ ] Deferred: will use Kafka + Spark watermarking in Phase 2

## Step 19: Feed Simulator ✅
- [x] Implement src/orchestrator/simulator.py (FeedSimulator class)
- [x] Replay historical Parquet data as simulated real-time stream
- [x] Configurable playback speed (1x real-time, Nx faster, 0 = no delay)
- [x] Per-record delay based on timestamp gaps, capped at 5s per gap
- [x] All sources supported (spy, vix, options, news, consolidated)
- [x] BaseSource-compatible stream_realtime() interface for StreamingRunner DI
- [x] stop_event support for graceful interruption
- [x] Config: simulator section in settings.yaml (speed_multiplier, max_delay_per_gap_sec)
- [x] CLI: simulate command (--source, --date, --speed)
- [x] Unit tests (24 tests — init, load, streaming, stop_event, delay cap, stubs, stats, source dirs)

## Step 20: Integration Tests ✅
- [x] tests/integration/test_historical_flow.py (10 tests)
  - TestSPYHistoricalFlow: multi-day backfill, checkpoint/resume, deduplication, invalid filtering (5)
  - TestVIXHistoricalFlow: VIX backfill with validator (1)
  - TestNewsHistoricalFlow: news backfill, article_id dedup within batch (2)
  - TestParquetOutput: sorted timestamps, no duplicate timestamps (2)
- [x] tests/integration/test_realtime_flow.py (9 tests)
  - TestSimulatorStandalone: SPY/VIX/news replay, stop_event, timestamp ordering (5)
  - TestSimulatorWithStreamingRunner: full pipeline, duplicate handling, invalid filtering, batch flushing (4)
- [x] tests/integration/test_full_pipeline.py (5 tests)
  - TestFullPipeline: end-to-end ingest+consolidate, idempotent consolidation (2)
  - TestSchemaMonitorOnPipeline: baseline capture, no-drift, drift detection (3)
- [x] All mock-based — no live API calls

## Step 21: Documentation ✅
- [x] Rewrite README.md — CLI reference (16 commands), data schemas (5), project structure, monitoring, config
- [x] Create docs/API_REFERENCE.md — module-by-module class reference (27 classes)
- [x] Create docs/WORK_LOG.md — implementation history (21 steps with dates, commits, test counts)
- [x] Create config/examples/ — 3 annotated YAML configs (backfill_only, full_pipeline, streaming)
- [x] Full test suite verification (543 passed, 7 skipped)

## Step 22: Data Purge Manager + Memory Leak Fixes ✅
- [x] Add retention + processing config sections to settings.yaml
- [x] Add max_error_types to monitoring.performance config
- [x] Create src/utils/purge_manager.py (PurgeManager class)
  - Per-category retention (raw_data, processed_data, performance_metrics, schema_drift, checkpoints, heartbeat)
  - Category-to-path mapping with file pattern filters (e.g. checkpoint_*.json)
  - Dry-run mode, graceful error handling, summary reporting
- [x] Add purge CLI command (--category, --retention-days, --dry-run/--no-dry-run)
- [x] Export PurgeManager from src/utils/__init__.py
- [x] Fix Deduplicator unbounded _seen set → OrderedDict with LRU eviction (max_size param)
- [x] Update streaming runners to read max_size from config["processing"]["deduplication"]["max_size"]
- [x] Fix PerformanceMonitor unbounded _throughput → _prune_throughput() (1-hour window)
- [x] Fix ErrorAggregator unbounded error types → OrderedDict with max_error_types LRU eviction
- [x] Update streaming runners to pass max_error_types from config
- [x] Vectorize Consolidator _compute_greeks_flat (iterrows → df.apply with boolean mask)
- [x] Unit tests: 12 purge + 5 dedup LRU + 2 perf pruning + 2 error LRU = 21 new tests
- [x] Full test suite verification (564 passed, 7 skipped)

## Step 25: Options Strike Selection Fix ✅
- [x] Replace `discovery_range_pct: 0.05` (±5% wide range) with `strike_increment: 0.5` in `config/pipeline_v2.yaml`
- [x] Add `_compute_strikes()` to `TargetedOptionsDownloader` — mathematically computes exact target strikes:
      - Calls: n strikes immediately above opening via `math.ceil`
      - Puts:  n strikes immediately at or below opening via `math.floor`
      - Edge case: if opening lands exactly on a strike boundary, calls start one increment up
- [x] Rewrite `discover_targeted()` to query Polygon with tight range (only the 4 target strikes) instead of ±5%
- [x] Match returned contracts by exact strike with 1-cent tolerance (floating-point safe)
- [x] Add `TestComputeStrikes` (6 tests): fractional opening, exact boundary, strict above/below invariants, user example
- [x] Live test confirmed: strike logic correct (opening=593.88 → calls [594.0, 594.5], puts [593.5, 593.0])
- [x] Live test confirmed: options API returns empty — free tier limitation, not a code bug
- [x] Full test suite: 647 passing + 7 skipped

## Step 24: Retry Handler Refinements ✅
- [x] Exponential backoff for all retried errors (5xx + 429): `initial_wait * base^(attempt-1)`, capped at `max_wait`
- [x] Auth failures (401, 403): log WARNING + return None immediately — no retry, prevents account lockout
- [x] New `SkippableError` exception for data quality issues and schema drift: log WARNING + return None, no retry
- [x] `with_retry` restructured from raw tenacity decorator → outer wrapper that intercepts SkippableError and auth errors
- [x] Updated 2 existing tests, added 7 new tests (auth skip, SkippableError, backoff growth across all 5xx/429)
- [x] Full test suite: 641 passing + 7 skipped

## Step 23: Feature Engineering & Analysis Rebuild ✅
- [x] Add `streamlit>=1.30.0` to requirements.txt
- [x] Create `config/pipeline_v2.yaml` (date range, lag windows, options targeting, scanner, reporting config)
- [x] Create `src/data_sources/minute_downloader.py` (MinuteDownloader — SPY + VIX bulk month download)
- [x] Create `src/data_sources/targeted_options_downloader.py` (TargetedOptionsDownloader — 2 calls + 2 puts per day)
- [x] Create `src/processing/feature_engineer.py` (FeatureEngineer — lagged % change + IV features)
- [x] Create `src/processing/options_scanner.py` (OptionsScanner — 20%+ move detector with event CSV)
- [x] Create `src/utils/space_reporter.py` (SpaceReporter — storage size tree + compression estimates)
- [x] Create `src/utils/hardware_monitor.py` (HardwareMonitor — CPU/memory/disk tracking + decorator)
- [x] Create `src/reporting/__init__.py` + `src/reporting/dashboard.py` (3-tab Streamlit dashboard)
- [x] Add 7 new CLI commands to `src/cli.py`:
      download-minute, download-options-targeted, engineer-features,
      scan-options, report-space, report-hardware, dashboard
- [x] Unit tests: 68 new tests across 6 files
- [x] Full test suite: 632 passing + 7 skipped (live market hours tests)

## Step 26: Massive.com Options Download Pipeline ✅
- [x] Create `src/data_sources/options_ticker_builder.py` (OptionsTickerBuilder)
  - Pure math, all `@staticmethod`, no config/API/I/O
  - `build_ticker()` — formats `O:SPY250304C00601000`-style ticker string
  - `compute_strikes()` — n calls (strictly above opening) + n puts (at or below), boundary-safe
  - `next_trading_day()` — first Mon–Fri strictly after date, skips weekends
  - `next_friday()` — first Friday strictly after date
- [x] Create `src/data_sources/contract_selector.py` (ContractSelector)
  - TEST mode: prompts user once per cycle (underlying, increment, n_calls, n_puts, expiry convention); uses those params for every date
  - PROD mode: calls `massive.list_options_contracts()`, filters to nearest n_calls/n_puts strikes
  - Expiry conventions: `next_calendar_day`, `next_trading_day`, `next_friday`, `fixed`
  - Injectable `_input_fn` for testing (no real `input()` calls in tests)
  - Output schema: `{ticker, strike, contract_type, expiry_date, underlying}`
- [x] Rewrite `src/data_sources/massive_options_downloader.py` (MassiveOptionsDownloader)
  - Zero ticker-construction logic — receives contract list from `ContractSelector`
  - `get_opening_price(date)` — reads from local SPY Parquet, no API call
  - `download_tickers(contracts, date)` — `ThreadPoolExecutor` parallel `list_aggs()` calls
  - `run(start_date, end_date)` — full date loop; calls `selector.prompt_once()` once before tqdm loop
  - Resume support: skips contracts whose Parquet already exists
  - Output: `data/raw/options/minute/{safe_ticker}/{date}.parquet`
- [x] Add `download-massive-options` CLI command to `src/cli.py`
  - Flags: `--start-date`, `--end-date`, `--mode [test|prod]`, `--resume/--no-resume`
  - `--mode` overrides `pipeline_v2.contract_selector.mode` from config
- [x] Add `massive>=2.2.0` to `requirements.txt`; installed in project venv
- [x] Fix `config/sources.yaml` — replace `${MASSIVE_API_KEY}` with `""` so downloader's own fallback chain picks up `POLYGON_API_KEY`
- [x] Update `config/pipeline_v2.yaml` — add `contract_selector` section and `max_workers` to `massive_options`
- [x] Unit tests: 43 (OptionsTickerBuilder) + 38 (ContractSelector) + 35 (MassiveOptionsDownloader) = 116 new tests
- [x] Live test: 21 trading days of March 2025 downloaded — 42 contracts, 3,541 bars in 26 seconds
- [x] Full test suite: 763 passing + 7 skipped

## Step 27: OptionsScanner Summary Metrics ✅
- [x] Add `_last_scan_stats` dict to `OptionsScanner.__init__` to persist contract-days + total-bars across scan/report calls
- [x] Update `scan()` to load each Parquet once, count contract-days and bars, store in `_last_scan_stats`, pass pre-loaded DataFrame to `_scan_single` via optional `_df` param (avoids double reads)
- [x] Replace `generate_report()` console output with 8 required metrics:
  - Contract-days scanned + total minute bars
  - Total events
  - Events per contract-day: min / median / max (zeros included for no-event days)
  - Total >20% minutes, positive-minute rate (%)
  - Duration above 20%: median / mean
  - Event distribution by trigger hour (ET) — ASCII bar chart
- [x] 9 new unit tests (TestScanStats ×3, TestGenerateReportMetrics ×6)
- [x] Full test suite: 774 passing + 7 skipped

## Step 28: Full-Year Data Collection & Scan ✅
- [x] Download SPY minute bars for full year: Mar 2025 → Feb 2026
  - 241 trading days, 189,742 bars, ~10 MB raw
  - CLI: `download-minute --ticker SPY --start-date 2025-04-01 --end-date 2026-02-19 --resume`
- [x] Download options minute bars (Massive.com free tier)
  - 68 unique dates, 124 contract parquets, ~3 MB raw
  - Per-date parallel watcher: as each SPY date lands, immediately triggers options download for that date
  - ⚠ Massive free tier limitation confirmed: ~3 months of options history only
    Coverage: Mar 2025 + Dec 2025 → Feb 2026; Apr–Nov 2025 returns empty
- [x] Feature engineering: 239 SPY feature files + 125 options feature files
- [x] Full-year scan (Mar 2025 → Feb 2026):
  - 125 contract-days scanned, 44,971 minute bars
  - 544 events detected (20%+ intraday moves)
  - Events/contract-day: min=0, median=4.0, max=12
  - Total >20% minutes: 24,929 (55.43% positive-minute rate)
  - Duration above 20%: median 8.5 min / mean 45.8 min
  - Event peak hours: 09:xx–10:xx (morning) and 15:xx (gamma into close)
- [x] Architectural insight documented: per-day interleaved SPY+options download needed (options only requires SPY open price, not full day; rate-limit wait window should be used for parallel options fetch)

## Future
- [ ] Upgrade Massive plan for full 12-month options history (Apr–Nov 2025 gap)
- [ ] VIX data integration (upgrade massive.com plan)
- [ ] Per-day interleaved `download-day` command (SPY open → options, parallel within rate-limit window)
- [ ] LSTM model training
- [ ] Signal validator implementation
- [ ] MLflow integration
- [ ] Backtesting framework

---
**Total tests: 774 passing + 7 live (skipped outside market hours) | Last updated: 2026-02-20**
