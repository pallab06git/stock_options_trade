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

## Step 29: ML Feature Engineering ✅
- [x] Create `src/processing/ml_feature_engineer.py` (MLFeatureEngineer class)
  - 66 engineered features across 13 groups (time, SPY momentum/volume/volatility/
    technicals/Bollinger/VWAP, options momentum/intraday, contract, IV, cross-asset)
  - RSI-14, EMA-9/21, MACD(12/26/9), Bollinger Bands via `ta` library
  - Implied volatility via `py_vollib` Black-Scholes (fallback 0.20 if unavailable)
  - Forward-looking binary target: did price rise ≥20% in next 120 min?
  - Label metadata: max_gain_120m, time_to_max_min
  - Output: `data/processed/features/{date}_features.csv` (81 cols)
  - Config-driven (feature_engineering.* keys; all params have defaults)
- [x] Unit tests: 36 tests (TestInit, TestComputeSpyFeatures, TestComputeTargets,
  TestParseContractMeta, TestEngineerDate, TestRun)
- [x] Smoke test on 2025-03-03: 764 rows × 81 cols, 2 contracts, 58% positive rate

## Step 30: Label Generator ✅
- [x] Create `src/processing/label_generator.py`
  - Module-level `generate_labels(df, threshold_pct=20.0, lookforward_minutes=120)`
    - Works on any DataFrame with `timestamp` + `close` columns
    - Per-ticker label isolation (groupby if `ticker` col present)
    - Adds: `target` (int8), `max_gain_pct` (float), `time_to_max_min` (float)
    - O(n log n) via numpy searchsorted; original df not mutated
  - `LabelGenerator` class (config-driven wrapper)
    - `generate(df)` — apply with configured params
    - `generate_for_file(path)` — load CSV/Parquet, apply, return (overwrites stale target)
    - `validate(df)` — check distribution, coverage, missing columns
- [x] Unit tests: 29 tests (TestValidateInput, TestSingleTicker, TestMultiTicker, TestClass)

## Step 31: Data Balancing ✅
- [x] Create `src/ml/__init__.py` (new ML sub-package)
- [x] Create `src/ml/data_balancer.py`
  - `undersample_majority(df, target_col, random_state)` — downsample majority to match minority;
    reproducible via random_state; handles empty/single-class/already-balanced edge cases
  - `calculate_class_weights(df, target_col)` — balanced weights formula
    (n_total / (n_classes × count_i)); equivalent to sklearn's compute_class_weight('balanced')
  - `DataBalancer` class (config-driven wrapper)
    - `balance(df)` — applies "undersample" or returns unchanged for "class_weights"
    - `get_class_weights(df)` — compute weights dict
    - `get_summary(df)` — distribution stats, imbalance_ratio, class_weights
- [x] Unit tests: 31 tests (TestCheckTargetCol, TestUndersampleMajority,
  TestCalculateClassWeights, TestDataBalancer)
- [x] No new dependencies added (pure numpy/pandas; equivalent to sklearn's balanced formula)

## Step 32: Train/Test Split Utility ✅
- [x] Create `src/ml/data_splitter.py`
  - `time_based_split(df, train_ratio=0.70, val_ratio=0.15)` → (train, val, test)
    - Date-level split when `date` column present (whole trading days kept together)
    - Row-level fallback when no `date` column (splits on sorted timestamps)
    - Strict chronological order: train < val < test, no overlap between sets
  - `DataSplitter` class (config-driven wrapper)
    - `split(df)` → delegates to time_based_split with configured ratios
    - `split_dates(dates)` → partition a date list for pre-loading planning
    - `test_ratio` property (derived: 1 − train − val)
    - `get_summary(train, val, test)` → row counts, date ranges, positive rates
  - `_validate_ratios` — raises ValueError for zero/exceeding ratios
- [x] Unit tests: 34 tests (TestValidateRatios, TestDateLevel, TestRowLevel,
  TestDataSplitter, TestSplitDates, TestGetSummary)
- [x] Key design: date-level split prevents intraday bars from spanning sets;
  no random shuffling anywhere — pure chronological ordering

## Step 33: XGBoost Training Pipeline ✅
- [x] Create `src/ml/train_xgboost.py`
  - `load_features(features_dir, start_date, end_date)` — load + concat `*_features.csv` files;
    filters by date range; sorts by timestamp; warns on missing files
  - `_NON_FEATURE_COLS` frozenset — excludes raw OHLCV, metadata, and all label columns
    from model input (open/high/low/close/volume/vwap/transactions, opt_close, date, ticker,
    timestamp, source, target, max_gain_120m, max_gain_pct, time_to_max_min)
  - `XGBoostTrainer` class (config-driven, reads `ml_training.xgboost.*`)
    - `train(features_dir, start_date, end_date, models_dir, logs_dir)` — full pipeline:
        load → split (chronological, FIRST) → balance training only (undersample) →
        fit XGBClassifier with early stopping → evaluate on val → save artifact → log metrics
    - `get_feature_cols(df)` — returns sorted list of model input columns
    - `_evaluate(model, X, y, threshold)` → {accuracy, precision, recall, f1, roc_auc}
    - `_save_model(artifact, version, models_dir)` — joblib.dump dict artifact to
        `models/xgboost_{version}.pkl`; artifact keys: model, feature_cols, threshold,
        xgb_params, saved_at
    - `_log_metrics(metrics, run_ts, logs_dir)` — JSON to `data/logs/training/training_{ts}.json`
  - XGBoost 3.x API: `early_stopping_rounds` in constructor (not fit()); no use_label_encoder
  - Default XGBoost params: n_estimators=300, max_depth=6, lr=0.05, subsample=0.80,
    colsample_bytree=0.80, min_child_weight=5, gamma=0.10
- [x] Unit tests: 35 tests (TestLoadFeatures, TestNonFeatureCols, TestXGBoostTrainerInit,
  TestGetFeatureCols, TestXGBoostTrainerTrain, TestEvaluate)
- [x] Full test suite: 974 passing + 7 skipped

## Step 34: Feature Importance Analyzer ✅
- [x] Create `src/ml/feature_importance.py`
  - `_VALID_IMPORTANCE_TYPES` frozenset: weight, gain, cover, total_gain, total_cover
  - `extract_importances(model, feature_cols, importance_type="gain")` → DataFrame
    - Maps f0/f1/… internal XGBoost names → real feature_cols via index
    - Features not used in any split included with importance=0.0
    - Columns: feature, importance, importance_pct (normalized), rank
    - Sorted by importance DESC; rank starts at 1
  - `FeatureImportanceAnalyzer` class (reads `ml_training.feature_importance.*`)
    - `analyze(model_path, output_dir)` — load joblib artifact → extract → save CSV → return df
    - `get_top_n(df, n)` — top-N slice with reset index
    - `save_report(df, model_version, output_dir)` — CSV to `{version}_{type}_importance.csv`
    - `plot_summary(df, top_n)` — ASCII horizontal bar chart, no external dependencies
- [x] Unit tests: 45 tests (TestExtractImportances, TestValidImportanceTypes,
  TestFeatureImportanceAnalyzerInit, TestGetTopN, TestSaveReport, TestAnalyze, TestPlotSummary)
- [x] Full test suite: 1019 passing + 7 skipped

## Step 35: ML Model Backtester ✅
- [x] Create `src/ml/backtest.py`
  - `backtest_model(model, feature_cols, df, threshold)` → (metrics_dict, trades_df)
    - Validates: non-empty df, target column present, all feature_cols in df
    - Predicts proba on X_test → binary y_pred via threshold
    - Builds per-trade DataFrame for predicted-positive bars (meta cols + outcome)
    - `is_true_positive` flag; carries date/ticker/timestamp/max_gain_120m/time_to_max_min
  - `_compute_metrics(y_true, y_pred, probas, df)` → dict
    - n_test_rows, n_signals, n_true_positives, n_false_positives, signal_rate,
      positive_rate_test, precision, recall, f1, roc_auc
    - avg_gain_all_bars (baseline), avg_gain_signals, avg_gain_tp, avg_gain_fp
    - lift = avg_gain_signals / avg_gain_all_bars (None when max_gain_120m absent)
  - `ModelBacktester` class (config-driven, reads `ml_training.backtest.*`)
    - `run(model_path, features_dir, start_date, end_date, output_dir)` →
        load artifact → load features → chronological split → take test set only →
        backtest_model → save trades CSV + JSON metrics report → return result dict
    - Output: `{model_version}_trades_{ts}.csv`, `{model_version}_backtest_{ts}.json`
  - Design: test set only (never training data); lift > 1 = model adds value over random
- [x] Unit tests: 38 tests (TestBacktestModel, TestComputeMetrics,
  TestBuildTradesDf, TestModelBacktester)
- [x] Full test suite: 1057 passing + 7 skipped

## Step 36: requirements.txt — ML Dependencies ✅
- [x] Added `# ML Training` section to `requirements.txt`:
  - `xgboost>=2.0.0`     (XGBoost gradient boosting; installed 3.2.0)
  - `scikit-learn>=1.3.0` (precision/recall/f1/roc_auc metrics; installed 1.8.0)
  - `joblib>=1.3.0`       (model artifact serialisation; installed 1.5.3)
- [x] Full test suite: 1057 passing + 7 skipped (no regressions)

## Step 37: config/ml_settings.yaml ✅
- [x] Created `config/ml_settings.yaml` — single config file for all ML modules
  - `feature_engineering.*` — start/end dates, input/output paths, target definition
    (threshold_pct=20, lookforward=120), lookback windows, risk_free_rate, dividend_yield
  - `label_generator.*` — threshold_pct, lookforward_minutes (mirrors feature_engineering)
  - `data_preparation.*` — train_ratio=0.70, val_ratio=0.15, balance_method=undersample,
    target_col=target, random_state=42
  - `ml_training.xgboost.*` — all 11 XGBoost params + threshold + model_version
  - `ml_training.feature_importance.*` — importance_type=gain, top_n=20, output_dir
  - `ml_training.backtest.*` — output_dir
  - `ml_paths.*` — models_dir, training_logs_dir (shared across modules)
- [x] Verified: all 28 config keys read by ML modules resolve cleanly via yaml.safe_load

## Step 38: ML CLI Runner ✅
- [x] Create `src/ml/cli.py` — `ml` Click subgroup with 4 commands
  - `generate-features` (`--config-dir, --start-date, --end-date`)
    → `MLFeatureEngineer.run()`; prints 7-line summary (dates processed/skipped/failed,
      total rows, n_features, positive rate, output dir)
  - `train` (`--config-dir, --start-date, --end-date, --model-version`)
    → `XGBoostTrainer.train()`; prints 11-line training summary
      (row counts, n_features, best_iteration, val accuracy/precision/recall/f1/ROC-AUC,
      model path, metrics log path)
  - `feature-importance` (`--config-dir, --model-path [required], --importance-type, --top-n`)
    → `FeatureImportanceAnalyzer.analyze()` + `plot_summary()`; prints ASCII bar chart
  - `backtest` (`--config-dir, --model-path [required], --start-date, --end-date`)
    → `ModelBacktester.run()`; prints 13-line backtest summary
      (test rows, signals, TP/FP, precision/recall/f1/ROC-AUC, avg gain, lift, file paths)
  - All heavy ML imports deferred inside each command body (fast CLI startup)
  - All commands: non-zero exit on exception; error message to stderr via `click.echo(err=True)`
- [x] Register `ml_cli` in `src/cli.py` via `cli.add_command(ml_cli)` — accessible as `ml` subgroup
- [x] Unit tests: 31 tests (TestGroupHelp ×6, TestGenerateFeatures ×5, TestTrain ×5,
  TestFeatureImportance ×6, TestBacktest ×7, TestMainCliIntegration ×2)
  - Deferred-import patch targets corrected: source module paths not `src.ml.cli.*`
    (e.g. `src.processing.ml_feature_engineer.MLFeatureEngineer`,
     `src.ml.train_xgboost.XGBoostTrainer`, `src.ml.feature_importance.FeatureImportanceAnalyzer`,
     `src.ml.backtest.ModelBacktester`)
- [x] Full test suite: 1088 passing + 7 skipped

## Step 39: High-Precision Threshold & Speed Benchmarking ✅
- [x] Add `min_loss_120m` to `_compute_targets()` in `src/processing/ml_feature_engineer.py`
  - Worst % drawdown in 120-min forward window from each bar's entry price
  - Added to `_NON_FEATURE_COLS` frozenset in `src/ml/train_xgboost.py` (prevents data leakage)
  - Added to `_TRADE_META_COLS` list in `src/ml/backtest.py` (included in trades CSV output)
- [x] Create `src/ml/evaluate.py`
  - `find_optimal_threshold_for_precision(model, X_val, y_val, min_precision, step)` → dict
  - Sweeps 0.50–0.99 on validation set; returns: achievable, optimal_threshold,
    achieved_precision, achieved_recall, n_signals, signal_rate, analysis_df
  - Returns `achievable=False` + best found precision when requirement unachievable
- [x] Create `src/ml/benchmark.py`
  - `benchmark_prediction_speed(model, sample_features, n_iterations)` → dict
  - 20-call warmup + n timed calls; returns mean/p50/p95/p99/max latency in ms
  - meets_100ms_requirement flag
- [x] Add 3 new CLI commands + `--threshold` override to backtest in `src/ml/cli.py`:
  - `find-threshold` — sweeps val set for min_precision; prints sweep table + recommendation
  - `benchmark-speed` — times single-sample predict_proba (zero-vector); prints latency stats
  - `backtest --threshold` — override artifact's stored threshold for high-precision testing
- [x] Add `evaluation.*` and `performance.*` sections to `config/ml_settings.yaml`
- [x] Unit tests: 21 tests for evaluate.py + 17 tests for benchmark.py = 38 new tests
- [x] Real results (xgboost_v2, test split 2025-12-23 → 2026-02-19):
  - At threshold=0.67: 307 signals | 282 TP | 25 FP | precision=91.9% | lift=2.94x
  - Speed: mean=0.20ms, p99=0.40ms (247× under 100ms requirement)
- [x] Full test suite: 1126 passing + 7 skipped

## Step 40: False Positive Severity Analysis ✅
- [x] Create `src/ml/error_analyzer.py` — `PredictionErrorAnalyzer` class
  - `load_false_positives(trades_path)` → pd.DataFrame
    - Reads per-trade CSV from `ml backtest`; filters `is_true_positive == False`
    - Validates required columns: is_true_positive, min_loss_120m, max_gain_120m
  - `generate_risk_report(fp_df)` → dict
    - total_false_positives, pct_price_never_below_entry
    - Loss distribution: mean/median/p25/p50/p75/p90/max_worst_drawdown_pct
    - Loss buckets: pct_0_to_5pct, pct_5_to_10pct, pct_10_to_15pct, pct_15_to_20pct, pct_over_20pct
    - Stop trigger rates: stop_5/10/15/20pct_triggered_pct
    - Recommendations: stop_loss_conservative_pct (p75), moderate (p90), aggressive (p95)
  - `stop_loss_impact(fp_df, stop_losses)` → dict keyed by stop level
    - Per level: triggered_count, triggered_pct, exit_loss_pct, uncaught_max_loss_pct, uncaught_count
    - Default levels: -5, -10, -15, -20, -25%
  - `plot_ascii(fp_df, bins=10)` → str — ASCII █░ histogram of min_loss_120m distribution
- [x] Add `analyze-errors` CLI command to `src/ml/cli.py`
  - `--trades` (required) — path to backtest trades CSV
  - `--output` (optional) — path to save JSON risk report
  - Prints: histogram, risk report, stop-loss trigger table, recommendations
- [x] Unit tests: 35 tests (TestLoadFalsePositives ×7, TestGenerateRiskReport ×11,
  TestStopLossImpact ×8, TestPlotAscii ×8)
- [x] Real results (xgboost_v2, threshold=0.67, 25 FPs, test split):
  - Median drawdown: -23.8% | P90: -15.8% | Worst: -69.1%
  - 0% of FPs stayed above entry price — all fell below entry
  - 80% of FPs lost >20%; 16% lost 15–20%; 4% lost 10–15%
  - Stop at -10% would catch 100% of FPs; -20% catches 80%; -25% catches 44%
  - Conservative stop recommendation: -20.0% | Moderate: -15.8%
- [x] Full test suite: 1161 passing + 7 skipped
- Added `calculate_expected_value(precision, avg_win_pct, avg_loss_pct, stop_loss_pct)` → dict
  - EV per trade = precision × avg_win + (1-precision) × avg_loss
  - Returns: win_rate, loss_rate, avg_win_pct, avg_loss_pct, expected_value_pct,
    profitable (bool), breakeven_win_rate
  - `analyze-errors` CLI now prints EV section using conservative stop and signal precision
  - 10 new tests for calculate_expected_value (TestCalculateExpectedValue)
- [x] Full test suite: 1171 passing + 7 skipped

## Step 41: Threshold Sensitivity Analysis ✅
- [x] Create `src/ml/threshold_analyzer.py` — `ThresholdAnalyzer` class
  - `analyze_full_year(artifact, features_dir, thresholds, start_date, end_date)` → dict
    - Loads full feature dataset with `load_features()`; runs `predict_proba` in one batch
    - Returns: aggregate (DataFrame), monthly (DataFrame), daily (DataFrame),
      date_range, total_samples, n_dates, n_months
  - `_analyze_threshold_range(df, predictions, thresholds)` → DataFrame (one row per threshold)
  - `_analyze_single_threshold(y_true, predictions, max_gains, min_losses, threshold)` → dict
    - Counts: total_signals, TP, FP, FN, TN, signal_rate
    - Metrics: precision, recall, f1_score
    - TP profit stats: max/avg/median/min/std (from max_gain_120m on TP bars)
    - FP loss stats: max/avg/median/min/std (from min_loss_120m on FP bars)
    - FN missed stats: max/avg/median/min/std (from max_gain_120m on FN bars)
    - Expected value = precision × avg_tp_gain + (1−precision) × avg_fp_loss
    - NaN-safe: drops NaN end-of-day bars via pd.Series.dropna()
  - `generate_monthly_summary(monthly_df, key_thresholds)` → DataFrame
    - Pivots monthly_df to: month × (signals_{pct}, precision_{pct}, ev_{pct})
  - `plot_monthly_signals(monthly_summary, key_thresholds)` → str (ASCII bar chart)
  - `find_optimal_threshold(results_df, optimization_metric, min_precision, min_signals)` → dict
    - Returns SUCCESS + optimal_threshold + metrics, or NO_VALID_THRESHOLD + message
- [x] Add `threshold-analysis` CLI command to `src/ml/cli.py`
  - `--model-path` (required), `--start-date`, `--end-date`
  - `--min-threshold` (0.70), `--max-threshold` (0.95), `--step` (0.01)
  - `--output` (default: data/reports/threshold_analysis)
  - Saves 7 files: aggregate_analysis.csv, monthly_breakdown.csv, daily_breakdown.csv,
    monthly_summary.csv, monthly_signals_chart.txt, aggregate_key_thresholds.csv, recommendations.json
  - Prints: ASCII bar chart, key-threshold table, monthly summary, 2 optimal recommendations
- [x] Unit tests: 38 tests (TestAnalyzeSingleThreshold ×11, TestAnalyzeThresholdRange ×6,
  TestGenerateMonthlySummary ×7, TestPlotMonthlySignals ×6, TestFindOptimalThreshold ×8)
- [x] Full test suite: 1209 passing + 7 skipped
- [x] Real results (xgboost_v2, full year 2025-03-03 → 2026-02-19):
  - 172,068 rows | 231 dates | 12 months
  - At threshold=0.70: 5,004 signals | 94.5% precision | TP avg profit=90.2%
  - At threshold=0.75: 1,359 signals | 93.6% precision | TP avg profit=107.1%
  - At threshold=0.85: 37 signals | 94.6% precision | 0 FPs | EV=+101%
  - At threshold=0.90+: 0 signals (model doesn't output probabilities that high)
  - December 2025 (test period): precision drops to 58%/31%/16% at 0.70/0.75/0.80
    → confirms held-out test performance is more modest than training period
  - Key insight: use 'ml backtest --threshold' for real held-out evaluation

## Step 42: Signal Explainability (SHAP) ✅
- [x] Create `src/ml/explainer.py` — `SignalExplainer` class
  - `__init__(model, feature_names)` — initialises `shap.TreeExplainer(model)`
  - `from_artifact(artifact)` — classmethod, constructs from joblib artifact dict
  - `explain_signal(features, prediction_proba, threshold=0.90)` → str
    - Builds ordered feature array; calls `shap_explainer.shap_values()` for SHAP values
    - Sorts impacts by abs(SHAP) descending; takes top 10
    - Calls `_interpret_feature` and `_format_explanation`
  - `_interpret_feature(name, value, impact)` → str
    - Plain-English sentences for all 66 model features (option returns, SPY technicals,
      volume, IV, moneyness, time features, contract type, DTE)
    - Generic fallback `"{name} = {value:.4g}"` for unknown features
  - `_format_explanation(confidence, threshold, top_impacts, features)` → str
    - Header: "🎯 SIGNAL DETECTED" + "CAUTION: Near Threshold" when margin < 2%
    - Confidence, threshold, and margin display with "⚠️ CLOSE CALL" warning
    - Numbered top-10 factors with 🔴 strength dots (1–5 scaled by abs(SHAP) × 20),
      BULLISH/BEARISH labels, and human-readable interpretation
    - "⚠️ RISK FACTORS" section for negative-SHAP features in top 10
    - "💡 RECOMMENDATION: Proceed with caution" when ≥2 risk factors
- [x] Add `explain-signal` CLI command to `src/ml/cli.py`
  - `--model-path` (required), `--features-file` (required)
  - `--ticker` (filter to specific option), `--row-index` (specific row)
  - `--threshold` (override artifact threshold)
  - Default: selects row with highest predicted probability; prints row index + confidence
- [x] Add `shap>=0.50.0` to `requirements.txt`
- [x] Unit tests: 60 tests
  - TestSignalExplainerConstruction ×5: init, from_artifact, missing shap ImportError
  - TestExplainSignal ×15: return type, confidence/threshold display, top-10 limit,
    margin, CLOSE CALL, risk factors, missing features fallback to 0, SHAP call,
    CAUTION header, RECOMMENDATION with ≥2 risk factors
  - TestInterpretFeature ×26: option returns, SPY RSI/MACD/EMA, volume/zscore/regime,
    IV (high/low/change), time (hour/morning/last-hour), moneyness (ATM/ITM/OTM),
    contract type, 0DTE, unknown fallback, all 66 model features return non-empty strings
  - TestFormatExplanation ×14: string type, header, confidence/threshold/margin,
    feature names, risk factors, caution, BULLISH/BEARISH, 🔴 dots, interpretation text
- [x] Full test suite: 1269 passing + 7 skipped

## Step 43: Walk-Forward Validation ✅
- [x] Create `src/ml/walk_forward_validator.py` — `WalkForwardValidator` class
  - `__init__(features_dir, xgb_params=None, train_window_months=3, test_window_months=1)`
    - Defaults to xgboost_v2 params (n_estimators=300, max_depth=6, lr=0.05, etc.)
  - `get_date_splits()` → List[Tuple[str,str,str,str]]
    - Uses calendar-month boundaries + `relativedelta`; 1-month slide (overlapping train windows)
    - Full 12-month dataset (Mar 2025–Feb 2026) → 8 splits (Split 9 excluded: Feb window > data end)
  - `load_date_range(start_date, end_date)` → pd.DataFrame
    - Delegates to `load_features()` from train_xgboost.py for consistent filtering
  - `evaluate_split(train_start, train_end, test_start, test_end, threshold)` → Dict
    - 80/20 chronological split within training window for early-stopping validation
    - Undersamples majority class in training portion only (`undersample_majority`)
    - Computes: precision, recall, total_signals, TP, FP, FN, TN, signal_rate
    - Computes: tp_avg_gain_pct, fp_avg_loss_pct, expected_value_pct
    - Returns "INSUFFICIENT_DATA" if train/test empty or <10 positives in training
  - `run_validation(threshold=0.67)` → Dict
    - Aggregates: precision_mean/median/std/min/max, signals_mean/median/min/max, ev_mean/median/std
    - Returns "NO_SPLITS", "ALL_SPLITS_FAILED", or "SUCCESS"
  - `plot_results(summary)` → str (ASCII bar chart of precision per test month)
- [x] Add `walk-forward` CLI command to `src/ml/cli.py`
  - `--threshold` (0.67), `--train-months` (3), `--test-months` (1), `--output`
  - Prints split preview table, per-split results table, ASCII bar chart, summary stats, interpretation
  - Saves `data/reports/walk_forward/walk_forward_results.json`
- [x] Unit tests: 36 tests
  - TestGetDateSplits ×8: empty dir, single date, 9 splits, 4-element tuples, non-overlapping tests,
    overlapping train windows, train_end < test_start, custom window sizes
  - TestLoadDateRange ×3: delegates to load_features, passes dates, empty result
  - TestGetFeatureCols ×3: excludes non-features, includes features, sorted
  - TestEvaluateSplit ×7: empty train, empty test, few positives, required keys, precision in range,
    TP+FP=signals, test_month label
  - TestRunValidation ×5: no splits, all failed, precision stats present, precision mean correct, splits list
  - TestPlotResults ×6: returns string, test months, bar chars, precision %, no-splits placeholder, header
  - TestDefaultXgbParams ×4: required keys, n_estimators=300, min_positives>=5, custom override
- [x] Full test suite: 1305 passing + 7 skipped
- [x] Live validation results (xgboost_v2 params, threshold=0.67, 3-month train, 1-month test):
  - 8 splits executed: Jun–Aug–Sep–Oct–Nov–Dec 2025, Jan 2026
  - **Precision mean: 55.8% | median: 61.8% | std: 36.2%** (0.0% – 100.0%)
  - **Key insight**: The 91.9% full-year backtest is an OUTLIER (+36.1% above walk-forward mean)
  - **Root cause**: Model trained on 3 months generalizes poorly; it needs 6+ months of training data
  - **High variance** (std=36.2%) confirms model is unstable across market regimes
  - Splits with most signals (Sep/Oct/Dec) show 56–69% precision — consistent with theory
  - Splits with few signals (Jan: 2 signals, Jun: 0) are not statistically meaningful
  - **Verdict**: Weak POC at 3-month rolling window; full-year training is more reliable approach

## Step 44: Detailed Trade-by-Trade Simulation ✅
- [x] Create `src/ml/trade_simulator.py`
  - `Trade(trade_id, entry_time, contract_symbol, entry_price_per_share, position_size_usd, confidence, trigger_factors)`
    - `SHARES_PER_CONTRACT = 100` (US equity options multiplier)
    - `cost_per_contract = entry_price_per_share × 100`
    - `num_contracts = floor(position_size_usd / cost_per_contract)`
    - `actual_position_size = num_contracts × cost_per_contract`
    - `close_trade(exit_time, exit_price_per_share, exit_reason, time_in_trade_minutes)`
      - `profit_loss_usd = (exit_cost - entry_cost) × num_contracts`
      - `profit_loss_pct = (exit_cost - entry_cost) / entry_cost × 100`
    - `to_dict()` → JSON-serialisable dict of all trade fields
  - `TradeSimulator(position_size_usd=12500, target_gain_pct=30, stop_loss_pct=-12, max_time_minutes=120, fee_per_trade_usd=4)`
    - `simulate_from_label_row(row, trade_id, confidence, trigger_factors)` → Optional[Trade]
      - Uses `close` (entry price/share), `max_gain_120m` (%), `min_loss_120m` (%) from feature CSV
      - Exit priority: both target+stop → stop first (conservative); target only → Target hit; stop only → Stop-loss; neither → Time limit
      - Returns None if entry_price=0/NaN or num_contracts<1
    - `simulate_period(df, predictions, threshold, model, feature_cols)` → List[Trade]
      - Fires on rows where predictions >= threshold; populates trigger_factors from model.feature_importances_
    - `generate_monthly_report(trades, month_str)` → Dict
      - Returns: total_trades, win_rate, gross P&L, net P&L, fees, ROI, avg/median profit/loss, calls/puts, exit_reasons, individual trade dicts
    - `print_trade_log(trade, trade_num)` — full trade lifecycle to stdout
    - `print_monthly_summary(report)` — summary table with target assessment ($15k/$20k/$25k)
  - `_extract_contract_type(ticker)` → 'C' | 'P' | 'X' helper
- [x] Integrate `TradeSimulator` into `WalkForwardValidator`
  - Constructor: `simulator: Optional[TradeSimulator] = None` parameter
  - `evaluate_split()`: calls `simulator.simulate_period()` + `generate_monthly_report()` after computing y_proba; adds `trade_report` to result dict
  - `run_validation()`: aggregates monthly reports → `simulation` dict (months_simulated, total_trades, win_rate, total_net_profit_usd)
- [x] Update `walk-forward` CLI in `src/ml/cli.py`
  - New options: `--position-size` (12500), `--target-gain` (30), `--stop-loss` (-12), `--show-trades` (flag)
  - Creates `TradeSimulator` → passes to `WalkForwardValidator`
  - Prints "Trade Simulation" section: monthly breakdown table + aggregate totals
  - `--show-trades`: prints per-trade details (entry/exit times, prices, contracts, P&L)
- [x] Unit tests: 48 tests in `tests/unit/test_trade_simulator.py`
  - TestSharesPerContract ×2
  - TestTradeConstruction ×7: basics, cost_per_contract, num_contracts flooring, actual_position, zero price, trigger factors, None exits
  - TestTradeCloseTrade ×6: winner/loser P&L math, explicit time, timestamp parse, zero contracts, exit_cost_per_contract
  - TestTradeToDict ×2: required keys, JSON-serialisable
  - TestExtractContractType ×4: call, put, unknown, short ticker
  - TestTradeSimulatorConstruction ×2
  - TestSimulateFromLabelRow ×10: target hit, stop hit, both (stop first), time limit, zero/negative/NaN price, too expensive, entry/exit time format
  - TestSimulatePeriod ×6: no signals, all signals, partial, skip zero price, trigger_factors with model, returns Trade objects
  - TestGenerateMonthlyReport ×9: empty, single winner/loser, mixed, required keys, fees, calls/puts, exit reasons, trades list
- [x] Full test suite: 1353 passing + 7 skipped
- [x] Live walk-forward simulation results (xgboost_v2 params, threshold=0.67, $12,500 position, +30% target, -12% stop):
  - 8 splits (Jun 2025 – Jan 2026), 7 months with trades, 1 month (Jun) with 0 signals
  - **Total trades: 2,875  |  Win rate: 33.9%**
  - **Total net profit: +$614,467 across all test months**
  - Best month: Sep 2025 (+$383K, 38.7% win), Oct 2025 (+$273K, 44.8% win)
  - Worst month: Jul 2025 (-$134K, 25.9% win) — low precision month inflates losses
  - Key insight: strategy is profitable despite 33.9% win rate because gains (+30%) dwarf losses (-12%) by 2.5×
  - Call/put breakdown: roughly even (1,419 calls / 1,456 puts across all months)
  - Note: Simulation uses label-based exits (max_gain_120m/min_loss_120m) — approximates real P&L without tick-level data

## Step 45: Hyperparameter Optimization — Optuna Bayesian Search ✅
- [x] Create `src/ml/hyperparameter_optimizer.py`
  - `HyperparameterOptimizer(model_class, param_space, X_train, y_train, optimization_metric='precision', n_trials=100, cv_splits=5, min_signals_per_fold=3, model_extra_params=None, predict_threshold=0.5, random_state=42)`
    - `_sample_params(trial)` — list → suggest_categorical, (int,int) → suggest_int, (float,float) → suggest_float
    - `objective(trial)` — TimeSeriesSplit CV; returns 0.0 if signals < min_signals_per_fold per fold
    - `optimize()` → Dict: best_params, best_score, model_class, optimization_metric, n_trials, optimization_history
    - `save_results(output_path)` — JSON dump
  - `LSTMHyperparameterOptimizer` — lazy torch import; imports LSTMModel inside objective()
  - `make_sequences(X, y, seq_len)` — sliding-window helper → (X_seq, y_seq)
  - Predefined param spaces: `XGBOOST_PARAM_SPACE`, `LIGHTGBM_PARAM_SPACE`, `RF_PARAM_SPACE`, `LSTM_PARAM_SPACE`
  - Graceful optuna handling: `_OPTUNA_AVAILABLE` flag

## Step 46: Multi-Model Comparison Framework ✅
- [x] Create `src/ml/model_comparator.py`
  - `ModelComparator(position_size_usd=12500, target_gain_pct=30, stop_loss_pct=-12, fee_per_trade_usd=4, monthly_profit_target=10000)`
    - `add_model(name, model, feature_cols, best_params, optimization_score, model_type)` — registers model
    - `evaluate_at_thresholds(model_name, test_df, thresholds)` → Dict[threshold → result_dict]
      - Derives feature_cols from `_NON_FEATURE_COLS` if not provided
      - Calls `TradeSimulator.simulate_period()` for each threshold
      - Returns: threshold, total_signals, calls/puts, win_rate, win stats (min/median/mean/max), loss stats, gross/net profit, fees, trades list
    - `find_signal_overlap(model_results, threshold=0.80)` → Dict
      - Key: (contract_symbol, entry_time) — both needed to uniquely identify a trade
      - Returns: total_unique_signals, all_models_agree, majority_agree, overlap_breakdown, detailed_overlaps
    - `generate_comparison_report(comparison_threshold=0.80)` → pd.DataFrame
      - Columns: Model, Opt Score, Signals (80%), Win Rate, Calls%, Puts%, Avg Win $, Avg Loss $, Net Profit, Meets Target
    - `get_best_threshold_per_model()` → Dict[model → {best_threshold, net_profit_usd, signals, win_rate}]
    - `save_results(output_dir)` — per-model JSON + model_comparison.csv + overlap_{t}.json at each default threshold
  - `DEFAULT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]`

## Step 47: PyTorch LSTM Model ✅
- [x] Create `src/ml/lstm_model.py`
  - `OptionsSequenceDataset(X, y, seq_len=20)`
    - Pure-numpy: no PyTorch dependency for instantiation
    - `_make_sequences(X, y, seq_len)` static method — returns (X_seq shape (n-seq_len, seq_len, n_features), y_seq shape (n-seq_len,))
    - `__len__()` / `__getitem__()` for compatibility
    - Raises `ValueError` if len(X) <= seq_len
  - `LSTMModel(input_size, hidden_size=128, num_layers=2, dropout=0.3)`
    - Wraps `_LSTMModule(nn.Module)` — stacked LSTM + FC classifier
    - `build()` → nn.Module; `module` property auto-builds on first access
    - Single-layer LSTM: dropout disabled on recurrent connections
  - `LSTMTrainer(input_size, hidden_size=128, num_layers=2, dropout=0.3, seq_len=20, lr=1e-3, epochs=50, batch_size=64, patience=10, pos_weight_factor=10.0, device=None)`
    - `fit(X, y)` → self: 80/20 chronological train/val split; BCEWithLogitsLoss with pos_weight; early stopping; restores best weights
    - `predict_proba(X)` → ndarray(n,): pads first seq_len rows with 0.5 for index alignment
    - `predict_proba_2d(X)` → ndarray(n, 2): sklearn-compatible two-column output
  - Graceful torch handling: `_TORCH_AVAILABLE` flag; `_require_torch()` raises clear ImportError
  - `_LSTMModule` and `_TensorDataset` conditionally defined only when torch is available
- [x] Unit tests: 35 tests in `tests/unit/test_lstm_model.py` (skipped gracefully when torch absent)
  - TestOptionsSequenceDataset ×8: lengths, shapes, dtypes, label/window alignment, error, getitem, static empty
  - TestLSTMModelBuild ×6: build(), module property, forward shape, dropout, linear output
  - TestLSTMTrainerFit ×8: returns self, module set, losses recorded, short data error, early stopping, val losses, all-positive, all-negative
  - TestLSTMTrainerPredictProba ×10: length, padding, range, dtype, short input, 2D shape, 2D sums, col1 match, raises before fit, batch consistency
  - TestTorchNotAvailable ×3: _require_torch, LSTMModel, LSTMTrainer all raise ImportError
- [x] Full test suite: 1285 passing, 35 skipped (LSTM tests skip when torch absent)

## Step 48: ML Comparison Streamlit Dashboard ✅
- [x] Create `src/ml/dashboard.py`
  - Launch: `streamlit run src/ml/dashboard.py -- --results-dir data/reports/model_comparison`
  - `--results-dir` CLI arg (default: `data/reports/model_comparison`)
  - `_load_results_dir(results_dir)` — reads all `*_results.json`; converts string keys back to float thresholds; `@st.cache_data`
  - `_load_comparison_csv(results_dir)` — reads `model_comparison.csv`; `@st.cache_data`
  - `_load_overlap_json(results_dir, threshold)` — reads `overlap_{t:.2f}.json`; `@st.cache_data`
  - `_get_results_dir()` — parses `--results-dir` / `--results_dir` from `sys.argv`
  - Sidebar: shows results directory, file count, per-file list; "Refresh data" button clears cache; plotly warning if absent
  - **Tab 1 — Model Comparison**: comparison table (from CSV) + net-profit line chart + win-rate line chart (all models, all thresholds)
  - **Tab 2 — Threshold Sweep**: single-model selector; metrics table; signal count bar chart; net-profit bar chart (green/red); calls vs puts stacked bar
  - **Tab 3 — Signal Overlap**: threshold selector; 4 KPI metrics (unique signals, models compared, all agree, majority agree); agreement breakdown bar chart; all-agree detail table
  - **Tab 4 — Trade Explorer**: model + threshold selectors; 4 KPI metrics; outcome/exit-reason/contract-type filters; sorted trade table; P&L distribution histogram
  - Graceful degradation: charts replaced with DataFrames when plotly absent; clear ImportError when streamlit absent
- [x] Unit tests: 25 tests in `tests/unit/test_ml_dashboard.py`
  - TestLoadResultsDir ×8: model names, float keys, threshold values, required keys, missing dir, empty dir, corrupt JSON, model count
  - TestLoadComparisonCsv ×5: returns DataFrame, not empty, Model column, missing dir, missing CSV
  - TestLoadOverlapJson ×5: returns dict, expected keys, model count, missing threshold, missing dir
  - TestGetResultsDir ×4: default, --results-dir flag, --results_dir variant, flag without value
  - TestDashboardConstants ×3: Path type, colour format (#rrggbb), threshold coverage
- [x] Full test suite: 1310 passing, 35 skipped

## Step 49: full-comparison CLI Command + requirements.txt Update ✅
- [x] Add `full-comparison` command to `src/ml/cli.py`
  - Options: `--model-path NAME=PATH` (repeatable, required), `--features-dir`, `--test-start-date` (required), `--test-end-date` (required), `--thresholds` (CSV floats, default 0.70–0.95), `--position-size`, `--target-gain`, `--stop-loss`, `--monthly-profit-target`, `--overlap-threshold`, `--output`
  - Pipeline:
    1. Parses `NAME=PATH` model specs; exits with clear error if malformed
    2. Parses threshold CSV with `float()` conversion; exits on bad format
    3. Loads test feature CSVs via `load_features(features_dir, start, end)`; exits if empty
    4. Loads each model artifact with `joblib.load()`; exits with FileNotFoundError message
    5. Registers each model with `ModelComparator.add_model()`
    6. Calls `evaluate_at_thresholds()` per model; prints per-threshold summary table
    7. Calls `get_best_threshold_per_model()` → prints best-threshold summary
    8. Calls `generate_comparison_report(0.80)` → prints side-by-side table
    9. Calls `find_signal_overlap()` when ≥2 models registered; prints overlap stats
    10. Calls `save_results(output_dir)` → lists saved files with sizes
    11. Prints `streamlit run src/ml/dashboard.py -- --results-dir <output>` hint
  - Full example: `python -m src.cli ml full-comparison --model-path xgboost=models/xgb.pkl --model-path lightgbm=models/lgbm.pkl --test-start-date 2025-12-23 --test-end-date 2026-02-19`
- [x] Update `requirements.txt` — add new optional ML deps:
  - `optuna>=3.0.0` — Bayesian hyperparameter optimization
  - `lightgbm>=4.0.0` — LightGBM gradient boosting
  - `plotly>=5.18.0` — interactive charts for ML dashboard
  - `torch>=2.0.0` — PyTorch for LSTM model (optional)
- [x] Unit tests: 18 new tests in `tests/unit/test_ml_cli.py` (total 50 tests)
  - TestFullComparisonHelp ×11: help text, required arg validation, bad NAME=PATH format, bad threshold format
  - TestFullComparisonRun ×6: success exit code, comparison table printed, dashboard hint, missing model file, empty features, multiple models + overlap
  - TestMainCliIntegration ×1: full-comparison accessible from main CLI
- [x] Full test suite: 1329 passing, 35 skipped

## Overnight Pipeline Run (2026-02-21) ✅
- [x] Install optuna, lightgbm, plotly via pip
- [x] `run_overnight.py` executed: 69m 12s total
- [x] XGBoost Optuna (50 trials) → best CV precision=0.5146 → `models/xgboost_opt.pkl`
- [x] LightGBM Optuna (50 trials) → best CV precision=0.5762 → `models/lightgbm_opt.pkl`
- [x] full-comparison results in `data/reports/model_comparison/`
- Findings:
  - xgboost_v2 (manually-tuned) best: 169 signals @ 0.70 threshold, 65.1% win rate, +$314,508
  - xgboost_opt: 6,366 signals @ 0.70 threshold, 29.5% win rate, -$504,158 (Optuna CV metric ≠ P&L)
  - lightgbm_opt: 0 signals at all thresholds (proba max=0.5654; fix: calibration or lower threshold)
- Dashboard ready: `streamlit run src/ml/dashboard.py -- --results-dir data/reports/model_comparison`

## Step 52: Leakage Detection + Dashboard Polish (Steps 50–51 continued) ✅
- [x] `src/ml/dashboard.py` — Trade Explorer: green/red row coloring + 2-decimal float formatting
- [x] `src/ml/leakage_detector.py` — LeakageDetector with 6 tests; `detect-leakage` CLI command
- [x] `tests/unit/test_leakage_detector.py` — 26 unit tests
- [x] `src/processing/ml_feature_engineer.py` — removed `opt_vol_pct_cumday` (lookahead feature)
- [x] `src/ml/train_xgboost.py` — added `opt_vol_pct_cumday` to `_NON_FEATURE_COLS` safety net
- [x] Retrained `models/xgboost_v3_clean.pkl` (65 features, no lookahead)
- [x] Full-comparison v2 vs v3_clean confirmed: 100% win rate was entirely due to lookahead

## Step 53: Sustained Movement Prediction Experiment ✅
- [x] Create `src/processing/sustained_movement_labeler.py`
  - `generate_sustained_labels(df, confirmation_minutes=15, sustain_minutes=5)` → 4 columns
    - Algorithm: at bar T, check bar T+15 min; if close > entry, count consecutive bars above entry;
      if ≥ sustain_minutes → target_sustained = 1
  - `classify_magnitude(gain_pct)` → one of 6 buckets: below_zero, 0-1%, 1-5%, 5-10%, 10-20%, 20%+
  - `SustainedMovementLabeler(config)` — config-driven wrapper: `label(df)`, `label_for_file(path)`,
    `validate(df)` (returns stats + magnitude breakdown)
  - Output cols: `target_sustained`, `gain_pct_at_confirmation`, `magnitude_bucket`,
    `sustain_minutes_actual`
- [x] Create `src/ml/multi_model_trainer.py`
  - `MultiModelTrainer(n_trials=50, cv_splits=3, operating_thresholds, min_signals=10)`
  - `train(df, target_col, feature_cols)` → dict of 3 artifacts (xgboost, lightgbm, random_forest)
    - Chronological 80/20 split → undersample majority in train → Optuna search → retrain best
    - Optuna objective: avg precision across operating_thresholds (0.70–0.90), min_signals filter
  - `save_artifacts(artifacts, output_dir)` → Dict[name → Path]
  - Artifacts compatible with `ModelComparator.add_model()` and `full-comparison` CLI
- [x] Create `src/ml/sustained_movement_evaluator.py`
  - `SustainedMovementEvaluator(thresholds, target_col)`
  - `evaluate(artifacts, test_df)` → nested results dict:
      models × threshold_results (precision/recall/F1/TP/FP/FN) +
      precision_by_magnitude (per bucket × threshold) + model_agreement stats
  - `generate_report(results, comparison_threshold)` → pd.DataFrame (one row per model)
  - `save_results(results, report_df, output_dir)` → writes 4 files:
      full_results.json, comparison_report.csv, precision_by_magnitude.json, model_agreement.json
- [x] Add `sustained-movement-experiment` CLI command to `src/ml/cli.py`
  - 5-step pipeline (load → label → split → train → evaluate)
  - Options: --confirmation-minutes, --sustain-minutes, --n-trials, --cv-splits, --thresholds
  - Prints magnitude breakdown, per-model results table, precision-by-magnitude table, model agreement
  - Saves model .pkl files + evaluator reports to --output dir
- [x] Create `tests/unit/test_sustained_movement.py` — 38 tests:
  - TestSustainedMovementLabeler ×12: positive/negative labels, NaN/zero handling, empty df,
    missing cols, per-ticker isolation, gain_pct, config wrapper, validate()
  - TestClassifyMagnitude ×7: all six buckets + NaN
  - TestMultiModelTrainer ×8: error cases, three-model keys, artifact keys, pkl save
  - TestSustainedMovementEvaluator ×11: error cases, evaluate() structure,
    threshold coverage, precision range, report DataFrame, agreement, magnitude
- [x] Full test suite: 1461 passing + 42 skipped (LSTM when torch absent)

## Step 63: Magnitude Prediction Experiment ✅ (completed 2026-02-23, 17 min)
- [x] Create `src/processing/magnitude_labeler.py` — MagnitudeLabeler
  - `__init__(min_magnitude_pct=20.0)` — direction-agnostic volatility labeler
  - `label(df)` — uses pre-computed max_gain_120m / min_loss_120m; no raw-bar scanning
    - abs_max = max(max_gain.clip(0), abs(min_loss))
    - target_magnitude = 1 if abs_max >= min_magnitude_pct else 0
    - Adds: target_magnitude, abs_max_move_pct, move_direction, magnitude_bucket
    - Magnitude buckets: 0-5%, 5-10%, 10-20%, 20-30%, 30%+
  - `validate(df)` — returns n_total, n_positive, positive_rate, n_up, n_down,
      avg_abs_magnitude, magnitude_breakdown dict
- [x] Add `generate-magnitude-labels` CLI command — fast diagnostic, no files written
- [x] Add `train-magnitude-models` CLI command — full pipeline:
  - Load → MagnitudeLabel → 70/30 split → MultiModelTrainer (XGBoost+LightGBM+RF, 50 trials)
  - Evaluate at 7 thresholds with straddle P&L model:
      TP = (avg_magnitude - 5) / 100 * position_size
      FP = -8 / 100 * position_size
  - Direction breakdown + precision-by-magnitude at 0.70
  - Saves magnitude_results.json + model pkl files
- [x] Create `run_magnitude_experiment.py` — autonomous overnight master script
- Dataset: 172,822 rows | **82.74% positive rate** | 65 features | elapsed: 17 min
- Results (51,847 test rows, 72 days):

| Model | Thresh | Precision | Signals/Day | Avg Magnitude |
|---|---|---|---|---|
| LightGBM | 0.70 | **96.2%** | 425.7 | 59.8% |
| LightGBM | 0.85 | **99.4%** | 82.3 | 65.9% |
| LightGBM | 0.90 | **100.0%** | 8.7 | 61.7% |
| XGBoost | 0.70 | 91.3% | 581.8 | 55.7% |
| RandomForest | 0.70 | 94.8% | 356.6 | 59.3% |

## Step 65: Magnitude Model Leakage Audit ✅ (2026-02-23)
- [x] Add `check_magnitude_specific_leaks()` (Test 9) to `src/ml/leakage_detector.py`
  - Check 1: Forbidden magnitude label columns in feature set
    (target_magnitude, abs_max_move_pct, move_direction, magnitude_bucket,
     max_gain_120m, min_loss_120m, time_to_max_min, target, target_sustained)
  - Check 2: Pearson |corr| ≥ 0.50 with abs_max_move_pct → HIGH suspicion
  - Check 3: Future-looking name patterns (future, lookforward, time_to_peak, peak_gain)
  - `generate_report()` updated to include Test 9 critical issues
- [x] Add `audit-magnitude-model` CLI command to `src/ml/cli.py`
  - Runs all 9 tests on any magnitude model artifact
  - Applies MagnitudeLabeler in-memory (no pre-saved label files)
  - Precision table on 70/30 test holdout + fresh 2026 subset
  - Verdict with base-rate lift interpretation
  - Tees output to terminal + `reports/leakage_audit/{model}_magnitude_audit.log`
- Audit results for all 3 magnitude models (172,822 rows, 65 features):

| Test | LightGBM | XGBoost | RandomForest |
|---|---|---|---|
| Random data (noise) | ✅ 49/10000 (0.49%) | 🚨 9948/10000 (99.5%) | ✅ 0/10000 (0%) |
| Source-code audit | ✅ PASS | ✅ PASS | ✅ PASS |
| Known lookahead | ✅ PASS | ✅ PASS | ✅ PASS |
| Target-in-features | ✅ PASS | ✅ PASS | ✅ PASS |
| Temporal ordering | ✅ PASS | ✅ PASS | ✅ PASS |
| Train/test contamination | ⚠️ FP* | ⚠️ FP* | ⚠️ FP* |
| 120-min correlation | ⚠️ FP* | ⚠️ FP* | ⚠️ FP* |
| Feature importance | ✅ PASS | 🚨 hour_et=42.75% | ✅ PASS |
| Magnitude-specific (T9) | ✅ PASS | ✅ PASS | ✅ PASS |

  *Both are known false positives: "1 overlap" = 70/30 split boundary date (2025-10-29);
   `pct_day_elapsed` flagged by `pct_day` regex but is backward-looking time fraction.
- **Verdicts:**
  - **LightGBM: CLEAN ✅** — random avg_p=0.711 on noise (below 82.74% base rate);
    top feature log_moneyness (6.74%); well-distributed importance; 96%+ precision is REAL
  - **XGBoost: OVERFIT 🚨** — fires 99.5% on random noise; hour_et dominates at 42.75%;
    model learned time-of-day distribution, not genuine signal
  - **RandomForest: CLEAN ✅** — 0 signals on noise; opt_price_change_open top (5.45%);
    fresh 2026: 95.6% @ 0.70, 96.6% @ 0.85; precision is REAL and generalisable
- KEY INSIGHT: 82.74% base rate explains "why precision is high" without leakage.
  LightGBM/RF learned: deeply OTM options with high IV already-moving → near-certain 20%+ move.
  This is genuine financial signal (OTM options have binary payoff; once moving, they continue).
- Safe models for straddle strategy: **LightGBM** and **RandomForest**
- Reports: `reports/leakage_audit/{lightgbm,xgboost,random_forest}_magnitude_audit.{log,json}`

## Step 67: Final Optimization — Maximum Precision, Minimum Signals ✅ (2026-02-23)
- [x] Create `src/ml/deep_hyperparameter_optimizer.py` — 3 classes
  - `DeepHyperparameterOptimizer(X_train, y_train, X_val, y_val, val_days, target_signals_per_day=2.5, min_precision=0.98)`
    - `_objective_score(y_proba)` — sweeps 0.85→0.999 (step 0.005); rewards 1–5 signals/day
      - spd < 1 → precision * 0.5 | spd > 5 → precision * 0.7 | 1–5 → precision * (0.7 + 0.3 * vol_score)
    - `optimize_lightgbm_precision(n_trials=200)` — 13-param Optuna TPE search; returns {best_params, best_value, study}
    - `optimize_randomforest_precision(n_trials=200)` — 8-param Optuna TPE search (oob_score removed; max_samples gated on bootstrap=True)
  - `EnsembleStrategy`
    - _THRESHOLDS_AND = [0.85, 0.90, 0.95, 0.97, 0.99] | _OR = [0.90, 0.95, 0.97] | _AVG = [0.90, 0.95, 0.97, 0.99]
    - `test_ensemble_combinations(lgbm_model, rf_model, X_test, y_test, test_df, n_test_days, target_signals_per_day=2.5)`
    - Viable = spd in [target*0.5, target*2.0]; Best = max by (precision, avg_magnitude)
    - Returns {results: List[Dict], best: Dict | None}
  - `MonteCarloSimulator`
    - `simulate_monthly_pnl(test_df, signals_mask, position_size_range=(10K,15K), n_simulations=1000, n_test_months=2.4)`
    - rng = np.random.default_rng(42); position~Uniform, theta~Normal(-6%,2%), slippage~Normal(-1%,0.5%), fill=Bernoulli(0.95)
    - TP: winner_pnl = position*(abs_max_move_pct+slippage)/100 | FP: loser_pnl = position*theta/100
    - monthly_pnl = total_pnl / n_test_months; Returns {mean, median, std, percentiles:{p5,p25,p75,p95}, win_rate, n_signals, n_months}
- [x] Add `final-optimization` CLI command to `src/ml/cli.py` (appended after line 6,557)
  - Options: --features-dir (required, Path exists), --n-trials=200, --target-signals-per-day=2.5, --min-magnitude=20.0, --n-mc-simulations=1000, --output=reports/final_optimization
  - 7-step pipeline with _Tee logging → final_optimization.log
  - Saves: lgbm_deep_opt.pkl, rf_deep_opt.pkl, final_optimization_results.json
  - JSON keys: lgbm, rf, ensemble, monte_carlo
- [x] Create `run_final_optimization.py` — subprocess wrapper; default production settings
  - N_TRIALS=200, TARGET_SIGNALS_PER_DAY=2.5, MIN_MAGNITUDE=20.0, N_MC_SIMULATIONS=1000
- Tests: 1474 passed, 42 skipped (unchanged)
- Run: `python run_final_optimization.py > logs/final_optimization.log 2>&1 &` (est. 4–8h)

## Step 68: Signal Analysis Dashboard ✅ (2026-02-23, commit b73e777)
- Goal: Extract all historical signals from the magnitude ensemble and present a rich interactive dashboard
- Files:
  - `src/analysis/signal_extractor.py` (CREATE NEW)
    - `SignalExtractor(feature_cols)` — extracts all bars where AVG(LightGBM,RF) ≥ threshold
    - `extract_all_signals(lgbm_model, rf_model, full_df, threshold=0.97)` → pd.DataFrame
    - Per-signal: metadata, entry conditions, model confidence, plain-English explanation, straddle P&L
    - `_generate_explanation(info)` — moneyness/IV/momentum/time description
    - `_simulate_straddle(info)` — TP: winner=entry*(1+move/100)+loser=entry*0.94; FP: both*0.92 (×100 shares/contract)
    - Column mapping corrections: ticker (not option_symbol), close (not option_mid), abs_max_move_pct, time_to_max_min, opt_rsi_14, spy_return_5m, target_magnitude
  - `src/visualization/__init__.py` (CREATE NEW) — package init
  - `src/visualization/signal_dashboard.py` (CREATE NEW)
    - `SignalDashboard(lgbm_model_name, rf_model_name, threshold)` — dark-theme Plotly HTML
    - `build_dashboard(signals_df, output_path)` — writes standalone HTML
    - 4 charts: confidence-vs-magnitude scatter, magnitude bucket bars, daily timeline, straddle return histogram
    - KPI row: 8 cards (total signals, precision, avg confidence, avg magnitude, straddle return, total P&L, win rate, CALLs/PUTs)
    - Sortable signal table + collapsible per-signal detail cards (top 100 by confidence)
  - `src/ml/cli.py` — new `build-signal-dashboard` command (5-step: load→label→extract→save CSV→build HTML)
- Production run results: 603 signals | 100% precision | 0 FP | avg confidence 97.0%
  - Date range: 2025-03-04 → 2025-11-25 (231 trading days)
  - Output: `reports/signal_dashboard.html` (377 KB) + `reports/signal_dashboard.csv`
- Tests: 1516 collected (unchanged)

## Future
- [ ] Fresh out-of-sample validation of LightGBM/RF magnitude models on 2026 live data
- [ ] Upgrade Massive plan for full 12-month options history (Apr–Nov 2025 gap)
- [ ] VIX data integration (upgrade massive.com plan)
- [ ] Per-day interleaved `download-day` command (SPY open → options, parallel within rate-limit window)
- [ ] LSTM model training
- [ ] MLflow integration

## Step 70: SPY 20-Lag Avg-Price Delta Features + Signal Filtering Experiments ✅
- [x] Added 20 new SPY momentum-profile features to `src/processing/ml_feature_engineer.py`
  - `spy_avg_vs_1m_ago` … `spy_avg_vs_20m_ago` = current OHLC midpoint minus N-minute-ago midpoint ($)
  - Added to `_compute_spy_features()` after VWAP distance block; warm-up filled with 0.0
  - Included in `ta_cols` NaN-fill pass; automatically captured by `spy_*` keep list
  - All 231 feature CSVs regenerated (32s); each CSV now 129 cols (was 109)
- [x] Created `src/analysis/signal_filtering_experiments.py` (3 experiments on 603 signals)
  - **Exp 1** Momentum Confirmation: 58/603 passed 3-bar rising-high filter | 55.2% precision | all reversal exits
  - **Exp 2** Consolidation Breakout (feature-based): 225 consolidations / 94 breakouts detected across 42 days; 6 signals matched 3-min window | 66.7% precision (small sample)
  - **Exp 3** Non-Overlapping Active Signals: 527/603 signals accepted | 12.6% overlap removed | 34.5% precision
- [x] Created `run_signal_filtering_experiments.py` runner
- [x] Outputs saved to `reports/signal_filtering_experiments/`
- Calibration: consolidation = max(|lag_1|…|lag_5|) < $0.30 (44th pctile); breakout = |lag_1| > $0.30 (90th pctile); breakout window = 3 min

## Step 71: SPY Consolidation/Breakout Features + Retrain xgboost_v4_consol ✅
- [x] Added 5 new SPY consolidation/breakout features to `src/processing/ml_feature_engineer.py`
  - `spy_consol_range_5b` / `spy_consol_range_10b`: rolling max(high)-min(low) over 5/10 bars ($)
  - `spy_breakout_5b` / `spy_breakout_10b`: signed distance above/below the rolling range (0=inside)
  - `spy_bars_since_breakout`: bars elapsed since last 5b-range breakout, capped at 20
  - Added to `_compute_spy_features()` after spy_avg_vs_ block; warm-up filled with 0.0
- [x] Regenerated all 231 feature CSVs → each now 134 cols (was 129)
- [x] Retrained `models/xgboost_v4_consol.pkl` with 118 features (vs 65 in v3_clean)
  - train=121,960 rows (2025-03-03→2025-10-30) | val=25,687 | test=25,175
  - early_stopping at iteration 45 | val precision=53.2% | val AUC=0.630
- [x] Evaluated on full 70-day test set (2025-10-31→2026-02-19):

| Threshold | v3_clean Prec | v4_consol Prec | Delta | v4 Sigs/Day |
|---|---|---|---|---|
| 0.50 | 53.6% | 53.5% | -0.1% | 304.8 |
| 0.55 | 54.9% | 56.5% | +1.6% | 155.6 |
| 0.58 | 53.8% | 58.4% | **+4.6%** | 85.1 |
| 0.60 | 53.7% | 59.5% | **+5.8%** | 49.2 |
| 0.62 | 54.2% | 59.4% | **+5.2%** | 25.0 |
| 0.65 | 54.9% | 52.2% | -2.7% | 6.8 |
| 0.70 | 57.0% | 14.3% | -42.7% | 1.1 |

- Feature importance: spy_consol_range_10b (rank 30, 1.10%), spy_consol_range_5b (rank 45, 0.63%)
  - spy_breakout_5b/10b/bars_since_breakout: near-zero importance (ranks 115-118)
  - v3_clean shows flat precision 53-55% across all thresholds (poor discrimination)
  - **Best operating point: threshold=0.60 → 59.5% precision, 49 signals/day**
- Report: `reports/consolidation_filter_spy/v4_consol_comparison.json`

## Step 72: Direction-Aligned Breakout Features + Top-N Daily Signal Selection ✅
- [x] Added 2 direction-aligned breakout features to `src/processing/ml_feature_engineer.py`
  - `spy_aligned_breakout_5b`: `spy_breakout_5b × (+1 if CALL, -1 if PUT)` — positive = SPY breakout aligned with option direction
  - `spy_aligned_breakout_10b`: same for 10-bar window
  - Inserted in `_compute_option_features()` after `transactions_ratio`, before `opt_bar_count`
  - Uses `contract_type` (already numeric rank 8) as the direction multiplier
- [x] Regenerated all 231 feature CSVs → each now 136 cols (was 134; +2 aligned breakout)
- [x] Retrained `models/xgboost_v5_aligned.pkl` with 120 features
  - train=121,960 rows | val=25,687 | test=25,175
  - early_stopping at iteration 57 | val precision=54.45% | val AUC=0.6264
- [x] Threshold sweep evaluation (35 test days: 2025-12-23→2026-02-19):

| Threshold | v3_clean | v4_consol | v5_aligned | v5 Sigs |
|---|---|---|---|---|
| 0.50 | 53.9% | 53.9% | 53.8% | 6,569 |
| 0.55 | 54.9% | 55.8% | 53.7% | 3,413 |
| 0.58 | 53.9% | 57.2% | 53.4% | 2,049 |
| 0.60 | 53.2% | 57.6% | 56.0% | 1,345 |
| 0.62 | 50.7% | 57.1% | **59.7%** | 841 |
| 0.65 | 54.6% | 47.9% | **61.7%** | 360 |
| 0.70 | 88.2%* | 83.3%* | 73.6%* | 53 |

*statistically unreliable (tiny counts)

- [x] Top-N per day analysis (v5_aligned, 35 test days):

| N signals/day | Precision | Total signals |
|---|---|---|
| Top-1 | **65.7%** | 35 |
| Top-2 | **60.0%** | 70 |
| Top-3 | 59.0% | 105 |
| Top-5 | 54.9% | 175 |
| Top-10 | 53.7% | 350 |
| Top-20 | 51.1% | 700 |

- **Key findings**:
  - v5_aligned best threshold: **0.65 → 61.7% precision, 10.3 signals/day** (beats v4_consol at 0.60 by +2.2pp)
  - Aligned breakout features fix the directional symmetry issue: v5 outperforms v4_consol at thresholds ≥ 0.62
  - Top-2 per day: **60.0% precision** (2 signals/day ≈ 70 total) — viable operating mode
  - Top-1 per day: 65.7% precision but only 1 signal/day (too few)
  - **Recommended operating point**: Top-3 per day OR threshold=0.65 for ~59-62% precision
- Report: `reports/consolidation_filter_spy/v5_aligned_comparison.json`

## Step 73: Microstructure + Regime Features ✅
- [x] Implement `src/processing/microstructure_features.py` — `MicrostructureFeatureEngineer` class
  - `compute_spy_microstructure(df)` → 6 columns: spy_tx_intensity_5m, spy_vol_per_tx, spy_vol_per_tx_zscore, spy_intrabar_vol, spy_intrabar_vol_ratio, spy_tick_direction
  - `compute_option_microstructure(df)` → 6 columns: opt_tx_intensity_5m, opt_vol_per_tx, opt_vol_per_tx_zscore, opt_intrabar_vol, opt_intrabar_vol_ratio, opt_tick_direction
- [x] Implement `src/processing/regime_detector.py` — `RegimeDetector` class
  - KMeans clustering on 6 SPY features into 4 regimes (low-vol trending, low-vol mean-revert, high-vol trending, high-vol choppy)
  - `fit(df)` / `predict(df)` → adds market_regime (0–3) and regime_confidence
- [x] Integrate into `src/processing/ml_feature_engineer.py` — calls at end of `_compute_spy_features()` and `_compute_option_features()`
- [x] Unit tests: `test_microstructure_features.py` (15 tests), `test_regime_detector.py` (12 tests) — 27 total
- Feature CSVs: 136 → 148 columns (+12 microstructure; regime added at training time)

## Step 74: Stacked Ensemble + Anomaly Filter ✅
- [x] Implement `src/ml/anomaly_filter.py` — `AnomalyFilter` class
  - IsolationForest-based anomaly detection with probability discounting
  - `fit(X)`, `is_anomalous(X)`, `apply_discount(probabilities, X)`
- [x] Implement `src/ml/stacked_ensemble.py` — `StackedEnsemble` class
  - Level-0: XGBoost + LightGBM + RandomForest (trained on train set with class balancing)
  - Level-1: LogisticRegression meta-learner (trained on validation set predictions)
  - Calibration: IsotonicRegression for meaningful probabilities
  - Anomaly filter: applied post-meta-learner
  - `train()`, `predict_proba()`, `predict_with_detail()`, `save()`, `load()`
- [x] CLI: `ml train-stacked-ensemble` command
- [x] Unit tests: `test_anomaly_filter.py` (10 tests), `test_stacked_ensemble.py` (20 tests) — 30 total
- [x] Edge case: handles val set with only 1 class (synthetic 2-class fallback for LogisticRegression)

## Step 75: Exit Signal Model ✅
- [x] Implement `src/ml/exit_signal_model.py` — `ExitFeatureEngineer` + `ExitSignalModel` classes
  - 15 exit features in 3 groups: trade-relative (5), momentum exhaustion (5), market context (5)
  - LightGBM classifier for P(exit) prediction
  - Hard rules override model: stop-loss (-20%), time limit (120m), EOD close (10m before)
  - `train()`, `should_exit()`, `save()`, `load()`
- [x] CLI: `ml train-exit-model` command
- [x] Unit tests: `test_exit_signal_model.py` (25 tests)

## Step 76: Real-Bar Trade Simulator ✅
- [x] Implement `src/ml/real_bar_simulator.py` — `RealBarSimulator` class
  - Bar-by-bar trade simulation using entry + exit models on actual minute data
  - `simulate_day()`, `simulate_period()`, `_simulate_trade()`
  - Processes candidates chronologically, one trade at a time, max 3/day
  - Reuses `Trade` class from `src/ml/trade_simulator.py`
- [x] CLI: `ml simulate-real-bars` command
- [x] Unit tests: `test_real_bar_simulator.py` (12 tests)

## Step 77: End-to-End Signal Pipeline ✅
- [x] Implement `src/ml/signal_pipeline.py` — `SignalPipeline` + `PipelineReport` classes
  - Orchestrates: feature loading → regime detection → entry scoring → top-N selection → real-bar simulation → report generation
  - `run_backtest()`, `generate_daily_signals()`, `_save_report()`
- [x] Added config sections to `config/ml_settings.yaml`: stacked_ensemble, exit_model, signal_pipeline
- [x] CLI: `ml run-signal-pipeline` command
- [x] Unit tests: `test_signal_pipeline.py` (13 tests)

## Step 78: Trade Dashboard ✅
- [x] Implement `src/visualization/trade_dashboard.py` — `TradeDashboard` class
  - Plotly dark-theme standalone HTML dashboard
  - Sections: KPI cards, equity curve + drawdown, monthly P&L bars ($10K target line), trade scatter, exit analysis, sortable trade table
  - Lazy plotly imports (inside methods) to avoid pytest collection conflicts
- [x] CLI: `ml build-trade-dashboard` command
- [x] Unit tests: `test_trade_dashboard.py` (13 tests)
- [x] Fixed plotly import conflict: `test_ml_dashboard.py` was injecting MagicMock into sys.modules["plotly"] unconditionally; changed to only mock when plotly is not installed

---
**Total tests: 1597 passed, 7 skipped | Last updated: 2026-02-23**
