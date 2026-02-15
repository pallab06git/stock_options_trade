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

## Step 14: VIX Data (Week 5)
- [ ] VIX historical (Polygon I:VIX)
- [ ] VIX real-time (WebSocket)
- [ ] Integration test

## Step 15: News Data (Week 5)
- [ ] Implement src/data_sources/news_client.py
- [ ] Sentiment extraction
- [ ] 5-min polling (configurable)
- [ ] Integration test

## Step 16: Data Consolidation (Week 6)
- [ ] Implement src/processing/consolidator.py
- [ ] Time alignment (SPY + Options + VIX + News)
- [ ] Compute Greeks (py_vollib)
- [ ] Compute momentum arrays
- [ ] Compute technical indicators (pandas-ta)
- [ ] On-demand execution
- [ ] Integration test (1 day consolidated)

## Step 17: Schema Drift Detection (Week 6)
- [ ] Implement src/monitoring/schema_monitor.py
- [ ] Daily alerts
- [ ] Continue processing despite drift

## Step 18: Late Data Handler (Week 6)
- [ ] Implement src/processing/late_data_handler.py
- [ ] Configurable rejection window
- [ ] Quarantine late data

## Step 19: Feed Simulator (Week 7)
- [ ] Implement src/orchestrator/simulator.py
- [ ] Replay historical data as real-time
- [ ] Configurable playback speed
- [ ] All sources supported

## Step 20: Integration Tests (Week 7)
- [ ] tests/integration/test_historical_flow.py
- [ ] tests/integration/test_realtime_flow.py
- [ ] tests/integration/test_consolidation.py
- [ ] Full pipeline tests

## Step 21: Documentation (Week 8)
- [ ] Update README.md with examples
- [ ] Create docs/API_REFERENCE.md
- [ ] Create config/examples/
- [ ] Full system test (1 week data)
- [ ] Generate work log

## Future (Phase 2 - ML)
- [ ] Feature engineering pipeline
- [ ] LSTM model training
- [ ] Signal validator implementation
- [ ] MLflow integration
- [ ] Backtesting framework

---
**Total tests: 367 passing + 7 live (skipped outside market hours) | Last updated: 2026-02-14**
