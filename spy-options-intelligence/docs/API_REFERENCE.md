# API Reference

Module-by-module reference for the SPY Options Intelligence platform.

---

## Data Sources

### BaseSource

**Module**: `src.data_sources.base_source`

Abstract base class for all data source implementations.

```python
BaseSource(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Establish connection to data source |
| `disconnect` | `() -> None` | Close connection to data source |
| `fetch_historical` | `(start_date: str, end_date: str, **kwargs) -> Generator[Dict]` | Yield historical records for a date range |
| `stream_realtime` | `(**kwargs) -> Generator[Dict]` | Yield real-time records via WebSocket |
| `validate_record` | `(record: Dict) -> bool` | Validate a single data record |

---

### PolygonEquityClient

**Module**: `src.data_sources.polygon_client`

Fetch equity per-second aggregate bars from Polygon.io REST and WebSocket APIs.

```python
PolygonEquityClient(
    config: Dict[str, Any],
    connection_manager: ConnectionManager,
    ticker: str = "SPY"
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Verify REST client availability |
| `disconnect` | `() -> None` | No-op (ConnectionManager owns lifecycle) |
| `fetch_historical` | `(start_date: str, end_date: str) -> Generator[Dict]` | Yield equity aggregates for a date range |
| `stream_realtime` | `(**kwargs) -> Generator[Dict]` | Stream real-time aggregates via WebSocket. Accepts `stop_event` kwarg |
| `validate_record` | `(record: Dict) -> bool` | Validate required fields, timestamp, OHLC prices |

**Example**:
```python
from src.data_sources.polygon_client import PolygonEquityClient
from src.utils.connection_manager import ConnectionManager

cm = ConnectionManager(config)
client = PolygonEquityClient(config, cm, ticker="SPY")
client.connect()

for record in client.fetch_historical("2025-01-02", "2025-01-03"):
    print(record["close"], record["volume"])
```

---

### PolygonOptionsClient

**Module**: `src.data_sources.polygon_options_client`

Discover and stream SPY options contracts within a configurable strike range.

```python
PolygonOptionsClient(
    config: Dict[str, Any],
    connection_manager: ConnectionManager
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `fetch_opening_price` | `(date: str) -> float` | Get underlying ticker's opening price for a date |
| `discover_contracts` | `(date: str, opening_price: float) -> List[Dict]` | Find contracts within ±1% strike range |
| `save_contracts` | `(contracts: List[Dict], date: str) -> Path` | Persist contracts as JSON |
| `load_contracts` | `(date: str) -> List[Dict]` | Load saved contracts for a date |
| `stream_realtime` | `(date: str, stop_event: Event = None) -> Generator[Dict]` | Stream real-time options aggregates via WebSocket |

**Example**:
```python
from src.data_sources.polygon_options_client import PolygonOptionsClient
from src.utils.connection_manager import ConnectionManager

cm = ConnectionManager(config)
client = PolygonOptionsClient(config, cm)

opening = client.fetch_opening_price("2025-01-15")
contracts = client.discover_contracts("2025-01-15", opening)
path = client.save_contracts(contracts, "2025-01-15")
```

---

### PolygonVIXClient

**Module**: `src.data_sources.polygon_vix_client`

Fetch VIX index aggregates from Polygon.io (ticker `I:VIX`).

```python
PolygonVIXClient(
    config: Dict[str, Any],
    connection_manager: ConnectionManager
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Verify REST client availability |
| `disconnect` | `() -> None` | No-op |
| `fetch_historical` | `(start_date: str, end_date: str) -> Generator[Dict]` | Yield VIX aggregates for a date range |
| `stream_realtime` | `(**kwargs) -> Generator[Dict]` | Stream real-time VIX via WebSocket (Market.Indices) |
| `validate_record` | `(record: Dict) -> bool` | Validate VIX record fields and OHLC |

**Example**:
```python
from src.data_sources.polygon_vix_client import PolygonVIXClient
from src.utils.connection_manager import ConnectionManager

cm = ConnectionManager(config)
client = PolygonVIXClient(config, cm)

for record in client.fetch_historical("2025-01-02", "2025-01-03"):
    print(record["close"], record["source"])  # source="vix"
```

---

### PolygonNewsClient

**Module**: `src.data_sources.news_client`

Fetch news articles from Polygon.io with sentiment extraction.

```python
PolygonNewsClient(
    config: Dict[str, Any],
    connection_manager: ConnectionManager
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Verify REST client availability |
| `disconnect` | `() -> None` | No-op |
| `fetch_historical` | `(start_date: str, end_date: str) -> Generator[Dict]` | Yield news articles for a date range |
| `stream_realtime` | `(**kwargs) -> Generator[Dict]` | Poll for new articles at configurable interval |
| `validate_record` | `(record: Dict) -> bool` | Validate timestamp and title fields |

**Example**:
```python
from src.data_sources.news_client import PolygonNewsClient
from src.utils.connection_manager import ConnectionManager

cm = ConnectionManager(config)
client = PolygonNewsClient(config, cm)

for article in client.fetch_historical("2025-01-15", "2025-01-15"):
    print(article["title"], article["sentiment"])
```

---

### MinuteDownloader

**Module**: `src.data_sources.minute_downloader`

Bulk-download SPY (and optionally VIX) per-minute bars for a full calendar month.

```python
MinuteDownloader(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `download_month` | `(ticker: str, year: int, month: int) -> int` | Download all trading days in a month; returns record count |
| `download_range` | `(ticker: str, start_date: str, end_date: str) -> int` | Download an arbitrary date range; returns record count |

**Example**:
```python
from src.data_sources.minute_downloader import MinuteDownloader

dl = MinuteDownloader(config)
count = dl.download_month("SPY", year=2025, month=3)
# Writes data/raw/spy/2025-03-01.parquet, 2025-03-03.parquet, ...
```

---

### TargetedOptionsDownloader

**Module**: `src.data_sources.targeted_options_downloader`

Download a small, targeted set of options per day (2 ATM calls + 2 ATM puts) to stay within free-tier API limits.

```python
TargetedOptionsDownloader(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `download_day` | `(date: str) -> List[Dict]` | Download ATM options for one trading day |
| `run` | `(start_date: str, end_date: str) -> Dict[str, Any]` | Resilient batch run; skips dates with no data; returns stats |

**Example**:
```python
from src.data_sources.targeted_options_downloader import TargetedOptionsDownloader

dl = TargetedOptionsDownloader(config)
stats = dl.run("2025-03-01", "2025-03-31")
```

---

### OptionsTickerBuilder

**Module**: `src.data_sources.options_ticker_builder`

Pure math helper for constructing options tickers and computing ATM strikes. All methods are `@staticmethod` — no config, no API calls, no I/O.

```python
OptionsTickerBuilder  # no constructor — use static methods directly
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `build_ticker` | `(underlying: str, strike: float, contract_type: str, expiry_date: str) -> str` | Format `O:SPY250304C00601000`-style ticker. `contract_type` is `"call"` or `"put"` (case-insensitive). `expiry_date` is `YYYY-MM-DD`. |
| `compute_strikes` | `(opening_price: float, n_calls: int, n_puts: int, strike_increment: float) -> Dict[str, List[float]]` | Return `{"calls": [...], "puts": [...]}`. Calls are strictly above opening; puts are at or below. If opening lands exactly on a boundary, calls start one increment up. |
| `next_trading_day` | `(date: str) -> str` | Return first Mon–Fri strictly after `date` (YYYY-MM-DD), skipping weekends. |
| `next_friday` | `(date: str) -> str` | Return first Friday strictly after `date`. |

**Example**:
```python
from src.data_sources.options_ticker_builder import OptionsTickerBuilder

ticker = OptionsTickerBuilder.build_ticker("SPY", 601.0, "call", "2025-03-04")
# → "O:SPY250304C00601000"

strikes = OptionsTickerBuilder.compute_strikes(600.25, n_calls=2, n_puts=2, strike_increment=0.5)
# → {"calls": [601.0, 601.5], "puts": [600.0, 599.5]}
```

---

### ContractSelector

**Module**: `src.data_sources.contract_selector`

Handshake module between contract discovery and bar download. Returns a uniform contract list regardless of mode.

```python
ContractSelector(
    config: Dict[str, Any],
    mode: str = "test",           # "test" or "prod"
    api_key: Optional[str] = None,  # required for prod mode
    _input_fn: Callable = input,  # injectable for testing
)
```

| Method / Property | Signature | Description |
|--------|-----------|-------------|
| `needs_prompt` | `-> bool` (property) | `True` if TEST mode and `prompt_once()` has not been called yet |
| `underlying` | `-> str` (property) | Returns underlying ticker from PROD config or TEST params |
| `prompt_once` | `() -> None` | Interactive: ask user once for underlying, increment, n_calls, n_puts, expiry convention. Idempotent — safe to call multiple times. Raises `RuntimeError` in PROD mode. |
| `get_contracts` | `(date: str, opening_price: float) -> List[Dict]` | Return filtered contract list. TEST: constructs tickers via `OptionsTickerBuilder`. PROD: calls `massive.list_options_contracts()`, tries up to `expiration_search_days` expiries. Auto-calls `prompt_once()` in TEST if not yet called. |
| `_resolve_expiry` | `(date: str, convention: str, offset: int = 1, fixed_date: str = None) -> str` | Resolve an expiry convention to a `YYYY-MM-DD` string. `offset` shifts the base date by that many calendar days (used for PROD retry loop). |

Output schema per contract dict:
```python
{
    "ticker":        str,   # e.g. "O:SPY250304C00601000"
    "strike":        float, # e.g. 601.0
    "contract_type": str,   # "call" or "put"
    "expiry_date":   str,   # "YYYY-MM-DD"
    "underlying":    str,   # e.g. "SPY"
}
```

**Example**:
```python
from src.data_sources.contract_selector import ContractSelector

# TEST mode — interactive prompt once, then reuse for all dates
sel = ContractSelector(config, mode="test")
sel.prompt_once()  # asks: SPY / 1.0 / 1 / 1 / next_trading_day
contracts = sel.get_contracts("2025-03-03", 600.25)
# → [{"ticker": "O:SPY250304C00601000", "strike": 601.0, ...}, ...]
```

---

### MassiveOptionsDownloader

**Module**: `src.data_sources.massive_options_downloader`

Generic parallel options minute-bar downloader via `massive.RESTClient.list_aggs()`. Contains zero ticker-construction or strike-selection logic — all of that is delegated to `ContractSelector`.

```python
MassiveOptionsDownloader(
    config: Dict[str, Any],
    api_key: str,
    selector: ContractSelector,
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_config` | `(config, selector) -> MassiveOptionsDownloader` (classmethod) | Construct from config, resolving API key automatically: `MASSIVE_API_KEY` → `massive.api_key` → `POLYGON_API_KEY` → `polygon.api_key`. |
| `get_opening_price` | `(date: str, underlying: str = "SPY") -> float` | Read opening price from local Parquet — no API call. Uses `data/raw/{underlying.lower()}/{date}.parquet`. |
| `download_tickers` | `(contracts: List[Dict], date: str, resume: bool = True) -> int` | Download bars for multiple contracts in parallel via `ThreadPoolExecutor`. Returns total bars written. |
| `run` | `(start_date: str, end_date: str, resume: bool = True) -> Dict[str, Any]` | Full pipeline for a date range. Calls `selector.prompt_once()` once before the loop. Returns stats: `{dates_processed, dates_skipped, contracts_found, total_bars}`. |

Output path: `data/raw/options/minute/{safe_ticker}/{date}.parquet`
`safe_ticker` = ticker with `:` replaced by `_`, e.g. `O_SPY250304C00601000`.

**Example**:
```python
from src.data_sources.contract_selector import ContractSelector
from src.data_sources.massive_options_downloader import MassiveOptionsDownloader

sel = ContractSelector(config, mode="test")
dl = MassiveOptionsDownloader.from_config(config, sel)
stats = dl.run("2025-03-01", "2025-03-31")
# → {"dates_processed": 21, "dates_skipped": 8, "contracts_found": 42, "total_bars": 3541}
```

---

## Sinks

### BaseSink

**Module**: `src.sinks.base_sink`

Abstract base class for data sink implementations.

```python
BaseSink(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Establish connection to sink |
| `disconnect` | `() -> None` | Close connection |
| `write_batch` | `(records: List[Dict], partition_key: str = None) -> None` | Write batch of records |
| `write_single` | `(record: Dict, partition_key: str = None) -> None` | Write a single record |
| `check_duplicate` | `(record: Dict) -> bool` | Check if record already exists |
| `overwrite` | `(records: List[Dict], partition_key: str) -> None` | Replace all data in a partition |

---

### ParquetSink

**Module**: `src.sinks.parquet_sink`

Write records to date-partitioned Parquet files with Snappy compression.

```python
ParquetSink(
    config: Dict[str, Any],
    dedup_subset: List[str] = None    # Compound dedup keys (e.g. ["ticker", "timestamp"])
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> None` | Ensure base directory exists |
| `disconnect` | `() -> None` | No-op |
| `write_batch` | `(records: List[Dict], partition_key: str = None) -> None` | Write batch to date-partitioned Parquet |
| `write_single` | `(record: Dict, partition_key: str = None) -> None` | Delegate to `write_batch` |
| `check_duplicate` | `(record: Dict) -> bool` | Check if timestamp exists in partition file |
| `overwrite` | `(records: List[Dict], partition_key: str) -> None` | Replace all data in a partition |

**Example**:
```python
from src.sinks.parquet_sink import ParquetSink

sink = ParquetSink(config)
sink.connect()
sink.write_batch(records, partition_key="2025-01-15")
# File written to: data/raw/spy/2025-01-15.parquet
```

---

## Processing

### RecordValidator

**Module**: `src.processing.validator`

Validate data records against per-source schemas (equity, vix, options, news).

```python
RecordValidator(source: str)    # "equity", "vix", "options", "news"
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `for_equity` | `(cls, ticker: str) -> RecordValidator` | Class method — create validator for any equity ticker |
| `validate` | `(record: Dict) -> bool` | Validate a single record |
| `validate_batch` | `(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]` | Split into (valid, invalid) lists |
| `get_validation_errors` | `(record: Dict) -> List[str]` | Human-readable error strings |

**Example**:
```python
from src.processing.validator import RecordValidator

validator = RecordValidator("equity")
valid, invalid = validator.validate_batch(records)
```

---

### Deduplicator

**Module**: `src.processing.deduplicator`

Track and filter duplicate records using an in-memory set.

```python
Deduplicator(key_field: str = "timestamp")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_duplicate` | `(record: Dict) -> bool` | Check and track a record |
| `deduplicate_batch` | `(records: List[Dict]) -> List[Dict]` | Remove duplicates (keep last occurrence) |
| `reset` | `() -> None` | Clear all tracked keys |
| `seen_count` | `-> int` (property) | Number of unique keys tracked |

**Example**:
```python
from src.processing.deduplicator import Deduplicator

dedup = Deduplicator(key_field="article_id")
unique = dedup.deduplicate_batch(articles)
```

---

### Consolidator

**Module**: `src.processing.consolidator`

Merge SPY, VIX, Options, and News into a per-option-per-minute enriched dataset with technical indicators, Greeks, momentum, and sentiment.

```python
Consolidator(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `consolidate` | `(date: str) -> Dict[str, Any]` | Run full pipeline for a trading day; returns stats |

**Pipeline steps**: Load sources → aggregate to 1-min bars → align VIX via `merge_asof` → compute RSI, MACD, Bollinger Bands → compute momentum (ROC) → attach news sentiment → flatten per-option rows → compute Greeks (delta, gamma, theta, vega, rho, IV).

**Example**:
```python
from src.processing.consolidator import Consolidator

consolidator = Consolidator(config)
stats = consolidator.consolidate("2025-01-15")
# Output: data/processed/consolidated/2025-01-15.parquet
```

---

### TrainingDataPrep

**Module**: `src.processing.training_data_prep`

Generate ML training datasets from consolidated data by adding forward-looking target prices.

```python
TrainingDataPrep(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `prepare` | `(dates: List[str]) -> Dict[str, Any]` | Add target_future_prices columns and filter by coverage |

**Example**:
```python
from src.processing.training_data_prep import TrainingDataPrep

prep = TrainingDataPrep(config)
stats = prep.prepare(["2025-01-15", "2025-01-16"])
# Output: data/processed/training/2025-01-15.parquet, ...
```

---

### FeatureEngineer

**Module**: `src.processing.feature_engineer`

Compute lagged percentage-change features and implied-volatility features from SPY minute bars.

```python
FeatureEngineer(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `engineer` | `(date: str) -> Dict[str, Any]` | Compute features for one trading day; returns stats |
| `engineer_range` | `(start_date: str, end_date: str) -> Dict[str, Any]` | Compute features for a date range; returns stats |

**Example**:
```python
from src.processing.feature_engineer import FeatureEngineer

fe = FeatureEngineer(config)
stats = fe.engineer("2025-03-04")
# Output: data/processed/features/2025-03-04.parquet
```

---

### OptionsScanner

**Module**: `src.processing.options_scanner`

Scan options feature files for intraday moves ≥20% relative to a rolling reference window minimum. Emits a CSV report and prints 8 summary metrics to console.

```python
OptionsScanner(config: Dict[str, Any])
```

**Config keys** (`config["pipeline_v2"]["scanner"]`): `reference_window_minutes` (default 120), `trigger_threshold_pct` (default 20.0), `sustained_threshold_pct` (default 10.0).

**State**: `_last_scan_stats: Dict[str, Any]` — populated by `scan()`, consumed by `generate_report()`. Keys: `contract_days`, `total_bars`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `scan` | `(start_date: str, end_date: str) -> List[Dict]` | Scan all options feature Parquets in range; updates `_last_scan_stats` |
| `generate_report` | `(events, start_date, end_date) -> Path` | Write CSV + print 8 metrics: contract-days, bars, events, events/cday (min/med/max), >20% mins, rate%, duration med/mean, hour distribution |
| `load_reports` | `(start_date=None, end_date=None) -> DataFrame` | Load all movement CSVs, optionally filtered by date |
| `_scan_single` | `(feat_path, safe_ticker, date, _df=None) -> List[Dict]` | Scan one Parquet; accepts pre-loaded DataFrame to avoid double reads |

**Console output** (from `generate_report`):
```
Contract-days scanned:    125
Total minute bars:        44971
Total events:             544
Events/contract-day:      min=0 median=4.0 max=12
Total >20% minutes:       24929
Positive-minute rate:     55.43%
Duration >20% (med/mean): 8.5 / 45.8 min

Event distribution by trigger hour (ET):
  09:xx   104  ###########################
  10:xx   115  ##############################
```

**Example**:
```python
from src.processing.options_scanner import OptionsScanner

scanner = OptionsScanner(config)
events = scanner.scan("2025-03-03", "2026-02-19")
path = scanner.generate_report(events, "2025-03-03", "2026-02-19")
# Output: data/reports/options_movement/2025-03-03_2026-02-19_movement.csv
```

---

## Orchestration

### HistoricalRunner

**Module**: `src.orchestrator.historical_runner`

Orchestrate historical data ingestion: Source → Validator → Deduplicator → ParquetSink with checkpoint/resume.

```python
HistoricalRunner(
    config: Dict[str, Any],
    ticker: str = "SPY",
    connection_manager: ConnectionManager = None,
    client: BaseSource = None,
    validator: RecordValidator = None,
    deduplicator: Deduplicator = None
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(resume: bool = False) -> Dict[str, Any]` | Execute full pipeline; returns stats dict |

**Example**:
```python
from src.orchestrator.historical_runner import HistoricalRunner

runner = HistoricalRunner(config, ticker="SPY")
stats = runner.run(resume=True)
```

---

### StreamingRunner

**Module**: `src.orchestrator.streaming_runner`

Orchestrate real-time streaming: WebSocket → Buffer → Validator → Deduplicator → ParquetSink with heartbeat monitoring and market hours enforcement.

```python
StreamingRunner(
    config: Dict[str, Any],
    ticker: str = "SPY",
    connection_manager: ConnectionManager = None,
    client: BaseSource = None,
    validator: RecordValidator = None,
    deduplicator: Deduplicator = None
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `() -> Dict[str, Any]` | Execute streaming pipeline; returns stats dict |

---

### OptionsStreamingRunner

**Module**: `src.orchestrator.options_streaming_runner`

Orchestrate real-time options streaming with compound deduplication (ticker + timestamp).

```python
OptionsStreamingRunner(config: Dict[str, Any], date: str)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `() -> Dict[str, Any]` | Execute options streaming pipeline; returns stats dict |

---

### FeedSimulator

**Module**: `src.orchestrator.simulator`

Replay historical Parquet data as a simulated real-time stream with configurable playback speed.

```python
FeedSimulator(
    config: Dict[str, Any],
    source: str,          # "spy", "vix", "options", "news", "consolidated"
    date: str,            # YYYY-MM-DD
    speed: float = 1.0    # 1.0 = real-time, 10.0 = 10x, 0 = no delay
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_records` | `() -> List[Dict]` | Load and sort records from Parquet |
| `stream_realtime` | `(**kwargs) -> Generator[Dict]` | Yield records with simulated delays |
| `get_stats` | `() -> Dict[str, Any]` | Return simulation statistics |

**Example**:
```python
from src.orchestrator.simulator import FeedSimulator

sim = FeedSimulator(config, source="spy", date="2025-01-15", speed=10.0)
for record in sim.stream_realtime():
    process(record)
```

---

### ParallelRunner

**Module**: `src.orchestrator.parallel_runner`

Spawn one subprocess per ticker for parallel backfill of multiple equities.

```python
ParallelRunner(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(start_date, end_date, resume=False, config_dir="config") -> Dict[str, Dict]` | Spawn workers, wait for results |
| `load_registry` | `(cls) -> Dict` | Class method — load process registry from disk |

---

### ProcessManager

**Module**: `src.orchestrator.process_manager`

Start, stop, and query worker processes via the shared process registry.

```python
ProcessManager(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_workers` | `() -> List[Dict]` | List all workers with PID, status, alive flag |
| `stop_worker` | `(ticker: str) -> bool` | Send SIGTERM to a worker by ticker |
| `stop_all` | `() -> Dict[str, bool]` | Send SIGTERM to all running workers |

---

## Monitoring

### PerformanceMonitor

**Module**: `src.monitoring.performance_monitor`

Track pipeline performance metrics (latency, throughput, memory) and alert on threshold breaches.

```python
PerformanceMonitor(
    config: Dict[str, Any],
    session_label: str = "default"
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `start_operation` | `(operation: str) -> None` | Begin timing an operation |
| `end_operation` | `(operation: str, record_count: int = 0) -> List[str]` | Stop timer, record metrics, return alerts |
| `check_stale_operations` | `(timeout_seconds: float = None) -> List[str]` | Detect hung operations |
| `check_alerts` | `(operation, elapsed, record_count) -> List[str]` | Evaluate alert thresholds |
| `get_latency_stats` | `(operation: str) -> Dict[str, float]` | Compute p50, p95, p99 latency |
| `get_throughput` | `(operation: str, window_seconds: float = 60.0) -> float` | Average records/second |
| `get_memory_usage_mb` | `() -> float` | Current process RSS in MB |
| `should_dump_metrics` | `() -> bool` | True if dump interval has elapsed |
| `dump_metrics` | `() -> Path` | Write metrics snapshot to JSON |
| `get_summary` | `() -> Dict[str, Any]` | Summary dict for end-of-run logging |

---

### SchemaMonitor

**Module**: `src.monitoring.schema_monitor`

Detect schema drift between Parquet files and stored baselines.

```python
SchemaMonitor(
    config: Dict[str, Any],
    session_label: str = "default"
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `capture_baseline` | `(source: str, parquet_path: str) -> Dict` | Extract schema from Parquet metadata |
| `save_baseline` | `(source: str, baseline: Dict) -> Path` | Write baseline JSON to disk |
| `load_baseline` | `(source: str) -> Optional[Dict]` | Load baseline from disk |
| `check_drift` | `(source: str, parquet_path: str) -> List[str]` | Compare schema against baseline, return alerts |
| `detect_schema_changes` | `(baseline: Dict, current: Dict) -> Dict` | Pure diff (new, missing, type changes) |
| `format_alerts` | `(source: str, changes: Dict) -> List[str]` | Human-readable alert messages |
| `log_drift` | `(source: str, date: str, changes: Dict) -> Path` | Write drift event JSON |

---

### ErrorAggregator

**Module**: `src.monitoring.error_aggregator`

Aggregate errors over a sliding window and alert when the error rate exceeds a threshold.

```python
ErrorAggregator(
    config: Dict[str, Any],
    session_label: str = "default"
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `record_error` | `(error_type: str, error_msg: str) -> None` | Record an error occurrence |
| `record_success` | `() -> None` | Record a successful operation |
| `get_error_rate` | `(window_seconds: float = None) -> float` | Error rate as percentage |
| `should_alert` | `() -> bool` | True if error rate exceeds threshold |
| `get_error_summary` | `() -> Dict[str, Any]` | Grouped error counts and current rate |
| `get_recent_errors` | `(error_type: str = None) -> List[Tuple]` | Recent (timestamp, message) tuples |

---

### HealthDashboard

**Module**: `src.monitoring.health_dashboard`

Aggregate per-session monitoring metrics into a unified health view.

```python
HealthDashboard(metrics_dir: str = "data/logs/performance")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_all_sessions` | `() -> Dict[str, Dict]` | Scan metrics dir for latest per-session JSON files |
| `get_health_summary` | `() -> Dict[str, Dict]` | Combine process registry + metrics |
| `get_session_detail` | `(ticker: str) -> Optional[Dict]` | Detailed metrics for one session |
| `format_table` | `(summary: Dict) -> str` | Format health summary as a text table |

---

### HeartbeatMonitor

**Module**: `src.monitoring.heartbeat_monitor`

Monitor real-time stream health via periodic heartbeats written to disk.

```python
HeartbeatMonitor(config: dict, source_name: str)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `record_message` | `() -> None` | Increment message counter for next heartbeat |
| `should_send_heartbeat` | `() -> bool` | True if heartbeat interval has elapsed |
| `send_heartbeat` | `() -> None` | Write heartbeat metadata to file |
| `check_stalled_stream` | `() -> bool` | True if no heartbeat for >15 min |

---

## Utilities

### ConfigLoader

**Module**: `src.utils.config_loader`

Load and merge YAML configuration files with environment variable substitution.

```python
ConfigLoader(config_dir: str = "config", env_file: str = ".env")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `() -> dict` | Load .env, parse YAML files, substitute env vars, merge, validate |
| `reload` | `() -> dict` | Re-read all config files and return updated configuration |

**Example**:
```python
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader(config_dir="config")
config = loader.load()
```

---

### ConnectionManager

**Module**: `src.utils.connection_manager`

Manage Polygon SDK sessions with unified token-bucket rate limiting.

```python
ConnectionManager(config: dict)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_rest_client` | `() -> RESTClient` | Get shared Polygon REST client (lazy init) |
| `get_ws_client` | `(market: Market = Market.Stocks) -> WebSocketClient` | Create a new WebSocket client |
| `acquire_rate_limit` | `(source: str = None) -> bool` | Block until rate-limit token available |
| `handle_rate_limit_response` | `(retry_after: int = None) -> None` | Handle 429 by pausing token bucket |
| `health_check` | `() -> bool` | Check Polygon API connectivity |
| `close` | `() -> None` | Clean up REST client session |

---

### retry_handler — `with_retry`, `RetryableError`, `SkippableError`

**Module**: `src.utils.retry_handler`

Decorator factory for configurable retry logic with exponential backoff and selective skip behaviour.

#### `with_retry` — decorator factory

```python
with_retry(source: str = "default", config: Optional[Dict] = None) -> Callable
```

Wraps a function with retry and skip logic driven by `retry_policy.yaml` profiles. Behaviour by error type:

| Error | Behaviour |
|-------|-----------|
| `RetryableError` (5xx / 429) | Exponential backoff: `initial_wait * base^(attempt−1)`, capped at `max_wait`. Raises after `max_attempts`. |
| `RetryableError` (401 / 403) | Log WARNING + return `None` immediately. Never retried — prevents account lockout. |
| `SkippableError` | Log WARNING + return `None` immediately. No retry — bad data won't improve on retry. |

Config keys per profile: `max_attempts`, `initial_wait_seconds`, `max_wait_seconds`, `exponential_base`, `jitter`, `retry_on_status_codes`.

**Example**:
```python
from src.utils.retry_handler import with_retry, RetryableError, SkippableError

@with_retry(source="polygon", config=app_config)
def fetch():
    raise RetryableError("Server error", status_code=500)  # retried with backoff

@with_retry(source="polygon", config=app_config)
def validate_record(record):
    if record.get("close") is None:
        raise SkippableError("Null close price")  # logged and skipped
```

#### `RetryableError`

```python
RetryableError(message: str, status_code: int = 0)
```

Exception wrapping an HTTP status code. Raised by data source clients to signal retryable or auth-failure conditions.

#### `SkippableError`

```python
SkippableError(message: str)
```

Exception for data quality issues (malformed records, unexpected nulls) and schema drift. Causes the decorated function to log and return `None` without any retry.

---

### MarketHours

**Module**: `src.utils.market_hours`

Check market open/close status using configured hours and NYSE calendar.

```python
MarketHours(config: dict, api_key: str)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_market_open` | `(check_time: datetime = None) -> bool` | True if market is currently open |
| `seconds_until_market_open` | `() -> Optional[int]` | Seconds until next open, or None if open |
| `seconds_until_market_close` | `() -> Optional[int]` | Seconds until close, or None if closed |

---

### Logger

**Module**: `src.utils.logger`

Centralized logging with automatic credential redaction.

| Function | Signature | Description |
|----------|-----------|-------------|
| `setup_logger` | `(config: dict = None) -> Logger` | Configure the loguru logger singleton |
| `get_logger` | `() -> Logger` | Get the configured logger (auto-configures on first call) |
| `redact_sensitive` | `(message: str) -> str` | Replace sensitive patterns with `***REDACTED***` |

**Example**:
```python
from src.utils.logger import setup_logger, get_logger

setup_logger(config)
logger = get_logger()
logger.info("Pipeline started")
```

---

### PurgeManager

**Module**: `src.utils.purge_manager`

Delete old files according to per-category retention policies configured in `settings.yaml`.

```python
PurgeManager(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `purge` | `(category: str = None, retention_days: int = None, dry_run: bool = True) -> Dict[str, Any]` | Delete files older than retention threshold; returns stats. `dry_run=True` by default — set `False` to actually delete. |
| `purge_all` | `(dry_run: bool = True) -> Dict[str, Any]` | Purge all configured categories; returns per-category stats. |

**Example**:
```python
from src.utils.purge_manager import PurgeManager

pm = PurgeManager(config)
stats = pm.purge(category="raw_data", retention_days=30, dry_run=False)
```

---

### SpaceReporter

**Module**: `src.utils.space_reporter`

Walk the `data/` directory tree and report storage usage by category with compression estimates.

```python
SpaceReporter(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `report` | `() -> Dict[str, Any]` | Return size breakdown by category (bytes, file count, compression ratio) |
| `format_table` | `(report: Dict) -> str` | Format the report as a human-readable text table |

**Example**:
```python
from src.utils.space_reporter import SpaceReporter

sr = SpaceReporter(config)
print(sr.format_table(sr.report()))
```

---

### HardwareMonitor

**Module**: `src.utils.hardware_monitor`

Track CPU, memory, and disk usage via psutil with optional function-level profiling decorator.

```python
HardwareMonitor(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `snapshot` | `() -> Dict[str, Any]` | Return current CPU %, memory RSS (MB), and disk usage % |
| `profile` | `(fn: Callable) -> Callable` | Decorator — log resource usage before and after a function call |
| `report` | `() -> str` | Format a human-readable hardware summary string |

**Example**:
```python
from src.utils.hardware_monitor import HardwareMonitor

hm = HardwareMonitor(config)
print(hm.report())

@hm.profile
def heavy_computation():
    ...
```

---

## Reporting

### Dashboard

**Module**: `src.reporting.dashboard`

3-tab Streamlit dashboard for visualising SPY features, options scanner events, and system metrics.

```python
Dashboard(config: Dict[str, Any])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `() -> None` | Launch Streamlit app (called via `streamlit run`) |

Tabs:
- **SPY Features** — plot lagged returns, rolling volatility, and feature distributions from `data/processed/features/`
- **Options Scanner** — display significant-move events from the scanner CSV with filters
- **Hardware & Storage** — live CPU/memory/disk snapshot and data directory size breakdown

**Launch via CLI**:
```bash
python -m src.cli dashboard
# equivalent to: streamlit run src/reporting/dashboard.py
```
