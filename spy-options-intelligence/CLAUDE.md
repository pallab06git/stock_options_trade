# CLAUDE.md - Development Guidelines

## Project Objective

**Phase 1**: Multi-source data ingestion platform with dual-mode execution (historical + real-time).

**Core Functions**:
- Fetch SPY per-second aggregates from Polygon.io (REST + WebSocket)
- Track options within ±1% of SPY opening price
- Ingest VIX data via Massive API
- Collect news events via Polygon News API
- Support both historical backfill and real-time streaming
- Store in configurable sinks (Parquet, SQL, cloud databases)
- Production-grade error handling and monitoring

---

## Strict Scope

### ✅ IN SCOPE (Phase 1)

**Data Sources**:
- SPY per-second aggregates (historical + real-time)
- Options contracts (±1% strike range from opening price)
- VIX volatility index (per-second)
- News events with sentiment analysis

**Execution Modes**:
- Historical data pull (REST API)
- Real-time streaming (WebSocket)

**Architecture**:
- Configurable source and sink (local files, databases, cloud)
- Dual-mode orchestration (batch vs streaming)
- Parquet for individual feeds + SQL for consolidated view
- Generator-based processing for large datasets

**Resilience**:
- Configurable retry logic (default: 3 attempts)
- Duplicate detection and overwrite
- Schema drift handling with daily alerts
- Late data handling (default: reject older than current day)
- Performance monitoring (alert if commits > 5 min)
- Connection failure recovery
- Backlog/backpressure management

**Operational**:
- Comprehensive logging for each step
- Meaningful error messages for aggregated analysis
- Manual stop/start capability
- History correction with overwrite
- Real-time feed simulator from historical data

### ❌ OUT OF SCOPE (Phase 1)
- Kafka message queues
- Twilio alerting
- Web UI/dashboard
- CI/CD pipelines
- AWS deployment automation
- ML models or feature engineering
- Signal generation or trading logic

---

## Data Sources Detail

### 1. SPY Aggregates
- **REST**: `polygon.RESTClient.get_aggs()` for historical
- **WebSocket**: `polygon.WebSocketClient` with `Feed.Delayed` or `Feed.RealTime`
- **Frequency**: Per-second bars
- **Fields**: open, close, high, low, volume, vwap, timestamps

### 2. Options Contracts
- **Discovery**: `polygon.list_options_contracts()` at market open
- **Tracking**: Only contracts within ±1% of SPY opening price
- **Streaming**: `WebSocketClient` with `Market.Options`
- **Limit**: ≤100 contracts per connection (Polygon recommendation)

### 3. VIX Data
- **Source**: Massive API (similar interface to Polygon)
- **Symbol**: `I:VIX`
- **Frequency**: Per-second aggregates
- **Fields**: open, close, high, low, timestamps

### 4. News Events
- **API**: `polygon.list_ticker_news()`
- **Fields**: title, description, sentiment, published_utc, keywords, tickers
- **Filtering**: Configurable by ticker relevance

---

## Project Structure

```
spy-options-intelligence/
├── src/
│   ├── data_sources/
│   │   ├── polygon_client.py      # SPY + Options REST/WebSocket
│   │   ├── massive_client.py      # VIX REST/WebSocket
│   │   ├── news_client.py         # Polygon News API
│   │   └── base_source.py         # Abstract source interface
│   ├── sinks/
│   │   ├── parquet_sink.py        # Date-partitioned Parquet
│   │   ├── sql_sink.py            # SQL database (consolidated)
│   │   ├── cloud_sink.py          # S3/GCS/Azure (future)
│   │   └── base_sink.py           # Abstract sink interface
│   ├── orchestrator/
│   │   ├── historical_runner.py   # Batch mode orchestration
│   │   ├── streaming_runner.py    # Real-time mode orchestration
│   │   ├── simulator.py           # Historical → real-time simulator
│   │   └── task_manager.py        # Task tracking
│   ├── processing/
│   │   ├── deduplicator.py        # Duplicate detection
│   │   ├── validator.py           # Schema validation
│   │   ├── consolidator.py        # Multi-source join logic
│   │   └── late_data_handler.py   # Late arrival handling
│   ├── monitoring/
│   │   ├── performance_monitor.py # 5-min commit SLA
│   │   ├── schema_monitor.py      # Drift detection
│   │   └── error_aggregator.py    # Error pattern analysis
│   └── utils/
│       ├── config_loader.py       # YAML config parser
│       ├── logger.py              # Centralized logging
│       ├── retry_handler.py       # Configurable retry logic
│       └── connection_manager.py  # Connection pooling
├── config/
│   ├── settings.yaml              # Master configuration
│   ├── sources.yaml               # API keys, endpoints
│   ├── sinks.yaml                 # Storage destinations
│   └── retry_policy.yaml          # Failure handling rules
├── data/
│   ├── raw/
│   │   ├── spy/                   # SPY aggregates
│   │   ├── options/               # Options data
│   │   ├── vix/                   # VIX data
│   │   └── news/                  # News events
│   ├── processed/
│   │   ├── consolidated/          # Multi-source joined data
│   │   └── partitioned/           # Date-partitioned by source
│   └── logs/
│       ├── execution/             # Run logs
│       ├── errors/                # Error logs
│       └── performance/           # Performance metrics
└── tests/
    ├── unit/                      # Unit tests
    ├── integration/               # End-to-end tests
    └── mocks/                     # Mock data/responses
```

---

## Development Discipline

**Architecture-First Approach**:
1. **BEFORE any implementation**, propose technical architecture:
   - System architecture diagram
   - Technology stack for each layer
   - Tools/libraries for each functional module
   - Data flow and component interactions
   - Performance and scalability considerations
2. **Wait for approval** of architecture
3. **Only after approval**, proceed with implementation

**One Module at a Time**:
1. Complete one task
2. Test it
3. Document changes
4. **STOP** - wait for next instruction

**Rules**:
- No multi-module generation in one turn unless explicitly requested
- Build only what's requested now
- Comment code thoroughly for functionality understanding
- Architecture must be approved before writing code

---

## Task Completion Report Format

```markdown
## Task Completion Report

**Task**: [Brief description]

**Files Modified**:
- path/to/file.py (created/modified)

**Summary**: [2-3 sentences]

**Open Questions**: [Any ambiguities]
```

---

## Working Directory

All work: `/mnt/user-data/outputs/spy-options-intelligence/`

No copying files, archives, or moving work outside project.

**Required Reading:**
- `CLAUDE.md` - Development guidelines (this file)
- `SCOPE.md` - API specifications and requirements
- `SECURITY.md` - Security requirements and credential handling

---

## Key Principles

1. Dual-mode execution (historical + real-time)
2. Config-driven (YAML)
3. Modular and extensible
4. Production-grade error handling
5. Performance monitored
6. Resource-efficient streaming

---

## Real-time Streaming Optimization

**CRITICAL - Efficient Streaming Design:**

1. **No Per-Message Logging:**
   - Do NOT log every WebSocket message received
   - Do NOT write individual records to logs
   - Logging every tick is overkill and creates massive log files

2. **Heartbeat Monitoring (Every 5 Minutes):**
   - Update metadata file with timestamp
   - Track: messages received count, last message time, connection status
   - Example: `data/logs/heartbeat/stream_status.json`
   ```json
   {
     "last_heartbeat": "2025-02-12T14:35:00Z",
     "messages_received_last_5min": 1500,
     "connection_status": "active",
     "source": "polygon_spy"
   }
   ```

3. **Alert Threshold (15 Minutes):**
   - If no heartbeat update for >15 minutes, trigger alert
   - Alert indicates: stream stalled, connection lost, or performance issue
   - Log alert to errors/ directory

4. **Market Hours Awareness:**
   - **Active hours**: 9:30 AM - 4:00 PM ET, Monday-Friday ONLY
   - **Pre-market**: DISABLED (not in Phase 1 scope)
   - **After-hours**: DISABLED (not in Phase 1 scope)
   - **Holidays**: Fetched from Polygon API market calendar
   - Check market status before connecting WebSocket
   - Auto-shutdown at market close (4:00 PM ET)

5. **Resource Conservation:**
   - No streaming during weekends
   - No streaming during market holidays (NYSE/NASDAQ)
   - No streaming outside 9:30 AM - 4:00 PM ET
   - Prevents unnecessary compute/network usage

6. **Implementation Requirements:**
   ```python
   # Example structure
   class StreamingRunner:
       def is_market_open(self) -> bool:
           # Check: Mon-Fri, 9:30 AM - 4:00 PM ET
           # Return False outside these hours
           
       def start_stream(self):
           if not self.is_market_open():
               logger.info("Market closed, stream inactive")
               return
           # Start WebSocket connection
           
       def update_heartbeat(self):
           # Called every 5 minutes
           # Update metadata file
           
       def check_heartbeat_alert(self):
           # If last update >15 min ago, alert
   ```

7. **Configuration (settings.yaml):**
   ```yaml
   streaming:
     market_hours:
       timezone: "America/New_York"
       active_days: [0, 1, 2, 3, 4]  # Mon-Fri (0=Mon, 6=Sun)
       start_time: "09:30"
       end_time: "16:00"
       # Pre-market and after-hours DISABLED for Phase 1
     monitoring:
       heartbeat_interval_seconds: 300  # 5 minutes
       alert_threshold_seconds: 900     # 15 minutes
   ```

**Benefits:**
- Drastically reduced log file sizes
- Easy monitoring (check one heartbeat file)
- Automatic resource management
- Clear alert when stream fails
- Cost savings (no compute during off-hours)

---

## Agent Constraints

- Do not generate roadmaps or future task plans
- Do not introduce new dependencies unless explicitly requested
- Do not modify requirements.txt without instruction
- Do not create additional documentation files
- Do not expand folder structure beyond defined structure
- Do not anticipate Phase 2 or beyond

---

## Git Commit Policy

**Author Attribution:**
- ALL commits must be attributed to: Pallab Basu Roy
- NEVER add "Co-Authored-By: Claude" or similar AI attributions
- Git uses configured user name automatically
- Do not add AI attribution in commit messages

**Commit Message Format:**
```
Brief summary (50 chars or less)

Detailed description of changes:
- What was added/modified
- Why the change was made
- Any breaking changes or important notes
```

**Example - CORRECT:**
```
Add configuration system with YAML validation

Implemented config_loader.py with pydantic validation,
created settings.yaml and sources.yaml templates.
Supports environment variable overrides.
```

**Example - INCORRECT:**
```
Add configuration system

Co-Authored-By: Claude <noreply@anthropic.com>  ❌ NEVER DO THIS
```

**Policy Scope:**
- All commits in this project
- All future commits
- All branches
- All AI assistants working on this codebase

---

## Security Requirements

**CRITICAL - No Security Violations:**

1. **Never log or print sensitive information:**
   - API keys
   - Passwords
   - Database credentials
   - Access tokens
   - Secret keys
   - Usernames (when used with passwords)

2. **Redaction in logs:**
   - Replace sensitive values with `***REDACTED***`
   - Show only last 4 characters of keys (e.g., `****abc123`)
   - Never log full connection strings with credentials

3. **Environment variables only:**
   - All secrets must be in `.env` file (git-ignored)
   - Never hardcode credentials in code
   - Load via `os.getenv()` or `python-dotenv`

4. **Error messages:**
   - Sanitize error messages before logging
   - Do not expose credentials in stack traces
   - Mask sensitive data in exception handlers

5. **Console output:**
   - Never print API keys or passwords to stdout/stderr
   - Use `[REDACTED]` in debug prints
   - Verify all print/logging statements before commit

**Example - BAD:**
```python
logger.info(f"Connecting with API key: {api_key}")
print(f"Database: postgres://user:password@host/db")
```

**Example - GOOD:**
```python
logger.info(f"Connecting with API key: ****{api_key[-4:]}")
logger.info(f"Database: postgres://****@host/db")
```

---

## Execution Mode

Every task must:
- Modify only requested files
- Stop immediately after task completion
- Generate Task Completion Report
- Wait for further instruction

---

**Phase 1 = Multi-source data ingestion foundation. Historical + Real-time.**
