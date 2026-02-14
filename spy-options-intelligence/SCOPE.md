# SCOPE.md - Phase 1 Detailed Specifications

## Data Sources & API Reference

### 1. SPY Per-Second Aggregates

#### Historical (REST API)
```python
from polygon import RESTClient

client = RESTClient("YOUR_POLYGON_API_KEY")
aggs = client.get_aggs(
    ticker="SPY",
    multiplier=1,
    timespan="second",
    from_="2025-10-28",
    to="2025-10-28",
    limit=120
)
```

#### Real-time (WebSocket)
```python
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from typing import List

client = WebSocketClient(
    api_key="YOUR_POLYGON_API_KEY",
    feed=Feed.Delayed,      # or Feed.RealTime for paid plan
    market=Market.Stocks
)

client.subscribe("A.SPY")

def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        print(m)

client.run(handle_msg)
```

#### Response Schema
```json
{
  "results": [
    {
      "ev": "A",
      "sym": "SPY",
      "v": 200,
      "av": 8642007,
      "op": 425.66,
      "vw": 425.3981,
      "o": 425.39,
      "c": 425.39,
      "h": 425.39,
      "l": 425.39,
      "a": 425.3714,
      "z": 50,
      "s": 1730144868000,
      "e": 1730144869000
    }
  ],
  "status": "OK"
}
```

**Field Descriptions**:
- `ev`: Event type (A = aggregate bar)
- `sym`: Ticker symbol
- `v`: Tick volume (per second)
- `av`: Cumulative daily volume
- `op`: Day's official open
- `vw`: Volume-weighted average price
- `o`, `c`, `h`, `l`: Open, Close, High, Low for the second
- `a`: Intraday VWAP
- `z`: Average trade size for the window
- `s`, `e`: Start/end timestamps (Unix ms)

---

### 2. Options Contracts

#### Contract Discovery (REST API)
```python
from polygon import RESTClient
from datetime import datetime
import json

client = RESTClient("YOUR_POLYGON_API_KEY")

# Step 1: Fetch SPY open price at market open
open_price = 428.50  # Replace with live feed
lower = round(open_price * 0.99, 2)
upper = round(open_price * 1.01, 2)

# Step 2: Pull next-day expiry contracts within ±1%
contracts = client.list_options_contracts(
    underlying_ticker="SPY",
    expiration_date="2025-10-30",
    strike_price_gte=lower,
    strike_price_lte=upper,
    order="asc",
    limit=100,
    sort="ticker"
)

# Step 3: Store contracts for the day
contracts_list = [c for c in contracts]
path = f"./data/options/contracts/{datetime.now():%Y-%m-%d}_contracts.json"
with open(path, "w") as f:
    json.dump([c.__dict__ for c in contracts_list], f, indent=2)

print(f"✅ {len(contracts_list)} contracts saved → {path}")
```

#### Real-time Options Data (WebSocket)
```python
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from typing import List
import json, datetime, os

API_KEY = "YOUR_POLYGON_API_KEY"

# Load contracts discovered earlier today
today = datetime.datetime.now().strftime("%Y-%m-%d")
contracts_path = f"./data/options/contracts/{today}_contracts.json"

if not os.path.exists(contracts_path):
    raise FileNotFoundError(
        f"Contract file for {today} not found. Run contract discovery first."
    )

with open(contracts_path, "r") as f:
    contracts = json.load(f)

# Limit to manageable subset (Polygon recommends ≤100 per connection)
tickers = [c["ticker"] for c in contracts][:100]

client = WebSocketClient(
    api_key=API_KEY,
    feed=Feed.Delayed,     # use Feed.RealTime for paid plan
    market=Market.Options
)

# Subscribe dynamically to all discovered tickers
client.subscribe(*[f"AM.{t}" for t in tickers])

def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        print(m)

client.run(handle_msg)
```

**Key Requirements**:
- Discover contracts at market open based on SPY opening price
- Track only contracts within ±1% strike range
- Limit to ≤100 contracts per WebSocket connection
- Store contract metadata daily

---

### 3. VIX Volatility Index

#### Historical (REST API)
```bash
curl "https://api.massive.com/v3/reference/dividends?apiKey=YOUR_POLYGON_API_KEY"
```

#### Real-time (WebSocket)
```python
from massive import WebSocketClient
from massive.websocket.models import WebSocketMessage, Feed, Market
from typing import List

client = WebSocketClient(
    api_key="YOUR_MASSIVE_API_KEY",
    feed=Feed.Delayed,
    market=Market.Indices
)

# Subscribe to VIX aggregates (per second)
client.subscribe("A.I:VIX")  # Volatility Index

# Alternative subscriptions:
# client.subscribe("A.*")         # all aggregates
# client.subscribe("A.I:SPX")     # S&P 500
# client.subscribe("A.I:DJI")     # Dow Jones
# client.subscribe("A.I:NDX")     # Nasdaq-100

def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        print(m)

client.run(handle_msg)
```

#### Response Schema
```json
{
  "results": [
    {
      "cash_amount": 0.25,
      "currency": "USD",
      "declaration_date": "2024-10-31",
      "dividend_type": "CD",
      "ex_dividend_date": "2024-11-08",
      "frequency": 4,
      "id": "E416a068758f85277196150c3eb73a3331d04698856c141e883ad95710dd0b189",
      "pay_date": "2024-11-14",
      "record_date": "2024-11-11",
      "ticker": "AAPL"
    }
  ],
  "status": "OK",
  "request_id": "5a8e1e551dc3a1c2c203744543b40399"
}
```

**Aggregate Event Fields**:
- `ev`: Event type (AM)
- `sym`: Index symbol (I:VIX)
- `op`: Official opening value
- `o`: Window open
- `c`: Window close
- `h`: Window high
- `l`: Window low
- `s`: Start timestamp (ms)
- `e`: End timestamp (ms)

---

### 4. News Events

#### API Call
```python
from polygon import RESTClient
from polygon.rest.models import TickerNews

client = RESTClient("YOUR_POLYGON_API_KEY")

news_items = []
for n in client.list_ticker_news(
    order="asc",
    limit=10,
    sort="published_utc"
):
    news_items.append(n)

# Print timestamp and title
for idx, item in enumerate(news_items):
    if isinstance(item, TickerNews):
        print(f"{item.published_utc:<25}{item.title}")
        if idx == 20:
            break
```

#### Response Schema
```json
{
  "count": 1,
  "next_url": "https://api.polygon.io/v2/reference/news?cursor=...",
  "request_id": "831afdb0b8078549fed053476984947a",
  "results": [
    {
      "article_url": "https://uk.investing.com/news/stock-market-news/...",
      "author": "Sam Boughedda",
      "description": "UBS analysts warn that markets are underestimating...",
      "id": "8ec63877...",
      "image_url": "https://i-invdn-com.investing.com/news/LYNXNPEC4I0AL_L.jpg",
      "insights": [
        {
          "sentiment": "positive",
          "sentiment_reasoning": "UBS analysts are providing a bullish outlook...",
          "ticker": "UBS"
        }
      ],
      "keywords": ["Federal Reserve", "interest rates", "economic data"],
      "published_utc": "2024-06-24T18:33:53Z",
      "publisher": {
        "name": "Investing.com",
        "logo_url": "https://s3.polygon.io/public/assets/news/logos/investing.png"
      },
      "tickers": ["UBS"],
      "title": "Markets are underestimating Fed cuts: UBS"
    }
  ],
  "status": "OK"
}
```

**Field Descriptions**:
- `count`: Number of articles returned
- `next_url`: Pagination URL
- `article_url`: Link to full article
- `author`: Article author
- `description`: Summary/teaser
- `insights.sentiment`: positive/neutral/negative
- `insights.sentiment_reasoning`: Rationale for classification
- `keywords`: Extracted keywords for NLP
- `published_utc`: UTC timestamp (RFC 3339)
- `tickers`: Mentioned ticker symbols

---

## Failure Handling Requirements

### Retry Policy
- **Default**: 3 retry attempts
- **Configurable**: Via `config/retry_policy.yaml`
- **Backoff**: Exponential with jitter

### Required Failure Scenarios

1. **Connection Failure**
   - Network timeout
   - API unavailable
   - Authentication failure

2. **Schema Drift**
   - New fields added
   - Fields removed
   - Type changes
   - **Action**: Log once per day, continue processing

3. **Late Data Flow**
   - **Default**: Reject data older than current day (real-time mode)
   - **Configurable**: Grace period in settings
   - **Action**: Log and skip or write to separate partition

4. **Slowness**
   - **Threshold**: Alert if commits take >5 minutes (configurable)
   - **Metrics**: Track latency percentiles (p50, p95, p99)

5. **Backlog/Backpressure**
   - **Batching**: Configurable batch size for bulk processing
   - **Throttling**: Rate limiting for downstream sinks

6. **Incomplete Data**
   - Missing required fields
   - Null values in non-nullable columns
   - **Action**: Quarantine to error partition

### Performance Monitoring

- **Commit SLA**: 5 minutes (configurable)
- **Alert Triggers**:
  - Commit latency >5 min
  - Error rate >1% over 15-min window
  - Schema drift detected
  - Connection failures >3 consecutive attempts

---

## Storage Requirements

### Individual Feed Storage
Each data source stored separately:
```
data/raw/
├── spy/
│   └── date=YYYY-MM-DD/*.parquet
├── options/
│   └── date=YYYY-MM-DD/*.parquet
├── vix/
│   └── date=YYYY-MM-DD/*.parquet
└── news/
    └── date=YYYY-MM-DD/*.parquet
```

### Consolidated Storage (SQL)
Multi-source joined view at options level:
- SPY ticker data (for each second)
- VIX value (for each second)
- News events (time-aligned)
- Options data (primary)

**Configurable sinks**:
- Local Parquet
- PostgreSQL
- MySQL
- Snowflake
- S3/GCS/Azure (future)

---

## Operational Requirements

### Logging
- **Per-step completion logs**: "Fetched 10,000 SPY bars"
- **Error logs**: With context for aggregated analysis
- **Performance logs**: Latency, throughput metrics

**Real-time Streaming Logs:**
- **NO per-message logging** (too verbose, creates massive files)
- **Heartbeat every 5 minutes**: Update metadata file with:
  * Timestamp
  * Messages received in last 5 min
  * Connection status
  * Source identifier
- **Alert if no heartbeat for 15 minutes**: Indicates stream failure
- **Example heartbeat metadata**:
  ```json
  {
    "last_heartbeat": "2025-02-12T14:35:00Z",
    "messages_received_last_5min": 1500,
    "total_messages_today": 234567,
    "connection_status": "active",
    "source": "polygon_spy",
    "memory_usage_mb": 245
  }
  ```

### Market Hours Management
**Streaming Active Hours:**
- **Days**: Monday-Friday only (no weekends)
- **Hours**: 9:30 AM - 4:00 PM ET (regular market hours ONLY)
- **Pre-market**: DISABLED (not included in Phase 1)
- **After-hours**: DISABLED (not included in Phase 1)
- **Holidays**: Fetched from Polygon API market calendar endpoint
  - API: `GET /v1/marketstatus/upcoming`
  - Cached for current year
  - Fallback to empty list if API fails (log warning)

**Resource Conservation:**
- Auto-shutdown at market close (4:00 PM ET)
- Do not start stream if market is closed
- Check market status before WebSocket connection
- Prevents unnecessary compute/network usage during off-hours
- No streaming during holidays (NYSE/NASDAQ closures)

**Implementation:**
```python
# Check before starting stream
market_hours = MarketHours(config, polygon_api_key)
if not market_hours.is_market_open():
    logger.info("Market closed - streaming inactive")
    return

# During active stream
if market_hours.seconds_until_market_close() < 60:
    logger.info("Market close approaching - graceful shutdown")
    disconnect_websocket()
```

### Manual Controls
- **Stop**: Graceful shutdown with checkpoint
- **Start**: Resume from last checkpoint (only if market open)
- **Overwrite**: History correction mode

### Simulators
- Convert historical data dumps to real-time feed
- Configurable playback speed
- Support all data sources (SPY, options, VIX, news)
- Respect simulated market hours

---

## Configuration-Driven Execution

All parameters via YAML:
- API keys and endpoints
- Date ranges (historical mode)
- Ticker symbols
- Retry policies
- Performance thresholds
- Storage destinations
- Logging verbosity
- Feature flags (enable/disable sources)

---

**This document defines the complete Phase 1 scope.**
