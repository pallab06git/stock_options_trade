# SPY Options Intelligence Platform

> Historical data ingestion foundation for ML research

---

## Overview

Foundation for collecting and storing SPY historical data to support ML-based trading signal research. This project fetches per-second aggregate data from Polygon API and stores it in date-partitioned Parquet files.

---

## Features

- **Historical Data Ingestion**: Fetch SPY per-second aggregates from Polygon API
- **Parquet Storage**: Date-partitioned columnar storage with compression
- **Config-Driven**: YAML-based configuration for all parameters
- **Modular Design**: Separate ingestion, storage, and orchestration layers
- **Resumable**: Checkpoint-based execution for interrupted fetches

---

## Installation

### Prerequisites
- Python 3.10+
- Polygon.io API key ([free tier available](https://polygon.io))
- Claude Code CLI or Desktop App

### Setup
```bash
# Extract project
tar -xzf spy-options-intelligence.tar.gz
cd spy-options-intelligence/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY
```

### Development with Claude Code

**IMPORTANT: Architecture-First Approach**

1. Open project in Claude Code
2. Use initial prompt from `INITIAL_PROMPT.md`
3. Claude Code will generate architecture proposal (Step 0)
4. Review and approve architecture
5. Claude Code implements Steps 1-20 incrementally
6. Review each step before proceeding

See `INITIAL_PROMPT.md` for complete workflow.

---

## Usage

### Configuration
Edit `config/settings.yaml` to set:
- Date range for data collection
- Storage paths
- Logging preferences

### Run Data Ingestion
```bash
# Fetch data as configured
python -m src.orchestration.runner --config config/settings.yaml

# Override date range
python -m src.orchestration.runner --start-date 2024-06-01 --end-date 2024-06-30

# Resume interrupted fetch
python -m src.orchestration.runner --resume
```

---

## Data Schema

SPY per-second aggregate bars:
```python
{
    'timestamp': int64,      # Unix timestamp (ms)
    'open': float64,         # Opening price
    'high': float64,         # High price
    'low': float64,          # Low price
    'close': float64,        # Closing price
    'volume': int64,         # Trading volume
    'vwap': float64,         # Volume-weighted average price
    'ticker': string,        # "SPY"
    'date': string          # YYYY-MM-DD (partition key)
}
```

Storage: `data/processed/spy/date=YYYY-MM-DD/*.parquet`

---

## Project Structure

```
spy-options-intelligence/
├── config/              # YAML configuration files
├── src/
│   ├── ingestion/      # Polygon API client, data fetcher
│   ├── storage/        # Parquet writer, schemas
│   ├── orchestration/  # Master runner, task manager
│   └── utils/          # Config loader, logging
├── data/
│   ├── raw/            # Raw API cache
│   ├── processed/      # Parquet files (date-partitioned)
│   └── logs/           # Execution logs
└── tests/              # Unit tests
```

---

## Security

**Critical - Credential Protection:**
- All API keys must be in `.env` file (git-ignored, never committed)
- Credentials automatically redacted from logs (shows only last 4 chars)
- Never hardcode secrets in source code
- See `SECURITY.md` for complete guidelines

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

---

## License

[Specify your license]
