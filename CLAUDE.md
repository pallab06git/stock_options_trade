# CLAUDE.md - Development Guidelines

## Project Objective

**Phase 1**: Historical SPY data ingestion foundation.

- Fetch SPY per-second aggregates from Polygon.io REST API
- Store in local date-partitioned Parquet files
- Config-driven execution via YAML

---

## Strict Scope

### ✅ IN SCOPE
- Historical SPY data (Polygon REST API)
- Local Parquet storage (date-partitioned)
- YAML configuration
- Data validation, logging, error handling
- Resumable fetches

### ❌ OUT OF SCOPE
- No streaming, WebSocket, or real-time
- No Kafka, message queues
- No UI, dashboard, web interface
- No cloud deployment (AWS, GCP, Azure)
- No ML models or feature engineering
- No options/VIX/news data yet

---

## Development Discipline

**One Module at a Time**:
1. Complete one task
2. Test it
3. Document changes
4. **STOP** - wait for next instruction

**Rules**:
- No multi-module generation in one turn
- No full implementations without request
- Build only what's needed **now**
- No hooks for future phases (streaming, AWS, Kafka)

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

---

## Project Structure

```
spy-options-intelligence/
├── src/
│   ├── data_sources/    # Polygon client
│   ├── sinks/           # Parquet storage
│   ├── orchestrator/    # Task runner
│   └── utils/           # Config, logging
├── config/              # YAML
├── data/                # Local storage
└── tests/               # Tests
```

---

## Key Principles

1. Batch-first (not real-time)
2. Config-driven (YAML)
3. Modular
4. Testable
5. Simple

---

## Agent Constraints

- Do not generate roadmaps or future task plans
- Do not introduce new dependencies unless explicitly requested
- Do not modify requirements.txt without instruction
- Do not create additional documentation files
- Do not expand folder structure
- Do not anticipate Phase 2 or beyond

---

## Execution Mode

Every task must:
- Modify only requested files
- Stop immediately after task completion
- Generate Task Completion Report
- Wait for further instruction

---

**Phase 1 = Historical data collection. Nothing more.**