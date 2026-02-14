# SECURITY.md - Security Guidelines

## Critical Security Requirements

### No Credential Exposure

**NEVER log, print, or expose:**
- API keys (Polygon, Massive)
- Passwords
- Database credentials
- Access tokens
- Secret keys
- Bearer tokens
- Connection strings with credentials

---

## Implementation Rules

### 1. Environment Variables Only

```python
# GOOD
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

# BAD - Never hardcode
api_key = "pk_1234567890abcdef"
```

### 2. Logging with Redaction

```python
# GOOD - Redact sensitive data
logger.info(f"Connecting with key: ****{api_key[-4:]}")
logger.info(f"Database: postgres://****@{host}/{db}")

# BAD - Exposes secrets
logger.info(f"API Key: {api_key}")
logger.info(f"DB: postgres://user:pass@host/db")
```

### 3. Error Messages

```python
# GOOD - Sanitize before logging
try:
    client = PolygonClient(api_key=api_key)
except Exception as e:
    # Remove sensitive data from error
    safe_error = str(e).replace(api_key, "****")
    logger.error(f"Connection failed: {safe_error}")

# BAD - May leak credentials in stack trace
except Exception as e:
    logger.error(f"Error: {e}")  # Might contain api_key
```

### 4. Console Output

```python
# GOOD
print(f"Loaded config with {len(api_key)} character key")

# BAD
print(f"Using API key: {api_key}")
```

### 5. Configuration Files

```yaml
# config/sources.yaml - GOOD (references env var)
polygon:
  api_key: ${POLYGON_API_KEY}  # Loaded from .env
  base_url: https://api.polygon.io

# BAD - Never put actual keys in YAML
polygon:
  api_key: pk_1234567890abcdef
```

---

## Automatic Redaction Helper

Implement in `src/utils/logger.py`:

```python
import re

SENSITIVE_PATTERNS = [
    r'(api[_-]?key["\s:=]+)([a-zA-Z0-9_-]+)',
    r'(password["\s:=]+)([^\s"]+)',
    r'(token["\s:=]+)([a-zA-Z0-9_-]+)',
    r'(secret["\s:=]+)([a-zA-Z0-9_-]+)',
    r'(postgres://[^:]+:)([^@]+)(@)',  # DB passwords
]

def redact_sensitive(message: str) -> str:
    """Automatically redact sensitive information from log messages."""
    for pattern in SENSITIVE_PATTERNS:
        message = re.sub(
            pattern,
            lambda m: f"{m.group(1)}****{m.group(2)[-4:] if len(m.group(2)) > 4 else '****'}",
            message,
            flags=re.IGNORECASE
        )
    return message

# Usage in logger
logger.add(
    "logs/execution/{time}.log",
    filter=lambda record: redact_sensitive(record["message"])
)
```

---

## Testing Security

Every module must pass security test:

```python
# tests/unit/test_security.py
def test_no_api_key_in_logs(tmp_path):
    """Ensure API keys are never logged."""
    fake_key = "pk_test_1234567890abcdef"
    
    # Simulate logging
    logger.info(f"Connecting with key: {fake_key}")
    
    # Read log file
    with open(log_file) as f:
        log_content = f.read()
    
    # Assert full key is NOT in logs
    assert fake_key not in log_content
    
    # Assert redacted version IS in logs
    assert "****cdef" in log_content or "****" in log_content
```

---

## Code Review Checklist

Before committing any code, verify:

- [ ] No API keys in source code
- [ ] No passwords in source code
- [ ] All credentials loaded from `.env`
- [ ] `.env` is in `.gitignore`
- [ ] All logging statements redact sensitive data
- [ ] Error messages sanitized
- [ ] No `print()` statements with credentials
- [ ] Connection strings masked in logs
- [ ] Security tests pass

---

## .env File Template

```bash
# .env - NEVER COMMIT THIS FILE

# Polygon.io
POLYGON_API_KEY=pk_your_actual_key_here

# Massive.com
MASSIVE_API_KEY=your_massive_key_here

# Database (if used)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=spy_options
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Optional
LOG_LEVEL=INFO
```

---

## Git Protection

Ensure `.gitignore` includes:

```
# Environment files
.env
.env.local
.env.*.local

# Credential files
*.key
*.pem
credentials.json
secrets.yaml

# Logs (may contain leaked data)
*.log
logs/
```

---

## Incident Response

If credentials are accidentally committed:

1. **Immediately revoke** the exposed credentials
2. Generate new API keys
3. Remove from git history: `git filter-branch` or BFG Repo-Cleaner
4. Force push: `git push --force`
5. Notify team members to pull latest
6. Update `.env` with new credentials
7. Review all logs for potential exposure

---

**Security is non-negotiable. When in doubt, redact.**
