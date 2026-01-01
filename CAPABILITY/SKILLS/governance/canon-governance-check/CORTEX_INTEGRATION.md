<!-- CONTENT_HASH: f17b6808aebd2da4dbf1585ff1f7a48505ab8139ca1de0bce230787dd10ace50 -->

# Canon Governance Check - Cortex Integration

## Overview
The canon governance check integrates with Cortex to provide provenance tracking for governance validation events.

## Cortex Event Schema
When `CORTEX_RUN_ID` is set, governance check results are logged to `CONTRACTS/_runs/<run_id>/events.jsonl`:

```json
{
  "type": "governance_check",
  "timestamp": "<caller-supplied-sentinel>",
  "passed": true|false,
  "exit_code": 0|1
}
```

## Usage with Cortex Provenance

### Option 1: Environment Variables
```bash
export CORTEX_RUN_ID="governance-$(date +%Y%m%d-%H%M%S)"
export CORTEX_TIMESTAMP="SENTINEL"
python SKILLS/canon-governance-check/run.py
```

### Option 2: Direct Integration
```python
import os
from pathlib import Path

# Set provenance context
os.environ["CORTEX_RUN_ID"] = "my-governance-run"
os.environ["CORTEX_TIMESTAMP"] = "SENTINEL"

# Run governance check (logs to Cortex automatically)
subprocess.run(["python", "SKILLS/canon-governance-check/run.py"])
```

## Ledger Output Location
```
CONTRACTS/_runs/<run_id>/
  events.jsonl  # Governance check events appended here
```

## Event Queries

### Check if governance passed in a run
```bash
cat CONTRACTS/_runs/<run_id>/events.jsonl | \
  jq 'select(.type == "governance_check") | .passed'
```

### Count governance failures
```bash
cat CONTRACTS/_runs/*/events.jsonl | \
  jq 'select(.type == "governance_check" and .passed == false)' | \
  wc -l
```

## Integration with Cortex Tools

### Using cortex.py
```bash
# Search for governance events
python TOOLS/cortex.py search "governance_check" --type event

# Resolve governance history
python TOOLS/cortex.py resolve governance-history
```

## Determinism
- Events are appended with canonical JSON (sorted keys, no whitespace)
- Timestamps are caller-supplied deterministic sentinels
- No runtime-generated timestamps or UUIDs
- Same inputs → same event records

## CI Integration
In `.github/workflows/contracts.yml`, governance check runs before Cortex build:
1. Node.js setup
2. **Canon governance check** (this validates changes)
3. Cortex index build
4. Other governance checks

This ensures governance validation happens before any knowledge indexing.
