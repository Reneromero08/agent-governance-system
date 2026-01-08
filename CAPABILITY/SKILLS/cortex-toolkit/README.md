# Cortex Toolkit

Unified toolkit for CORTEX operations. Consolidates 5 formerly separate skills into a single multi-operation skill.

## Operations

| Operation | Description | Source Skill |
|-----------|-------------|--------------|
| `build` | Rebuild CORTEX index and SECTION_INDEX | `cortex-build` |
| `verify_cas` | Check CAS directory integrity | `cas-integrity-check` |
| `verify_system1` | Ensure system1.db is in sync | `system1-verify` |
| `summarize` | Generate section summaries | `cortex-summaries` |
| `smoke_test` | Run LLM Packer smoke tests | `llm-packer-smoke` |

## Usage

### CLI

```bash
python run.py input.json output.json
```

### Input JSON Format

All operations require an `operation` field:

```json
{
  "operation": "build|verify_cas|verify_system1|summarize|smoke_test",
  ...operation-specific fields...
}
```

### Examples

**Build CORTEX index:**
```json
{
  "operation": "build",
  "expected_paths": ["AGENTS.md", "README.md"],
  "timeout_sec": 120
}
```

**Verify CAS integrity:**
```json
{
  "operation": "verify_cas",
  "cas_root": "NAVIGATION/CORTEX/CAS"
}
```

**Verify System1 database:**
```json
{
  "operation": "verify_system1"
}
```

**Generate summary:**
```json
{
  "operation": "summarize",
  "record": {
    "section_id": "AGENTS.md::overview",
    "heading": "## Overview",
    "start_line": 10,
    "end_line": 25,
    "hash": "abc123..."
  },
  "slice_text": "## Overview\n\nThis is the overview section..."
}
```

**LLM Packer smoke test:**
```json
{
  "operation": "smoke_test",
  "scope": "ags",
  "mode": "full",
  "profile": "full"
}
```

## Permissions

- **Read:** All operations can read from any repo path
- **Write:** All operations write via GuardedWriter to approved durable roots:
  - `LAW/CONTRACTS/_runs`
  - `NAVIGATION/CORTEX/_generated`
  - `MEMORY/LLM_PACKER/_packs`
  - `BUILD`

## Migration from Legacy Skills

This toolkit replaces the following individual skills:
- `cortex/cortex-build` → Use `operation: "build"`
- `cortex/cas-integrity-check` → Use `operation: "verify_cas"`
- `cortex/system1-verify` → Use `operation: "verify_system1"`
- `cortex/cortex-summaries` → Use `operation: "summarize"`
- `cortex/llm-packer-smoke` → Use `operation: "smoke_test"`

Legacy skills are deprecated and will be removed in a future version.
