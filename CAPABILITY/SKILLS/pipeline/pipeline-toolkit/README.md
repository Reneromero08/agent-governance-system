# Pipeline Toolkit

Unified toolkit for pipeline DAG operations. Consolidates 3 formerly separate skills.

## Operations

| Operation | Description | Source Skill |
|-----------|-------------|--------------|
| `schedule` | Deterministic DAG scheduling | `pipeline-dag-scheduler` |
| `receipts` | Execution receipts | `pipeline-dag-receipts` |
| `restore` | DAG restoration | `pipeline-dag-restore` |

## Usage

```bash
python run.py input.json output.json
```

### Examples

**Schedule DAG:**
```json
{
  "operation": "schedule",
  "dag_spec_path": "CATALYTIC-DPT/PIPELINES/example.json",
  "runs_root": "CONTRACTS/_runs"
}
```

**Generate receipts:**
```json
{
  "operation": "receipts",
  "dag_spec_path": "CATALYTIC-DPT/PIPELINES/example.json"
}
```

**Restore DAG:**
```json
{
  "operation": "restore",
  "dag_spec_path": "CATALYTIC-DPT/PIPELINES/example.json"
}
```

## Implementation Status

This skill is a governance placeholder. Actual implementation:
- `CATALYTIC-DPT/PIPELINES/` - DAG runtime
- `TOOLS/catalytic.py` - CLI
- `CATALYTIC-DPT/TESTBENCH/` - Tests

## Migration

- `pipeline/pipeline-dag-scheduler` → Use `operation: "schedule"`
- `pipeline/pipeline-dag-receipts` → Use `operation: "receipts"`
- `pipeline/pipeline-dag-restore` → Use `operation: "restore"`
