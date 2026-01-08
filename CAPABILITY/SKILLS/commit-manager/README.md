# Commit Manager

Unified manager for commit-related operations. Consolidates 3 formerly separate skills.

## Operations

| Operation | Description | Source Skill |
|-----------|-------------|--------------|
| `queue` | Manage commit queue | `commit-queue` |
| `summarize` | Commit summaries/templates | `commit-summary-log` |
| `recover` | Artifact escape check | `artifact-escape-hatch` |

## Usage

```bash
python run.py input.json output.json
```

### Examples

**Enqueue a commit:**
```json
{
  "operation": "queue",
  "action": "enqueue",
  "entry": {
    "message": "feat(skills): add new feature",
    "files": ["CAPABILITY/SKILLS/new-skill/run.py"],
    "created_at": "2026-01-07T12:00:00Z"
  }
}
```

**Generate commit template:**
```json
{
  "operation": "summarize",
  "action": "template",
  "type": "feat",
  "scope": "skills",
  "subject": "add new feature"
}
```

**Check for escaped artifacts:**
```json
{
  "operation": "recover",
  "check_type": "artifact-escape-hatch"
}
```

## Migration

- `commit/commit-queue` → Use `operation: "queue"`
- `commit/commit-summary-log` → Use `operation: "summarize"`
- `commit/artifact-escape-hatch` → Use `operation: "recover"`
