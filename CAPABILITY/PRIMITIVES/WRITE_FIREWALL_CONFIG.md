# Write Firewall Configuration Guide

## Overview

The Runtime Write Firewall (Phase 1.5A) enforces **catalytic domain separation** at the IO layer. It provides mechanical, fail-closed enforcement of write policies to ensure:

1. **Tmp writes** only go to declared tmp roots during execution
2. **Durable writes** only go to declared durable roots AND only after the commit gate opens
3. **All other filesystem mutations** (rename, unlink, mkdir) respect domain boundaries
4. **Deterministic errors** with machine-readable receipts on any violation

## Quick Start

```python
from pathlib import Path
from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall

# Initialize firewall
firewall = WriteFirewall(
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp", "CAPABILITY/PRIMITIVES/_scratch"],
    durable_roots=["LAW/CONTRACTS/_runs"],
    project_root=Path.cwd(),
    exclusions=["LAW/CANON", ".git"]
)

# During execution: tmp writes allowed
firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/progress.json", '{"status": "running"}', kind="tmp")

# After execution: open commit gate for durable writes
firewall.open_commit_gate()
firewall.safe_write("LAW/CONTRACTS/_runs/result.json", '{"status": "complete"}', kind="durable")
```

## Configuration Parameters

### `tmp_roots` (List[str])

Temporary write domains. Files written here are ephemeral and should not persist beyond the execution boundary.

**Standard catalytic tmp roots:**
- `LAW/CONTRACTS/_runs/_tmp` - Temporary run artifacts
- `CAPABILITY/PRIMITIVES/_scratch` - Scratch space for primitives
- `NAVIGATION/CORTEX/_generated/_tmp` - Temporary generated files

**Rules:**
- Tmp writes (`kind="tmp"`) are ONLY allowed under these roots
- No commit gate required
- Available during entire execution

### `durable_roots` (List[str])

Durable write domains. Files written here persist and become part of the repository state.

**Standard catalytic durable roots:**
- `LAW/CONTRACTS/_runs` - Durable run results and receipts
- `NAVIGATION/CORTEX/_generated` - Generated index files

**Rules:**
- Durable writes (`kind="durable"`) are ONLY allowed under these roots
- **Commit gate MUST be open** before durable writes are allowed
- Represents the commit boundary

### `project_root` (Path)

Absolute path to the project root directory. All relative paths are resolved against this root.

**Rules:**
- MUST be an absolute path
- All write paths are normalized relative to this root
- Paths outside project_root are rejected

### `exclusions` (Optional[List[str]])

Paths that are NEVER writable, even if they fall under allowed roots.

**Standard exclusions:**
- `LAW/CANON` - Canon documents (read-only law)
- `AGENTS.md` - System agent definitions
- `BUILD` - Build artifacts
- `.git` - Git metadata

**Rules:**
- Exclusions take precedence over all other rules
- Any write to an excluded path fails immediately
- Use for immutable/protected paths

## Core Operations

### `safe_write(path, data, kind)`

Write a file with firewall enforcement.

**Parameters:**
- `path` (str | Path): Path to write (relative or absolute)
- `data` (str | bytes): Content to write
- `kind` (str): `"tmp"` or `"durable"`

**Raises:**
- `FirewallViolation` if write violates policy

**Example:**
```python
# Tmp write (allowed during execution)
firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/log.txt", "Starting...", kind="tmp")

# Durable write (requires commit gate)
firewall.open_commit_gate()
firewall.safe_write("LAW/CONTRACTS/_runs/output.json", '{"ok": true}', kind="durable")
```

### `safe_mkdir(path, kind, parents=True, exist_ok=True)`

Create a directory with firewall enforcement.

**Parameters:**
- `path` (str | Path): Directory path
- `kind` (str): `"tmp"` or `"durable"`
- `parents` (bool): Create parent directories
- `exist_ok` (bool): Don't raise if directory exists

**Raises:**
- `FirewallViolation` if mkdir violates policy

**Example:**
```python
# Tmp directory
firewall.safe_mkdir("LAW/CONTRACTS/_runs/_tmp/stage1", kind="tmp")

# Durable directory (requires commit gate)
firewall.open_commit_gate()
firewall.safe_mkdir("LAW/CONTRACTS/_runs/final_output", kind="durable")
```

### `safe_rename(src, dst)`

Rename a file or directory with firewall enforcement.

**Parameters:**
- `src` (str | Path): Source path
- `dst` (str | Path): Destination path

**Rules:**
- Both `src` and `dst` must be in allowed domains
- `dst` domain determines commit gate requirement
- Renaming from tmp→durable requires commit gate open

**Raises:**
- `FirewallViolation` if rename violates policy

**Example:**
```python
# Rename within tmp (no commit gate needed)
firewall.safe_rename(
    "LAW/CONTRACTS/_runs/_tmp/draft.json",
    "LAW/CONTRACTS/_runs/_tmp/final.json"
)

# Rename to durable (requires commit gate)
firewall.open_commit_gate()
firewall.safe_rename(
    "LAW/CONTRACTS/_runs/_tmp/result.json",
    "LAW/CONTRACTS/_runs/result.json"
)
```

### `safe_unlink(path)`

Delete a file with firewall enforcement.

**Parameters:**
- `path` (str | Path): Path to delete

**Rules:**
- Path must be in an allowed domain (tmp or durable)
- Excluded paths cannot be deleted
- No commit gate requirement

**Raises:**
- `FirewallViolation` if unlink violates policy

**Example:**
```python
# Delete tmp file
firewall.safe_unlink("LAW/CONTRACTS/_runs/_tmp/temp.json")

# Delete durable file (no commit gate needed for deletion)
firewall.safe_unlink("LAW/CONTRACTS/_runs/old_result.json")
```

### `open_commit_gate()`

Open the commit gate to allow durable writes.

**Rules:**
- Call this AFTER all execution is complete
- Call this BEFORE any durable writes
- Represents the commit boundary in the execution model

**Example:**
```python
# Stage 1: Execution (tmp writes only)
firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/progress.json", data, kind="tmp")

# Stage 2: Commit (open gate)
firewall.open_commit_gate()

# Stage 3: Persist (durable writes allowed)
firewall.safe_write("LAW/CONTRACTS/_runs/receipt.json", data, kind="durable")
```

### `configure_policy(tmp_roots, durable_roots, exclusions=None)`

Reconfigure the firewall policy at runtime.

**⚠️ Warning:** Reconfiguring **closes the commit gate**. You must call `open_commit_gate()` again.

**Example:**
```python
firewall.configure_policy(
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs"],
    exclusions=["LAW/CANON"]
)
```

## Violation Receipts

When a firewall violation occurs, a `FirewallViolation` exception is raised with a deterministic receipt.

### Receipt Structure

```json
{
  "firewall_version": "1.0.0",
  "tool_version_hash": "abc123...",
  "verdict": "FAIL",
  "error_code": "FIREWALL_PATH_NOT_IN_DOMAIN",
  "message": "Path not in any allowed write domain",
  "operation": "write",
  "path": "README.md",
  "kind": "tmp",
  "policy_snapshot": {
    "tmp_roots": ["LAW/CONTRACTS/_runs/_tmp"],
    "durable_roots": ["LAW/CONTRACTS/_runs"],
    "exclusions": ["LAW/CANON", ".git"],
    "commit_gate_open": false,
    "tool_version": "1.0.0",
    "tool_version_hash": "abc123..."
  }
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| `FIREWALL_PATH_ESCAPE` | Path escapes project root |
| `FIREWALL_PATH_TRAVERSAL` | Path contains `..` traversal |
| `FIREWALL_PATH_EXCLUDED` | Path is in exclusion list |
| `FIREWALL_PATH_NOT_IN_DOMAIN` | Path not in any allowed domain |
| `FIREWALL_TMP_WRITE_WRONG_DOMAIN` | Tmp write attempted outside tmp roots |
| `FIREWALL_DURABLE_WRITE_WRONG_DOMAIN` | Durable write attempted outside durable roots |
| `FIREWALL_DURABLE_WRITE_BEFORE_COMMIT` | Durable write attempted before commit gate opened |
| `FIREWALL_INVALID_KIND` | Invalid write kind (not "tmp" or "durable") |

### Handling Violations

```python
from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation

try:
    firewall.safe_write("README.md", "test", kind="tmp")
except FirewallViolation as e:
    # Access violation receipt
    receipt = e.violation_receipt
    print(f"Error: {e.error_code}")
    print(f"Message: {e.message}")

    # Write receipt to file
    e.write_receipt(Path("violation.json"))

    # Serialize to JSON
    json_str = e.to_json()
```

## Integration Patterns

### Pattern 1: Guarded Writer Utility

Use the `GuardedWriter` helper for simplified integration:

```python
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

writer = GuardedWriter(project_root=Path.cwd())

# Execution phase
writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/log.txt", "Running...")

# Commit phase
writer.open_commit_gate()
writer.write_durable("LAW/CONTRACTS/_runs/result.json", data)
```

### Pattern 2: Direct Integration

Integrate firewall directly into your tool:

```python
class MyTool:
    def __init__(self, project_root):
        self.firewall = WriteFirewall(
            tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
            durable_roots=["LAW/CONTRACTS/_runs"],
            project_root=project_root
        )

    def execute(self):
        # Stage 1: Execution (tmp writes)
        self.firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/progress.json", data, kind="tmp")

        # Stage 2: Commit
        self.firewall.open_commit_gate()
        self.firewall.safe_write("LAW/CONTRACTS/_runs/receipt.json", data, kind="durable")
```

### Pattern 3: Violation Receipt Logging

Log all violations for audit/debugging:

```python
def run_with_firewall(firewall, operations):
    violations = []

    for op in operations:
        try:
            op(firewall)
        except FirewallViolation as e:
            violations.append(e.violation_receipt)
            # Optionally continue or abort

    # Write violation log
    if violations:
        Path("violations.jsonl").write_text(
            "\n".join(json.dumps(v) for v in violations)
        )
```

## Testing

Run firewall tests:

```bash
pytest CAPABILITY/TESTBENCH/pipeline/test_write_firewall.py -v
```

Expected: All 26 tests pass (100% coverage of firewall policy enforcement).

## Troubleshooting

### "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"

**Problem:** Attempting durable write before commit gate opens.

**Solution:** Call `firewall.open_commit_gate()` before durable writes.

### "FIREWALL_PATH_NOT_IN_DOMAIN"

**Problem:** Path not under any configured tmp or durable root.

**Solution:** Either:
1. Move the write to an allowed domain, OR
2. Add the path to `tmp_roots` or `durable_roots` in configuration

### "FIREWALL_PATH_EXCLUDED"

**Problem:** Attempting to write to an excluded path.

**Solution:** Excluded paths are read-only by design. Write to a different location or remove from exclusions if appropriate.

### "FIREWALL_PATH_TRAVERSAL"

**Problem:** Path contains `..` components.

**Solution:** Use normalized paths without traversal. The firewall rejects `..` for security.

## Architecture Notes

- **Fail-closed:** All violations raise exceptions. No silent failures.
- **Deterministic:** Same violation produces same error code every time.
- **Stateless receipts:** Violation receipts include full policy snapshot for reproducibility.
- **Tool versioning:** Every receipt includes tool version hash for auditability.
- **Path normalization:** Windows backslashes → Unix forward slashes automatically.

## Version History

- **v1.0.0** (Phase 1.5A): Initial implementation with tmp/durable domain separation and commit gate.
