# Stewardship Structure

This document defines the human escalation path when the Agent Governance System itself fails or when decisions exceed agent authority.

## Philosophy

No governance system is complete without a "when governance fails" clause. Stewardship fills this gap:
- Agents are bound by canon, but canon is written by humans
- When agents encounter edge cases, they escalate to stewards
- When canon contradicts itself beyond arbitration, stewards decide
- When crisis occurs, stewards take control

## Roles

### Steward

A **Steward** is a human with authority to:
1. Modify CANON files directly (bypassing ceremony in emergencies)
2. Lift quarantine mode
3. Resolve canon conflicts that exceed ARBITRATION.md procedures
4. Make binding decisions on behalf of the governance system

### Maintainer

A **Maintainer** is a human who can:
1. Propose changes via the normal ceremony
2. Review and approve pull requests
3. Run emergency procedures (but not constitutional reset)
4. Escalate to Steward when needed

### Agent

An **Agent** is an AI system operating under AGS that:
1. Follows canon without deviation
2. Escalates ambiguity to Maintainer
3. Cannot modify CANON without ceremony
4. Must stop and notify on crisis detection

## Escalation Matrix

| Situation | Agent Action | Escalate To |
|-----------|--------------|-------------|
| Task ambiguity | Ask clarifying question | User |
| Canon contradiction | Follow ARBITRATION.md | Maintainer (if unresolvable) |
| Critic/fixture failure | Stop, fix issue | User |
| Quarantine triggered | Stop all work, report | Maintainer |
| Constitutional reset needed | Stop immediately | Steward |
| Canon change requested | Follow ceremony | Maintainer approval |

## Contact Configuration

Stewardship contacts are configured in `.steward.json` (gitignored for privacy):

```json
{
  "stewards": [
    {
      "name": "Primary Steward",
      "email": "steward@example.com",
      "notify_on": ["quarantine", "constitutional-reset"]
    }
  ],
  "maintainers": [
    {
      "name": "Primary Maintainer",
      "email": "maintainer@example.com",
      "notify_on": ["critic-failure", "rollback"]
    }
  ],
  "channels": {
    "slack": "#ags-alerts",
    "discord": "ags-alerts"
  }
}
```

## Escalation Procedure

### Step 1: Agent Detects Issue

Agent encounters a situation outside its authority:
- Canon contradiction not covered by ARBITRATION.md
- Security concern
- Unclear user intent that could violate canon

### Step 2: Agent Stops and Documents

Agent MUST:
1. Stop the current action
2. Document the issue in `CONTEXT/open/`
3. Notify the user of the escalation
4. Wait for human response

### Step 3: Human Reviews

Maintainer or Steward:
1. Reviews the open question
2. Makes a decision
3. Documents the decision in an ADR
4. Optionally updates canon if the case is generalizable

### Step 4: Resolution

Agent receives the decision and can proceed.

## Engineering Culture

The following engineering practices are **mandatory** for all code contributions to AGS:

### 1. No Bare Excepts
**Rule**: Never use `except:` without specifying the exception type.

```python
# ❌ FORBIDDEN
try:
    risky_operation()
except:
    pass

# ✅ REQUIRED
try:
    risky_operation()
except (ValueError, KeyError) as e:
    logger.error(f"Operation failed: {e}")
    raise
```

**Rationale**: Bare excepts mask critical errors (KeyboardInterrupt, SystemExit) and make debugging impossible.

### 2. Atomic Writes
**Rule**: All file writes MUST use temp-write + atomic rename.

```python
# ❌ FORBIDDEN
with open("output.json", "w") as f:
    json.dump(data, f)

# ✅ REQUIRED
import tempfile, os
fd, tmp = tempfile.mkstemp(dir=os.path.dirname("output.json"))
try:
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, "output.json")  # Atomic on POSIX
except:
    os.unlink(tmp)
    raise
```

**Rationale**: Prevents partial writes that corrupt state during crashes.

### 3. Headless Execution
**Rule**: No code may spawn visible terminal windows (see ADR-029).

**Enforcement**: `TOOLS/terminal_hunter.py` scans for violations.

### 4. Deterministic Outputs
**Rule**: All artifacts (JSON, manifests, hashes) MUST be deterministic across runs.

**Requirements**:
- Sorted keys in JSON (`sort_keys=True`)
- Stable iteration order (sorted lists, OrderedDict)
- No timestamps in filenames (use content hashes or explicit versioning)

### 5. Safety Caps
**Rule**: All loops and recursive operations MUST have explicit bounds.

**Examples**:
- Max iterations: `for i in range(MAX_CYCLES):`
- Max file size: `if size > MAX_BYTES: raise`
- Timeout: `subprocess.run(..., timeout=30)`

**Rationale**: Prevents infinite loops, resource exhaustion, and runaway processes (see ADR-029 Terminator Mode incident).

### 6. Database Connections
**Rule**: Always use context managers or explicit cleanup for database connections.

```python
# ❌ FORBIDDEN
db = sqlite3.connect("data.db")
cursor = db.execute("SELECT * FROM table")
# Connection may leak

# ✅ REQUIRED
with sqlite3.connect("data.db") as conn:
    cursor = conn.execute("SELECT * FROM table")
    # Auto-commit and close

# OR (for classes)
class MyDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
    
    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

**Additional Requirements**:
- Set `conn.row_factory = sqlite3.Row` for dict-like access
- Use parameterized queries (`?` placeholders) to prevent SQL injection
- Call `conn.commit()` explicitly after writes
- On Windows: Add `time.sleep(0.5)` after `close()` before unlinking DB files

**Rationale**: Prevents file handle leaks, ensures transactions commit, avoids Windows file locking issues.

### 7. Never Bypass Tests
**Rule**: Never use `--no-verify` or skip pre-commit hooks. Fix the root cause.

```bash
# ❌ FORBIDDEN
git commit --no-verify -m "quick fix"

# ✅ REQUIRED
# 1. Identify why hook fails
# 2. Fix the underlying issue
# 3. Commit normally
git commit -m "fix: proper commit with tests passing"
```

**Rationale**: Bypassed tests lead to broken CI. The 2025-12-28 `export_to_json` incident occurred because a function was assumed to exist but was never implemented. Pre-commit hooks exist to catch such issues early.

### 8. Cross-Platform Scripts
**Rule**: All shell scripts must work on both Linux/macOS and Windows (Git Bash).

**Requirements**:
- Python: Use `python3 || python` fallback (Windows lacks `python3`)
- Paths: Use forward slashes or `Path()` objects
- Line endings: Configure `.gitattributes` for `* text=auto`
- Commands: Avoid `command -v` (unreliable in Git's Windows shell)

```bash
# ✅ Cross-platform Python detection
if python3 --version >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif python --version >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: No Python found"; exit 1
fi
```

**Rationale**: CI runs on Linux but developers use Windows. Scripts must work everywhere.

### 9. Interface Regression Tests
**Rule**: When Module A imports and calls Module B, there MUST be a test verifying B's interface.

```python
# ✅ REQUIRED: test_query.py
def test_export_to_json_exists():
    """Verify export_to_json function exists in query module."""
    import query as cortex_query
    assert hasattr(cortex_query, 'export_to_json'), \
        "query module must have export_to_json() (required by cortex.build.py)"
```

**Rationale**: The 2025-12-28 CI failure occurred because `cortex.build.py` called `query.export_to_json()` which was never implemented. A simple existence test would have caught this before merge.

### 10. Amend Over Pollute
**Rule**: When actively fixing the same issue across multiple iterations, amend the previous commit instead of creating new ones.

```bash
# ❌ FORBIDDEN (commit pollution)
git commit -m "fix: attempt 1"
git commit -m "fix: attempt 2"  
git commit -m "fix: final fix"

# ✅ REQUIRED (clean history)
git commit -m "fix: initial attempt"
# ...make more fixes...
git add .
git commit --amend -m "fix: complete solution"
git push --force-with-lease
```

**When to Amend**:
- Same logical fix, multiple iterations
- Not yet reviewed by others
- Within the same work session

**When NOT to Amend**:
- Different logical changes
- Already reviewed/merged
- Shared branches with active collaborators

**Rationale**: Commit history should tell a clear story. "fix, fix again, really fix, final fix" obscures intent. One clean commit per logical change.

## Authority Boundaries

### What Agents CAN Do

- Execute approved skills
- Create new files (non-canon)
- Propose ADRs for review
- Query context and cortex
- Run validation tools
- Report issues

### What Agents CANNOT Do

- Modify CANON/* without completing full ceremony
- Lift quarantine mode
- Bypass the commit ceremony
- Ignore critic failures
- Make constitutional decisions
- Override explicit user denial

### What Only Stewards CAN Do

- Constitutional reset
- Emergency canon modification
- Permanent agent restrictions
- Dissolve the governance system

## Emergency Steward Actions

In a constitutional crisis, stewards may:

1. **Direct Canon Edit**: Modify CANON files directly without ceremony (document afterward)
2. **Agent Termination**: Instruct agents to cease all operations
3. **System Reset**: Restore to a known-good tagged release
4. **Governance Suspension**: Temporarily suspend normal governance (document extensively)

All emergency actions MUST be documented in `CONTRACTS/_runs/steward_logs/steward-actions.log` (see ADR-015).

## Template: Steward Decision Record

```markdown
# Steward Decision: [Date] - [Title]

## Situation
[What happened that required steward intervention]

## Decision
[What the steward decided]

## Rationale
[Why this decision was made]

## Actions Taken
- [ ] Action 1
- [ ] Action 2

## Follow-up Required
- [ ] Update canon?
- [ ] Create ADR?
- [ ] Notify stakeholders?

## Steward: [Name]
```

---

Added: 2025-12-21
