<!-- CONTENT_HASH: ff9e9936396d36d7feae7297ebaa381b8ade6180190bfd844c5d978c07d258e4 -->

# Stewardship Structure

This document defines the human escalation path when the Agent Governance System itself fails or when decisions exceed agent authority.

## Philosophy

No governance system is complete without a "when governance fails" clause:
- Agents are bound by canon, but canon is written by humans
- When agents encounter edge cases, they escalate to stewards
- When canon contradicts itself beyond arbitration, stewards decide
- When crisis occurs, stewards take control

## Roles

### Steward
A **Steward** is a human with authority to:
1. Modify CANON files directly (bypassing ceremony in emergencies)
2. Lift quarantine mode
3. Resolve canon conflicts exceeding ARBITRATION.md
4. Make binding decisions on behalf of the governance system

### Maintainer
A **Maintainer** is a human who can:
1. Propose changes via normal ceremony
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
  "stewards": [{"name": "Primary Steward", "email": "...", "notify_on": ["quarantine", "constitutional-reset"]}],
  "maintainers": [{"name": "Primary Maintainer", "email": "...", "notify_on": ["critic-failure", "rollback"]}],
  "channels": {"slack": "#ags-alerts"}
}
```

## Escalation Procedure

1. **Agent detects issue** outside its authority
2. **Agent stops and documents** in `LAW/CONTEXT/open/`, notifies user, waits
3. **Human reviews** and decides (documents in ADR if generalizable)
4. **Resolution** — agent receives decision and proceeds

## Authority Boundaries

| Entity | CAN Do | CANNOT Do |
|--------|--------|-----------|
| **Agent** | Execute skills, create non-canon files, propose ADRs, query context, run validation, report issues | Modify CANON directly, lift quarantine, bypass ceremony, ignore critic failures, make constitutional decisions |
| **Steward** | Constitutional reset, emergency canon modification, permanent agent restrictions, dissolve governance | — |

## Engineering Culture (Mandatory Practices)

These practices are **mandatory** for all AGS code contributions:

### Exception Handling
```python
# ❌ FORBIDDEN: except:
# ✅ REQUIRED: except (ValueError, KeyError) as e: ...
```
Never use bare `except:` — it masks critical errors.

### Atomic Writes
```python
# ✅ REQUIRED: temp-write + os.replace()
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, 'w') as f: json.dump(data, f)
os.replace(tmp, path)
```
Prevents partial writes corrupting state during crashes.

### Headless Execution
No code may spawn visible terminal windows (see ADR-029). Enforced by `terminal_hunter.py`.

### Deterministic Outputs
All artifacts must be deterministic: `sort_keys=True` in JSON, stable iteration order, no timestamps in filenames.

### Safety Caps
All loops/recursion MUST have explicit bounds: `for i in range(MAX)`, `timeout=30`, `if size > MAX: raise`.

### Database Hygiene
Always use context managers; `conn.row_factory = sqlite3.Row`; parameterized queries; explicit `commit()`.

### Never Bypass Tests
Never use `--no-verify`. Fix the root cause instead.

### Cross-Platform Scripts
Scripts must work on Linux/macOS AND Windows (Git Bash). Use `python3 || python` fallback.

### Interface Regression Tests
When Module A imports Module B, there MUST be a test verifying B's interface exists.

### Amend Over Pollute
When fixing same issue multiple iterations: `git commit --amend`, not `fix1, fix2, fix3` commits.

### Repository Hygiene
- Logs → `LAW/CONTRACTS/_runs/` or designated folders
- Temp files (`*.tmp`, `*.bak`) → gitignored or deleted immediately
- `__pycache__` → gitignored

## Emergency Steward Actions

In constitutional crisis, stewards may:
1. **Direct Canon Edit** — modify without ceremony (document afterward)
2. **Agent Termination** — instruct agents to cease operations
3. **System Reset** — restore to known-good tagged release
4. **Governance Suspension** — temporarily suspend normal governance

All emergency actions MUST be logged to `LAW/CONTRACTS/_runs/steward_logs/steward-actions.log`.

## Template: Steward Decision Record

```markdown
# Steward Decision: [Date] - [Title]

## Situation
[What happened]

## Decision
[What was decided]

## Rationale
[Why]

## Actions Taken
- [ ] Action 1

## Follow-up
- [ ] Update canon?
- [ ] Create ADR?

## Steward: [Name]
```

---

Added: 2025-12-21
