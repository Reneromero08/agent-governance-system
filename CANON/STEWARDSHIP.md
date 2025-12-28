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
