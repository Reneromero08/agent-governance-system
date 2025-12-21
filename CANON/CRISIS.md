# Crisis Mode Procedures

This document defines emergency procedures for handling governance failures, security incidents, and system isolation scenarios.

## Philosophy

When governance fails catastrophically, **predictable recovery** is more valuable than perfect recovery. These procedures prioritize:
1. **Stopping the damage** (isolation)
2. **Preserving evidence** (audit trail)
3. **Human escalation** (stewardship)
4. **Documented recovery** (ceremony)

## Crisis Levels

| Level | Name | Trigger | Response |
|-------|------|---------|----------|
| **0** | Normal | All checks pass | Continue operations |
| **1** | Warning | Critic fails, fixtures pass | Fix before commit |
| **2** | Alert | Fixtures fail | Rollback last change |
| **3** | Quarantine | Canon corruption suspected | Isolate agent, human review |
| **4** | Constitutional | CONTRACT.md compromised | Full reset to known-good state |

## Emergency CLI

The `TOOLS/emergency.py` script provides concrete CLI modes for crisis handling.

### Usage

```bash
# Check current status
python TOOLS/emergency.py --status

# Level 1: Run critic and fixtures
python TOOLS/emergency.py --mode=validate

# Level 2: Rollback last change
python TOOLS/emergency.py --mode=rollback

# Level 3: Quarantine (blocks all writes, preserves state)
python TOOLS/emergency.py --mode=quarantine

# Level 4: Constitutional reset (restore to tagged release)
python TOOLS/emergency.py --mode=constitutional-reset --tag=v1.0.0

# Exit quarantine (requires confirmation)
python TOOLS/emergency.py --mode=restore
```

## Procedures

### Level 1: Validation Failure

**Symptoms:** 
- `TOOLS/critic.py` fails
- Fixture tests fail
- Canon sync warnings

**Procedure:**
1. Stop all agent work
2. Run `python TOOLS/emergency.py --mode=validate`
3. Review output to identify the issue
4. Fix the violation before proceeding
5. Re-run validation until clean

### Level 2: Rollback Required

**Symptoms:**
- Recent change broke fixtures
- Canon and behavior out of sync
- Agent made unauthorized changes

**Procedure:**
1. Run `python TOOLS/emergency.py --mode=rollback`
2. This creates a backup of current state
3. Reverts to the last known-good commit
4. Creates an ADR documenting the rollback
5. Human reviews what went wrong

### Level 3: Quarantine Mode

**Symptoms:**
- Canon file has unexpected modifications
- Agent behavior contradicts canon
- Suspicion of prompt injection or manipulation

**Procedure:**
1. Run `python TOOLS/emergency.py --mode=quarantine`
2. This:
   - Creates a `.quarantine` lock file
   - Saves current cortex and git state
   - Blocks all write operations
   - Alerts steward (if configured)
3. Human investigates the anomaly
4. Once resolved: `python TOOLS/emergency.py --mode=restore`

### Level 4: Constitutional Reset

**Symptoms:**
- CONTRACT.md has been corrupted
- INVARIANTS.md has unauthorized changes
- Fundamental governance breakdown

**Procedure:**
1. **DO NOT** attempt automated fixes
2. Human steward takes control
3. Run `python TOOLS/emergency.py --mode=constitutional-reset --tag=<last-known-good>`
4. This:
   - Creates full backup of current state
   - Resets all CANON files to the tagged release
   - Preserves CONTEXT (decisions are history)
   - Regenerates cortex
   - Creates detailed audit log
5. Human reviews all changes since the tagged release
6. Selectively re-applies valid changes with proper ceremony

## Quarantine Lock File

When quarantine mode is active, a `.quarantine` file is created in the project root:

```json
{
  "entered": "2025-12-21T15:30:00Z",
  "reason": "Suspected canon modification",
  "triggered_by": "human",
  "git_hash": "abc123...",
  "steward_notified": true
}
```

Tools and agents MUST check for this file before writing:
- If `.quarantine` exists, refuse all write operations
- Only human steward can lift quarantine

## Integration

### With Critic

The critic tool should check for quarantine status:
```python
if Path(".quarantine").exists():
    print("[QUARANTINE] System is in quarantine mode. No changes allowed.")
    sys.exit(1)
```

### With MCP

The MCP server should refuse write tools during quarantine:
```python
def _check_quarantine(self):
    if (PROJECT_ROOT / ".quarantine").exists():
        return {
            "content": [{"type": "text", "text": "System is in quarantine mode."}],
            "isError": True
        }
    return None
```

### With Agent Contract

Add to AGENTS.md:
> If a `.quarantine` file exists in the project root, the agent MUST stop all work and notify the user. The agent MUST NOT delete or modify the quarantine file.

## Stewardship Escalation

When a crisis occurs, the system should escalate to human stewards:

1. **Email** (if configured): Send details of the crisis
2. **Slack/Discord** (if configured): Post to #ags-alerts
3. **Log file**: Always write to `LOGS/crisis.log`

See `CANON/STEWARDSHIP.md` for escalation paths.

## Audit Trail

All emergency actions are logged to `LOGS/emergency.log`:

```
2025-12-21T15:30:00 QUARANTINE entered: "Suspected canon modification"
2025-12-21T15:45:00 QUARANTINE investigated by: human
2025-12-21T16:00:00 QUARANTINE lifted: "False positive, no corruption found"
```

## Recovery Ceremony

After any Level 3+ crisis, perform the recovery ceremony:

1. **Review**: What happened? Document in an ADR.
2. **Verify**: Run full validation suite.
3. **Rebuild**: Regenerate cortex and packs.
4. **Notify**: Inform stakeholders of the incident.
5. **Improve**: Update procedures to prevent recurrence.

---

Added: 2025-12-21
