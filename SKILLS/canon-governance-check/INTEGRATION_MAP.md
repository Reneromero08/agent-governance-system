# Canon Governance Check - System Integration Map

## Overview
The canon governance check is now "alive" and integrated at multiple layers of the AGS system.

## Integration Layers

### 1. **Cortex Layer** (Knowledge & Provenance)
- **Integration Point**: `SKILLS/canon-governance-check/run.py`
- **Cortex Event Schema**: Logs to `CONTRACTS/_runs/<run_id>/events.jsonl`
- **Determinism**: Canonical JSON with caller-supplied timestamps
- **Usage**: 
  ```bash
  export CORTEX_RUN_ID="governance-check"
  export CORTEX_TIMESTAMP="SENTINEL"
  python SKILLS/canon-governance-check/run.py
  ```

### 2. **CI/CD Layer** (Continuous Enforcement)
- **Integration Point**: `.github/workflows/contracts.yml`
- **Execution Order**:
  1. Cortex index build
  2. Node.js setup
  3. **Canon governance check** ← Validates before further checks
  4. Critic checks
  5. Token linting
  6. Contract fixtures
  7. Pytest suite
- **Fail Behavior**: Blocks PR/push if changelog missing for significant changes

### 3. **Skill Layer** (Reusable Component)
- **Location**: `SKILLS/canon-governance-check/`
- **Components**:
  - `SKILL.md` - Documentation and usage
  - `run.py` - Python wrapper with Cortex integration
  - `scripts/pre-commit` - Git pre-commit hook
  - `CORTEX_INTEGRATION.md` - Provenance integration guide

### 4. **Git Hook Layer** (Pre-commit Prevention)
- **Installation**:
  ```bash
  cp SKILLS/canon-governance-check/scripts/pre-commit .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit  # Unix/Mac
  ```
- **Behavior**: Prevents commits when governance check fails

### 5. **Developer Layer** (Manual Invocation)
- **Core Script**: `TOOLS/check-canon-governance.js`
- **Usage**:
  ```bash
  node TOOLS/check-canon-governance.js          # Normal
  node TOOLS/check-canon-governance.js --verbose # Debug
  python SKILLS/canon-governance-check/run.py   # With Cortex
  ```

## Data Flow

```
Developer Change
    ↓
Git Add/Commit
    ↓
Pre-commit Hook (optional) ──→ Governance Check ──→ Fail: Block commit
    ↓ Pass                                         ↓ Pass
Git Push
    ↓
GitHub CI
    ↓
Cortex Build ──→ Governance Check ──→ Fail: Block PR/merge
    ↓ Pass                           ↓ Pass
Cortex Provenance Ledger
    ↓
Merge to Main
```

## What Gets Checked

### Behavior Changes (❌ Blocks without CANON/CHANGELOG)
- `TOOLS/*.py` - Runtime tools
- `CATALYTIC-DPT/PRIMITIVES/*.py` - Core primitives
- `CATALYTIC-DPT/PIPELINES/*.py` - Pipeline runtime
- `SKILLS/*` (non-.md files) - Agent capabilities
- `.github/workflows/*.yml` - CI enforcement

### Rule Changes (❌ Blocks without CANON/CHANGELOG)
- `CANON/*.md` - Canon specifications
- `CATALYTIC-DPT/SPECTRUM/*.md` - Frozen law
- `CATALYTIC-DPT/SCHEMAS/*.json` - Schema definitions
- `SCHEMAS/*.json` - Contract schemas

### Decision Changes (❌ Blocks without CANON/CHANGELOG)
- `CONTEXT/decisions/ADR-*.md` - Architecture Decision Records

### Warnings (⚠️ Warns but allows)
- CAT-DPT changes without `CATALYTIC-DPT/CHANGELOG.md`
- Canon changes without `AGENTS.md` sync

### Informational (ℹ️ Info only)
- `CANON/planning/*` - Planning docs
- `CONTEXT/planning/*` - Planning records
- `*ROADMAP*.md` - Roadmap updates

## Cortex Queries

### Find governance failures
```bash
cat CONTRACTS/_runs/*/events.jsonl | \
  jq 'select(.type == "governance_check" and .passed == false)'
```

### Count governance checks
```bash
cat CONTRACTS/_runs/*/events.jsonl | \
  jq 'select(.type == "governance_check")' | wc -l
```

### Get latest governance status
```bash
cat CONTRACTS/_runs/*/events.jsonl | \
  jq 'select(.type == "governance_check")' | \
  tail -n 1 | \
  jq '.passed'
```

## Implementation Files

| Component | Path | Purpose |
|-----------|------|---------|
| Core Script | `TOOLS/check-canon-governance.js` | Main governance logic (Node.js) |
| Skill Wrapper | `SKILLS/canon-governance-check/run.py` | Cortex integration wrapper (Python) |
| Skill Docs | `SKILLS/canon-governance-check/SKILL.md` | Usage documentation |
| Cortex Docs | `SKILLS/canon-governance-check/CORTEX_INTEGRATION.md` | Provenance integration guide |
| Pre-commit | `SKILLS/canon-governance-check/scripts/pre-commit` | Git hook template |
| CI Config | `.github/workflows/contracts.yml` | GitHub Actions integration |

## Success Metrics

✅ **Active Enforcement**:
- CI blocks un-documented changes
- Pre-commit hook available for local enforcement
- Cortex tracks all governance checks

✅ **Multi-Layer Defense**:
1. Developer: Manual check before commit
2. Git: Pre-commit hook (optional)
3. CI: Automated check on push/PR
4. Cortex: Provenance ledger with event history

✅ **Observable**:
- All governance events logged to Cortex
- Query-able history of checks
- Deterministic event records

## Next Steps (Optional)

1. **Auto-install pre-commit hook** in onboarding script
2. **Add governance status badge** to README
3. **Create governance metrics dashboard** from Cortex events
4. **Extend to subsystem changelogs** (warn → error for CAT-DPT)
