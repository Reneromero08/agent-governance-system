---
name: canon-governance-check
version: "0.1.0"
description: Enforces changelog updates for significant changes to CANON/, TOOLS/, schemas, and ADRs
compatibility: all
---

<!-- CONTENT_HASH: 6ccf92d5a58730e071eab8db36222e2c03e3779f43e2056dcceac11528ab4ba1 -->

**required_canon_version:** >=3.0.0




# Canon Governance Check Skill

**Version:** 0.1.0

**Status:** Active

**Required Canon Version:** >=2.6.0 <3.0.0

## Purpose
Enforces documentation hygiene by requiring changelog updates when significant system changes are made.

## What it checks

### Behavior Changes (requires CANON/CHANGELOG.md)
- `TOOLS/` - Runtime behavior
- `CATALYTIC-DPT/PRIMITIVES/` and `CATALYTIC-DPT/PIPELINES/` - Core runtime
- `SKILLS/` - Agent capabilities
- `.github/workflows/` - CI enforcement

### Rule Changes (requires CANON/CHANGELOG.md)
- `CANON/*.md` - Canon specifications
- `CATALYTIC-DPT/SPECTRUM/*.md` - Frozen law
- `SCHEMAS/*.json` - Contract definitions

### Decision Changes (requires CANON/CHANGELOG.md)
- `CONTEXT/decisions/*.md` - Architecture Decision Records

### Warnings (not errors)
- CAT-DPT changes without `CATALYTIC-DPT/CHANGELOG.md` update
- Canon changes without `AGENTS.md` sync

## Usage

### CLI
```bash
# Normal check
node TOOLS/check-canon-governance.js

# Verbose mode (show all changed files)
node TOOLS/check-canon-governance.js --verbose
```

### As a Skill
```bash
python SKILLS/canon-governance-check/run.py
```

### In CI
Automatically runs in `.github/workflows/contracts.yml` on every push/PR.

### Pre-commit Hook
```bash
# Install pre-commit hook
cp SKILLS/canon-governance-check/scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Exit Codes
- `0` - Pass (no significant changes or changelog updated)
- `1` - Fail (significant changes without changelog)

## Cortex Integration
When run with `CORTEX_RUN_ID` set, logs governance check events to the Cortex provenance ledger.

## Implementation
- **Core script**: `TOOLS/check-canon-governance.js` (Node.js)
- **Skill wrapper**: `SKILLS/canon-governance-check/run.py`
- **Pre-commit hook**: `SKILLS/canon-governance-check/scripts/pre-commit`

**required_canon_version:** >=3.0.0

