---
uuid: 4a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
title: "Skill Consolidation & Hardening Roadmap"
section: roadmap
bucket: capability/skills
author: Antigravity
priority: High
created: 2026-01-07
modified: 2026-01-07
status: Proposed
summary: "Consolidation plan for reducing skill fragmentation (39 skills -> 25 skills) by merging CORTEX, MCP, COMMIT, and PIPELINE utilities into unified toolkits with 100% documentation coverage."
tags:
- maintenance
- refactoring
- skills
- consolidation
---
<!-- CONTENT_HASH: bf860b2f34693064393bbf8680ba7f3f41cf28a36bd7f369140fb09cf3844012 -->

# Skill Consolidation & Hardening Roadmap

**Date:** 2026-01-07  
**Status:** PROPOSED  
**Objective:** Reduce skill fragmentation, improve discoverability, and harden toolkits.

---

## 1. Executive Summary

Current state analysis reveals significant fragmentation in the `CAPABILITY/SKILLS` directory (39 total skills). This roadmap consolidates 18 fragmented utility skills into 4 cohesive toolkits, ensuring 100% documentation coverage and test fixture presence.

| Metric | Before | After |
|--------|--------|-------|
| Total Skills | 39 | ~25 |
| Skills with README | 5 | 25 |
| Skills with Fixtures | ~20 | 25 |
| Fragmented Utilities | 18 | 0 |

**Consolidation Targets:**
- CORTEX: 5 skills → 1 toolkit
- MCP: 7 skills → 1 toolkit  
- COMMIT: 3 skills → 1 toolkit
- PIPELINE: 3 skills → 1 toolkit

---

## 2. Dependencies & Preconditions

Before starting consolidation:

- [ ] All existing skill tests must pass (`pytest CAPABILITY/TESTBENCH/`)
- [ ] No active branches depending on skills being consolidated
- [ ] Backup of current skill implementations in git history
- [ ] AGENTS.md reviewed for skill references

**Risk Mitigation:**
- Each phase creates the new toolkit FIRST, then migrates logic
- Old skills remain until new toolkit tests pass
- Deprecation notices added before removal
- All changes are reversible via git

---

## 3. Phase 1: CORTEX Toolkit Consolidation

**Priority:** HIGH  
**Goal:** Merge 5 fragmented CORTEX skills into a single `cortex-toolkit`.

### 3.1 Skills to Consolidate

| Source Skill | New Operation | Purpose |
|--------------|---------------|---------|
| `cortex-build` | `build` | Build CORTEX indexes |
| `cas-integrity-check` | `verify_cas` | CAS blob verification |
| `system1-verify` | `verify_system1` | System1 DB verification |
| `cortex-summaries` | `summarize` | Generate summaries |
| `llm-packer-smoke` | `smoke_test` | LLM Packer smoke tests |

### 3.2 Implementation

**Location:** `CAPABILITY/SKILLS/cortex/cortex-toolkit/`

```json
{
  "operation": "build|verify_cas|verify_system1|summarize|smoke_test",
  "target": "optional target path"
}
```

### 3.3 Exit Criteria

- [ ] `cortex-toolkit` created with all 5 operations
- [ ] All operations pass individual tests
- [ ] README.md documents all operations
- [ ] fixtures/ contains test inputs for each operation
- [ ] Old skills marked deprecated (add `status: deprecated` to SKILL.md)
- [ ] Old skill directories removed
- [ ] AGENTS.md references updated

---

## 4. Phase 2: MCP Toolkit Consolidation

**Priority:** HIGH  
**Goal:** Merge 7 fragmented MCP skills into a single `mcp-toolkit`.

### 4.1 Skills to Consolidate

| Source Skill | New Operation | Purpose |
|--------------|---------------|---------|
| `mcp-builder` | `build` | Build MCP servers |
| `mcp-access-validator` | `validate_access` | Access control validation |
| `mcp-extension-verify` | `verify_extension` | Extension verification |
| `mcp-message-board` | `message_board` | Message board operations |
| `mcp-precommit-check` | `precommit` | Pre-commit checks |
| `mcp-smoke` | `smoke` | Smoke testing |
| `mcp-adapter` | `adapt` | MCP adapter logic |

### 4.2 Implementation

**Location:** `CAPABILITY/SKILLS/mcp/mcp-toolkit/`

```json
{
  "operation": "build|validate_access|verify_extension|message_board|precommit|smoke|adapt",
  "config": {}
}
```

### 4.3 Exit Criteria

- [ ] `mcp-toolkit` created with all 7 operations
- [ ] MCP server configuration updated to use new toolkit
- [ ] All operations pass tests
- [ ] README.md with complete documentation
- [ ] fixtures/ with test inputs
- [ ] Old skills deprecated and removed

---

## 5. Phase 3: COMMIT Manager Consolidation

**Priority:** MEDIUM  
**Goal:** Merge 3 commit-related skills into `commit-manager`.

### 5.1 Skills to Consolidate

| Source Skill | New Operation | Purpose |
|--------------|---------------|---------|
| `commit-queue` | `queue` | Manage commit queue |
| `commit-summary-log` | `summarize` | Generate commit summaries |
| `artifact-escape-hatch` | `recover` | Emergency artifact recovery |

### 5.2 Implementation

**Location:** `CAPABILITY/SKILLS/commit/commit-manager/`

```json
{
  "operation": "queue|summarize|recover",
  "target": "commit or artifact reference"
}
```

### 5.3 Exit Criteria

- [ ] `commit-manager` created with all 3 operations
- [ ] All operations pass tests
- [ ] README.md with documentation
- [ ] fixtures/ with test inputs
- [ ] Old skills deprecated and removed

---

## 6. Phase 4: PIPELINE Toolkit Consolidation

**Priority:** MEDIUM  
**Goal:** Merge 3 pipeline skills into `pipeline-toolkit`.

### 6.1 Skills to Consolidate

| Source Skill | New Operation | Purpose |
|--------------|---------------|---------|
| `pipeline-dag-scheduler` | `schedule` | DAG scheduling |
| `pipeline-dag-receipts` | `receipts` | Receipt generation |
| `pipeline-dag-restore` | `restore` | DAG restoration |

### 6.2 Implementation

**Location:** `CAPABILITY/SKILLS/pipeline/pipeline-toolkit/`

```json
{
  "operation": "schedule|receipts|restore",
  "dag_id": "pipeline identifier"
}
```

### 6.3 Exit Criteria

- [ ] `pipeline-toolkit` created with all 3 operations
- [ ] All operations pass tests
- [ ] README.md with documentation
- [ ] fixtures/ with test inputs
- [ ] Old skills deprecated and removed

---

## 7. Phase 5: Documentation & Hardening

**Priority:** MEDIUM  
**Goal:** Ensure every remaining skill has standardized documentation and test fixtures.

### 7.1 Skills Requiring README

After consolidation, remaining skills needing README.md:

**Governance (9 skills):**
- [ ] admission-control
- [ ] canon-governance-check
- [ ] canon-migration
- [ ] canonical-doc-enforcer (has README ✓)
- [ ] ci-trigger-policy
- [ ] intent-guard
- [ ] invariant-freeze
- [ ] master-override
- [ ] repo-contract-alignment

**Utilities (8 skills):**
- [ ] doc-merge-batch-skill (has README ✓)
- [ ] doc-update
- [ ] example-echo
- [ ] file-analyzer
- [ ] pack-validate
- [ ] powershell-bridge
- [ ] prompt-runner
- [ ] skill-creator

**Agents (2 skills):**
- [ ] ant-worker
- [ ] workspace-isolation (has README ✓)

### 7.2 README Template

Each README must contain:
1. **Purpose** — What the skill does
2. **Operations** — Available operations with descriptions
3. **Usage** — CLI and JSON examples
4. **Inputs/Outputs** — Schema documentation
5. **Permissions** — Required access rights

### 7.3 Fixture Requirements

Each skill must have:
- `fixtures/basic/input.json` — Minimal valid input
- `fixtures/basic/expected.json` — Expected output (if deterministic)

### 7.4 Exit Criteria

- [ ] All 25 remaining skills have README.md
- [ ] All 25 remaining skills have fixtures/
- [ ] All skills pass `runner.py` validation
- [ ] Zero governance violations from `critic.py`

---

## 8. Success Criteria (Overall)

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Total Skills | ≤25 | `ls CAPABILITY/SKILLS/*/ \| wc -l` |
| README Coverage | 100% | `find CAPABILITY/SKILLS -name README.md \| wc -l` |
| Fixture Coverage | 100% | `find CAPABILITY/SKILLS -type d -name fixtures \| wc -l` |
| Test Pass Rate | 100% | `pytest CAPABILITY/TESTBENCH/` |
| Fragmented Utils | 0 | Manual audit |

---

## 9. Timeline (Estimated)

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: CORTEX | 2-3 hours | None |
| Phase 2: MCP | 3-4 hours | Phase 1 complete |
| Phase 3: COMMIT | 1-2 hours | None |
| Phase 4: PIPELINE | 1-2 hours | None |
| Phase 5: Docs | 2-3 hours | Phases 1-4 complete |

**Total Estimated Effort:** 9-14 hours

---

## 10. Rollback Plan

If consolidation causes issues:

1. **Immediate:** Revert git commits for affected phase
2. **Old skills preserved:** Git history contains all original implementations
3. **No data loss:** Skills only contain logic, no state
4. **Gradual rollout:** Each phase is independent, partial rollback possible
