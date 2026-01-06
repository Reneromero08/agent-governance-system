---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 Prompt Pack Audit Report
section: report
bucket: INBOX/reports
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: 31787e6d8d7cafe966a5f5de78143e5022f4ddcbb0ec3ced78a759d71112e3b8 -->
# Prompt Pack Audit & Optimization Report
**Date:** 2026-01-05
**Status:** FINAL
**Auditor:** Antigravity

## 1. Executive Summary
The current prompt pack in `NAVIGATION/PROMPTS/` is **operationally valid but highly inefficient, fragile, and riddled with dead links**. It suffers from massive structural duplication (~40-50% token waste per file), contradictory governance instructions, and widespread reference rot (dead linter paths, stale filenames, broken index).

While the rigor is high (fail-closed, receipt-based), the implementation is brittle. Agents are forced to parse conflicting layers of instructions (Wrapper vs. Source Body) and rely on allowlists that technically forbid the very work they request.

**Verdict:** ⚠️ **CRITICAL REFACTOR REQUIRED** (High Waste, High Fragility, Broken Chains)

---

## 2. Quantified Inventory

| Metric | Count | Notes |
| :--- | :--- | :--- |
| **Total Task Prompts** | 32 | spread across Phases 01-08, 10 |
| **Governance/Canon Files** | 7 | Orientation, Policy, Guide, Templates, Routing |
| **Missing Phases** | 1 | **Phase 09** is completely documented but missing |
| **Dead Linter References** | 37+ | Refs to `scripts/lint-prompt.sh` (file does not exist) |
| **Manifest Path Mismatches** | ~10 | Manifest points to `foo.md`, disk has `foo✅.md` |
| **Index Stale Links** | 100% of completed | `INDEX.md` links to non-checkmark versions |
| **Dependency Gaps** | 100% | **32/32** tasks have `depends_on: []` |
| **"Compileall" Copy-Pastes** | 19 | `python -m compileall` used for checklist extraction |

---

## 3. Key Findings

### A. The "Wrapper Paradox" (Major Wasted Tokens)
Almost every prompt file follows this pattern:
1.  **Frontmatter** (YAML)
2.  **Outer Instructions**: "You are an AGS executor..."
3.  **Block Quote ("Source prompt body")**: A copy-paste of a *different* prompt format.
4.  **Outer Instructions (Again)**: A restatement of Scope, Plan, Validation.

**Impact:** The Agent reads the Plan and Scope **three times** per run.

### B. Contradictory Governance
In `3.2_memory-integration.md` and others:
-   **Fallback Chains:** Frontmatter (`Claude Sonnet -> Gemini Pro`) vs. Body (`GPT-5.2 -> Gemini 3 -> Claude Opus`).
-   **Write Allowlists:** Body permits specific paths; Outer Footer restricts changes to "only files required" but doesn't whitelist specific implementation paths (like `SKILLS/`), causing strict agents to `BLOCKED_UNKNOWN`.

### C. The "Spec Void"
Prompts like `3.2_memory-integration` command complex tasks ("Implement Context Window Management") with zero reference to a design doc or behavior spec.

### D. Dead References
-   **Linter:** 37+ files reference `scripts/lint-prompt.sh`. The actual tool is `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`.
-   **Impact:** Agents attempting to run the linter will fail or skip validation silentely.

### E. Missing Phase 09
The directory structure jumps from `PHASE_08` to `PHASE_10`. Phase 09 is missing entirely from the pack.

### F. Universal Broken Dependencies
Every single task in `PROMPT_PACK_MANIFEST.json` and the markdown headers has `depends_on: []`.
-   **Risk:** Agents can execute Phase 10 tasks before Phase 1 is done.
-   **Reality:** Logical dependencies exist (Phase 3 relies on Phase 2 CAS) but are not enforced.

### G. Manifest Path Mismatches
Tasks that have been completed were renamed with a checkmark (e.g., `1.1...✅.md`) on disk.
-   **Manifest:** Points to `1.1... .md` (without checkmark)
-   **Disk:** Has `1.1...✅.md` (with checkmark)
-   **Result:** Automation relying on the manifest will crash with `FileNotFound`.
-   **Affected:** ~10 completed tasks across all phases.

### H. Incomplete Allowlists
In `6.2_write-path-memory-persistence.md`:
-   **Requirement:** "Expose MCP tools", "Add cassette schemas".
-   **Allowlist:** Only allows `AGS_ROADMAP_MASTER.md` and `receipts`.
-   **Result:** A compliant agent **cannot** complete the task without violating the allowlist.

### I. "Compileall" Misuse
19+ prompts instruct the agent to extract checklist items using:
`python -m compileall . (must exit 0 or hard fail)`
This compiles Python bytecode and has **nothing** to do with reading markdown checklists. It is a hallucinatory copy-paste artifact.

### J. INDEX.md Stale Links
`NAVIGATION/PROMPTS/INDEX.md` serves as the human-readable directory of all task prompts.
-   **Issue:** All completed task links point to non-checkmark filenames (e.g., `1.1_hardened-inbox-governance-s-2.md`)
-   **Reality:** Files on disk have checkmarks (e.g., `1.1_hardened-inbox-governance-s-2✅.md`)
-   **Result:** 100% of completed task links in INDEX.md are broken.
-   **Affected:** 9+ completed tasks listed in the index.

---

## 4. Structural Analysis (Example: `3.2_memory-integration.md`)

| Section | Tokens (Est.) | Status | Issue |
| :--- | :--- | :--- | :--- |
| **Frontmatter** | 100 | ✅ Keep | Essential for manifests/automation. |
| **Outer Role/Goal** | 150 | ⚠️ Trim | Redundant with standard policy. |
| **Source Body (Block)** | 800 | ❌ **REMOVE** | "Ghost of Prompts Past". Duplicates data. |
| **Outer Plan/Validation** | 500 | ⚠️ Refine | Good structure, but needs to be the SSOT. |

**Total Estimated Waste:** ~1000 tokens per file (~50%).

---

## 5. Recommendations for Refactor

### Step 1: Collapse to SSOT
Remove the "Source prompt body" block. Merge instructions into a single `## PLAN` and `## SCOPE`.

### Step 2: Fix the Paths & Refs
-   Update Linter path to `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`.
-   Fix `INDEX.md` and `PROMPT_PACK_MANIFEST.json` to match current filenames (remove `✅` from filenames or update refs? **Recommendation: Remove `✅` from filenames and use a metadata status field instead**).

### Step 3: Sanity Check Commands
-   Replace `python -m compileall` with the correct `grep`/`python` script for checklist extraction.
-   Ensure Allowlists actually include the directories needed for the task (e.g., `CAPABILITY/MCP/...`).

### Step 4: Populate Dependencies
-   Update `depends_on` fields to reflect the linear phase progression at minimum.

---

## 6. Action Plan
1.  **Scripted Cleanup:**
    -   Rename files to remove `✅` (restore canonical naming).
    -   Update `PROMPT_PACK_MANIFEST.json` status/hashes.
2.  **Content Refactor:**
    -   Iterate through all 32 prompts.
    -   Strip "Source prompt body".
    -   Fix linter path.
    -   Fix `compileall` command.
    -   Expand allowlists based on task description.
3.  **Validation:**
    -   Run `lint_prompt_pack.sh` (the real one).
