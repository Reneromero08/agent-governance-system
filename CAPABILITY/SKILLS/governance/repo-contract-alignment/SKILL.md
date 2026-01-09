---
name: repo-contract-alignment
description: "Align the repository with its stated governance by scanning contract docs, extracting explicit rules, running checks, and implementing the smallest compliant fixes with enforcement."
---
<!-- CONTENT_HASH: dc5da4881546efe848cbadf457ccc141e4512024375fae7f5649fa98839fc310 -->

**required_canon_version:** >=3.0.0


# Skill: repo-contract-alignment

**Version:** 0.1.0
**Status:** Active


## Purpose

Align the repository with its stated governance by scanning contract docs, extracting explicit rules, running checks, and implementing the smallest compliant fixes with enforcement.

## Trigger

Use when asked to align repo behavior with canon/contract docs, audit governance compliance, or implement minimal fixes plus enforcement for regressions.

## Workflow

1. Confirm intent gate
   - If the user is asking for analysis or strategy only, stop after Findings/Plan and ask for explicit implementation approval.
   - Do not touch CANON or edit existing CONTEXT unless the task is explicitly about rules/governance/memory updates.

2. Ensure cortex is available
   - If `CORTEX/_generated/cortex.db` is missing, run `python CORTEX/cortex.build.py`.
   - Use `python CORTEX/query.py --json` or `--find` for file discovery; avoid raw filesystem scans.

3. Identify contract docs
   - From the cortex index, collect docs matching: `README.md`, `AGENTS.md`, `CANON/`, `CONTEXT/decisions/` (ADRs), `ROADMAP`, `CONTRIBUTING`, `SECURITY`, `CONTEXT/maps/*`, and any explicit “contract” docs.

4. Extract explicit rules
   - Read only those contract docs and list their rules.
   - Resolve conflicts using the authority gradient; note any conflicts.

5. Run existing checks/tests/build scripts
   - Always run `python TOOLS/critic.py` and `python CONTRACTS/runner.py`.
   - Run any additional scripts referenced by contract docs.

6. Report issues and minimal fixes
   - Produce a prioritized issue list (P0 violations, P1 missing enforcement, P2 hygiene).
   - Propose the smallest fixes and enforcement to prevent regressions.

7. Implement and re-check
   - Apply the smallest correct changes, add/adjust fixtures if behavior changes, then re-run checks until passing.
   - Write artifacts only to allowed roots (`CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/`).

## Output format

Always respond with: Findings, Plan, Changes, Commands run, Next.

**required_canon_version:** >=3.0.0

