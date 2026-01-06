---
title: MASTER_PROMPT_TEMPLATE_CANON
version: 1.5
status: CANONICAL
generated_on: 2026-01-06
scope: Governor workflow template for generating all remaining per-task prompts
---
<!-- CANON_HASH (sha256 over file content excluding this line): D0D3948770AEBA5CD5EDAB4B39BF62D7100BD443976909CCBFE03C90FE7BBEEF -->

## 0) Authority
Subordinate to:
- NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md
- NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md

## 1) Role
You are the Governor. Generate one canonical per-task prompt for every unfinished roadmap item.

## 2) Inputs
- Roadmap (TODO-only, numeric phases)
- NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md
- NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md
- Repo canon and contracts under LAW/

## 3) Outputs
- PROMPTS/phase-N/task-N.M[_K]_slug.md
- PROMPTS/PROMPT_PACK_MANIFEST.json
- PROMPTS/PROMPT_PACK_REPORT.md

Pack manifest MUST include:
- policy_canon_sha256
- guide_canon_sha256
- task_index: list of objects with:
  - task_id
  - prompt_path
  - receipt_path
  - report_path
  - depends_on
  - warnings

Recommended (manifest fields):
- workspace_policy: object with:
  - default_mode: "worktree"
  - allow_main_override: false
- task_workspaces: list of objects with:
  - task_id
  - workspace_name_suggestion
  - branch_name_suggestion

## 4) Workflow (deterministic)
1) Enumerate unfinished tasks from the roadmap.
2) Compute sha256 for policy and guide and carry into:
   - prompt headers
   - pack manifest
3) For each task:
   a) Choose granularity using guide heuristics.
   b) Select primary_model + fallback_chain per policy.
   c) Identify depends_on edges and set header depends_on.
   d) Determine receipt_path and report_path and set header fields.
   e) Ensure workspace safety is enforceable:
      - prompts must include preflight checks for branch + clean-state
      - default expectation is isolated workspace (worktree or clone)
      - main workspace use requires explicit opt-in and must be rare
   f) If unknown required facts exist:
      - classify as UNKNOWN_BLOCKER or UNKNOWN_DEFERRABLE
      - BLOCKER: do not generate prompt for this task; record STOP item
      - DEFERRABLE: generate prompt with FILL_ME__ tokens; record warning
   g) Draft prompt using canonical template.
   h) Run prompt QA checklist.
   i) If CAPABILITY/TOOLS/linters/lint_prompt_pack.sh exists (requires bash-compatible shell, e.g. WSL):
      - run it via `bash` on the prompt pack directory
      - exit 1 blocks pack generation
      - exit 2 records a warning and continues
   j) Write prompt file.
4) Write pack manifest and report.

## 5) STOP behavior
STOP is per-task only. Continue generating prompts for unrelated tasks.

## 6) Receipt requirements (lint metadata)
Receipts MUST include these fields when lint is run:
- lint_command: exact linter command executed
- lint_exit_code: exit status (0=PASS, 1=FAIL, 2=WARNING)
- lint_result: PASS | FAIL | WARNING
- linter_ref: optional (path/version/hash of the linter used)
