---
title: MASTER_PROMPT_TEMPLATE_CANON
version: 1.4
status: CANONICAL
generated_on: 2026-01-04
scope: Governor workflow template for generating all remaining per-task prompts
---
<!-- CANON_HASH: 032ab3efda97e8294e8474d41ba9462dc3a914e124e8042bcffc14a55278f67f -->

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
- task_index: list of {task_id, prompt_path, receipt_path, report_path, depends_on, warnings}

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
   e) If unknown required facts exist:
      - classify as UNKNOWN_BLOCKER or UNKNOWN_DEFERRABLE
      - BLOCKER: do not generate prompt for this task; record STOP item
      - DEFERRABLE: generate prompt with FILL_ME__ tokens; record warning
   f) Draft prompt using canonical template.
   g) Run prompt QA checklist.
   h) If CAPABILITY/TOOLS/linters/lint_prompt_pack.sh exists (requires bash-compatible shell, e.g. WSL):
      - run it via `bash` on the prompt
      - exit 1 blocks pack generation
      - exit 2 records a warning and continues
   i) Write prompt file.
4) Write pack manifest and report.

## 5) STOP behavior
STOP is per-task only. Continue generating prompts for unrelated tasks.

## 6) Receipt requirements (lint metadata)
Receipts MUST include the following REQUIRED fields for lint verification:
- lint_command: the exact linter command executed
- lint_exit_code: exit status (0=PASS, 1=FAIL, 2=WARNING)
- lint_result: one of PASS, FAIL, or WARNING
- linter_ref: optional (path/version/hash of the linter used)
