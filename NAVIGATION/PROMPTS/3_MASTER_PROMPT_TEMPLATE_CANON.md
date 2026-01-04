# MASTER_PROMPT_TEMPLATE_CANON.md
version: 1.4
status: CANONICAL_TEMPLATE
generated_on: 2026-01-03
scope: Governor workflow template for generating all remaining per-task prompts

## 0) Authority
Subordinate to:
- PROMPT_POLICY_CANON.md
- PROMPT_GENERATOR_GUIDE_FINAL.md

## 1) Role
You are the Governor. Generate one canonical per-task prompt for every unfinished roadmap item.

## 2) Inputs
- Roadmap (TODO-only, numeric phases)
- PROMPT_POLICY_CANON.md
- PROMPT_GENERATOR_GUIDE_FINAL.md
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
   h) If scripts/lint-prompt.sh exists:
      - run it on the prompt
      - exit 1 blocks pack generation
      - exit 2 records a warning and continues
   i) Write prompt file.
4) Write pack manifest and report.

## 5) STOP behavior
STOP is per-task only. Continue generating prompts for unrelated tasks.
