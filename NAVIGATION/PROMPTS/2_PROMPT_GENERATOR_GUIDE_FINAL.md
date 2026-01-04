# PROMPT_GENERATOR_GUIDE_FINAL.md
version: 1.4
status: CANONICAL_PROCEDURE
generated_on: 2026-01-03
scope: Procedures + templates for generating per-task prompts (subordinate to PROMPT_POLICY_CANON)

## 0) Authority
Subordinate to PROMPT_POLICY_CANON.md.

## 1) Operating principle
No guessing. STOP is per-task. Continue generating prompts for unrelated tasks.

## 2) Granularity heuristics (deterministic guidance)
Micro is favored when:
- touches fewer than 3 files, and
- about 100 lines or less, and
- no cross-module type or import dependencies.

Slice is favored when:
- all changes remain inside one bounded package/context, and
- acceptance criteria are verifiable without cross-cutting refactors.

Patchset is favored when:
- correctness depends on synchronized changes across modules, or
- splitting would break type boundaries or required interfaces.

## 3) Prompt QA checklist (generator self-check)
A prompt is invalid unless it passes:
- Header contains policy_canon_sha256 and guide_canon_sha256
- Numeric phase and task_id only
- Explicit write allowlist, no placeholders
- receipt_path and report_path are explicit and in header
- REQUIRED FACTS are verifiable or unknowns are classified (BLOCKER vs DEFERRABLE)
- DEFERRABLE facts have FILL_ME__ tokens and are recorded as warnings in the manifest
- PLAN is explicit and procedural
- VALIDATION is defined (or valid N/A case)
- ARTIFACTS includes receipt and report paths and matches header paths
- No forbidden inference terms (use hex-escaped regex as in policy)

If scripts/lint-prompt.sh exists, the Governor MUST run it on each generated prompt.

## 4) Canonical per-task prompt template
```text
HEADER
- phase: <N>
- task_id: <N.M or N.M.K>
- slug: <kebab-case>
- policy_canon_sha256: <sha256>
- guide_canon_sha256: <sha256>
- depends_on: [] (or list of task_ids)
- primary_model: <string>
- fallback_chain: [<string>, ...]
- receipt_path: <path>
- report_path: <path>
- max_report_lines: N/A (or value, optional)

ROLE + MODEL
- Primary: <MODEL>
- Fallback chain: <MODEL_A> -> <MODEL_B> -> <MODEL_C>

GOAL
<One paragraph tied to Phase/Task ID.>

SCOPE (WRITE ALLOWLIST)
- Allowed writes:
  - <path>
- Allowed deletes/renames:
  - N/A
- Forbidden writes:
  - Everything else.

REQUIRED FACTS (VERIFY, DO NOT GUESS)
- Fact: <needed>
  - Verify via: <command> and <expected signal>
  - If unknown:
    - UNKNOWN_BLOCKER -> STOP for this task
    - UNKNOWN_DEFERRABLE -> use FILL_ME__<KEY> and record warning in manifest

PLAN (EXPLICIT STEPS)
1) ...
2) ...
3) ...

VALIDATION (DONE = GREEN)
- Commands:
  - <command>
- Pass criteria:
  - <specific>
- If failing:
  - Iterate if salvageable, else revert; do not claim success.

ARTIFACTS (RECEIPT + REPORT)
- Receipt path: <path>.json
- Report path: <path>.md

EXIT CRITERIA
- [ ] <measurable condition>
```

## 5) Executor safety and planner-only blocks
Canonical prompts MUST be executor-safe by default.

If planner-only notes are needed, they MUST be wrapped in:
BEGIN_PLANNER_ONLY
...
END_PLANNER_ONLY

Executor MUST ignore content inside that block.
No other transform is required.

## 6) Dependency tracking
- Use depends_on in the header when a task requires outputs from another task.
- Governor records mapping task_id -> receipt_path in the pack manifest.
- Executor resolves dependency receipts via the manifest and requires result OK before proceeding.
