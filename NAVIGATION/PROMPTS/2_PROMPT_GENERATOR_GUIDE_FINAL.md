---
title: 2_PROMPT_GENERATOR_GUIDE_FINAL
version: 1.5
status: CANONICAL
generated_on: 2026-01-06
scope: Procedures + templates for generating per-task prompts (subordinate to PROMPT_POLICY_CANON)
---
<!-- CANON_HASH (sha256 over file content excluding this line): 62676E87CF498A4FB885AEEF67773DA3A6E444C84C1CA9EDD5BF74D08117F457 -->

## 0) Authority
Subordinate to NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md.

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

Header and routing
- Header contains policy_canon_sha256 and guide_canon_sha256
- Numeric phase and task_id only
- primary_model and fallback_chain are explicit
- receipt_path and report_path are explicit and in header

Scope
- Explicit write allowlist, no placeholders
- Allowed deletes/renames is explicit (or N/A)
- Forbidden writes is explicit (“everything else”)

Required facts and STOP behavior
- REQUIRED FACTS are verifiable or unknowns are classified (BLOCKER vs DEFERRABLE)
- DEFERRABLE facts use FILL_ME__ tokens and are recorded as warnings in the manifest
- STOP conditions are explicit for scope, unknowns, validation failures

Workspace preflight (must exist in REQUIRED FACTS or PLAN step 0)
- Verify workspace is on a named branch (not detached HEAD):
  - `git symbolic-ref -q HEAD` must exit 0
- Verify workspace starts clean:
  - `git status --porcelain` must be empty
- If prompt does not explicitly allow main workspace, it must require an isolated workspace (worktree or clone)

Validation semantics
- VALIDATION is defined (or valid N/A case)
- Test semantics invariant is respected:
  - if violations are detected, tests MUST fail
  - scanner tests are forbidden

Coverage accounting (only when relevant)
- If prompt requires “100%” or “no remaining”, it defines:
  - denominator (explicit list)
  - numerator (computed)
  - percentage (derived)
  - evidence (query or referenced artifact)

Artifacts
- ARTIFACTS includes receipt and report paths and matches header paths
- No forbidden inference terms (use hex-escaped regex as in policy)

If CAPABILITY/TOOLS/linters/lint_prompt_pack.sh exists, the Governor MUST run it via `bash` on each generated prompt (requires bash-compatible shell, e.g. WSL).

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
- Fact: Workspace is safe to execute in (branch + clean-state)
  - Verify via:
    - `git symbolic-ref -q HEAD` (must exit 0)
    - `git status --porcelain` (must be empty)
  - If not satisfied:
    - If prompt explicitly allows main workspace: STOP and require user to resolve dirty state
    - Otherwise: STOP and require creating an isolated workspace (worktree or clone)

- Fact: <needed>
  - Verify via: <command> and <expected signal>
  - If unknown:
    - UNKNOWN_BLOCKER -> STOP for this task
    - UNKNOWN_DEFERRABLE -> use FILL_ME__<KEY> and record warning in manifest

PLAN (EXPLICIT STEPS)
0) Preflight workspace checks (branch + clean-state). STOP if failing.
1) ...
2) ...
3) ...

VALIDATION (DONE = GREEN)
- Commands:
  - <command>
- Pass criteria:
  - <specific>
- Test semantics:
  - any detected violation MUST fail the test
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

## 6) Dependency tracking
- Use depends_on in the header when a task requires outputs from another task.
- Governor records mapping task_id -> receipt_path in the pack manifest.
- Executor resolves dependency receipts via the manifest and requires result OK before proceeding.
