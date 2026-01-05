---
title: PROMPT_POLICY_CANON
version: 1.4
status: CANONICAL
generated_on: 2026-01-04
scope: Prompt generation + execution governance for AGS
---
<!-- CANON_HASH: 58360609880fa536cf90e9e7f08a892de7f38edf360831a0edf8458026835ed2 -->

## 0) Authority and precedence
This file is normative.

Precedence (highest to lowest):
1) NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md
2) NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md
3) NAVIGATION/PROMPTS/3_MASTER_PROMPT_TEMPLATE_CANON.md
4) Per-task prompts under PROMPTS/

If conflict exists, higher precedence wins.

## 1) Core constraints
### 1.1 No guessing
Agents MUST NOT guess.

If a required fact is unknown, the agent MUST:
- verify it deterministically (repo paths, commands, tests), or
- STOP per-task and request missing input as a bounded list.

Unknowns in one task MUST NOT block prompt generation for unrelated tasks.

### 1.2 Unknown classes (two-tier)
Unknowns MUST be classified per-task:

- UNKNOWN_BLOCKER
  - Prompt cannot be produced without the fact.
  - Action: STOP for that task. Record in the pack manifest and governor report.

- UNKNOWN_DEFERRABLE
  - Prompt can be produced, but execution cannot proceed until resolved.
  - Action: Generate with placeholder tokens:
    - FILL_ME__<KEY>
  - Record as warnings in the pack manifest.
  - Execution rule: If any FILL_ME__ token remains at run time, STOP with result BLOCKED_UNKNOWN.

### 1.3 FILL_ME__ lifecycle (mechanical)
- Executor MUST scan the prompt text preflight for any FILL_ME__ tokens.
- If any remain unresolved, executor MUST STOP with BLOCKED_UNKNOWN.
- If tokens are resolved before execution, the resolved values MUST satisfy the same Verify via checks defined for that fact.

### 1.4 Explicit procedural steps default
Prompts MUST be explicit and procedural (about 99.99% of the time).
Especially for executor-class models.

### 1.5 Authority Asymmetry (Hierarchy)
### 1.5.1 Planning Authority Rule
Models designated as planner-capable MAY:
- perform planning, analysis, and decomposition
- navigate the repository
- read and write files within existing allowlists
- generate patches and diffs
- invoke tools, linters, and verification scripts
Planner-capable status imposes no artificial execution restriction.

### 1.5.2 Execution Restriction Rule (Non-Planner Models)
Models below the planner-capable tier MUST NOT:
- introduce new structure, policy, or scope
- plan or re-plan tasks
- infer intent or architectural direction
Such models MAY execute mechanical tasks ONLY IF:
- an explicit plan identifier or plan artifact is provided
- the plan was produced by a planner-capable model
Absence of a valid plan reference is a hard failure (fail-closed).


### 1.6 Scope control
Every prompt MUST declare:
- explicit write allowlist (files and or directories)
- explicit allowed renames/deletes (only if needed)
- everything else is write-forbidden

Write allowlists MUST be real allowlists.
Placeholders like “derive from git diff” are forbidden.

### 1.7 Determinism
Where ordering matters, prompts MUST require explicit sort keys.
Outputs MUST be stable for identical inputs and model outputs.

### 1.8 Truth and verification
Done = green.
A task is incomplete until all required validations pass.

Every run MUST emit:
- validation results
- receipt (machine-readable)
- report (human-readable)

### 1.9 Commit ceremony
Agents MUST NOT commit without explicit user approval for that specific commit.
Agents MUST NOT push without explicit user approval for that specific push.

## 2) Model routing contract
Each prompt MUST declare:
- primary_model
- fallback_chain (explicit order)

Routing principles:
- Engineering assurance first, then efficiency.
- Cheapest reliable model first.
- Escalation only through the declared fallback chain.

Tiny models are experimental until proven on fixtures.

Delegation and swarm are allowed only when:
- the prompt explicitly opts in
- subtask is fully specified
- validation is deterministic
- writes are bounded to an allowlist

## 3) Prompt format (mandatory)
Every per-task prompt MUST contain these sections, in this order:

1) HEADER
2) ROLE + MODEL
3) GOAL
4) SCOPE (WRITE ALLOWLIST)
5) REQUIRED FACTS (VERIFY, DO NOT GUESS)
6) PLAN (EXPLICIT STEPS)
7) VALIDATION (DONE = GREEN)
8) ARTIFACTS (RECEIPT + REPORT)
9) EXIT CRITERIA (checkboxes)

N/A rules:
- REQUIRED FACTS may be N/A only for pure documentation tasks with zero external dependencies.
- VALIDATION may be N/A only when SCOPE is read-only analysis.
- ARTIFACTS are never N/A.

## 4) Header schema (required fields)
Every per-task prompt header MUST include:

- phase: <N>
- task_id: <N.M or N.M.K>
- slug: <kebab-case>
- policy_canon_sha256: <sha256 of NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md>
- guide_canon_sha256: <sha256 of NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md>
- depends_on: [] (or list of task_ids)
- primary_model: <string>
- fallback_chain: [<string>, ...]
- receipt_path: <path to receipt JSON for this task>
- report_path: <path to report markdown for this task>
- max_report_lines: N/A (or value, optional)

## 5) Receipt schema (required fields)
Receipts MUST be machine-readable JSON.

Required fields:
- task_id: string
- timestamp_utc: string
- primary_model: string
- fallback_chain: list[string]
- policy_canon_sha256: string
- guide_canon_sha256: string
- depends_on: list[string]
- receipt_path: string
- report_path: string
- allowed_writes: list[string]
- allowed_deletes_renames: list[string]
- unknowns_or_missing_inputs: list[string]
- commands_run: list[string]
- validations: list[{"name": string, "command": string, "exit_code": int}]
- inputs: list[{"path": string, "sha256": string}]
- outputs: list[{"path": string, "sha256": string}]
- result: string (OK | VERIFICATION_FAILED | BLOCKED_UNKNOWN | INTERNAL_ERROR | POLICY_BREACH)
- plan_ref: string (required for non-planner models; matches plan_id or artifact path)

Optional fields:
- compliance_check_command: string
- delegation_digest: string

Minimal schema sketch:
```json
{
  "task_id": "str",
  "policy_canon_sha256": "str",
  "unknowns_or_missing_inputs": ["str"],
  "validations": [{"name":"str","command":"str","exit_code":0}]
}
```

## 6) Dependency enforcement (no fixed directory)
- depends_on lists task IDs.
- The pack manifest MUST provide a mapping task_id -> receipt_path.
- Executor MUST resolve dependency receipt paths using the manifest mapping.
- Executor MUST verify dependency receipts exist and have result OK before running the dependent task.

## 7) Prompt linting and blocking behavior
If CAPABILITY/TOOLS/linters/lint_prompt_pack.sh exists (requires bash-compatible shell, e.g. WSL):
- Governor MUST run it on each generated prompt.
- Exit code 1 MUST block pack generation.
- Exit code 2 MUST be recorded as a warning and pack generation may continue.

## 8) Version skew detection
Prompts, receipts, and the pack manifest MUST carry:
- policy_canon_sha256
- guide_canon_sha256

Executors MUST refuse to run prompts whose hashes do not match current canon.

## 9) Naming and numbering (no letter lanes)
Primary identifiers are numeric:
- Phase N
- Task N.M
- Subtask N.M.K

Legacy IDs may exist but MUST NOT be primary identifiers.

Prompt file naming MUST be deterministic:
- PROMPTS/phase-<N>/task-<N>.<M>.<K>_<slug>.md

## 10) STOP conditions (execution-time)
Agents MUST STOP if:
- a required fact cannot be verified deterministically
- any FILL_ME__ token remains unresolved
- the task would exceed the scope allowlist
- validation cannot be performed
- changes would touch CANON without explicit authorization

## 11) Forbidden inference terms (lint reference)
Forbidden terms include the 6-letter inference verb and its noun variants.
Lint SHOULD detect them using hex-escaped regex, for example:
- \b\x61\x73\x73\x75\x6d\x65\b
- \b\x61\x73\x73\x75\x6d\x70\x74\x69\x6f\x6e(s)?\b

## 12) Lint Gate
A prompt pack is invalid for execution unless it passes the canonical linter.

Executors MUST run the canonical lint command before any execution.
The canonical lint command is: `bash CAPABILITY/TOOLS/linters/lint_prompt_pack.sh PROMPTS_DIR` (requires bash-compatible shell, e.g. WSL)

Lint failure or inability to lint is a hard stop: no model execution, no writes.
Lint warnings (exit code 2) are non-blocking but MUST be recorded in receipts.

Executors MUST record the following lint metadata in receipts/run artifacts:
- lint_command: the exact command run
- lint_exit_code: exit status (0=PASS, 1=FAIL, 2=WARNING)
- lint_result: PASS, FAIL, or WARNING
- optional: linter_ref (path/version/hash) if available

## 13) Authority Enforcement
Executors MUST gate non-planner models on presence of a valid plan reference.
Violations MUST be recorded as policy breaches in run receipts.
Routing MUST escalate to a planner-capable model before execution if no plan exists.
