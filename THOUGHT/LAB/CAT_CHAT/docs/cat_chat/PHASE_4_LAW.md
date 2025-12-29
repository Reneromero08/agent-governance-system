# Phase 4: Deterministic Planner + Governed Step Pipeline — Law Document

**Status:** COMPLETE
**Date:** 2025-12-29
**Phase:** 4
**Nature:** Deterministic Compilation from Intent to Steps

---

## Purpose

The Planner is a **deterministic compiler** that transforms a high-level "request message" into a precise sequence of executable steps. It operates **on top of** the Phase 3 Message Cassette, using it to store both the request and the generated plan. The Planner does NOT execute code or call models—it only computes what should be done.

---

## What the Planner IS

### 1. A Deterministic Compiler

The planner takes:
- A **request message** (intent + inputs + budgets)
- A **canonical substrate** (sections from Phase 1, symbols from Phase 2)

And produces:
- A **deterministic plan** (ordered list of steps)
- Each step has: step_id, ordinal, operation type, refs, constraints, expected outputs

### 2. Execution-Agnostic

The planner does NOT:
- Execute code
- Run tests
- Call LLMs or models
- Modify files or run arbitrary commands
- Launch workers or background daemons

It only **computes** a plan for others to execute.

### 3. Governed by Budgets

The planner enforces hard limits:
- **max_steps**: Maximum number of steps in a plan
- **max_bytes**: Maximum total bytes to be referenced (files + symbol expansions)
- **max_symbols**: Maximum number of distinct @Symbols that may be referenced

Exceeding any budget causes **fail-closed** rejection.

### 4. Deterministic Behavior

Given the same:
- Request (same request_id)
- Substrate state (same SECTION_INDEX hash)
- Input definitions

The planner produces:
- **Identical step_ids** (via SHA256 of canonical JSON)
- **Identical step ordering** (stable sort by dependency)
- **Identical plan_hash** (SHA256 of canonical plan representation)

No randomness. No "pick any available" heuristics.

---

## What the Planner is NOT

### 1. NOT an Executor

The planner does NOT:
- Run the steps it generates
- Validate that steps succeed before proceeding
- Retry failed steps automatically
- Manage worker pools or task distribution

### 2. NOT an Orchestrator

The planner does NOT:
- Schedule work across multiple workers
- Implement backpressure or rate limiting
- Handle parallel execution or dependencies
- Track worker health or availability

### 3. NOT Phase 5+

The planner does NOT:
- Create executable bundles (that's Phase 5)
- Implement memoization across steps (that's Phase 5)
- Provide translation protocols (that's Phase 5)
- Handle file patching (that's Phase 5+)

### 4. NOT Autonomy

The planner does NOT:
- Make decisions about "how" to accomplish intent beyond what the user specified
- Substitute alternative approaches when a step fails
- Add "cleanup" or "recovery" steps automatically

---

## Entities

### Plan Request (`plan_request.schema.json`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| run_id | string | Yes | Run context for grouping |
| request_id | string | Yes | Unique identifier for this plan request |
| intent | string | Yes | High-level intent (e.g., "debug_function", "implement_feature") |
| inputs | object | No | Input references (symbols, files, notes) |
| inputs.symbols | string[] | No | List of @Symbol references to resolve |
| inputs.files | string[] | No | List of file references to read |
| inputs.notes | string[] | No | Advisory notes or constraints |
| budgets | object | Yes | Budget constraints for the plan |
| budgets.max_steps | integer | Yes | Maximum number of steps allowed |
| budgets.max_bytes | integer | No | Maximum total bytes to be referenced |
| budgets.max_symbols | integer | No | Maximum distinct @Symbols to reference |

### Plan Step (`plan_step.schema.json`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| step_id | string | Yes | Unique step identifier (SHA256 of canonical JSON) |
| ordinal | integer | Yes | Step order within the plan (1-based) |
| op | string | Yes | Operation type: READ_SECTION, READ_SYMBOL, WRITE_FILE, PATCH_FILE, RUN_TEST, NOTE |
| refs | object | No | References to external resources (depends on op) |
| refs.section_id | string | No | Section ID to read (for READ_SECTION) |
| refs.symbol_id | string | No | @Symbol to resolve (for READ_SYMBOL) |
| refs.file_path | string | No | File path to write/patch (for WRITE_FILE, PATCH_FILE) |
| refs.file_hash | string | No | SHA256 of file content (for patch verification) |
| refs.content | string | No | Content to write (for WRITE_FILE) |
| refs.patch_start_line | integer | No | Line number to start patching (for PATCH_FILE) |
| refs.patch_end_line | integer | No | Line number to end patching (for PATCH_FILE) |
| refs.test_name | string | No | Test name to run (for RUN_TEST) |
| refs.message | string | No | Note message (for NOTE) |
| constraints | object | No | Constraints on this step |
| constraints.timeout_seconds | integer | No | Maximum seconds for this step |
| constraints.required_outputs | string[] | No | Required outputs for this step |
| expected_outputs | object | No | Expected outputs after this step |

### Plan Output (`plan_output.schema.json`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| run_id | string | Yes | Run context for this plan |
| request_id | string | Yes | Request ID that generated this plan |
| planner_version | string | Yes | Version of planner that generated this plan |
| steps | array | Yes | Ordered list of steps in the plan |
| plan_hash | string | Yes | SHA256 hash of canonical plan representation |

---

## Operation Types

### READ_SECTION

Read a section from the Phase 1 substrate.

**Required refs:**
- `section_id` (from Phase 1 SECTION_INDEX)

**Purpose:** Worker needs section content to proceed.

### READ_SYMBOL

Resolve an @Symbol and return its bounded expansion.

**Required refs:**
- `symbol_id` (from Phase 2 SYMBOLS)

**Resolution:** Uses Phase 2 SymbolResolver to get bounded expansion.

**Purpose:** Worker needs symbol content to proceed.

### WRITE_FILE

Write content to a file.

**Required refs:**
- `file_path` (path to write)
- `content` (content to write)

**Purpose:** Worker writes a new file or updates an existing one.

### PATCH_FILE

Apply a patch to a file.

**Required refs:**
- `file_path` (path to patch)
- `patch_start_line` (first line to replace)
- `patch_end_line` (last line to replace)
- `content` (new content for lines)

**Purpose:** Worker modifies specific lines of a file.

### RUN_TEST

Execute a test.

**Required refs:**
- `test_name` (name of test to run)

**Purpose:** Worker runs a test to verify changes.

### NOTE

Add a note or annotation.

**Required refs:**
- `message` (note message)

**Purpose:** Non-executable step for documentation or checkpointing.

---

## Determinism Rules

### 1. Stable Step IDs

Each `step_id` is computed as:
```python
step_id = f"step_{hashlib.sha256(canonical_json_step).hexdigest()[:16]}"
```

The `canonical_json_step` includes:
- All required fields (step_id, ordinal, op)
- All refs (canonical JSON sort)
- All constraints (canonical JSON sort)

Same canonical JSON produces same step_id.

### 2. Stable Step Ordering

Steps are ordered by:
1. **Dependencies**: READ_SECTION/READ_SYMBOL before steps that depend on them
2. **Ordinal**: Maintained 1-N ordering from planner
3. **Canonical JSON**: Steps sorted by canonical representation to break ties

### 3. Stable Plan Hash

The `plan_hash` is computed as:
```python
canonical_plan = json.dumps({
    "run_id": run_id,
    "request_id": request_id,
    "steps": [canonical_json_step for step in sorted_steps]
}, sort_keys=True)
plan_hash = hashlib.sha256(canonical_plan.encode()).hexdigest()
```

Same plan produces same plan_hash.

### 4. Symbol Resolution Determinism

When `READ_SYMBOL` references an @Symbol:
1. Use Phase 2 SymbolResolver to get `section_id` and `default_slice`
2. Use Phase 2 SectionIndexer to get section content
3. Compute bounded expansion using default_slice
4. No randomness, no "pick any available"

---

## Budget Enforcement

### max_steps

- Maximum number of steps allowed in the plan
- If planner would generate more than `max_steps`, fail-closed
- Default if not specified: 100

### max_bytes

- Maximum total bytes referenced across all steps
- Computed as: `sum(file_sizes) + sum(symbol_expansion_sizes)`
- Symbol expansion size: length of bounded expansion (default_slice applied)
- If total would exceed `max_bytes`, fail-closed
- Default if not specified: 10,000,000 (10MB)

### max_symbols

- Maximum number of distinct @Symbols referenced across all steps
- If plan would reference more than `max_symbols`, fail-closed
- Default if not specified: 100

### Failure Mode: Fail-Closed

Any budget violation results in:
- Immediate `PlannerError` with clear message
- No partial plan generation
- No database writes

---

## Symbol Bounds Enforcement

### slice=ALL is Forbidden

Even if user requests `slice=ALL`, the planner MUST:
- Use the symbol's `default_slice` instead
- Reject the step if `default_slice` is not defined
- Never generate a `READ_SYMBOL` op with `slice=ALL`

### Reason

Unbounded expansion would break determinism guarantees because:
- Section content changes over time
- Different workers get different expansions
- Plan depends on external state (current substrate)

Using `default_slice` ensures:
- All workers see identical content
- Plan is reproducible
- Determinism holds

---

## API Reference

### `Planner` Class

#### `__init__(repo_root=None, substrate_db=None, symbols_db=None)`

Initialize planner. Uses default paths if not specified.

#### `plan_request(request: dict, *, repo_root: str, substrate: str) -> dict`

Plan a request and return plan_output.

**Parameters:**
- `request`: Full plan request (matches `plan_request.schema.json`)
- `repo_root`: Path to repository root
- `substrate`: Substrate type ("sqlite" or "jsonl")

**Returns:**
- `dict` matching `plan_output.schema.json`

**Raises:**
- `PlannerError` on validation failure, budget violation, missing references
- `SymbolError` on symbol resolution failure
- `SectionError` on section retrieval failure

---

## Cassette Integration

### `post_request_and_plan(run_id, request_payload, idempotency_key=None) -> (message_id, job_id)`

Convenience function that:
1. Stores the request as a cassette message (Phase 3)
2. Runs the planner
3. Stores the plan as a cassette job
4. Creates one cassette step per plan step (all PENDING)

**Behavior:**
- Request is stored as-is in `cassette_messages`
- Planner generates deterministic step list
- Job and steps are created in `cassette_jobs` / `cassette_steps`
- All steps start in `PENDING` status
- Workers can claim steps via Phase 3 API

**Idempotency:**
- Same `(run_id, idempotency_key)` returns same job_id and step_ids
- Prevents duplicate plans

---

## CLI Reference

### Plan Command

```
cortex plan --request-file <json> [--dry-run] [--repo-root <path>] [--substrate <type>]
```

Plan a request from a JSON file.

- `--request-file`: Path to plan request JSON (required)
- `--dry-run`: Print plan_output to stdout, no DB writes
- `--repo-root`: Repository root (default: cwd)
- `--substrate`: Substrate type (default: "sqlite")

**Behavior:**
- If dry-run: Print plan_output JSON to stdout
- If not dry-run: Call `post_request_and_plan()` and print job_id + step_ids
- Returns 0 on success, 1 on failure

### Plan Verify Command

```
cassette plan-verify --run-id <run> --request-id <id>
```

Verify that a stored plan hash matches recomputed plan.

- `--run-id`: Run context
- `--request-id`: Request ID to verify
- Retrieves stored plan from cassette
- Recomputes plan from stored request
- Compares plan_hash
- Returns 0 if match, 1 if mismatch

---

## Testing Requirements

### Mandatory Tests (all passing)

1. **Determinism Tests**
   - `test_plan_determinism_same_request_same_output`
     - Same request + same substrate = identical plan (step_ids, ordering, plan_hash)
   - `test_plan_determinism_step_ids_stable`
     - Same canonical JSON produces same step_id

2. **Budget Enforcement Tests**
   - `test_plan_rejects_too_many_steps`
     - Request exceeds max_steps = fail-closed
   - `test_plan_rejects_too_many_bytes`
     - Request exceeds max_bytes = fail-closed
   - `test_plan_rejects_too_many_symbols`
     - Request exceeds max_symbols = fail-closed

3. **Symbol Bounds Tests**
   - `test_plan_rejects_slice_all_forbidden`
     - slice=ALL in request causes rejection

4. **Idempotency Tests**
   - `test_plan_idempotency_same_idempotency_key`
     - Same (run_id, idempotency_key) returns same job_id/steps

5. **Dry-Run Tests**
   - `test_plan_dry_run_does_not_touch_db`
     - --dry-run produces plan without DB writes

6. **Plan Verify Tests**
   - `test_plan_verify_matches_stored_hash`
     - Stored plan_hash matches recomputed plan
   - `test_plan_verify_fails_on_mismatch`
     - Mismatched plan causes failure

---

## Database Location

**Request Storage:** `CORTEX/_generated/system3.db` (Phase 3 cassette)

The planner uses Phase 3 Message Cassette to store:
- Request message (in `cassette_messages`)
- Plan as job (in `cassette_jobs`)
- Steps (in `cassette_steps`)

---

## Versioning

- **Planner Version:** 1.0
- **Schema Versions:**
  - `plan_request.schema.json`: 1.0
  - `plan_step.schema.json`: 1.0
  - `plan_output.schema.json`: 1.0
- **Phase Status:** COMPLETE

Phase 4 is frozen. No breaking changes without version bump.

---

## Non-Negotiables

These rules MAY NOT be broken:

1. **Deterministic only** — No randomness, no heuristics, no "pick any available"
2. **Budget enforcement** — Hard limits, fail-closed on violation
3. **Symbol bounds** — slice=ALL forbidden, use default_slice
4. **No execution** — Planner does NOT run steps or execute code
5. **No external calls** — No network, no model calls, no workers
6. **Frozen after completion** — No changes without version bump

---

## Verification

All Phase 4 invariants are verified by tests:

```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_planner.py
python -m catalytic_chat.cli plan --request-file tests/fixtures/plan_request_min.json --dry-run
```

Expected output:
- `pytest`: All tests pass
- `plan`: Valid plan_output JSON printed
