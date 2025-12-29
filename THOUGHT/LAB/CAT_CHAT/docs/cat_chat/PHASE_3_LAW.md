# Phase 3: Message Cassette — Law Document

**Status:** COMPLETE
**Date:** 2025-12-29
**Phase:** 3
**Nature:** Execution-Agnostic Ledger (DB-First)

---

## Purpose

The Message Cassette is an **authoritative, append-only execution ledger** for Catalytic Chat. It stores the immutable record of work claims, receipts, and job state transitions. It enforces invariants at the database level via SQLite triggers, ensuring that even application bugs cannot corrupt the ledger.

---

## What the Message Cassette IS

### 1. An Execution Ledger

The cassette records:
- **Messages**: Intent signals posted by USER, PLANNER, SYSTEM, or WORKER
- **Jobs**: Execution units derived from messages
- **Steps**: Claimable units within jobs
- **Receipts**: Immutable proof of step completion with outcomes

### 2. Append-Only Immutable Store

Once written, data never changes:
- `cassette_messages`: No UPDATE, no DELETE
- `cassette_receipts`: No UPDATE, no DELETE
- `cassette_jobs`: Immutable after creation
- `cassette_steps`: Mutate ONLY via allowed FSM transitions

This immutability is enforced by SQLite triggers, not by convention.

### 3. Deterministic Behavior

- Claims are always deterministic: oldest job, lowest ordinal
- No randomness, no "pick any available"
- Identical inputs produce identical outputs

---

## What the Message Cassette is NOT

### 1. NOT an Execution Engine

The cassette does NOT:
- Execute code
- Call models or AI systems
- Spawn workers or processes
- Orchestrate or schedule work
- Implement backpressure or queuing policies
- Depend on external runtimes (Ollama, LLMs, etc.)

### 2. NOT Phase 4+

The cassette does NOT implement:
- Schedulers or task distribution
- Retry logic or exponential backoff
- Work-stealing or work balancing
- Vector similarity or discovery
- Translation protocols

### 3. NOT a Worker Pool

The cassette does NOT:
- Maintain worker connections
- Track worker health or availability
- Route work to specific workers
- Manage worker lifecycles

---

## Entities

### Message (`cassette_messages`)

| Field | Type | Constraint | Purpose |
|-------|------|------------|---------|
| message_id | TEXT | PRIMARY KEY | Unique message identifier |
| run_id | TEXT | NOT NULL | Run context for grouping |
| source | TEXT | NOT NULL, CHECK | USER, PLANNER, SYSTEM, or WORKER |
| idempotency_key | TEXT | UNIQUE(run_id, idempotency_key) | Prevent duplicate messages |
| payload_json | TEXT | NOT NULL | Message payload (JSON) |
| created_at | TEXT | NOT NULL, DEFAULT | ISO-8601 timestamp |

**Invariants:**
- Append-only (UPDATE/DELETE blocked by triggers)
- One unique (run_id, idempotency_key) pair

### Job (`cassette_jobs`)

| Field | Type | Constraint | Purpose |
|-------|------|------------|---------|
| job_id | TEXT | PRIMARY KEY | Unique job identifier |
| message_id | TEXT | NOT NULL, FK | Parent message |
| intent | TEXT | NOT NULL | Job intent (from payload) |
| ordinal | INTEGER | NOT NULL | Order within message |
| created_at | TEXT | NOT NULL, DEFAULT | ISO-8601 timestamp |

**Invariants:**
- Immutable after creation (no UPDATE/DELETE by API)
- References valid message_id (FK enforced)
- Ordinal is positive

### Step (`cassette_steps`)

| Field | Type | Constraint | Purpose |
|-------|------|------------|---------|
| step_id | TEXT | PRIMARY KEY | Unique step identifier |
| job_id | TEXT | NOT NULL, FK | Parent job |
| ordinal | INTEGER | NOT NULL | Order within job |
| status | TEXT | NOT NULL, CHECK | PENDING, LEASED, or COMMITTED |
| lease_owner | TEXT | NULLABLE | Worker who holds lease |
| lease_expires_at | TEXT | NULLABLE | ISO-8601 timestamp when lease expires |
| fencing_token | INTEGER | NOT NULL, DEFAULT 0 | Monotonic claim counter |
| payload_json | TEXT | NOT NULL | Step payload (JSON) |
| created_at | TEXT | NOT NULL, DEFAULT | ISO-8601 timestamp |

**Invariants:**
- Status transitions only via allowed FSM edges
- Lease fields can ONLY be set via PENDING→LEASED transition
- Lease expiration enforced
- Fencing token increments on each claim

### Receipt (`cassette_receipts`)

| Field | Type | Constraint | Purpose |
|-------|------|------------|---------|
| receipt_id | TEXT | PRIMARY KEY | Unique receipt identifier |
| step_id | TEXT | NOT NULL, FK | Completed step |
| job_id | TEXT | NOT NULL, FK | Parent job |
| worker_id | TEXT | NOT NULL | Worker who completed step |
| fencing_token | INTEGER | NOT NULL | Token from claim |
| outcome | TEXT | NOT NULL, CHECK | SUCCESS, FAILURE, or ABORTED |
| receipt_json | TEXT | NOT NULL | Receipt payload (JSON) |
| created_at | TEXT | NOT NULL, DEFAULT | ISO-8601 timestamp |

**Invariants:**
- Append-only (UPDATE/DELETE blocked by triggers)
- References valid step_id and job_id (FK enforced)
- One receipt per completed step (enforced by FSM)

---

## Allowed State Transitions (FSM)

### Step Status FSM

```
PENDING ──[claim_step]──> LEASED ──[complete_step]──> COMMITTED
  │                                       │
  └──────────[requeue]─────────────┘
```

**Allowed Transitions:**
- `PENDING → LEASED`: Via `claim_step()` when worker claims
- `LEASED → COMMITTED`: Via `complete_step()` when worker completes
- `LEASED → PENDING`: Via explicit requeue (not yet implemented in Phase 3)

**Forbidden Transitions (blocked by triggers):**
- `PENDING → COMMITTED`: Skips leasing
- `COMMITTED → LEASED`: Cannot re-lease completed step
- `COMMITTED → PENDING`: Cannot rollback completed step
- `LEASED → LEASED`: Duplicate lease

---

## Lease Enforcement

### Lease Assignment (`claim_step`)

A step can only be leased if:
1. Current status is `PENDING`
2. `lease_expires_at` is set to a future timestamp
3. `fencing_token` is incremented from previous value
4. No other worker holds the lease

### Lease Verification (`complete_step`)

Completion fails if:
1. Step is not currently `LEASED`
2. `lease_owner` does not match `worker_id`
3. `fencing_token` does not match the stored value
4. `lease_expires_at` has passed (expired lease)

### Lease Expiration

When a lease expires:
- The step remains in `LEASED` status
- `complete_step()` will fail with "lease expired"
- Requeuing is NOT automatic (must be explicit)

---

## Deterministic Claim Selection

When `claim_step(run_id, worker_id)` is called:

1. Select ALL pending steps for the run:
   ```sql
   SELECT s.step_id, s.job_id, j.message_id, s.ordinal, s.fencing_token
   FROM cassette_steps s
   JOIN cassette_jobs j ON s.job_id = j.job_id
   JOIN cassette_messages m ON j.message_id = m.message_id
   WHERE s.status = 'PENDING' AND m.run_id = ?
   ORDER BY m.created_at ASC, j.ordinal ASC, s.ordinal ASC
   ```

2. Claim the first result (LIMIT 1)
3. Set lease_owner, lease_expires_at, fencing_token
4. Update status to `LEASED`

**No randomness is involved.** Identical states produce identical behavior.

---

## Authority

### Database-Level Authority

All invariants are enforced by the SQLite database itself:

1. **Append-only triggers**:
   - `tr_messages_append_only_update`
   - `tr_messages_append_only_delete`
   - `tr_receipts_append_only_update`
   - `tr_receipts_append_only_delete`

2. **FSM enforcement triggers**:
   - `tr_steps_fsm_illegal_1`: Blocks PENDING→COMMITTED
   - `tr_steps_fsm_illegal_2`: Blocks LEASED→PENDING
   - `tr_steps_fsm_illegal_3`: Blocks COMMITTED→LEASED

3. **Lease protection trigger**:
   - `tr_steps_lease_prevent_direct_set`: Blocks direct lease field changes

### API Layer Authority

The Python API (`message_cassette.py`) is a **thin layer** that:
- Constructs SQL statements
- Relies on DB triggers to reject violations
- Raises `MessageCassetteError` immediately on any failure
- Never duplicates DB-level logic in application code

---

## API Reference

### `MessageCassette` Class

#### `__init__(repo_root=None, db_path=None)`

Initialize the cassette. Uses `CORTEX/_generated/system3.db` by default.

#### `post_message(payload, run_id, source, idempotency_key=None) -> (message_id, job_id)`

Post a message to the cassette, creating a job and initial step.

- **Raises**: `MessageCassetteError` on validation failure
- **Idempotent**: Same `(run_id, idempotency_key)` returns same IDs

#### `claim_step(run_id, worker_id, ttl_seconds=300) -> dict`

Claim a pending step for execution.

- **Returns**: Dict with step_id, job_id, message_id, ordinal, payload, fencing_token, lease_expires_at
- **Raises**: `MessageCassetteError` if no pending steps
- **Deterministic**: Always selects oldest job, lowest ordinal

#### `complete_step(run_id, step_id, worker_id, fencing_token, receipt_payload, outcome) -> receipt_id`

Complete a step with a receipt.

- **Raises**: `MessageCassetteError` on:
  - Invalid outcome
  - Step not found
  - Wrong run_id
  - Step not leased
  - Wrong worker
  - Stale token
  - Expired lease

#### `verify_cassette(run_id=None) -> None`

Verify cassette integrity and invariant enforcement.

- **Checks**:
  - PRAGMA foreign_keys is ON
  - All required tables exist
  - All required triggers exist
  - No expired leases
- **Raises**: `MessageCassetteError` on verification failure
- **Output**: Prints `PASS: All invariants verified` or `FAIL: N issue(s) found` to stderr

---

## CLI Reference

```
cortex cassette verify --run-id <id>
```
Verify cassette integrity. Returns 0 on pass, 1 on fail.

```
cortex cassette post --json <file> --run-id <id> --source <src> [--idempotency-key <k>]
```
Post a message from a JSON file. Source must be USER, PLANNER, SYSTEM, or WORKER.

```
cortex cassette claim --run-id <id> --worker <id> [--ttl <seconds>]
```
Claim a pending step. TTL defaults to 300 seconds.

```
cortex cassette complete --run-id <id> --step <id> --worker <id> --token <n> --receipt <file> --outcome <out>
```
Complete a step. Outcome must be SUCCESS, FAILURE, or ABORTED.

---

## Testing Requirements

### Mandatory Tests (all passing)

1. `test_messages_append_only_trigger_blocks_update_delete`
   - Proves UPDATE/DELETE on messages blocked by triggers

2. `test_receipts_append_only_trigger_blocks_update_delete`
   - Proves UPDATE/DELETE on receipts blocked by triggers

3. `test_fsm_illegal_transition_blocked`
   - Proves illegal FSM transitions blocked by triggers

4. `test_claim_deterministic_order`
   - Proves claim selection is deterministic

5. `test_complete_rejects_stale_token`
   - Proves completion rejects stale tokens

6. `test_complete_rejects_expired_lease`
   - Proves completion rejects expired leases

7. `test_receipt_requires_existing_step_job_fk`
   - Proves referential integrity enforced

### Adversarial Tests

Tests attempt raw SQL bypass to prove DB triggers fire:
- Direct UPDATE on messages (blocked)
- Direct UPDATE on receipts (blocked)
- Direct DELETE on messages (blocked)
- Direct DELETE on receipts (blocked)
- Direct FSM jumps (blocked)
- Direct lease field changes (blocked)

---

## Database Location

**Path:** `CORTEX/_generated/system3.db`

**Rationale:** Separate database for isolation from Phase 1/2 data.

---

## Versioning

- **DB Version:** 1 (stored in `cassette_meta` table)
- **Schema Version:** 1 (triggers enforce invariants)
- **Phase Status:** COMPLETE

Phase 3 is frozen. No schema changes without version bump.

---

## Non-Negotiables

These rules MAY NOT be broken:

1. **SQLite only** — No alternative substrates for Phase 3
2. **Append-only messages/receipts** — Enforced by triggers
3. **Deterministic claims** — No randomness allowed
4. **Fail-closed** — Any violation aborts the operation
5. **No execution** — The cassette never executes code
6. **No Phase 4+ features** — No schedulers, no discovery, no translation
7. **Frozen after completion** — No changes without version bump

---

## Verification

All Phase 3 invariants are verified by tests:

```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_message_cassette.py
python -m catalytic_chat.cli cassette verify --run-id test_run
```

Expected output:
- `pytest`: All tests pass
- `verify`: `PASS: All invariants verified`
