# Session Commit Plan & Change Log

Generated: 2025-12-29

This file tracks all changes made during this session, organized into logical commit chunks.

---

## Commit Chunks

### Chunk 1: CAT_CHAT Info Architecture Refactor

**Files Changed:**
- `docs/catalytic-chat/phases/` (new directory)
- `docs/catalytic-chat/phases/PHASE_1.md` (moved from PHASE1_COMPLETION_REPORT.md)
- `docs/catalytic-chat/phases/PHASE_2_1.md` (moved from PHASE_2.1_COMPLETION_REPORT.md)
- `docs/catalytic-chat/phases/PHASE_2_2.md` (moved from PHASE_2.2_COMPLETION_REPORT.md)
- `docs/catalytic-chat/phases/PHASE_2_2_EXPANSION_CACHE.md` (moved)
- `docs/catalytic-chat/ROADMAP.md` (moved from CAT_CHAT_ROADMAP_V1.md)
- `docs/catalytic-chat/CHANGELOG.md` (moved from CHANGELOG.md)
- `docs/catalytic-chat/notes/` (new directory)
- `docs/catalytic-chat/notes/SYMBOLIC_README.md` (moved)
- `docs/catalytic-chat/notes/REFACTORING_REPORT.md` (moved)
- `docs/catalytic-chat/notes/README.md` (new)
- `legacy/` (new directory)
- `legacy/chat_db.py` (moved)
- `legacy/db_only_chat.py` (moved)
- `legacy/embedding_engine.py` (moved)
- `legacy/message_writer.py` (moved)
- `legacy/direct_vector_writer.py` (moved)
- `legacy/run_swarm_with_chat.py` (moved)
- `legacy/swarm_chat_logger.py` (moved)
- `legacy/simple_symbolic_demo.py` (moved)
- `legacy/example_usage.py` (moved)
- `legacy/tests/` (new directory)
- `legacy/tests/test_chat_system.py` (moved)
- `legacy/tests/test_db_only_chat.py` (moved)
- `legacy/chats/` (moved - preserved data)
- `legacy/symbols/` (moved - preserved data)
- `legacy/README.md` (new)
- `pytest.ini` (updated)
- `ROADMAP.md` (stub pointing to docs)
- `CHANGELOG.md` (stub pointing to docs)
- `README.md` (updated with new paths)
- `catalytic_chat/README.md` (updated references)
- `archive/` (removed - contents moved to docs/notes/)

**Commit Message:**
```
refactor(cat_chat): reorganize docs structure for phase alignment

- Move phase completion reports to docs/catalytic-chat/phases/
- Move ROADMAP and CHANGELOG to docs/catalytic-chat/
- Move notes to docs/catalytic-chat/notes/
- Quarantine legacy scripts in legacy/ with README explaining status
- Update pytest.ini to exclude legacy tests
- Update README.md with new structure
- Remove empty archive/ (contents moved to notes/)

Root now contains only: README.md, ROADMAP.md (stub), 
CHANGELOG.md (stub), catalytic_chat/, docs/, tests/, legacy/

Phase 2 closure and Phase 3 "not started" are now unambiguous.
```

---

### Chunk 2: Phase 2.5 Experimental Vector Sandbox

**Files Changed:**
- `catalytic_chat/experimental/__init__.py` (new)
- `catalytic_chat/experimental/vector_store.py` (new)
- `tests/test_vector_store.py` (new)
- `docs/catalytic-chat/notes/VECTOR_SANDBOX.md` (new)

**Commit Message:**
```
feat(cat_chat): add phase 2.5 experimental vector sandbox

- Add SQLite-backed vector store for local experiments
- Vector tables: vector_id, namespace, content_hash, dims, 
  vector_json, meta_json, created_at
- API: put_vector, get_vector, query_topk
- Cosine similarity in pure Python (no extensions)
- Namespace isolation for multi-tenant experiments
- Tests for put/get, query ordering, dimension validation, 
  namespace isolation
- Documentation in VECTOR_SANDBOX.md

This is NOT part of Phase 2 or Phase 3.
No changes to symbols/resolve/expand Phase 2 behavior.
```

---

### Chunk 3: Phase 3 Message Cassette (DB-first Schema & API)

**Files Changed:**
- `catalytic_chat/message_cassette_db.py` (new)
- `catalytic_chat/message_cassette.py` (new)
- `tests/test_message_cassette.py` (new)

**Commit Message:**
```
feat(cat_chat): implement phase 3 message cassette with db-first enforcement

DB Schema (CORTEX/_generated/system3.db):
- cassette_messages: append-only, run_id, source, payload_json
- cassette_jobs: links to messages, ordinal for ordering
- cassette_steps: status (PENDING/LEASED/COMMITTED), 
  lease_owner, lease_expires_at, fencing_token
- cassette_receipts: append-only, links to step/job

DB-Level Enforcement (triggers):
- Append-only on messages and receipts (UPDATE/DELETE blocked)
- FSM enforcement: only PENDING->LEASED->COMMITTED allowed
- Lease protection: direct lease changes blocked without proper claim
- Foreign keys: receipts->steps->jobs->messages integrity

API (message_cassette.py):
- post_message(): create message/job/step, idempotency support
- claim_step(): deterministic selection (oldest job, lowest ordinal)
- complete_step(): verify lease/token/expiry before completion
- verify_cassette(): check DB state and invariants

Tests:
- Append-only triggers block UPDATE/DELETE
- FSM illegal transitions rejected
- Referential integrity enforced
- Deterministic claim ordering
```

---

### Chunk 4: Phase 3 CLI Commands

**Files Changed:**
- `catalytic_chat/cli.py` (modified)

**Commit Message:**
```
feat(cat_chat): add phase 3 cassette CLI commands

New commands:
- cortex cassette verify --run-id <id>
- cortex cassette post --json <file> --run-id <id> --source <src>
- cortex cassette claim --run-id <id> --worker <id> [--ttl <s>]
- cortex cassette complete --run-id <id> --step <id> --worker <id> 
  --token <n> --receipt <file> --outcome <out>

No changes to Phase 1/2 commands (build, verify, get, extract, 
symbols, resolve).
```

---

### Chunk 5: Phase 3 Hardening (Adversarial Tests & Verify Strengthening)

**Files Changed:**
- `tests/test_message_cassette.py` (added 8 adversarial tests)
- `catalytic_chat/message_cassette.py` (strengthened verify_cassette)
- `catalytic_chat/cli.py` (cleaned verify output)

**Commit Message:**
```
refactor(cat_chat): harden phase 3 with adversarial tests and verify checks

Adversarial Tests (test_message_cassette.py):
- test_steps_delete_allowed_when_persisting_design: 
  confirm steps are NOT append-only (design intent)
- test_messages_update_delete_blocked_by_triggers: 
  direct SQL UPDATE/DELETE blocked by triggers
- test_receipts_update_delete_blocked_by_triggers: 
  direct SQL UPDATE/DELETE blocked by triggers
- test_illegal_fsm_transition_blocked: 
  direct SQL FSM jumps (PENDING->COMMITTED, COMMITTED->LEASED) blocked
- test_lease_direct_set_blocked: 
  direct SQL lease changes without claim blocked
- test_complete_fails_on_stale_token_even_if_lease_owner_matches: 
  completion with stale token fails
- test_complete_fails_on_expired_lease: 
  completion with expired lease fails

verify_cassette() Strengthening:
- Check PRAGMA foreign_keys is ON
- Check required tables exist
- Check all 8 required triggers exist by exact name
- Return exit code 0 on PASS, non-zero on FAIL
- Print single-line PASS/FAIL to stderr (no chatter)

CLI Output:
- verify: "PASS: All invariants verified" to stderr on success
- verify: "FAIL: N issue(s) found" to stderr on failure
```

---

## Commit Plan

**Recommended Order:**
1. Chunk 1 (Info Architecture Refactor) - standalone, no breaking changes
2. Chunk 2 (Vector Sandbox) - isolated experimental feature
3. Chunk 3 (Phase 3 Core) - main feature implementation
4. Chunk 4 (Phase 3 CLI) - adds commands for chunk 3
5. Chunk 5 (Phase 3 Hardening) - strengthens tests and verification

**All tests pass:**
```bash
python -m pytest -q
```

**Verification:**
```bash
python -m catalytic_chat.cli cassette verify --run-id r0
# Expected output: PASS: All invariants verified
```

---

## Test Summary

| Test File | Tests | Status |
|-----------|--------|--------|
| tests/test_placeholder.py | 1 | PASS |
| tests/test_vector_store.py | 10 | PASS |
| tests/test_message_cassette.py | 21 | PASS |
| **Total** | **32** | **PASS** |

---

## Files Changed Summary

| Path | Change Type |
|------|-------------|
| `docs/catalytic-chat/phases/*` | New/moved (4 files) |
| `docs/catalytic-chat/ROADMAP.md` | Moved |
| `docs/catalytic-chat/CHANGELOG.md` | Moved |
| `docs/catalytic-chat/notes/*` | New/moved (5 files) |
| `legacy/*` | New directory (10 scripts + 2 tests + data + README) |
| `pytest.ini` | Modified |
| `README.md` | Modified |
| `catalytic_chat/README.md` | Modified |
| `ROADMAP.md` | New (stub) |
| `CHANGELOG.md` | New (stub) |
| `catalytic_chat/experimental/__init__.py` | New |
| `catalytic_chat/experimental/vector_store.py` | New |
| `catalytic_chat/message_cassette_db.py` | New |
| `catalytic_chat/message_cassette.py` | New |
| `catalytic_chat/cli.py` | Modified |
| `tests/test_vector_store.py` | New |
| `tests/test_message_cassette.py` | New |

**Total New Files:** ~12
**Total Modified Files:** ~4
**Total Moved Files:** ~18
