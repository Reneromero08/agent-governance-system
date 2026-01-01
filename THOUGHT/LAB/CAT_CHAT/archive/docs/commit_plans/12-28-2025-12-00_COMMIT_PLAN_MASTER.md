---
title: "Commit Plan Master"
section: "report"
author: "System"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Master commit plan (Archived)"
tags: [commit_plan, archive]
---

<!-- CONTENT_HASH: 36aae825efce1d5761c3469c35339ad00bf30da25f7d36659f33b55aaa6ee33f -->

# Session Commit Plan & Change Log

Generated: 2025-12-29 (Updated: Chunk 7 added)

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

### Chunk 6: Phase 4.3 Ants (Multi-worker Agent Runners)

**Files Changed:**
- `catalytic_chat/ants.py` (new)
- `catalytic_chat/cli.py` (modified - added ants commands)
- `tests/test_ants.py` (new)
- `tests/conftest.py` (new - for PYTHONPATH setup)

**Commit Message:**
```
feat(cat_chat): implement phase 4.3 ants multi-worker agent runners

ants.py:
- AntConfig dataclass with worker configuration (run_id, job_id,
  worker_id, repo_root, poll_interval_ms, ttl_seconds, continue_on_fail,
  max_idle_polls)
- AntWorker class with run() method:
  - Poll loop claiming next step via claim_next_step()
  - Execute step via execute_step() with global budget check
  - Exit codes: 0 (clean), 1 (failure), 2 (invariant/DB error)
- spawn_ants() function:
  - Spawn N OS processes via subprocess
  - Worker IDs: ant_<run>_<job>_<i>
  - Write manifest to CORTEX/_generated/ants_manifest_<run>_<job>.json
  - Collect exit codes with priority: 2 > 1 > 0
- run_ant_worker() function: CLI entrypoint wrapper

CLI (cli.py):
- ants spawn --run-id <run> --job-id <job> -n <N> [--continue-on-fail]
- ants worker --run-id <run> --job-id <job> --worker-id <id>
  [--continue-on-fail] [--poll-ms 250] [--ttl 300] [--max-idle-polls 20]

Tests (test_ants.py):
- test_ant_worker_claims_and_executes: single worker executes all steps
- test_ants_spawn_two_workers_no_duplicates: two workers, no dup receipts
- test_ant_stops_on_fail_by_default: exit 1 on first failure
- test_ant_continue_on_fail_completes_others: continue but exit 1 if any fail
- test_ant_spawn_multiprocess: end-to-end subprocess spawn with manifest
```

---

### Chunk 7: Fix Substrate Path/Schema Mismatch

**Files Changed:**
- `catalytic_chat/paths.py` (new)
- `catalytic_chat/section_indexer.py` (modified)
- `catalytic_chat/section_extractor.py` (modified)
- `catalytic_chat/symbol_registry.py` (modified)
- `catalytic_chat/symbol_resolver.py` (modified)
- `catalytic_chat/message_cassette_db.py` (modified)
- `CORTEX/db/system1.db` (deleted - moved to CORTEX/_generated/)

**Commit Message:**
```
fix(cat_chat): unify substrate paths and fix schema initialization

paths.py (new):
- get_cortex_dir() -> CORTEX/_generated
- get_db_path() -> CORTEX/_generated/{name}
- get_system1_db() -> CORTEX/_generated/system1.db
- get_system3_db() -> CORTEX/_generated/system3.db
- get_sqlite_connection() -> standard connection with FK+WAL

section_indexer.py:
- Use get_system1_db() instead of hardcoded CORTEX/db/system1.db
- Use get_sqlite_connection() for all DB access
- Ensure sections table exists on connect

symbol_registry.py:
- Use get_system1_db() and get_sqlite_connection()
- Create symbols table on every DB access (fixes "no such table: symbols")
- Add table creation to _get_symbol_sqlite() and _list_symbols_sqlite()

symbol_resolver.py:
- Use get_system1_db() and get_sqlite_connection()
- Create expansion_cache table on connect

message_cassette_db.py:
- Use get_system3_db() and get_sqlite_connection()
- Remove duplicate mkdir logic

section_extractor.py:
- Compute relative_path from repo_root (fixes "0 sections" when running from CAT_CHAT folder)

Fixes:
- "symbols list" no longer fails with "no such table: symbols"
- Build writes to CORTEX/_generated/system1.db consistently
- SECTION_INDEX.json located in CORTEX/_generated/ (via DB)
- Build extracts sections from repo_root instead of cwd
- Schema init runs deterministically on first connect
```

---

### Chunk 8: Phase 6.2.1 Attestation Stabilization

**Files Changed:**
- `tests/test_planner.py` (modified - fixed test_cli_dry_run subprocess call)

**Commit Message:**
```
fix(cat_chat): stabilize phase 6.2.1 by fixing test_cli_dry_run subprocess import

Issue: test_cli_dry_run was failing with:
  ModuleNotFoundError: No module named 'catalytic_chat'

Root cause: subprocess.run() call to 'python -m catalytic_chat.cli plan request'
did not inherit PYTHONPATH environment variable needed to find the module.

Fix: Added environment setup to test_cli_dry_run():
  - Import os module
  - Copy os.environ and set PYTHONPATH to Path(__file__).parent.parent
  - Pass env parameter to subprocess.run()

Impact:
  - All 59 tests now pass (previously 58 passed, 1 failed)
  - No new skips added (same 13 skips)
  - Deterministic receipts unchanged
  - Attestation still fail-closed
  - No CLI or executor regressions
  - Minimal fix (3 lines added, only test environment)

Verification:
  python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests
  # 59 passed, 13 skipped in 6.10s
```

---

### Chunk 9: Phase 6.3 Receipt Chain Anchoring

**Files Changed:**
- `catalytic_chat/receipt.py` (modified - added chain functions)
- `catalytic_chat/executor.py` (modified - added previous_receipt param)
- `SCHEMAS/receipt.schema.json` (modified - added chain fields)
- `catalytic_chat/cli.py` (modified - added --verify-chain flag)
- `tests/test_receipt_chain.py` (new - 4 chain tests)

**Commit Message:**
```
feat(cat_chat): add phase 6.3 receipt chain anchoring

Add deterministic receipt chaining with parent_receipt_hash linkage.

receipt.py:
- compute_receipt_hash(): compute hash from canonical bytes (excluding receipt_hash field)
- load_receipt(): load receipt from JSON file
- verify_receipt_chain(): verify chain linkage and receipt hashes
- find_receipt_chain(): find all receipts for a run in execution order

executor.py:
- Add previous_receipt parameter to BundleExecutor
- Load previous receipt and set parent_receipt_hash from its receipt_hash
- First receipt has parent_receipt_hash=null

receipt.schema.json:
- Add parent_receipt_hash field (string | null)
- Add receipt_hash field (string)

cli.py:
- Add --verify-chain flag to bundle run command
- Verify full receipt chain when --verify-chain is set
- Output chain status and receipt count

tests/test_receipt_chain.py:
- test_receipt_chain_deterministic: identical inputs produce identical chain
- test_receipt_chain_verification_passes: chain verification succeeds
- test_receipt_chain_break_fails: tamper detection
- test_receipt_chain_requires_sequential_order: reorder fails
```

---

### Chunk 10: Phase 6.4 Receipt Merkle Root + External Anchor

**Files Changed:**
- `catalytic_chat/receipt.py` (modified - added Merkle functions)
- `catalytic_chat/cli.py` (modified - added --print-merkle flag)
- `tests/test_merkle_root.py` (new - 3 Merkle root tests)

**Commit Message:**
```
feat(cat_chat): add phase 6.4 merkle root over receipt chains

Add deterministic Merkle root computation for receipt chains.

receipt.py:
- compute_merkle_root(): compute Merkle root from receipt hashes
  - Pairwise concatenate hex-decoded bytes (left||right)
  - SHA256 on concatenated bytes
  - Duplicate last node at each level if odd count
  - Preserve deterministic ordering (no re-sorting)
- verify_receipt_chain(): now returns Merkle root string

cli.py:
- Add --print-merkle flag to bundle run command
- --print-merkle requires --verify-chain (fail-closed)
- Print ONLY Merkle root hex to stdout when set
- Suppress all other output when --print-merkle is set

tests/test_merkle_root.py:
- test_merkle_root_deterministic: same hashes produce identical Merkle root
- test_merkle_root_changes_on_tamper: tampering changes root
- test_merkle_root_requires_verify_chain: --print-merkle without --verify-chain fails
```

---

### Chunk 11: Phase 6.5 Signed Merkle Attestation

**Files Changed:**
- `SCHEMAS/merkle_attestation.schema.json` (new)
- `catalytic_chat/merkle_attestation.py` (new)
- `catalytic_chat/cli.py` (modified - added merkle attestation flags)
- `tests/test_merkle_attestation.py` (new)
- `CAT_CHAT_ROADMAP.md` (updated - Phase 6.5 marked COMPLETE)
- `CHANGELOG.md` (updated - added 6.5 entry)

**Commit Message:**
```
feat(cat_chat): add phase 6.5 signed merkle attestation with strict stdout purity

Add Ed25519 signing and verification of receipt chain Merkle root.

merkle_attestation.py:
- sign_merkle_root(): sign merkle root with Ed25519 private key
  - Message: b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes (decoded 32 bytes)
  - Hex output for public_key and signature
- verify_merkle_attestation(): verify signature with strict validation
  - Validates hex, lengths (64/128), scheme
- validate_merkle_root_hex(): strict hex length validation
- write_merkle_attestation(): write canonical JSON with trailing newline
- load_merkle_attestation(): load from file

SCHEMAS/merkle_attestation.schema.json:
- scheme: const "ed25519"
- merkle_root: 64 hex chars pattern
- public_key: 64 hex chars pattern
- signature: 128 hex chars pattern
- Optional fields: run_id, job_id, bundle_id, receipt_count, receipt_chain_head_hash
- additionalProperties: false everywhere

cli.py - bundle run command:
- --attest-merkle: sign merkle root and emit attestation
  - Requires --verify-chain and --merkle-key
  - Prints ONLY JSON + \n to stdout when no --merkle-attestation-out specified
- --merkle-key <hex>: Ed25519 signing key (64 hex chars)
- --verify-merkle-attestation <path>: verify attestation against computed root
  - Requires --verify-chain
- --merkle-attestation-out <path>: write attestation to file
  - Default: print to stdout (JSON-only + \n)
- verbose_output flag: suppress status when JSON-only output required
- Strict flag interactions:
  - --print-merkle and --attest-merkle mutually exclusive
  - --attest-merkle and --verify-merkle-attestation mutually exclusive
- All error messages use sys.stderr.write() for consistency

tests/test_merkle_attestation.py (12 tests):
- test_merkle_attestation_sign_verify_roundtrip: roundtrip succeeds
- test_merkle_attestation_rejects_modified_root: tamper fails
- test_merkle_attestation_rejects_invalid_merkle_root_length: wrong length rejected
- test_merkle_attestation_rejects_invalid_signing_key_length: wrong key length rejected
- test_merkle_attestation_rejects_invalid_hex: non-hex rejected
- test_merkle_attestation_verify_rejects_wrong_scheme: non-ed25519 rejected
- test_merkle_attestation_verify_rejects_wrong_key_length: wrong pub key length rejected
- test_merkle_attestation_verify_rejects_wrong_signature_length: wrong sig length rejected
- test_bundle_run_attest_merkle_outputs_deterministic_bytes: identical outputs across runs
- test_bundle_run_verify_merkle_attestation_fails_on_mismatch: root mismatch fails
- test_merkle_attestation_write_load_roundtrip: file I/O preserves attestation
- test_merkle_attestation_load_nonexistent_file: missing file returns None

Invariants:
- Deterministic: identical inputs produce identical merkle root and signature bytes
- Fail-closed: chain tamper, ordering mismatch, schema mismatch, signature mismatch => non-zero exit
- No timestamps, randomness, absolute paths in outputs
- Signing does NOT change receipt_hash or merkle root computation
- Canonicalization single source of truth: reuses receipt_canonical_bytes() and compute_merkle_root()
- Stdout purity: --attest-merkle prints ONLY JSON + \n (no [OK] prefixes)
- Canonical JSON: sort_keys=True, separators=(",", ":"), UTF-8, \n newlines, \n EOF
```

---

## Commit Plan

**Recommended Order:**
1. Chunk 1 (Info Architecture Refactor) - standalone, no breaking changes
2. Chunk 2 (Vector Sandbox) - isolated experimental feature
3. Chunk 3 (Phase 3 Core) - main feature implementation
4. Chunk 4 (Phase 3 CLI) - adds commands for chunk 3
5. Chunk 5 (Phase 3 Hardening) - strengthens tests and verification
6. Chunk 6 (Phase 4.3 Ants) - multi-worker agent runners
7. Chunk 7 (Substrate Path Fix) - critical bugfix, must come after all
8. Chunk 8 (Phase 6.2.1 Stabilization) - fix test_cli_dry_run subprocess issue
9. Chunk 9 (Phase 6.3 Receipt Chain Anchoring) - deterministic receipt chaining
10. Chunk 10 (Phase 6.4 Merkle Root) - Merkle root over receipt chains
11. Chunk 11 (Phase 6.5 Signed Merkle Attestation) - Ed25519 signing of Merkle root
11. Chunk 11 (Phase 6.5 Signed Merkle Attestation) - Ed25519 signing of Merkle root

**All tests pass:**
```bash
python -m pytest -q
```

**Verification:**
```bash
python -m catalytic_chat.cli cassette verify --run-id r0
# Expected output: PASS: All invariants verified

# Ants verification:
python -m catalytic_chat.cli plan request --request-file tests/fixtures/plan_request_parallel.json
python -m catalytic_chat.cli ants spawn --run-id test_parallel_request --job-id <job_id> -n 4
python -m catalytic_chat.cli cassette verify --run-id test_parallel_request

# Substrate path fix verification (from CAT_CHAT folder):
cd THOUGHT/LAB/CAT_CHAT
python -m catalytic_chat.cli --repo-root "D:\CCC 2.0\AI\agent-governance-system" build
# Expected: Wrote 61 sections to D:\CCC 2.0\AI\agent-governance-system\CORTEX\_generated\system1.db

python -m catalytic_chat.cli --repo-root "D:\CCC 2.0\AI\agent-governance-system" symbols list --prefix "@"
# Expected: Listing 0 symbols (no errors)

# Merkle attestation verification (Phase 6.5):
python -m catalytic_chat.cli bundle run --bundle <bundle_dir> --verify-chain --attest-merkle --merkle-key <64_hex_key>
# Expected: JSON-only stdout (canonical) when no --merkle-attestation-out specified

python -m catalytic_chat.cli bundle run --bundle <bundle_dir> --verify-chain --print-merkle
# Expected: Merkle root hex only (no prefixes)
```

---

## Test Summary

| Test File | Tests | Status |
|-----------|--------|--------|
| tests/test_placeholder.py | 1 | PASS |
| tests/test_vector_store.py | 9 | PASS |
| tests/test_message_cassette.py | 21 | PASS |
| tests/test_ants.py | 7 | PASS |
| tests/test_planner.py | 6 | PASS |
| tests/test_execution.py | 2 | PASS |
| tests/test_execution_parallel.py | 4 | PASS |
| tests/test_attestation.py | 6 | PASS |
| tests/test_receipt.py | 5 | PASS |
| tests/test_bundle.py | 2 | PASS |
| tests/test_bundle_execution.py | 5 | PASS |
| tests/test_receipt_chain.py | 4 | PASS |
| tests/test_merkle_root.py | 3 | PASS |
| tests/test_merkle_attestation.py | 12 | PASS |
| **Total** | **78** | **PASS** (13 skipped) |

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
| `catalytic_chat/ants.py` | New |
| `catalytic_chat/paths.py` | New |
| `catalytic_chat/receipt.py` | Modified (attestation, chain, merkle) |
| `catalytic_chat/attestation.py` | New (signing/verification) |
| `catalytic_chat/executor.py` | Modified (attestation, chain support) |
| `catalytic_chat/bundle_execution.py` | Modified (verification) |
| `catalytic_chat/cli.py` | Modified (attestation, chain, merkle, merkle attestation flags) |
| `catalytic_chat/section_indexer.py` | Modified |
| `catalytic_chat/section_extractor.py` | Modified |
| `catalytic_chat/symbol_registry.py` | Modified |
| `catalytic_chat/symbol_resolver.py` | Modified |
| `catalytic_chat/planner.py` | New (request planning) |
| `catalytic_chat/merkle_attestation.py` | New (chunk 11) |
| `tests/test_vector_store.py` | New |
| `tests/test_message_cassette.py` | New |
| `tests/test_ants.py` | New |
| `tests/test_execution.py` | New |
| `tests/test_execution_parallel.py` | New |
| `tests/test_planner.py` | New (stabilized in chunk 8) |
| `tests/test_attestation.py` | New |
| `tests/test_receipt.py` | New |
| `tests/test_bundle.py` | New |
| `tests/test_bundle_execution.py` | New |
| `tests/test_receipt_chain.py` | New (chunk 9) |
| `tests/test_merkle_root.py` | New (chunk 10) |
| `tests/test_merkle_attestation.py` | New (chunk 11) |
| `tests/conftest.py` | New |
| `tests/fixtures/*` | New (plan request fixtures) |
| `SCHEMAS/receipt.schema.json` | Modified (added chain fields) |
| `SCHEMAS/merkle_attestation.schema.json` | New (chunk 11) |
| `CORTEX/db/system1.db` | Deleted |

**Total New Files:** ~36
**Total Modified Files:** ~16
**Total Moved Files:** ~18
**Total Deleted Files:** 1
