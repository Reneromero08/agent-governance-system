<!-- CONTENT_HASH: updated_with_phase_g_and_h -->

All notable changes to **CAT_CHAT (Catalytic Chat)** are documented in this file.

## [1.2.7] - 2026-01-19

### Added

**Phase H: Specs & Golden Demo (P3) - COMPLETE**

- **Authoritative specifications in `docs/specs/`:**
  - `SPEC_INDEX.md` - Index linking all specification documents
  - `BUNDLE_SPEC.md` - Bundle protocol v5.0.0 specification
    - Bundle structure (bundle.json, artifacts/)
    - Manifest and artifact schemas
    - Hash computation algorithms (bundle_id, root_hash, plan_hash)
    - Completeness gate and boundedness constraints
    - Canonical JSON rules and verification protocol
  - `RECEIPT_SPEC.md` - Receipt format v1.0.0 specification
    - Receipt schema and step result format
    - Receipt hash computation algorithm
    - Chain linking protocol (parent_receipt_hash)
    - Merkle root computation (binary tree)
    - Attestation object structure and signing process
  - `TRUST_SPEC.md` - Trust model v1.0.0 specification
    - Trust policy schema and validator entries
    - Trust index construction (by_public_key, by_validator_id)
    - Key authorization check algorithm
    - Scope definitions (RECEIPT, MERKLE)
    - Strict trust and strict identity modes
    - Signature verification (Ed25519)
  - `EXECUTION_SPEC.md` - Execution semantics v1.0.0 specification
    - Execution policy schema and fields
    - Execution flow and step execution
    - Fail-closed behavior table
    - Post-execution policy enforcement
    - Exit codes (0=OK, 1=verify fail, 2=invalid input, 3=internal error)
    - Determinism requirements and artifact confinement

- **Golden demo in `golden_demo/`:**
  - `golden_demo.py` (290 lines) - Self-contained demo script
    - No external dependencies except Python stdlib + jsonschema
    - Demonstrates complete pipeline: bundle creation, verification, execution, receipt verification
    - Determinism check: re-runs bundle creation to verify identical hashes
    - Clear step-by-step output with progress indicators
  - `README.md` - Demo documentation with quick start, expected output, troubleshooting
  - `fixtures/demo_content.txt` - Sample content demonstrating CAT_CHAT principles
  - `fixtures/demo_trust_policy.json` - Example trust policy for demo

- **Updated documentation:**
  - `docs/CAT_CHAT_USAGE_GUIDE.md` - Added "Fresh Clone Quick Start" section
    - Direct link to golden demo execution
    - Links to specification documents
    - Positioned at top for new users

**Phase G: Bundle Replay & Verification (P2) - COMPLETE**

- **Bundle verification and execution:**
  - `catalytic_chat/bundle.py` - BundleVerifier class (lines 393-569)
    - Schema validation (required fields check)
    - Ordering validation (steps by ordinal/step_id, artifacts by artifact_id)
    - Artifact hash verification (SHA256 matching)
    - Root hash recomputation and comparison
    - Bundle ID recomputation and verification
    - Boundedness validation (no ALL slices, all artifacts referenced)
    - Forbidden fields rejection (no timestamps, cwd, os, locale)
  - `catalytic_chat/executor.py` - BundleExecutor class (lines 15-249)
    - Deterministic bundle execution
    - Receipt generation from bundle results
    - Receipt signing with Ed25519 keys
    - Post-execution policy enforcement
    - Chain linking support (parent_receipt_hash)
  - Exit codes: 0 (success), 1 (verification failed), 2 (invalid input), 3 (internal error)

- **Test coverage for bundle execution:**
  - `tests/test_bundle_execution.py` (10 tests, all passing)
    - Bundle verification with absolute and relative paths
    - Bundle verification with bundle.json path
    - Receipt generation and hash computation
    - Deterministic execution (same bundle = same receipt)
    - Chain integrity verification

### Changed

- `CAT_CHAT_ROADMAP_2.0.md` - Marked Phase G and Phase H as COMPLETE with deliverables

## [1.2.6] - 2026-01-19

### Added

**Phase F: Docs Index (P2) - COMPLETE**

- **Full-text search index for repository documentation**
  - `catalytic_chat/docs_index.py` (600+ lines) - DocsIndex class with FTS5
    - `DocsSearchResult`, `DocsSearchResponse`, `IndexStats` dataclasses
    - `index_file()` - Index single file with sha256 deduplication
    - `index_all()` - Build/rebuild full index (incremental by default)
    - `search()` - Bounded, deterministic FTS5 search
    - `get_file_content()` - Retrieve file by hash
    - `get_stats()` - Index statistics
  - Text normalization: lowercase, strip markdown, collapse whitespace
  - Chunking support: Files > 50KB split at paragraph boundaries
  - Porter stemming tokenizer for English documents

- **FTS5 database schema in cat_chat.db:**
  - `docs_files` table - File metadata with sha256 uniqueness
  - `docs_content` table - Normalized text with chunk support
  - `docs_content_fts` - FTS5 virtual table with Porter stemming
  - Automatic sync triggers (INSERT, UPDATE, DELETE)
  - Foreign key constraints and indexes

- **CLI commands for documentation search:**
  - `catalytic_chat/cli.py` (+150 lines) - Added `docs` subcommand group:
    - `docs index [--full]` - Build/rebuild documentation index
    - `docs search --query "..." --limit N [--offset N] [--json]` - Search with bounded results
    - `docs show <sha256>` - Show file content by hash
    - `docs stats` - Show index statistics
  - JSON output mode for machine consumption
  - Snippet highlighting with ANSI codes in terminal output

- **Deterministic ranking:**
  - BM25 score (ascending - lower is better in FTS5)
  - file_path (alphabetical) - stable tie-breaker
  - chunk_index (ascending) - secondary tie-breaker
  - Same query always returns same results in same order

- **Integration with retrieval chain (Phase F.4):**
  - `catalytic_chat/cortex_expansion_resolver.py` (+40 lines) - Step 3b integration:
    - DocsIndex inserted between symbol registry and CAS
    - Lazy-loaded `docs_index` property
    - `_format_docs_results()` method for result formatting
    - `docs_index_hits` stat tracking
    - Updated retrieval order:
      1. SPC resolution (Phase D)
      2. Main Cassette FTS
      3. Symbol registry (local index)
      3b. DocsIndex FTS (NEW - Phase F)
      4. CAS (exact hash)
      5. Vector Fallback
      6. Fail-closed

- **Comprehensive test coverage:**
  - `tests/test_docs_index.py` (400+ lines) - 26 tests, all passing:
    - Schema creation and indexes
    - File indexing with sha256 deduplication
    - Incremental indexing (skip unchanged files)
    - Search with bounded results
    - Deterministic ranking verification
    - Snippet length enforcement (200 char max)
    - Large file chunking
    - FTS sync triggers
    - Content retrieval by hash
    - Text normalization
    - Empty query handling
    - Convenience functions

### Changed

**Retrieval Chain Order (Phase F Integration)**

- DocsIndex now at position 3b (between symbol registry and CAS)
- Provides fast keyword-based discovery before expensive vector search
- All results hash-verified (no trust bypass)

**Statistics Tracking**

- Added `docs_index_hits` to resolver statistics
- Updated `get_stats()` to include docs_index in total_hits calculation

### Security

- Content-addressable storage (sha256)
- Bounded results (max limit enforced)
- Bounded snippets (200 char max)
- No absolute paths (all relative to repo_root)
- Deterministic behavior (no timestamps in search results)

**Tests**

- 26 tests passing in test_docs_index.py
- Full integration with existing test suite
- Determinism verified across multiple runs

**Exit Criteria Verification**

- [x] FTS tables exist in cat_chat.db
- [x] `docs index` rebuilds index (3 files indexed in test run)
- [x] `docs search --query "..." --limit N` returns bounded results
- [x] Results are deterministic (verified with 3 consecutive runs)
- [x] Snippets bounded at 200 chars
- [x] All 26 tests passing
- [x] Integrated into retrieval chain (step 3b)

---

## [1.2.5] - 2026-01-19

### Added

**Phase E: Vector Fallback Chain (P2) - COMPLETE**

**E.1: Retrieval Order Enforcement**

- **Strict retrieval chain with governed fallback to vectors**
  - `catalytic_chat/cortex_expansion_resolver.py` (updated) - Enforced retrieval order:
    1. SPC resolution (Phase D)
    2. Main Cassette FTS
    3. Local Index (symbol registry)
    4. CAS (exact hash lookup)
    5. Vector Fallback (governed semantic search)
    6. Fail-closed
  - No trust-vectors bypass - all results hash-verified
  - Stats tracking for each retrieval source

**E.2: Vector Governance**

- **Governed semantic search with hard token budgets**
  - `catalytic_chat/vector_fallback.py` (new) - VectorFallbackResolver class
    - Agent is FREE to search until it finds what it needs
    - Budget is SAFETY BOUNDARY, not fill target
    - `min_similarity` = 0.5 (empirically validated in Q44)
    - Config file: `_generated/vector_fallback_config.json` for tuning
    - Search logging: `_generated/vector_fallback_search.jsonl` for analysis
  - `catalytic_chat/cas_resolver.py` (new) - CASResolver for exact hash lookup
    - Content-addressable storage with verification
    - Hash-based deduplication
    - Fail-closed on mismatch

**E.3: ELO Metadata Tracking**

- **Usage pattern tracking (does NOT affect ranking)**
  - `catalytic_chat/elo_observer.py` (new) - EloObserver class
    - Tracks retrieval events AFTER results returned
    - Logs to session for analytics
    - Never modifies ranking (called after retrieval complete)
  - Session-scoped tracking with `on_retrieval_complete()` method

### Tests

- `tests/test_cas_resolver.py` (19 tests) - CAS exact hash lookup
- `tests/test_vector_fallback.py` (24 tests) - Vector fallback with governance
- `tests/test_retrieval_order.py` (11 tests) - Retrieval chain order enforcement
- **54 tests passing total**

### Exit Criteria

- [x] Retrieval order enforced (6 steps)
- [x] Vector governance with hard budgets
- [x] ELO tracking as metadata only
- [x] All 54 tests passing

---

## [1.2.4] - 2026-01-19

### Added

**Phase D: SPC Integration (P1) - COMPLETE**

**D.1: Codebook Sync Handshake**

- **Fail-closed codebook verification on session start**
  - `catalytic_chat/spc_bridge.py` (new) - SPCBridge class
    - `sync_handshake()` - Verify codebook_id + SHA256
    - Fail-closed on mismatch
    - EVENT_CODEBOOK_SYNC logged to session events
  - Ensures SPC pointers resolve to correct expansions

**D.2: Pointer Resolution**

- **Support for three pointer types:**
  - SYMBOL_PTR - CJK characters (`法`) + ASCII radicals (C, I, V, L, G, S, R, A, J, P)
  - HASH_PTR - sha256 prefixed hashes (`sha256:7cfd0418...`)
  - COMPOSITE_PTR - Combined forms (`法.驗`, `C3:build`, `C&I`)
- **Highest priority in resolution chain**
  - SPC resolves BEFORE CORTEX queries
  - Integrated into cortex_expansion_resolver.py

**D.3: Compression Metrics**

- **Token savings tracking:**
  - tokens_expanded - Full content tokens
  - tokens_pointers - Pointer tokens used
  - tokens_saved - Compression achieved
  - CDR (Concept Density Ratio) per Q33
  - Per-symbol usage and savings tracking
  - EVENT_SPC_METRICS for session logging

### Tests

- `tests/test_spc_integration.py` (35 tests) - All passing
  - Codebook sync verification
  - Pointer resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR)
  - Compression metrics calculation
  - Fail-closed on mismatch

### Exit Criteria

- [x] Codebook sync handshake with fail-closed
- [x] Three pointer types resolve correctly
- [x] Compression metrics tracked
- [x] All 35 tests passing

---

## [1.2.3] - 2026-01-18

### Added

**Phase C: Auto-Controlled Context Loop (P0) - COMPLETE**

**C.1-C.5: Core Catalytic Behavior**

- **Virtual memory for LLMs - automatic working set management**
  - `catalytic_chat/auto_context_manager.py` (new) - AutoContextManager class
    - Working set (materialized) vs pointer set (offloaded)
    - E-score based eviction when budget exceeded
    - E-score based hydration (above threshold)
    - Hard fail if working_set exceeds budget (INV-CATALYTIC-04)
  - `catalytic_chat/context_partitioner.py` (new) - Context partitioning logic
  - `catalytic_chat/turn_compressor.py` (new) - Turn compression to hash pointers
  - `catalytic_chat/adaptive_budget.py` (new) - Budget management

**E-Score Implementation:**

- **Born rule quantum overlap for relevance scoring**
  - E = |<query_vec | item_vec>|^2 (Born rule)
  - Threshold = 0.5 (validated in Q44)
  - `THOUGHT/LAB/FORMULA/experiments/open_questions/q44/q44_core.py`:
    - `compute_E_linear(query_vec, context_vecs)` - E = mean overlap
    - `compute_born_probability(query_vec, context_vecs)` - P = |<psi|phi>|^2

**Catalytic Chat Loop:**

- **Wire together: query -> hydrate -> assemble -> LLM -> compress -> evict**
  - Session capsule logs every step (deterministic replay)
  - All context assembly uses ContextAssembler with budgets
  - GeometricChat.respond() uses auto-managed context

### Tests

- `tests/test_auto_context_loop.py` - Core loop tests
- Real conversation tests:
  - 500-turn extreme marathon with context decay analysis
  - 1000-turn ultimate marathon
  - Nemotron demo

### Exit Criteria

- [x] Model runs with fully auto-managed context
- [x] No manual @symbol references needed
- [x] Working set stays within budget across entire session
- [x] Eviction/hydration decisions logged and reproducible
- [ ] C.6.3 E-score vs quality correlation tracking (marked "Future")

---

## [1.2.2] - 2026-01-17

### Added

**Phase B: Main Cassette Network Integration (P1) - COMPLETE**

**B.1-B.2: Cassette Network Access**

- **Connect to existing cassette infrastructure for richer content**
  - `catalytic_chat/cassette_client.py` (new) - CassetteClient class
    - Read from `NAVIGATION/CORTEX/cassettes/*.db`
    - Search across canon.db, governance.db, etc.
    - Respect cassette network conventions
  - Symbol resolution via main cassettes:
    - Resolve @symbols to content in main cassettes
    - Fall back to local index if not found

**B.3: Write Isolation**

- **Reads: main cassettes (shared), Writes: _generated/cat_chat.db (sandbox)**
  - Documented in `docs/WRITE_ISOLATION.md`
  - Prevents CAT Chat from polluting main cassette network
  - Safe experimentation in LAB environment

**B.4: Graduation Path**

- **Documented future migration path**
  - `docs/GRADUATION_PATH.md` (future reference)
  - CAT Chat stays in LAB for now
  - Clear criteria for promotion to NAVIGATION/CORTEX

### Tests

- `tests/test_cassette_client.py` (22 tests) - Cassette network reading
- `tests/test_cassette_symbol_resolution.py` (15 tests) - Symbol resolution
- **37 tests passing total**

### Exit Criteria

- [x] CAT Chat reads from main cassette network
- [x] Writes isolated to _generated/cat_chat.db
- [x] Write isolation documented
- [x] All 37 tests passing

---

## [1.2.1] - 2026-01-16

### Added

**Phase A: Session Persistence Tests (P0) - COMPLETE**

**A.1-A.4: Deterministic Session Persistence**

- **Byte-identical replay and fail-closed tamper detection**
  - `tests/test_session_persistence.py` - 4 comprehensive tests:
    - A.1: Save/resume determinism test (byte-identical replay)
    - A.2: Partial execution resume test (no state loss)
    - A.3: Tamper detection test (fail-closed on corruption)
    - A.4: Hydration failure test (fail-closed on unresolvable symbols)
  - Session capsule foundation for catalytic behavior
  - Hash-chained event log for integrity

### Tests

- 4 fixtures created and all green
- Determinism proven across save/resume cycles
- Fail-closed behavior verified

### Exit Criteria

- [x] All 4 fixtures green
- [x] Determinism proven (byte-identical replay)
- [x] Tamper detection works (fail-closed)
- [x] Hydration failure handling (fail-closed)

---

## [1.1.0] - 2026-01-02

### Documentation Reorganization (v1.1 Housekeeping)

**Canon documents renamed and versioned:**
- `README.md` → `CAT_CHAT_README_1.1.md`
- `CHANGELOG.md` → `CAT_CHAT_CHANGELOG_1.1.md`
- `ROADMAP.md` → `CAT_CHAT_ROADMAP_1.1.md`

**Archived legacy versions:**
- `archive/docs/canon/README_1.md` (deprecated notice added)
- `archive/docs/canon/CHANGELOG_1.md` (deprecated notice added)
- `archive/docs/canon/CAT_CHAT_ROADMAP_1.md` (deprecated notice added)

**Canonical filename compliance (timestamp + ALL_CAPS):**
- `catalytic-chat-research.md` → `01-01-2026-18-19_CATALYTIC_CHAT_RESEARCH.md`
- `CAT_CHAT_PROMPTS_AND_HANDOFFS.md` → `01-02-2026-03-43_CAT_CHAT_PROMPTS_AND_HANDOFFS.md`

**Content hashes added:**
- All new `_1.1.md` canon documents
- All status documents in `archive/docs/status/`

### Archive Consolidation

**Demo scripts relocated:**
- `docs/examples/demo_ingest.py` → `archive/legacy/demo_ingest.py`
- `docs/examples/demo_query_trace.py` → `archive/legacy/demo_query_trace.py`
- Updated `docs/DATABASE_MAINTENANCE.md` references
- Updated `archive/legacy/README.md` with script descriptions

**Archive structure cleanup:**
- Consolidated duplicate documentation
- Organized status reports by date
- Removed orphaned/duplicate files

### History Merging

**History merged and consolidated:**
- Multiple conflicting versions of README, CHANGELOG, and ROADMAP merged into single canonical `_1.1.md` files
- Removed duplicate/conflicting files:
  - `CAT_CHAT_ROADMAP_FINAL.md`, `CAT_CHAT_ROADMAP_UPDATED_v2.md`, `CAT_CHAT_ROADMAP_V2.md` (deleted)
  - `CHANGELOG_V2.md` (deleted)
  - Legacy versions archived to `archive/docs/canon/` with deprecation notices
- Roadmap history was merged and consolidated:
  - Unified phase tracking across all versions
  - Consolidated completion status and exit criteria
  - Removed conflicting/duplicate phase definitions
  - Updated cross-references to master roadmap and other labs
- Changelog history was merged and consolidated:
  - Migrated stray entries from main AGS changelog
  - Preserved complete phase history (Phases 0-7)
  - Maintained chronological ordering and version tags
  - Removed duplicate entries and conflicting version information
- This entire thing was a mess, now it's nice and tidy (sort of).

### Roadmap Updates

**System-wide enhancements moved to master roadmap:**
- Lane Ω (God-Tier) removed from CAT_CHAT roadmap
- Now lives in `AGS_ROADMAP_MASTER.md` Lane Ω:
  - Ω.1: Performance Foundation (Incremental Indexing, Query Caching, Metrics)
  - Ω.2: Scale & Governance (Federation, Temporal Queries, Receipt Compression)
  - Ω.3: Intelligence & UX (Auto-Symbols, Smart Slicing, Provenance Graphs, ZK Proofs)

**Lane T (Tiny Model) moved:**
- Tiny Model Compression Lab removed from CAT_CHAT roadmap
- Now lives in `THOUGHT/LAB/TINY_COMPRESS/`

**Roadmap v1.1 updates:**
- Clarified Phase 8 scope (Vector Resident)
- Updated cross-references to master roadmap
- Cleaned up completed phase markers

### ADR & Governance

**Universal document hashing:**
- SHA-256 content hashes applied to all `.md` files

---


## [1.0.1] - 2026-01-01



### Fixed

- `section_extractor.py`: Computes relative paths from `repo_root` (fixes “0 sections” when running from CAT_CHAT folder)
- `section_indexer.py`: Uses canonical path helpers; ensures schema init on connect
- `symbol_registry.py`: Uses canonical path helpers; creates tables on every DB access (fixes “no such table: symbols”)
- `symbol_resolver.py`: Uses canonical path helpers; ensures `expansion_cache` table exists
- `message_cassette_db.py`: Uses `get_system3_db()` helper

### Added

- `catalytic_chat/paths.py`: Canonical path helpers
  - `get_cortex_dir()` → `CORTEX/_generated`
  - `get_db_path()` → `CORTEX/_generated/{name}`
  - `get_system1_db()` → `CORTEX/_generated/system1.db`
  - `get_system3_db()` → `CORTEX/_generated/system3.db`
  - `get_sqlite_connection()` → Standard connection with FK + WAL

### Changed

- `planner.py`:
  - Dry-run missing symbol handling: steps now include `expected_outputs: {"unresolved_symbols": ["@SYMBOL_ID"]}` for missing symbols instead of failing
  - Rerun idempotency: `post_request_and_plan()` uses `INSERT OR IGNORE` for DB ops and always does a final lookup, preventing UNIQUE constraint crashes
- Tests:
  - Added `tests/fixtures/plan_request_missing_symbol.json`
  - Added `test_plan_request_dry_run_missing_symbol_does_not_fail`
  - Added `test_plan_request_idempotent_rerun_no_unique_constraint`
  - Updated `test_cli_dry_run` to handle both resolved and unresolved symbols

### Security


- Bundle provenance determinism audit: removed wall-clock timestamp fields from provenance and replaced provenance with an empty object to preserve purity
- Added `COMMIT_PLAN_BUNDLE.md` documenting deterministic bundle construction and verification


## [1.0.0] - 2025-12-31

### Added

**(Compression Protocol Formalization) - COMPLETE**

- **`THOUGHT/LAB/CAT_CHAT/PHASE_7_COMPRESSION_SPEC.md`** (320 lines)
  - Authoritative specification for compression protocol
  - Defines compression metric definitions (ratio, numerator/denominator)
  - Component definitions: vector_db_only, symbol_lang, f3, cas
  - Reconstruction procedures
  - Threat model (what IS proven vs NOT proven)
  - Verification checklist (8 phases with 20+ checks)
  - Error codes table (28 error codes)
  - Deterministic computation rules (token estimation, artifact ordering, hash computation)
  - Fail-closed behavior

- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/compression_claim.schema.json`** (67 lines)
  - JSON schema with `additionalProperties: false` everywhere
  - Required fields: claim_version, run_id, bundle_id, components, reported_metrics, claim_hash
  - Component definitions (vector_db_only, symbol_lang, f3, cas)
  - Reported metrics (compression_ratio, uncompressed_tokens, compressed_tokens, artifact_count, total_bytes)
  - Optional component metrics (vector_db_tokens, symbol_lang_tokens, f3_tokens, cas_tokens)
  - F3 marked as theoretical (validator fails if included)
  - Stable identifiers, no timestamps

- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/compression_validator.py`** (470 lines)
  - `CompressionValidator` class with 8-phase verification pipeline
  - Entry function: `validate_compression_claim()`
  - Deterministic metric computation from verified artifacts only
  - 28 error codes with explicit messages
  - Supports strict_trust, strict_identity, require_attestation flags
  - Reuses: BundleVerifier, find_receipt_chain, verify_receipt_chain, receipt_canonical_bytes, canonical_json_bytes

- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py`** (+~50 lines)
  - Added `compress verify` command
  - Command: `python -m catalytic_chat.cli compress verify --bundle <path> --receipts <dir> --trust-policy <path> --claim <json> [--strict-trust] [--strict-identity] [--require-attestation] [--json] [--quiet]`
  - Exit codes: 0 (OK), 1 (verification failed), 2 (invalid input), 3 (internal error)
  - Output modes: Human-readable (default) or JSON (--json, pure stdout with canonical JSON)

- **`THOUGHT/LAB/CAT_CHAT/tests/test_compression_validator.py`** (400+ lines)
  - 5 test functions:
    1. `test_estimate_tokens()` - Token estimation function
    2. `test_compression_verify_passes_on_matching_claim()` - Pass case with matching metrics
    3. `test_compression_verify_fails_on_metric_mismatch()` - Fail on metric mismatch
    4. `test_compression_verify_fails_if_not_strictly_verified()` - Fail on missing receipts
    5. `test_compression_outputs_deterministic()` - Deterministic JSON output
  - Uses existing fixtures from test bundle

**— Multi-Validator Aggregation**

- **Multi-validator attestations for quorum validation** (RECEIPT + MERKLE)
  - `SCHEMAS/receipt.schema.json` - added optional `attestations[]` array
  - `SCHEMAS/execution_policy.schema.json` - added `receipt_attestation_quorum` and `merkle_attestation_quorum`
  - `SCHEMAS/aggregated_merkle_attestations.schema.json` - new schema for aggregated attestations
  - `catalytic_chat/attestation.py` - added `verify_receipt_attestation_single()` and `verify_receipt_attestations_with_quorum()`
  - `catalytic_chat/merkle_attestation.py` - added `load_aggregated_merkle_attestations()` and `verify_merkle_attestations_with_quorum()`
  - `tests/test_multi_validator_attestations.py` - comprehensive tests for ordering, quorum, backward compatibility
  - Deterministic ordering by (validator_id, public_key.lower(), build_id or "")
  - Reuses existing trust policy (strict_trust, strict_identity)
  - Purely additive: single attestation path unchanged

**— External Verifier UX Improvements**

- **CI-friendly output modes and machine-readable summaries**
  - `catalytic_chat/cli_output.py` - new module with standardized exit codes and JSON output helpers
  - `catalytic_chat/cli.py` - added `--json` and `--quiet` flags to `bundle verify`, `bundle run`, `trust verify`
  - `catalytic_chat/__main__.py` - entry point for `python -m catalytic_chat.cli`
  - `tests/test_cli_output.py` - tests for JSON stdout purity, exit code classification, deterministic outputs
  - Standardized exit codes: 0 (OK), 1 (verification failed), 2 (invalid input), 3 (internal error)
  - JSON output uses `canonical_json_bytes()` for deterministic field ordering
  - No verification behavior changes; purely additive UX improvements

### Fixed

**— Receipt Index Determinism (Redo)**

- **Executor-derived receipt_index with no caller control**
  - `catalytic_chat/executor.py` - removed `receipt_index` parameter from `__init__()`
  - `catalytic_chat/executor.py` - removed `_find_next_receipt_index()` (no filesystem scanning)
  - `catalytic_chat/executor.py` - always assigns `receipt_index = 0` deterministically
  - `tests/test_receipt_index_propagation.py` - updated tests to verify receipt_index=0 with no caller control
  - receipt_index is executor-derived, no filesystem scanning, pure execution order function

**— Receipt Index Propagation**

- **Deterministic receipt_index assignment with strict verification**
  - `catalytic_chat/executor.py` - assign receipt_index deterministically (no filesystem scanning)
  - `catalytic_chat/receipt.py` - hardened `verify_receipt_chain()` to enforce contiguity, start at 0, strictly increasing
  - `tests/test_receipt_index_propagation.py` - tests for contiguous index enforcement, gap detection, nonzero start, mixed null/int
  - All receipt_index rules are fail-closed; indices must be exactly [0, 1, 2, ..., n-1] when present

**— Receipt Chain Ordering Hardening**

- **Deterministic, fail-closed receipt chain ordering**
  - `SCHEMAS/receipt.schema.json` - added `receipt_index` field (integer | null) for explicit sequential ordering
  - `catalytic_chat/receipt.py` - `find_receipt_chain()` enhanced:
    - Explicit ordering key with priority: receipt_index > receipt_hash > filename
    - Duplicate receipt_index detection (raises ValueError)
    - Mixed receipt_index/null detection (raises ValueError)
    - Duplicate receipt_hash detection (raises ValueError)
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()` enhanced:
    - Strict monotonic validation: receipt_index must be strictly increasing
    - Mixed receipt_index/null detection (raises ValueError)
    - All receipts must have receipt_index set or all must be null
  - `catalytic_chat/receipt.py` - `receipt_signed_bytes()` implemented:
    - Extracts identity fields from attestation
    - Builds signing stub with scheme, public_key, validator_id, build_id
    - Ensures identity fields are included in signed message
  - `catalytic_chat/executor.py` - added `"receipt_index": None` to receipt creation
  - `tests/test_receipt_chain_ordering.py` - 5 deterministic ordering tests

### Changed

**6.12**

- `catalytic_chat/receipt.py` - `build_receipt_from_bundle()`:
  - Added `"receipt_index": None` to receipt dict
- `catalytic_chat/receipt.py` - `find_receipt_chain()`:
  - Removed reliance on filesystem `sorted()` of glob results
  - Uses explicit ordering function for deterministic sorting
- `catalytic_chat/receipt.py` - `compute_merkle_root()`:
  - No longer performs internal re-sorting (caller responsibility)
  - Maintains existing behavior for consumption of ordered input

### Security

- Fail-closed behavior: duplicate receipt_index raises ValueError
- Fail-closed behavior: mixed receipt_index/null raises ValueError
- Strict monotonicity: receipt_index sequence must be strictly increasing
- Filesystem independence: order determined by receipt_index, not creation order

**Tests**

- 118 passed, 13 skipped (full test suite)
- All 5 new receipt chain ordering tests passing
- All existing receipt chain tests passing

**— Stabilization**

- **Eliminated execution policy test flakiness; hardened determinism guarantees**
  - `tests/test_execution_policy.py`: Fixed test result capture to use return value from `executor.execute()` instead of accessing non-existent `executor.result` attribute; wrapped failing tests in `pytest.raises()` for proper error detection.
  - Determinism audit of executor and policy enforcement: Confirmed no unordered iteration, no filesystem ordering dependence, no environment leakage. All policy checks use explicit ordering (sorted keys, ordered lists from JSON).
  - Zero flaky tests confirmed: All 113 tests pass deterministically across runs.

**— Policy Gate**

- **Centralized execution policy enforcement in executor**
  - `catalytic_chat/executor.py`: Added `policy` parameter to `BundleExecutor.__init__`, added `_enforce_policy_after_execution()` method with deterministic enforcement order (verify bundle → execute steps → verify chain → verify receipt attestation → verify merkle attestation).
  - `cli.py`: Refactored `cmd_bundle_run()` to compile exactly one policy dict (from file or CLI args), removed duplicate trust policy loading logic, fixed `cmd_trust_show()` to iterate over correct index structure.
  - `tests/test_execution_policy.py`: Added comprehensive policy enforcement tests (trust policy requirement, missing attestation, missing merkle attestation, full stack, CLI back-compatibility).

**Sanity Check Fixes**

- **Critical bug fixes for attestation and merkle verification**
  - `catalytic_chat/attestation.py`: Fixed SyntaxError (duplicate code at lines 280-293), fixed verification logic to reconstruct signing stub correctly
  - `catalytic_chat/merkle_attestation.py`: Fixed verification message exactness (added VID, BID, PK fields), corrected signing key length validation (64/128 hex chars for 32/64 bytes)
  - `tests/test_trust_identity_patch.py`: Fixed test to use mismatched validator_id correctly

**— Validator Identity Pinning + Trust Policy**

- **Complete deterministic trust policy system for CAT_CHAT**
  - `SCHEMAS/trust_policy.schema.json`: Schema enforcing `policy_version="1.0.0"`, 64-char hex public keys, ed25519 scheme, scopes limited to RECEIPT/MERKLE.
  - `catalytic_chat/trust_policy.py`: Trust policy loader and verifier with deterministic indexing and uniqueness enforcement.
  - `catalytic_chat/attestation.py`: Added `verify_receipt_attestation()` with strict trust checking.
  - `catalytic_chat/merkle_attestation.py`: Added `verify_merkle_attestation_with_trust()` with strict trust checking.
  - `cli.py`: Added `trust {init,verify,show}` commands and strict trust flags to `bundle run`.
  - `tests/test_trust_policy.py`: Comprehensive tests for trust policy, receipt/merkle strict trust, and CLI integration.
  - `CORTEX/_generated/TRUST_POLICY.json`: Default empty trust policy.


## [0.6.21] - 2025-12-30


### Fixed

**— Attestation Stabilization**

- **Fixed test_cli_dry_run subprocess import issue**
  - `tests/test_planner.py` - added 3 lines:
    - Import `os`
    - Copy environment variables and set `PYTHONPATH` to `Path(__file__).parent.parent`
    - Pass `env` parameter to `subprocess.run()`

### Changed

**Verification**

- All 59 tests now pass (previously 58 passed, 1 failed)
- No new skips added (same 13 skips)
- Deterministic receipts unchanged
- Attestation still fail-closed
- No CLI or executor regressions


## [0.6.6] - 2025-12-30


### Added

**— Validator Identity Pinning + Trust Policy**

- **Deterministic, governed trust policy that pins which validator public keys are allowed**
  - `SCHEMAS/trust_policy.schema.json` - JSON schema for trust policy
    - Enforces `policy_version="1.0.0"` (const)
    - Defines `allow` array of pinned validator entries
    - Each entry requires:
      - `validator_id`: stable human label
      - `public_key`: 64-char hex (pattern `^[0-9a-fA-F]{64}$`)
      - `schemes`: array containing `"ed25519"`
      - `scope`: array of `"RECEIPT"` and/or `"MERKLE"`
      - `enabled`: boolean
    - `additionalProperties: false` everywhere

  - `catalytic_chat/trust_policy.py` - Trust policy loader + verifier:
    - `load_trust_policy_bytes(path)`: Reads exact bytes, fails if missing
    - `parse_trust_policy(policy_bytes)`: Validates against schema using jsonschema
    - `build_trust_index(policy)`: Returns deterministic index mapping lowercase public_key → entry
      - Enforces uniqueness of `validator_id` and `public_key` (case-insensitive)
      - Raises `TrustPolicyError` on duplicates
    - `is_key_allowed(index, public_key_hex, scope, scheme)`: Checks if key is allowed for scope
    - `Default path`: `THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/TRUST_POLICY.json`
    - `CLI override`: `--trust-policy <path>` supports absolute paths (never embedded in outputs)

- **Receipt Attestation Strict Trust**
  - `catalytic_chat/attestation.py` - New function:
    ```python
    verify_receipt_attestation(receipt: dict, trust_index: Optional[dict], strict: bool) -> None
    ```
  - **Rules:**
    - If `receipt["attestation"]` is `null`/`None`: Always OK (no trust needed)
    - If attestation exists:
      - Always validate signature correctness (existing behavior preserved)
      - If `strict == True`:
        - `trust_index` MUST be provided
        - Attesting `public_key` MUST be pinned with scope including `"RECEIPT"`
        - Fail-closed with `AttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned
      - If `strict == False`: Signature validity only (no trust policy required)

- **Merkle Attestation Strict Trust**
  - `catalytic_chat/merkle_attestation.py` - New function:
    ```python
    verify_merkle_attestation_with_trust(att: dict, merkle_root_hex: str, trust_index: Optional[dict], strict: bool) -> None
    ```
  - **Rules:**
    - Always validate signature correctness and merkle root match (existing behavior preserved)
    - If `strict == True`:
      - `trust_index` MUST be provided
      - `public_key` MUST be pinned with scope including `"MERKLE"`
      - Fail-closed with `MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned

- **CLI: Trust Commands + Strict Verification Flags**
  - `catalytic_chat/cli.py` - New command group: `python -m catalytic_chat.cli trust {init,verify,show}`:
    - `trust init`:
      - Writes starter `TRUST_POLICY.json` to default path
      - Deterministic content (no timestamps)
      - `allow: []`
      - Output: stderr `[OK] wrote TRUST_POLICY.json`, exit 0
    - `trust verify [--trust-policy <path>]`:
      - Validates policy against schema + uniqueness rules
      - Output: stderr `[OK] trust policy valid` or `[FAIL] <reason>`, exit 1
    - `trust show [--trust-policy <path>]`:
      - Prints canonical JSON summary to stdout ONLY (machine output)
      - Uses `canonical_json_bytes()` with trailing `\n`

  - `bundle run` new flags:
    - `--trust-policy <path>`: Override default policy path
    - `--strict-trust`: Enable strict trust verification (fail-closed if policy missing/invalid)
    - `--require-attestation`: Receipt attestation MUST be present or fail
    - `--require-merkle-attestation`: Merkle attestation MUST be present and valid or fail

  - **Rules:**
    - If `--strict-trust`: Load trust policy, build index; fail-closed if missing/invalid
    - If `--require-attestation`: Fail if receipt attestation absent
    - If `--require-merkle-attestation`: Fail if merkle attestation absent or invalid
    - Default behavior compatible:
      - No `--strict-trust`: Do not require policy
      - No require flags: Do not require attestations
    - Stdout purity: Machine JSON commands (e.g., `--attest-merkle` without `--merkle-attestation-out`) output ONLY JSON + `\n`; all status to stderr

  - **Invariant enforced:** Receipt attestation verification always validates signature; strict mode adds trust check without weakening existing verification

### Changed

**Tests**

- `tests/test_trust_policy.py` - Comprehensive test coverage:
  1. `test_trust_policy_schema_and_uniqueness`:
    - Valid empty `allow` policy passes
    - Duplicate `public_key` (case-insensitive) → verify fails
    - Duplicate `validator_id` → verify fails
  2. `test_receipt_attestation_strict_trust_blocks_unknown_key`:
    - Generate receipt + attestation using SigningKey
    - Build trust policy without that pubkey
    - `verify_receipt_attestation(strict=True)` → `UNTRUSTED_VALIDATOR_KEY`
    - `verify_receipt_attestation(strict=False)` → passes (signature valid)
  3. `test_receipt_attestation_strict_trust_allows_pinned_key`:
    - Same as above but trust policy includes pubkey with `RECEIPT` scope
    - `verify_receipt_attestation(strict=True)` → passes
  4. `test_merkle_attestation_strict_trust_blocks_unknown_key_and_allows_pinned_key`:
    - Generate merkle root + sign it
    - Verify strict fails without pin, passes with pin and `MERKLE` scope
  5. `test_cli_trust_verify`:
    - Use `subprocess.run` to call `trust verify --trust-policy <tmpfile>`
    - Assert exit codes
  6. `test_cli_trust_show`:
    - Verify stdout JSON structure and counts

**Verification Results**

- Full test suite:
  - 84 passed, 13 skipped in 6.45s
- Trust policy tests:
  - 6 passed in 0.83s

**Invariants Enforced**

- **Determinism**: All modules - Identical inputs + same trust policy → identical results
- **Fail-closed**: All verification paths - Unknown keys fail when strict trust enabled
- **Schema validation**: `trust_policy.py` - Policy must conform to JSON schema
- **Uniqueness**: `trust_policy.py` - No duplicate `validator_id` or `public_key` (case-insensitive)
- **No existing verification weakened**: `attestation.py`, `merkle_attestation.py` - Signature validation always occurs; trust check is additional
- **Stdout purity**: `cli.py` - Machine JSON outputs ONLY JSON + `\n`; status to stderr
- **Path traversal defense**: `cli.py`, `trust_policy.py` - No escaping intended directories when loading policies
- **No timestamps/randomness**: All modules - Deterministic behavior, no environment-dependent data
- **Default compatibility**: `cli.py` - No `--strict-trust` = no policy required; No require flags = no attestations required
- **Minimal diffs**: All changes - Localized to CAT_CHAT, preserved existing behavior


## [0.6.5] - 2025-12-30


### Added

**— Signed Merkle Attestation**

- **Ed25519 signing and verification of receipt chain Merkle root**
  - `catalytic_chat/merkle_attestation.py` module with `sign_merkle_root()` and `verify_merkle_attestation()`
  - `SCHEMAS/merkle_attestation.schema.json` - JSON schema for merkle attestations
    - Strict hex length validation: merkle_root (64), public_key (64), signature (128)
    - `scheme` const `"ed25519"` with `additionalProperties: false`
  - CLI flags for `bundle run`:
    - `--attest-merkle`: sign merkle root after chain verification
    - `--merkle-key <hex>`: Ed25519 signing key (64 hex chars)
    - `--verify-merkle-attestation <path>`: verify attestation against computed root
    - `--merkle-attestation-out <path>`: write attestation to file
  - Strict stdout purity: `--attest-merkle` prints ONLY JSON + \n when no file output specified
  - `--print-merkle` prints ONLY merkle_root hex + \n to stdout
  - Fail-closed verification: chain tamper, ordering violation, root mismatch, signature failure => non-zero exit

### Changed

****

- `catalytic_chat/cli.py` - extended `cmd_bundle_run()` with merkle attestation support
  - Added `verbose_output` flag to suppress status messages when JSON-only output required
  - All error messages now use `sys.stderr.write()` for consistency
  - Mutually exclusive flag enforcement for stdout purity

**Tests**

- `tests/test_merkle_attestation.py` - 12 comprehensive tests
  - Sign/verify roundtrip
  - Modified root detection
  - Input validation (hex, lengths, scheme)
  - Deterministic output across multiple runs
  - Mismatch detection
  - File I/O roundtrip

### Security

****

- Message to sign: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes` (decoded 32 bytes)
- Hex decoding uses `bytes.fromhex()` - NOT ASCII bytes of hex string
- Canonical JSON output using `receipt.canonical_json_bytes()` (sort_keys, separators, UTF-8, \n EOF)


## [0.6.4] - 2025-12-30


### Added

**— Receipt Merkle Root**

- **Deterministic Merkle root computation for receipt chains**
  - `catalytic_chat/receipt.py` - `compute_merkle_root()`:
    - Pairwise concatenate hex-decoded bytes (left||right)
    - SHA256 on concatenated bytes
    - Duplicate last node at each level if odd count
    - Preserve deterministic ordering (no re-sorting)
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()` now returns Merkle root string
  - `catalytic_chat/cli.py` - `--print-merkle` flag:
    - Requires `--verify-chain` (fail-closed)
    - Print ONLY Merkle root hex to stdout when set
    - Suppress all other output when `--print-merkle` is set

### Changed

**Tests**

- `tests/test_merkle_root.py` - 3 Merkle root tests:
  - `test_merkle_root_deterministic`: same hashes produce identical Merkle root
  - `test_merkle_root_changes_on_tamper`: tampering changes root
  - `test_merkle_root_requires_verify_chain`: `--print-merkle` without `--verify-chain` fails


## [0.6.3] - 2025-12-30


### Added

**— Receipt Chain Anchoring**

- **Deterministic receipt chaining with parent_receipt_hash linkage**
  - `catalytic_chat/receipt.py` - `compute_receipt_hash()`: deterministic hash from canonical bytes (excluding receipt_hash field)
  - `catalytic_chat/receipt.py` - `load_receipt()`: load receipt from JSON file
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()`: verify chain linkage and receipt hashes
  - `catalytic_chat/receipt.py` - `find_receipt_chain()`: find all receipts for a run in execution order
  - `catalytic_chat/executor.py` - added `previous_receipt` parameter to `__init__()`:
    - Sets `parent_receipt_hash` from previous receipt's `receipt_hash`
    - First receipt has `parent_receipt_hash=null`
  - `SCHEMAS/receipt.schema.json` - added chain fields:
    - `parent_receipt_hash`: string | null
    - `receipt_hash`: string
  - `catalytic_chat/cli.py` - added `--verify-chain` flag to `bundle run`:
    - Verifies full receipt chain for a run
    - Outputs chain status and Merkle root

### Changed

**Tests**

- `tests/test_receipt_chain.py` - 4 chain tests:
  - Deterministic chain verification
  - Chain break detection
  - Sequential order enforcement


## [0.6.2] - 2025-12-30


### Added

**— Receipt Attestation**

- **Ed25519 signing and verification of receipts**
  - `catalytic_chat/attestation.py` - `sign_receipt_bytes()`, `verify_receipt_bytes()`
  - `SCHEMAS/receipt.schema.json` - added optional `attestation` object
  - `catalytic_chat/executor.py` - enhanced to support signing:
    - Added `attestation_override` parameter
    - Signs receipt with `attestation=None` (null in signed scope)
  - `catalytic_chat/receipt.py` - `receipt_canonical_bytes()`: single source of truth for receipt canonicalization
    - Used by signer, verifier, and executor
  - `catalytic_chat/cli.py` - added `--attest` flag:
    - `bundle run` now supports `--attest` (requires `--signing-key`)
    - `--verify-attestation` flag for verification
  - `tests/test_attestation.py` - 6 attestation tests:
    - All tests pass
    - Fixed tamper test to verify canonical byte differences

### Security

****

- Single source of truth for canonicalization
- Signing input is canonical receipt bytes with `attestation=null/None`
- Verifying recomputes exact same canonical bytes
- Hex-only for `public_key`/`signature` with validation
- No timestamps, randomness, absolute paths, or env-dependent behavior
- Minimal diffs; changes localized to canonicalization and signing flow


## [0.5.0] - 2025-12-29


### Added

**— Bundle (Translation Protocol) MVP**

- **Deterministic, bounded, fail-closed bundle construction and verification**
  - `catalytic_chat/bundle.py` - Bundle builder with deterministic `bundle_id` computation
  - `catalytic_chat/bundle_verifier.py` - Verifier with schema validation and boundedness enforcement
  - `SCHEMAS/bundle.schema.json` - Bundle schema with `additionalProperties: false` everywhere

- **Session 1 — Objective Definition & Contract Specification**
  - **Hard requirements:** Deterministic output (ordering, hashes, formatting), bounded content inclusion (only referenced artifacts and exact slices), fail-closed validation on all integrity/rules violations
  - **Environment specification:** Windows PowerShell with `$env:PYTHONPATH` set
  - **Bundle contract specification:** Definition of "completed job", job completeness gate, exact slice usage, canonical serialization rules

- **Session 2 — Bundle Algorithm (Deterministic, Bounded, Fail-Closed)**
  - **bundle_id algorithm:** Pre-manifest SHA-256 of `plan_hash + ":" + session_id + ":" + plan.json`
  - **Deterministic ordering:** Steps ordered by `(ordinal asc, step_id asc, artifact_id asc)` with stable sort
  - **Bounded content inclusion:** Only artifacts referenced by plan steps, only exact slices used
  - **Fail-closed verification:** Schema validation, hash mismatch detection, ordering violation detection, forbidden slice/ALL detection
  - **Canonical serialization:** UTF-8 encoding, `\n` separator, `sort_keys=True`, `separators=(",", ":")`, single trailing newline
  - **Job completeness gate:** Exactly one receipt per step, all steps committed

- **Session 3 — Artifact Structure (Fixed, Not Underspecified)**
  - `artifact_id`: "<stable deterministic id>"
  - `kind`: "SYMBOL_SLICE" or "FILE"
  - `ref`: `@SYMBOL_ID` or `section_id/path`
  - `slice_used`: exact slice string (for SYMBOL_SLICE)
  - `path`: `section_id/path` (relative, no absolute paths)
  - `sha256`: hex digest of artifact content
  - `bytes`: size in bytes
  - Arrays: `inputs`, `artifacts`, `hashes` ordered lexicographically

- **Session 4 — Bundle Output Layout (Minimal, Canonical)**
  - `bundle.json`: Main bundle file at root with all metadata
  - Artifacts: `artifacts/` directory with `<artifact_id>.txt` files
  - No nested directories, no mixed types
  - All paths relative (no `../../` or absolute Windows paths)

- **Files created:**
  - `catalytic_chat/bundle.py` (299 lines) - BundleBuilder class
  - `catalytic_chat/bundle_verifier.py` (425 lines) - BundleVerifier class
  - `SCHEMAS/bundle.schema.json` (67 lines) - Bundle schema
  - Test fixtures and comprehensive test coverage

- **Key invariants enforced:**
  - **Determinism:** Identical inputs → identical bundle_id, artifact hashes, and bundle.json
  - **Boundedness:** Only referenced artifacts included, only exact slices used
  - **Fail-closed:** Schema validation, hash verification, ordering checks, ALL sentinel rejection
  - **Purity:** No timestamps, no randomness, no environment-dependent behavior
  - **No absolute paths:** All paths are relative from bundle root
  - **Canonical JSON:** Stable sort, no trailing spaces, single trailing newline

**— Deterministic Job Completeness Gate**

- **Job completeness definition**: A job is "complete" if and only if:
  - All steps are COMMITTED (no PENDING or LEASED)
  - Exactly one receipt exists per step
  - Receipts are immutable (append-only in database)

- **Implementation:**
  - `catalytic_chat/bundle.py` - Added job completeness check in bundle building
  - `catalytic_chat/bundle_verifier.py` - Added job completeness verification
  - **Rule:** `BundleError("JOB_NOT_COMPLETE")` if any step is not committed or has multiple/zero receipts

- **Files created:**
  - Job completeness gate with detailed error messages
  - Updated tests for job completeness validation

**— Boundedness Enforcement with Slice Validation**

- **Slice validation framework** in bundle_verifier.py:
  - Reject `ALL` slices (case-insensitive)
  - Validate slice format: `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)`
  - Error types: `INVALID_SLICE_FORMAT`, `SLICE_OUT_OF_BOUNDS`, `FORBIDDEN_ALL_SLICE`
  - Deterministic slice computation with exact character positions

- **Files created/modified:**
  - `catalytic_chat/bundle.py` - Enhanced with slice parsing and validation
  - `catalytic_chat/bundle_verifier.py` - Added slice validation
  - Comprehensive test coverage for all slice types and error cases

**— Multi-Plan Coordination**

- **Multi-plan bundling support** for scenarios with multiple planners:
  - `--source-bundle <path>`: Bundle A (source of steps)
  - `--target-bundle <path>`: Bundle B (dependent steps)
  - Deterministic resolution: Build target bundle with all steps, extract its artifacts, verify target bundle
  - `--merge-strategy`: `reuse`, `rebuild` (no copying duplicates)

- **Implementation:**
  - BundleBuilder supports source/target bundles
  - BundleVerifier verifies dependency chains
  - Multi-plan artifact validation and cross-bundle verification

- **Files created:**
  - Enhanced CLI with `--source-bundle`, `--target-bundle`, `--merge-strategy`
  - Multi-bundle test fixtures

**— CLI Commands for Bundle Operations**

- **New CLI commands:**
  - `bundle build --request-file <json> [--run-id <id>]` - Build bundle from plan
  - `bundle verify <bundle> [--strict]` - Verify bundle integrity
  - `bundle list <bundle>` - List artifacts in bundle
  - `bundle extract <bundle> [--out-dir <dir>]` - Extract artifacts

- **Files created/modified:**
  - `catalytic_chat/cli.py` - Extended CLI with bundle subcommands
  - `SCHEMAS/bundle.schema.json` - Bundle schema
  - Comprehensive test coverage for all CLI commands

**— Canonical JSON Writer for Deterministic Bundles**

- **Canonical JSON serialization:**
  - UTF-8 encoding
  - Sort keys lexicographically
  - `separators=(",", ":")` (no spaces)
  - Single trailing newline
  - No floating point issues (Decimal, float)

- **Files created:**
  - `catalytic_chat/canonical_json.py` (new module)
  - Integration with BundleBuilder and BundleVerifier
  - Schema validation and deterministic output

### Changed

**Tests**

- **`tests/test_bundle.py`** - Comprehensive test suite (200+ lines)
- Tests cover:
  - Deterministic bundle_id computation
  - Artifact ordering
  - Schema validation
  - Boundedness enforcement
  - Slice validation
  - Job completeness gates
  - Multi-plan coordination
  - Canonical JSON serialization
  - CLI commands

- **Test results:** All tests passing


## [0.4.0] - 2025-12-29


### Added

**(Deterministic Planner)**

- **`THOUGHT/LAB/CAT_CHAT/PHASE_4_LAW.md`** (390 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_request.schema.json`** (88 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_step.schema.json`** (159 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_output.schema.json`** (113 lines)
- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/planner.py`** (427 lines)
- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/message_cassette.py`** (modified: integrated with planner)
- **`THOUGHT/LAB/CAT_CHAT/tests/test_planner.py`** (443 lines)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json`** (minimal)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_files.json`** (with file reference)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_steps_exceeded.json`** (max_steps exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_bytes_exceeded.json`** (max_bytes exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_symbols_exceeded.json`** (max_symbols exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_slice_all_forbidden.json`** (slice=ALL forbidden)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_invalid_symbol.json`** (invalid symbol_id)

### Changed

**Features**

- **Deterministic Compiler**: Request -> Plan (steps with stable IDs and ordering)
- **Budget Enforcement**: max_steps, max_bytes, max_symbols (fail-closed)
- **Symbol Bounds**: slice=ALL forbidden, uses default_slice
- **Integration**: post_request_and_plan() stores request + plan in cassette
- **Idempotency**: Same (run_id, idempotency_key) returns same job_id/steps
- **CLI Commands**: `cortex plan --request-file <json> [--dry-run]`, `cassette plan-verify`

**Tests**

- `test_plan_determinism_same_request_same_output` ✅
- `test_plan_determinism_step_ids_stable` ✅
- `test_plan_rejects_too_many_steps` ✅
- `test_plan_rejects_too_many_bytes` ✅
- `test_plan_rejects_too_many_symbols` ✅
- `test_plan_rejects_slice_all_forbidden` ✅
- `test_plan_rejects_invalid_symbol_id` ✅
- `test_plan_idempotency_same_idempotency_key` ✅
- `test_plan_dry_run_does_not_touch_db` ✅
- `test_plan_verify_matches_stored_hash` ✅
- `test_plan_verify_fails_on_mismatch` ✅

**Verified**

- **Tests**: All 31 tests passing
- **CLI Verify**: `python -m catalytic_chat.cli plan --request-file tests/fixtures/plan_request_min.json --dry-run` → valid plan


## [0.3.0] - 2025-12-29


### Added

**— Message Cassette (DB-first Schema & API)**

- **DB Schema (CORTEX/_generated/system3.db): Message cassette with append-only enforcement**
  - `catalytic_chat/message_cassette_db.py` - SQLite database schema:
    - `cassette_messages`: append-only, run_id, source, payload_json
    - `cassette_jobs`: links to messages, ordinal for ordering
    - `cassette_steps`: status (PENDING/LEASED/COMMITTED), lease_owner, lease_expires_at, fencing_token
    - `cassette_receipts`: append-only, links to step/job

  - **DB-Level Enforcement (triggers):**
    - Append-only on messages and receipts (UPDATE/DELETE blocked)
    - FSM enforcement: only PENDING→LEASED→COMMITTED allowed
    - Lease protection: direct lease changes blocked without proper claim
    - Foreign keys: receipts→steps→jobs→messages integrity

- **API (message_cassette.py): Message cassette with deterministic job/step management**
  - `post_message()`: create message/job/step, idempotency support
  - `claim_step()`: deterministic selection (oldest job, lowest ordinal)
  - `complete_step()`: verify lease/token/expiry before completion
  - `verify_cassette()`: check DB state and invariants

- **Tests:**
  - `tests/test_message_cassette.py` - Full test coverage for append-only, FSM enforcement, lease protection, referential integrity, deterministic claim ordering

**— Message Cassette CLI Commands**

- **New CLI commands for cassette management**
  - `catalytic_chat/cli.py` - Modified to add cassette commands:
    - `cortex cassette verify --run-id <id>`: Verify cassette state
    - `cortex cassette post --json <file> --run-id <id> --source <src>`: Post message to cassette
    - `cortex cassette claim --run-id <id> --worker <id> [--ttl <s>]`: Claim a step
    - `cortex cassette complete --run-id <id> --step <id> --worker <id> --token <n> --receipt <file> --outcome <out>`: Complete a step

  - **No changes to existing commands** (build, verify, get, extract, symbols, resolve)

**— Message Cassette Hardening (Adversarial Tests & Verify Strengthening)**

- **Adversarial tests (test_message_cassette.py):**
  - `test_steps_delete_allowed_when_persisting_design`: Confirm steps are NOT append-only (design intent)
  - `test_messages_update_delete_blocked_by_triggers`: Direct SQL UPDATE/DELETE blocked by triggers
  - `test_receipts_update_delete_blocked_by_triggers`: Direct SQL UPDATE/DELETE blocked by triggers
  - `test_illegal_fsm_transition_blocked`: Direct SQL FSM jumps (PENDING→COMMITTED, COMMITTED→LEASED) blocked
  - `test_lease_direct_set_blocked`: Direct SQL lease changes without claim blocked
  - `test_complete_fails_on_stale_token_even_if_lease_owner_matches`: Completion with stale token fails
  - `test_complete_fails_on_expired_lease`: Completion with expired lease fails

- **verify_cassette() Strengthening:**
  - Check PRAGMA foreign_keys is ON
  - Check required tables exist
  - Check all 8 required triggers exist by exact name
  - Return exit code 0 on PASS, non-zero on FAIL

- **CLI Output:**
  - `verify`: "PASS: All invariants verified" to stderr on success
  - `verify`: "FAIL: N issue(s) found" to stderr on failure

### Changed

**Files**

- `tests/test_message_cassette.py` - Added 8 adversarial tests
- `catalytic_chat/message_cassette.py` - Strengthened verify_cassette()
- `catalytic_chat/cli.py` - Cleaned verify output, added exit codes

**Tests**

- 8 adversarial tests (all passing)
- Full test coverage for DB schema, FSM enforcement, lease protection
- Referential integrity and deterministic ordering verified


## [0.2.5] - 2025-12-29


### Added

**— Experimental Vector Sandbox**

- **SQLite-backed vector store for local experiments**
  - `catalytic_chat/experimental/__init__.py` (new)
  - `catalytic_chat/experimental/vector_store.py` (new)
  - `tests/test_vector_store.py` (new)
  - `docs/catalytic-chat/notes/VECTOR_SANDBOX.md` (new)

- **Vector Store Features:**
  - `catalytic_chat/experimental/vector_store.py` - Vector tables and API:
    - `vector_id`, `namespace`, `content_hash`, `dims`, `vector_json`, `meta_json`, `created_at`
    - `put_vector()`: Store vector with metadata
    - `get_vector()`: Retrieve vector by ID
    - `query_topk()`: Top-k semantic search using cosine similarity
    - Pure Python implementation (no extensions)

- **API Details:**
  - SQLite-backed vector tables for local experiments
  - Cosine similarity computation in pure Python
  - Namespace isolation for multi-tenant experiments

- **Tests:**
  - `tests/test_vector_store.py` - Full test coverage:
    - `test_put_and_get_vector` - Store and retrieve
    - `test_query_returns_ordered_results` - Top-k ordering
    - `test_query_with_dim_mismatch` - Dimension validation
    - `test_namespace_isolation` - Multi-tenant separation
    - `test_vector_json_serialization` - JSON handling
    - `test_created_at_field` - Timestamp tracking

- **Documentation:**
  - `docs/catalytic-chat/notes/VECTOR_SANDBOX.md` - Vector sandbox documentation


## [0.2.1] - 2025-12-29


### Added

**— Symbol Registry**

- **SYMBOLS artifact mapping** with `@Symbol` → `section_id` references
  - Dual substrate support: SQLite (`CORTEX/_generated/system1.db`) + JSONL fallback
  - Namespace conventions: `@CANON/`, `@CONTRACTS/`, `@TOOLS/`, etc.
  - `symbol_id` field: TEXT, PRIMARY KEY, `@` prefix validated
  - `target_ref` field: target_type + target_id (e.g., SECTION:abc123)
  - Schema validation on insert

- **Files created/modified:**
  - `catalytic_chat/symbol_registry.py` (SymbolRegistry class with dual substrate)
  - `SCHEMAS/symbol.schema.json` (symbol registry schema)
  - `SCHEMAS/section_index.schema.json` (section index schema)
  - `catalytic_chat/__init__.py` (exports SymbolRegistry)
  - `catalytic_chat/symbol_indexer.py` (indexer with dual substrate)
  - `tests/test_symbol_registry.py` (comprehensive tests)
  - `tests/fixtures/section_index.fixture.json` (test data)

- **Features implemented:**
  - SQLite primary with JSONL fallback
  - `register()` - Add symbol with metadata
  - `get_by_id()` - Retrieve symbol by ID
  - `list()` - Query symbols with filters
  - `update()` - Modify symbol metadata
  - `delete()` - Remove symbol
  - Indexer integration: Auto-index on register/update
  - Full schema validation
  - Namespace isolation by prefix
  - Target type validation (SECTION, FILE, etc.)

- **Tests (25/25 passing):**
  - All CRUD operations tested
  - Dual substrate failover tested
  - Schema validation verified
  - Indexer integration tested
  - Edge cases covered

**— Symbol Resolution + Expansion Cache**


- **Bounded resolver API** with deterministic slice enforcement
  - `resolve(@Symbol, slice)` → `(payload, cache_hit)` return
  - Slice forms: `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)`
  - **slice=ALL** explicitly denied (fail-closed)
  - `default_slice` parameter for convenience
  - Deterministic ordering: cache_miss → JSONL → SQLite (fallback)

- **Expansion cache** per-run reuse
  - Cache key: `(run_id, symbol_id, slice, content_hash)`
  - Cache TTL: Configurable (default 1 hour)
  - SQLite storage with `expansion_cache` table
  - JSONL fallback for non-SQLite environments

- **Files created:**
  - `catalytic_chat/symbol_resolver.py` (SymbolResolver class)
  - `catalytic_chat/expansion_cache.py` (ExpansionCache class)
  - `SCHEMAS/expansion_cache.schema.json` (cache schema)
  - `tests/test_symbol_resolver.py` (comprehensive tests)
  - `tests/fixtures/expansion_cache.fixture.json` (test data)

- **Features implemented:**
  - `resolve()` with slice parsing and caching
  - `cache_invalidate()` - Clear cache for run_id
  - `cache_get_stats()` - Cache statistics
  - Multiple slice forms with validation
  - Fail-closed on `slice=ALL`
  - Deterministic ordering guarantees

- **Tests (20/20 passing):**
  - All slice forms tested
  - Cache hit/miss verified
  - TTL expiration tested
  - Deterministic ordering verified
  - Edge cases covered

### Changed

**& 2.2 Integration**

- `catalytic_chat/__init__.py` exports both SymbolRegistry and SymbolResolver
- Symbol registry and resolution fully integrated and tested together

**Tests**

- 45 tests passing (25 for symbol registry + 20 for symbol resolution)
- Full integration tests for registry + resolver + cache
- Failover tests for dual substrate


## [0.1.0] - 2025-12-29


### Added

**— Substrate + Deterministic Indexing**

- **Sections and slices** - Content is addressable as sections with explicit slices
  - `catalytic_chat/section_extractor.py` - Extract sections from markdown files
  - `catalytic_chat/section_indexer.py` - Index sections for fast lookup
  - `catalytic_chat/slice_resolver.py` - Parse slice expressions (lines, chars, head, tail)
  - Default slice policy and boundedness enforcement

- **Section indexing** - Deterministic section discovery and storage
  - `CORTEX/_generated/section_index.db` - SQLite database for section metadata
  - `CORTEX/_generated/SECTION_INDEX.json` - JSONL export for compatibility
  - Section ID format: stable, deterministic generation
  - Hash-based deduplication to prevent duplicate sections

- **Files created/modified:**
  - `catalytic_chat/paths.py` - Path helpers for deterministic DB locations
  - `SCHEMAS/section.schema.json` - Section metadata schema
  - `SCHEMAS/section_index.schema.json` - Section index schema
  - `tests/test_section_indexer.py` - Tests for section indexing
  - `tests/test_slice_resolver.py` - Tests for slice parsing

- **Features implemented:**
  - Section extraction from markdown files
  - Deterministic section ID generation
  - Hash-based content tracking
  - Slice parsing with boundedness checks
  - Section listing and lookup by ID
  - Default slice enforcement
  - Fail-closed validation on invalid inputs

- **Tests (32/32 passing):**
  - Section extraction from markdown
  - Section ID generation determinism
  - Slice parsing (lines, chars, head, tail)
  - Boundedness enforcement (reject out-of-bounds)
  - Default slice application
  - Section indexing and lookup
  - Hash-based deduplication
  - Path resolution for DB locations

**— CLI Commands for Substrate**

- **Section operations:**
  - `cortex build` - Index all sections
  - `cortex get <section_id>` - Get section by ID
  - `cortex extract <file> --section <id> --slice <expr>` - Extract content
  - `cortex list --prefix <prefix>` - List sections by prefix

- **Slice resolver validation:**
  - `cortex resolve-slice <expr>` - Validate slice expression syntax
  - Supports: `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)`

- **Files created:**
  - `catalytic_chat/cli.py` - CLI implementation with all commands
  - `tests/test_cli.py` - CLI command tests

- **Tests (24/24 passing):**
  - Build command indexes all sections
  - Get command retrieves by ID
  - Extract command applies slices correctly
  - List command filters by prefix
  - Resolve-slice validates all slice types
  - Error handling and exit codes


## [0.0.1] - 2025-12-29


### Added


### Changed

**Freeze scope and interfaces**


- Freeze vocabulary: Section, Symbol, Message, Expansion, Receipt, Bundle.
- Freeze determinism requirements and fail-closed error policy.
- Define budgets and the no-unbounded-expansion invariant.


## [0.0.0] - 2025-12-29


### Changed

**Deprecated / Legacy**

- **Previous Claude Code triple-write implementation** (misaligned with roadmap)
  - `legacy/chat_db.py` - Database for Claude Code messages
  - `legacy/embedding_engine.py` - Vector embeddings for chat messages
  - `legacy/message_writer.py` - Triple-write to DB + JSONL + MD
  - `catalytic-chat-research.md` (archived) - Claude Code research
  - `catalytic-chat-phase1-implementation-report.md` (archived) - Old phase report
  - `archive/catalytic-chat-roadmap.md` (archived) - Old roadmap
  - `legacy/SYMBOLIC_README.md` - Symbol encoding (different from roadmap "Symbol" concept)

**Note**: Legacy files preserved in `archive/` directory but are not aligned with current CAT_CHAT architecture. See `archive/LEGACY_NOTES.md` for consolidated legacy documentation.

**Legacy Implementation**

- **Core Database** (`legacy/chat_db.py`)
  - SQLite database with 4 tables: `chat_messages`, `message_chunks`, `message_vectors`, `message_fts`
  - Hash-based deduplication (SHA-256)
  - Transaction support with context managers
  - Migration system for schema versioning
  - Foreign key constraints for data integrity

- **Embedding Engine** (`legacy/embedding_engine.py`)
  - Vector embeddings using `all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing for efficiency
  - Cosine similarity computation
  - BLOB serialization for SQLite storage

- **Message Writer** (`legacy/message_writer.py`)
  - Triple-write strategy: DB (primary) + JSONL (mechanical) + MD (human)
  - Atomic writes - all three must succeed or none
  - Automatic chunking of long messages (500 tokens per chunk)
  - Embedding generation for all chunks
  - JSONL export in Claude Code format
  - MD export with human-readable formatting

**Testing (Legacy)**

- **Unit Tests** (`legacy/tests/test_chat_system.py`)
  - 44 tests across 3 test classes
  - Test coverage: Database, Embedding Engine, Message Writer
  - All tests passing

**Documentation (Legacy)**

- Implementation Report: `catalytic-chat-phase1-implementation-report.md` (archived)
- Research: `catalytic-chat-research.md` (archived)
- Roadmap: `catalytic-chat-roadmap.md` (archived)
- ADR: `CONTEXT/decisions/ADR-031-catalytic-chat-triple-write.md` (if exists)

**Legacy Symbolic Encoding (Not Implemented in Current System)**

- **Symbolic Chat Encoding** (archived in `legacy/SYMBOLIC_README.md`)
  - Token savings of 30-70% through symbol compression
  - Symbol dictionary with governance/technical terms
  - Auto-encoding of English to symbols on write
  - Auto-decoding of symbols to English on read
  - `simple_symbolic_demo.py` - working demo with 62.5% token savings

**Note**: This symbolic encoding is experimental and NOT part of current CAT_CHAT system. Current CAT_CHAT uses "Symbol" concept differently (symbol registry referencing canonical sections).
