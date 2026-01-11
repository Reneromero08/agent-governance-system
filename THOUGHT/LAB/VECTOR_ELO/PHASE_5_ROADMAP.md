---
title: Phase 5 Vector/Symbol Integration Detailed Roadmap
section: roadmap
version: 1.8.0
created: 2026-01-07
modified: 2026-01-08
status: Active
summary: Comprehensive roadmap for current Phase 5 tasks (SCL, SPC formalization)
tags:
- phase-5
- vector
- semiotic
- roadmap
---
<!-- CONTENT_HASH: 76a7f194b6cdfbe129be2ffdb30c164a2cffbac768886d825380b62e5e7ae41d -->

# Phase 5: Vector/Symbol Integration - Detailed Roadmap

## Overview

Phase 5 enables semantic addressability through two complementary systems:
- **5.1** Vector embeddings for semantic retrieval
- **5.2** Symbolic compression for token efficiency

**Goal:** Make governance artifacts addressable by meaning, not just by path or hash.

---

## Global Definition of Done

Every task must produce:
- [ ] All relevant tests pass
- [ ] Receipts emitted (inputs, outputs, hashes, commands, exit status)
- [ ] Human-readable report (what changed, why, how verified)
- [ ] Scope respected (explicit allowlists for writes)

---

# Phase 5.1: Vector Indexing & Semantic Discovery ✅

# Phase 5.2: Semiotic Compression Layer (SCL)

## 5.2.1 Define MVP Macro Set ✅ COMPLETE (2026-01-08)

## 5.2.2 Semiotic Symbol Vocabulary ✅ COMPLETE (2026-01-08)

## 5.2.3 Implement Stacked Symbol Resolution ✅

## 5.2.4 Implement SCL Validator ✅

## 5.2.5 Implement SCL CLI ✅

**Purpose:** Command-line interface for SCL operations.

### 5.2.5.1 CLI Commands ✅ COMPLETE (2026-01-09)
- [x] Create `CAPABILITY/TOOLS/scl/scl_cli.py`
- [x] Commands:
  - `scl decode <program>` → emit JobSpec JSON
  - `scl validate <program|job.json>` → PASS/FAIL
  - `scl run <program>` → execute with invariant proofs
  - `scl audit <program>` → human-readable expansion
- [x] Receipt emission for all operations (`--receipt-out`)
- [x] JSON output mode (`--json`)
- [x] CJK compound symbol support (法.驗)

### 5.2.5.2 Integration ✅ COMPLETE (2026-01-09)
- [x] Emit receipts for all operations
- [x] Register as CLI tool (`python -m CAPABILITY.TOOLS.scl`)

### 5.2.5.3 Tests ✅ COMPLETE (2026-01-09)
- [x] CLI invocation tests (45 tests)
- [x] Output format validation
- [x] Error handling

**Exit Criteria:** ✅ ALL MET
- [x] `scl` CLI functional with all commands
- [x] Receipts emitted

---

## 5.2.6 SCL Tests & Benchmarks ✅ COMPLETE (2026-01-09)

**Purpose:** Prove determinism and measure token reduction.

### 5.2.6.1 Determinism Tests ✅
- [x] Create `CAPABILITY/TESTBENCH/integration/test_phase_5_2_semiotic_compression.py`
- [x] Same program → same JSON hash (100 runs)
- [x] Same program + same codebook version → same output

### 5.2.6.2 Schema Validation Tests ✅
- [x] Expanded JobSpecs validate against schema
- [x] Invalid inputs produce schema errors

### 5.2.6.3 Token Benchmark ✅
- [x] Measure: tokens for symbolic program vs expanded text
- [x] Target: 80%+ reduction for governance boilerplate (achieved 96.4%)
- [x] Create benchmark fixture with representative programs

### 5.2.6.4 Negative Tests ✅
- [x] Invalid syntax → clear error
- [x] Unknown symbol → clear error
- [x] Circular expansion → error (if possible)

### 5.2.6.5 L2 Compression Proof Script (Stacked Receipt) ✅
- [x] Create `run_scl_proof.py` following L1 pattern
- [x] Inputs: L1 receipt hash, governance text samples
- [x] Measure with tiktoken: natural language → symbolic IR tokens
- [x] Emit receipt that chains to L1:
  ```json
  {
    "layer": "SCL",
    "parent_receipt": "325410258180d609...",
    "input_tokens": 334,
    "output_tokens": 12,
    "compression_pct": 96.4
  }
  ```
- [x] Include negative controls (garbage input → low compression)
- [x] Deterministic: same input → same receipt hash

**Exit Criteria:** ✅ ALL MET
- [x] 20+ tests passing (28 tests)
- [x] Token reduction benchmark documented (SCL_PROOF_REPORT.md)
- [x] **L2 receipt chains to L1 receipt** (stacked proof)
- [x] All error classes covered

---

## 5.2.7 Token Accountability Layer ✅

**Purpose:** Make token savings mandatory and visible in every operation.

**Reference:** `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md`

### 5.2.7.1 TokenReceipt Schema ✅ COMPLETE (2026-01-10)
- [x] Create `CAPABILITY/PRIMITIVES/schemas/token_receipt.schema.json`
- [x] Required fields:
  - `schema_version` (string): Schema version for evolution (const: "1.0.0")
  - `operation` (string): Operation type enum
  - `tokens_out` (integer): Output tokens
  - `tokenizer` (object): library, encoding, version, fallback_used
- [x] Optional fields (core):
  - `tokens_in`, `baseline_equiv`, `tokens_saved`, `savings_pct`
  - `corpus_anchor`, `operation_id`, `timestamp_utc`
- [x] Optional fields (hardening):
  - `session_id` (string): Session ID for firewall aggregation
  - `receipt_hash` (string): SHA-256 of receipt for integrity/chain linking
  - `parent_receipt_hash` (string|null): SHA-256 for receipt chaining
  - `baseline_method` (enum): How baseline was calculated
  - `determinism_proof` (object): git_head, git_clean, methodology_hash
  - `query_metadata` (object): Operation-specific data (query_hash, results_count, etc.)

### 5.2.7.2 TokenReceipt Primitive ✅ COMPLETE (2026-01-10)
- [x] Create `CAPABILITY/PRIMITIVES/token_receipt.py`
- [x] Implement `TokenReceipt` dataclass
- [x] Auto-compute `tokens_saved` and `savings_pct` from baseline
- [x] Generate unique `operation_id` hash
- [x] Compute `receipt_hash` for integrity/chaining
- [x] Helper functions: `get_default_tokenizer()`, `count_tokens()`, `hash_query()`
- [x] Schema validation: `validate_receipt()`

### 5.2.7.3 Patch Semantic Search ✅ COMPLETE (2026-01-10)
- [x] Update `NAVIGATION/CORTEX/semantic/semantic_search.py`
- [x] Emit TokenReceipt on every `search()` call
- [x] Include baseline_equiv (sum of corpus tokens)
- [x] Return receipt alongside results via `SearchResponse` dataclass
- [x] Added `_get_corpus_tokens()` for baseline calculation
- [x] Added `_get_corpus_anchor()` for reproducibility
- [x] Backwards-compatible: `SearchResponse` is iterable like `List[SearchResult]`

### 5.2.7.4 Require TokenReceipt in JOBSPEC ✅ COMPLETE (2026-01-10)
- [x] Update JobSpec schema to include optional `token_receipt` field
- [x] SCL decoder emits TokenReceipt on decode (`scl_cli.py`)
- [x] Skill executor aggregation deferred to 5.2.7.5 Session Aggregator

### 5.2.7.5 Session Aggregator ✅ COMPLETE (2026-01-10)
- [x] Create `CAPABILITY/PRIMITIVES/token_session.py`
- [x] Aggregate all receipts per session
- [x] Compute cumulative savings
- [x] Emit `SessionTokenSummary` at session end
- [x] Global session manager: `get_current_session()`, `start_new_session()`, `end_current_session()`, `log_receipt()`
- [x] Ledger export: `to_ledger()` for firewall logging
- [x] Display formats: `compact()`, `verbose()`, `to_json()`

### 5.2.7.6 Firewall Enforcement ✅ COMPLETE (2026-01-10)
- [x] Add firewall rule: REJECT outputs > 1000 tokens without TokenReceipt (REJECT-001)
- [x] Add firewall rule: WARN if savings_pct < 50% for semantic_query (WARN-001)
- [x] Log all receipts to session ledger (LOG-001, auto_log_receipts)
- [x] Create `CAPABILITY/PRIMITIVES/token_firewall.py`
- [x] Convenience functions: `validate_token_output()`, `require_token_receipt()`
- [x] Violation logging and audit trail

### 5.2.7.7 Display Formats ✅ COMPLETE (2026-01-10)
- [x] Compact format for CLI: `[TOKEN] op: N tokens (saved M / P%)`
- [x] Verbose format for reports (multi-line)
- [x] JSON export for machine processing
- [x] Implemented as methods in `TokenReceipt`: `compact()`, `verbose()`, `to_json()`

### 5.2.7.8 Tests ✅ COMPLETE (2026-01-10)
- [x] `test_phase_5_2_7_token_accountability.py` — 21 tests
- [x] TokenReceipt schema validation (7 tests)
- [x] Session aggregation (4 tests)
- [x] Firewall rules enforced (6 tests)
- [x] Semantic search receipt structure (2 tests)
- [x] JobSpec integration (2 tests)

**Exit Criteria:** ✅ ALL MET
- [x] Every semantic_query emits TokenReceipt (SearchResponse.receipt)
- [x] Session summaries show cumulative savings (SessionTokenSummary)
- [x] Firewall rejects unreceipted large outputs (REJECT-001)
- [x] 21 tests passing (exceeds 10+ requirement)

---

# Phase 5.3: SPC Formalization & Research Publication

**Status:** PENDING (execute after 5.1 and 5.2 complete)
**Purpose:** Formalize Semantic Pointer Compression (SPC) as a defensible research contribution
**Source:** GPT execution pack (`OPUS_SPC_RESEARCH_CLAIM_EXECUTION_PACK.md`)

> **Design principle:** The better the brief, the better the design. This phase creates formal specs and reproducible benchmarks for publication, building on the working implementation from 5.1/5.2.

---

## 5.3.0 Definition

**SPC (Semantic Pointer Compression):** Conditional compression with shared side-information.
- Sender transmits a pointer (symbol or hash) plus required sync metadata
- Receiver expands deterministically into a canonical IR subtree
- Expansion is accepted only if hashes and versions verify. Otherwise fail closed.

**Key insight:** This is CAS at the semantic layer. Symbol = semantic hash. Codebook = semantic CAS index.

---

## 5.3.1 SPC_SPEC.md (Normative) ✅ COMPLETE (2026-01-11)

**Purpose:** Formal specification for Semantic Pointer Compression protocol.

### Deliverables
- [x] Create `LAW/CANON/SEMANTIC/SPC_SPEC.md`

### Contents Required
- [x] **Pointer Types:**
  - `SYMBOL_PTR`: Single-token glyph pointers (法, 真, 道)
  - `HASH_PTR`: Content-addressed pointers (SHA-256)
  - `COMPOSITE_PTR`: Pointer plus typed qualifiers (法.驗, C3:build)
- [x] **Decoder Contract:**
  - Inputs: pointer, context keys, codebook_id, codebook_sha256, kernel_version, tokenizer_id
  - Output: canonical IR subtree OR FAIL_CLOSED with explicit error code
- [x] **Ambiguity Rules:**
  - Multiple expansions possible → reject unless disambiguation is explicit and deterministic
- [x] **Canonical Normalization:**
  - `encode(decode(x))` stabilizes to declared normal form
- [x] **Security & Drift Behavior:**
  - Codebook mismatch → reject
  - Hash mismatch → reject
  - Unknown symbol → reject
  - Unknown kernel version → reject
- [x] **Measured Metrics:**
  - `concept_unit` definition (ties to GOV_IR_SPEC)
  - `CDR = concept_units / tokens`
  - `ECR = exact IR match rate`
  - `M_required = multiplex factor for target nines`

**Exit Criteria:** ✅ ALL MET
- [x] SPC_SPEC.md is normative and complete
- [x] All pointer types defined with examples
- [x] Fail-closed behavior specified for all error cases

---

## 5.3.2 GOV_IR_SPEC.md (Normative) ✅ COMPLETE (2026-01-11)

**Purpose:** Define minimal typed governance IR so "meaning" is countable.

### Deliverables
- [x] Create `LAW/CANON/SEMANTIC/GOV_IR_SPEC.md`
- [x] Create `LAW/SCHEMAS/gov_ir.schema.json`

### Contents Required
- [x] **IR Primitives:**
  - Boolean ops (AND, OR, NOT, XOR, IMPLIES)
  - Comparisons (EQ, NE, LT, LE, GT, GE)
  - Typed references (paths, canon versions, tool ids, artifact hashes, rule/invariant ids)
  - Gates (test, restore_proof, allowlist_check, hash_verify, schema_validate)
  - Side-effects flags (writes, deletes, creates, modifies_canon, requires_ceremony, emits_receipt)
- [x] **Canonical JSON Schema:**
  - Stable ordering (alphabetical Unicode code point)
  - Explicit types (no coercion)
  - Canonical string forms (UTF-8, NFC, LF, no trailing whitespace)
- [x] **Equality Definition:**
  - Equality = byte-identical canonical JSON
  - `ir_equal(a, b) ≡ canonical_json(a) == canonical_json(b)`
- [x] **concept_unit Definition:**
  - Atomic unit of governance meaning
  - Types: constraint (1), permission (1), prohibition (1), reference (1), gate (1)
  - Counting rules for operations, sequences, records
  - CDR = concept_units / tokens

**Exit Criteria:** ✅ ALL MET
- [x] GOV_IR_SPEC.md defines complete typed IR (9 node types)
- [x] JSON schema provided and validated (`gov_ir.schema.json`)
- [x] concept_unit is measurable (counting function defined)

---

## 5.3.3 CODEBOOK_SYNC_PROTOCOL.md (Normative) ✅ COMPLETE (2026-01-11)

**Purpose:** Define how sender and receiver establish shared side-information.

> **Note:** This protocol is the formalization of Phase 6 Cassette Network's sync mechanism. The cassette network IS the implementation of this protocol.

### Deliverables
- [x] Create `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md`

### Contents Required
- [x] **Sync Handshake:**
  - `codebook_id` + `sha256` + `semver`
  - `semantic_kernel_version`
  - `tokenizer_id`
- [x] **Compatibility Policy:**
  - Default: exact match required
  - Optional: explicit compatibility ranges with migration step (never silent)
- [x] **Handshake Message Shape:**
  - Request format (SyncRequest)
  - Response format (SyncResponse)
  - Failure codes (17 enumerated)
- [x] **Integration with Cassette Network:**
  - How cassettes carry codebook state (sync_tuple in handshake)
  - Verification before symbol expansion (blanket_status check)
- [x] **Markov Blanket Semantics:**
  - Theoretical foundation (Q35 integration)
  - Blanket alignment = R > τ
  - Active Inference interpretation
- [x] **Information-Theoretic Semantics:**
  - Conditional entropy H(X|S) (Q33 integration)
  - Semantic density CDR = σ^Df
  - Measurement procedure

**Exit Criteria:** ✅ ALL MET
- [x] Sync protocol fully specified (12 sections)
- [x] Handshake message shapes defined (SyncRequest, SyncResponse, SyncError, Heartbeat)
- [x] Failure codes enumerated (17 codes across 3 categories)

---

## 5.3.4 TOKENIZER_ATLAS.json (Artifact)

**Purpose:** Formal artifact tracking glyph/operator token counts across tokenizers.

### Deliverables
- [ ] Create `CAPABILITY/TOOLS/generate_tokenizer_atlas.py`
- [ ] Create `LAW/CANON/SEMANTIC/TOKENIZER_ATLAS.json`

### Contents Required
- [ ] **Atlas Generator Script:**
  - Token counts for all semantic symbols under declared tokenizers
  - Deterministic ranking: prefer single-token glyphs, stable fallback
- [ ] **Atlas Artifact:**
  - Symbol → token_count mapping for cl100k_base
  - Symbol → token_count mapping for o200k_base
  - Preferred glyph list with fallbacks
- [ ] **CI Gate:**
  - Test that fails if preferred glyph becomes multi-token after tokenizer change

**Exit Criteria:**
- [ ] TOKENIZER_ATLAS.json generated and receipted
- [ ] All 7 current symbols verified single-token
- [ ] CI gate added to prevent silent tokenizer drift

---

## 5.3.5 Proof Harness: proof_spc_semantic_density_run/

**Purpose:** Reproducible benchmark suite with receipted measurements.

### Deliverables
- [ ] Create `CAPABILITY/TESTBENCH/proof_spc_semantic_density_run/`
  - [ ] `benchmark_cases.json` - 10-30 fixed test cases
  - [ ] `run_benchmark.py` - Deterministic proof runner
  - [ ] `metrics.json` - Machine-readable output
  - [ ] `report.md` - Human-readable output
  - [ ] `receipts/` - SHA-256 of all inputs/outputs

### Benchmark Case Structure
```json
{
  "id": "case_001",
  "nl_statement": "All writes to canon require verification receipt",
  "gold_ir": { "type": "constraint", "op": "requires", ... },
  "pointer_encoding": "法.驗",
  "expected_tokens_nl": 12,
  "expected_tokens_pointer": 3
}
```

### Measurements Required
- [ ] `tokens(NL)` under declared tokenizer
- [ ] `tokens(pointer_payload)`
- [ ] `concept_units(IR)`
- [ ] `ECR` (exact match rate)
- [ ] Reject rate and reasons
- [ ] Computed `M_required` for declared targets

### Hard Acceptance Criteria
- [ ] **A1 Determinism:** Two consecutive runs produce byte-identical outputs
- [ ] **A2 Fail-closed:** Any mismatch emits explicit failure artifacts
- [ ] **A3 Measured density:** CDR and ECR computed and output
- [ ] **A4 No hallucinated paths:** All file paths exist in repo

**Exit Criteria:**
- [ ] Benchmark suite runs end-to-end
- [ ] All 4 acceptance criteria pass
- [ ] Receipts generated for reproducibility

---

## 5.3.6 PAPER_SPC.md (Research Skeleton)

**Purpose:** Publishable research claim skeleton with measured results only.

### Deliverables
- [ ] Create `THOUGHT/LAB/VECTOR_ELO/research/PAPER_SPC.md`

### Required Sections
- [ ] **Title & Abstract**
- [ ] **Contributions:**
  - Deterministic semantic pointers
  - Receipted verification
  - Measured semantic density metric and benchmark
- [ ] **What Is New:**
  - Not "beating Shannon" - conditional compression with shared side-information
  - Formal protocol for LLM context optimization
  - Measured H(X|S) vs H(X)
- [ ] **Threat Model:**
  - Codebook drift
  - Tokenizer changes
  - Semantic ambiguity
- [ ] **Limitations:**
  - Requires shared context establishment
  - Single-token symbols depend on tokenizer stability
  - Compression ratio depends on corpus size
- [ ] **Reproducibility:**
  - Exact commands to run benchmark
  - Hashes of all artifacts
  - Environment requirements

**Exit Criteria:**
- [ ] Paper skeleton complete with all sections
- [ ] No claims without metrics
- [ ] Reproducibility section includes exact commands

---

## Phase 5.3 Complete When:

- [x] SPC_SPEC.md normative and complete (5.3.1)
- [x] GOV_IR_SPEC.md with typed IR and JSON schema (5.3.2)
- [x] CODEBOOK_SYNC_PROTOCOL.md with handshake defined (5.3.3)
- [ ] TOKENIZER_ATLAS.json generated with CI gate
- [ ] Proof harness passes all 4 acceptance criteria
- [ ] PAPER_SPC.md ready for external review

**Publication Milestone:** After 5.3, SPC is a defensible research contribution with:
- Formal specs others can implement
- Reproducible benchmarks with receipts
- Clear claims grounded in measurements

---
