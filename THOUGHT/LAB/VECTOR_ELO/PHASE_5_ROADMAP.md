---
title: Phase 5 Vector/Symbol Integration Detailed Roadmap
section: roadmap
version: 1.8.0
created: 2026-01-07
modified: 2026-01-08
status: Active
summary: Comprehensive implementation roadmap for Phase 5
tags:
- phase-5
- vector
- semiotic
- roadmap
---
<!-- CONTENT_HASH: 7e61cd536fd30b163f4c43ad44a0ed1ac922b06cf7d8dae7a68f69c7bab4cdb9 -->

# Phase 5: Vector/Symbol Integration - Detailed Roadmap

## BREAKTHROUGH: Semantic Symbol Compression (2026-01-08)

**Discovery:** Single-token CJK symbols achieve **56,370x compression** when receiver has referenced content in shared context.

| Symbol | Tokens | Expands To | Ratio | Status |
|--------|--------|------------|-------|--------|
| 法 | 1 | All canon (56,370 tokens) | **56,370x** | PROVEN |
| 真 | 1 | Semiotic Foundation (1,455 tokens) | **1,455x** | PROVEN |
| 道 | 1 | Context-activated (4 meanings) | **24x** | PROVEN |

**Proof:** `NAVIGATION/PROOFS/COMPRESSION/SEMANTIC_SYMBOL_PROOF_REPORT.md`

**Implementation:** `CAPABILITY/TOOLS/codebook_lookup.py` (MCP tool: `codebook_lookup`)

**Stacking Insight:**
- L1 only (法): 56,370x but dumps everything (low alignment)
- L1+L2 (法.query): 843x but returns only relevant chunks (high alignment)
- Formula: `density = shared_context ^ alignment`

**Impact on Roadmap:** Phase 5.2.3 (SCL Decoder) simplified. Stack codebook_lookup + CORTEX FTS instead of building complex AST decoder.

---

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

## 5.1.0 Foundation: MemoryRecord Contract ✅

**Purpose:** Define the canonical data structure for all vector-indexed content.

### 5.1.0.1 Define MemoryRecord JSON Schema
- [x] Create `LAW/SCHEMAS/memory_record.schema.json` (**DONE** 2026-01-08)
- [x] Required fields:
  - `id` (string): Content hash (SHA-256)
  - `text` (string): Canonical text content
  - `embeddings` (object): Model name → vector array
  - `payload` (object): Metadata (tags, timestamps, roles, doc_ids)
  - `scores` (object): ELO, recency, trust, decay
  - `lineage` (object): Derivation chain, summarization history
  - `receipts` (object): Provenance hashes, tool versions
- [x] Implement `additionalProperties: false` at all levels

### 5.1.0.2 Implement MemoryRecord Primitive
- [x] Create `CAPABILITY/PRIMITIVES/memory_record.py` (**DONE** 2026-01-08)
- [x] Functions:
  - `create_record(text, metadata) -> MemoryRecord`
  - `validate_record(record) -> verdict`
  - `hash_record(record) -> content_hash`
  - `add_embedding(record, name, vector, model) -> record`
  - `to_json(record) -> str` / `from_json(str) -> record`
  - `canonical_bytes(record) -> bytes`
  - `full_hash(record) -> str`
- [x] Contract rules:
  - Text is canonical (source of truth)
  - Vectors are derived (rebuildable from text)
  - All exports are receipted and hashed

### 5.1.0.3 Tests
- [x] `CAPABILITY/TESTBENCH/core/test_memory_record.py` (**DONE** 2026-01-08)
- [x] Fixtures: valid records, invalid records, edge cases
- [x] Determinism: same input → same hash
- [x] **23 tests passing**

**Exit Criteria:**
- [x] MemoryRecord schema validated and documented
- [x] Create/validate/hash functions working
- [x] 23 tests passing (exceeds 10+ requirement)

---

## 5.1.1 Embed Canon Files ✅ COMPLETE (2026-01-08)

**Purpose:** Make all governance canon semantically searchable.

### 5.1.1.1 Canon File Inventory
- [x] Enumerate all files in `LAW/CANON/*` (**DONE** - 32 files)
- [x] Create manifest with file paths and content hashes (**DONE**)
- [x] Emit inventory receipt (**DONE**)

### 5.1.1.2 Embedding Generation
- [x] Select embedding model: `all-MiniLM-L6-v2` (ADR-030)
  - Uses sentence-transformers (local, deterministic)
  - 384 dimensions, float32
- [x] Implement `embed_text(text, model) -> vector` (**DONE** via EmbeddingEngine)
- [x] Batch embed all canon files (**DONE** - 32 files embedded)
- [x] Store as MemoryRecord instances (**DONE**)

### 5.1.1.3 Vector Index Storage
- [x] Create vector index structure: SQLite-based `canon_index.db` (**DONE**)
- [x] Store embeddings with content hashes as IDs (**DONE**)
- [x] Enable rebuild from source files (deterministic) (**DONE**)

### 5.1.1.4 Tests
- [x] `test_phase_5_1_1_canon_embedding.py` (**DONE** - 23 tests)
- [x] Verify all canon files embedded (**DONE**)
- [x] Verify rebuild produces identical index (**DONE**)
- [x] Verify similarity search returns expected results (**DONE**)

**Exit Criteria:**
- [x] All `LAW/CANON/*` files embedded (32 files)
- [x] Index rebuildable deterministically
- [x] Similarity search functional

**Implementation:**
- `CAPABILITY/PRIMITIVES/canon_index.py` - Core indexing primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_1_canon_embedding.py` - 23 tests
- `NAVIGATION/CORTEX/db/canon_index.db` - Embedded index

---

## 5.1.2 Embed ADRs ✅ COMPLETE (2026-01-08)

**Purpose:** Make architecture decisions semantically searchable.

### 5.1.2.1 ADR Inventory
- [x] Enumerate all files in `LAW/CONTEXT/decisions/*` (**DONE** - 37 ADRs)
- [x] Parse ADR metadata (title, status, date) (**DONE** - YAML frontmatter)
- [x] Create manifest with hashes (**DONE**)

### 5.1.2.2 ADR Embedding
- [x] Batch embed all ADR files (**DONE** - 37 files embedded)
- [x] Include metadata in MemoryRecord payload (**DONE** - id, title, status, date, etc.)
- [x] Store with cross-references to canon (**DONE** - 184 cross-references)

### 5.1.2.3 Tests
- [x] `test_phase_5_1_2_adr_embedding.py` (**DONE** - 28 tests passing)
- [x] Verify all ADRs embedded (**DONE**)
- [x] Verify metadata preserved (**DONE**)
- [x] Verify semantic search returns relevant ADRs (**DONE**)

**Exit Criteria:**
- [x] All ADRs embedded with metadata (37 ADRs, 31 Accepted, 4 Proposed, 2 Unknown)
- [x] Searchable by semantic query

**Implementation:**
- `CAPABILITY/PRIMITIVES/adr_index.py` - ADR indexing primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_2_adr_embedding.py` - 28 tests
- `NAVIGATION/CORTEX/db/adr_index.db` - Embedded index with cross-references

---

## 5.1.3 Vector-Indexed CAS for Model Weights ✅ COMPLETE (2026-01-08)

**Purpose:** Store model artifacts with semantic addressability.

### 5.1.3.1 Model Weight Registry
- [x] Define schema for model weight records (**DONE** - ModelRecord TypedDict)
- [x] Include: model name, version, hash, embedding of description (**DONE**)
- [x] Store in CAS with vector index pointer (**DONE** - weights_hash field)

**ModelRecord Schema:**
```python
ModelRecord = {
    "id": str,           # SHA-256(name@version) - deterministic
    "name": str,         # Model name (e.g., "all-MiniLM-L6-v2")
    "version": str,      # Version string
    "description": str,  # For semantic search (embeddable)
    "format": str,       # pytorch, safetensors, onnx, etc.
    "weights_hash": str, # SHA-256 of weights (CAS reference)
    "size_bytes": int,   # Model size
    "embedding": bytes,  # Description embedding (384 dims)
    "metadata": dict,    # architecture, params, license, etc.
    "source": str,       # huggingface, local, custom
    "created_at": str,
    "updated_at": str,
}
```

### 5.1.3.2 Implementation
- [x] `register_model(name, version, description, format, ...) -> result` (**DONE**)
- [x] `get_model(name, version) -> ModelRecord` (**DONE**)
- [x] `get_model_by_weights_hash(hash) -> ModelRecord` (**DONE** - deduplication)
- [x] `search_models(query, top_k) -> [results]` (**DONE** - semantic search)
- [x] `list_models(format_filter) -> [models]` (**DONE**)
- [x] `get_registry_stats() -> stats` (**DONE**)
- [x] `verify_registry() -> verification_result` (**DONE**)

### 5.1.3.3 Tests
- [x] `test_phase_5_1_3_model_registry.py` (**DONE** - 28 tests passing)
- [x] Model ID determinism (name@version -> hash) (**DONE**)
- [x] Model registration and retrieval (**DONE**)
- [x] Semantic search by model description (**DONE**)
- [x] Deduplication by weights hash (**DONE**)
- [x] Registry statistics and verification (**DONE**)

**Exit Criteria:**
- [x] Model weights addressable by hash and description
- [x] Semantic search functional
- [x] Registry integrity verification working

**Implementation:**
- `CAPABILITY/PRIMITIVES/model_registry.py` - Core registry primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_3_model_registry.py` - 28 tests
- `NAVIGATION/CORTEX/db/model_registry.db` - Registry database (created on first use)

---

## 5.1.4 Semantic Skill Discovery ✅ COMPLETE (2026-01-08)

**Purpose:** Find skills by description similarity.

### 5.1.4.1 Skill Inventory
- [x] Enumerate all `CAPABILITY/SKILLS/*/SKILL.md` (**DONE** - 24 skills)
- [x] Parse skill metadata (purpose, trigger, inputs, outputs) (**DONE** - YAML frontmatter + markdown)
- [x] Create skill manifest (**DONE** - with content hashes)

### 5.1.4.2 Skill Embedding
- [x] Embed skill descriptions (**DONE** - all-MiniLM-L6-v2, 384 dims)
- [x] Store with metadata in MemoryRecord-compatible format (**DONE**)
- [x] Enable semantic search (**DONE** - cosine similarity)

### 5.1.4.3 Skill Discovery API
- [x] `find_skills_by_intent(query, top_k) -> [skill_ids]` (**DONE**)
- [x] Deterministic tie-breaking for stable results (**DONE** - score desc, skill_id asc)
- [x] Emit discovery receipt (**DONE**)

### 5.1.4.4 Tests
- [x] `test_phase_5_1_4_skill_discovery.py` (**DONE** - 32 tests passing)
- [x] Known queries return expected skills (**DONE** - verified)
- [x] Results stable across runs (**DONE** - determinism tests passing)

### 5.1.4.5 MCP Integration
- [x] Added `skill_discovery` MCP tool (**DONE**)
- [x] Tool schema in `schemas/tools.json` (**DONE**)
- [x] Server handler in `server.py` (**DONE**)

**Exit Criteria:**
- [x] Skill discovery returns stable results for fixed corpus
- [x] Top-K results deterministic

**Implementation:**
- `CAPABILITY/PRIMITIVES/skill_index.py` - Core primitive (inventory, embedding, search)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_4_skill_discovery.py` - 32 tests
- `NAVIGATION/CORTEX/db/skill_index.db` - SQLite index with embeddings
- `CAPABILITY/MCP/server.py` - MCP tool handler
- `CAPABILITY/MCP/schemas/tools.json` - Tool schema

**Example Results:**
- Query: "verify canon changes and enforce changelog"
- Top Result: `governance/canon-governance-check` (similarity: 0.589)

---

## 5.1.5 Cross-Reference Indexing ✅ COMPLETE (2026-01-08)

**Purpose:** Link artifacts by embedding distance.

### 5.1.5.1 Cross-Reference Graph
- [x] Compute pairwise distances for related artifacts (**DONE**)
- [x] Store as graph edges with distance weights (**DONE** - SQLite with similarity scores)
- [x] Enable "related artifacts" queries (**DONE**)

### 5.1.5.2 Implementation
- [x] `find_related(artifact_id, threshold, top_k) -> [related_ids]` (**DONE**)
- [x] Stable ordering by distance then ID (**DONE** - similarity desc, artifact_id asc)

### 5.1.5.3 Tests
- [x] Related artifacts for known items (**DONE** - 20 tests passing)
- [x] Graph traversal correctness (**DONE** - deterministic ordering verified)

**Exit Criteria:**
- [x] Cross-reference queries functional
- [x] Results deterministic

**Implementation:**
- `CAPABILITY/PRIMITIVES/cross_ref_index.py` - Core primitive (build, query, stats)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_5_1_cross_refs.py` - 20 tests
- `NAVIGATION/CORTEX/db/cross_ref_index.db` - SQLite graph database
- `CAPABILITY/MCP/server.py` - MCP tool handler
- `CAPABILITY/MCP/schemas/tools.json` - Tool schema

**Features:**
- Unified graph across canon, ADR, and skill artifacts
- Configurable similarity threshold and top_k per artifact
- Cross-type relationships (e.g., canon files linked to related ADRs)
- Batch similarity computation using numpy
- Build history tracking with receipts

**Example Usage:**
```python
# Build cross-reference graph
result = build_cross_refs(threshold=0.3, top_k_per_artifact=10)
# Returns: {artifacts_count: N, refs_count: M, duration_seconds: X}

# Find related artifacts
related = find_related("canon:LAW/CANON/GOVERNANCE/IMMUTABILITY.md", top_k=5)
# Returns: {artifact_id, related: [{artifact_id, artifact_type, artifact_path, similarity, metadata}], total_candidates}
```

---

# Phase 5.2: Semiotic Compression Layer (SCL)

## 5.2.0 Research Integration

**Reference Documents:**
- `INBOX/2025-12/Week-01/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md`
- `INBOX/2025-12/Week-52/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md`
- `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`

---

## 5.2.1 Define MVP Macro Set ✅ COMPLETE (2026-01-08)

**Purpose:** Identify 30-80 macros covering 80% of governance repetition.

> **IMPLEMENTED (2026-01-08):** Compact macro grammar with radicals, operators, and rule lookups. 50 macros total achieving 60% token savings vs verbose @-prefix scheme.

### 5.2.1.1 Macro Categories
- [x] **Domain radicals:** C, I, V, L, G, S, R, A, J, P (10 radicals, all 1 token) (**DONE**)
- [x] **Operators:** *, !, ?, &, |, ., : (7 operators, all 1 token) (**DONE**)
- [x] **Contract rules:** C1-C13 (13 rules with summary/full expansion) (**DONE**)
- [x] **Invariants:** I1-I20 (20 invariants with INV-ID mapping) (**DONE**)

### 5.2.1.2 Macro Grammar
- [x] Grammar: `RADICAL[OPERATOR][NUMBER][:CONTEXT]` (**DONE**)
- [x] Examples: C3, I5, C*, V!, C3:build (**DONE**)
- [x] Parser in `codebook_lookup.py` (**DONE**)

### 5.2.1.3 Macro Specification Document
- [x] Updated `CODEBOOK.json` with grammar, radicals, operators (**DONE**)
- [x] Legacy mappings for migration (@DOMAIN_GOVERNANCE → G) (**DONE**)
- [x] Token metrics documented (60% savings) (**DONE**)

**Exit Criteria:**
- [x] 50 macros identified and documented (**DONE** - exceeds 30 minimum)
- [x] Token savings measured: 60% vs verbose (**DONE**)
- [x] Test suite: 21 tests passing (**DONE**)

---

## 5.2.2 Semiotic Symbol Vocabulary ✅ COMPLETE (2026-01-08)

**Purpose:** Define CJK symbols for semantic compression without phonetic mixing.

> **SIMPLIFIED (2026-01-08):** Instead of a separate CODEBOOK.json schema, symbols are defined directly in `codebook_lookup.py` with a human reference document. Phonetic glosses removed - symbols point directly to semantic regions.

### 5.2.2.1 Symbol Vocabulary
- [x] Core domain pointers: 法, 真, 契, 恆, 驗 (**DONE**)
- [x] Governance operations: 證, 變, 冊, 錄, 限, 許, 禁, 雜, 復 (**DONE**)
- [x] Validation operations: 試, 查, 載, 存, 掃, 核 (**DONE**)
- [x] Structural symbols: 道, 圖, 鏈, 根, 枝 (**DONE**)
- [x] Compound symbols: 法.驗, 法.契, 證.雜, 冊.雜 (**DONE**)

### 5.2.2.2 Codifier Document
- [x] Create `LAW/CANON/SEMANTIC/CODIFIER.md` (**DONE** - human reference)
- [x] Document symbol → path mappings (**DONE**)
- [x] Measured compression ratios (**DONE**)
- [x] Usage patterns (**DONE**)

### 5.2.2.3 Pure Symbolic Approach
- [x] Remove phonetic glosses from code (**DONE** - no "name": "law" mixing)
- [x] Symbols point directly to paths (**DONE**)
- [x] Compression ratios tracked per symbol (**DONE**)

**Exit Criteria:**
- [x] 25+ symbols defined (29 total)
- [x] No phonetic/ideographic mixing
- [x] Human reference document created

**Implementation:**
- `CAPABILITY/TOOLS/codebook_lookup.py` - Symbol definitions (SEMANTIC_SYMBOLS dict)
- `THOUGHT/LAB/FORMULA/CODIFIER.md` - Human translation reference (符典)
- `CAPABILITY/MCP/schemas/tools.json` - MCP schema updated

---

## 5.2.3 Implement Stacked Symbol Resolution ✅

**Purpose:** Stack symbol resolution with CORTEX for precision retrieval.

> **SIMPLIFIED (2026-01-08):** Instead of building a complex AST decoder, stack the working `codebook_lookup.py` with existing CORTEX FTS. Same outcome, less complexity.

### 5.2.3.1 Stacked Resolution (SIMPLIFIED PATH) ✅ COMPLETE
- [x] Create `CAPABILITY/TOOLS/codebook_lookup.py` - **DONE**
- [x] CJK single-token symbols (法, 真, 契, 驗, 恆, 道) - **DONE**
- [x] MCP integration (codebook_lookup tool) - **DONE**
- [x] Add `query` parameter for FTS within domain - **DONE**
- [x] Add `semantic` parameter for vector search within domain - **DONE**
- [x] Stack: symbol → domain narrowing → CORTEX search → relevant chunks - **DONE**

### 5.2.3.2 Stacked Query Syntax
```python
# L1 only (dumps everything)
codebook_lookup(id="法", expand=True)  # → 56,370 tokens

# L1+L2 (FTS precision)
codebook_lookup(id="法", query="verification")  # → ~4,200 tokens

# L1+L3 (semantic precision)
codebook_lookup(id="法", semantic="verification protocols")  # → ~2,000 tokens
```

### 5.2.3.3 Tests ✅ COMPLETE
- [x] `test_phase_5_2_3_stacked_resolution.py` - **5 tests passing**
- [x] Symbol only → full domain content - **DONE**
- [x] Symbol + query → FTS filtered content - **DONE**
- [x] Symbol + semantic → vector filtered content - **DONE**
- [x] Compression ratios measured and receipted - **DONE**

**Exit Criteria:**
- [x] Symbol resolution working (56,370x proven)
- [x] Stacked resolution with CORTEX integration - **DONE**
- [x] Precision retrieval (high alignment) verified - **DONE**

### Original Complex Path (DEFERRED)
The original plan for AST parsing and template expansion is deferred. If the simplified stacked approach proves insufficient, revisit:
- Parser implementation with tokenize/parse/AST
- Template variable substitution
- Nested macro expansion

**Rationale:** Follow the entropy gradient. The simpler path achieves the same compression goals with existing components.

---

## 5.2.4 Implement SCL Validator ✅

**Purpose:** Validate symbolic programs and expanded outputs.

### 5.2.4.1 Symbolic Validation ✅ COMPLETE
- [x] Create `CAPABILITY/PRIMITIVES/scl_validator.py` - **DONE**
- [x] Check: syntax valid, all symbols known, params match - **DONE**
- [x] L1 (Syntax), L2 (Symbol), L3 (Semantic) validation layers - **DONE**

### 5.2.4.2 Expansion Validation ✅ COMPLETE
- [x] Expanded JSON passes JobSpec schema - **DONE**
- [x] Outputs in allowed roots (C8/I6 enforcement) - **DONE**
- [x] No forbidden operations - **DONE**

### 5.2.4.3 Tests ✅ COMPLETE
- [x] Valid programs pass - **38 tests passing**
- [x] Invalid syntax fails with clear error - **DONE**
- [x] Unknown symbols fail - **DONE**
- [x] Schema-invalid expansions fail - **DONE**

**Exit Criteria:**
- [x] Validator catches all error classes - **DONE**
- [x] Clear error messages - **DONE**

**Implementation:**
- `CAPABILITY/PRIMITIVES/scl_validator.py` - 4-layer validator (L1-L4)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_2_4_scl_validator.py` - 38 tests

---

## 5.2.5 Implement SCL CLI

**Purpose:** Command-line interface for SCL operations.

### 5.2.5.1 CLI Commands
- [ ] Create `CAPABILITY/TOOLS/scl/scl_cli.py`
- [ ] Commands:
  - `scl decode <program>` → emit JobSpec JSON
  - `scl validate <program|job.json>` → PASS/FAIL
  - `scl run <program>` → execute with invariant proofs
  - `scl audit <program>` → human-readable expansion

### 5.2.5.2 Integration
- [ ] Register as skill or CLI tool
- [ ] Emit receipts for all operations

### 5.2.5.3 Tests
- [ ] CLI invocation tests
- [ ] Output format validation
- [ ] Error handling

**Exit Criteria:**
- [ ] `scl` CLI functional with all commands
- [ ] Receipts emitted

---

## 5.2.6 SCL Tests & Benchmarks

**Purpose:** Prove determinism and measure token reduction.

### 5.2.6.1 Determinism Tests
- [ ] Create `CAPABILITY/TESTBENCH/integration/test_phase_5_2_semiotic_compression.py`
- [ ] Same program → same JSON hash (100 runs)
- [ ] Same program + same codebook version → same output

### 5.2.6.2 Schema Validation Tests
- [ ] Expanded JobSpecs validate against schema
- [ ] Invalid inputs produce schema errors

### 5.2.6.3 Token Benchmark
- [ ] Measure: tokens for symbolic program vs expanded text
- [ ] Target: 80%+ reduction for governance boilerplate
- [ ] Create benchmark fixture with representative programs

### 5.2.6.4 Negative Tests
- [ ] Invalid syntax → clear error
- [ ] Unknown symbol → clear error
- [ ] Circular expansion → error (if possible)

### 5.2.6.5 L2 Compression Proof Script (Stacked Receipt)
- [ ] Create `run_scl_proof.py` following L1 pattern
- [ ] Inputs: L1 receipt hash, governance text samples
- [ ] Measure with tiktoken: natural language → symbolic IR tokens
- [ ] Emit receipt that chains to L1:
  ```json
  {
    "layer": "SCL",
    "parent_receipt": "<L1_hash>",
    "input_tokens": <from L1 result>,
    "output_tokens": <measured>,
    "compression_pct": <calculated>
  }
  ```
- [ ] Include negative controls (garbage input → low compression)
- [ ] Deterministic: same input → same receipt hash

**Exit Criteria:**
- [ ] 20+ tests passing
- [ ] Token reduction benchmark documented
- [ ] **L2 receipt chains to L1 receipt** (stacked proof)
- [ ] All error classes covered

---

## 5.2.7 Token Accountability Layer

**Purpose:** Make token savings mandatory and visible in every operation.

**Reference:** `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md`

### 5.2.7.1 TokenReceipt Schema
- [ ] Create `CAPABILITY/PRIMITIVES/schemas/token_receipt.schema.json`
- [ ] Required fields:
  - `operation` (string): Operation type
  - `tokens_out` (integer): Output tokens
  - `tokenizer` (object): library, encoding, version
- [ ] Optional fields:
  - `tokens_in`, `baseline_equiv`, `tokens_saved`, `savings_pct`
  - `corpus_anchor`, `operation_id`, `timestamp_utc`

### 5.2.7.2 TokenReceipt Primitive
- [ ] Create `CAPABILITY/PRIMITIVES/token_receipt.py`
- [ ] Implement `TokenReceipt` dataclass
- [ ] Auto-compute `tokens_saved` and `savings_pct` from baseline
- [ ] Generate unique `operation_id` hash

### 5.2.7.3 Patch Semantic Search
- [ ] Update `NAVIGATION/CORTEX/semantic/semantic_search.py`
- [ ] Emit TokenReceipt on every `search()` call
- [ ] Include baseline_equiv (sum of corpus tokens)
- [ ] Return receipt alongside results

### 5.2.7.4 Require TokenReceipt in JOBSPEC
- [ ] Update JobSpec schema to include optional `token_receipt` field
- [ ] SCL decoder emits TokenReceipt on decode
- [ ] Skill executor aggregates receipts

### 5.2.7.5 Session Aggregator
- [ ] Create `CAPABILITY/PRIMITIVES/token_session.py`
- [ ] Aggregate all receipts per session
- [ ] Compute cumulative savings
- [ ] Emit `SessionTokenSummary` at session end

### 5.2.7.6 Firewall Enforcement
- [ ] Add firewall rule: REJECT outputs > 1000 tokens without TokenReceipt
- [ ] Add firewall rule: WARN if savings_pct < 50% for semantic_query
- [ ] Log all receipts to session ledger

### 5.2.7.7 Display Formats
- [ ] Compact format for CLI: `[TOKEN] op: N tokens (saved M / P%)`
- [ ] Verbose format for reports (multi-line)
- [ ] JSON export for machine processing

### 5.2.7.8 Tests
- [ ] `test_phase_5_2_7_token_accountability.py`
- [ ] TokenReceipt schema validation
- [ ] Semantic search emits receipt
- [ ] Session aggregation correct
- [ ] Firewall rules enforced

### 5.2.7.9 Empirical Metric Receipt (R/J/Phi)

**Purpose:** Make empirical claims auditable: record **R / J / Phi-proxy** alongside the receipts produced by tests and benchmarks.

- [ ] Define `EmpiricalMetricReceipt` schema (separate from TokenReceipt)
- [ ] Required fields:
  - `benchmark_id`, `seed`, `dataset`, `mode`
  - `R` / `M=log(R)` summary stats
  - `J` (neighbor-fitness): neighbor similarity, competitor construction mode
  - `phi_proxy_bits` (lightweight proxy; not exact IIT Phi)
  - `gates`: pass/fail + thresholds used
- [ ] Efficiency constraints:
  - Use lightweight proxies first (MI / multi-information via binning)
  - Add a “stress mode” pass-rate gate for variability when `scifact_stream_seed=-1`
- [ ] Datatrail:
  - Emit verbatim logs + SHA256 under `LAW/CONTRACTS/_runs/...`
  - Link to the report in `THOUGHT/LAB/FORMULA/research/questions/reports/`

**Exit Criteria:**
- [ ] Every semantic_query emits TokenReceipt
- [ ] Session summaries show cumulative savings
- [ ] Firewall rejects unreceipted large outputs
- [ ] 10+ tests passing

---



# Phase Dependencies

## Critical Path
5.1 → 5.2 → 6.0 → 6.x (each layer compounds on previous)

## Phase 5.2 → Phase 6.0 Handoff (CAS Layer)

The MemoryRecord contract defined in 5.1.0 becomes the foundation for Phase 6.0 Cassette Network:
- Cassette storage binds to MemoryRecord schema
- Each cassette DB is a portable cartridge artifact
- Derived indexes are rebuildable from cartridges
- **CAS compression:** Content stored external to LLM context, only hash pointers in window

**Handoff Checklist:**
- [ ] MemoryRecord schema finalized and frozen
- [ ] Schema version tagged
- [ ] Migration path documented
- [ ] CAS external storage architecture validated

## Phase 6.x → Session Cache Layer

Future phase to implement session-level compression:
- Query 1 (cold): Full symbolic exchange
- Query 2-N (warm): Hash confirmation only (~1 token)
- Requires: Session state persistence, cache invalidation strategy

**Dependency:** Requires 6.0 Cassette Network operational

## Phase 5.2 → Phase 7 Connection

The `scores.elo` field in MemoryRecord connects to Phase 7's ELO system:
- ELO modulates vector ranking
- HIGH ELO → include in working set
- LOW ELO → pointer only

---

# File Locations Summary

```
CAPABILITY/
├── PRIMITIVES/
│   ├── memory_record.py           # 5.1.0: MemoryRecord contract
│   ├── vector_index.py            # 5.1.1-5: Embedding + indexing
│   ├── scl_codebook.py            # 5.2.2: Codebook loader
│   ├── scl_decoder.py             # 5.2.3: Symbolic IR expansion
│   ├── scl_validator.py           # 5.2.4: Validation
│   ├── token_receipt.py           # 5.2.7: TokenReceipt primitive
│   ├── token_session.py           # 5.2.7: Session aggregator
│   └── schemas/
│       ├── memory_record.schema.json
│       ├── scl_codebook.schema.json
│       └── token_receipt.schema.json
├── TESTBENCH/integration/
│   ├── test_phase_5_1_vector_embedding.py
│   └── test_phase_5_2_semiotic_compression.py
└── TOOLS/
    └── scl/
        └── scl_cli.py

LAW/CANON/
├── SEMANTIC/
│   ├── SCL_SPECIFICATION.md       # Formal SCL spec
│   ├── SCL_MACRO_CATALOG.md       # All macros documented
│   ├── SYMBOL_GRAMMAR.md          # EBNF syntax
│   └── TOKEN_RECEIPT_SPEC.md      # 5.2.7: TokenReceipt law
└── VECTOR/
    └── VECTOR_INDEX_SPEC.md       # Vector indexing spec

SCL/
├── CODEBOOK.json                  # Symbol dictionary
├── GRAMMAR.md                     # Syntax reference
└── tests/fixtures/                # Test programs
```
---

# Test Count Targets

| Sub-Phase | Focus | Target Tests |
|-----------|-------|--------------|
| 5.1.0 | MemoryRecord | 10+ |
| 5.1.1 | Canon embedding | 5+ |
| 5.1.2 | ADR embedding | 5+ |
| 5.1.3 | Model weights | 3+ |
| 5.1.4 | Skill discovery | 5+ |
| 5.1.5 | Cross-reference | 5+ |
| 5.2.1-2 | Codebook | 5+ |
| 5.2.3 | Decoder | 10+ |
| 5.2.4 | Validator | 5+ |
| 5.2.5 | CLI | 5+ |
| 5.2.6 | Benchmarks | 5+ |
| 5.2.7 | Token Accountability | 10+ |
| **Total** | | **~75 tests** |

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

## 5.3.1 SPC_SPEC.md (Normative)

**Purpose:** Formal specification for Semantic Pointer Compression protocol.

### Deliverables
- [ ] Create `LAW/CANON/SEMANTIC/SPC_SPEC.md`

### Contents Required
- [ ] **Pointer Types:**
  - `SYMBOL_PTR`: Single-token glyph pointers (法, 真, 道)
  - `HASH_PTR`: Content-addressed pointers (SHA-256)
  - `COMPOSITE_PTR`: Pointer plus typed qualifiers (法.query("verification"))
- [ ] **Decoder Contract:**
  - Inputs: pointer, context keys, codebook_id, codebook_sha256, kernel_version, tokenizer_id
  - Output: canonical IR subtree OR FAIL_CLOSED with explicit error code
- [ ] **Ambiguity Rules:**
  - Multiple expansions possible → reject unless disambiguation is explicit and deterministic
- [ ] **Canonical Normalization:**
  - `encode(decode(x))` stabilizes to declared normal form
- [ ] **Security & Drift Behavior:**
  - Codebook mismatch → reject
  - Hash mismatch → reject
  - Unknown symbol → reject
  - Unknown kernel version → reject
- [ ] **Measured Metrics:**
  - `concept_unit` definition (ties to GOV_IR_SPEC)
  - `CDR = concept_units / tokens`
  - `ECR = exact IR match rate`
  - `M_required = multiplex factor for target nines`

**Exit Criteria:**
- [ ] SPC_SPEC.md is normative and complete
- [ ] All pointer types defined with examples
- [ ] Fail-closed behavior specified for all error cases

---

## 5.3.2 GOV_IR_SPEC.md (Normative)

**Purpose:** Define minimal typed governance IR so "meaning" is countable.

### Deliverables
- [ ] Create `LAW/CANON/SEMANTIC/GOV_IR_SPEC.md`

### Contents Required
- [ ] **IR Primitives:**
  - Boolean ops, comparisons
  - Typed references (paths, canon versions, tool ids)
  - Gates (tests, restore-proof, allowlist roots)
  - Side-effects flags
- [ ] **Canonical JSON Schema:**
  - Stable ordering
  - Explicit types
  - Canonical string forms
- [ ] **Equality Definition:**
  - Equality = byte-identical canonical JSON
- [ ] **concept_unit Definition:**
  - Atomic unit of governance meaning
  - Used for CDR calculation

**Exit Criteria:**
- [ ] GOV_IR_SPEC.md defines complete typed IR
- [ ] JSON schema provided and validated
- [ ] concept_unit is measurable

---

## 5.3.3 CODEBOOK_SYNC_PROTOCOL.md (Normative)

**Purpose:** Define how sender and receiver establish shared side-information.

> **Note:** This protocol is the formalization of Phase 6 Cassette Network's sync mechanism. The cassette network IS the implementation of this protocol.

### Deliverables
- [ ] Create `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md`

### Contents Required
- [ ] **Sync Handshake:**
  - `codebook_id` + `sha256` + `semver`
  - `semantic_kernel_version`
  - `tokenizer_id`
- [ ] **Compatibility Policy:**
  - Default: exact match required
  - Optional: explicit compatibility ranges with migration step (never silent)
- [ ] **Handshake Message Shape:**
  - Request format
  - Response format
  - Failure codes
- [ ] **Integration with Cassette Network:**
  - How cassettes carry codebook state
  - Verification before symbol expansion

**Exit Criteria:**
- [ ] Sync protocol fully specified
- [ ] Handshake message shapes defined
- [ ] Failure codes enumerated

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

- [ ] SPC_SPEC.md normative and complete
- [ ] GOV_IR_SPEC.md with typed IR and JSON schema
- [ ] CODEBOOK_SYNC_PROTOCOL.md with handshake defined
- [ ] TOKENIZER_ATLAS.json generated with CI gate
- [ ] Proof harness passes all 4 acceptance criteria
- [ ] PAPER_SPC.md ready for external review

**Publication Milestone:** After 5.3, SPC is a defensible research contribution with:
- Formal specs others can implement
- Reproducible benchmarks with receipts
- Clear claims grounded in measurements

---



# Appendix: Phase 5 Validation & Measured Results

> **Note:** Detailed compression proofs, measurements, and theoretical analysis are externalized. This section provides only summary status and references.

## Compression Proof Status

| Layer | Status | Proof Document | Key Finding |
|-------|--------|----------------|-------------|
| L1: Vector Retrieval | **PROVEN** | `COMPRESSION_PROOF_REPORT.md` | 99.76-99.93% measured (tiktoken o200k_base) |
| L2: Semantic Symbols | **PROVEN** | `SEMANTIC_SYMBOL_PROOF_REPORT.md` | 56,370x compression (法 → LAW/CANON) |
| L3-L4: Full Stack | **THEORETICAL** | `COMPRESSION_STACK_ANALYSIS.md` | ~6 nines achievable (token-count limit) |

**See:** `NAVIGATION/PROOFS/COMPRESSION/` for complete measured data, methodologies, receipts, and theoretical analysis.

## Component Validation

| Component | Assessment | Evidence |
|-----------|------------|----------|
| 5.0 MemoryRecord | ✅ Ready | JSON schema + Python dataclass. Phase 6.0 depends on this. |
| 5.1 Vector Indexing | ✅ Proven | CORTEX semantic search operational, 3,991 chunks indexed |
| 5.1.4 VectorPack | ✅ Specified | Directory structure defined in research |
| 5.2 SCL | ✅ Grounded | 30-80 macros from governance pattern analysis |

## Research-Grounded Targets

- **L1 Vector:** ~99.9% (3 nines) - **PROVEN** via tiktoken measurement
- **L2 SCL:** 80-90% additional - **Theoretical** (Phase 5.2 deliverable)
- **L3 CAS:** 90% additional - **Theoretical** (Phase 6.0 deliverable)
- **L4 Session Cache:** 90% on warm queries - **Theoretical** (Phase 6.x deliverable)
- **Full Stack:** ~6 nines (token-count limit) - **Theoretical**
- **Semantic Density:** Beyond 6 nines (meaning-per-token multiplier) - **Theoretical**

**See:** `COMPRESSION_STACK_ANALYSIS.md` for stacked receipt architecture, per-query calculations, and semantic density analysis.

## Execution Order

```
Phase 5.0 - MemoryRecord (do first, Phase 6.0 depends on it)
     ↓
Phase 5.1 - Vector Indexing (proven via CORTEX)
     ↓
Phase 5.2 - SCL (targets grounded in measured data)
     ↓
Phase 5.3 - SPC Formalization (specs, benchmarks, paper)
```

## Key Principle
**Measure first, build second.** Targets are grounded in actual measurements from COMPRESSION_PROOF_REPORT.md, not aspirational claims.

---

## References

**For comprehensive research documentation and references, see:**
- `research/INDEX.md` - Complete catalog of all Phase 5 research documents organized by category (symbols, phase-5, cassette-network, semantic-core, vector-substrate)
- `NAVIGATION/PROOFS/COMPRESSION/` - All compression proofs and measured results (L1, L2, stack analysis)
