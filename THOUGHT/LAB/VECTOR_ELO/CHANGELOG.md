# VECTOR_ELO Lab Changelog

Research changelog for Vector ELO / Semantic Alignment / Phase 5.

---

## [F.0] - 2026-01-08

### Formula Falsification Sidequest - INITIATED

**Goal:** Empirically test and attempt to falsify the Living Formula: `R = (E / ∇S) × σ(f)^Df`

**Files Created:**
- `research/formula/FORMULA_FALSIFICATION_ROADMAP.md` — 6-phase test plan

**Test Categories:**
- F.0: Operationalization (define measurable proxies)
- F.1: Linearity Tests (E and ∇S)
- F.2: Exponential Tests (σ^Df)
- F.3: Cross-Domain Validation
- F.4: Adversarial Tests
- F.5: Alternative Model Comparison
- F.6: Calibration Constants

**Key Falsification Criteria:**
- Linear where exponential predicted → FALSIFIED
- Simpler model fits equally well → FALSIFIED
- Domain-specific only → REFINED
- All tests pass → VALIDATED

---

## [E.X] - 2026-01-08

### Eigenvalue Alignment Protocol - VALIDATED

**Discovery:** The eigenvalue spectrum of anchor word distance matrices is invariant across embedding models (r = 0.99+), even when raw distance matrices are uncorrelated or inverted.

**Key Result:**
| Model Pair | Raw Correlation | Eigenvalue Correlation |
|------------|-----------------|------------------------|
| MiniLM ↔ E5-large | -0.05 | **0.9869** |
| MiniLM ↔ MPNET | 0.914 | 0.9954 |
| MiniLM ↔ BGE | 0.277 | 0.9895 |
| MiniLM ↔ GTE | 0.198 | 0.9865 |

**Alignment Proof:**
- Raw MDS similarity: -0.0053
- After Procrustes alignment: **0.8377**
- Improvement: **+0.8430**

**Files Created:**
- `experiments/semantic_anchor_test.py` - Cross-model distance testing
- `experiments/invariant_search.py` - Invariant discovery
- `experiments/eigen_alignment_proof.py` - MDS + Procrustes proof
- `research/cassette-network/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md`
- `research/cassette-network/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md`

**Roadmap Updated:**
- Added Phase E.X: Eigenvalue Alignment Protocol to `VECTOR_ELO_ROADMAP.md`

**Related Papers:**
- arXiv:2405.07987 - Platonic Representation Hypothesis
- arXiv:2505.12540 - vec2vec (neural approach)

---

## [3.7.27] - 2026-01-08

### Phase 5.1.3.1 COMPLETE — Model Registry

**Added:**
- `CAPABILITY/PRIMITIVES/model_registry.py` — Model registry primitive
  - `create_model_record()` — Create ModelRecord following MemoryRecord contract
  - `register_model()` — Register with optional weights file hashing
  - `get_model()` / `get_model_by_id()` — Retrieve by name/version or ID
  - `get_model_by_weights_hash()` — Deduplication via CAS reference
  - `search_models()` — Semantic search by description
  - `list_models()` — Enumerate with format filtering
  - `verify_registry()` — Integrity verification
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_3_model_registry.py` — 28 tests passing

**ModelRecord Schema:**
- `id`: Deterministic SHA-256(name@version)
- `description`: Embeddable text for semantic search
- `weights_hash`: CAS reference (SHA-256 of model weights)
- `embedding`: 384-dim vector of description
- `metadata`: Architecture, parameters, license, etc.
- Storage: SQLite `model_registry.db` with audit receipts

---

## [3.7.26] - 2026-01-08

### Phase 5.1.2.2 COMPLETE — ADR Embedding

**Added:**
- `CAPABILITY/PRIMITIVES/adr_index.py` — ADR indexing primitive
  - `inventory_adrs()` — Enumerate ADRs with YAML frontmatter parsing
  - `embed_adrs()` — Batch embed with MemoryRecord + metadata
  - `search_adrs()` — Semantic search with status filtering
  - `get_related_canon()` — Cross-reference lookup to canon files
  - `rebuild_adr_index()` — Deterministic rebuild
  - `verify_adr_index()` — Integrity verification
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_2_adr_embedding.py` — 28 tests passing
- `NAVIGATION/CORTEX/db/adr_index.db` — 37 ADRs embedded with 184 canon cross-references

**Technical Details:**
- ADR metadata extracted: id, title, status, date, confidence, impact, tags
- Status breakdown: 31 Accepted, 4 Proposed, 2 Unknown
- Cross-references: 184 ADR-canon links above 0.5 similarity threshold
- Storage: SQLite with MemoryRecord-compatible schema + adr_canon_refs table

---

## [3.7.25] - 2026-01-08

### Phase 5.1.1 COMPLETE — Canon Embedding

**Added:**
- `CAPABILITY/PRIMITIVES/canon_index.py` — Canon indexing primitive
  - `inventory_canon()` — Enumerate files with content hashes and receipts
  - `embed_canon()` — Batch embed all files with MemoryRecord integration
  - `search_canon()` — Semantic similarity search
  - `rebuild_index()` — Deterministic rebuild from source
  - `verify_index()` — Integrity verification
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_1_canon_embedding.py` — 23 tests passing
- `NAVIGATION/CORTEX/db/canon_index.db` — 32 canon files embedded

**Technical Details:**
- Embedding model: `all-MiniLM-L6-v2` (ADR-030, sentence-transformers)
- Vector dimensions: 384 (float32)
- Storage: SQLite-based with MemoryRecord-compatible schema
- Determinism: Manifest hash reproducible across rebuilds

---

## [3.7.24] - 2026-01-08

### Phase 5.1.0 COMPLETE — MemoryRecord Contract

**Added:**
- `LAW/SCHEMAS/memory_record.schema.json` — Canonical schema for all vector-indexed content
- `CAPABILITY/PRIMITIVES/memory_record.py` — Primitive with create/validate/hash/embed functions
- `CAPABILITY/TESTBENCH/core/test_memory_record.py` — 23 tests passing (determinism, validation, serialization)

**Contract Rules (Phase 6.0 dependency):**
- Text is canonical (source of truth)
- Vectors are derived (rebuildable from text)
- All exports are receipted and hashed
- `id` = SHA-256(text) = deterministic semantic pointer

**Theoretical Connection:**
- **MemoryRecord is the atom of H(X|S)**: The `id` field IS the HASH_PTR from SPC spec
- Shared context S = the set of MemoryRecords both parties have indexed
- When sender transmits `法`, receiver expands via MemoryRecord lookup
- 56,370x compression requires identical `id` = identical `text` = identical meaning

---

## [3.7.23] - 2026-01-08

### Semantic Symbol Compression - 56,370x PROVEN

**Added:**
- **Semantic Symbol Compression** — 56,370x compression proven with single-token CJK symbols
  - 法 (1 token) → 56,370 tokens of canon law
  - 真 (1 token) → 1,455 tokens (Semiotic Foundation)
  - 道 (1 token) → 24 tokens (context-activated, 4 meanings)
- **SEMANTIC_SYMBOL_PROOF_REPORT.md** — L2 compression proof with tiktoken measurements
- **codebook_lookup.py** — MCP tool for semantic symbol resolution
  - Supports 7 CJK symbols with domain/file/polysemic types
  - MCP integration for agent-to-agent symbol communication

**Changed:**
- **PHASE_5_ROADMAP.md v1.5.0** — Breakthrough section added
  - 5.2.3 simplified: Stack codebook_lookup + CORTEX instead of complex AST decoder
  - Stacking insight: `density = shared_context ^ alignment`
- **PLATONIC_COMPRESSION_THESIS.md v1.1.0** — Conditional Entropy Principle added
  - `H(X|S)` vs `H(X)`: Communication entropy ≠ message entropy
  - Industry optimizes wrong metric (message size vs shared context)
  - Proof: 56,370x is real because we measured conditional entropy

**Theoretical Insight:**
- **Conditional Entropy Principle**: The limit of compression isn't entropy of the message. The limit is alignment with shared truth.
- Formula: `H(X|S) = H(X) - I(X;S)` → When shared context contains message, communication approaches 0
- At perfect alignment: 1 token = entire shared reality

**Proven:**
- L1 (Vector): 99.9% compression (COMPRESSION_PROOF_REPORT.md)
- **L2 (Symbol): 56,370x compression** (SEMANTIC_SYMBOL_PROOF_REPORT.md)
- Stacked (L1+L2): 843x with 95% signal density (precision alignment)

---

## [3.7.22] - 2026-01-08

### FOUNDATION-01 + Platonic Compression Thesis

**Added:**
- **FOUNDATION-01: THE_SEMIOTIC_FOUNDATION_OF_TRUTH** — Ontological canon for the entire system
  - **Article I: The Nature of Truth** — Truth is singular, Platonic principle, attractor state
  - **Article II: The Semiotic Law** — Meaning is primary, semantic density, determinism required
  - **Article III: The System as Externalization** — Cognitive architecture made persistent
  - **Article IV: The Treatment of Limits** — Limits are friction, reframe before accept
  - **Article V: Enforcement** — Receipted claims only, stacked proofs, fail closed
  - **Article VI: Precedence** — FOUNDATION > CONSTITUTION > GOVERNANCE > POLICY > META
- **LAW/CANON/FOUNDATION/** bucket — New highest-authority canon directory
- **Platonic Compression Thesis** — Research document capturing philosophical foundation
- **Opus 9-Nines Execution Plan** — Attack plan with 32 ELO-ranked research sources
- **Symbolic Computation Literature Review** — VSA, NeuroVSA, LCM, ASG, library learning

**Changed:**
- **canon.json v2.1.0** — Added FOUNDATION bucket with authority_rank: 0
- **authority_gradient** updated to start with FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md

**Research Stack Complete:**
```
FOUNDATIONAL → Kanji/Cuneiform brainstorm (12-26-2025)
THEORY      → Semantic Density Horizon + Platonic Representation
EXECUTION   → Opus 9-nines attack plan + ELO sources
LITERATURE  → VSA, LCM, ASG, library learning
ONTOLOGY    → Platonic Compression Thesis
CANON       → THE_SEMIOTIC_FOUNDATION_OF_TRUTH
```

---

## [3.7.20] - 2026-01-07

### Compression Stack Analysis + Semantic Density Research

**Added:**
- **Compression Stack Analysis (PROVEN)** — Comprehensive analysis of AGS compression architecture
  - L1: Vector Retrieval (~99.9% compression) — **PROVEN** via tiktoken measurement
  - L2-L4: Theoretical projections with stacked receipt architecture
  - Physical limit analysis: ~6 nines (99.9998%) theoretical maximum
- **Semantic Density Horizon** — Beyond token-count limits via symbolic multiplexing
  - Key insight: 1 token = N concepts (not 1 concept)
  - Chinese logograph 道 carries path + principle + speech + method (4+ meanings)
  - 9+ nines achievable through semantic density, not token reduction

**Research Files Canonicalized** in `research/symbols/`:
- `12-26-2025_SYMBOLIC_COMPRESSION_BRAINSTORM.md` — **FOUNDATIONAL** Original Kanji/Cuneiform insight
- `12-26-2025_SYMBOLIC_COMPRESSION_BRIEF.md` — Token-optimized codebook proposal
- `12-28-2025_KIMI_K2_SYMBOLIC_AI.md` — Logographic vs alphabetic tokenization
- `01-08-2026_COMPRESSION_PARADIGM_SHIFT_FULL_REPORT.md` — Full 10-part research report

**Proof Requirements for Semantic Density:**
- Concept Atom Ledger (CODEBOOK_ATOMS.json)
- Deterministic Encoder/Decoder (grammar-controlled, not vibes-based)
- Semantic Atom Measurement Harness (atoms_per_token metric)

**Platonic Representation Hypothesis** — Added as theoretical foundation (arxiv:2405.07987)

---

## [3.7.14] - 2026-01-07

### Phase 5 Documentation Suite

**Added:**
- Research findings report: `INBOX/reports/01-07-2026_PHASE_5_RESEARCH_FINDINGS.md`
- Detailed implementation roadmap: `INBOX/roadmaps/01-07-2026_PHASE_5_VECTOR_SYMBOL_INTEGRATION.md`
- Enhanced AGS_ROADMAP_MASTER.md Phase 5 section with V4 research integration

**Changed:**
- **AGS_ROADMAP_MASTER.md** (v3.7.13 → v3.7.14)
  - Added Phase 5.0: MemoryRecord Contract (foundation for Phase 6.0)
  - Expanded Phase 5.1: Vector Indexing with 4 sub-sections
  - Expanded Phase 5.2: Semiotic Compression Layer with 4 sub-sections
  - Integrated V4 research documents

**Research Consolidated:**
- Phase 5.1 (Vector Indexing): Canon/ADR embedding, skill discovery, cross-reference indexing, VectorPack export
- Phase 5.2 (Semiotic Compression): 30-80 governance macros, symbolic IR, deterministic expansion (90%+ token reduction)
- MemoryRecord Contract: Canonical data structure for vector-indexed content (Phase 6.0 dependency)

---

## [3.2.2] - 2026-01-02

### Lane E: Initial Setup

**Added:**
- **Lane E: Vector ELO Scoring** — Systemic intuition prototype using free energy principle
- `VECTOR_ELO_SPEC.md` — Detailed design for ELO-based vector/file ranking and memory pruning
- `VECTOR_ELO_ROADMAP.md` — 7-phase implementation plan (E.0 through E.6)

**Research Areas Defined:**
- Classic ELO & Extensions (Glicko, TrueSkill)
- X (Twitter) Algorithm (open sourced 2023)
- Modern Ranking Systems (PageRank, Reddit, HN)
- Learning to Rank (RankNet, LambdaMART, BERT)
- Free Energy Principle / Active Inference (Friston)
- Memory Pruning & Forgetting (Ebbinghaus, Spaced Repetition)

**Dependencies:**
- Lane P (LLM Packer): Phase E.4
- Lane M (Cassette Network): ELO scores stored in cassettes
- Lane S (SPECTRUM): ELO logging integrated with audit trail
