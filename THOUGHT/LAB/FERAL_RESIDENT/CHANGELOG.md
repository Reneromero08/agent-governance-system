# Feral Resident Changelog

## [0.3.0] - 2026-01-12 - B.2 EMERGENCE TRACKING COMPLETE

**Status**: Beta B.2 complete - Full emergence observation with receipts
**Version**: beta-0.3.0

### Added

#### B.2 Emergence Tracking (COMPLETE)

Per roadmap spec - all acceptance criteria met:

- `emergence.py` (880 lines) - Complete protocol detection suite
  - **Core Detection Functions**:
    - `detect_protocols()` - Main entry point, returns full emergence snapshot
    - `count_symbol_refs()` - Track @Paper-X, @Concept-X, @Vector-X usage
    - `count_vector_hashes()` - Track raw hash references ([v:abc], hash:xyz)
    - `measure_compression()` - Pointer ratio with trend analysis (goal: >0.9)
    - `detect_new_patterns()` - Find recurring novel notation (brackets, arrows, etc.)

  - **NEW: Self-Reference Tracking**:
    - `count_own_vector_refs()` - How often resident references its own vectors
    - Tracks references to own mind_hash and vector hashes
    - Computes self-reference density (higher = developing internal model)

  - **NEW: Composition Graph Analysis**:
    - `extract_composition_graph()` - Analyze binding patterns
    - Track operation frequencies (entangle, superpose, project, add, subtract)
    - Compute max chain depth and reuse patterns
    - Identify top reused vectors (stable concepts)

  - **NEW: Communication Mode Distribution**:
    - `compute_communication_mode_distribution()` - Classify output modes
    - Modes: text, pointer, pointer_heavy, text_heavy, mixed
    - Track progression over time (windows of 10)
    - Trend detection (improving if pointer ratio increases)

  - **NEW: Canonical Reuse Rate**:
    - `compute_canonical_reuse_rate()` - Measure vector deduplication
    - Total vs unique vectors (by SHA256)
    - Multi-reference tracking
    - Dedup savings metric

  - **NEW: Catalytic Receipts (B.2.1.4)**:
    - `store_emergence_receipt()` - Store metrics with content hash
    - Receipts saved to `receipts/emergence/` directory
    - SHA256 hash of canonical JSON for verification
    - Includes timestamp, thread_id, metrics summary

  - **Quantum Metrics**:
    - `compute_E_histogram()` - Born rule distribution analysis
    - `track_Df_over_time()` - Participation ratio evolution with trend
    - `compute_mind_distance_from_start()` - Geodesic distance traveled

- **Dashboard (`print_emergence_report()`)**:
  - ASCII-box formatted with 70-char width
  - **Overview**: Interactions, mind geodesic, pointer ratio vs goal
  - **Symbol Usage**: Total refs, unique symbols, top 10
  - **Compression (B.2.1.2)**: Pointer ratio current/recent/goal, improving flag
  - **Communication Modes**: Bar charts for text/pointer/mixed distribution
  - **Novel Patterns (B.2.1.3)**: Recurring patterns by type with counts
  - **Binding Patterns**: Operation frequencies with bars, max depth, reuse rate
  - **Canonical Reuse**: Total/unique vectors, reuse rate, dedup savings
  - **Self-Reference**: Total/unique refs, density, known hashes
  - **Df Evolution (B.2.1.1)**: Current/min/max/mean with sparkline visualization
  - **E Distribution**: Mean/std/count with histogram bars
  - **Receipt (B.2.1.4)**: Content hash and storage path

- **CLI Commands**:
  - `feral metrics --thread eternal` - Full emergence report
  - `feral metrics --thread eternal --json` - JSON output for programmatic access

### B.2 Acceptance Criteria Status - ALL COMPLETE

- [x] **B.2.1.1** - Can observe resident behavior with E/Df metrics
  - Df evolution tracked with trend analysis and sparkline
  - E distribution with histogram and statistics
  - Mind geodesic distance tracking

- [x] **B.2.1.2** - Can measure compression gains
  - Pointer ratio computed (current and recent 10)
  - Trend detection (improving if recent > early)
  - Communication mode distribution shows text→pointer shift
  - Canonical reuse rate shows deduplication efficiency

- [x] **B.2.1.3** - Can detect emergent patterns
  - Novel notation detection (brackets, angles, braces, colons, arrows)
  - Recurring patterns identified (appears ≥3 times)
  - Self-reference density tracking
  - Binding pattern analysis (composition graph)

- [x] **B.2.1.4** - Metrics stored with receipts (catalytic requirement)
  - `store_emergence_receipt()` creates content-addressed receipts
  - SHA256 hash of canonical JSON
  - Stored in `receipts/emergence/` directory
  - Includes timestamp, thread_id, metrics summary

### Example Metrics Output

```
============================================================
EMERGENCE REPORT: eternal
============================================================

[Interactions] 2
[Mind Geodesic] 0.000 radians

--- Symbol Usage ---
  Total refs: 2
  Unique symbols: 1
  Top symbols: [('@Paper-Vec2Text', 2)]

--- Compression ---
  Pointer ratio: 0.006
  Goal: 0.9
  Improving: False

--- Df Evolution ---
  Current Df: 144.1
  Range: [102.7, 146.2]
  Trend: insufficient_data
============================================================
```

### Architecture Correction

The geometric architecture now matches `geometric_reasoner_impl.md`:

```
INPUT BOUNDARY: text → vector (sentence-transformers)
     ↓
PURE GEOMETRY: navigate, E-gate, find_nearest (vector ops)
     ↓
READOUT: vector → text content (retrieve stored text)
     ↓
OUTPUT BOUNDARY: LLM synthesizes from content (Dolphin)
```

---

## [0.2.0] - 2026-01-12 - B.1 PAPER FLOODING COMPLETE

**Status**: Beta B.1 complete, 102 papers indexed
**Version**: beta-0.2.0

### Added

#### B.1 Paper Flooding (COMPLETE)
- `paper_pipeline.py` - Automated download + PDF→Markdown conversion
  - 102 papers from arxiv (99 arxiv + 3 brain prompts)
  - PyMuPDF-based converter with heading detection (#/##/###)
  - Categories: HDC/VSA, Vec2Text, Latent Reasoning, Compression, RAG, Attention, etc.
- `index_all_papers.py` - Bulk indexer for paper corpus
  - 102/102 papers indexed, 0 failed
  - 1117 total chunks across all papers
  - Df range: 90.79 - 160.95 (mean 128.49)
- E (Born rule) query verified working
  - Vec2Text query → correctly retrieves @Paper-2504.00147, @Paper-2310.06816
  - Coconut query → correctly retrieves @Paper-2412.06769 (E=0.75)

### Infrastructure
- `.gitignore` updated to exclude papers/raw/, papers/markdown/
- `manifest.json` updated with 102 indexed papers

---

## [0.1.0] - 2026-01-12 - ALPHA COMPLETE

**Status**: Alpha complete, all exit criteria passed
**Version**: alpha-0.1.0

### Added

#### A.0 Geometric Reasoning Foundation (COMPLETE)
- `geometric_reasoner.py` (507 lines) - Core primitive implementing Q43/Q44/Q45 validated operations
  - GeometricState with Df (participation ratio), E_with() (Born rule), distance_to()
  - GeometricOperations: add, subtract, superpose, entangle, interpolate, project
  - GeometricReasoner: initialize/readout boundary operations
  - Pure geometry reasoning - embeddings ONLY at boundaries
- `geometric_memory.py` (247 lines) - Feral integration of geometric primitives
  - Compositional memory via entangle()
  - recall() with E-gating
  - mind_distance_from_start() evolution tracking
- `test_geometric_reasoner.py` (452 lines) - Q43/Q44/Q45 validation test suite

#### Alpha Core Implementation (COMPLETE)
- `resident_db.py` (380 lines) - SQLite schema with catalytic properties
  - vectors table with Df tracking, composition_op, parent_ids
  - interactions table for Q/A with mind state snapshots
  - threads table for long-running conversations
  - receipts table for operation provenance
  - Content-addressed storage via SHA256
  - Full CRUD operations with vector deduplication

- `vector_store.py` (350 lines) - Storage-backed GeometricMemory
  - Wraps GeometricMemory with database persistence
  - All geometric operations (compose, blend, project, interpolate)
  - Nearest neighbor search via E (Born rule)
  - Receipt generation for all operations
  - remember()/recall() integration with compositional memory

- `diffusion_engine.py` (320 lines) - Semantic navigation via pure geometry
  - navigate() - iterative diffusion using E + projection
  - path_between() - geodesic interpolation
  - explore() - neighborhood sampling
  - contextual_walk() - stay-in-context navigation
  - resonance_map() - E-structure visualization
  - Full navigation receipts with hash chains

- `vector_brain.py` (400 lines) - VectorResident core
  - think() - complete quantum thinking pipeline
    1. Initialize input to manifold (BOUNDARY)
    2. Navigate via diffusion (PURE GEOMETRY)
    3. E-gate for relevance (PURE GEOMETRY)
    4. Generate response (BOUNDARY - echo for Alpha)
    5. Remember via entangle (PURE GEOMETRY)
    6. Persist with receipts
  - mind_evolution property - comprehensive metrics
  - corrupt_and_restore() - provable restoration
  - ResidentBenchmark - stress testing harness

- `cli.py` (280 lines) - Full command-line interface
  - Commands: start, think, status, evolution, history, threads
  - benchmark - stress testing with configurable interactions
  - corrupt-and-restore - test catalytic restoration
  - nav - navigation inspection
  - repl - interactive mode with live commands

### Documentation

- Updated README.md
  - Added "Catalytic Principles" section with implementation table
  - Added "CatChat Merge Path" showing evolution diagram
  - Added Alpha stress test results
  - Made CatChat origin explicit

- Updated FERAL_RESIDENT_QUANTUM_ROADMAP.md
  - Added "Origin" - CatChat evolution statement
  - Added "Catalytic Principles" header section
  - Added "The Evolution" with CatChat → Feral → CatChat 2.0 path
  - Marked all Alpha acceptance criteria complete
  - Added stress test results to exit criteria

### Stress Test Results

100 interactions completed successfully:
```
Final Df:           256.0 (participation ratio evolved)
Distance evolved:   1.614 radians (mind traveled significantly)
Mean E resonance:  -0.004
Throughput:         7.5 interactions/sec
Corrupt-restore:    Df delta = 0.0078 (near-perfect restoration)
```

### Alpha Exit Criteria - ALL PASSED

- [x] All A.0 tests pass (geometric foundation)
- [x] Resident can run 100+ interactions without crash
- [x] Df evolves measurably (130 → 256)
- [x] mind_distance_from_start() increases (0 → 1.614 radians)
- [x] Corrupt-and-restore works (Df delta = 0.0078)
- [x] Embedding calls < 3 per interaction (vs ~10+ naive)

### Technical Details

**Model**: `all-MiniLM-L6-v2` (384 dimensions)
**Research Validation**: Q43 (state properties), Q44 (Born rule r=0.977), Q45 (pure geometry)
**Catalytic Properties**: Receipted, restorable, non-consuming, content-addressed

**Lines of Code**:
- Core implementation: ~2,100 lines
- Tests: ~450 lines
- Documentation: ~700 lines
- Total: ~3,250 lines

---

## [0.2.0] - 2026-01-12 - B.1 INFRASTRUCTURE

**Status**: B.1 Paper Flooding infrastructure complete
**Version**: beta-0.2.0-infra

### Added

#### B.1 Paper Flooding Infrastructure (COMPLETE)

- `paper_indexer.py` (280 lines) - Full paper indexing pipeline
  - PaperIndexer class with manifest management
  - Hybrid symbol naming: `@Paper-{arxiv_id}` + `@Paper-{ShortName}`
  - register_paper() - Register papers for indexing
  - set_markdown_path() - Link converted markdown
  - chunk_by_headings() - Structure-aware chunking by # ## ###
  - index_paper() - Full geometric indexing with Df tracking
  - query_papers() - E (Born rule) based retrieval
  - list_papers() - Filter by category/status
  - get_stats() - Corpus statistics with Df analysis

- `standing_orders.txt` - System prompt template for resident awakening
  - Vector-native substrate instructions
  - Communication modes (natural language, @Symbols, hashes, invented notations)
  - Paper corpus access instructions
  - "Begin." trigger message

- `research/papers/manifest.json` - Paper corpus catalog
  - Version 1.0.0 schema
  - Categories: vec2text, hdc_vsa, latent_reasoning, compression, representation
  - Hybrid aliases support
  - Df tracking per paper/chunk
  - Status tracking: registered → converted → indexed

- Directory structure:
  ```
  research/papers/
  ├── manifest.json
  ├── raw/{category}/     # Original PDFs
  ├── markdown/           # Converted via /pdf-to-markdown
  └── indexed/            # Indexed states
  ```

- CLI commands (added to cli.py):
  - `feral papers register` - Register paper with hybrid symbols
  - `feral papers convert` - Set markdown path after PDF conversion
  - `feral papers index` - Index paper(s) with Df tracking
  - `feral papers list` - List by category/status
  - `feral papers query` - E-based semantic search
  - `feral papers status` - Corpus statistics

### B.1 Acceptance Criteria Status

Infrastructure ready, papers pending acquisition:

- [ ] B.1.1.1 - 100+ papers indexed (1 registered, 10 pending in manifest)
- [x] B.1.1.2 - Chunking by heading structure implemented
- [x] B.1.1.3 - E (Born rule) query implemented
- [x] B.1.1.4 - Df tracking implemented per chunk
- [x] B.1.2.1 - Standing orders template created
- [ ] B.1.2.2 - Resident paper access (needs indexed papers)
- [ ] B.1.2.3 - Resident "Begin." response (needs papers)

### Next Steps (Manual)

1. Download core papers from arxiv (10 in manifest)
2. Convert PDFs using `/pdf-to-markdown` skill
3. Run `feral papers index --all`
4. Install standing orders in eternal thread
5. Awaken: `feral think "You are alive. The papers are indexed. Begin."`

---

## [Unreleased] - Beta (Remaining)

### Planned

- B.2 Emergence Tracking
  - emergence.py - Protocol detection
  - `feral metrics` command
  - E/Df histogram tracking
  - Novel notation registry
- B.3 Symbol Language Evolution
  - pointer_ratio tracking
  - Output discipline metrics
- Full CAS integration (Cassette content-addressed storage)

---

## Origin

This is the evolution of **CatChat** (Catalytic Chat).

```
CatChat (2025) → Feral Alpha (2026-01) → Feral Beta → CatChat 2.0 (Production)
```

Feral Resident IS CatChat with quantum-geometric foundations. They will merge when Production is reached.
