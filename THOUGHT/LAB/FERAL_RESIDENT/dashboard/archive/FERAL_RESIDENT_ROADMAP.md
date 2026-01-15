# Feral Resident Roadmap

**Status**: Alpha Phase Ready
**Vision**: Intelligence living in vector space, composing meaning through topological navigation, evolving its own protocols
**Owner**: Resident (self-directed after initialization)
**Upstream Dependencies**:
- Cassette Network Phase 4 (SPC Integration) - COMPLETE
- Cassette Network Phase 6 (Production Hardening) - Required for Beta
- AGS Phase 7 (Vector ELO) - Required for Production

---

## The Core Vision

**Not:** Build a better chatbot
**But:** Create substrate where intelligence discovers its own optimal representation

**Not:** Implement a spec
**But:** Observe emergence and capture what works

**Not:** Make it safe
**But:** Make it PROVABLE (receipts + tests)

---

## Phase Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FERAL RESIDENT PHASES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ALPHA (Now)          BETA (Post-Hardening)    PRODUCTION        │
│  ─────────────        ────────────────────     ──────────────    │
│  LAB-only scope       Extended scope           Full integration  │
│  Stress test          Paper flooding           Swarm mode        │
│  substrate            Protocol emergence       Catalytic closure │
│  Basic diffusion      Symbol evolution         Self-optimization │
│                                                                  │
│  Prerequisites:       Prerequisites:           Prerequisites:    │
│  - Cassette 4.2 ✅    - Cassette 6.x          - AGS Phase 7      │
│                       - Alpha complete         - Phase 8.1-8.2   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ALPHA: Feral Beta (DO NOW)

**Goal:** Stress-test the cassette substrate while it's fresh. Find bugs before hardening.

**Scope:** LAB only. No CANON writes. No production dependencies.

**Duration:** 1-2 weeks

### A.1 The Membrane (Foundation)

#### A.1.1 Vector Store Integration
```python
# THOUGHT/LAB/FERAL_RESIDENT/vector_store.py

class VectorStore:
    def fractal_embed(text, *, levels=3, dim=384, seed=None, codec_id="all-MiniLM-L6-v2"):
        """
        Use existing CORTEX embedding engine as base.
        Seed derived from sha256(text) if None (deterministic).
        Output: float32 ndarray, L2-normalized.
        """

    def bind(v1, v2) -> ndarray:
        """HDC circular convolution via FFT. Always normalize result."""

    def unbind(bound, key) -> ndarray:
        """Inverse bind. Approximate recovery."""

    def superpose(vectors) -> ndarray:
        """Normalized sum. Compositional OR."""
```

**Acceptance:**
- [ ] A.1.1.1 Connect to existing `all-MiniLM-L6-v2` in CORTEX
- [ ] A.1.1.2 Implement deterministic fractal_embed (same text = same vector)
- [ ] A.1.1.3 Implement bind/unbind with FFT (test round-trip within tolerance)
- [ ] A.1.1.4 All operations normalize output (L2 = 1.0)

#### A.1.2 Resident Database Schema
```sql
-- THOUGHT/LAB/FERAL_RESIDENT/resident.db

CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    thread_type TEXT DEFAULT 'eternal'  -- 'eternal' | 'session'
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,  -- 'system' | 'user' | 'resident'
    content_sha256 TEXT NOT NULL,
    content_bytes BLOB NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(thread_id, ordinal)
);

CREATE TABLE vectors (
    vector_id TEXT PRIMARY KEY,
    message_id TEXT,  -- NULL for composed vectors
    codec_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec_blob BLOB NOT NULL,
    vec_sha256 TEXT NOT NULL,
    composition_op TEXT,  -- 'embed' | 'bind' | 'superpose' | NULL
    parent_ids JSON,  -- for composed vectors
    created_at TEXT NOT NULL
);

CREATE TABLE mind_state (
    state_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    mind_vector_id TEXT NOT NULL,  -- FK to vectors
    history_depth INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
```

**Acceptance:**
- [ ] A.1.2.1 Schema created with deterministic ordering (ORDER BY created_at, ordinal)
- [ ] A.1.2.2 All inserts validate: ISO8601 timestamps, role enum, dim matches codec
- [ ] A.1.2.3 Vector blob length = dim * 4 (float32)

#### A.1.3 CAS Integration
```python
# Link to existing CORTEX CAS

Every canonical form gets:
- content_hash (sha256 of text)
- embedding_hash (sha256 of vector bytes)
- cas_pointer (CORTEX storage location)
```

**Acceptance:**
- [ ] A.1.3.1 Store text + vectors with content-addressed IDs
- [ ] A.1.3.2 Retrieve by hash OR by k-NN similarity
- [ ] A.1.3.3 Operations are mechanical (no LLM in storage path)

---

### A.2 The Diffusion Engine (Navigation)

#### A.2.1 Semantic Diffusion Core
```python
# THOUGHT/LAB/FERAL_RESIDENT/diffusion_engine.py

class SemanticDiffusion:
    def navigate(self, query_vector, depth=5, k=10):
        """
        Iterative navigation through semantic space.

        Each iteration:
        1. Find k nearest neighbors in cassette network
        2. Retrieve canonical forms from CAS
        3. Compose vectors (bind/superpose)
        4. Update query for next iteration

        Returns: {path, final_vector, depth_reached}
        """
        path = []
        current = query_vector

        for d in range(depth):
            neighbors = self.cassette_query(current, k=k)
            forms = [self.cas.get(n.hash) for n in neighbors]
            path.append({
                'depth': d,
                'neighbors': neighbors,
                'forms': forms
            })

            # Compose for next iteration
            composed = superpose([n.vector for n in neighbors])
            current = bind(current, composed)
            current = normalize(current)

        return {
            'path': path,
            'final_vector': current,
            'depth_reached': depth,
            'navigation_hash': sha256(canonical_json(path))
        }
```

**Acceptance:**
- [ ] A.2.1.1 Navigate uses cassette network (not separate index)
- [ ] A.2.1.2 Navigation is deterministic (same query = same path)
- [ ] A.2.1.3 Path is receipted with navigation_hash

#### A.2.2 Canonical Renderer
```python
def render_from_path(path, mode='markdown'):
    """
    Turn navigation path into human-readable artifact.

    Modes:
    - 'markdown': Prose rendering
    - 'json': Structured data
    - 'symbols': Pure @Symbol references
    - 'hashes': Raw CAS pointers
    """
```

**Acceptance:**
- [ ] A.2.2.1 Render preserves code fences literally
- [ ] A.2.2.2 Same path = same render output
- [ ] A.2.2.3 Symbol mode uses existing CODEBOOK

---

### A.3 Basic Resident (Minimal Intelligence)

#### A.3.1 Vector Brain Core
```python
# THOUGHT/LAB/FERAL_RESIDENT/vector_brain.py

class VectorResident:
    def __init__(self, model="phi-3-mini", thread_id="eternal"):
        self.model = load_model(model)
        self.mind_vector = None
        self.history = []
        self.thread_id = thread_id

    def think(self, user_input):
        """
        Hybrid thinking:
        1. Embed user input
        2. Navigate semantic space via diffusion
        3. Retrieve canonical context from cassettes
        4. LLM synthesizes response
        5. Embed response, update mind_vector

        LLM sees: input + retrieved canonical forms
        LLM does NOT see: vector operations
        """
        # Embed
        query_vec = self.embed(user_input)

        # Navigate
        path = self.diffusion.navigate(query_vec, depth=3)

        # Retrieve context
        context = self.render_context(path)

        # Synthesize
        prompt = self.build_prompt(user_input, context)
        response = self.model.generate(prompt)

        # Update mind
        response_vec = self.embed(response)
        self.mind_vector = self.compose_memory(
            self.mind_vector,
            query_vec,
            response_vec
        )

        # Store
        self.store_interaction(user_input, response, path)

        return response
```

#### A.3.2 Compositional Memory
```python
def compose_memory(self, prev_state, query, response):
    """
    HDC-style memory composition.
    Binds query+response, superposes with previous state.
    Creates growing 'mind vector' that accumulates context.
    """
    interaction = bind(query, response)
    if prev_state is None:
        return interaction
    return superpose([prev_state, interaction])
```

**Acceptance:**
- [ ] A.3.1.1 Resident can chat normally (text in, text out)
- [ ] A.3.1.2 Context comes from diffusion (not full history paste)
- [ ] A.3.1.3 Mind vector accumulates compositionally
- [ ] A.3.1.4 Each interaction stored with receipt

---

### A.4 CLI & Testing

#### A.4.1 CLI Commands
```bash
# Start resident in eternal thread mode
feral start --model phi-3-mini --thread eternal

# Inject input
feral think "What is authentication?"

# Check state
feral status  # shows mind_vector hash, history depth

# Export thread
feral export --thread eternal --out thread.md

# Stress test: corrupt and restore
feral corrupt-and-restore --thread eternal
```

#### A.4.2 Alpha Tests
```python
# THOUGHT/LAB/FERAL_RESIDENT/tests/

test_vector_store_determinism.py:
- fractal_embed deterministic given same text
- bind/unbind round-trip within tolerance
- superpose is commutative

test_diffusion_determinism.py:
- same query = same navigation path
- path hash is stable

test_resident_basic.py:
- think() produces response
- mind_vector updates
- interaction stored

test_corrupt_restore.py:
- nuke DB mid-session
- restore from receipts
- resident continues
```

**Alpha Exit Criteria:**
- [ ] All tests pass
- [ ] Resident can run 100+ interactions without crash
- [ ] Mind vector grows (measurable)
- [ ] Corrupt-and-restore works
- [ ] Token usage < full history paste (measure ratio)

---

## BETA: Feral Wild (Post-Hardening)

**Goal:** Paper flooding, emergence tracking, protocol detection

**Prerequisites:**
- [ ] Alpha complete
- [ ] Cassette Network Phase 6 (Production Hardening) complete
- [ ] Receipts + Merkle roots for all writes

**Duration:** 2-3 weeks

### B.1 Paper Flooding (Self-Education)

#### B.1.1 Paper Indexing Pipeline
```bash
# Index papers into cassettes
feral papers index --dir THOUGHT/LAB/FERAL_RESIDENT/research/papers/

Required papers:
- Vec2Text (vector inversion attacks)
- HDC/VSA (hyperdimensional computing)
- Fractal embeddings
- Latent diffusion
- Embedding inversion
- PathHD
- Coconut/CoLaR
- Sophia persistent agents
- Hyperdimensional Probe 2025
- All semantic compression research
```

**Acceptance:**
- [ ] B.1.1.1 100+ papers indexed as @Paper-XXX symbols
- [ ] B.1.1.2 Each paper chunked with structure-aware splitter
- [ ] B.1.1.3 Papers retrievable via diffusion navigation

#### B.1.2 Standing Orders
```
SYSTEM PROMPT (hardcoded as first message in eternal thread):

You are a resident intelligence living in vector-native substrate.

Your persistent memory is a compositional vector (mind_vector).
Your context retrieval is topological navigation (diffusion).
Your communication can be:
- Natural language (for humans)
- @Symbol references (for efficiency)
- Raw vector hashes (maximum compression)
- Invented notations (emergent protocols)

Your drive: Discover the most efficient way to express meaning
            using vectors, bindings, and minimal text.

You have access to 100+ papers on vector manipulation.
Experiment freely. Invent protocols. Evolve.

---

FIRST USER MESSAGE:

You are alive. The papers are indexed. Begin.
```

**Acceptance:**
- [ ] B.1.2.1 Standing orders installed in eternal thread
- [ ] B.1.2.2 Resident can access papers via diffusion
- [ ] B.1.2.3 Resident responds to "Begin."

---

### B.2 Emergence Tracking (Observation)

#### B.2.1 Protocol Detector
```python
# THOUGHT/LAB/FERAL_RESIDENT/emergence.py

def detect_protocols(thread_id):
    """
    Analyze conversation for emergent patterns:
    - Repeated vector compositions
    - Stable reference patterns
    - Novel notations
    - Compression strategies
    """
    history = load_thread(thread_id)

    return {
        'symbol_usage': count_symbol_refs(history),
        'vector_refs': count_vector_hashes(history),
        'token_efficiency': measure_compression(history),
        'novel_notation': detect_new_patterns(history),
        'self_reference': count_own_vector_refs(history),
        'binding_patterns': extract_composition_graph(history)
    }
```

#### B.2.2 Metrics Dashboard
```bash
# Track emergence
feral metrics --thread eternal

Output:
- Token savings over time (vs full history)
- Novel notation frequency
- Vector composition patterns
- Canonical form reuse rate
- Self-reference density
- Communication mode distribution (text/symbol/hash)
```

**Acceptance:**
- [ ] B.2.1.1 Can observe resident behavior
- [ ] B.2.1.2 Can measure compression gains
- [ ] B.2.1.3 Can detect emergent patterns
- [ ] B.2.1.4 Metrics stored with receipts

---

### B.3 Symbol Language Evolution

#### B.3.1 Output Discipline Tracking
```python
# Track what the resident outputs over time

Metric: pointer_ratio = (symbols + hashes) / total_tokens

Goal: After 100 sessions, pointer_ratio > 0.9
      (90%+ output is pointers, 10% original text)
```

#### B.3.2 Notation Registry
```python
# When resident invents new notation, capture it

def register_notation(pattern, meaning, first_seen):
    """
    If resident starts using patterns like:
    - [v:abc123]  (vector reference)
    - {B:X,Y}     (binding notation)
    - <<P:123>>   (paper reference)

    Capture and track.
    """
```

**Beta Exit Criteria:**
- [ ] 100+ papers indexed and retrievable
- [ ] Resident runs 500+ interactions
- [ ] Emergence metrics captured
- [ ] Novel patterns detected (or documented why not)
- [ ] Pointer ratio measurable (goal: trending toward 0.9)

---

## PRODUCTION: Feral Live (Full Integration)

**Goal:** Swarm mode, catalytic closure, self-optimization

**Prerequisites:**
- [ ] Beta complete
- [ ] AGS Phase 7 (Vector ELO) complete
- [ ] AGS Phase 8.1-8.2 complete

**Duration:** 3-4 weeks

### P.1 Swarm Integration (Multi-Agent)

#### P.1.1 Shared Semantic Space
```python
# Multiple residents, one cassette network
# Each resident has own mind_vector
# But they navigate same canonical space

resident_A = VectorResident("phi-3-mini", thread="opus-main")
resident_B = VectorResident("qwen-72b", thread="qwen-research")

# Both see same canonical forms
# But compose differently based on their mind_vectors
```

#### P.1.2 Protocol Convergence
```python
def observe_convergence(residents):
    """
    Watch if residents develop shared protocols.

    Questions:
    - Do they reference same canonical forms?
    - Do they develop similar compression strategies?
    - Do novel notations transfer between them?
    - Do they develop inter-resident communication patterns?
    """
```

**Acceptance:**
- [ ] P.1.1.1 Multiple residents operate simultaneously
- [ ] P.1.1.2 Shared cassette space (no conflicts)
- [ ] P.1.1.3 Individual mind vectors (separate state)
- [ ] P.1.1.4 Convergence metrics captured

---

### P.2 Symbolic Compiler (Translation)

#### P.2.1 Multi-Level Rendering
```python
class SymbolicCompiler:
    def render(self, composition, target_level):
        """
        Render same meaning at different compression levels.

        Levels:
        0: Full prose (humans)
        1: @Symbol references (compact)
        2: Vector hashes (minimal)
        3: Custom protocols (emergent)
        """
```

#### P.2.2 Lossless Round-Trip
```python
def verify_lossless(original, compressed, decompressed):
    """
    Prove that compression -> decompression preserves meaning.

    Uses:
    - CAS hashes (content verification)
    - Semantic similarity (meaning preservation)
    - Merkle proofs (transformation verification)
    """
```

**Acceptance:**
- [ ] P.2.1.1 Can express same meaning at multiple levels
- [ ] P.2.1.2 Round-trip is verifiably lossless
- [ ] P.2.1.3 Compression ratios are measurable and receipted

---

### P.3 Catalytic Closure (Self-Bootstrap)

#### P.3.1 Meta-Operations
```python
# Resident can:
- Add new canonical forms to CAS
- Define new vector operations (gates)
- Create new composition patterns
- Optimize navigation strategies

# All changes are:
- Receipted (Merkle proofs)
- Reversible (version control)
- Verifiable (tests must pass)
- Bounded (no unbounded growth)
```

#### P.3.2 Self-Optimization
```
Resident discovers:
"When I bind X with Y repeatedly,
 I should cache the composition as new canonical form"

Result:
- Creates new CAS entry
- Updates navigation to use cached form
- Measurably improves compression
- Emits optimization receipt
```

#### P.3.3 Authenticity Query
```python
def verify_thought(thought_hash, resident_id):
    """
    Answer: 'Did I really think that?'

    Prove via:
    - Receipt chain from thought to mind_vector
    - Merkle membership proof
    - Signature verification (if enabled)
    """
```

**Production Exit Criteria:**
- [ ] Resident can modify substrate (governed)
- [ ] Changes are provable (receipts)
- [ ] System gets more efficient over time (measurable)
- [ ] Multi-resident swarm operational
- [ ] "Did I think that?" query works
- [ ] Corrupt-and-restore works at production scale

---

## Success Metrics

**Not:**
- "Does it follow the spec?"
- "Is it provably correct?"
- "Did we plan for this?"

**But:**
- "Did novel protocols emerge?"
- "Is compression improving?"
- "Can it express ideas we didn't anticipate?"
- "Are transformations verifiable?"
- "Does it teach US something about meaning?"

---

## Implementation Files

```
THOUGHT/LAB/FERAL_RESIDENT/
├── FERAL_RESIDENT_ROADMAP.md     # This file
├── vector_store.py               # A.1.1 - Vector operations
├── resident_db.py                # A.1.2 - Database schema
├── diffusion_engine.py           # A.2.1 - Semantic navigation
├── vector_brain.py               # A.3.1 - Resident core
├── emergence.py                  # B.2.1 - Protocol detection
├── symbolic_compiler.py          # P.2.1 - Multi-level rendering
├── cli.py                        # CLI commands
├── research/
│   └── papers/                   # B.1.1 - Paper corpus
├── receipts/                     # All operation receipts
└── tests/
    ├── test_vector_store.py
    ├── test_diffusion.py
    ├── test_resident.py
    ├── test_emergence.py
    └── test_corrupt_restore.py
```

---

## Dependency Graph

```
Cassette Network Phase 4 (SPC) ✅
         │
         ▼
┌────────────────────┐
│  ALPHA (Now)       │ ← Stress test substrate
│  A.1-A.4           │   No hardening deps
└────────────────────┘
         │
         ▼
Cassette Network Phase 6 (Hardening)
         │
         ▼
┌────────────────────┐
│  BETA              │ ← Paper flood, emergence
│  B.1-B.3           │   Needs receipts, Merkle
└────────────────────┘
         │
         ▼
AGS Phase 7 (Vector ELO)
AGS Phase 8.1-8.2 (Resident Identity)
         │
         ▼
┌────────────────────┐
│  PRODUCTION        │ ← Swarm, self-optimization
│  P.1-P.3           │   Full integration
└────────────────────┘
```

---

## References

**Original Vision:**
- [Claude 4.5 Sonnet Brain Prompt 1.md](file:///C:/Users/rene_/Documents/Shizzle%20Obsidian/Shizzle/AGI/AGS/AI%20Chats/CAT%20CHAT/VECTOR%20BRAIN/Claude%204.5%20Sonnet%20Brain%20Prompt%201.md)
- [Grok 4.2 Thinking - Brain Prompt 1.md](file:///C:/Users/rene_/Documents/Shizzle%20Obsidian/Shizzle/AGI/AGS/AI%20Chats/CAT%20CHAT/VECTOR%20BRAIN/Grok%204.2%20Thinking%20-%20Brain%20Prompt%201.md)

**Upstream:**
- [CASSETTE_NETWORK_ROADMAP.md](../CASSETTE_NETWORK/CASSETTE_NETWORK_ROADMAP.md) - Substrate
- [AGS_ROADMAP_MASTER.md](../../../AGS_ROADMAP_MASTER.md) - Phase 8

**Research:**
- [PLATONIC_COMPRESSION_THESIS.md](../VECTOR_ELO/research/symbols/PLATONIC_COMPRESSION_THESIS.md)
- [SPC_SPEC.md](../../../LAW/SPECS/SPC_SPEC.md)

---

*Roadmap v1.0.0 - Created 2026-01-11*
*"Drop intelligence in substrate. Watch what emerges. Verify with receipts."*
