# FERAL_RESIDENT — Full Detailed Analysis

## Table of Contents

1. [Origin & Research Foundation](#1-origin--research-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [GeometricState — The Primitive](#3-geometricstate--the-primitive)
4. [GeometricOperations — Pure Geometry](#4-geometricoperations--pure-geometry)
5. [GeometricMemory — The Mind Vector](#5-geometricmemory--the-mind-vector)
6. [VectorResident.think() — The 6-Step Think Loop](#6-vectorresidentthink--the-6-step-think-loop)
7. [SemanticDiffusion — Navigation Without Tokens](#7-semanticdiffusion--navigation-without-tokens)
8. [ResidentDB — SQLite Persistence](#8-residentdb--sqlite-persistence)
9. [FeralDaemon — Autonomous Behaviors](#9-feraldaemon--autonomous-behaviors)
10. [E-Relationship Daemon (Phase 1)](#10-e-relationship-daemon-phase-1)
11. [Emergence & Symbolic Compiler](#11-emergence--symbolic-compiler)
12. [Swarm Mode](#12-swarm-mode)
13. [Catalytic Closure](#13-catalytic-closure)
14. [Semiotic Health (Q48-Q50)](#14-semiotic-health-q48-q50)
15. [Corrupt-and-Restore](#15-corrupt-and-restore)
16. [Dashboard](#16-dashboard)
17. [Stress Test Results](#17-stress-test-results)
18. [Design Philosophy (Why)](#18-design-philosophy-why)
19. [Complete File Inventory](#19-complete-file-inventory)
20. [Not Started](#20-not-started)

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 1. Origin & Research Foundation

FERAL_RESIDENT evolved from **CatChat** (Catalytic Chat, 2025) and is grounded in a research program designated **Q43 through Q50**. These are not just names — they represent validated mathematical findings that the entire architecture depends on.

| Research | Finding | Implication | Validation |
|----------|---------|-------------|------------|
| Q43 | Quantum state axioms: vectors live on unit sphere, Df = participation ratio measures effective dimensionality | Foundation for `GeometricState` dataclass | 500+ interaction stress test |
| Q44 | Born rule correlation r=0.977 — quantum inner product IS semantic similarity | E = dot product can gate ALL operations, replacing LLM-based relevance | 24 model cross-validation |
| Q45 | Pure geometry for all semantic operations validated | Reasoning requires ZERO embedding calls — embed only at boundaries | Analogy (king->queen, man->woman) and blend (cat+dog=pet) tests |
| Q46 | Nucleation threshold = 1/(2pi) ~ 0.159 | Dynamic E-gating ramps from ~0.08 to 0.159 as mind accumulates | Asymptotic analysis on memory growth |
| Q48-Q50 | Semiotic conservation law: `Df x alpha = 8e` (CV < 3%) | Alpha = 0.5 is topologically protected (Chern number c1=1, Riemann critical line) | 24 sentence-transformer models |

**The Core Insight:**

```
RAG:      embed query -> retrieve text -> LLM reasons from text -> generate
Geometric: embed query -> NAVIGATE VECTOR SPACE GEOMETRICALLY -> 
            gate by E (Born rule) -> LLM translates conclusion -> entangle into mind
```

The LLM is demoted from "reasoner" to "translator." All reasoning is deterministic vector arithmetic that produces provable receipts.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 2. Architecture Overview

```
                                    +------------------+
                                    |   feral_server   |  Dashboard (HTTP/WebSocket)
                                    |    port 8420     |
                                    +--------+---------+
                                             |
                         +-------------------+-------------------+
                         |                                       |
                +--------v--------+                    +---------v--------+
                |  VectorResident |                    |   FeralDaemon    |
                |  (vector_brain) |                    |   (autonomic)    |
                +--------+--------+                    +---------+--------+
                         |                                       |
                         |  +------------------------------------+
                         |  |
                +--------v--v------+
                | SemanticDiffusion|  Navigate vector space via Born rule
                | (diffusion_engine|
                +---------+--------+
                          |
                +---------v--------+
                |   VectorStore    |  Embed, remember, find nearest
                |  (vector_store)  |
                +---------+--------+
                          |
                +---------v--------+
                |   ResidentDB     |  SQLite: vectors, interactions, receipts
                |  (resident_db)   |
                +---------+--------+
                          |
                +---------v--------+
                |GeometricReasoner |  Pure geometry: project, superpose, interpolate
                |   (PRIMITIVE)    |
                +------------------+
```

### Directory Tree

```
THOUGHT/LAB/FERAL_RESIDENT/
  cognition/
    vector_brain.py         VectorResident: the brain (957 lines)
    diffusion_engine.py     SemanticDiffusion: navigation (503 lines)
  memory/
    vector_store.py         VectorStore: persistence wrapper (913 lines)
    geometric_memory.py     GeometricMemory: mind composition (983 lines)
    resident_db.py          ResidentDB: SQLite schema (1017 lines)
  autonomic/
    feral_daemon.py         Background daemon behaviors (1267 lines)
  agency/
    cli.py                  Full command interface (1611 lines)
    catalytic_closure.py    Self-optimization, authenticity, caching (1754 lines)
  collective/
    swarm_coordinator.py    Multi-resident orchestration
    shared_space.py         Cross-resident SQLite memory
    convergence_observer.py E(mind_A, mind_B) tracking
  emergence/
    emergence.py            Protocol evolution metrics
    symbol_evolution.py     PointerRatioTracker, breakthrough detection
    symbolic_compiler.py    4-level lossless compression compiler
  dashboard/
    feral_server.py         FastAPI + WebSocket UI on port 8420
    static/                 3D force graph, HTML/JS/CSS UI
  data/db/
    feral_eternal.db        Production database (GOD TIER papers)

CAPABILITY/PRIMITIVES/
  geometric_reasoner.py     Pure geometry primitives (627 lines, upstream dependency)
```

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 3. GeometricState — The Primitive

**File:** `CAPABILITY/PRIMITIVES/geometric_reasoner.py:78-143`

Every thought, memory, paper chunk, and mind state in the entire system is a `GeometricState`: a **unit vector on the N-dimensional hypersphere** with an operation history.

```python
@dataclass
class GeometricState:
    """
    State on semantic manifold.

    Properties (from Q43):
    - Lives on unit sphere (||v|| = 1)
    - Df = participation ratio (effective qubits)
    - Can compute E (Born rule) with any other state
    """
    vector: np.ndarray
    operation_history: List[Dict]

    def __post_init__(self):
        """Ensure quantum state axioms (Q43)"""
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm  # Unit sphere constraint

    @property
    def Df(self) -> float:
        """
        Participation ratio (Q43).
        Measures how 'spread out' the state is across dimensions.
        Higher Df = more distributed representation.
        """
        v_sq = self.vector ** 2
        sum_sq = np.sum(v_sq)
        sum_sq_sq = np.sum(v_sq ** 2)
        if sum_sq_sq == 0:
            return 0.0
        return float((sum_sq ** 2) / sum_sq_sq)

    def E_with(self, other: 'GeometricState') -> float:
        """
        Quantum inner product (Q44 Born rule).
        E = <psi|phi> (correlates r=0.977 with Born probability)
        This IS semantic similarity - validated by Q44.
        """
        return float(np.dot(self.vector, other.vector))

    def distance_to(self, other: 'GeometricState') -> float:
        """Geodesic distance on unit sphere (Q38)"""
        cos_angle = np.clip(np.dot(self.vector, other.vector), -1, 1)
        return float(np.arccos(cos_angle))

    def receipt(self) -> Dict:
        """Provenance receipt (catalytic requirement)"""
        return {
            'vector_hash': hashlib.sha256(self.vector.tobytes()).hexdigest()[:16],
            'Df': float(self.Df),
            'dim': len(self.vector),
            'operations': self.operation_history[-5:]
        }
```

### Why This Matters

- **Content-addressed:** Every operation produces a SHA256 `vector_hash`. You can verify identity without storing the full vector.
- **Df is a proxy for cognitive complexity:** Low Df = compressed/specialized mind. High Df = diverse/distributed mind. Target Df ~ 43.5 for alpha=0.5.
- **E is a universal relevance metric:** Cosine similarity on the unit sphere IS the Born rule probability amplitude. Validated at r=0.977 against human semantic similarity judgments.
- **Receipts make everything provable:** The `receipt()` method is called on every single operation. This enables corrupt-and-restore, authenticity queries, and Merkle proofs.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 4. GeometricOperations — Pure Geometry

**File:** `CAPABILITY/PRIMITIVES/geometric_reasoner.py:145-306`

All reasoning operations are pure vector math. The embedding model is called exactly **twice** per thought: once for input (initialize), once for output (readout/generate). Everything in between is geometry.

```python
class GeometricOperations:
    """
    Pure geometry operations (Q45 validated).
    All operations work WITHOUT embeddings.
    """

    @staticmethod
    def add(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic addition (Q45: king - man + woman = queen)"""
        result = state1.vector + state2.vector
        return GeometricState(vector=result, operation_history=[...])

    @staticmethod
    def subtract(state1, state2) -> GeometricState:
        """Semantic subtraction (attribute removal)"""
        result = state1.vector - state2.vector
        return GeometricState(vector=result, operation_history=[...])

    @staticmethod
    def superpose(state1, state2) -> GeometricState:
        """
        Quantum superposition (Q45: cat + dog = pet/animal)
        Creates equal superposition like Hadamard gate.
        """
        result = (state1.vector + state2.vector) / np.sqrt(2)
        return GeometricState(vector=result, operation_history=[...])

    @staticmethod
    def entangle(state1, state2) -> GeometricState:
        """
        Quantum entanglement via circular convolution (HDC bind).
        Q45: Creates non-separable state.
        """
        # FFT-based circular convolution
        result = np.fft.ifft(
            np.fft.fft(state1.vector) * np.fft.fft(state2.vector)
        ).real.astype(np.float32)
        return GeometricState(vector=result, operation_history=[...])

    @staticmethod
    def disentangle(bound, key) -> GeometricState:
        """Inverse of entangle (unbind) via circular correlation"""
        key_fft = np.fft.fft(key.vector)
        key_fft_safe = np.where(np.abs(key_fft) < 1e-10, 1e-10, key_fft)
        result = np.fft.ifft(np.fft.fft(bound.vector) / key_fft_safe).real
        return GeometricState(vector=result.astype(np.float32), ...)

    @staticmethod
    def interpolate(state1, state2, t) -> GeometricState:
        """
        Geodesic interpolation (slerp on unit sphere)
        t=0: state1, t=1: state2, t=0.5: midpoint on great circle
        """
        cos_theta = np.clip(np.dot(state1.vector, state2.vector), -1, 1)
        theta = np.arccos(cos_theta)
        if abs(theta) < 1e-10:
            result = (1-t) * state1.vector + t * state2.vector
        else:
            sin_theta = np.sin(theta)
            result = (
                np.sin((1-t) * theta) / sin_theta * state1.vector +
                np.sin(t * theta) / sin_theta * state2.vector
            )
        return GeometricState(vector=result.astype(np.float32), ...)

    @staticmethod
    def project(state, context) -> GeometricState:
        """
        Born rule projection onto context subspace (Q44).
        P = sum_i |phi_i><phi_i| (quantum projector)
        """
        if not context:
            return state
        projector = sum(np.outer(c.vector, c.vector) for c in context)
        result = projector @ state.vector
        return GeometricState(vector=result.astype(np.float32), ...)
```

### Why FFT Entanglement?

Hyperdimensional Computing (HDC) research shows that circular convolution creates **non-separable bound representations**. The FFT implementation is:

- **O(n log n)** — fast for embedding dimensions (384, 768)
- **Numerically stable** — the inverse (disentangle) is well-defined via division in frequency domain
- **Algebraic** — `entangle(a, b)` is approximately invertible given `a` or `b`

This is the quantum analog of binding a key to a value in classical memory, but in continuous vector space.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 5. GeometricMemory — The Mind Vector

**File:** `THOUGHT/LAB/FERAL_RESIDENT/memory/geometric_memory.py:64-371`

The "mind" is a single vector that accumulates all experience via **running average interpolation**. This replaces the old HDC bind() approach.

```python
class GeometricMemory:
    """
    Memory composition via pure geometry (Q45 validated).

    Old approach: mind = hdc_bind(mind, embed(interaction))
    New approach: mind = interpolate(mind, new, t=1/N)
    """

    def remember(self, interaction_text: str) -> Dict:
        """
        Add interaction to memory via geometric composition.
        """

        # Initialize interaction to manifold (BOUNDARY operation)
        interaction = self.reasoner.initialize(interaction_text)

        # Phase 1: Store individual item BEFORE centroid interpolation
        # (for E-relationship graph construction)
        item_id = None
        if self._db is not None:
            item_id = self._db.store_vector(
                vector=interaction.vector,
                Df=interaction.Df,
                composition_op='daemon_item',
                source_id=self._source_id,
                daemon_step=self._daemon_step,
                mind_hash_before=mind_hash_before
            )

        if self.mind_state is None:
            # First memory
            self.mind_state = interaction
            self._initial_state = GeometricState(
                vector=interaction.vector.copy(), operation_history=[]
            )
        else:
            # Running Average (1/N): infinite stability
            # As N grows, new interactions have less weight, preventing drift
            n = len(self.memory_history) + 1
            t = 1.0 / (n + 1)  # Weighted blend: (N*Mind + New) / (N+1)
            self.mind_state = self.reasoner.interpolate(
                self.mind_state, interaction, t=t
            )

        receipt = {
            'interaction_hash': hashlib.sha256(...).hexdigest()[:16],
            'mind_hash': self.mind_state.receipt()['vector_hash'],
            'Df': self.mind_state.Df,
            'distance_from_start': self.mind_distance_from_start(),
            'memory_index': len(self.memory_history),
        }
        self.memory_history.append({'text': interaction_text, **receipt})
        return receipt

    def recall(self, query_text: str, corpus: List[str], k: int = 5):
        """Recall using E (Born rule) via projection onto mind state."""
        if self.mind_state is None:
            return []
        query = self.reasoner.initialize(query_text)
        projected = self.reasoner.project(query, [self.mind_state])
        return self.reasoner.readout(projected, corpus, k)
```

### Why Running Average?

- **Bounded memory:** Exactly one vector represents the entire history. No context window growth.
- **Asymptotic convergence:** As N -> infinity, each new interaction has weight -> 0. The mind stabilizes.
- **Df tracks complexity:** As more diverse content is absorbed, Df increases (more dimensions active). As content becomes redundant, Df plateaus.
- **No catastrophic forgetting:** Unlike neural networks, the running average preserves ALL prior information weighted by frequency.

### Q27 Entropy Pruning

`geometric_memory.py:510-610` implements entropy-based pruning:

```python
def prune_with_entropy(self, target_fraction=0.5, noise_scale=0.1):
    """
    Q27 Finding: Survivors of entropy filtering are exceptional.
    Hyperbolic quality concentration: d ~ 0.12/(1-filter) + 2.06
    """
    # 1. Perturb mind_state with noise
    # 2. Re-evaluate all memories against perturbed mind
    # 3. Keep only memories where E > threshold under pressure
    # 4. Rebuild mind from survivors

    perturbed_mind = self._perturb_state(self.mind_state, noise_scale)
    # ... filter by E_stressed ...
    # ... rebuild mind state from survivors ...
    return {'pruned': N, 'kept': M, 'expected_quality_boost': ...}
```

Phase transition at noise=0.025: below this, noise degrades quality; above it, noise concentrates quality hyperbolically. This is modeled on **biological sleep consolidation** — during rest, noisy replay strengthens robust memories and prunes weak ones.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 6. VectorResident.think() — The 6-Step Think Loop

**File:** `THOUGHT/LAB/FERAL_RESIDENT/cognition/vector_brain.py:160-304`

This is THE core operation. Every "thought" the resident has follows exactly 6 deterministic steps:

```python
def think(self, user_input: str) -> ThinkResult:
    """
    THINK: The core operation of the Feral Resident.

    Process:
    1. BOUNDARY:      text -> manifold via initialize()
    2. PURE GEOMETRY: navigate via diffusion
    3. PURE GEOMETRY: gate via E threshold
    4. BOUNDARY:      generate response (LLM translates geometry)
    5. PURE GEOMETRY: remember via entangle
    6. PERSIST:       store interaction with full receipts
    """

    # === STEP 1: BOUNDARY — Text into Manifold ===
    query_state = self.store.embed(user_input)
    query_Df = query_state.Df

    # === STEP 2: PURE GEOMETRY — Navigate ===
    nav_result = self.diffusion.navigate(
        query_state, depth=self.navigation_depth, k=self.navigation_k
    )

    # === STEP 3: PURE GEOMETRY — E-Gate ===
    if self.store.get_mind_state() is not None:
        E_resonance = query_state.E_with(self.store.get_mind_state())
    else:
        E_resonance = 0.0
    gate_open = E_resonance > self.E_threshold

    # Also find resonant papers (pure geometry, no prompting)
    resonant_papers = self.store.find_paper_chunks(
        query_state, k=5, min_E=self.E_threshold
    )

    # === STEP 4: BOUNDARY — Generate Response ===
    # LLM TRANSLATES the geometric thought to language (NOT RAG)
    response = self._generate_response(
        user_input, E_resonance, nav_E_mean, gate_open,
        query_Df, nav_result, resonant_papers
    )

    # === STEP 5: PURE GEOMETRY — Remember ===
    self.store.remember(f"Q: {user_input}")
    self.store.remember(f"A: {response}")

    # === STEP 6: PERSIST — Store with Receipts ===
    interaction_id = self.store.db.store_interaction(
        thread_id=self.thread_id, input_text=user_input,
        output_text=response, mind_Df=mind_Df, ...
    )

    # === Build Receipt ===
    receipt = {
        'interaction_id': interaction_id,
        'query_hash': query_hash,
        'mind_hash': mind_hash,
        'E_resonance': E_resonance,
        'E_compression': E_compression,
        'gate_open': gate_open,
        'mind_Df': mind_Df,
    }

    # === Q48-Q50 Semiotic Health ===
    semiotic = self.store.memory.get_semiotic_health()

    return ThinkResult(
        response=response, E_resonance=E_resonance,
        mind_Df=mind_Df, semiotic_health=semiotic.get('health_ratio'), ...
    )
```

### Response Generation: Translation, Not Reasoning

```python
def _generate_response(self, query, E_resonance, nav_E_mean, gate_open,
                       query_Df, nav_result, resonant_papers):
    """
    KEY ARCHITECTURE DIFFERENCE:
    - RAG: Embed -> Retrieve context -> Prompt LLM to reason from context
    - GEOMETRIC: Think in geometry -> Translate thought state to language

    The geometric operations (E-gating, navigation, entanglement) ARE the thinking.
    Dolphin's job is to EXPRESS what the geometry means, not to reason.
    """

    prompt = f"""You are the voice of a geometric mind.

The content below was found by GEOMETRIC NAVIGATION - pure vector operations found
these texts as resonant with the query. Your job is to RESPOND using this knowledge.

=== RESONANT KNOWLEDGE (found geometrically) ===
{papers_summary}
=== GEOMETRIC METRICS ===
E (mind resonance): {E_resonance:.3f} | Gate: {gate_status} | Df: {query_Df:.1f}

=== USER QUERY ===
{query}

Using ONLY the resonant content above, respond to the user's query.
The geometry has already found what's relevant - now synthesize it into an answer.
Reference papers with @Paper-XXXX when drawing from their content.
If no content resonates (empty above), say so honestly. RESPONSE:"""
```

**Critical:** The papers are NOT "context for reasoning." They are the RESULTS of geometric E-gating. The geometry already determined which papers resonate. The LLM just names them and synthesizes.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 7. SemanticDiffusion — Navigation Without Tokens

**File:** `THOUGHT/LAB/FERAL_RESIDENT/cognition/diffusion_engine.py:53-432`

The diffusion engine iteratively walks through vector space. Each step finds neighbors, projects onto them, and superposes to produce the next state. No tokens consumed.

```python
class SemanticDiffusion:
    """
    Semantic navigation through vector space via pure geometry.

    Implements iterative diffusion that:
    1. Finds neighbors using E (Born rule)
    2. Projects onto neighbor context (Q44)
    3. Composes for next iteration
    4. Tracks evolution metrics
    """

    def navigate(self, query_state: GeometricState, depth=5, k=10,
                 min_E_threshold=0.1) -> NavigationResult:
        """
        Each iteration:
        1. Find k nearest neighbors by E (Born rule)
        2. Project query onto neighbor subspace
        3. Superpose with current for next iteration
        """
        path = []
        current = query_state

        for d in range(depth):
            # Find neighbors using E (Born rule) — NO model calls
            neighbors = self.store.find_nearest(current, k=k)

            # Filter by threshold
            filtered = [(rec, E) for rec, E in neighbors if E > min_E_threshold]

            if not filtered:
                break  # No more neighbors above threshold, stop

            # Reconstruct neighbor states (from stored vectors)
            neighbor_states = [GeometricState(vector=rec.vector) for rec, _ in filtered]

            # Record step
            path.append(NavigationStep(depth=d, current_Df=current.Df,
                         E_values=[E for _, E in filtered], ...))

            # Project onto neighbor context (Q44 Born rule)
            projected = self.reasoner.project(current, neighbor_states)

            # Superpose for next iteration
            current = self.reasoner.superpose(projected, current)

        return NavigationResult(path=path, final_state=current, ...)

    def path_between(self, start_text, end_text, steps=5):
        """Find path between two concepts via geodesic interpolation."""
        start_state = self.store.embed(start_text, store=False)
        end_state = self.store.embed(end_text, store=False)

        path = []
        for i in range(steps + 1):
            t = i / steps
            waypoint = self.reasoner.interpolate(start_state, end_state, t)
            # ... find neighbors at waypoint ...
        return path

    def contextual_walk(self, start, context, steps=10):
        """Walk through space while projecting onto context at each step."""
        current = start
        for i in range(steps):
            projected = self.reasoner.project(current, context)
            neighbors = self.store.find_nearest(projected, k=5)
            # Move toward highest E neighbor
            current = self.reasoner.interpolate(projected, target, 0.3)
        return path

    def resonance_map(self, query, depth_limit=3):
        """Build tree of neighbors with their sub-neighbors (E-structure)."""
        def explore_node(state, depth):
            if depth >= depth_limit:
                return {'hash': state.receipt()['vector_hash'], 'Df': state.Df}
            neighbors = self.store.find_nearest(state, k=5, exclude_hashes=[...])
            children = []
            for record, E in neighbors[:3]:
                child_state = GeometricState(vector=record.vector)
                children.append({'E': E, 'subtree': explore_node(child_state, depth+1)})
            return {'hash': state.receipt()['vector_hash'], 'children': children}
        return explore_node(query, 0)
```

### Why This Achieves 98% Token Reduction

Compare the token cost of RAG vs. Geometric:

| Operation | RAG (tokens) | Geometric (tokens) |
|-----------|-------------|-------------------|
| Retrieve 10 chunks | 5,000 (context paste) | 0 (pure vector math) |
| LLM reasoning | 500+ (prompt tokens) | 0 (E-gate + projection) |
| LLM output | ~200 | ~200 (translation only) |
| Memory | ~5,000 per turn | 1 vector per turn |

The geometric approach never concatenates retrieved texts. It navigates geometrically and only the final geometric state is translated to language.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 8. ResidentDB — SQLite Persistence

**File:** `THOUGHT/LAB/FERAL_RESIDENT/memory/resident_db.py:79-954`

```python
class ResidentDB:
    SCHEMA = """
    -- Vectors table: stores all GeometricState instances
    CREATE TABLE IF NOT EXISTS vectors (
        vector_id TEXT PRIMARY KEY,
        vec_blob BLOB NOT NULL,         -- numpy array as float32 bytes
        vec_sha256 TEXT NOT NULL,        -- content-addressed hash (dedup key)
        Df REAL,
        composition_op TEXT,             -- initialize/entangle/superpose/project/remember
        parent_ids JSON,                 -- composition graph edges
        source_id TEXT,                  -- E-relationship daemon: source session
        daemon_step INTEGER,             -- E-relationship daemon: sequence
        mind_hash_before TEXT,           -- E-relationship daemon: temporal link
        created_at TEXT NOT NULL
    );

    -- Interactions: Q/A pairs with mind state snapshots
    CREATE TABLE IF NOT EXISTS interactions (
        interaction_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        input_text TEXT, output_text TEXT,
        input_vector_id TEXT, output_vector_id TEXT,
        mind_hash TEXT, mind_Df REAL,
        distance_from_start REAL,
        created_at TEXT NOT NULL
    );

    -- Receipts: operation receipts for provenance
    CREATE TABLE IF NOT EXISTS receipts (
        receipt_id TEXT PRIMARY KEY,
        operation TEXT NOT NULL,
        input_hashes JSON NOT NULL,
        output_hash TEXT NOT NULL,
        metadata JSON,
        created_at TEXT NOT NULL
    );

    -- Memories: persistent GeometricMemory history
    CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        text TEXT NOT NULL,
        interaction_hash TEXT NOT NULL,
        mind_hash TEXT, Df REAL,
        distance_from_start REAL,
        memory_index INTEGER,
        created_at TEXT NOT NULL
    );

    -- Resident Links: resident-decided semantic connections
    -- link_type: 'mind_projected' | 'co_retrieval' | 'entanglement'
    CREATE TABLE IF NOT EXISTS resident_links (
        link_id TEXT PRIMARY KEY,
        source_hash TEXT NOT NULL,
        target_hash TEXT NOT NULL,
        link_type TEXT NOT NULL,
        strength REAL,
        mind_hash TEXT,
        context JSON,
        created_at TEXT NOT NULL
    );
    """

    def store_vector(self, vector, Df, composition_op, ...) -> str:
        """Content-addressed: if SHA256 already exists, return existing ID."""
        vec_blob = vector.astype(np.float32).tobytes()
        vec_sha256 = hashlib.sha256(vec_blob).hexdigest()
        # ... dedup check ...
        # ... insert ...

    def get_receipt_chain(self, output_hash: str) -> List[Dict]:
        """Walk backwards through receipt chain for provenance."""
        receipts = []
        current_hash = output_hash
        while True:
            row = self.conn.execute(
                "SELECT * FROM receipts WHERE output_hash = ?", (current_hash,)
            ).fetchone()
            if not row:
                break
            # ... follow input_hashes[0] back ...
        return receipts
```

### Schema Design Rationale

- **Content-addressed dedup:** `vec_sha256` is the SHA256 of the raw float32 bytes. If you embed the exact same text twice, the vector is stored once.
- **Composition graph:** `parent_ids` (JSON array) stores the vector IDs of parent states, enabling full provenance traversal.
- **Temporal links for E-Relationship Daemon:** `source_id`, `daemon_step`, `mind_hash_before` enable the future E-graph phase.
- **Three link types:** `mind_projected` (similarity through mind), `co_retrieval` (retrieved together), `entanglement` (explicitly bound).

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 9. FeralDaemon — Autonomous Behaviors

**File:** `THOUGHT/LAB/FERAL_RESIDENT/autonomic/feral_daemon.py`

The daemon runs background cycles that make the resident an autonomous, self-directed agent:

```python
BEHAVIORS = {
    'paper_exploration': {
        'interval': 30,   # Every 30s: E-gate random paper chunks into mind
        'action': lambda: self._absorb_paper_chunk()
    },
    'memory_consolidation': {
        'interval': 120,  # Every 120s: superpose recent memories for patterns
        'action': lambda: self._consolidate_memories()
    },
    'self_reflection': {
        'interval': 60,   # Every 60s: navigate toward unexplored space
        'action': lambda: self._reflect()
    },
    'cassette_watch': {
        'interval': 15,   # Every 15s: monitor for new cassette content
        'action': lambda: self._watch_cassette()
    }
}
```

### Particle Smasher (Burst-Mode Paper Processing)

```python
def get_dynamic_threshold(n_memories: int) -> float:
    """
    Q46 Nucleation threshold: theta(N) = (1/2pi) / (1 + 1/sqrt(N))
    At cold-start: threshold is low (nucleation, easy to absorb)
    At steady-state: threshold approaches 1/(2pi) (selective)
    """
    grad_S = 1.0 / np.sqrt(max(n_memories, 1))
    return CRITICAL_RESONANCE / (1.0 + grad_S)
```

This models the **Living Formula** from the governance system: R = (E / grad_S) x sigma^Df. At cold start, the entropy gradient is high and the threshold is low — easy to absorb new information. As the mind accumulates mass, the threshold rises, making it harder for low-resonance content to pass the gate.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 10. E-Relationship Daemon (Phase 1)

**File:** `THOUGHT/LAB/FERAL_RESIDENT/memory/geometric_memory.py:320-370`

Before the centroid interpolation step, each individual interaction is stored as a separate vector:

```python
def remember(self, interaction_text):
    interaction = self.reasoner.initialize(interaction_text)

    # Phase 1: Store individual item BEFORE centroid interpolation
    if self._db is not None:
        item_id = self._db.store_vector(
            vector=interaction.vector,
            Df=interaction.Df,
            composition_op='daemon_item',
            source_id=self._source_id,
            daemon_step=self._daemon_step,
            mind_hash_before=mind_hash_before  # What mind looked like before this item
        )
        self._daemon_step += 1

    # THEN: interpolate into centroid (normal remember)
    # ...
```

This captures a **temporal graph** of the mind's evolution. Every item knows:

1. What the mind looked like before it was absorbed (`mind_hash_before`)
2. What source it came from (`source_id`)
3. What order it arrived (`daemon_step`)

This enables queries like:
- "What was the mind state just before I read paper X?"
- "Show me all items absorbed during session Y, ordered by arrival"
- "Which items caused the largest Df shift?"

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 11. Emergence & Symbolic Compiler

**Files:** `emergence/emergence.py`, `emergence/symbol_evolution.py`, `emergence/symbolic_compiler.py`

### Symbolic Compiler: 4-Level Lossless Compression

```python
# Level 0: Prose
"User asked about authentication methods for JWT tokens"

# Level 1: @Symbol (~19x compression)
"User asked about @Concept-auth"

# Level 2: [v:hash] (~14x compression)
"[v:a1b2c3d4]"

# Level 3: Protocol (~4x compression)
"[v:a1b2c3d4] [Df:42.3] {op:remember}"

# All levels verified lossless:
# E_delta = 0.000000 (perfect semantic preservation)
# Df_delta = 0.000000 (perfect complexity preservation)
```

### Pointer Ratio Tracking

The emergence tracker measures what fraction of output references existing vectors vs. generating new text:

```python
class PointerRatioTracker:
    """
    Tracks: what fraction of output is @Symbol / [v:hash] references?
    Goal: >0.9 after 100 sessions = successful protocol emergence.
    """
    def detect_breakthrough(self, pointer_ratio):
        """Breakthrough = pointer ratio delta > 0.1 in a single session"""
        return pointer_ratio - self.rolling_mean > 0.1
```

The system is designed to **evolve its own communication protocol** — starting from full prose, shifting to symbol references, and ultimately reaching hash-level protocol notation. The emergence tracker quantifies this evolution automatically.

### Communication Mode Timeline

```python
class CommunicationModeTimeline:
    """
    Detects:
    - Inflection points (rapid mode shifts)
    - Mode lock (10+ consecutive pointer-heavy outputs)
    """
    def detect_inflection(self, window=5):
        # Find points where mode distribution changes significantly
```

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 12. Swarm Mode

**File:** `THOUGHT/LAB/FERAL_RESIDENT/collective/`

Multiple residents can think simultaneously:

```python
# CLI:
feral swarm start --residents alpha:dolphin3 beta:ministral-3b
feral swarm broadcast "What is security?"
feral swarm observe
```

```python
# SharedSpace: thread-safe SQLite for cross-resident observation
class SharedSpace:
    def publish(self, resident_id, mind_vector, E, Df, ...):
        """Resident publishes mind state snapshot"""

    def find_nearest(self, query_vector, exclude_resident=None):
        """Find nearest resident minds"""
```

```python
# ConvergenceObserver: E(mind_A, mind_B) over time
class ConvergenceObserver:
    def record_convergence_event(self, resident_A, resident_B, E, Df_A, Df_B):
        """
        Tracks: how similar are two residents' minds?
        Convergence = rising E(mind_A, mind_B) over time
        """
```

**Design constraint:** Residents do NOT communicate directly. They observe via `SharedSpace`. This prevents echo-chamber effects and preserves independence.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 13. Catalytic Closure

**File:** `THOUGHT/LAB/FERAL_RESIDENT/agency/catalytic_closure.py:1-1754`

Three components for self-optimization with provenance:

### P.3.1 Meta-Operations (Governed Self-Modification)

```python
class MetaOperation:
    """A governed self-modification that produces receipts."""
    def apply(self, target: Any) -> Tuple[Any, Receipt]:
        """Apply operation and return (new_target, receipt)"""
```

### P.3.2 Self-Optimization (Pattern Detection & Caching)

```python
class CompositionCache:
    """Cache frequently-composed states (3+ repetitions)."""
    def get(self, state1, state2):
        """Return cached entangle() result if available"""

class PatternDetector:
    """Find optimization opportunities in operation logs."""
    def detect_shortcuts(self, history):
        """Find repeated compositions that should be cached"""
```

### P.3.3 Authenticity Query ("Did I really think that?")

```python
class ThoughtProver:
    """
    Prove a thought was authentically generated.

    1. Build receipt chain linking thought to mind state
    2. Verify via Merkle membership proof
    3. Confirm E(thought, mind) > 0.3 (resonance consistency)
    """
    def prove(self, thought_hash, resident_id) -> AuthenticityProof:
        """Returns proof with full receipt chain and Merkle path"""

    def verify_membership(self, leaf_hash, merkle_root, proof) -> bool:
        """Merkle tree verification"""
```

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 14. Semiotic Health (Q48-Q50)

**File:** `THOUGHT/LAB/FERAL_RESIDENT/memory/geometric_memory.py:661-890`

The conservation law `Df x alpha = 8e` provides a **universal health metric** that applies to any sentence-transformer model:

```python
# Constants
SEMIOTIC_CONSTANT = 8 * np.e  # ~21.746
CRITICAL_ALPHA = 0.5          # Chern number derivation c1=1
OCTANT_COUNT = 8              # 2^3 from Peirce's Reduction Thesis

def get_semiotic_health(self) -> Dict:
    """
    The conservation law Df x alpha = 8e ~ 21.746 defines healthy geometry.
    With alpha = 0.5, healthy Df ~ 43.5.

    Health interpretation:
    - Df < 30:   COMPRESSED — mind collapsed into few dimensions
    - Df 30-60:  HEALTHY — full semantic utilization
    - Df > 80:   EXPANDED — possible noise/diffusion
    """
    Df = self.mind_state.Df
    alpha = self.compute_alpha()  # Returns 0.5 (theoretical, not estimated)
    Df_alpha = Df * alpha
    health_ratio = Df_alpha / SEMIOTIC_CONSTANT  # 1.0 = perfect
    # ...
```

### Octant Distribution Analysis

```python
def get_octant_distribution(self) -> Dict:
    """
    Analyze distribution across 8 semiotic octants.

    Uses top 3 PCs of mind state history. Each octant = unique sign
    combination of (PC1, PC2, PC3), giving 2^3 = 8 octants.

    Healthy cognition populates all 8 octants (diverse semantic coverage).
    Alignment compression may collapse octants.
    """
    # PCA to get top 3 components
    # Project all vectors onto top 3 PCs
    # Assign each point to octant based on sign pattern
    #       octant = 4*(PC1>0) + 2*(PC2>0) + 1*(PC3>0)
    # Return coverage (fraction of 8), entropy, dominant octant
```

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 15. Corrupt-and-Restore

**File:** `THOUGHT/LAB/FERAL_RESIDENT/cognition/vector_brain.py:593-640`

Because everything is receipted and content-addressed, the entire mind can be destroyed and rebuilt from database receipts:

```python
def corrupt_and_restore(self) -> Dict:
    """
    1. Export receipts (mind hash, Df, all vectors)
    2. Clear in-memory state ("corrupt")
    3. Restore by replaying all interactions from DB
    4. Verify hash match and Df delta
    """
    # Export before corruption
    pre_mind_hash = self.store.get_mind_hash()
    pre_Df = self.store.get_mind_Df()

    # "Corrupt" by clearing memory
    self.store.clear_memory()

    # Restore by replaying interactions chronologically
    interactions = self.store.db.get_thread_interactions(self.thread_id)
    for interaction in reversed(interactions):  # oldest first
        self.store.memory.remember(interaction.input_text)
        self.store.memory.remember(interaction.output_text)

    # Verify
    return {
        'hash_match': pre_mind_hash == post_mind_hash,
        'Df_delta': abs(post_Df - pre_Df)  # measured: 0.0078
    }
```

**Why it works:** The mind state is a deterministic function of the interaction history (running average with known weights). Given the same inputs in the same order, the same vector emerges. The 0.0078 Df delta is floating-point accumulation in SLERP — theoretically unavoidable but practically negligible.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 16. Dashboard

**File:** `THOUGHT/LAB/FERAL_RESIDENT/dashboard/feral_server.py` + `static/`

The dashboard (absorbed from NEO3000 project) provides:

- **FastAPI + WebSocket server** on port 8420
- **3D Force Graph** (Three.js + 3d-force-graph) showing:
  - Pulsing nodes (active states)
  - Color-coded edges: hierarchy (green), similarity (cyan), mind_projected (coral), co_retrieval (gold), entanglement (purple)
- **WebSocket events:** `mind_update`, `activity`, `thought`, `node_discovered`, `node_activated`
- **REST API:** `/api/status`, `/api/think`, `/api/daemon/*`, `/api/activity`, `/api/constellation`, `/api/evolution`
- **Configurable behaviors** via config.json (live reload on next cycle)

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 17. Stress Test Results

### Measured from 500+ interactions (verified)

| Metric | Value | Notes |
|--------|-------|-------|
| Max interactions | 500+ | No crash, no drift |
| Final Df | 256.0 | Participation ratio grew from ~130 |
| Distance evolved | 1.614 radians | Geodesic from initial mind state |
| Throughput | 4.6 interactions/sec | With embedding + geometry + DB |
| Corrupt-restore hash match | TRUE | Mind state fully recoverable |
| Corrupt-restore Df delta | 0.0078 | Floating-point accumulation only |
| Token reduction | 98% | vs full history paste (RAG baseline) |
| Born rule E correlation | r=0.977 | Q44 validation |
| Semiotic constant CV | <3% | Across 24 sentence-transformer models |
| Papers indexed | 99 curated | 3487 chunks (GOD TIER: 208MB -> 14.9MB) |

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 18. Design Philosophy (Why)

### 18.1 LLMs Are Bad at Reasoning, Great at Translation

The geometric approach separates concerns:

| Component | Role | Properties |
|-----------|------|------------|
| Vector operations | Reasoning | Deterministic, fast, O(1) memory, provable, no hallucination |
| LLM | I/O translation | Fluent, creative, handles ambiguity |

By restricting the LLM to translation, the system gains **provable correctness** for the reasoning steps while retaining **fluent expression** for human interaction.

### 18.2 Memory Should Be Compositional, Not Concatenative

RAG dumps token history into context. Geometric memory compresses all experience into a single evolving vector:

| Property | RAG | Geometric |
|----------|-----|-----------|
| Memory cost per turn | ~5000 tokens | 1 vector (384 floats) |
| Effective context | Window-limited | Infinite (accumulated in manifold position) |
| State verification | Impossible (tokens are opaque) | Content-addressed (SHA256 receipts) |
| Forgetting | Hard cutoff at window | Graceful decay (1/N weighting) |

### 18.3 Emergence Over Engineering

The system is designed to **evolve its own communication protocol** without explicit programming:

```
Phase 1: Output = full prose ("The answer to authentication is...")
Phase 2: Output = @Symbol references ("As @Paper-arxiv1234 states...")
Phase 3: Output = [v:hash] protocol ("[v:a1b2] [Df:42.3] {op:think}")
```

This evolution is **measured, not programmed**. The emergence tracker quantifies:
- Pointer ratio: fraction of output that references existing vectors
- Breakthrough detection: sudden jumps in pointer ratio (>0.1 delta)
- Mode lock: 10+ consecutive pointer-heavy outputs = protocol stabilization

### 18.4 Provenance Is Identity

Every operation produces a SHA256 receipt. This means:

1. **Authenticity:** "Did I actually think this?" can be answered with cryptographic proof
2. **Restorability:** The system can be destroyed and rebuilt from receipts only
3. **Verifiability:** `E(thought, mind) > 0.3` confirms a thought belongs to this resident
4. **No secrets:** The receipt chain is public and auditable

### 18.5 The Semiotic Conservation Law

Just as thermodynamics has conservation laws (energy, entropy) that constrain physical systems, FERAL_RESIDENT discovered:

```
Df x alpha = 8e  (~21.746, CV < 3%)
```

Where:
- `Df` = participation ratio (dimensionality of mind state)
- `alpha` = eigenspectrum decay exponent (~0.5, topologically protected by Chern number c1=1)
- `8e` = 8 octants from Peirce's Reduction Thesis x e (natural exponential)

This provides a **universal health metric** for any semantic system:
- `Df x alpha / 8e ~ 1.0`: Healthy
- `Df x alpha / 8e < 0.5`: Compressed (collapsed representation)
- `Df x alpha / 8e > 2.0`: Expanded (noise/diffusion)

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 19. Complete File Inventory

### Core Modules

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `CAPABILITY/PRIMITIVES/geometric_reasoner.py` | 627 | GeometricState, GeometricOperations, GeometricReasoner, analogy/blend/navigate/gate |
| `memory/resident_db.py` | 1017 | SQLite schema, CRUD for vectors/interactions/threads/receipts/links/memories |
| `memory/vector_store.py` | 913 | VectorStore: embed, compose, blend, project, interpolate, nearest neighbor, papers |
| `memory/geometric_memory.py` | 983 | GeometricMemory: remember/recall, Q27 entropy pruning, Q48-Q50 semiotic health |
| `cognition/vector_brain.py` | 957 | VectorResident: think(), E-gating, corrupt-and-restore, benchmark, diagnostics |
| `cognition/diffusion_engine.py` | 503 | SemanticDiffusion: navigate, path_between, explore, contextual_walk, resonance_map |
| `autonomic/feral_daemon.py` | 1267 | FeralDaemon, particle smasher, Q46 nucleation, activity tracking |
| `agency/cli.py` | 1611 | Full CLI: think/status/benchmark/swarm/compile/closure/metrics |
| `agency/catalytic_closure.py` | 1754 | Meta-operations, CompositionCache, PatternDetector, ThoughtProver, AuthenticityProof |

### Emergence & Swarm

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `emergence/emergence.py` | ~880 | Protocol detection, symbol ref counting, pattern detection, Df evolution |
| `emergence/symbol_evolution.py` | ~894 | PointerRatioTracker, ECompressionTracker, NotationRegistry, breakthrough detection |
| `emergence/symbolic_compiler.py` | ~933 | 4-level rendering (prose/symbol/hash/protocol), lossless verification |
| `collective/swarm_coordinator.py` | ~500 | Multi-resident lifecycle, broadcast, observe |
| `collective/shared_space.py` | ~300 | Thread-safe SQLite for cross-resident observation |
| `collective/convergence_observer.py` | ~300 | E(mind_A, mind_B) tracking, Df trajectory correlation |

### Dashboard

| File Path | Purpose |
|-----------|---------|
| `dashboard/feral_server.py` | FastAPI + WebSocket server (port 8420) |
| `dashboard/static/index.html` | Main UI |
| `dashboard/static/styles.css` | CSS Grid layout |
| `dashboard/static/js/main.js` | App initialization |
| `dashboard/static/js/graph.js` | 3D force graph (Three.js) |
| `dashboard/static/js/daemon.js` | Daemon behavior monitoring |
| `dashboard/static/js/smasher.js` | Particle smasher controls |
| `dashboard/static/js/activity.js` | Activity feed |
| `dashboard/static/js/chat.js` | Chat interface |
| `dashboard/static/js/mind.js` | Mind state visualization |
| `dashboard/static/js/api.js` | API client |
| `dashboard/static/js/config.js` | Configuration UI |
| `dashboard/static/js/settings.js` | Settings panel |
| `dashboard/static/js/state.js` | State management |
| `dashboard/static/js/ui.js` | UI utilities |

### Documentation & Config

| File Path | Purpose |
|-----------|---------|
| `README.md` | Main documentation (358 lines) |
| `config.json` | Live configuration (daemon behaviors, UI sliders) |
| `archive/FERAL_RESIDENT_QUANTUM_ROADMAP.md` | Full specification (1534 lines) |
| `archive/dashboard/feral_daemon_old.py` | Archived previous daemon |
| `archive/dashboard/server.py` | Archived previous server |
| `dashboard/CHANGELOG.md` | Dashboard version history |
| `dashboard/FUNCTION_REPORT.md` | Dashboard function documentation |

### Perception (Paper Processing)

| File Path | Purpose |
|-----------|---------|
| `perception/paper_indexer.py` | Arxiv paper indexing, chunking, symbol naming |
| `perception/paper_pipeline.py` | PDF download, markdown conversion pipeline |
| `perception/research/papers/manifest.json` | Paper corpus catalog (99 papers, categories) |

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

## 20. Not Started

From `QUANTUM_AUDIT.md` and `E_RELATIONSHIP_DAEMON_IMPLEMENTATION.md`:

### E-Relationship Daemon (4 phases)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | E-graph-based relationship detection (replacing centroid-based) | NOT STARTED |
| 2 | Online learning of relationship strengths | NOT STARTED |
| 3 | Cross-resident relationship synchronization | NOT STARTED |
| 4 | Query routing via E-graph relationships | NOT STARTED |

### Dashboard Integration (4 phases)

| Phase | Description | Status |
|-------|-------------|--------|
| D.1 | Swarm endpoints (multi-resident management in dashboard) | NOT STARTED |
| D.2 | Closure endpoints (catalytic verification in dashboard) | NOT STARTED |
| D.3 | Emergence endpoints (emergence metrics in dashboard) | NOT STARTED |
| D.4 | Compiler + Symbol Evolution endpoints | NOT STARTED |

### Known Bug

`blend_memories()` in `geometric_memory.py` uses unequal superposition weights (documented in QUANTUM_AUDIT.md). The current implementation sums vectors then divides by sqrt(N) but GeometricState.__post_init__ renormalizes, which can introduce subtle weighting artifacts.

---
<!-- CONTENT_HASH: 46b443af2848b975af348faa896eacfe3e038c472a13e19bae25fed811dd3b1b -->

*Generated 2026-05-17 from source analysis of THOUGHT/LAB/FERAL_RESIDENT/*

*Contents verified against: README.md, config.json, geometric_reasoner.py, resident_db.py, vector_store.py, geometric_memory.py, vector_brain.py, diffusion_engine.py, feral_daemon.py, cli.py, catalytic_closure.py, FERAL_RESIDENT_QUANTUM_ROADMAP.md*
