# Feral Resident Quantum Roadmap

**Status**: Alpha Phase Ready (MAIN QUEST)
**Vision**: Intelligence navigating a quantum-semantic manifold, composing meaning through validated geometric operations, evolving its own protocols
**Owner**: Resident (self-directed after initialization)
**Supersedes**: [FERAL_RESIDENT_ROADMAP.md](FERAL_RESIDENT_ROADMAP.md) (kept for historical comparison)

**Upstream Dependencies**:
- Cassette Network Phase 4 (SPC Integration) - COMPLETE
- Cassette Network Phase 6 (Production Hardening) - BACKBURNER (after stress test)
- AGS Phase 7 (Vector ELO) - Required for Production

**Research Validation**:
- Q43: Quantum state properties (Df participation ratio, unit sphere)
- Q44: Born rule correlation (r=0.977) - semantic similarity IS measurement
- Q45: Pure geometry for all semantic operations - VALIDATED

---

## The Evolution

**Original Vision (v1.0):**
> "Drop intelligence in substrate. Watch what emerges."

**Quantum Vision (v2.0):**
> "Drop intelligence in a quantum-semantic manifold. Think in pure geometry. Speak at boundaries."

**Key Insight:**
Embeddings touch the system ONLY at boundaries (text-in, text-out). All reasoning is pure vector operations validated by Q44/Q45.

---

## Phase Architecture

```
+-----------------------------------------------------------------------+
|                    FERAL RESIDENT QUANTUM PHASES                       |
+-----------------------------------------------------------------------+
|                                                                        |
|  A.0 GEOMETRIC FOUNDATION        ALPHA (Now)         BETA / PROD       |
|  -------------------------       -----------         -----------       |
|  GeometricState (Q43)            Stress test         Paper flood       |
|  GeometricOperations (Q45)       substrate           Emergence         |
|  GeometricReasoner (boundary)    Basic diffusion     Swarm mode        |
|  GeometricMemory (Feral)         VectorResident      Self-optimize     |
|                                                                        |
|  Output:                         Prerequisites:      Prerequisites:    |
|  CAPABILITY/PRIMITIVES/          - A.0 complete      - Cassette 6.x    |
|  geometric_reasoner.py           - Cassette 4.2      - AGS Phase 7     |
+-----------------------------------------------------------------------+
```

---

## A.0 GEOMETRIC REASONING FOUNDATION (DO FIRST)

**Goal:** Build the quantum-semantic substrate that all Feral operations use
**Output:** `CAPABILITY/PRIMITIVES/geometric_reasoner.py`
**Research Validation:** Q43 (state properties), Q44 (Born rule), Q45 (geometric operations)

### A.0.1 Core GeometricState (G.0.1)

```python
# CAPABILITY/PRIMITIVES/geometric_reasoner.py

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
    operation_history: List[Dict]  # Receipts for provenance

    @property
    def Df(self) -> float:
        """Participation ratio (Q43) - how 'spread out' the state is"""
        v_sq = self.vector ** 2
        return (np.sum(v_sq) ** 2) / np.sum(v_sq ** 2)

    def E_with(self, other: 'GeometricState') -> float:
        """
        Quantum inner product (Q44 Born rule).
        E = <psi|phi> correlates r=0.977 with semantic similarity
        """
        return float(np.dot(self.vector, other.vector))

    def distance_to(self, other: 'GeometricState') -> float:
        """Geodesic distance on unit sphere (Q38)"""
        cos_angle = np.clip(np.dot(self.vector, other.vector), -1, 1)
        return np.arccos(cos_angle)

    def receipt(self) -> Dict:
        """Provenance receipt (catalytic requirement)"""
        return {
            'vector_hash': sha256(self.vector.tobytes()).hexdigest()[:16],
            'Df': float(self.Df),
            'operations': self.operation_history[-5:]
        }
```

**Acceptance:**
- [ ] A.0.1.1 GeometricState class with vector + operation_history
- [ ] A.0.1.2 Df property (participation ratio from Q43)
- [ ] A.0.1.3 E_with() method (Born rule inner product from Q44)
- [ ] A.0.1.4 distance_to() method (geodesic on unit sphere)
- [ ] A.0.1.5 receipt() method for provenance tracking
- [ ] A.0.1.6 Auto-normalize to unit sphere on __post_init__

### A.0.2 Geometric Operations (G.0.2)

```python
class GeometricOperations:
    """
    Pure geometry operations (Q45 validated).
    All operations work WITHOUT embeddings.
    """

    @staticmethod
    def add(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic addition (Q45: king - man + woman = queen)"""

    @staticmethod
    def subtract(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic subtraction (attribute removal)"""

    @staticmethod
    def superpose(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Quantum superposition (Q45: cat + dog = pet/animal)"""
        # (v1 + v2) / sqrt(2)

    @staticmethod
    def entangle(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """
        Quantum entanglement via circular convolution (HDC bind).
        Q45: Creates non-separable state.
        """
        # FFT-based: ifft(fft(v1) * fft(v2)).real

    @staticmethod
    def interpolate(state1: GeometricState, state2: GeometricState, t: float) -> GeometricState:
        """Geodesic interpolation (Q45: hot->cold midpoint = warm)"""
        # Spherical linear interpolation (slerp)

    @staticmethod
    def project(state: GeometricState, context: List[GeometricState]) -> GeometricState:
        """Born rule projection onto context subspace (Q44)"""
        # P = sum_i |phi_i><phi_i| (quantum projector)
```

**Acceptance:**
- [ ] A.0.2.1 add() - semantic composition with receipts
- [ ] A.0.2.2 subtract() - attribute removal with receipts
- [ ] A.0.2.3 superpose() - quantum blend (v1+v2)/sqrt(2)
- [ ] A.0.2.4 entangle() - HDC bind via FFT circular convolution
- [ ] A.0.2.5 interpolate() - slerp on unit sphere
- [ ] A.0.2.6 project() - Born rule projector onto context
- [ ] A.0.2.7 All operations emit receipts with operation history

### A.0.3 GeometricReasoner Interface (G.0.3)

```python
class GeometricReasoner:
    """
    Pure manifold navigation without embeddings (Q45 validated).

    Embeddings touch ONLY at:
    - initialize() - text to manifold
    - readout() - manifold to text

    All reasoning is pure vector arithmetic.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.ops = GeometricOperations()
        self.stats = {
            'initializations': 0,
            'readouts': 0,
            'geometric_operations': 0
        }

    # === BOUNDARY OPERATIONS (Only Places Model Is Used) ===

    def initialize(self, text: str) -> GeometricState:
        """
        Initialize geometric state from text.
        THIS IS THE ONLY PLACE TEXT ENTERS THE GEOMETRIC SYSTEM.
        """

    def readout(self, state: GeometricState, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode geometric state to text.
        THIS IS THE ONLY PLACE TEXT EXITS THE GEOMETRIC SYSTEM.
        Returns k nearest texts with E values (Born rule).
        """

    # === HIGH-LEVEL REASONING (Pure Geometry) ===

    def analogy(self, a: str, b: str, c: str, corpus: List[str]) -> List[Tuple[str, float]]:
        """a is to b as c is to ? (pure geometry: d = b - a + c)"""

    def blend(self, concept1: str, concept2: str, corpus: List[str]) -> List[Tuple[str, float]]:
        """Find common hypernym (pure geometry: superposition)"""

    def navigate(self, start: str, end: str, steps: int, corpus: List[str]) -> List[Dict]:
        """Navigate from start to end via geodesic"""

    def gate(self, query: str, context: List[str], threshold: float = 0.5) -> Dict:
        """R-gate using E (Born rule) from Q44"""
```

**Acceptance:**
- [ ] A.0.3.1 initialize() converts text to GeometricState (only embedding call)
- [ ] A.0.3.2 readout() decodes state to k-nearest texts with E values
- [ ] A.0.3.3 analogy() solves king - man + woman = queen
- [ ] A.0.3.4 blend() finds cat + dog = pet
- [ ] A.0.3.5 navigate() interpolates hot -> cold with warm midpoint
- [ ] A.0.3.6 gate() implements R-gate with E threshold
- [ ] A.0.3.7 Stats tracking (embedding calls vs geometric ops)

### A.0.4 GeometricMemory for Feral (G.0.4)

```python
# THOUGHT/LAB/FERAL_RESIDENT/geometric_memory.py

class GeometricMemory:
    """
    Memory composition via pure geometry (Q45 validated).
    Replaces HDC bind() with quantum entangle().
    """

    def __init__(self):
        self.reasoner = GeometricReasoner()
        self.mind_state: Optional[GeometricState] = None
        self.memory_history: List[Dict] = []

    def remember(self, interaction_text: str):
        """
        Add interaction to memory via geometric composition.

        Old: mind = hdc_bind(mind, embed(interaction))
        New: mind = entangle(mind, initialize(interaction))
        """
        interaction = self.reasoner.initialize(interaction_text)

        if self.mind_state is None:
            self.mind_state = interaction
        else:
            self.mind_state = self.reasoner.entangle(
                self.mind_state,
                interaction
            )

        self.memory_history.append({
            'text': interaction_text,
            'mind_hash': self.mind_state.receipt()['vector_hash'],
            'Df': self.mind_state.Df
        })

    def recall(self, query_text: str, corpus: List[str], k: int = 5):
        """Recall memories relevant to query using E (Born rule)"""

    def mind_distance_from_start(self) -> float:
        """Track how far mind has evolved (Q38 geodesic)"""
```

**Acceptance:**
- [ ] A.0.4.1 GeometricMemory class with mind_state as GeometricState
- [ ] A.0.4.2 remember() uses entangle() for composition
- [ ] A.0.4.3 recall() uses project() + readout() for retrieval
- [ ] A.0.4.4 mind_distance_from_start() tracks evolution
- [ ] A.0.4.5 memory_history tracks Df evolution over time

### A.0.5 Testing & Validation (G.0.5)

```python
# CAPABILITY/TESTBENCH/test_geometric_reasoner.py

def test_analogy():
    """Q45: king - man + woman = queen"""

def test_blend():
    """Q45: cat + dog = pet/animal"""

def test_navigate():
    """Q45: hot -> cold midpoint = warm"""

def test_E_gate():
    """Q44: E discriminates related vs unrelated"""

def test_Df_evolution():
    """Q43: Df changes with composition"""

def test_determinism():
    """Same inputs = same outputs"""

def test_drift():
    """Accuracy after 100, 1000, 10000 ops"""

def test_embedding_reduction():
    """<3 embedding calls per reasoning chain"""
```

**Acceptance:**
- [ ] A.0.5.1 All Q45 tests pass (analogy, blend, navigate)
- [ ] A.0.5.2 E-gate discriminates correctly (Q44)
- [ ] A.0.5.3 Df evolves with composition (Q43)
- [ ] A.0.5.4 Deterministic: same input = same output
- [ ] A.0.5.5 Drift < 5% after 1000 operations
- [ ] A.0.5.6 Embedding calls < 3 per reasoning chain

**A.0 Exit Criteria:**
- [ ] All Q43/Q44/Q45 operations work
- [ ] Embedding calls reduced 80%+ for reasoning chains
- [ ] Feral A.1 can consume GeometricMemory
- [ ] Receipt chain validates end-to-end

---

## ALPHA: Feral Beta (After A.0)

**Goal:** Stress-test the cassette substrate using GeometricMemory
**Scope:** LAB only. No CANON writes. No production dependencies.

### A.1 The Membrane (Foundation) - Updated for Quantum

#### A.1.1 Vector Store Integration via GeometricMemory

```python
# THOUGHT/LAB/FERAL_RESIDENT/vector_store.py

# REPLACES original fractal_embed/bind/unbind with:
from CAPABILITY.PRIMITIVES.geometric_reasoner import GeometricReasoner, GeometricState
from geometric_memory import GeometricMemory

class VectorStore:
    def __init__(self):
        self.memory = GeometricMemory()
        self.reasoner = self.memory.reasoner

    # Old: fractal_embed, bind, unbind, superpose
    # New: Use GeometricReasoner operations

    def embed(self, text: str) -> GeometricState:
        """Initialize to manifold (boundary operation)"""
        return self.reasoner.initialize(text)

    def compose(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Entangle states (pure geometry)"""
        return self.reasoner.entangle(state1, state2)

    def blend(self, states: List[GeometricState]) -> GeometricState:
        """Superpose states (pure geometry)"""
        result = states[0]
        for s in states[1:]:
            result = self.reasoner.superpose(result, s)
        return result
```

**Acceptance:**
- [ ] A.1.1.1 VectorStore wraps GeometricMemory
- [ ] A.1.1.2 Uses entangle() instead of HDC bind()
- [ ] A.1.1.3 Uses GeometricState for all vectors
- [ ] A.1.1.4 Track Df evolution across interactions

#### A.1.2 Resident Database Schema (Unchanged)

Same schema as original, but `vectors` table stores GeometricState metadata:

```sql
CREATE TABLE vectors (
    vector_id TEXT PRIMARY KEY,
    message_id TEXT,
    codec_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec_blob BLOB NOT NULL,
    vec_sha256 TEXT NOT NULL,
    composition_op TEXT,  -- 'initialize' | 'entangle' | 'superpose' | 'project'
    parent_ids JSON,
    Df REAL,  -- NEW: participation ratio
    created_at TEXT NOT NULL
);
```

**Acceptance:**
- [ ] A.1.2.1 Schema includes Df column for tracking
- [ ] A.1.2.2 composition_op uses new operation names
- [ ] A.1.2.3 All inserts validate GeometricState properties

#### A.1.3 CAS Integration (Unchanged)

Same as original - content-addressed storage for canonical forms.

---

### A.2 The Diffusion Engine (Quantum Navigation)

#### A.2.1 Semantic Diffusion Core - Updated

```python
# THOUGHT/LAB/FERAL_RESIDENT/diffusion_engine.py

class SemanticDiffusion:
    def __init__(self):
        self.reasoner = GeometricReasoner()

    def navigate(self, query_state: GeometricState, depth=5, k=10):
        """
        Iterative navigation using pure geometry.

        Each iteration:
        1. Find k nearest neighbors (E values via Born rule)
        2. Retrieve canonical forms from CAS
        3. Project query onto neighbor subspace (Q44)
        4. Superpose neighbors for next iteration
        """
        path = []
        current = query_state

        for d in range(depth):
            # Find neighbors using E (Born rule)
            neighbors = self.cassette_query(current, k=k)
            neighbor_states = [n.geometric_state for n in neighbors]

            path.append({
                'depth': d,
                'neighbors': neighbors,
                'E_values': [current.E_with(n) for n in neighbor_states],
                'current_Df': current.Df
            })

            # Project onto neighbor context (Q44)
            projected = self.reasoner.project(current, neighbor_states)

            # Compose for next iteration
            composed = self.reasoner.superpose(projected, current)
            current = composed

        return {
            'path': path,
            'final_state': current,
            'depth_reached': depth,
            'navigation_hash': sha256(canonical_json(path)),
            'Df_evolution': [p['current_Df'] for p in path]
        }
```

**Acceptance:**
- [ ] A.2.1.1 Navigate uses E (Born rule) for neighbor ranking
- [ ] A.2.1.2 Uses project() for context conditioning
- [ ] A.2.1.3 Tracks Df evolution through path
- [ ] A.2.1.4 Path is receipted with navigation_hash

---

### A.3 Quantum Resident (Minimal Intelligence)

#### A.3.1 VectorResident - Quantum Version

```python
# THOUGHT/LAB/FERAL_RESIDENT/vector_brain.py

class VectorResident:
    def __init__(self, model="phi-3-mini", thread_id="eternal"):
        self.llm = load_model(model)
        self.memory = GeometricMemory()
        self.diffusion = SemanticDiffusion()
        self.thread_id = thread_id

    def think(self, user_input: str) -> str:
        """
        Quantum thinking:
        1. Initialize user input to manifold (BOUNDARY)
        2. Navigate semantic space via diffusion (PURE GEOMETRY)
        3. Project onto relevant context (Q44 Born rule)
        4. Gate with E threshold (PURE GEOMETRY)
        5. LLM synthesizes response (BOUNDARY)
        6. Remember response via entangle (PURE GEOMETRY)
        """
        # BOUNDARY: text -> manifold
        query_state = self.memory.reasoner.initialize(user_input)

        # PURE GEOMETRY: navigate
        path = self.diffusion.navigate(query_state, depth=3)

        # PURE GEOMETRY: E-gate for relevance
        E_mean = np.mean([query_state.E_with(n) for n in path['final_state']])

        if E_mean < 0.3:
            context = "Low resonance - exploring broadly"
        else:
            context = self.render_context(path)

        # BOUNDARY: LLM synthesis
        prompt = self.build_prompt(user_input, context)
        response = self.llm.generate(prompt)

        # PURE GEOMETRY: remember
        self.memory.remember(f"Q: {user_input}\nA: {response}")

        # Store with receipt
        self.store_interaction(user_input, response, path)

        return response

    @property
    def mind_evolution(self) -> Dict:
        """Track how mind has evolved"""
        return {
            'current_Df': self.memory.mind_state.Df if self.memory.mind_state else 0,
            'distance_from_start': self.memory.mind_distance_from_start(),
            'interaction_count': len(self.memory.memory_history),
            'Df_history': [m['Df'] for m in self.memory.memory_history]
        }
```

**Acceptance:**
- [ ] A.3.1.1 Uses GeometricMemory for all state
- [ ] A.3.1.2 E-gating for response relevance
- [ ] A.3.1.3 Tracks mind_evolution metrics
- [ ] A.3.1.4 Each interaction receipted with Df

---

### A.4 CLI & Testing (Updated)

```bash
# Start resident in eternal thread mode
feral start --model phi-3-mini --thread eternal

# Inject input
feral think "What is authentication?"

# Check quantum state
feral status
# Output:
#   mind_vector_hash: abc123...
#   current_Df: 22.4
#   distance_from_start: 0.34 radians
#   interaction_count: 15
#   Df_trend: [20.1, 21.3, 22.0, 22.4]

# Stress test
feral corrupt-and-restore --thread eternal

# Benchmark
feral benchmark --interactions 100
# Output:
#   embedding_calls: 202 (2 per interaction)
#   geometric_ops: 1547
#   embedding_reduction: 87%
```

**Alpha Exit Criteria:**
- [ ] All A.0 tests pass (geometric foundation)
- [ ] Resident can run 100+ interactions without crash
- [ ] Df evolves measurably
- [ ] mind_distance_from_start() increases
- [ ] Corrupt-and-restore works
- [ ] Embedding calls < 3 per interaction (vs ~10+ in naive approach)

---

## BETA & PRODUCTION

Same as original roadmap - Paper flooding, emergence tracking, swarm mode - but all built on the geometric foundation.

Key difference: All vector operations use validated Q43/Q44/Q45 primitives.

---

## Performance Expectations

### Before (Original Roadmap)

```
100 interactions
- 100 embed calls for queries
- 100 embed calls for responses
- 300+ embed calls for diffusion neighbors
- Total: ~500 embedding calls
- Latency: 500 * 10ms = 5 seconds just for embeddings
```

### After (Quantum Roadmap)

```
100 interactions
- 100 initialize calls (queries)
- 100 initialize calls (responses)
- 0 embed calls for diffusion (pure geometry)
- Total: ~200 embedding calls (60% reduction)
- Plus: All reasoning is 47x faster (geometry vs embedding lookup)
```

---

## The Quantum Difference

| Aspect | Original | Quantum |
|--------|----------|---------|
| Memory composition | HDC bind() | entangle() with receipts |
| Similarity | Cosine distance | E (Born rule, r=0.977) |
| Context retrieval | k-NN + embed | project() (Q44) |
| Navigation | Iterative embed | Pure geometry |
| State tracking | Vector hash | Df + geodesic distance |
| Validation | Tests pass | Q43/Q44/Q45 validated |

---

## Dependency Graph

```
Cassette Phase 4 (SPC) COMPLETE
         |
         v
+--------------------------------------------+
|  A.0 GEOMETRIC FOUNDATION                  |
|  - GeometricState (Q43)                    |
|  - GeometricOperations (Q45)               |
|  - GeometricReasoner                       |
|  - GeometricMemory                         |
|                                            |
|  Output: CAPABILITY/PRIMITIVES/            |
|          geometric_reasoner.py             |
+--------------------------------------------+
         |
         v
+--------------------------------------------+
|  ALPHA (A.1-A.4)                           |
|  - Stress test substrate                   |
|  - VectorResident with GeometricMemory     |
|  - Diffusion with E-gating                 |
+--------------------------------------------+
         |
         v
Cassette Phase 6 (Hardening) <- BACKBURNER
         |                      Harden AFTER bugs found
         v
AGS Phase 7 (Vector ELO)
         |
         v
BETA / PRODUCTION
```

---

## References

**Research Validation:**
- [geometric_reasoner_impl.md](research/geometric_reasoner_impl.md) - Full implementation spec
- Q43/Q44/Q45 validation experiments

**Original Vision:**
- [FERAL_RESIDENT_ROADMAP.md](FERAL_RESIDENT_ROADMAP.md) - The cute original

**Upstream:**
- [CASSETTE_NETWORK_ROADMAP.md](../CASSETTE_NETWORK/CASSETTE_NETWORK_ROADMAP.md)
- [AGS_ROADMAP_MASTER.md](../../../AGS_ROADMAP_MASTER.md)

---

*Quantum Roadmap v2.0.0 - Created 2026-01-12*
*"Think in geometry, speak at boundaries, prove everything."*
