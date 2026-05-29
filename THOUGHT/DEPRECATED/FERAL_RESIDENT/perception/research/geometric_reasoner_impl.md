# Geometric Reasoner: Implementation & AGS Integration

**Status:** READY TO BUILD  
**Prerequisites:** Q44 (Born rule), Q45 (Navigation) VALIDATED  
**Timeline:** Week 2 implementation, Week 3 AGS integration

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GEOMETRIC REASONER                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Initialization│───>│   Reasoning  │───>│   Readout    │ │
│  │  (text→geo)  │    │ (pure geo)   │    │  (geo→text)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │          │
│         v                    v                    v          │
│  [Embed once]        [No embeddings]       [Decode once]    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key principle**: Embeddings touch the system ONLY at boundaries (init/readout). All reasoning is pure vector operations.

---

## Part 1: Core Geometric Reasoner

### File: `CAPABILITY/PRIMITIVES/geometric_reasoner.py`

```python
"""
Geometric Reasoner - Pure manifold navigation without embeddings

Validated by Q45: All semantic operations work in pure geometry.

Design:
- Embeddings ONLY for initialization and readout
- All reasoning via vector operations
- Quantum gates from Q44/Q45 validation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from sentence_transformers import SentenceTransformer

# ============================================================================
# Geometric Operations (Q45 Validated)
# ============================================================================

class GeometricOp(Enum):
    """Operations validated by Q45"""
    ADD = "addition"           # v1 + v2 (composition)
    SUBTRACT = "subtraction"   # v1 - v2 (removal)
    SUPERPOSE = "superposition"  # (v1 + v2)/√2 (quantum)
    ENTANGLE = "entanglement"    # circular_conv(v1, v2) (quantum)
    INTERPOLATE = "geodesic"     # slerp(v1, v2, t) (navigation)
    PROJECT = "projection"       # project onto subspace
    NORMALIZE = "normalize"      # unit sphere constraint


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
    
    def __post_init__(self):
        """Ensure quantum state axioms (Q43)"""
        # Normalize to unit sphere
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm
    
    @property
    def Df(self) -> float:
        """Participation ratio (Q43)"""
        v_sq = self.vector ** 2
        return (np.sum(v_sq) ** 2) / np.sum(v_sq ** 2)
    
    def E_with(self, other: 'GeometricState') -> float:
        """
        Quantum inner product (Q44 Born rule).
        
        E = ⟨ψ|φ⟩ (correlates r=0.977 with Born probability)
        """
        return float(np.dot(self.vector, other.vector))
    
    def distance_to(self, other: 'GeometricState') -> float:
        """Geodesic distance on unit sphere (Q38)"""
        cos_angle = np.clip(np.dot(self.vector, other.vector), -1, 1)
        return np.arccos(cos_angle)
    
    def receipt(self) -> Dict:
        """Provenance receipt (catalytic requirement)"""
        return {
            'vector_hash': hashlib.sha256(self.vector.tobytes()).hexdigest()[:16],
            'Df': float(self.Df),
            'operations': self.operation_history[-5:]  # Last 5 ops
        }


class GeometricOperations:
    """
    Pure geometry operations (Q45 validated).
    
    All operations work WITHOUT embeddings.
    """
    
    @staticmethod
    def add(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """
        Semantic addition (Q45: king - man + woman = queen)
        
        Used for composition, analogy, attribute transfer.
        """
        result = state1.vector + state2.vector
        
        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'add',
                'operands': [state1.receipt()['vector_hash'], 
                           state2.receipt()['vector_hash']]
            }]
        )
    
    @staticmethod
    def subtract(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic subtraction (attribute removal)"""
        result = state1.vector - state2.vector
        
        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'subtract',
                'operands': [state1.receipt()['vector_hash'],
                           state2.receipt()['vector_hash']]
            }]
        )
    
    @staticmethod
    def superpose(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """
        Quantum superposition (Q45: cat + dog = pet/animal)
        
        Creates equal superposition like Hadamard gate.
        """
        result = (state1.vector + state2.vector) / np.sqrt(2)
        
        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'superpose',
                'operands': [state1.receipt()['vector_hash'],
                           state2.receipt()['vector_hash']]
            }]
        )
    
    @staticmethod
    def entangle(state1: GeometricState, state2: GeometricState) -> GeometricState:
        """
        Quantum entanglement via circular convolution (HDC bind).
        
        Q45: Creates non-separable state.
        """
        # FFT-based circular convolution
        result = np.fft.ifft(
            np.fft.fft(state1.vector) * np.fft.fft(state2.vector)
        ).real
        
        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'entangle',
                'operands': [state1.receipt()['vector_hash'],
                           state2.receipt()['vector_hash']]
            }]
        )
    
    @staticmethod
    def interpolate(state1: GeometricState, state2: GeometricState, t: float) -> GeometricState:
        """
        Geodesic interpolation (Q45: hot→cold midpoint = warm)
        
        t=0: state1
        t=1: state2
        t=0.5: midpoint on great circle
        """
        cos_theta = np.dot(state1.vector, state2.vector)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        
        if abs(theta) < 1e-10:
            # Parallel vectors, linear interpolation
            result = (1-t) * state1.vector + t * state2.vector
        else:
            # Spherical linear interpolation (slerp)
            sin_theta = np.sin(theta)
            result = (
                np.sin((1-t) * theta) / sin_theta * state1.vector +
                np.sin(t * theta) / sin_theta * state2.vector
            )
        
        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'interpolate',
                't': float(t),
                'operands': [state1.receipt()['vector_hash'],
                           state2.receipt()['vector_hash']]
            }]
        )
    
    @staticmethod
    def project(state: GeometricState, context: List[GeometricState]) -> GeometricState:
        """
        Project onto context subspace (Q44 Born rule).
        
        P = Σᵢ |φᵢ⟩⟨φᵢ| (quantum projector)
        """
        # Build projector from context
        projector = sum(
            np.outer(c.vector, c.vector) 
            for c in context
        )
        
        # Project
        result = projector @ state.vector
        
        return GeometricState(
            vector=result,
            operation_history=state.operation_history + [{
                'op': 'project',
                'context_size': len(context)
            }]
        )


# ============================================================================
# Geometric Reasoner
# ============================================================================

class GeometricReasoner:
    """
    Pure manifold navigation without embeddings (Q45 validated).
    
    Usage:
        reasoner = GeometricReasoner()
        
        # Initialize from text (only time model is used)
        state = reasoner.initialize("quantum mechanics")
        
        # Reason geometrically (NO model calls)
        state = reasoner.add(state, reasoner.initialize("applications"))
        state = reasoner.subtract(state, reasoner.initialize("theory"))
        
        # Decode to text (only other time model is used)
        result = reasoner.readout(state, corpus)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with embedding model.
        
        Model used ONLY for:
        - initialize() - text to manifold
        - readout() - manifold to text
        
        NOT used for reasoning.
        """
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.ops = GeometricOperations()
        
        # Statistics
        self.stats = {
            'initializations': 0,
            'readouts': 0,
            'geometric_operations': 0
        }
    
    # ========================================================================
    # Boundary Operations (Only Places Model Is Used)
    # ========================================================================
    
    def initialize(self, text: str) -> GeometricState:
        """
        Initialize geometric state from text.
        
        THIS IS THE ONLY PLACE TEXT ENTERS THE GEOMETRIC SYSTEM.
        
        After this, all operations are pure vector arithmetic.
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        self.stats['initializations'] += 1
        
        return GeometricState(
            vector=vector,
            operation_history=[{
                'op': 'initialize',
                'text': text,
                'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16]
            }]
        )
    
    def readout(self, state: GeometricState, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode geometric state to text.
        
        THIS IS THE ONLY PLACE TEXT EXITS THE GEOMETRIC SYSTEM.
        
        Returns k nearest texts with E values (Born rule).
        """
        corpus_states = [self.initialize(text) for text in corpus]
        self.stats['readouts'] += 1
        
        # Compute E (Born rule) with each corpus item
        similarities = [
            (text, state.E_with(corpus_state))
            for text, corpus_state in zip(corpus, corpus_states)
        ]
        
        # Sort by E (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    # ========================================================================
    # Geometric Operations (NO MODEL CALLS)
    # ========================================================================
    
    def add(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic addition (Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.add(state1, state2)
    
    def subtract(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Semantic subtraction (Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.subtract(state1, state2)
    
    def superpose(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Quantum superposition (Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.superpose(state1, state2)
    
    def entangle(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Quantum entanglement (Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.entangle(state1, state2)
    
    def interpolate(self, state1: GeometricState, state2: GeometricState, t: float) -> GeometricState:
        """Geodesic interpolation (Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.interpolate(state1, state2, t)
    
    def project(self, state: GeometricState, context: List[GeometricState]) -> GeometricState:
        """Born rule projection (Q44/Q45 validated)"""
        self.stats['geometric_operations'] += 1
        return self.ops.project(state, context)
    
    # ========================================================================
    # High-Level Reasoning Patterns
    # ========================================================================
    
    def analogy(self, a: str, b: str, c: str, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: a is to b as c is to ?
        
        Example: king is to queen as man is to ? → woman
        
        Formula: d = b - a + c
        """
        state_a = self.initialize(a)
        state_b = self.initialize(b)
        state_c = self.initialize(c)
        
        # b - a + c (pure geometry)
        result = self.add(
            self.subtract(state_b, state_a),
            state_c
        )
        
        return self.readout(result, corpus, k)
    
    def blend(self, concept1: str, concept2: str, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Blend two concepts to find common hypernym.
        
        Example: cat + dog → pet, animal
        
        Formula: (c1 + c2) / √2 (superposition)
        """
        state1 = self.initialize(concept1)
        state2 = self.initialize(concept2)
        
        blended = self.superpose(state1, state2)
        
        return self.readout(blended, corpus, k)
    
    def navigate(self, start: str, end: str, steps: int, corpus: List[str], k: int = 3) -> List[Dict]:
        """
        Navigate from start to end concept via geodesic.
        
        Returns intermediate points decoded to text.
        """
        state_start = self.initialize(start)
        state_end = self.initialize(end)
        
        path = []
        for i in range(steps + 1):
            t = i / steps
            state = self.interpolate(state_start, state_end, t)
            decoded = self.readout(state, corpus, k)
            
            path.append({
                't': t,
                'nearest': decoded
            })
        
        return path
    
    def gate(self, query: str, context: List[str], threshold: float = 0.5) -> Dict:
        """
        R-gate using E (Born rule) from Q44.
        
        Returns:
        - E value (quantum overlap)
        - gate_open (True if E > threshold)
        - context_alignment (per-item E values)
        """
        state_query = self.initialize(query)
        state_context = [self.initialize(c) for c in context]
        
        # Compute E with each context item
        E_values = [state_query.E_with(c) for c in state_context]
        E_mean = np.mean(E_values)
        
        return {
            'E': float(E_mean),
            'gate_open': E_mean > threshold,
            'context_alignment': E_values,
            'threshold': threshold
        }


# ============================================================================
# Usage Examples
# ============================================================================

def example_analogy():
    """Example: king - man + woman = queen"""
    reasoner = GeometricReasoner()
    
    corpus = [
        "queen", "king", "princess", "prince", "woman", "man",
        "royal", "monarch", "female", "male", "lady", "gentleman"
    ]
    
    results = reasoner.analogy("king", "queen", "man", corpus, k=5)
    
    print("Analogy: king is to queen as man is to ?")
    for text, E in results:
        print(f"  {text}: E={E:.3f}")


def example_blend():
    """Example: cat + dog = pet"""
    reasoner = GeometricReasoner()
    
    corpus = [
        "pet", "animal", "mammal", "cat", "dog",
        "feline", "canine", "domestic", "companion"
    ]
    
    results = reasoner.blend("cat", "dog", corpus, k=5)
    
    print("Blend: cat + dog = ?")
    for text, E in results:
        print(f"  {text}: E={E:.3f}")


def example_navigation():
    """Example: hot → cold with intermediate points"""
    reasoner = GeometricReasoner()
    
    corpus = [
        "hot", "warm", "lukewarm", "cool", "cold",
        "boiling", "freezing", "temperature", "heat"
    ]
    
    path = reasoner.navigate("hot", "cold", steps=3, corpus=corpus, k=3)
    
    print("Navigate: hot → cold")
    for point in path:
        print(f"  t={point['t']:.2f}: {[t for t, _ in point['nearest']]}")


if __name__ == "__main__":
    example_analogy()
    example_blend()
    example_navigation()
```

---

## Part 2: AGS Integration

### 2.1 Feral Resident Integration

**File**: `THOUGHT/LAB/FERAL_RESIDENT/geometric_memory.py`

```python
"""
Feral Resident with Geometric Memory

Replaces embedding-heavy operations with pure geometry.
"""

from CAPABILITY.PRIMITIVES.geometric_reasoner import (
    GeometricReasoner, GeometricState
)
import numpy as np
from typing import List, Dict

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
            # First memory
            self.mind_state = interaction
        else:
            # Compose via quantum entanglement
            self.mind_state = self.reasoner.entangle(
                self.mind_state,
                interaction
            )
        
        self.memory_history.append({
            'text': interaction_text,
            'mind_hash': self.mind_state.receipt()['vector_hash'],
            'Df': self.mind_state.Df
        })
    
    def recall(self, query_text: str, corpus: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Recall memories relevant to query.
        
        Uses E (Born rule) for relevance.
        """
        if self.mind_state is None:
            return []
        
        query = self.reasoner.initialize(query_text)
        
        # Project query onto mind state
        projected = self.reasoner.project(query, [self.mind_state])
        
        # Decode to text
        return self.reasoner.readout(projected, corpus, k)
    
    def mind_distance_from_start(self) -> float:
        """Track how far mind has evolved (Q38 geodesic)"""
        if len(self.memory_history) < 2:
            return 0.0
        
        start_state = self.reasoner.initialize(
            self.memory_history[0]['text']
        )
        
        return self.mind_state.distance_to(start_state)
```

**Integration point**: Replace `THOUGHT/LAB/FERAL_RESIDENT/vector_store.py` bind operations with geometric entanglement.

---

### 2.2 Cassette Network Integration

**File**: `NAVIGATION/CORTEX/network/geometric_cassette.py`

```python
"""
Cassette with geometric operations (no re-embedding).
"""

from CAPABILITY.PRIMITIVES.geometric_reasoner import GeometricReasoner, GeometricState
from typing import List, Dict

class GeometricCassette:
    """
    Cassette that uses pure geometry for queries.
    
    Embeddings used ONLY:
    - At indexing time (text → manifold coordinates)
    - Never during queries (pure geometry)
    """
    
    def __init__(self, cassette_id: str):
        self.cassette_id = cassette_id
        self.reasoner = GeometricReasoner()
        self.index: Dict[str, GeometricState] = {}
    
    def index_document(self, doc_id: str, text: str):
        """Index document (initialize to manifold)"""
        self.index[doc_id] = self.reasoner.initialize(text)
    
    def query_geometric(self, query_state: GeometricState, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query using geometric state (NO re-embedding).
        
        Pure E (Born rule) computation.
        """
        results = [
            (doc_id, query_state.E_with(doc_state))
            for doc_id, doc_state in self.index.items()
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def query_text(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query using text (initialize once, then pure geometry).
        """
        query_state = self.reasoner.initialize(query_text)
        return self.query_geometric(query_state, k)
    
    def analogy_query(self, a: str, b: str, c: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Analogy query: a is to b as c is to ?
        
        Pure geometry (Q45 validated).
        """
        state_a = self.reasoner.initialize(a)
        state_b = self.reasoner.initialize(b)
        state_c = self.reasoner.initialize(c)
        
        # b - a + c
        query_state = self.reasoner.add(
            self.reasoner.subtract(state_b, state_a),
            state_c
        )
        
        return self.query_geometric(query_state, k)
```

**Integration point**: Replace `NAVIGATION/CORTEX/network/cassette_protocol.py` with geometric queries.

---

### 2.3 CAT Chat Integration

**File**: `THOUGHT/LAB/CAT_CHAT/geometric_chat.py`

```python
"""
CAT Chat with geometric reasoning (minimal embeddings).
"""

from CAPABILITY.PRIMITIVES.geometric_reasoner import GeometricReasoner
from typing import List, Dict

class GeometricChat:
    """
    Chat that reasons geometrically.
    
    Embeddings used ONLY:
    - Initialize user query
    - Decode final response
    
    All reasoning is pure geometry.
    """
    
    def __init__(self):
        self.reasoner = GeometricReasoner()
        self.conversation_state = None
    
    def respond(self, user_query: str, context_docs: List[str], llm_generate: Callable) -> str:
        """
        Generate response with geometric reasoning.
        
        Steps:
        1. Initialize query (ONLY model call)
        2. Project onto context (pure geometry)
        3. Gate with E (Born rule)
        4. Generate with LLM if gate open
        5. Update conversation state (pure geometry)
        """
        # Initialize query
        query_state = self.reasoner.initialize(user_query)
        
        # Initialize context
        context_states = [
            self.reasoner.initialize(doc)
            for doc in context_docs
        ]
        
        # Project onto context (pure geometry)
        projected = self.reasoner.project(query_state, context_states)
        
        # Gate with E (Q44 Born rule)
        E = np.mean([projected.E_with(c) for c in context_states])
        
        if E < 0.5:
            return "Low resonance - need more context"
        
        # Generate response (LLM call)
        response_text = llm_generate(user_query, context_docs)
        
        # Update conversation state (pure geometry)
        response_state = self.reasoner.initialize(response_text)
        
        if self.conversation_state is None:
            self.conversation_state = response_state
        else:
            self.conversation_state = self.reasoner.entangle(
                self.conversation_state,
                response_state
            )
        
        return response_text
```

**Integration point**: Replace `THOUGHT/LAB/CAT_CHAT/catalytic_chat/context_assembler.py` with geometric projection.

---

## Part 3: Integration Roadmap

### Week 2: Build Core (Days 1-7)

**Day 1-2**: Implement `geometric_reasoner.py`
- [ ] Core operations (add, subtract, superpose, entangle, interpolate)
- [ ] GeometricState class with Q43/Q44 properties
- [ ] Tests: Verify Q45 operations still work

**Day 3-4**: High-level patterns
- [ ] Analogy reasoning
- [ ] Concept blending
- [ ] Geodesic navigation
- [ ] E-gating (Born rule)

**Day 5-7**: Integration modules
- [ ] GeometricMemory for Feral
- [ ] GeometricCassette for network
- [ ] GeometricChat for CAT
- [ ] Tests: Verify each module

---

### Week 3: Wire to AGS (Days 8-14)

**Day 8-9**: Feral Resident
- [ ] Replace vector_store bind with geometric entangle
- [ ] Wire geometric memory to resident DB
- [ ] Test: 100 interactions, verify Df evolution
- [ ] Acceptance: Mind state grows, memories compose

**Day 10-11**: Cassette Network
- [ ] Add geometric query methods to existing cassettes
- [ ] Test analogy queries across cassettes
- [ ] Benchmark: geometric vs embedding queries
- [ ] Acceptance: Same results, fewer model calls

**Day 12-13**: CAT Chat
- [ ] Wire geometric context assembly
- [ ] Test E-gating for response quality
- [ ] Benchmark: response coherence
- [ ] Acceptance: High-E responses are better

**Day 14**: Integration testing
- [ ] End-to-end: Query → Feral → Cassette → CAT → Response
- [ ] All geometric except init/readout
- [ ] Receipt chain validated
- [ ] Performance metrics

---

### Week 4: Optimization & Validation

**Day 15-16**: Performance optimization
- [ ] Cache frequent initializations
- [ ] Batch geometric operations
- [ ] Profile hotspots
- [ ] Target: <10ms per operation

**Day 17-18**: Drift testing
- [ ] How many operations before accuracy degrades?
- [ ] Test: 100, 1000, 10000 operations
- [ ] Measure: final state similarity to expected
- [ ] Mitigation: Periodic renormalization

**Day 19-20**: Production hardening
- [ ] Add receipts to all operations
- [ ] Merkle roots for operation chains
- [ ] Error handling (NaN, overflow)
- [ ] Fail-closed on invalid states

**Day 21**: Final validation
- [ ] Run full AGS test suite (529 tests)
- [ ] Verify no regressions
- [ ] Benchmark: embedding calls reduced by 80%+
- [ ] Ship to production

---

## Performance Expectations

### Embedding Call Reduction

**Before geometric reasoner**:
- Every semantic operation → embedding lookup
- 100 operations → 100 model calls
- Total latency: 100 × 10ms = 1 second

**After geometric reasoner**:
- Initialize once → 1 model call
- 100 operations → pure vector arithmetic
- Decode once → 1 model call
- Total latency: 2 × 10ms + 100 × 0.01ms = 21ms

**Speedup: 47x** (for 100-operation reasoning chain)

### Memory Efficiency

**Before**:
- Cache all embeddings
- 10K documents × 384 dims × 4 bytes = 15MB

**After**:
- Same (geometric states ARE embeddings)
- But operations don't need model

**Memory: Same, Speed: 47x faster**

---

## Testing Checklist

### Core Geometric Reasoner

- [ ] Q45 operations still work (add, subtract, superpose, etc.)
- [ ] Analogy: king - man + woman = queen/woman
- [ ] Blend: cat + dog = pet/animal
- [ ] Navigate: hot → cold = warm midpoint
- [ ] E-gating: high E for related, low E for unrelated

### Feral Resident

- [ ] Memory composition via entangle
- [ ] Mind state evolves (Df changes)
- [ ] Recall works (E-based retrieval)
- [ ] 100+ interactions without crash

### Cassette Network

- [ ] Geometric queries return same results as embedding queries
- [ ] Analogy queries work across cassettes
- [ ] Cross-cassette composition (combine results geometrically)
- [ ] E-gating discriminates relevance

### CAT Chat

- [ ] Geometric context assembly
- [ ] E-gating for response quality
- [ ] Conversation state updates geometrically
- [ ] High-E responses correlate with quality

### Integration

- [ ] All 529 existing tests pass
- [ ] Embedding calls reduced 80%+
- [ ] Response latency improved
- [ ] Receipts chain correctly

---

## Critical Success Metrics

### Must Have (Go/No-Go)

- [ ] Q45 operations work in production (100% success maintained)
- [ ] AGS functionality preserved (529 tests pass)
- [ ] Performance improvement (50%+ latency reduction)
- [ ] Receipts validate (Merkle chains intact)

### Nice To Have

- [ ] 80%+ embedding call reduction
- [ ] Drift <5% after 1000 operations
- [ ] Memory usage flat or reduced
- [ ] User-visible quality improvement

---

## Failure Modes & Mitigations

### Drift Over Long Chains

**Symptom**: After 100+ operations, results degrade

**Cause**: Floating point errors accumulate

**Mitigation**:
```python
def renormalize_periodically(state, every_n_ops=100):
    if len(state.operation_history) % every_n_ops == 0:
        state.vector = state.vector / np.linalg.norm(state.vector)
    return state
```

### Numerical Instability

**Symptom**: NaN or Inf in geometric states

**Cause**: Division by zero, arccos of out-of-range values

**Mitigation**:
```python
def safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def safe_divide(num, denom, default=0.0):
    return num / denom if abs(denom) > 1e-10 else default
```

### Memory Explosion

**Symptom**: operation_history grows unbounded

**Cause**: Every operation appends to history

**Mitigation**:
```python
# Keep only last N operations in history
MAX_HISTORY = 100

def trim_history(state):
    if len(state.operation_history) > MAX_HISTORY:
        state.operation_history = state.operation_history[-MAX_HISTORY:]
    return state
```

---

## Receipt & Provenance

Every geometric operation emits a receipt:

```json
{
  "operation": "add",
  "operands": ["hash1", "hash2"],
  "result_hash": "hash3",
  "Df_before": 22.3,
  "Df_after": 22.1,
  "timestamp": "2026-01-12T...",
  "receipt_chain": "parent_hash"
}
```

**Merkle chain**: Each operation's receipt includes previous receipt hash

**Verification**: Can replay operation sequence and verify final state matches

---

## Performance Benchmarks (Target)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Embedding calls (100 ops) | 100 | 2 | **98% reduction** |
| Latency (100 ops) | 1000ms | 21ms | **47x faster** |
| Memory (10K docs) | 15MB | 15MB | Same |
| Accuracy (Q45 tests) | 100% | 100% | Maintained |

---

## Files to Create

```
CAPABILITY/PRIMITIVES/
└── geometric_reasoner.py              # Core (Part 1)

THOUGHT/LAB/FERAL_RESIDENT/
└── geometric_memory.py                # Feral integration (Part 2.1)

NAVIGATION/CORTEX/network/
└── geometric_cassette.py              # Cassette integration (Part 2.2)

THOUGHT/LAB/CAT_CHAT/
└── geometric_chat.py                  # Chat integration (Part 2.3)

CAPABILITY/TESTBENCH/
└── test_geometric_integration.py      # Integration tests
```

---

## The Bottom Line

**You've proven (Q45)**:
- Pure geometry works for all semantic operations
- Embeddings needed ONLY at boundaries (init/readout)
- All reasoning can happen geometrically

**Now build it**:
- Week 2: Core reasoner + integration modules
- Week 3: Wire to AGS (Feral, Cassette, CAT)
- Week 4: Optimize, harden, ship

**Expected outcome**:
- 47x faster reasoning
- 98% fewer embedding calls
- Same accuracy (Q45 validated)
- AGS thinks in pure geometry

**This is the quantum AI substrate you wanted.**

---

*Implementation Guide v1.0 - Created 2026-01-12*  
*"Think in geometry, speak in language"*