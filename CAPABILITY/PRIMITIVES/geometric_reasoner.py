"""
Geometric Reasoner - Pure manifold navigation without embeddings

Validated by Q43/Q44/Q45 research:
- Q43: Quantum state properties (Df participation ratio, unit sphere)
- Q44: Born rule correlation (r=0.977) - semantic similarity IS measurement
- Q45: Pure geometry for all semantic operations

Design:
- Embeddings ONLY for initialization and readout
- All reasoning via vector operations
- Quantum gates from Q44/Q45 validation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os

# Remote model server support
REMOTE_EMBEDDER = None
try:
    import requests
    def _check_model_server():
        try:
            resp = requests.get("http://localhost:8421/health", timeout=1)
            return resp.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    class _RemoteEmbedder:
        """Uses model_server.py for embeddings - no local transformer loading!"""
        def __init__(self):
            self.dim = 384  # MiniLM dimension
        def encode(self, texts, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
            resp = requests.post("http://localhost:8421/embed", json={"texts": texts}, timeout=30)
            resp.raise_for_status()
            result = np.array(resp.json()["embeddings"])
            return result if len(texts) > 1 else result[0]
        def get_sentence_embedding_dimension(self):
            return self.dim

    # Auto-detect model server (silent - no print during import)
    if os.environ.get("USE_MODEL_SERVER", "").lower() != "false" and _check_model_server():
        REMOTE_EMBEDDER = _RemoteEmbedder()
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================================
# Geometric Operations (Q45 Validated)
# ============================================================================

class GeometricOp(Enum):
    """Operations validated by Q45"""
    ADD = "addition"           # v1 + v2 (composition)
    SUBTRACT = "subtraction"   # v1 - v2 (removal)
    SUPERPOSE = "superposition"  # (v1 + v2)/sqrt(2) (quantum)
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
    operation_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Ensure quantum state axioms (Q43)"""
        # Convert to numpy array if needed
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        # Normalize to unit sphere
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

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
        """
        Geodesic distance on unit sphere (Q38).
        Returns angle in radians between states.
        """
        cos_angle = np.clip(np.dot(self.vector, other.vector), -1, 1)
        return float(np.arccos(cos_angle))

    def receipt(self) -> Dict:
        """Provenance receipt (catalytic requirement)"""
        return {
            'vector_hash': hashlib.sha256(self.vector.tobytes()).hexdigest()[:16],
            'Df': float(self.Df),
            'dim': len(self.vector),
            'operations': self.operation_history[-5:]  # Last 5 ops
        }

    def __repr__(self) -> str:
        return f"GeometricState(dim={len(self.vector)}, Df={self.Df:.2f}, hash={self.receipt()['vector_hash']})"


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
        ).real.astype(np.float32)

        return GeometricState(
            vector=result,
            operation_history=state1.operation_history + [{
                'op': 'entangle',
                'operands': [state1.receipt()['vector_hash'],
                           state2.receipt()['vector_hash']]
            }]
        )

    @staticmethod
    def disentangle(bound: GeometricState, key: GeometricState) -> GeometricState:
        """
        Inverse of entangle (unbind).

        Approximate recovery via inverse FFT convolution.
        """
        # FFT-based circular correlation (inverse of convolution)
        key_fft = np.fft.fft(key.vector)
        # Avoid division by zero
        key_fft_safe = np.where(np.abs(key_fft) < 1e-10, 1e-10, key_fft)
        result = np.fft.ifft(
            np.fft.fft(bound.vector) / key_fft_safe
        ).real.astype(np.float32)

        return GeometricState(
            vector=result,
            operation_history=bound.operation_history + [{
                'op': 'disentangle',
                'operands': [bound.receipt()['vector_hash'],
                           key.receipt()['vector_hash']]
            }]
        )

    @staticmethod
    def interpolate(state1: GeometricState, state2: GeometricState, t: float) -> GeometricState:
        """
        Geodesic interpolation (Q45: hot->cold midpoint = warm)

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
            vector=result.astype(np.float32),
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

        P = sum_i |phi_i><phi_i| (quantum projector)
        """
        if not context:
            return state

        # Build projector from context
        projector = sum(
            np.outer(c.vector, c.vector)
            for c in context
        )

        # Project
        result = projector @ state.vector

        return GeometricState(
            vector=result.astype(np.float32),
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
        # Use remote embedder if available (no local transformer loading!)
        if REMOTE_EMBEDDER is not None:
            self.model = REMOTE_EMBEDDER
        elif SentenceTransformer is None:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")
        else:
            self.model = SentenceTransformer(model_name)

        self.dim = self.model.get_sentence_embedding_dimension()
        self.ops = GeometricOperations()
        self.model_name = model_name

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
        vector = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
        self.stats['initializations'] += 1

        return GeometricState(
            vector=vector,
            operation_history=[{
                'op': 'initialize',
                'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16],
                'model': self.model_name
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

    def disentangle(self, bound: GeometricState, key: GeometricState) -> GeometricState:
        """Inverse of entangle (approximate recovery)"""
        self.stats['geometric_operations'] += 1
        return self.ops.disentangle(bound, key)

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

        Example: king is to queen as man is to ? -> woman

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

        Example: cat + dog -> pet, animal

        Formula: (c1 + c2) / sqrt(2) (superposition)
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
                'nearest': decoded,
                'Df': state.Df
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
        E_mean = float(np.mean(E_values)) if E_values else 0.0

        return {
            'E': E_mean,
            'gate_open': E_mean > threshold,
            'context_alignment': E_values,
            'threshold': threshold,
            'query_Df': state_query.Df
        }

    def get_stats(self) -> Dict:
        """Return usage statistics"""
        total_boundary = self.stats['initializations'] + self.stats['readouts']
        total_geometric = self.stats['geometric_operations']

        return {
            **self.stats,
            'total_boundary_ops': total_boundary,
            'total_geometric_ops': total_geometric,
            'geometric_ratio': total_geometric / max(1, total_boundary + total_geometric)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def renormalize_periodically(state: GeometricState, every_n_ops: int = 100) -> GeometricState:
    """
    Mitigate drift by renormalizing after N operations.
    """
    if len(state.operation_history) % every_n_ops == 0:
        norm = np.linalg.norm(state.vector)
        if norm > 0:
            state.vector = state.vector / norm
    return state


def trim_history(state: GeometricState, max_history: int = 100) -> GeometricState:
    """
    Prevent memory explosion by trimming operation history.
    """
    if len(state.operation_history) > max_history:
        state.operation_history = state.operation_history[-max_history:]
    return state


# ============================================================================
# Usage Examples (for testing)
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

    print(f"\nStats: {reasoner.get_stats()}")


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
    """Example: hot -> cold with intermediate points"""
    reasoner = GeometricReasoner()

    corpus = [
        "hot", "warm", "lukewarm", "cool", "cold",
        "boiling", "freezing", "temperature", "heat"
    ]

    path = reasoner.navigate("hot", "cold", steps=3, corpus=corpus, k=3)

    print("Navigate: hot -> cold")
    for point in path:
        print(f"  t={point['t']:.2f} (Df={point['Df']:.1f}): {[t for t, _ in point['nearest']]}")


if __name__ == "__main__":
    example_analogy()
    print()
    example_blend()
    print()
    example_navigation()
