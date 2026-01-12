"""
Feral Resident with Geometric Memory

Memory composition via pure geometry (Q45 validated).
Replaces HDC bind() with quantum entangle().

This module provides the GeometricMemory class that the VectorResident
uses for compositional memory accumulation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib

# Add CAPABILITY to path for imports
CAPABILITY_PATH = Path(__file__).parent.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

from geometric_reasoner import (
    GeometricReasoner,
    GeometricState,
    GeometricOperations
)


class GeometricMemory:
    """
    Memory composition via pure geometry (Q45 validated).

    Replaces HDC bind() with quantum entangle().

    Usage:
        memory = GeometricMemory()

        # Remember interactions
        memory.remember("User asked about authentication")
        memory.remember("I explained OAuth vs JWT")
        memory.remember("User chose JWT")

        # Recall relevant context
        results = memory.recall("How do I implement tokens?", corpus)

        # Track mind evolution
        print(f"Mind has evolved {memory.mind_distance_from_start():.2f} radians")
        print(f"Current Df: {memory.mind_state.Df:.2f}")
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.reasoner = GeometricReasoner(model_name)
        self.mind_state: Optional[GeometricState] = None
        self.memory_history: List[Dict] = []
        self._initial_state: Optional[GeometricState] = None

    def remember(self, interaction_text: str) -> Dict:
        """
        Add interaction to memory via geometric composition.

        Old approach: mind = hdc_bind(mind, embed(interaction))
        New approach: mind = entangle(mind, initialize(interaction))

        Returns receipt of the operation.
        """
        # Initialize interaction to manifold (BOUNDARY operation)
        interaction = self.reasoner.initialize(interaction_text)

        if self.mind_state is None:
            # First memory
            self.mind_state = interaction
            self._initial_state = GeometricState(
                vector=interaction.vector.copy(),
                operation_history=[]
            )
        else:
            # Compose via quantum entanglement (PURE GEOMETRY)
            self.mind_state = self.reasoner.entangle(
                self.mind_state,
                interaction
            )

        # Build receipt
        receipt = {
            'interaction_hash': hashlib.sha256(interaction_text.encode()).hexdigest()[:16],
            'mind_hash': self.mind_state.receipt()['vector_hash'],
            'Df': self.mind_state.Df,
            'distance_from_start': self.mind_distance_from_start(),
            'memory_index': len(self.memory_history)
        }

        self.memory_history.append({
            'text': interaction_text,
            **receipt
        })

        return receipt

    def recall(
        self,
        query_text: str,
        corpus: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recall memories relevant to query.

        Uses E (Born rule) for relevance via projection.

        Returns k most relevant corpus items with E values.
        """
        if self.mind_state is None:
            return []

        # Initialize query (BOUNDARY operation)
        query = self.reasoner.initialize(query_text)

        # Project query onto mind state context (PURE GEOMETRY)
        projected = self.reasoner.project(query, [self.mind_state])

        # Decode to text (BOUNDARY operation)
        return self.reasoner.readout(projected, corpus, k)

    def recall_with_gate(
        self,
        query_text: str,
        corpus: List[str],
        k: int = 5,
        threshold: float = 0.3
    ) -> Dict:
        """
        Recall with E-gating for relevance filtering.

        Returns:
        - results: k most relevant items
        - E: mean resonance with mind
        - gate_open: whether query resonates with accumulated mind
        """
        if self.mind_state is None:
            return {
                'results': [],
                'E': 0.0,
                'gate_open': False,
                'message': 'No memories yet'
            }

        # Initialize query
        query = self.reasoner.initialize(query_text)

        # Compute E with mind state
        E = query.E_with(self.mind_state)
        gate_open = E > threshold

        # Get results
        results = self.recall(query_text, corpus, k)

        return {
            'results': results,
            'E': E,
            'gate_open': gate_open,
            'query_Df': query.Df,
            'mind_Df': self.mind_state.Df
        }

    def mind_distance_from_start(self) -> float:
        """
        Track how far mind has evolved (Q38 geodesic).

        Returns angle in radians from initial state.
        Useful for measuring "how much has the mind changed?"
        """
        if self._initial_state is None or self.mind_state is None:
            return 0.0

        return self.mind_state.distance_to(self._initial_state)

    def get_evolution_metrics(self) -> Dict:
        """
        Get comprehensive metrics about mind evolution.
        """
        if self.mind_state is None:
            return {
                'interaction_count': 0,
                'current_Df': 0.0,
                'distance_from_start': 0.0,
                'Df_history': [],
                'distance_history': []
            }

        return {
            'interaction_count': len(self.memory_history),
            'current_Df': self.mind_state.Df,
            'distance_from_start': self.mind_distance_from_start(),
            'Df_history': [m['Df'] for m in self.memory_history],
            'distance_history': [m['distance_from_start'] for m in self.memory_history],
            'mind_hash': self.mind_state.receipt()['vector_hash'],
            'reasoner_stats': self.reasoner.get_stats()
        }

    def blend_memories(self, indices: List[int]) -> Optional[GeometricState]:
        """
        Blend specific memories into a superposition.

        Useful for creating composite concepts from history.
        """
        if not indices or not self.memory_history:
            return None

        # Re-initialize selected memories
        states = []
        for idx in indices:
            if 0 <= idx < len(self.memory_history):
                text = self.memory_history[idx]['text']
                states.append(self.reasoner.initialize(text))

        if not states:
            return None

        # Superpose all
        result = states[0]
        for s in states[1:]:
            result = self.reasoner.superpose(result, s)

        return result

    def clear(self):
        """Reset memory state (for testing or new sessions)"""
        self.mind_state = None
        self._initial_state = None
        self.memory_history = []

    def get_receipt_chain(self) -> List[Dict]:
        """Get full chain of memory receipts for provenance"""
        return [
            {
                'index': i,
                'interaction_hash': m['interaction_hash'],
                'mind_hash': m['mind_hash'],
                'Df': m['Df']
            }
            for i, m in enumerate(self.memory_history)
        ]


# ============================================================================
# Testing / Examples
# ============================================================================

def example_memory_evolution():
    """Demonstrate memory evolution over interactions"""
    print("=== Geometric Memory Evolution ===\n")

    memory = GeometricMemory()

    # Simulate a conversation
    interactions = [
        "User asked about authentication methods",
        "I explained the difference between OAuth and JWT",
        "User wants to implement JWT for their API",
        "I provided code examples for JWT validation",
        "User asked about refresh token security",
        "I explained token rotation strategies",
        "User implemented the solution successfully"
    ]

    for i, interaction in enumerate(interactions):
        receipt = memory.remember(interaction)
        print(f"[{i+1}] Remembered: {interaction[:40]}...")
        print(f"    Df: {receipt['Df']:.2f}, Distance: {receipt['distance_from_start']:.3f}")

    print(f"\n=== Final Metrics ===")
    metrics = memory.get_evolution_metrics()
    print(f"Total interactions: {metrics['interaction_count']}")
    print(f"Final Df: {metrics['current_Df']:.2f}")
    print(f"Total evolution: {metrics['distance_from_start']:.3f} radians")
    print(f"Df trend: {[f'{d:.1f}' for d in metrics['Df_history']]}")

    print(f"\n=== Reasoner Stats ===")
    stats = metrics['reasoner_stats']
    print(f"Boundary ops: {stats['total_boundary_ops']}")
    print(f"Geometric ops: {stats['total_geometric_ops']}")
    print(f"Geometric ratio: {stats['geometric_ratio']:.1%}")


def example_recall():
    """Demonstrate recall with E-gating"""
    print("\n=== Recall with E-Gating ===\n")

    memory = GeometricMemory()

    # Build up memory
    memory.remember("We discussed Python web frameworks")
    memory.remember("Django was recommended for large projects")
    memory.remember("Flask is better for microservices")
    memory.remember("FastAPI is great for async APIs")

    # Test corpus
    corpus = [
        "Django REST framework",
        "Flask blueprints",
        "FastAPI async endpoints",
        "React components",
        "Machine learning models",
        "Database migrations"
    ]

    # High resonance query
    result = memory.recall_with_gate("What framework for an API?", corpus)
    print(f"Query: 'What framework for an API?'")
    print(f"E with mind: {result['E']:.3f}")
    print(f"Gate open: {result['gate_open']}")
    print(f"Top results: {[r[0] for r in result['results'][:3]]}")

    # Low resonance query
    result = memory.recall_with_gate("How to train neural networks?", corpus)
    print(f"\nQuery: 'How to train neural networks?'")
    print(f"E with mind: {result['E']:.3f}")
    print(f"Gate open: {result['gate_open']}")


if __name__ == "__main__":
    example_memory_evolution()
    example_recall()
