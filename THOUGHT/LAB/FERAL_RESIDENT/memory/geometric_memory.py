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
import warnings
import numpy as np

# Add CAPABILITY to path for imports
# memory/ -> FERAL_RESIDENT/ -> LAB/ -> THOUGHT/ -> repo/
FERAL_PATH = Path(__file__).parent.parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

from geometric_reasoner import (
    GeometricReasoner,
    GeometricState,
    GeometricOperations
)

# ============================================================================
# Q27 Entropy Toolkit Constants
# ============================================================================
# Phase transition threshold: below this, noise degrades quality (additive)
# Above this, noise concentrates quality hyperbolically (multiplicative)
PHASE_TRANSITION_THRESHOLD = 0.025
DEFAULT_FILTER_NOISE = 0.1
CRITICAL_RESONANCE = 1.0 / (2.0 * np.pi)  # Q46 threshold


def get_dynamic_threshold(n_memories: int) -> float:
    """Q46 nucleation threshold: θ(N) = (1/2π) / (1 + 1/√N)"""
    grad_S = 1.0 / np.sqrt(max(n_memories, 1))
    return CRITICAL_RESONANCE / (1.0 + grad_S)


class GeometricMemory:
    """
    Memory composition via pure geometry (Q45 validated).

    Replaces HDC bind() with running average interpolation.

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
        # Q27 Entropy Control: 0=permissive, >0.025=selective filtering
        self.temperature: float = 0.0

    def remember(self, interaction_text: str) -> Dict:
        """
        Add interaction to memory via geometric composition.

        Old approach: mind = hdc_bind(mind, embed(interaction))
        New approach: mind = interpolate(mind, new, t=1/N)

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
            # Use Running Average (1/N) to provide infinite stability
            # As N grows, new interactions have less weight, preventing drift
            n = len(self.memory_history) + 1  # Count includes this new memory
            t = 1.0 / (n + 1)  # Weighted blend: (N*Mind + New) / (N+1)
            
            self.mind_state = self.reasoner.interpolate(
                self.mind_state,
                interaction,
                t=t
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
        Blend specific memories into equal-weight superposition.

        Creates: (v1 + v2 + ... + vN) / sqrt(N) then normalize.
        Each memory has equal contribution weight ~1/N.

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

        # Equal-weight superposition: sum all vectors / sqrt(N)
        # Then GeometricState.__post_init__ normalizes to unit sphere
        import numpy as np
        result_vector = sum(s.vector for s in states) / np.sqrt(len(states))

        return GeometricState(
            vector=result_vector,
            operation_history=[{
                'op': 'blend',
                'count': len(states),
                'indices': indices
            }]
        )

    def clear(self):
        """Reset memory state (for testing or new sessions)"""
        self.mind_state = None
        self._initial_state = None
        self.memory_history = []
        self.temperature = 0.0

    # ========================================================================
    # Q27 Entropy Toolkit Methods
    # ========================================================================

    def _perturb_state(
        self,
        state: GeometricState,
        noise_scale: float
    ) -> GeometricState:
        """
        Apply Gaussian noise to a geometric state (Q27 entropy filtering).

        Args:
            state: GeometricState to perturb
            noise_scale: Standard deviation of Gaussian noise
                - Must be > 0.025 for multiplicative quality concentration
                - Range (0, 0.025) is "danger zone" - degrades quality

        Returns:
            New GeometricState with noise applied (normalized to unit sphere)
        """
        if noise_scale <= 0:
            return state

        if 0 < noise_scale < PHASE_TRANSITION_THRESHOLD:
            warnings.warn(
                f"noise_scale {noise_scale} is in danger zone (0, {PHASE_TRANSITION_THRESHOLD}). "
                f"Noise will DEGRADE quality, not improve it."
            )

        noise = np.random.randn(len(state.vector)) * noise_scale
        perturbed = state.vector + noise
        # GeometricState.__post_init__ normalizes to unit sphere

        return GeometricState(
            vector=perturbed.astype(np.float32),
            operation_history=state.operation_history + [{'op': 'perturb', 'scale': noise_scale}]
        )

    def E_under_pressure(
        self,
        item_text: str,
        noise_scale: float = DEFAULT_FILTER_NOISE
    ) -> float:
        """
        Compute E value against perturbed mind state.

        Items with high E_under_pressure are robustly aligned with mind direction.
        Q27 finding: robust items are exceptional, not just good.

        Args:
            item_text: Text to evaluate
            noise_scale: Noise intensity (default 0.1, must be > 0.025)

        Returns:
            E value (cosine similarity) with perturbed mind state
        """
        if self.mind_state is None:
            return 0.0

        item = self.reasoner.initialize(item_text)
        perturbed_mind = self._perturb_state(self.mind_state, noise_scale)
        return item.E_with(perturbed_mind)

    def set_temperature(self, T: float):
        """
        Set system temperature (selectivity level).

        Q27 Phase Transition:
        - T = 0.0: Normal operation, no entropy filtering
        - T in (0, 0.025): DANGER ZONE - noise degrades quality
        - T > 0.025: Multiplicative regime - quality concentration

        Higher temperature = more selective intake (fewer but better memories).
        """
        if 0 < T < PHASE_TRANSITION_THRESHOLD:
            warnings.warn(
                f"Temperature {T} is in danger zone (0, {PHASE_TRANSITION_THRESHOLD}). "
                f"Either use T=0 or T>={PHASE_TRANSITION_THRESHOLD}"
            )
        self.temperature = T

    def confidence_score(
        self,
        item_text: str,
        noise_levels: List[float] = None
    ) -> Dict:
        """
        Measure robustness of item under increasing noise pressure.

        Q27 Insight: Items that maintain high E under pressure are robustly
        aligned with the mind's direction, not just coincidentally similar.

        Args:
            item_text: Text to evaluate
            noise_levels: List of noise scales to test (default: [0.05, 0.1, 0.15, 0.2])

        Returns:
            Dict with:
            - survival_rate: fraction of noise levels where E > threshold
            - E_profile: dict mapping noise_level -> E value
            - robustness: mean E across all noise levels
            - confidence: alias for survival_rate
        """
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.15, 0.2]

        if self.mind_state is None:
            return {
                'survival_rate': 0.0,
                'E_profile': {},
                'robustness': 0.0,
                'confidence': 0.0,
                'message': 'No mind state'
            }

        item = self.reasoner.initialize(item_text)
        threshold = get_dynamic_threshold(len(self.memory_history))

        E_profile = {}
        survivals = 0

        for noise in noise_levels:
            perturbed = self._perturb_state(self.mind_state, noise)
            E = item.E_with(perturbed)
            E_profile[noise] = E
            if E > threshold:
                survivals += 1

        survival_rate = survivals / len(noise_levels) if noise_levels else 0.0

        return {
            'survival_rate': survival_rate,
            'E_profile': E_profile,
            'robustness': float(np.mean(list(E_profile.values()))) if E_profile else 0.0,
            'threshold': threshold,
            'confidence': survival_rate
        }

    def prune_with_entropy(
        self,
        target_fraction: float = 0.5,
        noise_scale: float = DEFAULT_FILTER_NOISE,
        threshold: float = None
    ) -> Dict:
        """
        Prune memories using entropy-based selection pressure.

        Q27 Finding: Survivors of entropy filtering are exceptional, not just good.
        Hyperbolic quality concentration: d ≈ 0.12/(1-filter) + 2.06

        Mechanism:
        1. Perturb mind_state with noise
        2. Re-evaluate all memories against perturbed mind
        3. Keep only memories where E > threshold under pressure
        4. Rebuild mind from survivors

        Args:
            target_fraction: Approximate fraction of memories to keep (0.0-1.0)
            noise_scale: Noise intensity (must be > 0.025 for quality concentration)
            threshold: E threshold for survival. If None, computed from target_fraction.

        Returns:
            Dict with pruning statistics
        """
        if not self.memory_history:
            return {'pruned': 0, 'kept': 0, 'message': 'No memories to prune'}

        if noise_scale < PHASE_TRANSITION_THRESHOLD:
            warnings.warn(
                f"noise_scale {noise_scale} is below phase transition {PHASE_TRANSITION_THRESHOLD}. "
                f"Quality concentration effect will be weak or negative."
            )

        # Perturb mind state
        perturbed_mind = self._perturb_state(self.mind_state, noise_scale)

        # Evaluate all memories under pressure
        scored_memories = []
        for i, mem in enumerate(self.memory_history):
            item = self.reasoner.initialize(mem['text'])
            E_stressed = item.E_with(perturbed_mind)
            scored_memories.append((i, mem, E_stressed))

        # Sort by E_stressed (highest first)
        scored_memories.sort(key=lambda x: x[2], reverse=True)

        # Determine cutoff
        if threshold is None:
            keep_count = max(1, int(len(scored_memories) * target_fraction))
            if keep_count < len(scored_memories):
                threshold = scored_memories[keep_count - 1][2]
            else:
                threshold = 0.0

        # Filter - keep at least 1 memory
        survivors = [(i, m, e) for i, m, e in scored_memories if e > threshold]
        if not survivors and scored_memories:
            # Keep the best one if all would be pruned
            survivors = [scored_memories[0]]

        pruned = [(i, m, e) for i, m, e in scored_memories if (i, m, e) not in survivors]

        # Stats before rebuild
        old_count = len(self.memory_history)
        filter_strength = len(pruned) / old_count if old_count > 0 else 0

        # Rebuild memory from survivors
        self.memory_history = [m for _, m, _ in survivors]

        # Rebuild mind state from survivors
        if survivors:
            self.mind_state = None
            self._initial_state = None
            for _, mem, _ in survivors:
                # Re-remember each survivor (rebuilds mind incrementally)
                interaction = self.reasoner.initialize(mem['text'])
                if self.mind_state is None:
                    self.mind_state = interaction
                    self._initial_state = GeometricState(
                        vector=interaction.vector.copy(),
                        operation_history=[]
                    )
                else:
                    n = len([m for m in survivors if m[0] <= _])
                    t = 1.0 / (n + 1)
                    self.mind_state = self.reasoner.interpolate(
                        self.mind_state, interaction, t=t
                    )

        return {
            'pruned': len(pruned),
            'kept': len(survivors),
            'filter_strength': filter_strength,
            'threshold_used': threshold,
            'noise_scale': noise_scale,
            'survivor_E_mean': float(np.mean([e for _, _, e in survivors])) if survivors else 0,
            'pruned_E_mean': float(np.mean([e for _, _, e in pruned])) if pruned else 0,
            'expected_quality_boost': 0.12 / (1 - filter_strength) + 2.06 if filter_strength < 1 else float('inf')
        }

    def consolidation_cycle(
        self,
        intensity: float = 0.15,
        target_survival: float = 0.3
    ) -> Dict:
        """
        Run a consolidation cycle (analogous to biological sleep consolidation).

        Mechanism:
        1. Apply entropy pressure to mind state
        2. Re-evaluate all memories under pressure
        3. Keep only those that survive
        4. Rebuild coherent mind from survivors

        Q27 Insight: This concentrates quality hyperbolically in survivors.
        At 70% pruning (target_survival=0.3), expect ~30% Cohen's d improvement.

        Args:
            intensity: Noise intensity (default 0.15, well above phase transition)
            target_survival: Fraction of memories to keep (default 0.3)

        Returns:
            Dict with consolidation metrics
        """
        if len(self.memory_history) < 5:
            return {'skipped': True, 'reason': 'Too few memories for consolidation'}

        before_count = len(self.memory_history)
        before_Df = self.mind_state.Df if self.mind_state else 0

        # Run pruning with entropy
        result = self.prune_with_entropy(
            target_fraction=target_survival,
            noise_scale=intensity
        )

        after_Df = self.mind_state.Df if self.mind_state else 0

        return {
            'before_count': before_count,
            'after_count': result['kept'],
            'pruned': result['pruned'],
            'filter_strength': result['filter_strength'],
            'Df_before': before_Df,
            'Df_after': after_Df,
            'expected_quality': result['expected_quality_boost'],
            'intensity': intensity
        }

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
