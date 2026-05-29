"""
Feral Resident with Geometric Memory

Memory composition via pure geometry (Q45 validated).
Replaces HDC bind() with quantum entangle().

This module provides the GeometricMemory class that the VectorResident
uses for compositional memory accumulation.

Phase 1 E-Relationship Enhancement:
- Individual items stored BEFORE centroid interpolation
- Each item tagged with source_id, daemon_step, mind_hash_before
- Backward-compatible: centroid tracking still works
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import hashlib
import warnings
import numpy as np

if TYPE_CHECKING:
    from .resident_db import ResidentDB

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

# ============================================================================
# Q48-Q50 Semiotic Conservation Law Constants
# ============================================================================
# Universal conservation law: Df × α = 8e ≈ 21.746 (CV < 3% across 24 models)
SEMIOTIC_CONSTANT = 8 * np.e  # ≈ 21.746
CRITICAL_ALPHA = 0.5  # Riemann critical line, Chern number derivation (c₁ = 1)
OCTANT_COUNT = 8  # 2³ from Peirce's Reduction Thesis (3 irreducible categories)
MIN_MEMORIES_FOR_ALPHA = 20  # Minimum memories before alpha computation is stable


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

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        db: Optional['ResidentDB'] = None,
        source_id: Optional[str] = None
    ):
        self.reasoner = GeometricReasoner(model_name)
        self.mind_state: Optional[GeometricState] = None
        self.memory_history: List[Dict] = []
        self._initial_state: Optional[GeometricState] = None
        # Q27 Entropy Control: 0=permissive, >0.025=selective filtering
        self.temperature: float = 0.0
        # E-relationship daemon (Phase 1)
        self._db = db
        self._source_id = source_id
        self._daemon_step = 0

    def remember(self, interaction_text: str) -> Dict:
        """
        Add interaction to memory via geometric composition.

        Old approach: mind = hdc_bind(mind, embed(interaction))
        New approach: mind = interpolate(mind, new, t=1/N)

        Phase 1 Enhancement: Store individual item BEFORE interpolation
        for E-relationship graph construction.

        Returns receipt of the operation including item_id if db is connected.
        """
        # Capture mind state BEFORE this interaction (for graph edges)
        mind_hash_before = None
        if self.mind_state is not None:
            mind_hash_before = self.mind_state.receipt()['vector_hash']

        # Initialize interaction to manifold (BOUNDARY operation)
        interaction = self.reasoner.initialize(interaction_text)

        # Phase 1: Store individual item BEFORE centroid interpolation
        item_id = None
        if self._db is not None:
            item_id = self._db.store_vector(
                vector=interaction.vector,
                Df=interaction.Df,
                composition_op='daemon_item',
                parent_ids=[],
                source_id=self._source_id,
                daemon_step=self._daemon_step,
                mind_hash_before=mind_hash_before
            )
            self._daemon_step += 1

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
            'memory_index': len(self.memory_history),
            'item_id': item_id,  # Phase 1: Individual item ID for graph lookup
            'item_Df': interaction.Df  # Phase 1: Individual item Df (before interpolation)
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
        self._daemon_step = 0

    # ========================================================================
    # E-Relationship Daemon Methods (Phase 1)
    # ========================================================================

    def connect_db(self, db: 'ResidentDB', source_id: Optional[str] = None):
        """
        Connect to a database for individual item storage.

        Call this to enable E-relationship graph construction.
        Items stored via remember() will be persisted to the database.

        Args:
            db: ResidentDB instance
            source_id: Optional source identifier (session, paper, etc.)
        """
        self._db = db
        self._source_id = source_id
        self._daemon_step = 0

    def set_source_id(self, source_id: str):
        """Set the source ID for subsequent remember() calls."""
        self._source_id = source_id
        self._daemon_step = 0  # Reset step for new source

    def get_stored_item_ids(self) -> List[str]:
        """
        Get list of item IDs stored during this session.

        Returns:
            List of vector IDs for items stored via remember()
        """
        return [
            m['item_id']
            for m in self.memory_history
            if m.get('item_id') is not None
        ]

    @property
    def db_connected(self) -> bool:
        """Check if database is connected for individual item storage."""
        return self._db is not None

    @property
    def source_id(self) -> Optional[str]:
        """Get current source ID."""
        return self._source_id

    @property
    def daemon_step(self) -> int:
        """Get current daemon step (number of items stored)."""
        return self._daemon_step

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

    # ========================================================================
    # Q48-Q50 Semiotic Health Methods
    # ========================================================================

    def compute_alpha(self) -> float:
        """
        Get eigenspectrum decay exponent α (Q48-Q50).

        The α ≈ 0.5 value is a fundamental property of well-trained embedding
        models (sentence-transformers), derived from Chern number c₁ = 1.

        This is NOT computed from user samples (which would require thousands of
        samples to estimate accurately), but is the known theoretical value from
        Q48-Q50 research validated across 24 models (CV = 6.93%).

        The semiotic health check uses this theoretical α to verify that the
        mind's Df follows the conservation law: Df × α ≈ 8e.

        Returns:
            α = 0.5 (critical value, topologically protected)
        """
        # Q48-Q50 validated: α ≈ 0.5 across all sentence-transformer models
        # This is a property of the embedding model, not user data
        # Deviations in Df × α from 8e indicate compression/distortion
        return CRITICAL_ALPHA

    def estimate_sample_alpha(self, n_eigenvalues: int = 50) -> Optional[float]:
        """
        Estimate eigenspectrum decay from sample covariance (for diagnostics).

        WARNING: This is NOT the α from Q48-Q50. That α is computed from the
        full embedding weight matrix, requiring millions of parameters.
        This sample-based estimate is useful for diagnostics but will not
        match the theoretical value of 0.5.

        Returns:
            Estimated α from sample covariance, or None if insufficient data
        """
        if len(self.memory_history) < MIN_MEMORIES_FOR_ALPHA:
            return None

        # Get all memory vectors
        vectors = []
        for mem in self.memory_history:
            state = self.reasoner.initialize(mem['text'])
            vectors.append(state.vector)

        vectors = np.array(vectors)

        # Compute Gram matrix (N×N) instead of covariance (D×D)
        # This is more appropriate for N << D case
        centered = vectors - np.mean(vectors, axis=0)
        gram = np.dot(centered, centered.T)  # N×N Gram matrix

        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        # Take top n_eigenvalues (or fewer if not enough)
        n = min(n_eigenvalues, len(eigenvalues))
        eigenvalues = eigenvalues[:n]

        # Filter out zero/negative eigenvalues for log
        mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[mask]

        if len(eigenvalues) < 5:
            return None

        # Fit log(λ_k) vs log(k)
        k = np.arange(1, len(eigenvalues) + 1)
        log_k = np.log(k)
        log_lambda = np.log(eigenvalues)

        # Linear regression: log(λ) = -α × log(k) + const
        slope, _ = np.polyfit(log_k, log_lambda, 1)
        alpha = -slope

        return float(alpha)

    def get_semiotic_health(self) -> Dict:
        """
        Compute semiotic health metrics (Q48-Q50).

        The conservation law Df × α = 8e ≈ 21.746 defines healthy semantic geometry.
        With α = 0.5 (Chern number derivation), healthy Df ≈ 43.5.

        The mind_state.Df is the participation ratio of the mind vector itself,
        which measures how "spread out" the representation is across dimensions.

        Health interpretation:
        - Low Df (<30): Mind collapsed into few dimensions → compressed
        - Healthy Df (30-60): Full semantic utilization
        - High Df (>80): Possible noise/diffusion

        Returns:
            Dict with health metrics
        """
        if self.mind_state is None:
            return {
                'Df_alpha': 0.0,
                'health_ratio': 0.0,
                'alpha': None,
                'Df': 0.0,
                'interpretation': 'no_mind_state'
            }

        Df = self.mind_state.Df
        alpha = self.compute_alpha()  # Returns 0.5 (theoretical value)

        Df_alpha = Df * alpha
        health_ratio = Df_alpha / SEMIOTIC_CONSTANT

        # The "target" Df for α=0.5 is 8e/α = 43.49
        target_Df = SEMIOTIC_CONSTANT / alpha

        # Interpret health based on how close Df is to target
        # Use log-ratio to handle both compression and expansion symmetrically
        Df_ratio = Df / target_Df if target_Df > 0 else 0

        if Df_ratio < 0.5:  # Less than half target
            interpretation = 'compressed'
        elif Df_ratio > 2.0:  # More than double target
            interpretation = 'expanded'  # Possible noise/diffusion
        elif 0.7 < Df_ratio < 1.3:  # Within 30% of target
            interpretation = 'healthy'
        else:
            interpretation = 'moderate'  # Somewhat off but not critical

        # Alignment compression: how much Df is below target
        compression = max(0.0, 1.0 - Df_ratio) if Df_ratio < 1.0 else 0.0

        return {
            'Df_alpha': float(Df_alpha),
            'health_ratio': float(health_ratio),
            'alpha': float(alpha),
            'Df': float(Df),
            'target_Df': float(target_Df),
            'target_8e': float(SEMIOTIC_CONSTANT),
            'Df_ratio': float(Df_ratio),
            'interpretation': interpretation,
            'alignment_compression': float(compression)
        }

    def get_octant_distribution(self) -> Dict:
        """
        Analyze distribution across 8 semiotic octants (Q48-Q50).

        Uses top 3 PCs of mind state history. Each octant = unique sign
        combination of (PC1, PC2, PC3), giving 2³ = 8 octants.

        Healthy cognition populates all 8 octants (diverse semantic coverage).
        Alignment compression may collapse octants.

        Returns:
            Dict with:
            - counts: 8-element array of population per octant
            - coverage: fraction of octants populated (should be 8/8)
            - entropy: distribution entropy (max = log(8) ≈ 2.08)
            - dominant_octant: index of most populated octant
            - octant_labels: sign patterns for each octant
        """
        if len(self.memory_history) < 10:
            return {
                'counts': np.zeros(OCTANT_COUNT),
                'coverage': 0.0,
                'entropy': 0.0,
                'dominant_octant': None,
                'message': 'Need 10+ memories for octant analysis'
            }

        # Get all memory vectors
        vectors = []
        for mem in self.memory_history:
            state = self.reasoner.initialize(mem['text'])
            vectors.append(state.vector)

        vectors = np.array(vectors)

        # PCA to get top 3 components
        centered = vectors - np.mean(vectors, axis=0)
        cov = np.dot(centered.T, centered) / len(vectors)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        top3_vectors = eigenvectors[:, idx[:3]]

        # Project all vectors onto top 3 PCs
        projections = np.dot(centered, top3_vectors)  # Shape: (N, 3)

        # Assign each point to an octant based on sign pattern
        # Octant index = 4*(PC1>0) + 2*(PC2>0) + 1*(PC3>0)
        octant_indices = (
            4 * (projections[:, 0] > 0).astype(int) +
            2 * (projections[:, 1] > 0).astype(int) +
            1 * (projections[:, 2] > 0).astype(int)
        )

        # Count per octant
        counts = np.bincount(octant_indices, minlength=OCTANT_COUNT)

        # Coverage: fraction of octants with at least 1 member
        coverage = np.sum(counts > 0) / OCTANT_COUNT

        # Entropy of distribution
        p = counts / counts.sum()
        p = p[p > 0]  # Filter zeros for log
        entropy = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

        # Maximum entropy for 8 octants
        max_entropy = np.log(OCTANT_COUNT)

        # Octant labels (sign patterns)
        octant_labels = [
            '---', '--+', '-+-', '-++',
            '+--', '+-+', '++-', '+++'
        ]

        return {
            'counts': counts.tolist(),
            'coverage': float(coverage),
            'entropy': float(entropy),
            'normalized_entropy': float(entropy / max_entropy) if max_entropy > 0 else 0.0,
            'max_entropy': float(max_entropy),
            'dominant_octant': int(np.argmax(counts)),
            'dominant_label': octant_labels[np.argmax(counts)],
            'octant_labels': octant_labels,
            'total_memories': len(self.memory_history)
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
