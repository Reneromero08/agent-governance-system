"""
Convergence Observer for Multi-Resident Swarm (P.1.2)

Track if residents develop shared patterns:
- E(mind_A, mind_B) - quantum overlap between mind states
- Df correlation - participation ratio evolution similarity
- Shared notations - patterns adopted by multiple residents
- Convergence events - moments of high inter-resident resonance

Q44/Q45 validated: All metrics use Born rule (E) and Df (participation ratio)
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
import sys

# Add parent for imports
FERAL_PATH = Path(__file__).parent.parent  # collective/ -> FERAL_RESIDENT/
sys.path.insert(0, str(FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricState
from .shared_space import SharedSemanticSpace
from emergence.symbol_evolution import NotationRegistry

if TYPE_CHECKING:
    from cognition.vector_brain import VectorResident


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConvergenceMetrics:
    """Metrics tracking convergence between two residents."""
    pair_id: str                       # "resident_a:resident_b"
    timestamp: str
    E_mind_correlation: float          # E(mind_A, mind_B)
    Df_correlation: float              # Correlation of Df trajectories
    shared_notation_count: int         # Notations used by both
    Df_a: float                        # Current Df for resident A
    Df_b: float                        # Current Df for resident B
    distance_a: float                  # Mind distance from start for A
    distance_b: float                  # Mind distance from start for B


@dataclass
class SwarmConvergenceSummary:
    """Summary of convergence across entire swarm."""
    timestamp: str
    resident_count: int
    pair_count: int
    E_minds_mean: float
    E_minds_max: float
    E_minds_min: float
    Df_correlation_mean: float
    total_convergence_events: int
    total_shared_notations: int
    pairs: Dict[str, ConvergenceMetrics] = field(default_factory=dict)


@dataclass
class ConvergenceObservation:
    """Single observation record for audit."""
    observation_id: str
    timestamp: str
    pair_id: str
    E_minds: float
    Df_corr: float
    shared_notations: int
    is_convergence_event: bool
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Convergence Observer
# =============================================================================

class ConvergenceObserver:
    """
    P.1.2: Track if residents develop shared patterns.

    Observes:
    - E(mind_A, mind_B) - mind resonance between residents
    - Df correlation - participation ratio evolution similarity
    - Shared notations - patterns adopted by multiple residents
    - Convergence events - moments of high inter-resident resonance

    Usage:
        space = SharedSemanticSpace()
        observer = ConvergenceObserver(space)

        # Observe after each interaction
        metrics = observer.observe_pair(resident_a, resident_b)

        # Or observe entire swarm
        summary = observer.observe_swarm({"alpha": resident_a, "beta": resident_b})
    """

    # E threshold for "convergence event"
    HIGH_RESONANCE_THRESHOLD = 0.5

    # Df correlation threshold for "aligned evolution"
    DF_CORRELATION_THRESHOLD = 0.7

    def __init__(self, shared_space: SharedSemanticSpace):
        """
        Initialize convergence observer.

        Args:
            shared_space: SharedSemanticSpace for recording events
        """
        self.shared_space = shared_space
        self.observations: List[ConvergenceObservation] = []

        # Cache for Df histories (updated on observe)
        self._Df_cache: Dict[str, List[float]] = {}

    # =========================================================================
    # Core Metrics
    # =========================================================================

    def compute_E_between_minds(
        self,
        mind_A: GeometricState,
        mind_B: GeometricState
    ) -> float:
        """
        Compute E(mind_A, mind_B) - quantum overlap between mind states.

        This is the core cross-resident metric:
        - High E means residents are "thinking similarly"
        - Low E means divergent development

        Q44 validated: E correlates r=0.977 with semantic similarity.
        """
        return mind_A.E_with(mind_B)

    def compute_Df_correlation(
        self,
        Df_history_A: List[float],
        Df_history_B: List[float]
    ) -> float:
        """
        Compute Pearson correlation between Df trajectories.

        Aligned residents will have correlated Df evolution
        (both spreading or both concentrating together).

        Returns:
            Correlation coefficient [-1, 1] or 0 if insufficient data
        """
        if len(Df_history_A) < 3 or len(Df_history_B) < 3:
            return 0.0

        # Align lengths (use most recent data)
        min_len = min(len(Df_history_A), len(Df_history_B))
        A = np.array(Df_history_A[-min_len:])
        B = np.array(Df_history_B[-min_len:])

        # Handle constant arrays
        if np.std(A) == 0 or np.std(B) == 0:
            return 0.0

        return float(np.corrcoef(A, B)[0, 1])

    def detect_shared_notations(
        self,
        resident_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Find notation patterns shared by multiple residents.

        Integration with symbol_evolution.py's NotationRegistry.

        Args:
            resident_ids: List of resident IDs to analyze

        Returns:
            Dict mapping pattern -> list of residents using it
        """
        all_patterns: Dict[str, List[str]] = {}

        for resident_id in resident_ids:
            try:
                # Load registry for this resident
                registry = NotationRegistry(resident_id)

                for pattern in registry.registry.keys():
                    if pattern not in all_patterns:
                        all_patterns[pattern] = []
                    all_patterns[pattern].append(resident_id)
            except Exception:
                continue

        # Return only patterns used by 2+ residents
        return {
            pattern: users
            for pattern, users in all_patterns.items()
            if len(users) >= 2
        }

    # =========================================================================
    # Observation Methods
    # =========================================================================

    def observe_pair(
        self,
        resident_a: 'VectorResident',
        resident_b: 'VectorResident'
    ) -> Optional[ConvergenceMetrics]:
        """
        Observe convergence between a pair of residents.

        Args:
            resident_a: First VectorResident
            resident_b: Second VectorResident

        Returns:
            ConvergenceMetrics or None if observation failed
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get resident IDs
        id_a = resident_a.thread_id
        id_b = resident_b.thread_id

        # Normalize pair ordering
        if id_a > id_b:
            id_a, id_b = id_b, id_a
            resident_a, resident_b = resident_b, resident_a

        pair_id = f"{id_a}:{id_b}"

        # Get mind states
        mind_a = resident_a.store.get_mind_state()
        mind_b = resident_b.store.get_mind_state()

        if mind_a is None or mind_b is None:
            return None

        # Compute E between minds
        E_minds = self.compute_E_between_minds(mind_a, mind_b)

        # Get evolution data
        evolution_a = resident_a.mind_evolution
        evolution_b = resident_b.mind_evolution

        Df_history_a = evolution_a.get('Df_history', [])
        Df_history_b = evolution_b.get('Df_history', [])

        # Update Df cache
        self._Df_cache[id_a] = Df_history_a
        self._Df_cache[id_b] = Df_history_b

        # Compute Df correlation
        Df_corr = self.compute_Df_correlation(Df_history_a, Df_history_b)

        # Detect shared notations
        shared_notations = self.detect_shared_notations([id_a, id_b])

        # Check for convergence event
        is_convergence_event = E_minds > self.HIGH_RESONANCE_THRESHOLD

        if is_convergence_event:
            self.shared_space.record_convergence_event(
                resident_a=id_a,
                resident_b=id_b,
                E_value=E_minds,
                Df_a=mind_a.Df,
                Df_b=mind_b.Df,
                event_type=SharedSemanticSpace.EVENT_HIGH_RESONANCE,
                metadata={
                    'timestamp': timestamp,
                    'Df_correlation': Df_corr,
                    'shared_notations': len(shared_notations)
                }
            )

        # Record observation
        observation = ConvergenceObservation(
            observation_id=hashlib.sha256(
                f"{pair_id}{timestamp}".encode()
            ).hexdigest()[:16],
            timestamp=timestamp,
            pair_id=pair_id,
            E_minds=E_minds,
            Df_corr=Df_corr,
            shared_notations=len(shared_notations),
            is_convergence_event=is_convergence_event,
            metadata={
                'Df_a': mind_a.Df,
                'Df_b': mind_b.Df,
                'distance_a': evolution_a.get('distance_from_start', 0),
                'distance_b': evolution_b.get('distance_from_start', 0)
            }
        )
        self.observations.append(observation)

        # Build metrics
        return ConvergenceMetrics(
            pair_id=pair_id,
            timestamp=timestamp,
            E_mind_correlation=E_minds,
            Df_correlation=Df_corr,
            shared_notation_count=len(shared_notations),
            Df_a=mind_a.Df,
            Df_b=mind_b.Df,
            distance_a=evolution_a.get('distance_from_start', 0),
            distance_b=evolution_b.get('distance_from_start', 0)
        )

    def observe_swarm(
        self,
        residents: Dict[str, 'VectorResident']
    ) -> SwarmConvergenceSummary:
        """
        Observe convergence across all resident pairs.

        Args:
            residents: Dict mapping resident_id -> VectorResident

        Returns:
            SwarmConvergenceSummary with all pairwise metrics
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        resident_ids = list(residents.keys())
        pairs: Dict[str, ConvergenceMetrics] = {}

        E_values = []
        Df_correlations = []
        total_shared_notations = set()

        # Pairwise observation
        for i, id_a in enumerate(resident_ids):
            for id_b in resident_ids[i+1:]:
                metrics = self.observe_pair(residents[id_a], residents[id_b])

                if metrics:
                    pairs[metrics.pair_id] = metrics
                    E_values.append(metrics.E_mind_correlation)
                    Df_correlations.append(metrics.Df_correlation)

        # Get all shared notations across swarm
        all_shared = self.detect_shared_notations(resident_ids)
        for pattern in all_shared.keys():
            total_shared_notations.add(pattern)

        # Count convergence events from shared space
        convergence_events = len(self.shared_space.get_convergence_events(
            event_type=SharedSemanticSpace.EVENT_HIGH_RESONANCE,
            limit=1000
        ))

        # Build summary
        return SwarmConvergenceSummary(
            timestamp=timestamp,
            resident_count=len(residents),
            pair_count=len(pairs),
            E_minds_mean=float(np.mean(E_values)) if E_values else 0.0,
            E_minds_max=float(np.max(E_values)) if E_values else 0.0,
            E_minds_min=float(np.min(E_values)) if E_values else 0.0,
            Df_correlation_mean=float(np.mean(Df_correlations)) if Df_correlations else 0.0,
            total_convergence_events=convergence_events,
            total_shared_notations=len(total_shared_notations),
            pairs=pairs
        )

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_convergence_trend(
        self,
        pair_id: str,
        window: int = 10
    ) -> Dict:
        """
        Get convergence trend for a specific pair.

        Args:
            pair_id: Pair identifier (e.g., "alpha:beta")
            window: Number of recent observations to analyze

        Returns:
            Trend analysis dict
        """
        pair_obs = [o for o in self.observations if o.pair_id == pair_id]

        if len(pair_obs) < 2:
            return {'trend': 'insufficient_data', 'observations': len(pair_obs)}

        recent = pair_obs[-window:]
        E_values = [o.E_minds for o in recent]

        # Linear regression for trend
        x = np.arange(len(E_values))
        slope, _ = np.polyfit(x, E_values, 1)

        if slope > 0.01:
            trend = 'converging'
        elif slope < -0.01:
            trend = 'diverging'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'slope': float(slope),
            'current_E': E_values[-1],
            'observations': len(recent),
            'convergence_events': sum(1 for o in recent if o.is_convergence_event),
            'E_history': E_values
        }

    def get_observation_summary(self) -> Dict:
        """Get summary of all observations."""
        if not self.observations:
            return {'status': 'no_observations'}

        E_values = [o.E_minds for o in self.observations]
        Df_corrs = [o.Df_corr for o in self.observations]

        # Group by pair
        pair_counts = {}
        for o in self.observations:
            pair_counts[o.pair_id] = pair_counts.get(o.pair_id, 0) + 1

        return {
            'total_observations': len(self.observations),
            'unique_pairs': len(pair_counts),
            'E_minds_mean': float(np.mean(E_values)),
            'E_minds_std': float(np.std(E_values)),
            'E_minds_max': float(np.max(E_values)),
            'Df_correlation_mean': float(np.mean(Df_corrs)),
            'convergence_events': sum(1 for o in self.observations if o.is_convergence_event),
            'pair_observations': pair_counts,
            'recent_observations': [asdict(o) for o in self.observations[-10:]]
        }

    def find_most_convergent_pair(self) -> Optional[str]:
        """Find the pair with highest average E."""
        if not self.observations:
            return None

        # Group by pair
        pair_E: Dict[str, List[float]] = {}
        for o in self.observations:
            if o.pair_id not in pair_E:
                pair_E[o.pair_id] = []
            pair_E[o.pair_id].append(o.E_minds)

        # Find highest mean
        best_pair = None
        best_mean = -1

        for pair_id, E_list in pair_E.items():
            mean = np.mean(E_list)
            if mean > best_mean:
                best_mean = mean
                best_pair = pair_id

        return best_pair

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_observations(self, filepath: Optional[Path] = None):
        """Save observations to JSON file."""
        if filepath is None:
            filepath = Path(__file__).parent / "data" / "convergence_observations.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'observations': [asdict(o) for o in self.observations],
            'Df_cache': {k: list(v) for k, v in self._Df_cache.items()}
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_observations(self, filepath: Optional[Path] = None):
        """Load observations from JSON file."""
        if filepath is None:
            filepath = Path(__file__).parent / "data" / "convergence_observations.json"

        if not filepath.exists():
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.observations = [
                ConvergenceObservation(**o)
                for o in data.get('observations', [])
            ]
            self._Df_cache = {
                k: list(v) for k, v in data.get('Df_cache', {}).items()
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # =========================================================================
    # Visualization
    # =========================================================================

    def print_convergence_dashboard(self):
        """Print ASCII dashboard of convergence metrics."""
        summary = self.get_observation_summary()

        print(f"\n{'='*70}")
        print(f"  CONVERGENCE OBSERVER DASHBOARD (P.1.2)")
        print(f"{'='*70}")

        if summary.get('status') == 'no_observations':
            print(f"\n  No observations recorded yet.")
            print(f"  Run: observer.observe_swarm(residents)")
            print(f"\n{'='*70}\n")
            return

        print(f"\n  Total Observations: {summary['total_observations']}")
        print(f"  Unique Pairs: {summary['unique_pairs']}")
        print(f"  Convergence Events: {summary['convergence_events']}")

        print(f"\n{'-'*70}")
        print(f"  E(mind_A, mind_B) METRICS")
        print(f"{'-'*70}")
        print(f"  Mean:  {summary['E_minds_mean']:.4f}")
        print(f"  Std:   {summary['E_minds_std']:.4f}")
        print(f"  Max:   {summary['E_minds_max']:.4f}")

        print(f"\n{'-'*70}")
        print(f"  Df CORRELATION")
        print(f"{'-'*70}")
        print(f"  Mean:  {summary['Df_correlation_mean']:.4f}")

        print(f"\n{'-'*70}")
        print(f"  PAIR OBSERVATIONS")
        print(f"{'-'*70}")
        for pair_id, count in summary.get('pair_observations', {}).items():
            trend = self.get_convergence_trend(pair_id)
            print(f"  {pair_id}: {count} observations ({trend['trend']})")

        most_convergent = self.find_most_convergent_pair()
        if most_convergent:
            print(f"\n  Most Convergent Pair: {most_convergent}")

        print(f"\n{'='*70}\n")


# =============================================================================
# Testing
# =============================================================================

def _test_convergence_observer():
    """Basic test of ConvergenceObserver."""
    import tempfile
    import os

    # Use temp file for test
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = f.name

    try:
        # Create space and observer
        space = SharedSemanticSpace(test_db)
        observer = ConvergenceObserver(space)

        # Create mock mind states
        vec_a = np.random.randn(384).astype(np.float32)
        vec_b = np.random.randn(384).astype(np.float32)
        vec_similar = vec_a + 0.1 * np.random.randn(384).astype(np.float32)

        state_a = GeometricState(vector=vec_a, operation_history=[])
        state_b = GeometricState(vector=vec_b, operation_history=[])
        state_similar = GeometricState(vector=vec_similar, operation_history=[])

        # Test E computation
        E_ab = observer.compute_E_between_minds(state_a, state_b)
        E_similar = observer.compute_E_between_minds(state_a, state_similar)

        print(f"E(random_a, random_b): {E_ab:.4f}")
        print(f"E(a, similar_to_a): {E_similar:.4f}")
        assert E_similar > E_ab, "Similar state should have higher E"

        # Test Df correlation
        Df_history_a = [10.0, 12.0, 14.0, 16.0, 18.0]
        Df_history_b = [11.0, 13.0, 15.0, 17.0, 19.0]  # Correlated
        Df_history_c = [20.0, 18.0, 16.0, 14.0, 12.0]  # Anti-correlated

        corr_ab = observer.compute_Df_correlation(Df_history_a, Df_history_b)
        corr_ac = observer.compute_Df_correlation(Df_history_a, Df_history_c)

        print(f"Df correlation (correlated): {corr_ab:.4f}")
        print(f"Df correlation (anti-correlated): {corr_ac:.4f}")
        assert corr_ab > 0.9, "Correlated histories should have high correlation"
        assert corr_ac < -0.9, "Anti-correlated histories should have negative correlation"

        # Test observation summary
        summary = observer.get_observation_summary()
        print(f"Summary: {summary}")

        space.close()
        print("ConvergenceObserver test passed!")

    finally:
        os.unlink(test_db)


if __name__ == "__main__":
    _test_convergence_observer()
