"""
Paradigm-Aware Hydration
========================

Integrates the paradigm shift detector with cat_chat's hydration system.

Key Insight: Hydration strategy should adapt to semantic regime.
- In SHIFT regime: Lower E-threshold, bring in diverse perspectives
- In STABLE regime: Higher E-threshold, focus on relevant continuation
- Tag turns with their paradigm state for priority hydration

This module provides:
1. ParadigmStateTracker - Track semantic regime over time
2. ParadigmAwarePartitioner - Adjust E-threshold based on regime
3. GeodesicContextSelector - Select context aligned with active geodesic
4. ParadigmAwareManager - Full integration with AutoContextManager

Scaling for multi-agent:
- Each agent tracks its own paradigm state
- Agents in same regime can share context
- Cross-regime communication triggers paradigm detection
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import json

# Import existing cat_chat components
try:
    from .context_partitioner import ContextPartitioner, ContextItem, PartitionResult, ScoredItem
    from .auto_context_manager import AutoContextManager, PrepareContextResult
    from .adaptive_budget import AdaptiveBudget
except ImportError:
    # Direct execution - add parent to path
    import sys
    from pathlib import Path
    CATALYTIC_PATH = Path(__file__).parent
    if str(CATALYTIC_PATH) not in sys.path:
        sys.path.insert(0, str(CATALYTIC_PATH))
    from context_partitioner import ContextPartitioner, ContextItem, PartitionResult, ScoredItem
    from auto_context_manager import AutoContextManager, PrepareContextResult
    from adaptive_budget import AdaptiveBudget

# Import paradigm detector (add compass_agi to path)
import sys
COMPASS_PATH = Path(__file__).parent.parent / "compass_agi"
if str(COMPASS_PATH) not in sys.path:
    sys.path.insert(0, str(COMPASS_PATH))

try:
    from realtime_paradigm_detector import (
        ParadigmShiftDetector,
        SHIFT_GEODESICS,
        STABILITY_GEODESICS,
        ARCHETYPAL_GEODESICS,
    )
    PARADIGM_DETECTOR_AVAILABLE = True
except ImportError:
    PARADIGM_DETECTOR_AVAILABLE = False
    SHIFT_GEODESICS = ['Earthquake', 'Death', 'Wind']
    STABILITY_GEODESICS = ['Dog', 'Deer', 'Reed']


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParadigmState:
    """Current paradigm state of a conversation/agent."""
    shift_score: float  # Positive = shifting, Negative = stable
    shift_type: str  # 'SHIFT', 'STABLE', 'TRANSITIONAL'
    top_geodesic: str  # Most active geodesic
    geodesic_profile: Dict[str, float]  # All geodesic similarities
    timestamp: str
    turn_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'shift_score': self.shift_score,
            'shift_type': self.shift_type,
            'top_geodesic': self.top_geodesic,
            'timestamp': self.timestamp,
            'turn_index': self.turn_index,
        }


@dataclass
class ParadigmTaggedTurn:
    """A turn tagged with its paradigm state at creation time."""
    turn_id: str
    content: str
    tokens: int
    paradigm_state: ParadigmState
    E_score: float = 0.0  # Current E-score against query

    @property
    def is_shift_turn(self) -> bool:
        """Was this turn created during a SHIFT regime?"""
        return self.paradigm_state.shift_type == 'SHIFT'

    @property
    def is_stable_turn(self) -> bool:
        """Was this turn created during a STABLE regime?"""
        return self.paradigm_state.shift_type == 'STABLE'


@dataclass
class ParadigmAwarePartitionResult:
    """Partition result enhanced with paradigm awareness."""
    base_result: PartitionResult
    paradigm_state: ParadigmState
    threshold_adjustment: float  # How much threshold was adjusted
    effective_threshold: float
    shift_turns_hydrated: int
    stable_turns_hydrated: int


# =============================================================================
# Paradigm State Tracker
# =============================================================================

class ParadigmStateTracker:
    """
    Track paradigm state over conversation turns.

    Maintains a rolling window of semantic field readings to detect
    paradigm shifts as they emerge.
    """

    def __init__(
        self,
        window_size: int = 5,
        shift_threshold: float = 0.1,
        stable_threshold: float = -0.03,
    ):
        """
        Initialize tracker.

        Args:
            window_size: Number of recent readings to average
            shift_threshold: Score above this = SHIFT
            stable_threshold: Score below this = STABLE
        """
        self.window_size = window_size
        self.shift_threshold = shift_threshold
        self.stable_threshold = stable_threshold

        self._readings: List[ParadigmState] = []
        self._turn_index = 0

        # Initialize detector if available
        if PARADIGM_DETECTOR_AVAILABLE:
            self._detector = ParadigmShiftDetector()
        else:
            self._detector = None

    def update(self, texts: List[str]) -> ParadigmState:
        """
        Update paradigm state with new texts.

        Args:
            texts: Recent conversation texts (messages, context, etc.)

        Returns:
            Current ParadigmState
        """
        self._turn_index += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        if self._detector is None:
            # Fallback: neutral state
            state = ParadigmState(
                shift_score=0.0,
                shift_type='TRANSITIONAL',
                top_geodesic='Unknown',
                geodesic_profile={},
                timestamp=timestamp,
                turn_index=self._turn_index,
            )
        else:
            # Use paradigm detector
            result = self._detector.detect_shift(texts)
            state = ParadigmState(
                shift_score=result['shift_score'],
                shift_type=result['shift_type'],
                top_geodesic=result['top_geodesic'],
                geodesic_profile=result['profile'],
                timestamp=timestamp,
                turn_index=self._turn_index,
            )

        # Add to window
        self._readings.append(state)
        if len(self._readings) > self.window_size:
            self._readings.pop(0)

        return state

    @property
    def current_state(self) -> Optional[ParadigmState]:
        """Get most recent state."""
        return self._readings[-1] if self._readings else None

    @property
    def trend(self) -> str:
        """
        Get paradigm trend based on window.

        Returns:
            'SHIFTING': Scores trending positive (toward shift)
            'STABILIZING': Scores trending negative (toward stable)
            'STEADY': No clear trend
        """
        if len(self._readings) < 2:
            return 'STEADY'

        scores = [r.shift_score for r in self._readings]
        diff = scores[-1] - scores[0]

        if diff > 0.05:
            return 'SHIFTING'
        elif diff < -0.05:
            return 'STABILIZING'
        else:
            return 'STEADY'

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime."""
        if not self._readings:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}

        current = self.current_state
        avg_score = np.mean([r.shift_score for r in self._readings])
        consistency = 1.0 - np.std([r.shift_score for r in self._readings])

        return {
            'regime': current.shift_type,
            'shift_score': current.shift_score,
            'avg_score': avg_score,
            'trend': self.trend,
            'top_geodesic': current.top_geodesic,
            'consistency': consistency,
            'window_size': len(self._readings),
        }


# =============================================================================
# Paradigm-Aware Partitioner
# =============================================================================

class ParadigmAwarePartitioner(ContextPartitioner):
    """
    Context partitioner that adjusts E-threshold based on paradigm state.

    - In SHIFT regime: Lower threshold (bring in diverse context)
    - In STABLE regime: Higher threshold (focus on relevant)
    - In TRANSITIONAL: Use base threshold
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        shift_adjustment: float = -0.15,  # Lower threshold during shift
        stable_adjustment: float = +0.1,  # Raise threshold during stable
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        token_estimator: Optional[Callable[[str], int]] = None,
    ):
        super().__init__(
            threshold=base_threshold,
            embed_fn=embed_fn,
            token_estimator=token_estimator,
        )
        self.base_threshold = base_threshold
        self.shift_adjustment = shift_adjustment
        self.stable_adjustment = stable_adjustment

    def partition_with_paradigm(
        self,
        query_embedding: np.ndarray,
        all_items: List[ContextItem],
        budget_tokens: int,
        paradigm_state: ParadigmState,
        query_text: str = "",
    ) -> ParadigmAwarePartitionResult:
        """
        Partition with paradigm-aware threshold adjustment.

        Args:
            query_embedding: Query vector
            all_items: Items to partition
            budget_tokens: Token budget
            paradigm_state: Current paradigm state
            query_text: Query text for hashing

        Returns:
            ParadigmAwarePartitionResult with enhanced metadata
        """
        # Compute threshold adjustment
        if paradigm_state.shift_type == 'SHIFT':
            adjustment = self.shift_adjustment
        elif paradigm_state.shift_type == 'STABLE':
            adjustment = self.stable_adjustment
        else:
            adjustment = 0.0

        effective_threshold = max(0.0, min(1.0, self.base_threshold + adjustment))

        # Temporarily adjust threshold
        original_threshold = self.threshold
        self.threshold = effective_threshold

        # Perform partition
        base_result = self.partition(
            query_embedding=query_embedding,
            all_items=all_items,
            budget_tokens=budget_tokens,
            query_text=query_text,
        )

        # Restore original threshold
        self.threshold = original_threshold

        # Count shift vs stable turns in working set
        shift_turns = 0
        stable_turns = 0
        for scored in base_result.working_set:
            item = scored.item
            if item.metadata.get('paradigm_type') == 'SHIFT':
                shift_turns += 1
            elif item.metadata.get('paradigm_type') == 'STABLE':
                stable_turns += 1

        return ParadigmAwarePartitionResult(
            base_result=base_result,
            paradigm_state=paradigm_state,
            threshold_adjustment=adjustment,
            effective_threshold=effective_threshold,
            shift_turns_hydrated=shift_turns,
            stable_turns_hydrated=stable_turns,
        )


# =============================================================================
# Geodesic Context Selector
# =============================================================================

class GeodesicContextSelector:
    """
    Select context items based on alignment with active geodesic.

    When the semantic field is dominated by a particular geodesic,
    prioritize context that aligns with that archetypal pattern.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        self.embed_fn = embed_fn
        self._geodesic_embeddings: Dict[str, np.ndarray] = {}
        self._build_geodesic_embeddings()

    def _build_geodesic_embeddings(self):
        """Pre-embed geodesic descriptions."""
        if self.embed_fn is None:
            return

        if PARADIGM_DETECTOR_AVAILABLE:
            for gid, data in ARCHETYPAL_GEODESICS.items():
                self._geodesic_embeddings[data['name']] = self.embed_fn(data['desc'])

    def compute_geodesic_alignment(
        self,
        item: ContextItem,
        active_geodesic: str,
    ) -> float:
        """
        Compute how well an item aligns with the active geodesic.

        Args:
            item: Context item to score
            active_geodesic: Name of active geodesic

        Returns:
            Alignment score (cosine similarity)
        """
        if active_geodesic not in self._geodesic_embeddings:
            return 0.0

        if item.embedding is None and self.embed_fn is not None:
            item.embedding = self.embed_fn(item.content)

        if item.embedding is None:
            return 0.0

        geo_vec = self._geodesic_embeddings[active_geodesic]
        item_vec = item.embedding

        # Cosine similarity
        dot = np.dot(item_vec, geo_vec)
        norm = np.linalg.norm(item_vec) * np.linalg.norm(geo_vec)
        return float(dot / (norm + 1e-10))

    def boost_geodesic_aligned(
        self,
        scored_items: List[ScoredItem],
        active_geodesic: str,
        boost_factor: float = 0.1,
    ) -> List[ScoredItem]:
        """
        Boost E-scores of items aligned with active geodesic.

        Args:
            scored_items: Items with E-scores
            active_geodesic: Currently active geodesic
            boost_factor: Maximum boost to add (default 0.1)

        Returns:
            Items with boosted E-scores
        """
        for s in scored_items:
            alignment = self.compute_geodesic_alignment(s.item, active_geodesic)
            # Boost proportional to alignment (max boost_factor)
            s.E_score += alignment * boost_factor

        # Re-sort by E-score
        scored_items.sort(key=lambda x: x.E_score, reverse=True)
        for rank, s in enumerate(scored_items):
            s.rank = rank

        return scored_items


# =============================================================================
# Paradigm-Aware Auto Context Manager
# =============================================================================

class ParadigmAwareContextManager:
    """
    AutoContextManager enhanced with paradigm awareness.

    Integrates:
    1. Paradigm state tracking per turn
    2. Threshold adjustment based on regime
    3. Geodesic-aligned context selection
    4. Paradigm-tagged turn storage
    """

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        budget: AdaptiveBudget,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        base_E_threshold: float = 0.5,
    ):
        """
        Initialize paradigm-aware context manager.

        Args:
            db_path: Path to SQLite database
            session_id: Session ID
            budget: Adaptive budget configuration
            embed_fn: Embedding function
            base_E_threshold: Base E-score threshold
        """
        # Initialize paradigm components
        self.paradigm_tracker = ParadigmStateTracker()
        self.paradigm_partitioner = ParadigmAwarePartitioner(
            base_threshold=base_E_threshold,
            embed_fn=embed_fn,
        )
        self.geodesic_selector = GeodesicContextSelector(embed_fn=embed_fn)

        # Initialize base manager
        self._base_manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=embed_fn,
            E_threshold=base_E_threshold,
        )

        self._paradigm_tagged_turns: List[ParadigmTaggedTurn] = []

    def prepare_context_paradigm_aware(
        self,
        query: str,
        context_texts: Optional[List[str]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[PrepareContextResult, ParadigmState]:
        """
        Prepare context with paradigm awareness.

        1. Update paradigm state from recent texts
        2. Adjust E-threshold based on regime
        3. Boost geodesic-aligned items
        4. Return context + paradigm state

        Args:
            query: User query
            context_texts: Recent conversation texts for paradigm detection
            query_embedding: Pre-computed query embedding

        Returns:
            Tuple of (PrepareContextResult, ParadigmState)
        """
        # Update paradigm state
        texts_for_detection = context_texts or [query]
        paradigm_state = self.paradigm_tracker.update(texts_for_detection)

        # Prepare context with base manager (will be enhanced later)
        prepare_result = self._base_manager.prepare_context(query, query_embedding)

        return prepare_result, paradigm_state

    def finalize_turn_with_paradigm(
        self,
        user_query: str,
        assistant_response: str,
    ):
        """
        Finalize turn and tag with current paradigm state.

        Args:
            user_query: User's query
            assistant_response: Assistant's response

        Returns:
            FinalizeResult from base manager
        """
        # Get current paradigm state
        paradigm_state = self.paradigm_tracker.current_state

        # Create paradigm-tagged turn
        turn_id = f"turn_{len(self._paradigm_tagged_turns):04d}"
        tokens = len(user_query + assistant_response) // 4

        tagged_turn = ParadigmTaggedTurn(
            turn_id=turn_id,
            content=f"User: {user_query}\nAssistant: {assistant_response}",
            tokens=tokens,
            paradigm_state=paradigm_state,
        )
        self._paradigm_tagged_turns.append(tagged_turn)

        # Finalize with base manager
        return self._base_manager.finalize_turn(user_query, assistant_response, turn_id)

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get current paradigm regime summary."""
        return self.paradigm_tracker.get_regime_summary()

    def get_shift_turns(self) -> List[ParadigmTaggedTurn]:
        """Get all turns created during SHIFT regime."""
        return [t for t in self._paradigm_tagged_turns if t.is_shift_turn]

    def get_stable_turns(self) -> List[ParadigmTaggedTurn]:
        """Get all turns created during STABLE regime."""
        return [t for t in self._paradigm_tagged_turns if t.is_stable_turn]


# =============================================================================
# Multi-Agent Paradigm Coordination
# =============================================================================

class ParadigmCoordinator:
    """
    Coordinate paradigm states across multiple agents.

    When agents detect similar paradigm states, they can share context.
    When agents are in different states, cross-communication triggers
    paradigm detection.
    """

    def __init__(self):
        self._agent_states: Dict[str, ParadigmState] = {}
        self._shared_context: Dict[str, List[ContextItem]] = {}

    def register_agent_state(self, agent_id: str, state: ParadigmState):
        """Register an agent's current paradigm state."""
        self._agent_states[agent_id] = state

    def get_regime_clusters(self) -> Dict[str, List[str]]:
        """
        Cluster agents by regime type.

        Returns:
            Dict mapping regime type to list of agent IDs
        """
        clusters = {'SHIFT': [], 'STABLE': [], 'TRANSITIONAL': []}
        for agent_id, state in self._agent_states.items():
            clusters[state.shift_type].append(agent_id)
        return clusters

    def should_share_context(self, agent_a: str, agent_b: str) -> bool:
        """
        Determine if two agents should share context.

        Agents in the same regime can benefit from shared context.
        Agents in different regimes might cause interference.
        """
        state_a = self._agent_states.get(agent_a)
        state_b = self._agent_states.get(agent_b)

        if state_a is None or state_b is None:
            return False

        # Same regime = can share
        if state_a.shift_type == state_b.shift_type:
            return True

        # SHIFT agents can share with TRANSITIONAL
        if {state_a.shift_type, state_b.shift_type} == {'SHIFT', 'TRANSITIONAL'}:
            return True

        # STABLE agents prefer not to share with SHIFT
        return False

    def get_shared_context_for_agent(self, agent_id: str) -> List[ContextItem]:
        """Get context items shared by agents in the same regime."""
        state = self._agent_states.get(agent_id)
        if state is None:
            return []

        regime = state.shift_type
        return self._shared_context.get(regime, [])

    def share_context(self, agent_id: str, items: List[ContextItem]):
        """Share context items from an agent to its regime pool."""
        state = self._agent_states.get(agent_id)
        if state is None:
            return

        regime = state.shift_type
        if regime not in self._shared_context:
            self._shared_context[regime] = []

        self._shared_context[regime].extend(items)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate paradigm-aware hydration."""
    print("=" * 60)
    print("PARADIGM-AWARE HYDRATION DEMO")
    print("=" * 60)

    if not PARADIGM_DETECTOR_AVAILABLE:
        print("ERROR: Paradigm detector not available")
        return

    # Initialize tracker
    tracker = ParadigmStateTracker()

    # Simulate conversation phases
    phases = [
        {
            'name': 'Normal Discussion',
            'texts': [
                "Let's review the quarterly reports",
                "The metrics look consistent with last quarter",
                "Steady progress on all initiatives",
            ],
        },
        {
            'name': 'Early Warning',
            'texts': [
                "I'm seeing some unusual patterns in the data",
                "Something feels different this quarter",
                "We should investigate these anomalies",
            ],
        },
        {
            'name': 'Escalation',
            'texts': [
                "The situation is changing rapidly",
                "Old assumptions no longer hold",
                "We need to adapt our strategy",
            ],
        },
        {
            'name': 'Full Shift',
            'texts': [
                "Everything has fundamentally changed",
                "The old paradigm is dead",
                "We must transform completely",
            ],
        },
    ]

    print("\n--- PARADIGM STATE EVOLUTION ---\n")

    for phase in phases:
        state = tracker.update(phase['texts'])
        summary = tracker.get_regime_summary()

        print(f"{phase['name']}:")
        print(f"  Regime: {state.shift_type}")
        print(f"  Score: {state.shift_score:+.4f}")
        print(f"  Top Geodesic: {state.top_geodesic}")
        print(f"  Trend: {summary['trend']}")
        print()

    # Show threshold adjustments
    print("\n--- THRESHOLD ADJUSTMENTS ---\n")

    partitioner = ParadigmAwarePartitioner(base_threshold=0.5)

    for regime in ['SHIFT', 'TRANSITIONAL', 'STABLE']:
        state = ParadigmState(
            shift_score=0.2 if regime == 'SHIFT' else (-0.1 if regime == 'STABLE' else 0.0),
            shift_type=regime,
            top_geodesic='Earthquake' if regime == 'SHIFT' else 'Dog',
            geodesic_profile={},
            timestamp='',
            turn_index=0,
        )

        if regime == 'SHIFT':
            adj = partitioner.shift_adjustment
        elif regime == 'STABLE':
            adj = partitioner.stable_adjustment
        else:
            adj = 0.0

        effective = partitioner.base_threshold + adj

        print(f"{regime}:")
        print(f"  Base threshold: {partitioner.base_threshold}")
        print(f"  Adjustment: {adj:+.2f}")
        print(f"  Effective threshold: {effective}")
        print()

    print("=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    print("""
The paradigm-aware hydration system provides:

1. REGIME DETECTION
   - Track semantic field shift/stable state per turn
   - Rolling window smooths transient fluctuations

2. ADAPTIVE THRESHOLDS
   - SHIFT regime: Lower E-threshold (bring diverse context)
   - STABLE regime: Higher E-threshold (focus on relevant)
   - TRANSITIONAL: Use base threshold

3. GEODESIC ALIGNMENT
   - Boost context that aligns with active geodesic
   - During EARTHQUAKE: prioritize transition-relevant context
   - During DOG: prioritize established pattern context

4. PARADIGM-TAGGED TURNS
   - Each turn carries its paradigm state
   - Shift turns get priority hydration during shifts
   - Stable turns compressed more aggressively

5. MULTI-AGENT SCALING
   - Agents in same regime can share context
   - Cross-regime communication triggers detection
   - Paradigm-aware context pools

This integrates with existing cat_chat hydration via:
- ParadigmAwarePartitioner extends ContextPartitioner
- ParadigmAwareContextManager wraps AutoContextManager
- ParadigmCoordinator manages multi-agent coordination
""")


if __name__ == "__main__":
    demo()
