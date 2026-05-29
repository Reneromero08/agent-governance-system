"""
PARADIGM-AWARE HYDRATION DEMO

Demonstrates how paradigm detection integrates with cat_chat's hydration system.
Standalone demo - doesn't require full cat_chat imports.
"""

import numpy as np
from pathlib import Path
import sys

# Add compass_agi to path
COMPASS_PATH = Path(__file__).parent
if str(COMPASS_PATH) not in sys.path:
    sys.path.insert(0, str(COMPASS_PATH))

from realtime_paradigm_detector import (
    ParadigmShiftDetector,
    SHIFT_GEODESICS,
    STABILITY_GEODESICS,
)


class ParadigmStateTracker:
    """Track paradigm state over conversation turns."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._readings = []
        self._detector = ParadigmShiftDetector()

    def update(self, texts):
        """Update with new texts, return current state."""
        result = self._detector.detect_shift(texts)
        self._readings.append(result)
        if len(self._readings) > self.window_size:
            self._readings.pop(0)
        return result

    @property
    def trend(self):
        if len(self._readings) < 2:
            return 'STEADY'
        scores = [r['shift_score'] for r in self._readings]
        diff = scores[-1] - scores[0]
        if diff > 0.05:
            return 'SHIFTING'
        elif diff < -0.05:
            return 'STABILIZING'
        return 'STEADY'


class ParadigmAwarePartitioner:
    """Adjust E-threshold based on paradigm state."""

    def __init__(self, base_threshold=0.5):
        self.base_threshold = base_threshold
        self.shift_adjustment = -0.15  # Lower during shift
        self.stable_adjustment = +0.10  # Raise during stable

    def get_effective_threshold(self, regime):
        if regime == 'SHIFT':
            return self.base_threshold + self.shift_adjustment
        elif regime == 'STABLE':
            return self.base_threshold + self.stable_adjustment
        return self.base_threshold


def demo():
    print("=" * 70)
    print("PARADIGM-AWARE HYDRATION: How it Integrates with cat_chat")
    print("=" * 70)

    tracker = ParadigmStateTracker()
    partitioner = ParadigmAwarePartitioner()

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
        {
            'name': 'New Stability',
            'texts': [
                "We've established new procedures",
                "The team is loyal to the new approach",
                "Trust has been rebuilt on solid foundation",
            ],
        },
    ]

    print("\n--- PARADIGM STATE EVOLUTION ---\n")
    print(f"{'Phase':<20} {'Regime':<15} {'Score':>10} {'Threshold':>12} {'Trend':<12}")
    print("-" * 70)

    for phase in phases:
        state = tracker.update(phase['texts'])
        regime = state['shift_type']
        threshold = partitioner.get_effective_threshold(regime)
        trend = tracker.trend

        print(f"{phase['name']:<20} {regime:<15} {state['shift_score']:>+10.4f} {threshold:>12.2f} {trend:<12}")

    # Show the key integration points
    print("\n" + "=" * 70)
    print("INTEGRATION WITH cat_chat HYDRATION")
    print("=" * 70)

    print("""
CURRENT cat_chat HYDRATION:
- ContextPartitioner uses E-score (cosine similarity) to rank items
- Items above E-threshold go to working_set (hydrated)
- Items below E-threshold stay in pointer_set (compressed)
- Single fixed threshold for all turns

PARADIGM-AWARE ENHANCEMENT:

1. ADAPTIVE E-THRESHOLD
   -------------------------------------------
   Regime          | Adjustment | Effective
   -------------------------------------------
   SHIFT           | -0.15      | 0.35
   TRANSITIONAL    |  0.00      | 0.50
   STABLE          | +0.10      | 0.60
   -------------------------------------------

   WHY: During SHIFT, you need diverse perspectives (lower threshold).
        During STABLE, focus on most relevant (higher threshold).

2. PARADIGM-TAGGED TURNS
   - Each turn stores its paradigm state when created
   - SHIFT turns get priority hydration during future shifts
   - STABLE turns can stay compressed longer
   - Enables "regime-coherent" context windows

3. GEODESIC-ALIGNED CONTEXT SELECTION
   - Boost E-scores of items matching active geodesic
   - During EARTHQUAKE: prioritize transition content
   - During DOG: prioritize established pattern content
   - Alignment = cosine(item_embedding, geodesic_embedding)

4. MULTI-AGENT SCALING
   -------------------------------------------
   Agent A (SHIFT)  <---> Agent B (SHIFT)   = CAN SHARE CONTEXT
   Agent A (STABLE) <---> Agent B (STABLE)  = CAN SHARE CONTEXT
   Agent A (SHIFT)  <-X-> Agent B (STABLE)  = PREFER NOT TO SHARE
   -------------------------------------------

   WHY: Agents in same regime benefit from shared context.
        Cross-regime sharing may cause confusion.

IMPLEMENTATION IN AutoContextManager:

```python
# Before
result = partitioner.partition(
    query_embedding=query_vec,
    all_items=items,
    budget_tokens=30000
)

# After (paradigm-aware)
paradigm_state = tracker.update(recent_texts)
effective_threshold = partitioner.get_threshold(paradigm_state)
result = partitioner.partition_with_paradigm(
    query_embedding=query_vec,
    all_items=items,
    budget_tokens=30000,
    paradigm_state=paradigm_state
)
```

The paradigm detector adds ~50ms overhead per turn (embedding + cosine).
This is negligible compared to LLM latency.
""")

    # Final summary
    print("=" * 70)
    print("SCALING BENEFITS")
    print("=" * 70)

    print("""
1. REGIME-AWARE CONTEXT WINDOWS
   - Each agent maintains optimal context for its regime
   - Shift agents get broader context (lower threshold)
   - Stable agents get focused context (higher threshold)

2. CROSS-AGENT PARADIGM COORDINATION
   - ParadigmCoordinator tracks all agent states
   - Agents in same regime can share context pools
   - Reduces redundant hydration across swarm

3. TEMPORAL COHERENCE
   - Turns tagged with paradigm state
   - Future hydration respects temporal regime
   - "Shift memory" vs "stable memory" partitioning

4. EARLY WARNING PROPAGATION
   - When one agent detects SHIFT, can alert others
   - Swarm-level regime transitions
   - Coordinated context pivot

This transforms cat_chat from:
  "Hydrate based on relevance"
To:
  "Hydrate based on relevance AND semantic regime"

The compass tells you WHAT kind of moment you're in.
Hydration becomes paradigm-aware.
""")


if __name__ == "__main__":
    demo()
