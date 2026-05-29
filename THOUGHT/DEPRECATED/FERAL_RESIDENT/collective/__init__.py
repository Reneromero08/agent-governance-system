# collective/ - Social cognition
# Multi-agent coordination and observation

from .swarm_coordinator import (
    SwarmCoordinator,
    ResidentConfig,
    ResidentStatus,
    SwarmStatus,
    BroadcastResult,
)
from .shared_space import (
    SharedSemanticSpace,
    PublishedVector,
    ConvergenceEvent,
)
from .convergence_observer import (
    ConvergenceObserver,
    ConvergenceMetrics,
    SwarmConvergenceSummary,
    ConvergenceObservation,
)

__all__ = [
    # swarm_coordinator
    "SwarmCoordinator",
    "ResidentConfig",
    "ResidentStatus",
    "SwarmStatus",
    "BroadcastResult",
    # shared_space
    "SharedSemanticSpace",
    "PublishedVector",
    "ConvergenceEvent",
    # convergence_observer
    "ConvergenceObserver",
    "ConvergenceMetrics",
    "SwarmConvergenceSummary",
    "ConvergenceObservation",
]
