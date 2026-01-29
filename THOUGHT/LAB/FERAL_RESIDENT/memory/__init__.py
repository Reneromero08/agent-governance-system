# memory/ - Memory systems (episodic, semantic, procedural)
# All storage, retrieval, and memory composition operations

from .geometric_memory import GeometricMemory
from .vector_store import VectorStore
from .resident_db import ResidentDB, VectorRecord, InteractionRecord, ThreadRecord
from .e_graph import ERelationshipGraph, EEdge, RTier, E_THRESHOLD, COMPARE_TO_RECENT
from .e_patterns import EPatternDetector, Cluster, Bridge, UnionFind
from .e_query import EQueryEngine, RelatedItem, PathResult

__all__ = [
    # Core memory
    "GeometricMemory",
    "VectorStore",
    "ResidentDB",
    "VectorRecord",
    "InteractionRecord",
    "ThreadRecord",
    # E-relationship graph (Phase 2)
    "ERelationshipGraph",
    "EEdge",
    "RTier",
    "E_THRESHOLD",
    "COMPARE_TO_RECENT",
    # E-pattern detection (Phase 3)
    "EPatternDetector",
    "Cluster",
    "Bridge",
    "UnionFind",
    # E-query engine (Phase 4)
    "EQueryEngine",
    "RelatedItem",
    "PathResult",
]
