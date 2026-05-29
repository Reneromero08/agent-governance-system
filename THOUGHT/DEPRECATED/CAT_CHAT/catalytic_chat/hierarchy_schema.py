#!/usr/bin/env python3
"""
Hierarchy Schema (Phase J.1)

Defines the centroid hierarchy for O(log n) retrieval over 100K+ turns.

Hierarchy Levels:
- L0 (turns): Individual conversation turns
- L1 (100 turns): Centroid of 100 L0 vectors
- L2 (1000 turns): Centroid of 10 L1 centroids (100 * 10)
- L3 (10000 turns): Centroid of 10 L2 centroids (1000 * 10)

Each level aggregates CHILDREN_PER_LEVEL (100) children from the level below.
This enables O(log n) retrieval by searching top-down through the hierarchy.

Part of Phase J.1: Centroid Hierarchy Schema.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json

import numpy as np


# Level constants
L0 = 0  # Individual turns
L1 = 1  # Centroid of 100 turns
L2 = 2  # Centroid of 1000 turns (10 L1 nodes)
L3 = 3  # Centroid of 10000 turns (10 L2 nodes)

# Number of children per parent node at each level
# L1 has 100 L0 children, L2 has 100 L1 children, etc.
CHILDREN_PER_LEVEL = 100

# Embedding dimensions (must match vector_persistence.py)
EMBEDDING_DIM = 384
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32 = 4 bytes

# Human-readable level names
LEVEL_NAMES = {
    L0: "turn",
    L1: "century",      # 100 turns
    L2: "millennium",   # 1000 turns
    L3: "epoch",        # 10000 turns
}


def _now_iso() -> str:
    """Get ISO8601 timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class HierarchyNode:
    """
    A node in the centroid hierarchy.

    Represents either an individual turn (L0) or an aggregate centroid (L1-L3).

    Attributes:
        node_id: Unique identifier for this node
        session_id: Session this node belongs to
        level: Hierarchy level (L0, L1, L2, or L3)
        centroid: Vector centroid as numpy array (384 dimensions)
        parent_id: ID of parent node (None for root/unassigned)
        turn_count: Number of turns this node represents
        first_turn_seq: Sequence number of first turn in this node's span
        last_turn_seq: Sequence number of last turn in this node's span
        created_at: ISO timestamp of creation
        is_archived: Whether this node has been archived (for compaction)
        last_accessed_at: ISO timestamp of last access (for LRU eviction)
    """
    node_id: str
    session_id: str
    level: int
    centroid: np.ndarray
    parent_id: Optional[str] = None
    turn_count: int = 1
    first_turn_seq: Optional[int] = None
    last_turn_seq: Optional[int] = None
    created_at: str = field(default_factory=_now_iso)
    is_archived: bool = False
    last_accessed_at: Optional[str] = None

    def __post_init__(self):
        """Validate node after initialization."""
        # Validate level
        if self.level not in (L0, L1, L2, L3):
            raise ValueError(f"Invalid level: {self.level}. Must be L0-L3.")

        # Validate centroid shape
        if self.centroid is not None:
            if not isinstance(self.centroid, np.ndarray):
                raise TypeError(
                    f"centroid must be numpy array, got {type(self.centroid)}"
                )
            if self.centroid.shape != (EMBEDDING_DIM,):
                raise ValueError(
                    f"centroid must have shape ({EMBEDDING_DIM},), "
                    f"got {self.centroid.shape}"
                )

    @property
    def level_name(self) -> str:
        """Human-readable level name."""
        return LEVEL_NAMES.get(self.level, f"L{self.level}")

    @property
    def max_turns(self) -> int:
        """Maximum number of turns this level can represent."""
        return CHILDREN_PER_LEVEL ** self.level

    @property
    def is_full(self) -> bool:
        """Check if this node has reached its maximum turn capacity."""
        return self.turn_count >= self.max_turns

    def serialize_centroid(self) -> bytes:
        """Serialize centroid to bytes for database storage.

        Returns:
            bytes representation (1536 bytes = 384 * 4)
        """
        return self.centroid.astype(np.float32).tobytes()

    @classmethod
    def deserialize_centroid(cls, blob: bytes) -> np.ndarray:
        """Deserialize centroid from database storage.

        Args:
            blob: bytes representation from database

        Returns:
            numpy array of shape (384,)

        Raises:
            ValueError: If blob size is incorrect
        """
        if len(blob) != EMBEDDING_BYTES:
            raise ValueError(
                f"Expected {EMBEDDING_BYTES} bytes, got {len(blob)}"
            )
        return np.frombuffer(blob, dtype=np.float32).copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Note: centroid is converted to list for JSON compatibility.
        """
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "level": self.level,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "parent_id": self.parent_id,
            "turn_count": self.turn_count,
            "first_turn_seq": self.first_turn_seq,
            "last_turn_seq": self.last_turn_seq,
            "created_at": self.created_at,
            "is_archived": self.is_archived,
            "last_accessed_at": self.last_accessed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchyNode":
        """Create HierarchyNode from dictionary.

        Args:
            data: Dictionary with node attributes

        Returns:
            HierarchyNode instance
        """
        centroid = data.get("centroid")
        if centroid is not None and not isinstance(centroid, np.ndarray):
            centroid = np.array(centroid, dtype=np.float32)

        return cls(
            node_id=data["node_id"],
            session_id=data["session_id"],
            level=data["level"],
            centroid=centroid,
            parent_id=data.get("parent_id"),
            turn_count=data.get("turn_count", 1),
            first_turn_seq=data.get("first_turn_seq"),
            last_turn_seq=data.get("last_turn_seq"),
            created_at=data.get("created_at", _now_iso()),
            is_archived=data.get("is_archived", False),
            last_accessed_at=data.get("last_accessed_at"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "HierarchyNode":
        """Create HierarchyNode from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __eq__(self, other: object) -> bool:
        """Check equality (for testing)."""
        if not isinstance(other, HierarchyNode):
            return False
        return (
            self.node_id == other.node_id
            and self.session_id == other.session_id
            and self.level == other.level
            and self.parent_id == other.parent_id
            and self.turn_count == other.turn_count
            and self.first_turn_seq == other.first_turn_seq
            and self.last_turn_seq == other.last_turn_seq
            and np.allclose(self.centroid, other.centroid)
        )


def generate_node_id(session_id: str, level: int, sequence: int) -> str:
    """Generate a deterministic node ID.

    Format: h{level}_{session_id}_{sequence}

    Args:
        session_id: Session identifier
        level: Hierarchy level (0-3)
        sequence: Sequential number within this level

    Returns:
        Node ID string
    """
    return f"h{level}_{session_id}_{sequence}"


def get_turns_per_level(level: int) -> int:
    """Get the number of turns represented by one node at a given level.

    Args:
        level: Hierarchy level (0-3)

    Returns:
        Number of turns: 1 for L0, 100 for L1, 10000 for L2, 1000000 for L3
    """
    if level < 0:
        raise ValueError(f"Level must be non-negative, got {level}")
    return CHILDREN_PER_LEVEL ** level


if __name__ == "__main__":
    # Self-test
    print("Testing HierarchyNode...")

    # Create test node
    centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    node = HierarchyNode(
        node_id="h1_test_session_1",
        session_id="test_session",
        level=L1,
        centroid=centroid,
        turn_count=50,
        first_turn_seq=1,
        last_turn_seq=50,
    )

    print(f"Node ID: {node.node_id}")
    print(f"Level: {node.level} ({node.level_name})")
    print(f"Max turns: {node.max_turns}")
    print(f"Is full: {node.is_full}")

    # Test serialization
    blob = node.serialize_centroid()
    print(f"Serialized size: {len(blob)} bytes")
    assert len(blob) == EMBEDDING_BYTES

    # Test deserialization
    recovered = HierarchyNode.deserialize_centroid(blob)
    assert np.allclose(centroid, recovered)
    print("Serialization roundtrip passed")

    # Test to_dict/from_dict
    data = node.to_dict()
    recovered_node = HierarchyNode.from_dict(data)
    assert node == recovered_node
    print("Dict roundtrip passed")

    # Test to_json/from_json
    json_str = node.to_json()
    recovered_node = HierarchyNode.from_json(json_str)
    assert node == recovered_node
    print("JSON roundtrip passed")

    # Test level constants
    assert get_turns_per_level(L0) == 1
    assert get_turns_per_level(L1) == 100
    assert get_turns_per_level(L2) == 10000
    assert get_turns_per_level(L3) == 1000000
    print("Level calculations passed")

    # Test node ID generation
    node_id = generate_node_id("session_123", L2, 5)
    assert node_id == "h2_session_123_5"
    print("Node ID generation passed")

    print("\nAll tests passed!")
