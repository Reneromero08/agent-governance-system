"""Canonical Anchor Set for Vector Communication.

The anchor set is the shared vocabulary that enables cross-model
vector communication. Two models compute MDS coordinates using
the same anchors, then Procrustes rotation aligns their spaces.

The canonical set has 128 words covering:
- Concrete nouns (8)
- Abstract concepts (8)
- Actions (8)
- Properties (8)
- Relations (8)
- Quantities (8)
- Domains (8)
- Meta concepts (8)
- Extended: persons, emotions, senses, colors, directions, times, roles, elements (64)

The anchor hash is computed as SHA-256 of the sorted, newline-joined words.
This enables verification that two parties share the same anchor set.
"""

import hashlib
from typing import List


# Canonical 128-word anchor set
# DO NOT MODIFY - changing this breaks compatibility with existing keys
CANONICAL_128: List[str] = [
    # Concrete nouns
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    # Abstract concepts
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    # Actions
    "run", "walk", "think", "speak", "create", "destroy", "give", "take",
    # Properties
    "big", "small", "fast", "slow", "hot", "cold", "good", "bad",
    # Relations
    "above", "below", "inside", "outside", "before", "after", "with", "without",
    # Quantities
    "one", "many", "all", "none", "more", "less", "equal", "different",
    # Domains
    "science", "art", "music", "math", "language", "nature", "technology", "society",
    # Meta concepts
    "question", "answer", "problem", "solution", "cause", "effect", "begin", "end",
    # Extended set
    "person", "animal", "plant", "machine", "building", "road", "mountain", "river",
    # Emotions
    "happy", "sad", "angry", "calm", "excited", "bored", "curious", "confused",
    # Senses
    "see", "hear", "touch", "smell", "taste", "feel", "know", "believe",
    # Colors
    "red", "blue", "green", "white", "black", "bright", "dark", "clear",
    # Directions
    "north", "south", "east", "west", "up", "down", "left", "right",
    # Times
    "day", "night", "morning", "evening", "spring", "summer", "autumn", "winter",
    # Roles
    "mother", "father", "child", "friend", "enemy", "leader", "worker", "teacher",
    # Elements
    "earth", "fire", "air", "metal", "stone", "wood", "glass", "paper",
]


def compute_anchor_hash(anchors: List[str]) -> str:
    """Compute deterministic hash of an anchor set.

    The hash is computed over the sorted, newline-joined words.
    This ensures different orderings produce the same hash.

    Args:
        anchors: List of anchor words

    Returns:
        SHA-256 hash (first 16 hex chars for brevity)
    """
    canonical = "\n".join(sorted(anchors))
    full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return full_hash[:16]


def verify_anchor_hash(anchors: List[str], expected_hash: str) -> bool:
    """Verify an anchor set matches an expected hash.

    Args:
        anchors: List of anchor words
        expected_hash: Expected hash value

    Returns:
        True if hash matches, False otherwise
    """
    actual = compute_anchor_hash(anchors)
    return actual == expected_hash


def get_canonical_hash() -> str:
    """Get the hash of the canonical 128-word anchor set.

    Returns:
        SHA-256 hash of CANONICAL_128
    """
    return compute_anchor_hash(CANONICAL_128)


# Pre-computed hash for quick verification
CANONICAL_128_HASH = compute_anchor_hash(CANONICAL_128)
