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


# Optimized 64-word anchor set - most stable across models
# Selected by analyzing cross-model distance matrix correlations
# across nomic-embed-v1.5, all-MiniLM-L6-v2, and all-mpnet-base-v2
# Lower Procrustes residual = better cross-model alignment
STABLE_64: List[str] = [
    # Highest stability (0.60+): concrete nouns, nature, seasons
    "outside", "nature", "animal", "tree", "science", "summer", "water", "technology",
    "autumn", "plant", "mountain", "spring", "winter", "wood", "car", "building",
    # High stability (0.55+): objects, domains
    "book", "machine", "earth", "paper", "art", "music", "food", "stone",
    "space", "enemy", "river", "math", "house", "north", "effect", "dog",
    # Medium stability (0.50+): mixed
    "glass", "cat", "road", "walk", "know", "leader", "air", "teacher",
    "evening", "person", "destroy", "language", "morning", "see", "fire", "answer",
    # Lower stability (0.45+): actions, senses
    "fast", "child", "question", "speak", "problem", "dark", "night", "society",
    "touch", "taste", "think", "sad", "cold", "south", "give", "hear",
]

STABLE_64_HASH = compute_anchor_hash(STABLE_64)


# Ultra-stable 32-word anchor set - BEST for cross-model alignment
# Selected by per-anchor alignment error analysis across nomic, MiniLM, MPNet
# Achieves 59% residual reduction vs STABLE_64 (1.08 vs 2.63)
# Use when cross-model communication is more important than same-model redundancy
# Discovered 2026-01-17 via find_stable_anchors.py analysis
STABLE_32: List[str] = [
    # Lowest error (0.21-0.27): actions, effects, animals
    "destroy", "effect", "animal", "fast", "art", "cold", "child", "walk",
    # Low error (0.27-0.29): objects, concepts
    "stone", "think", "give", "space", "society", "glass", "touch", "air",
    # Medium-low error (0.29-0.30): nature, times
    "evening", "mountain", "book", "leader", "sad", "dog", "cat", "winter",
    # Medium error (0.30-0.32): buildings, people
    "wood", "morning", "know", "fire", "car", "building", "person", "enemy",
]

STABLE_32_HASH = compute_anchor_hash(STABLE_32)


def get_recommended_anchors(priority: str = "stability") -> List[str]:
    """Get recommended anchor set based on priority.

    Args:
        priority: "cross_model" for best cross-model alignment (STABLE_32),
                  "stability" for good balance (STABLE_64),
                  "coverage" for maximum semantic coverage (CANONICAL_128)

    Returns:
        Recommended anchor set
    """
    if priority == "cross_model":
        return STABLE_32
    elif priority == "stability":
        return STABLE_64
    else:
        return CANONICAL_128


def get_anchor_sets() -> dict:
    """Get all available anchor sets with their properties.

    Returns:
        Dict mapping name to (anchor_list, hash, description)
    """
    return {
        "CANONICAL_128": (CANONICAL_128, CANONICAL_128_HASH,
                         "Full coverage, max redundancy, higher residual"),
        "STABLE_64": (STABLE_64, STABLE_64_HASH,
                      "Good balance of coverage and stability"),
        "STABLE_32": (STABLE_32, STABLE_32_HASH,
                      "Best cross-model alignment, 59% lower residual"),
    }
