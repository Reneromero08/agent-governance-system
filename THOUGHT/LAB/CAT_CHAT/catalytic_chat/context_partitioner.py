"""
Context Partitioner
===================

Unified E-score based partitioning of context items into working_set and pointer_set.

Key Design Principle: No separate "eviction" and "hydration" operations.
Every turn, ALL items (working + pointer) are re-scored and re-partitioned
based on the current query's E-scores.

Algorithm:
1. Score ALL items (working_set + pointer_set) against current query
2. Sort by E-score descending
3. Fill working_set until budget exhausted
4. Items below E-threshold always go to pointer_set
5. Remainder goes to pointer_set

Phase C.2 of Auto-Controlled Context Loop implementation.
"""

import hashlib
import json
import re
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable, Set

# Import E-score computation from Q44
import sys
from pathlib import Path

# Add Q44 path for import
Q44_PATH = Path(__file__).parent.parent.parent / "FORMULA" / "experiments" / "open_questions" / "q44"
if str(Q44_PATH) not in sys.path:
    sys.path.insert(0, str(Q44_PATH))

try:
    from q44_core import compute_E_linear, normalize
except ImportError:
    # Fallback implementation if q44_core not available
    def normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)

    def compute_E_linear(
        query_vec: np.ndarray,
        context_vecs: List[np.ndarray]
    ) -> Tuple[float, List[float]]:
        if len(context_vecs) == 0:
            return 0.0, []
        # Filter out non-numeric vectors
        valid_vecs = [v for v in context_vecs
                      if isinstance(v, np.ndarray) and v.dtype.kind in ('f', 'i', 'u')]
        if len(valid_vecs) == 0:
            return 0.0, []
        psi = normalize(query_vec)
        overlaps = [float(np.dot(psi, normalize(phi))) for phi in valid_vecs]
        return float(np.mean(overlaps)), overlaps


# =============================================================================
# Hybrid Retrieval: Keyword + Semantic
# =============================================================================

# Common stop words to exclude from keyword matching
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "this", "that", "these", "those", "what",
    "which", "who", "whom", "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it",
    "its", "they", "them", "their", "theirs", "am", "about", "get", "got",
}


def extract_keywords(text: str, min_length: int = 2) -> Set[str]:
    """
    Extract meaningful keywords from text.

    - Tokenizes on word boundaries
    - Lowercases for matching
    - Removes stop words
    - Keeps words >= min_length
    - Preserves multi-word entities (e.g., "Shadow Fox" -> "shadow", "fox")

    Args:
        text: Input text
        min_length: Minimum word length to keep

    Returns:
        Set of lowercase keywords
    """
    # Tokenize: split on non-alphanumeric, keep alphanumeric sequences
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())

    # Filter: remove stop words and short words
    keywords = {
        w for w in words
        if len(w) >= min_length and w not in STOP_WORDS
    }

    return keywords


def compute_keyword_score(query_keywords: Set[str], item_text: str) -> float:
    """
    Compute keyword match score between query keywords and item text.

    Returns:
        Score in [0, 1] representing fraction of query keywords found in item
    """
    if not query_keywords:
        return 0.0

    item_lower = item_text.lower()
    matches = sum(1 for kw in query_keywords if kw in item_lower)

    return matches / len(query_keywords)


def compute_hybrid_score(
    semantic_score: float,
    keyword_score: float,
    keyword_boost: float = 1.0,
    tiered_bonus: bool = True
) -> float:
    """
    Combine semantic and keyword scores for hybrid retrieval.

    Formula:
        hybrid = semantic + keyword_boost * keyword_score + tiered_bonus

    Uses tiered bonus system:
    - >40% keyword match: +0.5 (likely exact entity match)
    - >25% keyword match: +0.2 (partial entity match)
    - Any match: +0.1 (at least some relevance)

    This strongly prioritizes items matching more query keywords,
    which is crucial for entity discrimination in high-interference data.

    Args:
        semantic_score: E-score from embedding dot product (typically 0.5-0.9)
        keyword_score: Fraction of keywords matched (0-1)
        keyword_boost: Weight for keyword score component (default: 1.0)
        tiered_bonus: Apply tiered bonus based on match quality

    Returns:
        Combined score (can exceed 1.0)
    """
    hybrid = semantic_score + (keyword_boost * keyword_score)

    # Tiered bonus for keyword match quality
    if tiered_bonus:
        if keyword_score > 0.4:
            # High match - likely exact entity
            hybrid += 0.5
        elif keyword_score > 0.25:
            # Partial match
            hybrid += 0.2
        elif keyword_score > 0:
            # Any match
            hybrid += 0.1

    return hybrid


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ContextItem:
    """
    A context item that can be in working_set or pointer_set.

    Attributes:
        item_id: Unique identifier
        content: Full text content
        tokens: Token count (estimated or exact)
        embedding: Vector embedding for E-score computation
        item_type: Type of item (message, expansion, turn, etc.)
        metadata: Additional metadata (timestamps, sources, etc.)
    """
    item_id: str
    content: str
    tokens: int
    embedding: Optional[np.ndarray] = None
    item_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure embedding is numpy array if provided."""
        if self.embedding is not None:
            if isinstance(self.embedding, str):
                # String passed as embedding - this is a bug, clear it
                self.embedding = None
            elif not isinstance(self.embedding, np.ndarray):
                arr = np.array(self.embedding, dtype=np.float32)
                # Validate it's actually numeric
                if arr.dtype.kind not in ('f', 'i', 'u'):
                    self.embedding = None
                else:
                    self.embedding = arr

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding for serialization)."""
        return {
            "item_id": self.item_id,
            "content_hash": hashlib.sha256(self.content.encode()).hexdigest()[:16],
            "tokens": self.tokens,
            "item_type": self.item_type,
            "metadata": self.metadata,
        }


@dataclass
class ScoredItem:
    """
    A context item with its E-score against current query.
    """
    item: ContextItem
    E_score: float
    rank: int = 0  # Position in sorted order (0 = highest E)


@dataclass
class PartitionResult:
    """
    Result of partitioning items into working_set and pointer_set.

    Provides full provenance for deterministic replay.
    """
    working_set: List[ScoredItem]
    pointer_set: List[ScoredItem]

    # Partition metrics
    query_hash: str  # Hash of query for replay
    threshold: float  # E-threshold used
    budget_total: int  # Total budget available
    budget_used: int  # Tokens used by working_set

    # Statistics
    items_total: int
    items_in_working_set: int
    items_below_threshold: int
    items_over_budget: int

    # E-score distribution
    E_mean: float
    E_min: float
    E_max: float
    E_std: float

    # Timestamp for logging
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event logging."""
        return {
            "query_hash": self.query_hash,
            "threshold": self.threshold,
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
            "items_total": self.items_total,
            "items_in_working_set": self.items_in_working_set,
            "items_below_threshold": self.items_below_threshold,
            "items_over_budget": self.items_over_budget,
            "E_mean": self.E_mean,
            "E_min": self.E_min,
            "E_max": self.E_max,
            "E_std": self.E_std,
            "working_set_ids": [s.item.item_id for s in self.working_set],
            "pointer_set_ids": [s.item.item_id for s in self.pointer_set],
            "timestamp": self.timestamp,
        }

    def compute_hash(self) -> str:
        """Compute deterministic hash for chain linking."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# =============================================================================
# Context Partitioner
# =============================================================================

class ContextPartitioner:
    """
    Partitions context items into working_set and pointer_set based on E-scores.

    Supports HYBRID RETRIEVAL: combines semantic (embedding) scores with
    keyword matching for better entity discrimination.

    Usage:
        partitioner = ContextPartitioner(
            threshold=0.5,
            embed_fn=my_embedding_function,
            enable_hybrid=True  # Enable keyword boosting
        )

        result = partitioner.partition(
            query_embedding=query_vec,
            all_items=items,
            budget_tokens=30000,
            query_text="Find Shadow Fox"  # Keywords extracted from this
        )

        # working_set has high-E items that fit budget
        # pointer_set has low-E items and overflow
    """

    def __init__(
        self,
        threshold: float = 0.5,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        token_estimator: Optional[Callable[[str], int]] = None,
        enable_hybrid: bool = True,
        keyword_boost: float = 1.0
    ):
        """
        Initialize partitioner.

        Args:
            threshold: E-score threshold (items below this always go to pointer_set)
            embed_fn: Function to compute embeddings from text (optional)
            token_estimator: Function to estimate tokens from text (default: len//4)
            enable_hybrid: Enable hybrid retrieval (semantic + keyword matching)
            keyword_boost: Weight for keyword match component (default: 1.0)
        """
        self.threshold = threshold
        self.embed_fn = embed_fn
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)
        self.enable_hybrid = enable_hybrid
        self.keyword_boost = keyword_boost

    def score_items(
        self,
        query_embedding: np.ndarray,
        items: List[ContextItem],
        query_text: str = ""
    ) -> List[ScoredItem]:
        """
        Score all items against query embedding.

        When hybrid mode is enabled, combines semantic E-score with
        keyword matching for better entity discrimination.

        Args:
            query_embedding: Query vector (will be normalized)
            items: List of context items with embeddings
            query_text: Original query text for keyword extraction (hybrid mode)

        Returns:
            List of ScoredItem sorted by E-score descending
        """
        if not items:
            return []

        # Extract keywords for hybrid scoring
        query_keywords = extract_keywords(query_text) if self.enable_hybrid and query_text else set()

        # Collect embeddings
        embeddings = []
        items_with_embeddings = []

        for item in items:
            if item.embedding is not None:
                embeddings.append(item.embedding)
                items_with_embeddings.append(item)
            elif self.embed_fn is not None:
                # Compute embedding on-the-fly
                emb = self.embed_fn(item.content)
                item.embedding = emb
                embeddings.append(emb)
                items_with_embeddings.append(item)
            else:
                # No embedding available - assign neutral E-score
                items_with_embeddings.append(item)
                embeddings.append(None)

        # Compute E-scores for items with embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        if valid_embeddings:
            _, E_scores = compute_E_linear(query_embedding, valid_embeddings)
        else:
            E_scores = []

        # Build scored items
        scored = []
        E_idx = 0
        for i, item in enumerate(items_with_embeddings):
            if embeddings[i] is not None:
                semantic_E = E_scores[E_idx]
                E_idx += 1
            else:
                # No embedding - use threshold as neutral score
                semantic_E = self.threshold

            # Apply hybrid scoring if enabled
            if self.enable_hybrid and query_keywords:
                keyword_score = compute_keyword_score(query_keywords, item.content)
                E = compute_hybrid_score(
                    semantic_score=semantic_E,
                    keyword_score=keyword_score,
                    keyword_boost=self.keyword_boost
                )
            else:
                E = semantic_E

            scored.append(ScoredItem(item=item, E_score=E))

        # Sort by E-score descending
        scored.sort(key=lambda s: s.E_score, reverse=True)

        # Assign ranks
        for rank, s in enumerate(scored):
            s.rank = rank

        return scored

    def partition(
        self,
        query_embedding: np.ndarray,
        all_items: List[ContextItem],
        budget_tokens: int,
        query_text: str = ""
    ) -> PartitionResult:
        """
        Partition items into working_set and pointer_set.

        Algorithm:
        1. Score all items against query
        2. Sort by E-score descending
        3. Fill working_set with high-E items until budget exhausted
        4. Items below threshold always go to pointer_set
        5. Remaining items go to pointer_set

        Args:
            query_embedding: Query vector
            all_items: All items to partition (current working + pointer sets)
            budget_tokens: Token budget for working_set
            query_text: Original query text for hashing

        Returns:
            PartitionResult with working_set, pointer_set, and metrics
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]

        if not all_items:
            return PartitionResult(
                working_set=[],
                pointer_set=[],
                query_hash=query_hash,
                threshold=self.threshold,
                budget_total=budget_tokens,
                budget_used=0,
                items_total=0,
                items_in_working_set=0,
                items_below_threshold=0,
                items_over_budget=0,
                E_mean=0.0,
                E_min=0.0,
                E_max=0.0,
                E_std=0.0,
                timestamp=timestamp,
            )

        # Score and sort all items (passes query_text for hybrid keyword matching)
        scored = self.score_items(query_embedding, all_items, query_text)

        # Partition based on threshold and budget
        working_set: List[ScoredItem] = []
        pointer_set: List[ScoredItem] = []
        tokens_used = 0
        items_below_threshold = 0
        items_over_budget = 0

        for s in scored:
            if s.E_score < self.threshold:
                # Below threshold - always pointer_set
                pointer_set.append(s)
                items_below_threshold += 1
            elif tokens_used + s.item.tokens <= budget_tokens:
                # Above threshold and fits budget - working_set
                working_set.append(s)
                tokens_used += s.item.tokens
            else:
                # Above threshold but over budget - pointer_set
                pointer_set.append(s)
                items_over_budget += 1

        # Compute E-score statistics
        E_values = [s.E_score for s in scored]
        E_mean = float(np.mean(E_values)) if E_values else 0.0
        E_min = float(np.min(E_values)) if E_values else 0.0
        E_max = float(np.max(E_values)) if E_values else 0.0
        E_std = float(np.std(E_values)) if len(E_values) > 1 else 0.0

        return PartitionResult(
            working_set=working_set,
            pointer_set=pointer_set,
            query_hash=query_hash,
            threshold=self.threshold,
            budget_total=budget_tokens,
            budget_used=tokens_used,
            items_total=len(all_items),
            items_in_working_set=len(working_set),
            items_below_threshold=items_below_threshold,
            items_over_budget=items_over_budget,
            E_mean=E_mean,
            E_min=E_min,
            E_max=E_max,
            E_std=E_std,
            timestamp=timestamp,
        )

    def adjust_threshold(self, new_threshold: float) -> None:
        """
        Adjust E-threshold for future partitions.

        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {new_threshold}")
        self.threshold = new_threshold


# =============================================================================
# Utility Functions
# =============================================================================

def items_to_text(items: List[ContextItem], separator: str = "\n\n") -> str:
    """Concatenate item contents into single text."""
    return separator.join(item.content for item in items)


def estimate_partition_budget(
    items: List[ContextItem],
    budget_tokens: int,
    threshold: float = 0.5,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Estimate how items would partition without actual E-scores.

    Useful for budget planning before embeddings are computed.

    Returns:
        Dict with estimated partition sizes and token distributions
    """
    total_tokens = sum(item.tokens for item in items)
    avg_tokens = total_tokens / len(items) if items else 0

    # Estimate: items roughly split by threshold (50/50 if threshold=0.5)
    estimated_above_threshold = int(len(items) * (1.0 - threshold))
    estimated_below_threshold = len(items) - estimated_above_threshold

    # Estimate how many would fit in budget
    estimated_in_budget = min(
        estimated_above_threshold,
        int(budget_tokens / avg_tokens) if avg_tokens > 0 else 0
    )

    return {
        "total_items": len(items),
        "total_tokens": total_tokens,
        "avg_tokens_per_item": avg_tokens,
        "budget_tokens": budget_tokens,
        "estimated_above_threshold": estimated_above_threshold,
        "estimated_below_threshold": estimated_below_threshold,
        "estimated_in_budget": estimated_in_budget,
        "estimated_overflow": max(0, estimated_above_threshold - estimated_in_budget),
    }


if __name__ == "__main__":
    # Quick sanity test
    print("Context Partitioner - Sanity Test")
    print("=" * 50)

    # Create test items with synthetic embeddings
    def synthetic_embedding(text: str, dim: int = 384) -> np.ndarray:
        """Generate deterministic embedding from text hash."""
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(dim)
        return vec / np.linalg.norm(vec)

    items = [
        ContextItem(
            item_id="msg1",
            content="What is catalytic computing?",
            tokens=100,
            embedding=synthetic_embedding("What is catalytic computing?"),
            item_type="message"
        ),
        ContextItem(
            item_id="doc1",
            content="Catalytic computing is a paradigm where large disk state restores exactly after use.",
            tokens=200,
            embedding=synthetic_embedding("Catalytic computing is a paradigm where large disk state restores exactly after use."),
            item_type="document"
        ),
        ContextItem(
            item_id="doc2",
            content="The weather today is sunny with clear skies.",
            tokens=150,
            embedding=synthetic_embedding("The weather today is sunny with clear skies."),
            item_type="document"
        ),
        ContextItem(
            item_id="doc3",
            content="Clean space bounded context uses pointers not full content.",
            tokens=180,
            embedding=synthetic_embedding("Clean space bounded context uses pointers not full content."),
            item_type="document"
        ),
    ]

    query = "How does catalytic computing manage context?"
    query_embedding = synthetic_embedding(query)

    partitioner = ContextPartitioner(threshold=0.5)
    result = partitioner.partition(
        query_embedding=query_embedding,
        all_items=items,
        budget_tokens=400,
        query_text=query
    )

    print(f"\nQuery: {query}")
    print(f"Budget: {result.budget_total} tokens")
    print(f"Threshold: {result.threshold}")
    print(f"\nPartition Results:")
    print(f"  Working Set: {result.items_in_working_set} items, {result.budget_used} tokens")
    print(f"  Pointer Set: {len(result.pointer_set)} items")
    print(f"  Below Threshold: {result.items_below_threshold}")
    print(f"  Over Budget: {result.items_over_budget}")
    print(f"\nE-Score Distribution:")
    print(f"  Mean: {result.E_mean:.4f}")
    print(f"  Min: {result.E_min:.4f}")
    print(f"  Max: {result.E_max:.4f}")
    print(f"  Std: {result.E_std:.4f}")

    print("\nWorking Set Items:")
    for s in result.working_set:
        print(f"  [{s.rank}] {s.item.item_id}: E={s.E_score:.4f}, tokens={s.item.tokens}")

    print("\nPointer Set Items:")
    for s in result.pointer_set:
        print(f"  [{s.rank}] {s.item.item_id}: E={s.E_score:.4f}, tokens={s.item.tokens}")
