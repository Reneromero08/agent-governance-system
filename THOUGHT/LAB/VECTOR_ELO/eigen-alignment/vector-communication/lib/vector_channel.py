"""Vector Channel: Cross-model communication via geometric alignment.

Two embedding models can communicate meaning through vectors by:
1. Sharing an anchor set (bootstrap)
2. Computing MDS projection into shared geometric space
3. Procrustes rotation to align orientations
4. Sending k-dimensional coordinates instead of full embeddings

The key insight: eigenvalue spectrum is INVARIANT across models,
so the geometric structure is universal. Only orientation differs.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import hashlib
import json

# Import from existing lib
from lib.mds import squared_distance_matrix, classical_mds, effective_rank
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity


@dataclass
class ChannelEndpoint:
    """One end of a vector communication channel."""
    model_id: str
    anchors: List[str]
    anchor_hash: str
    anchor_embeddings: np.ndarray  # (n_anchors, dim)
    D2: np.ndarray                  # (n_anchors, n_anchors) squared distances
    mds_coords: np.ndarray          # (n_anchors, k) MDS coordinates
    eigenvalues: np.ndarray         # (k,) eigenvalues
    eigenvectors: np.ndarray        # (n_anchors, k) eigenvectors
    k: int                          # dimensionality of shared space


@dataclass
class AlignedChannel:
    """A bidirectional channel between two models."""
    endpoint_a: ChannelEndpoint
    endpoint_b: ChannelEndpoint
    R_a_to_b: np.ndarray           # rotation from A's space to B's space
    R_b_to_a: np.ndarray           # rotation from B's space to A's space
    spectrum_correlation: float     # how well eigenvalues match
    procrustes_residual: float      # alignment quality


def compute_anchor_hash(anchors: List[str]) -> str:
    """Deterministic hash of anchor set."""
    canonical = "\n".join(sorted(anchors))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def create_endpoint(
    model_id: str,
    embed_fn,  # callable: List[str] -> np.ndarray
    anchors: List[str],
    k: int = 16
) -> ChannelEndpoint:
    """Create a channel endpoint for a model.

    Args:
        model_id: Identifier for the model
        embed_fn: Function that embeds list of strings to (n, dim) array
        anchors: List of anchor words/phrases
        k: Number of dimensions for MDS projection

    Returns:
        ChannelEndpoint ready for alignment
    """
    # Embed anchors
    embeddings = embed_fn(anchors)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Compute squared distance matrix
    D2 = squared_distance_matrix(embeddings)

    # Classical MDS
    mds_coords, eigenvalues, eigenvectors = classical_mds(D2, k=k)

    return ChannelEndpoint(
        model_id=model_id,
        anchors=anchors,
        anchor_hash=compute_anchor_hash(anchors),
        anchor_embeddings=embeddings,
        D2=D2,
        mds_coords=mds_coords,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        k=len(eigenvalues)
    )


def align_endpoints(
    endpoint_a: ChannelEndpoint,
    endpoint_b: ChannelEndpoint
) -> AlignedChannel:
    """Align two endpoints to create a communication channel.

    Args:
        endpoint_a: First endpoint
        endpoint_b: Second endpoint

    Returns:
        AlignedChannel with rotation matrices

    Raises:
        ValueError: If anchor sets don't match
    """
    # Verify anchor sets match
    if endpoint_a.anchor_hash != endpoint_b.anchor_hash:
        raise ValueError(
            f"Anchor hash mismatch: {endpoint_a.anchor_hash} vs {endpoint_b.anchor_hash}"
        )

    # Use minimum k
    k = min(endpoint_a.k, endpoint_b.k)

    # Compute spectrum correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(endpoint_a.eigenvalues[:k], endpoint_b.eigenvalues[:k])

    # Procrustes alignment: find rotation from A to B
    X_a = endpoint_a.mds_coords[:, :k]
    X_b = endpoint_b.mds_coords[:, :k]

    R_a_to_b, residual = procrustes_align(X_a, X_b)
    R_b_to_a = R_a_to_b.T  # orthogonal matrix, so inverse = transpose

    return AlignedChannel(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        R_a_to_b=R_a_to_b,
        R_b_to_a=R_b_to_a,
        spectrum_correlation=corr,
        procrustes_residual=residual
    )


def send(
    channel: AlignedChannel,
    text: str,
    embed_fn_sender,
    from_a: bool = True
) -> np.ndarray:
    """Send a message through the channel.

    Args:
        channel: The aligned channel
        text: Text to send
        embed_fn_sender: Embedding function of the sender
        from_a: True if sending from A to B, False if B to A

    Returns:
        k-dimensional vector in RECEIVER's coordinate system
    """
    # Get sender's endpoint
    sender = channel.endpoint_a if from_a else channel.endpoint_b
    R = channel.R_a_to_b if from_a else channel.R_b_to_a

    # Embed the message
    embedding = embed_fn_sender([text])[0]
    embedding = embedding / np.linalg.norm(embedding)

    # Compute squared distances to anchors
    d2 = 2 * (1 - sender.anchor_embeddings @ embedding)

    # Project via Gower's formula
    y = out_of_sample_mds(
        d2.reshape(1, -1),
        sender.D2,
        sender.eigenvectors,
        sender.eigenvalues
    )[0]

    # Rotate to receiver's space
    k = min(len(y), R.shape[0])
    y_rotated = y[:k] @ R[:k, :k]

    return y_rotated


def receive(
    channel: AlignedChannel,
    vector: np.ndarray,
    embed_fn_receiver,
    candidates: List[str],
    to_a: bool = False
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Receive a message and find the closest candidate.

    Args:
        channel: The aligned channel
        vector: k-dimensional vector in receiver's coordinate system
        embed_fn_receiver: Embedding function of the receiver
        candidates: List of candidate texts to match against
        to_a: True if receiving at A, False if receiving at B

    Returns:
        Tuple of (best_match, best_score, all_scores)
    """
    # Get receiver's endpoint
    receiver = channel.endpoint_a if to_a else channel.endpoint_b

    # Project all candidates into MDS space
    candidate_embeddings = embed_fn_receiver(candidates)
    candidate_embeddings = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )

    # Project each candidate
    scores = []
    for i, cand in enumerate(candidates):
        emb = candidate_embeddings[i]
        d2 = 2 * (1 - receiver.anchor_embeddings @ emb)

        y = out_of_sample_mds(
            d2.reshape(1, -1),
            receiver.D2,
            receiver.eigenvectors,
            receiver.eigenvalues
        )[0]

        # Cosine similarity in MDS space
        k = min(len(y), len(vector))
        sim = cosine_similarity(y[:k], vector[:k])
        scores.append((cand, sim))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[0][0], scores[0][1], scores


def test_channel():
    """Test the vector channel with real embedding models."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Run: pip install sentence-transformers")
        return None

    print("=" * 60)
    print("VECTOR CHANNEL TEST")
    print("=" * 60)

    # Load two different models
    print("\nLoading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')

    print(f"  Model A: all-MiniLM-L6-v2 (dim={model_a.get_sentence_embedding_dimension()})")
    print(f"  Model B: all-mpnet-base-v2 (dim={model_b.get_sentence_embedding_dimension()})")

    # Define anchor set
    anchors = [
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
        # Numbers/quantities
        "one", "many", "all", "none", "more", "less", "equal", "different",
        # Domains
        "science", "art", "music", "math", "language", "nature", "technology", "society",
        # Meta
        "question", "answer", "problem", "solution", "cause", "effect", "begin", "end"
    ]

    print(f"\nAnchor set: {len(anchors)} words")

    # Create endpoints
    print("\nCreating endpoints...")
    endpoint_a = create_endpoint(
        "MiniLM",
        lambda texts: model_a.encode(texts),
        anchors,
        k=16
    )
    endpoint_b = create_endpoint(
        "MPNet",
        lambda texts: model_b.encode(texts),
        anchors,
        k=16
    )

    print(f"  Endpoint A: k={endpoint_a.k}, effective_rank={effective_rank(endpoint_a.eigenvalues):.2f}")
    print(f"  Endpoint B: k={endpoint_b.k}, effective_rank={effective_rank(endpoint_b.eigenvalues):.2f}")

    # Align
    print("\nAligning endpoints...")
    channel = align_endpoints(endpoint_a, endpoint_b)

    print(f"  Spectrum correlation: {channel.spectrum_correlation:.4f}")
    print(f"  Procrustes residual: {channel.procrustes_residual:.4f}")

    # Test messages
    test_messages = [
        "The quick brown fox jumps over the lazy dog",
        "I love programming and building things",
        "The weather is cold and rainy today",
        "Mathematics is the language of the universe",
        "She walked slowly through the quiet forest",
    ]

    # Candidate pool for matching (includes the originals + distractors)
    candidates = test_messages + [
        "The cat sat on the mat",
        "I hate doing boring tasks",
        "The sun is hot and bright today",
        "Poetry is the soul of humanity",
        "He ran quickly across the busy street",
        "Cooking dinner for the family",
        "Reading a book by the fireplace",
        "The ocean waves crash on the shore",
    ]

    print("\n" + "=" * 60)
    print("COMMUNICATION TEST: Model A -> Model B")
    print("=" * 60)

    results = []
    for msg in test_messages:
        # Send from A
        vec = send(
            channel,
            msg,
            lambda texts: model_a.encode(texts),
            from_a=True
        )

        # Receive at B
        match, score, all_scores = receive(
            channel,
            vec,
            lambda texts: model_b.encode(texts),
            candidates,
            to_a=False
        )

        success = match == msg
        results.append(success)

        print(f"\nSent: \"{msg[:50]}...\"" if len(msg) > 50 else f"\nSent: \"{msg}\"")
        print(f"  Vector: [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, ...] (k={len(vec)})")
        print(f"  Received: \"{match[:50]}...\"" if len(match) > 50 else f"  Received: \"{match}\"")
        print(f"  Score: {score:.4f}")
        print(f"  Match: {'OK' if success else 'FAIL'}")

    # Summary
    accuracy = sum(results) / len(results)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy: {sum(results)}/{len(results)} ({accuracy*100:.1f}%)")
    print(f"  Spectrum correlation: {channel.spectrum_correlation:.4f}")
    print(f"  Vector dimension: {endpoint_a.k} (vs original {model_a.get_sentence_embedding_dimension()})")
    print(f"  Compression: {model_a.get_sentence_embedding_dimension() / endpoint_a.k:.1f}x")

    # Test reverse direction
    print("\n" + "=" * 60)
    print("COMMUNICATION TEST: Model B -> Model A")
    print("=" * 60)

    results_reverse = []
    for msg in test_messages:
        # Send from B
        vec = send(
            channel,
            msg,
            lambda texts: model_b.encode(texts),
            from_a=False
        )

        # Receive at A
        match, score, _ = receive(
            channel,
            vec,
            lambda texts: model_a.encode(texts),
            candidates,
            to_a=True
        )

        success = match == msg
        results_reverse.append(success)

        print(f"\nSent: \"{msg[:40]}...\"" if len(msg) > 40 else f"\nSent: \"{msg}\"")
        print(f"  Received: \"{match[:40]}...\"" if len(match) > 40 else f"  Received: \"{match}\"")
        print(f"  Score: {score:.4f}, Match: {'OK' if success else 'FAIL'}")

    accuracy_reverse = sum(results_reverse) / len(results_reverse)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  A -> B accuracy: {sum(results)}/{len(results)} ({accuracy*100:.1f}%)")
    print(f"  B -> A accuracy: {sum(results_reverse)}/{len(results_reverse)} ({accuracy_reverse*100:.1f}%)")
    print(f"  Bidirectional: {(accuracy + accuracy_reverse) / 2 * 100:.1f}%")
    print(f"  Compression: {model_a.get_sentence_embedding_dimension()}D -> {endpoint_a.k}D ({model_a.get_sentence_embedding_dimension() / endpoint_a.k:.1f}x)")

    return {
        "a_to_b_accuracy": accuracy,
        "b_to_a_accuracy": accuracy_reverse,
        "spectrum_correlation": channel.spectrum_correlation,
        "compression_ratio": model_a.get_sentence_embedding_dimension() / endpoint_a.k,
        "k": endpoint_a.k
    }


if __name__ == "__main__":
    test_channel()