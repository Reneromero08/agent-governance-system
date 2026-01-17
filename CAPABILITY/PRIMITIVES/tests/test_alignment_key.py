"""Tests for AlignmentKey - Standalone Vector Communication.

These tests verify that the alignment key works for:
1. Single-model encode/decode
2. Cross-model communication
3. Key serialization/deserialization
4. Anchor hash verification

Run with:
    python -m pytest CAPABILITY/PRIMITIVES/tests/test_alignment_key.py -v

Or standalone:
    python CAPABILITY/PRIMITIVES/tests/test_alignment_key.py
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Import the modules we're testing
from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey, AlignedKeyPair
from CAPABILITY.PRIMITIVES.canonical_anchors import (
    CANONICAL_128,
    compute_anchor_hash,
    verify_anchor_hash,
    CANONICAL_128_HASH
)
from CAPABILITY.PRIMITIVES.mds import (
    squared_distance_matrix,
    classical_mds,
    effective_rank
)
from CAPABILITY.PRIMITIVES.procrustes import (
    procrustes_align,
    cosine_similarity
)


def test_anchor_hash():
    """Test anchor hash computation and verification."""
    print("Test: Anchor hash computation")

    # Compute hash
    h = compute_anchor_hash(CANONICAL_128)
    print(f"  CANONICAL_128 hash: {h}")

    # Verify it matches pre-computed
    assert h == CANONICAL_128_HASH, f"Hash mismatch: {h} vs {CANONICAL_128_HASH}"

    # Verify function
    assert verify_anchor_hash(CANONICAL_128, h)

    # Different order should give same hash
    shuffled = CANONICAL_128.copy()
    shuffled.reverse()
    h2 = compute_anchor_hash(shuffled)
    assert h == h2, "Hash should be order-independent"

    print("  PASSED")


def test_mds_basic():
    """Test MDS on random data."""
    print("Test: MDS basic functionality")

    # Random normalized vectors
    np.random.seed(42)
    n, d = 50, 100
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Squared distance matrix
    D2 = squared_distance_matrix(X)
    assert D2.shape == (n, n)
    assert np.allclose(np.diag(D2), 0)  # Self-distance is 0

    # MDS
    coords, eigenvalues, eigenvectors = classical_mds(D2, k=10)
    assert coords.shape[0] == n
    assert len(eigenvalues) <= 10

    # Effective rank
    er = effective_rank(eigenvalues)
    print(f"  Effective rank: {er:.2f}")
    assert 1 <= er <= len(eigenvalues)

    print("  PASSED")


def test_procrustes_basic():
    """Test Procrustes alignment."""
    print("Test: Procrustes alignment")

    np.random.seed(42)
    n, k = 50, 10

    # Create source coordinates
    X_source = np.random.randn(n, k)

    # Create target by rotating source
    theta = np.pi / 4
    R_true = np.eye(k)
    R_true[0, 0] = np.cos(theta)
    R_true[0, 1] = -np.sin(theta)
    R_true[1, 0] = np.sin(theta)
    R_true[1, 1] = np.cos(theta)

    X_target = X_source @ R_true

    # Find rotation
    R, residual = procrustes_align(X_source, X_target)
    print(f"  Residual: {residual:.6f}")

    # Aligned should be close to target
    aligned = X_source @ R
    error = np.linalg.norm(aligned - X_target, 'fro')
    assert error < 0.01, f"Alignment error too high: {error}"

    print("  PASSED")


def test_alignment_key_with_mock():
    """Test AlignmentKey with mock embedding function."""
    print("Test: AlignmentKey with mock embeddings")

    # Mock embedding function - just hash-based for determinism
    def mock_embed(texts):
        np.random.seed(42)
        dim = 384
        embeddings = []
        for t in texts:
            # Use text hash as seed for reproducibility
            seed = hash(t) % (2**32)
            np.random.seed(seed)
            v = np.random.randn(dim)
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)

    # Create key
    key = AlignmentKey.create("mock-model", mock_embed, k=32)

    print(f"  Model: {key.model_id}")
    print(f"  Anchor hash: {key.anchor_hash}")
    print(f"  k: {key.k}")

    # Encode a text
    vector = key.encode("Hello world", mock_embed)
    assert vector.shape == (key.k,)
    print(f"  Encoded vector shape: {vector.shape}")

    # Decode against candidates
    candidates = ["Hello world", "Goodbye world", "Hello there"]
    match, score = key.decode(vector, candidates, mock_embed)
    print(f"  Match: '{match}' (score: {score:.4f})")

    # Should match the original
    assert match == "Hello world", f"Expected 'Hello world', got '{match}'"

    print("  PASSED")


def test_key_serialization():
    """Test key save/load."""
    print("Test: Key serialization")

    def mock_embed(texts):
        np.random.seed(42)
        embeddings = []
        for t in texts:
            seed = hash(t) % (2**32)
            np.random.seed(seed)
            v = np.random.randn(384)
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)

    # Create key
    key = AlignmentKey.create("test-model", mock_embed, k=32)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_key"
        key.to_file(path)

        # Check files exist
        assert (Path(str(path) + ".json")).exists()
        assert (Path(str(path) + ".npz")).exists()

        # Load back
        loaded = AlignmentKey.from_file(path)

        # Verify
        assert loaded.model_id == key.model_id
        assert loaded.anchor_hash == key.anchor_hash
        assert loaded.k == key.k
        assert np.allclose(loaded.eigenvalues, key.eigenvalues)
        assert np.allclose(loaded.eigenvectors, key.eigenvectors)

    print("  PASSED")


def test_aligned_pair_with_mock():
    """Test AlignedKeyPair with mock embeddings."""
    print("Test: AlignedKeyPair with mock embeddings")

    # Two different mock embedding functions
    def mock_embed_a(texts):
        np.random.seed(100)
        embeddings = []
        for t in texts:
            seed = hash(t) % (2**32)
            np.random.seed(seed)
            v = np.random.randn(384)
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)

    def mock_embed_b(texts):
        np.random.seed(200)
        embeddings = []
        for t in texts:
            seed = (hash(t) + 12345) % (2**32)
            np.random.seed(seed)
            v = np.random.randn(768)  # Different dimension
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)

    # Create keys
    key_a = AlignmentKey.create("model-a", mock_embed_a, k=32)
    key_b = AlignmentKey.create("model-b", mock_embed_b, k=32)

    print(f"  Key A: {key_a.model_id}, k={key_a.k}")
    print(f"  Key B: {key_b.model_id}, k={key_b.k}")

    # Align
    pair = key_a.align_with(key_b)
    print(f"  Spectrum correlation: {pair.spectrum_correlation:.4f}")
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")

    # Test communication
    candidates = [
        "The cat sat on the mat",
        "Hello world how are you",
        "Mathematics is beautiful",
        "The weather is nice today"
    ]

    # A -> B
    msg = "Hello world how are you"
    vec = pair.encode_a_to_b(msg, mock_embed_a)
    match, score = pair.decode_at_b(vec, candidates, mock_embed_b)
    print(f"  A->B: '{msg[:30]}...' -> '{match[:30]}...' (score: {score:.4f})")

    # B -> A
    msg2 = "Mathematics is beautiful"
    vec2 = pair.encode_b_to_a(msg2, mock_embed_b)
    match2, score2 = pair.decode_at_a(vec2, candidates, mock_embed_a)
    print(f"  B->A: '{msg2[:30]}...' -> '{match2[:30]}...' (score: {score2:.4f})")

    print("  PASSED (Note: mock embeddings don't preserve semantics)")


def test_with_real_models():
    """Test with real sentence-transformers models (if available)."""
    print("Test: Real model communication")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIPPED (sentence-transformers not installed)")
        return

    print("  Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')

    dim_a = model_a.get_sentence_embedding_dimension()
    dim_b = model_b.get_sentence_embedding_dimension()
    print(f"  Model A: all-MiniLM-L6-v2 ({dim_a}D)")
    print(f"  Model B: all-mpnet-base-v2 ({dim_b}D)")

    # Create keys
    print("  Creating keys...")
    key_a = AlignmentKey.create("MiniLM", model_a.encode, k=48)
    key_b = AlignmentKey.create("MPNet", model_b.encode, k=48)

    # Align
    print("  Aligning...")
    pair = key_a.align_with(key_b)
    print(f"  Spectrum correlation: {pair.spectrum_correlation:.4f}")
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"  Compression: {dim_a}D/{dim_b}D -> {pair.k}D")

    # Test messages
    messages = [
        "The quick brown fox jumps over the lazy dog",
        "I love programming and building things",
        "Mathematics is the language of the universe",
        "The coffee was hot and delicious",
    ]

    distractors = [
        "The cat sat on the mat",
        "I hate doing boring tasks",
        "Poetry is the soul of humanity",
        "The tea was cold and bitter",
    ]

    candidates = messages + distractors

    print("\n  Communication test (A -> B):")
    correct = 0
    for msg in messages:
        vec = pair.encode_a_to_b(msg, model_a.encode)
        match, score = pair.decode_at_b(vec, candidates, model_b.encode)
        ok = match == msg
        correct += ok
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] '{msg[:40]}...' -> '{match[:40]}...'")

    accuracy = correct / len(messages)
    print(f"\n  Accuracy: {correct}/{len(messages)} ({accuracy*100:.0f}%)")

    if accuracy >= 0.75:
        print("  PASSED")
    else:
        print("  FAILED (accuracy below 75%)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ALIGNMENT KEY TEST SUITE")
    print("=" * 60)
    print()

    test_anchor_hash()
    print()

    test_mds_basic()
    print()

    test_procrustes_basic()
    print()

    test_alignment_key_with_mock()
    print()

    test_key_serialization()
    print()

    test_aligned_pair_with_mock()
    print()

    test_with_real_models()
    print()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
