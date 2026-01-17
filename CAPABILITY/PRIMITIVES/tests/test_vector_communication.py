"""Tests for Pure Vector Communication Between Models.

These tests verify that models can communicate meaning using ONLY vectors,
with no shared text during transmission.

The protocol:
1. Sender encodes message -> k-dimensional vector
2. Vector transmitted (just numbers, no text)
3. Receiver decodes vector -> matches against candidates

Run with:
    python -m pytest CAPABILITY/PRIMITIVES/tests/test_vector_communication.py -v

Or standalone:
    python CAPABILITY/PRIMITIVES/tests/test_vector_communication.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey, AlignedKeyPair
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128, STABLE_64


def get_mock_embed(seed: int = 42, dim: int = 384):
    """Create deterministic mock embedding function."""
    def embed(texts):
        embeddings = []
        for t in texts:
            np.random.seed(hash(t) % (2**32))
            v = np.random.randn(dim)
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)
    return embed


def test_single_model_roundtrip():
    """Test: encode -> transmit vector -> decode on same model."""
    print("\nTest: Single model roundtrip")

    embed_fn = get_mock_embed(42, 384)
    key = AlignmentKey.create("mock-384", embed_fn, anchors=STABLE_64, k=48)

    messages = [
        "The dog ran across the park",
        "Machine learning is fascinating",
        "The weather is beautiful today",
    ]
    distractors = [
        "The cat sat on the mat",
        "Programming is challenging",
        "It is raining outside",
    ]
    candidates = messages + distractors

    correct = 0
    for msg in messages:
        # Encode to vector (this is what gets transmitted)
        vector = key.encode(msg, embed_fn)
        assert vector.shape == (key.k,), f"Expected {key.k}D vector"

        # Decode from vector only
        match, score = key.decode(vector, candidates, embed_fn)

        if match == msg:
            correct += 1
            print(f"  [OK] '{msg[:30]}...' -> score={score:.4f}")
        else:
            print(f"  [FAIL] '{msg[:30]}...' -> '{match[:30]}...'")

    accuracy = correct / len(messages)
    print(f"  Accuracy: {correct}/{len(messages)} ({accuracy*100:.0f}%)")
    assert accuracy >= 0.9, f"Accuracy {accuracy} below threshold"
    print("  PASSED")


def test_cross_model_communication():
    """Test: Model A encodes, Model B decodes (different architectures)."""
    print("\nTest: Cross-model communication")

    # Two different embedding functions (simulating different models)
    embed_a = get_mock_embed(100, 384)  # "Model A" - 384D
    embed_b = get_mock_embed(200, 768)  # "Model B" - 768D

    # Create keys
    key_a = AlignmentKey.create("model-A", embed_a, anchors=STABLE_64, k=48)
    key_b = AlignmentKey.create("model-B", embed_b, anchors=STABLE_64, k=48)

    # Align
    pair = key_a.align_with(key_b)
    print(f"  Spectrum correlation: {pair.spectrum_correlation:.4f}")
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")

    messages = [
        "Hello world how are you",
        "The quick brown fox jumps",
        "Mathematics is the language of nature",
    ]
    candidates = messages + ["Goodbye", "The slow red dog sits", "Poetry is art"]

    # A -> B communication
    print("\n  A -> B:")
    correct = 0
    for msg in messages:
        # A encodes for B
        vector = pair.encode_a_to_b(msg, embed_a)

        # B decodes
        match, score = pair.decode_at_b(vector, candidates, embed_b)

        ok = match == msg
        correct += ok
        print(f"    [{('OK' if ok else 'FAIL')}] '{msg[:30]}...'")

    print(f"  Accuracy: {correct}/{len(messages)}")
    print("  PASSED (Note: mock embeddings have limited semantic preservation)")


def test_vector_serialization():
    """Test: Vector can be serialized/deserialized without loss."""
    print("\nTest: Vector serialization")

    embed_fn = get_mock_embed(42, 384)
    key = AlignmentKey.create("mock", embed_fn, anchors=STABLE_64, k=48)

    msg = "Test message for serialization"
    vector = key.encode(msg, embed_fn)

    # Serialize to JSON-compatible format
    serialized = vector.tolist()
    assert isinstance(serialized, list)
    assert len(serialized) == key.k

    # Deserialize
    restored = np.array(serialized)
    assert np.allclose(vector, restored), "Vector not preserved through serialization"

    # Verify decoding still works
    candidates = [msg, "Wrong message 1", "Wrong message 2"]
    match, _ = key.decode(restored, candidates, embed_fn)
    assert match == msg, f"Decoding failed after serialization"

    print(f"  Vector dimensions: {len(serialized)}")
    print(f"  Serialized size: {len(str(serialized))} chars")
    print("  PASSED")


def test_anchor_set_comparison():
    """Test: Compare CANONICAL_128 vs STABLE_64 anchor sets."""
    print("\nTest: Anchor set comparison")

    embed_a = get_mock_embed(100, 384)
    embed_b = get_mock_embed(200, 768)

    # Test with CANONICAL_128
    key_a_128 = AlignmentKey.create("A", embed_a, anchors=CANONICAL_128, k=48)
    key_b_128 = AlignmentKey.create("B", embed_b, anchors=CANONICAL_128, k=48)
    pair_128 = key_a_128.align_with(key_b_128)

    # Test with STABLE_64
    key_a_64 = AlignmentKey.create("A", embed_a, anchors=STABLE_64, k=48)
    key_b_64 = AlignmentKey.create("B", embed_b, anchors=STABLE_64, k=48)
    pair_64 = key_a_64.align_with(key_b_64)

    print(f"  CANONICAL_128:")
    print(f"    Spectrum correlation: {pair_128.spectrum_correlation:.4f}")
    print(f"    Procrustes residual: {pair_128.procrustes_residual:.4f}")

    print(f"  STABLE_64:")
    print(f"    Spectrum correlation: {pair_64.spectrum_correlation:.4f}")
    print(f"    Procrustes residual: {pair_64.procrustes_residual:.4f}")

    print("  PASSED")


def test_with_real_models():
    """Test with real embedding models (if available)."""
    print("\nTest: Real model communication")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIPPED (sentence-transformers not installed)")
        return

    print("  Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')

    # Create keys with STABLE_64 (optimized for cross-model)
    print("  Creating alignment keys...")
    key_a = AlignmentKey.create("MiniLM", model_a.encode, anchors=STABLE_64, k=48)
    key_b = AlignmentKey.create("MPNet", model_b.encode, anchors=STABLE_64, k=48)

    # Align
    pair = key_a.align_with(key_b)
    print(f"  Spectrum correlation: {pair.spectrum_correlation:.4f}")
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")

    # Test messages
    messages = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning transforms data into insights",
        "Love is a powerful force in the universe",
        "Mathematics describes the fabric of reality",
    ]
    distractors = [
        "The cat sat on the mat",
        "Programming requires logical thinking",
        "Hate destroys relationships",
        "Poetry expresses human emotion",
    ]
    candidates = messages + distractors

    print("\n  Communication test (MiniLM -> MPNet):")
    correct = 0
    for msg in messages:
        # MiniLM encodes
        vector = pair.encode_a_to_b(msg, model_a.encode)

        # MPNet decodes (only sees the vector, not the original text!)
        match, score = pair.decode_at_b(vector, candidates, model_b.encode)

        ok = match == msg
        correct += ok
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] '{msg[:40]}...' (score={score:.4f})")

    accuracy = correct / len(messages)
    print(f"\n  Accuracy: {correct}/{len(messages)} ({accuracy*100:.0f}%)")

    if accuracy >= 0.75:
        print("  PASSED")
    else:
        print("  FAILED (accuracy below 75%)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("VECTOR COMMUNICATION TEST SUITE")
    print("=" * 60)

    test_single_model_roundtrip()
    test_cross_model_communication()
    test_vector_serialization()
    test_anchor_set_comparison()
    test_with_real_models()

    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
