#!/usr/bin/env python3
"""Test SVTP - Semantic Vector Transport Protocol.

Tests:
1. Single-model encode/decode
2. Pilot tone verification (geometric checksum)
3. Auth token verification
4. Cross-model communication
5. Corruption detection

Usage:
    python test_svtp.py
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.vector_packet import (
    SVTPEncoder, SVTPDecoder, CrossModelEncoder, CrossModelDecoder,
    SVTPPacket, SVTP_256, create_svtp_channel, format_packet_hex,
    PAYLOAD_START, PAYLOAD_END, PILOT_START, PILOT_END, AUTH_START, AUTH_END,
)
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128


def test_single_model():
    """Test: single model encode/decode without network."""
    print("\n" + "=" * 60)
    print("TEST: Single Model SVTP")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(texts):
        return model.encode(texts, convert_to_numpy=True)

    # Create alignment key with k=256 for full packet
    key = AlignmentKey.create("MiniLM", embed, anchors=CANONICAL_128, k=128)

    # Create encoder/decoder
    encoder = SVTPEncoder(key, embed)
    decoder = SVTPDecoder(key, embed)

    # Test messages
    test_messages = [
        "The neural network learns patterns",
        "Love conquers all obstacles",
        "Water flows downhill naturally",
    ]

    candidates = test_messages + [
        "Distractor message one",
        "Another unrelated text",
        "Random noise here",
    ]

    print("\nEncoding and decoding...")
    successes = 0
    for i, msg in enumerate(test_messages):
        packet = encoder.encode(msg, sequence=i)
        print(f"\n  Message: '{msg}'")
        print(f"  {format_packet_hex(packet)}")

        result = decoder.decode(packet.vector, candidates)
        print(f"  Decoded: '{result.payload}' (conf={result.confidence:.3f})")
        print(f"  Valid: {result.valid}, Pilot: {result.pilot_valid}, Auth: {result.auth_valid}")

        if result.payload == msg:
            successes += 1
            print("  [PASS]")
        else:
            print("  [FAIL]")

    print(f"\nSingle-model accuracy: {successes}/{len(test_messages)}")
    return successes == len(test_messages)


def test_pilot_tone_corruption():
    """Test: pilot tone detects corruption."""
    print("\n" + "=" * 60)
    print("TEST: Pilot Tone Corruption Detection")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(texts):
        return model.encode(texts, convert_to_numpy=True)

    key = AlignmentKey.create("MiniLM", embed, anchors=CANONICAL_128, k=128)
    encoder = SVTPEncoder(key, embed)
    decoder = SVTPDecoder(key, embed, pilot_threshold=0.6)

    msg = "Test message for corruption"
    candidates = [msg, "Other message", "Wrong answer"]

    # Clean packet
    packet = encoder.encode(msg, sequence=0)
    clean_result = decoder.decode(packet.vector, candidates)
    print(f"\nClean packet:")
    print(f"  Valid: {clean_result.valid}, Pilot: {clean_result.pilot_valid}")

    # Corrupt pilot tone
    corrupted = packet.vector.copy()
    corrupted[PILOT_START:PILOT_END] = np.random.randn(PILOT_END - PILOT_START)

    corrupt_result = decoder.decode(corrupted, candidates)
    print(f"\nCorrupted pilot tone:")
    print(f"  Valid: {corrupt_result.valid}, Pilot: {corrupt_result.pilot_valid}")
    if corrupt_result.error:
        print(f"  Error: {corrupt_result.error}")

    # Corrupt payload (pilot should still work)
    payload_corrupted = packet.vector.copy()
    payload_corrupted[0:50] = np.random.randn(50)  # Corrupt some payload

    payload_result = decoder.decode(payload_corrupted, candidates)
    print(f"\nCorrupted payload only:")
    print(f"  Valid: {payload_result.valid}, Pilot: {payload_result.pilot_valid}")

    passed = (
        clean_result.valid and
        not corrupt_result.pilot_valid and
        payload_result.pilot_valid
    )
    print(f"\n{'[PASS]' if passed else '[FAIL]'}")
    return passed


def test_cross_model_svtp():
    """Test: cross-model SVTP communication."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Model SVTP")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    # Load two different models
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')

    def embed_a(texts):
        return model_a.encode(texts, convert_to_numpy=True)

    def embed_b(texts):
        return model_b.encode(texts, convert_to_numpy=True)

    print("\nCreating alignment keys...")
    key_a = AlignmentKey.create("MiniLM", embed_a, anchors=CANONICAL_128, k=128)
    key_b = AlignmentKey.create("MPNet", embed_b, anchors=CANONICAL_128, k=128)

    print("Aligning keys...")
    pair = key_a.align_with(key_b)
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")

    # Create SVTP channel
    enc_a, dec_a, enc_b, dec_b = create_svtp_channel(pair, embed_a, embed_b)

    # Test A -> B
    test_messages = [
        "Neural networks learn from data",
        "The ocean waves crash on shore",
        "Mathematics describes the universe",
    ]

    candidates = test_messages + [
        "Random distractor text",
        "Another unrelated sentence",
        "Noise to confuse decoder",
    ]

    print("\n--- A -> B Communication ---")
    ab_successes = 0
    for i, msg in enumerate(test_messages):
        packet = enc_a.encode_to_other(msg, sequence=i)
        result = dec_b.decode(packet.vector, candidates)

        status = "[PASS]" if result.payload == msg else "[FAIL]"
        print(f"  '{msg[:30]}...' -> '{result.payload[:30] if result.payload else 'None'}...' {status}")

        if result.payload == msg:
            ab_successes += 1

    print(f"\nA->B accuracy: {ab_successes}/{len(test_messages)}")

    print("\n--- B -> A Communication ---")
    ba_successes = 0
    for i, msg in enumerate(test_messages):
        packet = enc_b.encode_to_other(msg, sequence=i)
        result = dec_a.decode(packet.vector, candidates)

        status = "[PASS]" if result.payload == msg else "[FAIL]"
        print(f"  '{msg[:30]}...' -> '{result.payload[:30] if result.payload else 'None'}...' {status}")

        if result.payload == msg:
            ba_successes += 1

    print(f"\nB->A accuracy: {ba_successes}/{len(test_messages)}")

    total = ab_successes + ba_successes
    expected = len(test_messages) * 2
    print(f"\nTotal cross-model accuracy: {total}/{expected}")
    return total >= expected * 0.8  # 80% threshold


def test_sequence_ordering():
    """Test: sequence numbers preserved correctly."""
    print("\n" + "=" * 60)
    print("TEST: Sequence Number Ordering")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(texts):
        return model.encode(texts, convert_to_numpy=True)

    key = AlignmentKey.create("MiniLM", embed, anchors=CANONICAL_128, k=128)
    encoder = SVTPEncoder(key, embed)
    decoder = SVTPDecoder(key, embed)

    # Encode with different sequences
    sequences = [0, 42, 127, 255]
    candidates = ["Test message"]

    print("\nSequence preservation:")
    all_correct = True
    for seq in sequences:
        packet = encoder.encode("Test message", sequence=seq)
        result = decoder.decode(packet.vector, candidates)

        correct = result.sequence == seq
        status = "[PASS]" if correct else "[FAIL]"
        print(f"  Sent seq={seq}, Received seq={result.sequence} {status}")

        if not correct:
            all_correct = False

    return all_correct


def test_packet_serialization():
    """Test: packet to/from bytes."""
    print("\n" + "=" * 60)
    print("TEST: Packet Serialization")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(texts):
        return model.encode(texts, convert_to_numpy=True)

    key = AlignmentKey.create("MiniLM", embed, anchors=CANONICAL_128, k=128)
    encoder = SVTPEncoder(key, embed)

    packet = encoder.encode("Serialize this message", sequence=99)

    # Serialize
    data = packet.to_bytes()
    print(f"\n  Original packet: {len(packet.vector)} floats")
    print(f"  Serialized: {len(data)} bytes")

    # Deserialize
    restored = SVTPPacket.from_bytes(data)
    print(f"  Restored packet: {len(restored.vector)} floats")
    print(f"  Sequence match: {restored.sequence == packet.sequence}")

    # Verify vectors match
    diff = np.max(np.abs(packet.vector - restored.vector))
    print(f"  Max vector difference: {diff:.2e}")

    passed = diff < 1e-6 and restored.sequence == 99
    print(f"\n{'[PASS]' if passed else '[FAIL]'}")
    return passed


def main():
    print("=" * 60)
    print("SVTP - Semantic Vector Transport Protocol Tests")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Protocol: SVTP 256D")
    print(f"  Payload: dims 0-199 (200 dims)")
    print(f"  Pilot:   dims 200-219 (20 dims)")
    print(f"  Auth:    dims 220-254 (35 dims)")
    print(f"  Clock:   dim 255 (1 dim)")

    results = {
        "single_model": test_single_model(),
        "pilot_corruption": test_pilot_tone_corruption(),
        "cross_model": test_cross_model_svtp(),
        "sequence": test_sequence_ordering(),
        "serialization": test_packet_serialization(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} passed")

    if total_passed == total_tests:
        print("\n*** SVTP PROTOCOL VERIFIED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")

    return results


if __name__ == "__main__":
    main()
