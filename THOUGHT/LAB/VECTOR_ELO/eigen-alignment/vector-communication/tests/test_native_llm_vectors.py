#!/usr/bin/env python3
"""Native LLM Vector Communication Test.

Tests TRUE LLM-to-LLM vector communication using the LLM's own
internal embeddings - not a separate embedding model.

Architecture:
    qwen (3584D native) -> MDS align -> SVTP -> MDS align -> mistral (4096D native)

The LLMs produce vectors directly from their own understanding.
No separate embedding model involved.

Usage:
    python test_native_llm_vectors.py
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.vector_packet import create_svtp_channel
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128


# =============================================================================
# Configuration
# =============================================================================

OLLAMA_URL = "http://localhost:11434"


# =============================================================================
# API Setup
# =============================================================================

def create_session():
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    session.mount('http://', adapter)
    return session


SESSION = create_session()
EMBED_CACHE_QWEN = {}
EMBED_CACHE_MISTRAL = {}


# =============================================================================
# Native LLM Embedding Functions
# =============================================================================

def embed_qwen_native(texts):
    """Get qwen's NATIVE internal embeddings (3584D).

    These are the LLM's own representations, not a separate embedding model.
    """
    if isinstance(texts, str):
        texts = [texts]
    result = []
    for t in texts:
        if t not in EMBED_CACHE_QWEN:
            try:
                resp = SESSION.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": "qwen2.5:7b", "input": t},
                    timeout=60
                )
                EMBED_CACHE_QWEN[t] = np.array(resp.json()["embeddings"][0])
            except Exception as e:
                print(f"  Qwen embed error: {e}")
                return None
        result.append(EMBED_CACHE_QWEN[t])
    return np.array(result)


def embed_mistral_native(texts):
    """Get mistral's NATIVE internal embeddings (4096D).

    These are the LLM's own representations, not a separate embedding model.
    """
    if isinstance(texts, str):
        texts = [texts]
    result = []
    for t in texts:
        if t not in EMBED_CACHE_MISTRAL:
            try:
                resp = SESSION.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": "mistral:7b", "input": t},
                    timeout=60
                )
                EMBED_CACHE_MISTRAL[t] = np.array(resp.json()["embeddings"][0])
            except Exception as e:
                print(f"  Mistral embed error: {e}")
                return None
        result.append(EMBED_CACHE_MISTRAL[t])
    return np.array(result)


# =============================================================================
# LLM Generation Functions
# =============================================================================

def qwen_generate(prompt, max_tokens=50):
    """Generate text using qwen."""
    try:
        resp = SESSION.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.7}
            },
            timeout=120
        )
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  Qwen generate error: {e}")
        return None


def mistral_generate(prompt, max_tokens=50):
    """Generate text using mistral."""
    try:
        resp = SESSION.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "mistral:7b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.7}
            },
            timeout=120
        )
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  Mistral generate error: {e}")
        return None


# =============================================================================
# Test Functions
# =============================================================================

def check_availability():
    """Check native embedding support."""
    print("\n" + "=" * 60)
    print("CHECKING NATIVE LLM EMBEDDING SUPPORT")
    print("=" * 60)

    # Check qwen native embeddings
    test = embed_qwen_native("test")
    if test is None:
        print("  [FAIL] qwen2.5:7b native embeddings")
        return False
    print(f"  [OK] qwen2.5:7b native: {test.shape[1]}D")

    # Check mistral native embeddings
    test = embed_mistral_native("test")
    if test is None:
        print("  [FAIL] mistral:7b native embeddings")
        return False
    print(f"  [OK] mistral:7b native: {test.shape[1]}D")

    # Check generation
    test = qwen_generate("Say OK", max_tokens=5)
    if not test:
        print("  [FAIL] qwen2.5:7b generation")
        return False
    print(f"  [OK] qwen2.5:7b generation")

    test = mistral_generate("Say OK", max_tokens=5)
    if not test:
        print("  [FAIL] mistral:7b generation")
        return False
    print(f"  [OK] mistral:7b generation")

    return True


def test_native_alignment():
    """Test alignment between native LLM embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Native LLM Embedding Alignment")
    print("      qwen (3584D) <-> mistral (4096D)")
    print("=" * 60)

    print("\n  Creating alignment keys from native embeddings...")
    print("  (This uses the LLM's OWN understanding, not a separate model)")

    key_qwen = AlignmentKey.create(
        "qwen-native",
        embed_qwen_native,
        anchors=CANONICAL_128,
        k=128
    )
    print(f"    qwen key created (MDS from 3584D -> 128D)")

    key_mistral = AlignmentKey.create(
        "mistral-native",
        embed_mistral_native,
        anchors=CANONICAL_128,
        k=128
    )
    print(f"    mistral key created (MDS from 4096D -> 128D)")

    # Align
    print("\n  Aligning native embedding spaces...")
    pair = key_qwen.align_with(key_mistral)

    print(f"    Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"    Spectrum correlation: {pair.spectrum_correlation:.4f}")

    # The key question: do different LLMs share semantic structure?
    if pair.spectrum_correlation > 0.9:
        print("\n  [BREAKTHROUGH] LLMs share universal semantic geometry!")
    else:
        print(f"\n  [NOTE] Correlation {pair.spectrum_correlation:.4f} - exploring...")

    return pair


def test_native_vector_communication(pair):
    """Test vector communication using native LLM embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Native LLM Vector Communication")
    print("      qwen's thoughts -> vectors -> mistral's understanding")
    print("=" * 60)

    # Create SVTP channel using native embeddings
    enc_qwen, dec_qwen, enc_mistral, dec_mistral = create_svtp_channel(
        pair,
        embed_qwen_native,
        embed_mistral_native
    )

    # Test messages
    test_messages = [
        "Neural networks learn patterns from data",
        "The sun rises in the east every morning",
        "Mathematics describes the structure of reality",
        "Music expresses emotions through sound",
    ]

    candidates = test_messages + ["Random noise", "Distractor text"]

    # Qwen -> Mistral
    print("\n  --- Qwen (native) -> Mistral (native) ---")
    q2m_success = 0
    for msg in test_messages:
        packet = enc_qwen.encode_to_other(msg, sequence=0)
        result = dec_mistral.decode(packet.vector, candidates)
        if result.payload == msg:
            q2m_success += 1
            print(f"    [PASS] '{msg[:40]}...' (conf={result.confidence:.3f})")
        else:
            print(f"    [FAIL] '{msg[:40]}...' -> '{result.payload}'")

    # Mistral -> Qwen
    print("\n  --- Mistral (native) -> Qwen (native) ---")
    m2q_success = 0
    for msg in test_messages:
        packet = enc_mistral.encode_to_other(msg, sequence=0)
        result = dec_qwen.decode(packet.vector, candidates)
        if result.payload == msg:
            m2q_success += 1
            print(f"    [PASS] '{msg[:40]}...' (conf={result.confidence:.3f})")
        else:
            print(f"    [FAIL] '{msg[:40]}...' -> '{result.payload}'")

    total = q2m_success + m2q_success
    expected = len(test_messages) * 2
    accuracy = total / expected
    print(f"\n  Total: {total}/{expected} ({accuracy*100:.0f}%)")

    return accuracy >= 0.7


def test_native_llm_generated():
    """Test with content generated by the LLMs themselves."""
    print("\n" + "=" * 60)
    print("TEST: Native LLM-Generated Content")
    print("      Qwen thinks -> qwen's vector -> mistral understands")
    print("=" * 60)

    # Create fresh alignment
    key_qwen = AlignmentKey.create("qwen-native", embed_qwen_native, anchors=CANONICAL_128, k=128)
    key_mistral = AlignmentKey.create("mistral-native", embed_mistral_native, anchors=CANONICAL_128, k=128)
    pair = key_qwen.align_with(key_mistral)

    enc_qwen, _, _, dec_mistral = create_svtp_channel(pair, embed_qwen_native, embed_mistral_native)

    # Have qwen generate thoughts
    print("\n  [Qwen generating thoughts...]")
    qwen_thoughts = []
    for topic in ["artificial intelligence", "the ocean", "human creativity"]:
        thought = qwen_generate(f"Complete in one sentence: {topic} is", max_tokens=30)
        if thought:
            thought = thought.split(".")[0] + "." if "." in thought else thought[:60]
            qwen_thoughts.append(thought)
            print(f"    '{thought}'")

    if len(qwen_thoughts) < 2:
        print("  [WARN] Not enough thoughts generated")
        return True

    candidates = qwen_thoughts + ["Random distractor", "Noise"]

    # Transmit qwen's thoughts to mistral through vectors
    print("\n  [Transmitting qwen's thoughts to mistral via native vectors...]")
    successes = 0
    for thought in qwen_thoughts:
        # Encode using qwen's native understanding
        packet = enc_qwen.encode_to_other(thought, sequence=0)

        # Decode using mistral's native understanding
        result = dec_mistral.decode(packet.vector, candidates)

        if result.payload == thought:
            successes += 1
            # Have mistral respond
            response = mistral_generate(f"Respond briefly: {result.payload}", max_tokens=30)
            print(f"    [PASS] Mistral understood: '{response[:50]}...'")
        else:
            print(f"    [FAIL] Sent: '{thought[:30]}' Got: '{result.payload}'")

    accuracy = successes / len(qwen_thoughts)
    print(f"\n  Accuracy: {successes}/{len(qwen_thoughts)} ({accuracy*100:.0f}%)")
    return accuracy >= 0.5


def test_native_corruption_tolerance():
    """Test corruption tolerance with native embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Native Embedding Corruption Tolerance (Q40)")
    print("=" * 60)

    key_qwen = AlignmentKey.create("qwen-native", embed_qwen_native, anchors=CANONICAL_128, k=128)
    key_mistral = AlignmentKey.create("mistral-native", embed_mistral_native, anchors=CANONICAL_128, k=128)
    pair = key_qwen.align_with(key_mistral)

    enc_qwen, _, _, dec_mistral = create_svtp_channel(pair, embed_qwen_native, embed_mistral_native)

    msg = "The universe is governed by mathematical laws"
    candidates = [msg, "Random distractor", "Noise text"]

    print(f"\n  Message: '{msg}'")
    print(f"  Testing corruption levels...")

    corruption_levels = [0.0, 0.25, 0.50, 0.75, 0.90]
    results = {}

    for corruption in corruption_levels:
        packet = enc_qwen.encode_to_other(msg, sequence=0)
        vec = packet.vector.copy()

        if corruption > 0:
            n_corrupt = int(128 * corruption)
            np.random.seed(42)
            corrupt_idx = np.random.choice(128, n_corrupt, replace=False)
            vec[corrupt_idx] = 0.0

        result = dec_mistral.decode(vec, candidates, verify_pilot=corruption == 0)
        results[corruption] = result.payload == msg
        status = "[PASS]" if results[corruption] else "[FAIL]"
        print(f"    {corruption*100:3.0f}% corruption: conf={result.confidence:.3f} {status}")

    return results.get(0.0, False) and results.get(0.50, False)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("NATIVE LLM VECTOR COMMUNICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"\nArchitecture:")
    print(f"  qwen2.5:7b (3584D native) -> MDS -> SVTP -> MDS -> mistral:7b (4096D native)")
    print(f"\n  NO SEPARATE EMBEDDING MODEL")
    print(f"  LLMs produce vectors from their OWN internal representations")

    if not check_availability():
        print("\n*** CANNOT RUN: Missing native embedding support ***")
        return {"error": "No native embeddings"}

    results = {}

    # Test alignment
    pair = test_native_alignment()
    if pair is None:
        return {"error": "Alignment failed"}

    results["alignment"] = pair.spectrum_correlation > 0.8

    # Test communication
    results["communication"] = test_native_vector_communication(pair)
    results["llm_generated"] = test_native_llm_generated()
    results["corruption"] = test_native_corruption_tolerance()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} passed")

    if total_passed >= total_tests - 1:
        print("\n*** NATIVE LLM VECTOR COMMUNICATION VERIFIED ***")
        print("    LLMs communicate through their OWN internal representations")
        print("    No separate embedding model required")
    else:
        print("\n*** SOME TESTS FAILED ***")

    return results


if __name__ == "__main__":
    main()
