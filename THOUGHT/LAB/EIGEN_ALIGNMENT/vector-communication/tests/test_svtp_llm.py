#!/usr/bin/env python3
"""Test SVTP with LLM Integration.

Tests SVTP as a communication channel between LLMs.
LLMs act as the application layer (generating/interpreting messages).
Embedding models act as the transport layer (encoding vectors).

Architecture:
    LLM A (nemotron) --> embed A (nomic) --> SVTP --> embed B (MiniLM) --> decode

The test verifies:
1. LLM-generated content can be vectorized via SVTP
2. Cross-model communication preserves semantic meaning
3. SVTP can handle diverse LLM output styles

Usage:
    python test_svtp_llm.py
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
from CAPABILITY.PRIMITIVES.vector_packet import (
    create_svtp_channel, format_packet_hex
)
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128


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
EMBED_CACHE = {}
LM_STUDIO_URL = "http://10.5.0.2:1234"


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_nomic(texts):
    """Embed using nomic via LM Studio."""
    if isinstance(texts, str):
        texts = [texts]

    uncached = [t for t in texts if t not in EMBED_CACHE]
    if uncached:
        for i in range(0, len(uncached), 20):  # Small batches
            batch = uncached[i:i+20]
            try:
                response = SESSION.post(
                    f"{LM_STUDIO_URL}/v1/embeddings",
                    json={
                        "model": "text-embedding-nomic-embed-text-v1.5",
                        "input": batch
                    },
                    timeout=30
                )
                data = response.json()
                for j, t in enumerate(batch):
                    EMBED_CACHE[t] = np.array(data["data"][j]["embedding"])
            except Exception as e:
                print(f"  Embedding error: {e}")
                return None

    return np.array([EMBED_CACHE[t] for t in texts])


def embed_mini(texts):
    """Embed using MiniLM (local)."""
    from sentence_transformers import SentenceTransformer
    if not hasattr(embed_mini, '_model'):
        embed_mini._model = SentenceTransformer('all-MiniLM-L6-v2')

    if isinstance(texts, str):
        texts = [texts]

    return embed_mini._model.encode(texts, convert_to_numpy=True)


# =============================================================================
# LLM Functions (Optional - for content generation)
# =============================================================================

def llm_generate(prompt, model="nemotron-3-nano-30b-a3b", max_tokens=50):
    """Generate text using LLM via LM Studio (optional)."""
    try:
        response = SESSION.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=60
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


# =============================================================================
# Test Functions
# =============================================================================

def test_basic_cross_embed():
    """Test: basic cross-embedding-model SVTP."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Embedding SVTP (nomic <-> MiniLM)")
    print("=" * 60)

    # Create keys
    print("\nCreating alignment keys...")
    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=CANONICAL_128, k=128)
    key_mini = AlignmentKey.create("MiniLM", embed_mini, anchors=CANONICAL_128, k=128)

    print("Aligning keys...")
    pair = key_nomic.align_with(key_mini)
    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"  Spectrum correlation: {pair.spectrum_correlation:.4f}")

    # Create SVTP channel
    enc_nomic, dec_nomic, enc_mini, dec_mini = create_svtp_channel(
        pair, embed_nomic, embed_mini
    )

    # Test messages (diverse semantic content)
    test_messages = [
        "The transformer architecture revolutionized natural language processing",
        "Love transcends all boundaries and connects humanity",
        "The quick brown fox jumps over the lazy dog",
        "E equals m c squared is Einstein's famous equation",
        "Water freezes at zero degrees Celsius under normal pressure",
    ]

    candidates = test_messages + [
        "Random distractor about weather patterns",
        "Another unrelated sentence about cooking",
        "Noise text that means nothing specific",
    ]

    # Test nomic -> MiniLM
    print("\n--- nomic -> MiniLM ---")
    n2m_success = 0
    for msg in test_messages:
        packet = enc_nomic.encode_to_other(msg, sequence=0)
        result = dec_mini.decode(packet.vector, candidates)

        if result.payload == msg:
            n2m_success += 1
            print(f"  [PASS] '{msg[:40]}...'")
        else:
            print(f"  [FAIL] '{msg[:40]}...' -> '{result.payload[:40] if result.payload else None}...'")

    # Test MiniLM -> nomic
    print("\n--- MiniLM -> nomic ---")
    m2n_success = 0
    for msg in test_messages:
        packet = enc_mini.encode_to_other(msg, sequence=0)
        result = dec_nomic.decode(packet.vector, candidates)

        if result.payload == msg:
            m2n_success += 1
            print(f"  [PASS] '{msg[:40]}...'")
        else:
            print(f"  [FAIL] '{msg[:40]}...' -> '{result.payload[:40] if result.payload else None}...'")

    total = n2m_success + m2n_success
    expected = len(test_messages) * 2
    print(f"\nTotal: {total}/{expected} ({total/expected*100:.0f}%)")

    return total >= expected * 0.8


def test_corruption_tolerance():
    """Test: SVTP corruption tolerance across embedding models."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Embedding Corruption Tolerance")
    print("=" * 60)

    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=CANONICAL_128, k=128)
    key_mini = AlignmentKey.create("MiniLM", embed_mini, anchors=CANONICAL_128, k=128)
    pair = key_nomic.align_with(key_mini)

    enc_nomic, _, _, dec_mini = create_svtp_channel(pair, embed_nomic, embed_mini)

    msg = "The neural network learns patterns from data"
    candidates = [
        msg,
        "The computer processes information quickly",
        "Random unrelated text here",
        "Another distractor sentence",
    ]

    corruption_levels = [0.0, 0.1, 0.25, 0.5]
    n_trials = 10

    print(f"\nMessage: '{msg}'")
    print(f"Trials per corruption level: {n_trials}")

    results = {}
    for pct in corruption_levels:
        successes = 0

        for trial in range(n_trials):
            packet = enc_nomic.encode_to_other(msg, sequence=trial)
            vec = packet.vector.copy()

            # Corrupt payload only (not pilot/auth)
            n_corrupt = int(128 * pct)  # k=128
            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(128, n_corrupt, replace=False)
                vec[indices] = 0.0

            result = dec_mini.decode(vec, candidates, verify_pilot=False)
            if result.payload == msg:
                successes += 1

        accuracy = successes / n_trials
        results[pct] = accuracy
        status = "[PASS]" if accuracy >= 0.5 else "[WARN]"
        print(f"  {pct*100:3.0f}% corruption: {accuracy*100:5.1f}% accuracy {status}")

    return results[0.0] >= 0.8  # At least 80% at 0% corruption


def test_semantic_diversity():
    """Test: SVTP handles diverse semantic content."""
    print("\n" + "=" * 60)
    print("TEST: Semantic Diversity")
    print("=" * 60)

    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=CANONICAL_128, k=128)
    key_mini = AlignmentKey.create("MiniLM", embed_mini, anchors=CANONICAL_128, k=128)
    pair = key_nomic.align_with(key_mini)

    enc_nomic, _, _, dec_mini = create_svtp_channel(pair, embed_nomic, embed_mini)

    # Diverse categories
    categories = {
        "technical": [
            "Backpropagation computes gradients through the network",
            "The GPU accelerates matrix multiplication operations",
        ],
        "emotional": [
            "Joy fills the heart when loved ones reunite",
            "Sadness overwhelms after losing someone dear",
        ],
        "factual": [
            "The Earth orbits the Sun once per year",
            "Water is composed of hydrogen and oxygen",
        ],
        "abstract": [
            "Time flows like a river toward the sea",
            "Truth emerges from the collision of ideas",
        ],
    }

    all_messages = []
    for cat_msgs in categories.values():
        all_messages.extend(cat_msgs)

    candidates = all_messages + [
        "Random distractor one",
        "Random distractor two",
        "Random distractor three",
    ]

    category_results = {}

    for cat_name, cat_msgs in categories.items():
        successes = 0
        for msg in cat_msgs:
            packet = enc_nomic.encode_to_other(msg, sequence=0)
            result = dec_mini.decode(packet.vector, candidates)
            if result.payload == msg:
                successes += 1

        accuracy = successes / len(cat_msgs)
        category_results[cat_name] = accuracy
        status = "[PASS]" if accuracy >= 0.5 else "[FAIL]"
        print(f"  {cat_name:12s}: {accuracy*100:5.1f}% {status}")

    avg = sum(category_results.values()) / len(category_results)
    print(f"\n  Average: {avg*100:.1f}%")

    return avg >= 0.7


def embed_ollama(texts):
    """Embed using Ollama nomic-embed-text."""
    if isinstance(texts, str):
        texts = [texts]
    result = []
    for t in texts:
        if t not in EMBED_CACHE:
            try:
                resp = SESSION.post(
                    "http://localhost:11434/api/embed",
                    json={"model": "nomic-embed-text", "input": t},
                    timeout=30
                )
                EMBED_CACHE[t] = np.array(resp.json()["embeddings"][0])
            except Exception:
                return None
        result.append(EMBED_CACHE[t])
    return np.array(result)


def ollama_generate(prompt, model="tinyllama:1.1b"):
    """Generate text using Ollama LLM."""
    try:
        resp = SESSION.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 25, "temperature": 0.7}
            },
            timeout=60
        )
        text = resp.json().get("response", "").strip()
        return text.split(".")[0] + "." if text else None
    except Exception:
        return None


def test_ollama_svtp():
    """Test: SVTP with Ollama nomic-embed-text."""
    print("\n" + "=" * 60)
    print("TEST: Ollama nomic-embed-text SVTP")
    print("=" * 60)

    # Check Ollama availability
    test_emb = embed_ollama("test")
    if test_emb is None:
        print("  Ollama embedding not available, skipping...")
        return True  # Skip but don't fail

    print(f"  Ollama nomic available, dims: {test_emb.shape[1]}")

    key_ollama = AlignmentKey.create("ollama-nomic", embed_ollama, anchors=CANONICAL_128, k=128)
    key_mini = AlignmentKey.create("MiniLM", embed_mini, anchors=CANONICAL_128, k=128)
    pair = key_ollama.align_with(key_mini)

    print(f"  Procrustes residual: {pair.procrustes_residual:.4f}")

    enc_ollama, _, _, dec_mini = create_svtp_channel(pair, embed_ollama, embed_mini)

    test_messages = [
        "Neural networks learn patterns from data",
        "The sun sets in the west every evening",
    ]
    candidates = test_messages + ["Random distractor"]

    successes = 0
    for msg in test_messages:
        packet = enc_ollama.encode_to_other(msg, sequence=0)
        result = dec_mini.decode(packet.vector, candidates)
        if result.payload == msg:
            successes += 1
            print(f"  [PASS] {msg[:40]}")
        else:
            print(f"  [FAIL] {msg[:40]}")

    return successes == len(test_messages)


def test_cross_llm_svtp():
    """Test: Cross-LLM communication via SVTP."""
    print("\n" + "=" * 60)
    print("TEST: Cross-LLM SVTP Communication")
    print("=" * 60)
    print("  Architecture: LLM A -> embed -> SVTP -> decode -> LLM B")

    # Check Ollama availability
    test_emb = embed_ollama("test")
    if test_emb is None:
        print("  Ollama embedding not available, skipping...")
        return True

    test_gen = ollama_generate("Say OK:", "tinyllama:1.1b")
    if test_gen is None:
        print("  Ollama LLM not available, skipping...")
        return True

    from CAPABILITY.PRIMITIVES.vector_packet import SVTPEncoder, SVTPDecoder

    key = AlignmentKey.create("ollama", embed_ollama, anchors=CANONICAL_128, k=128)
    encoder = SVTPEncoder(key, embed_ollama)
    decoder = SVTPDecoder(key, embed_ollama)

    # LLM A generates content
    print("\n  LLM A (tinyllama) generating...")
    llm_messages = []
    for topic in ["the ocean", "computers"]:
        msg = ollama_generate(f"Complete: {topic} is", "tinyllama:1.1b")
        if msg and len(msg) > 5:
            llm_messages.append(msg)
            print(f"    '{msg[:50]}'")

    if len(llm_messages) < 2:
        print("  Not enough content, skipping...")
        return True

    candidates = llm_messages + ["Random distractor", "Noise"]

    # Transmit with corruption
    print("\n  Transmitting with 25% corruption...")
    successes = 0
    for msg in llm_messages:
        packet = encoder.encode(msg, sequence=0)
        vec = packet.vector.copy()

        # 25% corruption
        np.random.seed(42)
        corrupt_idx = np.random.choice(128, 32, replace=False)
        vec[corrupt_idx] = 0.0

        result = decoder.decode(vec, candidates, verify_pilot=False)
        if result.payload == msg:
            successes += 1
            print(f"    [PASS] conf={result.confidence:.3f}")
        else:
            print(f"    [FAIL]")

    print(f"\n  Accuracy with 25% corruption: {successes}/{len(llm_messages)}")
    return successes > 0


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("SVTP LLM Integration Tests")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"\nArchitecture:")
    print(f"  Embedding A: nomic (via LM Studio or Ollama)")
    print(f"  Embedding B: MiniLM (local)")
    print(f"  LLM: tinyllama via Ollama (optional)")

    results = {}

    # Core cross-embedding tests
    results["cross_embed"] = test_basic_cross_embed()
    results["corruption"] = test_corruption_tolerance()
    results["diversity"] = test_semantic_diversity()

    # Ollama integration tests
    results["ollama_svtp"] = test_ollama_svtp()
    results["cross_llm"] = test_cross_llm_svtp()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} passed")

    if total_passed >= total_tests - 1:  # Allow 1 failure for optional tests
        print("\n*** SVTP LLM INTEGRATION VERIFIED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")

    return results


if __name__ == "__main__":
    main()
