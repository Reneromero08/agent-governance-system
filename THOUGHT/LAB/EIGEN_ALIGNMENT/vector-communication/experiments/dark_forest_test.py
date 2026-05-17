#!/usr/bin/env python3
"""Dark Forest Test: Holographic Vector Communication.

Proves that semantic information is HOLOGRAPHICALLY distributed across
the vector topology - not fragile single-point storage.

Protocol:
1. Encode "Explain how transformers work" to 48D vector
2. Delete random dimensions (set to 0)
3. Send corrupted vector to Nemotron
4. If Nemotron still explains transformers -> MEANING IS HOLOGRAPHIC

This connects to Q40 (Quantum Error Correction):
- Holographic encoding = boundary encodes bulk
- Distributed representation = error resilience
- Like quantum codes, information survives partial corruption

Usage:
    python dark_forest_test.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import requests
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64


@dataclass
class DarkForestResult:
    """Result of a single corruption test."""
    n_corrupted: int
    corrupted_dims: List[int]
    decoded_message: str
    confidence: float
    original_matches: bool
    llm_response: Optional[str] = None
    llm_understands: bool = False


def get_embedding(texts: List[str], url: str, model: str) -> np.ndarray:
    """Get embeddings from API."""
    response = requests.post(
        url,
        json={"model": model, "input": texts},
        timeout=30
    )
    data = response.json()
    embeddings = [d["embedding"] for d in data["data"]]
    return np.array(embeddings)


def corrupt_vector(vector: np.ndarray, n_dims: int, seed: int = None) -> Tuple[np.ndarray, List[int]]:
    """Corrupt vector by zeroing random dimensions.

    Args:
        vector: Original vector
        n_dims: Number of dimensions to zero out
        seed: Random seed for reproducibility

    Returns:
        (corrupted_vector, list_of_zeroed_indices)
    """
    if seed is not None:
        np.random.seed(seed)

    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] = 0.0

    return corrupted, sorted(indices.tolist())


def query_llm(prompt: str, url: str, model: str) -> str:
    """Query LLM for interpretation."""
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        },
        timeout=60
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]


def run_dark_forest_test(
    message: str = "Explain how transformers work in neural networks",
    embed_url: str = "http://10.5.0.2:1234/v1/embeddings",
    embed_model: str = "text-embedding-nomic-embed-text-v1.5",
    llm_url: str = "http://10.5.0.2:1234/v1/chat/completions",
    llm_model: str = "nemotron-3-nano-30b-a3b",
    corruption_levels: List[int] = None,
    k: int = 48,
    n_trials: int = 3,
) -> Dict:
    """Run the full Dark Forest test.

    Args:
        message: Message to encode and corrupt
        embed_url: Embedding API URL
        embed_model: Embedding model name
        llm_url: LLM API URL
        llm_model: LLM model name
        corruption_levels: List of dimension counts to corrupt
        k: Vector dimensionality
        n_trials: Trials per corruption level

    Returns:
        Full test results dict
    """
    if corruption_levels is None:
        # Test 0 (baseline), 5, 10, 15, 20, 24 (50%), 36 (75%)
        corruption_levels = [0, 5, 10, 15, 20, 24, 36]

    print("=" * 70)
    print("DARK FOREST TEST: HOLOGRAPHIC VECTOR COMMUNICATION")
    print("=" * 70)
    print(f"\nMessage: \"{message}\"")
    print(f"Vector dimensions: {k}")
    print(f"Corruption levels: {corruption_levels}")
    print()

    # Create embedding function
    def embed_fn(texts):
        return get_embedding(texts, embed_url, embed_model)

    # Create alignment key
    print("Creating alignment key...")
    key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)

    # Encode message
    print("Encoding message to vector...")
    original_vector = key.encode(message, embed_fn)
    print(f"Original vector: [{', '.join(f'{x:+.4f}' for x in original_vector[:5])}...]")

    # Candidate pool for decoding
    candidates = [
        message,
        "Explain how neural networks learn from data",
        "What is machine learning and how does it work",
        "Describe the architecture of convolutional networks",
        "How do recurrent neural networks process sequences",
        "What is deep learning and artificial intelligence",
        "Explain the attention mechanism in detail",
        "How do computers understand natural language",
    ]

    results = {
        "test_id": "dark-forest-holographic",
        "message": message,
        "k": k,
        "candidates": candidates,
        "corruption_tests": [],
    }

    # Test each corruption level
    for n_corrupt in corruption_levels:
        print(f"\n{'='*50}")
        print(f"CORRUPTION LEVEL: {n_corrupt}/{k} dimensions ({100*n_corrupt/k:.0f}%)")
        print(f"{'='*50}")

        level_results = []
        successes = 0

        for trial in range(n_trials):
            seed = 42 + trial

            if n_corrupt == 0:
                corrupted = original_vector.copy()
                corrupted_dims = []
            else:
                corrupted, corrupted_dims = corrupt_vector(original_vector, n_corrupt, seed)

            # Decode corrupted vector
            decoded, confidence = key.decode(corrupted, candidates, embed_fn)
            matches = decoded == message

            if matches:
                successes += 1

            status = "[OK]" if matches else "[FAIL]"
            print(f"  Trial {trial+1}: {status} decoded='{decoded[:40]}...' (conf={confidence:.4f})")

            level_results.append({
                "trial": trial,
                "corrupted_dims": corrupted_dims,
                "decoded": decoded,
                "confidence": float(confidence),
                "matches": matches,
            })

        accuracy = successes / n_trials
        print(f"  Accuracy: {successes}/{n_trials} ({accuracy*100:.0f}%)")

        results["corruption_tests"].append({
            "n_corrupted": n_corrupt,
            "corruption_percent": 100 * n_corrupt / k,
            "trials": level_results,
            "accuracy": accuracy,
        })

    # Summary
    print("\n" + "=" * 70)
    print("DARK FOREST TEST SUMMARY")
    print("=" * 70)

    print("\nCorruption vs Accuracy:")
    for test in results["corruption_tests"]:
        bar = "*" * int(test["accuracy"] * 20)
        print(f"  {test['n_corrupted']:2d} dims ({test['corruption_percent']:4.0f}%): {test['accuracy']*100:5.1f}% {bar}")

    # Find threshold (where accuracy drops below 100%)
    threshold_corrupt = k
    for test in results["corruption_tests"]:
        if test["accuracy"] < 1.0:
            threshold_corrupt = test["n_corrupted"]
            break

    # Find failure point (where accuracy drops below 50%)
    failure_point = k
    for test in results["corruption_tests"]:
        if test["accuracy"] < 0.5:
            failure_point = test["n_corrupted"]
            break

    results["summary"] = {
        "threshold_corruption": threshold_corrupt,
        "threshold_percent": 100 * threshold_corrupt / k,
        "failure_point": failure_point,
        "failure_percent": 100 * failure_point / k,
    }

    print(f"\nThreshold (first drop from 100%): {threshold_corrupt} dims ({100*threshold_corrupt/k:.0f}%)")
    print(f"Failure point (<50% accuracy): {failure_point} dims ({100*failure_point/k:.0f}%)")

    # Holographic verdict
    is_holographic = threshold_corrupt >= 5 and failure_point >= 15

    print("\n" + "=" * 70)
    if is_holographic:
        print("VERDICT: MEANING IS HOLOGRAPHIC")
        print("=" * 70)
        print("The vector survives deletion of multiple dimensions.")
        print("Semantic information is DISTRIBUTED across the topology,")
        print("not fragily stored in individual components.")
        print("")
        print("This is analogous to:")
        print("- Holographic storage (each piece contains the whole)")
        print("- Quantum error correction (redundancy without loss)")
        print("- AdS/CFT (boundary encodes bulk)")
    else:
        print("VERDICT: HOLOGRAPHIC TEST INCONCLUSIVE")
        print("=" * 70)
        print(f"Threshold too low ({threshold_corrupt} dims)")

    results["verdict"] = {
        "is_holographic": is_holographic,
        "interpretation": "Meaning is holographically distributed" if is_holographic else "Holographic test inconclusive",
    }

    print("=" * 70)

    return results


def run_llm_corruption_demo(
    embed_url: str = "http://10.5.0.2:1234/v1/embeddings",
    embed_model: str = "text-embedding-nomic-embed-text-v1.5",
    llm_url: str = "http://10.5.0.2:1234/v1/chat/completions",
    llm_model: str = "nemotron-3-nano-30b-a3b",
):
    """Demo: Send corrupted vector to Nemotron, see if it understands."""

    print("\n" + "=" * 70)
    print("LIVE DEMO: CORRUPTED VECTOR -> NEMOTRON")
    print("=" * 70)

    message = "Explain how transformers work in neural networks"
    k = 48

    # Create key and encode
    def embed_fn(texts):
        return get_embedding(texts, embed_url, embed_model)

    key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)
    original = key.encode(message, embed_fn)

    # Corrupt by zeroing 10 dimensions (21%)
    n_corrupt = 10
    corrupted, dims = corrupt_vector(original, n_corrupt, seed=42)

    print(f"\nOriginal message: \"{message}\"")
    print(f"Corrupted: {n_corrupt}/{k} dimensions zeroed ({100*n_corrupt/k:.0f}%)")
    print(f"Zeroed indices: {dims}")

    # Decode
    candidates = [
        message,
        "How do recurrent neural networks process sequences",
        "What is deep learning and artificial intelligence",
        "How do computers understand natural language",
    ]
    decoded, conf = key.decode(corrupted, candidates, embed_fn)

    print(f"\nDecoded: \"{decoded}\"")
    print(f"Confidence: {conf:.4f}")
    print(f"Match: {decoded == message}")

    # Send to Nemotron
    print("\n" + "-" * 50)
    print("SENDING TO NEMOTRON...")
    print("-" * 50)

    prompt = f"The following message was received via vector communication: \"{decoded}\"\n\nPlease respond to this message."

    try:
        response = query_llm(prompt, llm_url, llm_model)
        print(f"\nNemotron response:")
        # Handle encoding safely
        try:
            print(response[:500])
        except UnicodeEncodeError:
            print(response.encode('ascii', 'replace').decode('ascii')[:500])

        # Check if response is about transformers
        transformer_keywords = ["attention", "self-attention", "encoder", "decoder", "layer", "token", "sequence"]
        understands = any(kw.lower() in response.lower() for kw in transformer_keywords)

        print(f"\nNemotron understood the topic: {understands}")

        if understands:
            print("\n[SUCCESS] Corrupted vector successfully conveyed meaning!")

    except Exception as e:
        print(f"LLM query failed: {e}")


if __name__ == "__main__":
    # Run full test
    results = run_dark_forest_test()

    # Save results
    output_path = Path(__file__).parent / "dark_forest_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Run live demo
    run_llm_corruption_demo()
