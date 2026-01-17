#!/usr/bin/env python3
"""Quick Cross-Model Test - Verify STABLE_32 Improvement.

Simplified test focusing on the key comparison.
"""

import sys
from pathlib import Path
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64

# 32 most stable anchors
STABLE_32 = [
    "destroy", "effect", "animal", "fast", "art", "cold", "child", "walk",
    "stone", "think", "give", "space", "society", "glass", "touch", "air",
    "evening", "mountain", "book", "leader", "sad", "dog", "cat", "winter",
    "wood", "morning", "know", "fire", "car", "building", "person", "enemy",
]


def get_embeddings(texts, url="http://10.5.0.2:1234/v1/embeddings",
                   model="text-embedding-nomic-embed-text-v1.5"):
    response = requests.post(url, json={"model": model, "input": texts}, timeout=60)
    return np.array([d["embedding"] for d in response.json()["data"]])


def test_config(name, anchors, k, embed_a, embed_b, model_a_name, model_b_name):
    """Test a single configuration."""
    print(f"\n{name}: {len(anchors)} anchors, k={k}")

    key_a = AlignmentKey.create(model_a_name, embed_a, anchors=anchors, k=k)
    key_b = AlignmentKey.create(model_b_name, embed_b, anchors=anchors, k=k)
    pair = key_a.align_with(key_b)

    print(f"  Residual: {pair.procrustes_residual:.4f}")

    test_msg = "Explain how transformers work in neural networks"
    candidates = [
        test_msg,
        "Describe gradient descent optimization",
        "What is deep learning",
        "Love connects humanity",
    ]

    results = {}
    for pct in [0.0, 0.25, 0.50, 0.75]:
        n_corrupt = int(k * pct)
        successes = 0
        n_trials = 10

        for trial in range(n_trials):
            vec = pair.encode_a_to_b(test_msg, embed_a)
            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(len(vec), n_corrupt, replace=False)
                vec[indices] = 0.0
            match, _ = pair.decode_at_b(vec, candidates, embed_b)
            if match == test_msg:
                successes += 1

        results[f"{int(pct*100)}%"] = f"{successes*10}%"
        print(f"  {int(pct*100):2d}% corruption: {successes}/{n_trials}")

    return pair.procrustes_residual, results


def main():
    print("=" * 60)
    print("CROSS-MODEL QUICK TEST")
    print("=" * 60)

    # Load MiniLM
    print("\nLoading MiniLM...")
    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_nomic(texts):
        return get_embeddings(texts)

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    print("Testing nomic -> mini")

    # Test STABLE_64
    res1, acc1 = test_config("STABLE_64", STABLE_64, 48, embed_nomic, embed_mini, "nomic", "mini")

    # Test STABLE_32
    res2, acc2 = test_config("STABLE_32", STABLE_32, 31, embed_nomic, embed_mini, "nomic", "mini")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSTABLE_64 (64 anchors, k=48):")
    print(f"  Residual: {res1:.4f}")
    print(f"  Corruption tolerance: {acc1}")

    print(f"\nSTABLE_32 (32 anchors, k=31):")
    print(f"  Residual: {res2:.4f}")
    print(f"  Corruption tolerance: {acc2}")

    print(f"\nResidual reduction: {(res1-res2)/res1*100:.1f}%")
    print("\nRECOMMENDATION: Use STABLE_32 for cross-model communication")


if __name__ == "__main__":
    main()
