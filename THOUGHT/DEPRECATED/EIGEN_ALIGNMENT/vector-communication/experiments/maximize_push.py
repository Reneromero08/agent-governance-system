#!/usr/bin/env python3
"""Push for Maximum Cross-Model Accuracy.

Key finding from maximize_fast: More anchors + higher k = better!
ANCHOR_128 with k=64 achieved 90% at 50% corruption.

Now testing:
- ANCHOR_256, ANCHOR_512 with k=96, 128, 192
- Goal: Find configuration with 100% at 50% corruption

Usage:
    python maximize_push.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from large_anchor_generator import ANCHOR_256, ANCHOR_512, generate_anchor_set


# Setup
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    return session


SESSION = create_session()
EMBED_CACHE = {}


def get_embeddings_cached(texts, url, model):
    uncached = [t for t in texts if t not in EMBED_CACHE]
    if uncached:
        for i in range(0, len(uncached), 50):
            batch = uncached[i:i+50]
            print(f"    Embedding batch {i//50 + 1}/{(len(uncached)+49)//50}...", end="\r")
            response = SESSION.post(url, json={"model": model, "input": batch}, timeout=120)
            data = response.json()
            for j, t in enumerate(batch):
                EMBED_CACHE[t] = np.array(data["data"][j]["embedding"])
        print(" " * 50, end="\r")
    return np.array([EMBED_CACHE[t] for t in texts])


# Test data
TEST_MSG = "Explain how transformers work in neural networks"
CANDIDATES = [
    TEST_MSG,
    "Describe gradient descent optimization",
    "What is the attention mechanism",
    "Love connects all humanity",
    "The cat sat on the windowsill",
    "Computers process information",
    "Music evokes emotions",
    "Mountains stand tall",
]


def test_config(anchor_name, anchors, k, embed_a, embed_b, name_a, name_b, n_trials=30):
    """Test a single configuration."""
    if k >= len(anchors):
        return None

    print(f"  {anchor_name} k={k}...", end=" ", flush=True)

    try:
        key_a = AlignmentKey.create(name_a, embed_a, anchors=anchors, k=k)
        key_b = AlignmentKey.create(name_b, embed_b, anchors=anchors, k=k)
        pair = key_a.align_with(key_b)
    except Exception as e:
        print(f"Error: {e}")
        return None

    results = {"residual": pair.procrustes_residual}

    for pct in [0.0, 0.25, 0.50, 0.75, 0.90]:
        n_corrupt = int(k * pct)
        successes = 0

        for trial in range(n_trials):
            vec = pair.encode_a_to_b(TEST_MSG, embed_a)
            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(len(vec), n_corrupt, replace=False)
                vec[indices] = 0.0
            match, _ = pair.decode_at_b(vec, CANDIDATES, embed_b)
            if match == TEST_MSG:
                successes += 1

        results[f"{int(pct*100)}%"] = successes / n_trials

    print(f"50%={results['50%']*100:.0f}%, 75%={results['75%']*100:.0f}%")
    return results


def main():
    print("=" * 60)
    print("PUSH FOR MAXIMUM CROSS-MODEL ACCURACY")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    print("\nLoading models...")
    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    print("Models loaded.")

    # Large anchor sets
    ANCHOR_512_clean = ANCHOR_512[:512]
    ANCHOR_777 = generate_anchor_set(777)

    anchor_sets = {
        "ANCHOR_256": ANCHOR_256[:256],
        "ANCHOR_512": ANCHOR_512_clean,
        "ANCHOR_777": ANCHOR_777,
    }

    k_values = [64, 96, 128, 192, 256]

    all_results = []
    best_config = None
    best_50 = 0

    print("\n" + "=" * 60)
    print("TESTING nomic -> MiniLM (Large Scale)")
    print("=" * 60)

    for anchor_name, anchors in anchor_sets.items():
        print(f"\n{anchor_name} ({len(anchors)} anchors):")

        for k in k_values:
            if k >= len(anchors) - 1:
                continue

            result = test_config(
                anchor_name, anchors, k,
                embed_nomic, embed_mini, "nomic", "mini"
            )

            if result:
                exp = {
                    "anchors": anchor_name,
                    "n_anchors": len(anchors),
                    "k": k,
                    **result
                }
                all_results.append(exp)

                if result["50%"] > best_50:
                    best_50 = result["50%"]
                    best_config = exp

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Config':<15} {'k':>4} {'Res':>7} {'0%':>5} {'25%':>5} {'50%':>5} {'75%':>5} {'90%':>5}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: -x["50%"]):
        print(f"{r['anchors']:<15} {r['k']:>4} {r['residual']:>7.2f} "
              f"{r['0%']*100:>4.0f}% {r['25%']*100:>4.0f}% "
              f"{r['50%']*100:>4.0f}% {r['75%']*100:>4.0f}% {r['90%']*100:>4.0f}%")

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    if best_config:
        print(f"  Anchors: {best_config['anchors']} ({best_config['n_anchors']})")
        print(f"  k: {best_config['k']}")
        print(f"  Residual: {best_config['residual']:.4f}")
        print(f"  0% corruption: {best_config['0%']*100:.0f}%")
        print(f"  50% corruption: {best_config['50%']*100:.0f}%")
        print(f"  75% corruption: {best_config['75%']*100:.0f}%")
        print(f"  90% corruption: {best_config['90%']*100:.0f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if best_50 >= 0.95:
        print(f"\n  SUCCESS! Achieved {best_50*100:.0f}% at 50% corruption!")
        print("  Cross-model communication can match same-model with sufficient dimensionality.")
    elif best_50 >= 0.80:
        print(f"\n  GOOD! Achieved {best_50*100:.0f}% at 50% corruption.")
        print("  Further optimization possible with more anchors/dimensions.")
    else:
        print(f"\n  Current best: {best_50*100:.0f}% at 50% corruption.")
        print("  Fundamental limit may exist - need different approach.")

    return all_results, best_config


if __name__ == "__main__":
    results, best = main()

    # Save
    output_path = Path(__file__).parent / "maximize_push_results.json"
    with open(output_path, 'w') as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"\nResults saved to: {output_path}")
