#!/usr/bin/env python3
"""Optimized Cross-Model Communication Test.

Now that we've identified the 32 most stable anchors, test the full
corruption tolerance matrix to find the optimal configuration.

Key insight: Cross-model residual comes from anchor-level geometric
differences between models. Using only stable anchors dramatically
reduces residual and improves corruption tolerance.

Usage:
    python cross_model_optimized.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64


# =============================================================================
# Optimized Anchor Sets (discovered via stability analysis)
# =============================================================================

# 32 most stable anchors across nomic, MiniLM, and MPNet
STABLE_32 = [
    "destroy", "effect", "animal", "fast", "art", "cold", "child", "walk",
    "stone", "think", "give", "space", "society", "glass", "touch", "air",
    "evening", "mountain", "book", "leader", "sad", "dog", "cat", "winter",
    "wood", "morning", "know", "fire", "car", "building", "person", "enemy",
]

# 24 most stable
STABLE_24 = STABLE_32[:24]

# 16 most stable
STABLE_16 = STABLE_32[:16]


# =============================================================================
# Connection Setup
# =============================================================================

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    return session


SESSION = create_session()
EMBED_CACHE = {}


def get_embeddings_cached(texts, url, model):
    """Get embeddings with caching."""
    uncached = [t for t in texts if t not in EMBED_CACHE]
    if uncached:
        response = SESSION.post(url, json={"model": model, "input": uncached}, timeout=60)
        data = response.json()
        for i, t in enumerate(uncached):
            EMBED_CACHE[t] = np.array(data["data"][i]["embedding"])
    return np.array([EMBED_CACHE[t] for t in texts])


# =============================================================================
# Test Messages
# =============================================================================

MESSAGES = [
    "Explain how transformers work in neural networks",
    "Describe gradient descent optimization in machine learning",
    "What is the attention mechanism in deep learning",
    "Love is a powerful force that connects all humanity",
    "Fear can paralyze us or motivate us to action",
    "Time flows like a river toward an unknown sea",
]

DISTRACTORS = [
    "The cat sat on the warm windowsill",
    "Computers process binary information rapidly",
    "Music has power to evoke emotions",
]


# =============================================================================
# Main Test
# =============================================================================

def run_cross_model_optimized():
    """Run optimized cross-model communication test."""
    print("=" * 70)
    print("CROSS-MODEL COMMUNICATION: OPTIMIZED")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Load models
    print("\nLoading models...")
    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')
    model_mpnet = SentenceTransformer('all-mpnet-base-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    def embed_mpnet(texts):
        return model_mpnet.encode(texts, convert_to_numpy=True)

    models = {
        "nomic": embed_nomic,
        "mini": embed_mini,
        "mpnet": embed_mpnet,
    }

    # Test different anchor sets
    anchor_sets = {
        "STABLE_64": STABLE_64,
        "STABLE_32": STABLE_32,
        "STABLE_24": STABLE_24,
        "STABLE_16": STABLE_16,
    }

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "by_anchor_set": {},
    }

    candidates = MESSAGES + DISTRACTORS

    for anchor_name, anchors in anchor_sets.items():
        print(f"\n{'='*60}")
        print(f"ANCHOR SET: {anchor_name} ({len(anchors)} anchors)")
        print(f"{'='*60}")

        # Maximum k is anchors - 1
        k = min(len(anchors) - 1, 32)
        print(f"Using k={k}")

        # Create keys for all models
        keys = {}
        for name, embed_fn in models.items():
            keys[name] = AlignmentKey.create(name, embed_fn, anchors=anchors, k=k)

        # Test all pairs
        anchor_results = {"pairs": {}}

        pairs = [("nomic", "mini"), ("nomic", "mpnet"), ("mini", "mpnet")]

        for name_a, name_b in pairs:
            print(f"\n--- {name_a} -> {name_b} ---")

            key_a = keys[name_a]
            key_b = keys[name_b]
            embed_a = models[name_a]
            embed_b = models[name_b]

            # Align
            pair = key_a.align_with(key_b)
            print(f"Spectrum correlation: {pair.spectrum_correlation:.4f}")
            print(f"Procrustes residual: {pair.procrustes_residual:.4f}")

            # Test corruption levels
            corruption_results = []

            for pct in [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75]:
                n_corrupt = int(k * pct)
                if n_corrupt >= k:
                    continue

                successes = 0
                total = 0
                confidences = []

                for msg in MESSAGES:
                    for trial in range(5):  # 5 trials per message
                        vec = pair.encode_a_to_b(msg, embed_a)

                        if n_corrupt > 0:
                            np.random.seed(trial)
                            indices = np.random.choice(len(vec), n_corrupt, replace=False)
                            vec[indices] = 0.0

                        match, conf = pair.decode_at_b(vec, candidates, embed_b)
                        confidences.append(conf)
                        total += 1

                        if match == msg:
                            successes += 1

                accuracy = successes / total
                corruption_results.append({
                    "corruption_pct": pct * 100,
                    "n_corrupt": n_corrupt,
                    "accuracy": accuracy,
                    "mean_confidence": float(np.mean(confidences)),
                })

                status = "OK" if accuracy >= 0.9 else f"{accuracy*100:.0f}%"
                print(f"  {pct*100:4.0f}% ({n_corrupt:2d}/{k}): {status} (conf={np.mean(confidences):.3f})")

            # Find thresholds
            threshold_100 = 0
            threshold_90 = 0
            threshold_50 = 0

            for r in corruption_results:
                if r["accuracy"] >= 1.0:
                    threshold_100 = r["corruption_pct"]
                if r["accuracy"] >= 0.9:
                    threshold_90 = r["corruption_pct"]
                if r["accuracy"] >= 0.5:
                    threshold_50 = r["corruption_pct"]

            anchor_results["pairs"][f"{name_a}->{name_b}"] = {
                "spectrum_correlation": pair.spectrum_correlation,
                "procrustes_residual": pair.procrustes_residual,
                "corruption_results": corruption_results,
                "threshold_100": threshold_100,
                "threshold_90": threshold_90,
                "threshold_50": threshold_50,
            }

            print(f"\n  Thresholds: 100%@{threshold_100:.0f}%, 90%@{threshold_90:.0f}%, 50%@{threshold_50:.0f}%")

        results["by_anchor_set"][anchor_name] = {
            "n_anchors": len(anchors),
            "k": k,
            **anchor_results,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ANCHOR SET COMPARISON")
    print("=" * 70)

    print(f"\n{'Anchor Set':<12} | {'Anchors':>7} | {'k':>3} | {'Residual':>8} | {'100%@':>6} | {'90%@':>6} | {'50%@':>6}")
    print("-" * 70)

    for anchor_name in anchor_sets.keys():
        r = results["by_anchor_set"][anchor_name]
        # Average across pairs
        residuals = [p["procrustes_residual"] for p in r["pairs"].values()]
        thresh_100 = [p["threshold_100"] for p in r["pairs"].values()]
        thresh_90 = [p["threshold_90"] for p in r["pairs"].values()]
        thresh_50 = [p["threshold_50"] for p in r["pairs"].values()]

        print(f"{anchor_name:<12} | {r['n_anchors']:>7} | {r['k']:>3} | {np.mean(residuals):>8.4f} | {np.mean(thresh_100):>5.0f}% | {np.mean(thresh_90):>5.0f}% | {np.mean(thresh_50):>5.0f}%")

    # Best configuration
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find best by 90% threshold
    best_name = None
    best_thresh = -1
    for anchor_name in anchor_sets.keys():
        r = results["by_anchor_set"][anchor_name]
        avg_90 = np.mean([p["threshold_90"] for p in r["pairs"].values()])
        if avg_90 > best_thresh:
            best_thresh = avg_90
            best_name = anchor_name

    print(f"\nBest anchor set for cross-model communication: {best_name}")
    print(f"  - Achieves 90% accuracy at {best_thresh:.0f}% corruption average")

    r = results["by_anchor_set"][best_name]
    print(f"  - Uses {r['n_anchors']} anchors, k={r['k']}")
    print(f"  - Average residual: {np.mean([p['procrustes_residual'] for p in r['pairs'].values()]):.4f}")

    results["recommendation"] = {
        "best_anchor_set": best_name,
        "reason": f"Best 90% accuracy threshold ({best_thresh:.0f}%)",
    }

    return results


if __name__ == "__main__":
    results = run_cross_model_optimized()

    # Save results
    output_path = Path(__file__).parent / "cross_model_optimized_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
