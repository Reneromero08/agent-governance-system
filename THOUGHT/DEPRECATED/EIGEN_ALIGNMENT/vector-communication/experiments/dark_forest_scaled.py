#!/usr/bin/env python3
"""Dark Forest SCALED Test: Holographic Vector Communication at 110%.

Optimized version with connection pooling and batch processing.

Usage:
    python dark_forest_scaled.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64


# =============================================================================
# Connection Setup
# =============================================================================

def create_session():
    """Create session with connection pooling."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    return session


SESSION = create_session()


# =============================================================================
# Test Messages
# =============================================================================

MESSAGES = [
    # Technical
    "Explain how transformers work in neural networks",
    "Describe gradient descent optimization in machine learning",
    "What is the attention mechanism in deep learning",
    # Emotional
    "Love is a powerful force that connects all humanity",
    "Fear can paralyze us or motivate us to action",
    "Joy comes from meaningful connections with others",
    # Abstract
    "Time flows like a river toward an unknown sea",
    "Truth emerges from the collision of perspectives",
    "Beauty exists in the eye of the beholder alone",
    # Concrete
    "The dog ran across the green park this morning",
    "Water boils at one hundred degrees Celsius",
    "The red car drove down the mountain road",
    # Scientific
    "Quantum mechanics describes subatomic particle behavior",
    "DNA carries genetic information in organisms",
    "Gravity bends spacetime around massive objects",
]

DISTRACTORS = [
    "The cat sat on the warm windowsill",
    "Computers process binary information rapidly",
    "Music has power to evoke emotions",
    "Mountains stand tall against the sky",
    "Time heals all wounds eventually",
]


# =============================================================================
# Embedding with Batch + Cache
# =============================================================================

EMBED_CACHE = {}


def get_embeddings_batch(texts, url, model):
    """Get embeddings with caching."""
    # Check cache
    uncached = [t for t in texts if t not in EMBED_CACHE]

    if uncached:
        # Batch API call
        response = SESSION.post(
            url,
            json={"model": model, "input": uncached},
            timeout=60
        )
        data = response.json()

        for i, t in enumerate(uncached):
            EMBED_CACHE[t] = np.array(data["data"][i]["embedding"])

    return np.array([EMBED_CACHE[t] for t in texts])


# =============================================================================
# Corruption Strategies
# =============================================================================

def corrupt_random(vector, n_dims, seed):
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] = 0.0
    return corrupted


def corrupt_magnitude_high(vector, n_dims, seed=None):
    """Worst case: delete highest magnitude dimensions."""
    corrupted = vector.copy()
    indices = np.argsort(np.abs(vector))[-n_dims:]
    corrupted[indices] = 0.0
    return corrupted


def corrupt_sign_flip(vector, n_dims, seed):
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] = -corrupted[indices]
    return corrupted


def corrupt_gaussian_noise(vector, n_dims, seed):
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] += np.random.randn(n_dims) * np.std(vector) * 2
    return corrupted


STRATEGIES = {
    "random_zero": corrupt_random,
    "magnitude_high": corrupt_magnitude_high,
    "sign_flip": corrupt_sign_flip,
    "gaussian_noise": corrupt_gaussian_noise,
}


# =============================================================================
# Main Test
# =============================================================================

def run_scaled_test(
    embed_url="http://10.5.0.2:1234/v1/embeddings",
    embed_model="text-embedding-nomic-embed-text-v1.5",
):
    print("=" * 80)
    print("DARK FOREST SCALED TEST - 110%")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Pre-cache all embeddings
    print("\nPre-caching all embeddings...")
    all_texts = MESSAGES + DISTRACTORS
    get_embeddings_batch(all_texts, embed_url, embed_model)
    print(f"  Cached {len(EMBED_CACHE)} embeddings")

    def embed_fn(texts):
        return get_embeddings_batch(texts, embed_url, embed_model)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "by_k": {},
        "by_strategy": {},
        "minimum_dimensions": {},
    }

    candidates = MESSAGES + DISTRACTORS

    # ==========================================================================
    # Test different k values
    # ==========================================================================

    for k in [16, 32, 48]:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        print(f"{'='*60}")

        key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)

        k_results = {"by_message": [], "by_strategy": {}}

        for strategy_name, strategy_fn in STRATEGIES.items():
            print(f"\n  Strategy: {strategy_name}")

            strategy_results = []

            for corruption_pct in [0.0, 0.50, 0.75, 0.90, 0.94, 0.96, 0.98]:
                n_corrupt = int(k * corruption_pct)
                if n_corrupt >= k:
                    n_corrupt = k - 1

                total_trials = 0
                successes = 0
                confidences = []

                for msg in MESSAGES:
                    original = key.encode(msg, embed_fn)

                    for trial in range(3):  # 3 trials per message
                        if n_corrupt == 0:
                            corrupted = original.copy()
                        else:
                            corrupted = strategy_fn(original, n_corrupt, trial)

                        decoded, conf = key.decode(corrupted, candidates, embed_fn)
                        total_trials += 1
                        confidences.append(conf)

                        if decoded == msg:
                            successes += 1

                accuracy = successes / total_trials
                strategy_results.append({
                    "corruption_pct": corruption_pct * 100,
                    "n_corrupt": n_corrupt,
                    "accuracy": accuracy,
                    "mean_confidence": float(np.mean(confidences)),
                })

                status = "OK" if accuracy == 1.0 else f"{accuracy*100:.0f}%"
                print(f"    {corruption_pct*100:4.0f}% ({n_corrupt:2d}/{k} dims): {status} (conf={np.mean(confidences):.3f})")

            k_results["by_strategy"][strategy_name] = strategy_results

        results["by_k"][str(k)] = k_results

    # ==========================================================================
    # Find TRUE minimum dimensions
    # ==========================================================================

    print("\n" + "=" * 60)
    print("FINDING TRUE MINIMUM DIMENSIONS (k=48)")
    print("=" * 60)

    k = 48
    key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)

    # Test a single message extensively
    test_msg = "Explain how transformers work in neural networks"
    original = key.encode(test_msg, embed_fn)

    print(f"\nMessage: '{test_msg}'")
    print("\nTesting each dimension count (50 trials each)...")

    min_dims_results = {}
    n_trials = 50

    for n_keep in range(1, k + 1):
        n_corrupt = k - n_keep
        successes = 0

        for trial in range(n_trials):
            corrupted = corrupt_random(original, n_corrupt, trial)
            decoded, _ = key.decode(corrupted, candidates, embed_fn)
            if decoded == test_msg:
                successes += 1

        accuracy = successes / n_trials
        min_dims_results[n_keep] = accuracy

        if n_keep <= 10 or accuracy < 1.0:
            print(f"  Keep {n_keep:2d} dims: {accuracy*100:5.1f}%")

    # Find thresholds
    min_100 = k
    for n in range(k, 0, -1):
        if min_dims_results.get(n, 0) >= 1.0:
            min_100 = n
        else:
            break

    min_90 = k
    for n in range(1, k + 1):
        if min_dims_results.get(n, 0) >= 0.90:
            min_90 = n
            break

    min_50 = k
    for n in range(1, k + 1):
        if min_dims_results.get(n, 0) >= 0.50:
            min_50 = n
            break

    results["minimum_dimensions"] = {
        "k": k,
        "min_for_100_pct": min_100,
        "min_for_90_pct": min_90,
        "min_for_50_pct": min_50,
        "full_results": {str(k): v for k, v in min_dims_results.items()},
    }

    print(f"\n  Minimum for 100% accuracy: {min_100} dims ({min_100/k*100:.1f}% of vector)")
    print(f"  Minimum for 90% accuracy: {min_90} dims ({min_90/k*100:.1f}% of vector)")
    print(f"  Minimum for 50% accuracy: {min_50} dims ({min_50/k*100:.1f}% of vector)")

    # ==========================================================================
    # Worst-case analysis (magnitude_high)
    # ==========================================================================

    print("\n" + "=" * 60)
    print("WORST CASE ANALYSIS (delete highest magnitude dims)")
    print("=" * 60)

    worst_case_results = {}

    for n_keep in range(1, k + 1):
        n_corrupt = k - n_keep
        corrupted = corrupt_magnitude_high(original, n_corrupt)
        decoded, conf = key.decode(corrupted, candidates, embed_fn)
        success = decoded == test_msg
        worst_case_results[n_keep] = {"success": success, "confidence": float(conf)}

        if n_keep <= 10 or not success:
            status = "OK" if success else "FAIL"
            print(f"  Keep {n_keep:2d} dims (worst case): {status} (conf={conf:.3f})")

    # Find worst-case threshold
    worst_100 = k
    for n in range(k, 0, -1):
        if worst_case_results.get(n, {}).get("success", False):
            worst_100 = n
        else:
            break

    results["worst_case"] = {
        "strategy": "magnitude_high",
        "min_dims_for_success": worst_100,
        "full_results": {str(k): v for k, v in worst_case_results.items()},
    }

    print(f"\n  Worst-case minimum: {worst_100} dims ({worst_100/k*100:.1f}%)")

    # ==========================================================================
    # Cross-model test
    # ==========================================================================

    print("\n" + "=" * 60)
    print("CROSS-MODEL CORRUPTION TEST")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        model_mini = SentenceTransformer('all-MiniLM-L6-v2')
        model_mpnet = SentenceTransformer('all-mpnet-base-v2')

        def embed_mini(texts):
            return model_mini.encode(texts, convert_to_numpy=True)

        def embed_mpnet(texts):
            return model_mpnet.encode(texts, convert_to_numpy=True)

        # Create keys
        key_nomic = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=48)
        key_mini = AlignmentKey.create("mini", embed_mini, anchors=STABLE_64, k=48)
        key_mpnet = AlignmentKey.create("mpnet", embed_mpnet, anchors=STABLE_64, k=48)

        # Align nomic -> mpnet
        pair = key_nomic.align_with(key_mpnet)
        print(f"  Nomic -> MPNet alignment:")
        print(f"    Spectrum correlation: {pair.spectrum_correlation:.4f}")
        print(f"    Procrustes residual: {pair.procrustes_residual:.4f}")

        # Test cross-model corruption
        original_nomic = key_nomic.encode(test_msg, embed_fn)

        cross_model_results = []
        print(f"\n  Cross-model corruption tolerance:")

        for corruption_pct in [0.0, 0.50, 0.75, 0.90, 0.94]:
            n_corrupt = int(48 * corruption_pct)
            successes = 0

            for trial in range(10):
                corrupted = corrupt_random(original_nomic, n_corrupt, trial) if n_corrupt > 0 else original_nomic.copy()

                # Transform to MPNet space
                transformed = pair.R_a_to_b @ corrupted

                # Decode at MPNet
                decoded, conf = key_mpnet.decode(transformed, candidates, embed_mpnet)
                if decoded == test_msg:
                    successes += 1

            accuracy = successes / 10
            cross_model_results.append({"corruption_pct": corruption_pct * 100, "accuracy": accuracy})
            print(f"    {corruption_pct*100:4.0f}% corruption: {accuracy*100:.0f}%")

        results["cross_model"] = {
            "sender": "nomic",
            "receiver": "mpnet",
            "spectrum_correlation": float(pair.spectrum_correlation),
            "results": cross_model_results,
        }

    except ImportError:
        print("  sentence-transformers not available, skipping")
        results["cross_model"] = None

    # ==========================================================================
    # Final Summary
    # ==========================================================================

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    holographic_score = (k - min_100) / k  # How much can be deleted

    summary = {
        "holographic_score": holographic_score,
        "random_corruption_threshold": f"{100*(k-min_100)/k:.1f}%",
        "worst_case_threshold": f"{100*(k-worst_100)/k:.1f}%",
        "min_dims_needed_random": min_100,
        "min_dims_needed_worst": worst_100,
        "is_holographic": min_100 <= 5,
    }

    results["summary"] = summary

    print(f"\n  Holographic Score: {holographic_score*100:.1f}% of vector can be deleted")
    print(f"  Random corruption threshold: {summary['random_corruption_threshold']}")
    print(f"  Worst-case threshold: {summary['worst_case_threshold']}")
    print(f"  Minimum dims (random): {min_100}/48")
    print(f"  Minimum dims (worst): {worst_100}/48")

    if summary["is_holographic"]:
        print("\n" + "=" * 80)
        print("  VERDICT: MEANING IS HOLOGRAPHIC")
        print("  " + "=" * 76)
        print(f"  Only {min_100} dimensions needed for perfect accuracy!")
        print(f"  {holographic_score*100:.1f}% redundancy in the encoding!")
        print("=" * 80)
    else:
        print(f"\n  Minimum dims > 5: Partial holographic encoding")

    return results


if __name__ == "__main__":
    start = time.time()

    results = run_scaled_test()

    # Save results
    output_path = Path(__file__).parent / "dark_forest_scaled_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTime: {time.time() - start:.1f}s")
    print(f"Results: {output_path}")
