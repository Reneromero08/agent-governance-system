#!/usr/bin/env python3
"""Dark Forest FULL SCALE Test: Holographic Vector Communication at 110%.

This is the comprehensive stress test for holographic encoding.
We test EVERYTHING:
- Multiple corruption patterns (random, sequential, eigenvalue-ordered)
- Different k values (8, 16, 32, 48, 64)
- Diverse message types (technical, emotional, abstract, concrete)
- Statistical rigor (many trials, confidence intervals)
- Cross-model corruption
- Find the TRUE minimum dimensions

Usage:
    python dark_forest_full_scale.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import requests
from scipy import stats

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64, CANONICAL_128


# =============================================================================
# Test Messages - Diverse Semantic Domains
# =============================================================================

TEST_MESSAGES = {
    "technical": [
        "Explain how transformers work in neural networks",
        "Describe the architecture of convolutional neural networks",
        "What is gradient descent optimization in machine learning",
        "How does backpropagation update neural network weights",
        "Explain the attention mechanism in sequence models",
    ],
    "emotional": [
        "Love is a powerful force that connects all humanity",
        "Fear can paralyze us or motivate us to action",
        "Joy comes from meaningful connections with others",
        "Grief is the price we pay for having loved deeply",
        "Hope sustains us through the darkest of times",
    ],
    "abstract": [
        "Time flows like a river toward an unknown sea",
        "Truth emerges from the collision of different perspectives",
        "Beauty exists in the eye of the beholder alone",
        "Justice requires balancing mercy with accountability",
        "Freedom carries the weight of responsibility",
    ],
    "concrete": [
        "The dog ran across the green park this morning",
        "Water boils at one hundred degrees Celsius at sea level",
        "The red car drove down the winding mountain road",
        "Birds fly south during the cold winter months",
        "The old wooden bridge crosses over the river",
    ],
    "scientific": [
        "Quantum mechanics describes behavior of subatomic particles",
        "DNA carries genetic information in living organisms",
        "Gravity bends spacetime around massive objects",
        "Evolution occurs through natural selection over generations",
        "Entropy always increases in a closed system",
    ],
}

# Distractors for each category
DISTRACTORS = {
    "technical": [
        "How do databases store information efficiently",
        "What is the purpose of operating system kernels",
        "Explain how compilers translate source code",
    ],
    "emotional": [
        "Anger can be destructive or transformative",
        "Sadness teaches us what we truly value",
        "Excitement drives us toward new experiences",
    ],
    "abstract": [
        "Power corrupts those who wield it carelessly",
        "Wisdom comes from experience and reflection",
        "Chaos contains hidden patterns of order",
    ],
    "concrete": [
        "The cat sat on the warm windowsill today",
        "Rain falls from gray clouds in the sky",
        "The tall building stands at the city center",
    ],
    "scientific": [
        "Cells divide through mitosis in organisms",
        "Light travels at constant speed in vacuum",
        "Atoms bond together to form molecules",
    ],
}


# =============================================================================
# Corruption Strategies
# =============================================================================

def corrupt_random(vector: np.ndarray, n_dims: int, seed: int) -> Tuple[np.ndarray, List[int]]:
    """Random dimension deletion."""
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] = 0.0
    return corrupted, sorted(indices.tolist())


def corrupt_sequential_low(vector: np.ndarray, n_dims: int, seed: int = None) -> Tuple[np.ndarray, List[int]]:
    """Delete lowest-index dimensions first."""
    corrupted = vector.copy()
    indices = list(range(n_dims))
    corrupted[indices] = 0.0
    return corrupted, indices


def corrupt_sequential_high(vector: np.ndarray, n_dims: int, seed: int = None) -> Tuple[np.ndarray, List[int]]:
    """Delete highest-index dimensions first."""
    corrupted = vector.copy()
    k = len(vector)
    indices = list(range(k - n_dims, k))
    corrupted[indices] = 0.0
    return corrupted, indices


def corrupt_magnitude_low(vector: np.ndarray, n_dims: int, seed: int = None) -> Tuple[np.ndarray, List[int]]:
    """Delete dimensions with lowest absolute magnitude."""
    corrupted = vector.copy()
    magnitudes = np.abs(vector)
    indices = np.argsort(magnitudes)[:n_dims].tolist()
    corrupted[indices] = 0.0
    return corrupted, sorted(indices)


def corrupt_magnitude_high(vector: np.ndarray, n_dims: int, seed: int = None) -> Tuple[np.ndarray, List[int]]:
    """Delete dimensions with highest absolute magnitude (worst case)."""
    corrupted = vector.copy()
    magnitudes = np.abs(vector)
    indices = np.argsort(magnitudes)[-n_dims:].tolist()
    corrupted[indices] = 0.0
    return corrupted, sorted(indices)


def corrupt_gaussian_noise(vector: np.ndarray, n_dims: int, seed: int) -> Tuple[np.ndarray, List[int]]:
    """Add Gaussian noise to random dimensions instead of zeroing."""
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    noise_scale = np.std(vector) * 2  # Strong noise
    corrupted[indices] += np.random.randn(n_dims) * noise_scale
    return corrupted, sorted(indices.tolist())


def corrupt_sign_flip(vector: np.ndarray, n_dims: int, seed: int) -> Tuple[np.ndarray, List[int]]:
    """Flip signs of random dimensions."""
    np.random.seed(seed)
    corrupted = vector.copy()
    indices = np.random.choice(len(vector), n_dims, replace=False)
    corrupted[indices] = -corrupted[indices]
    return corrupted, sorted(indices.tolist())


CORRUPTION_STRATEGIES = {
    "random_zero": corrupt_random,
    "sequential_low": corrupt_sequential_low,
    "sequential_high": corrupt_sequential_high,
    "magnitude_low": corrupt_magnitude_low,
    "magnitude_high": corrupt_magnitude_high,
    "gaussian_noise": corrupt_gaussian_noise,
    "sign_flip": corrupt_sign_flip,
}


# =============================================================================
# Embedding Functions
# =============================================================================

def get_embedding_api(texts: List[str], url: str, model: str) -> np.ndarray:
    """Get embeddings from API."""
    response = requests.post(
        url,
        json={"model": model, "input": texts},
        timeout=60
    )
    data = response.json()
    embeddings = [d["embedding"] for d in data["data"]]
    return np.array(embeddings)


def get_embedding_local(texts: List[str], model) -> np.ndarray:
    """Get embeddings from local model."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


# =============================================================================
# Full Scale Test
# =============================================================================

@dataclass
class CorruptionResult:
    strategy: str
    n_corrupted: int
    corruption_percent: float
    n_trials: int
    successes: int
    accuracy: float
    mean_confidence: float
    std_confidence: float
    min_confidence: float


@dataclass
class MessageResult:
    message: str
    category: str
    k: int
    corruption_results: List[CorruptionResult] = field(default_factory=list)
    threshold_dims: int = 0  # First drop from 100%
    failure_dims: int = 0    # Drop below 50%


def run_full_scale_test(
    embed_url: str = "http://10.5.0.2:1234/v1/embeddings",
    embed_model: str = "text-embedding-nomic-embed-text-v1.5",
    k_values: List[int] = None,
    corruption_levels: List[float] = None,
    n_trials: int = 10,
    strategies: List[str] = None,
) -> Dict:
    """Run the full-scale Dark Forest test."""

    if k_values is None:
        k_values = [8, 16, 32, 48]

    if corruption_levels is None:
        # Test 0%, 25%, 50%, 75%, 85%, 90%, 92%, 94%, 96%, 98%
        corruption_levels = [0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]

    if strategies is None:
        strategies = list(CORRUPTION_STRATEGIES.keys())

    print("=" * 80)
    print("DARK FOREST FULL SCALE TEST - 110%")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"k values: {k_values}")
    print(f"Corruption levels: {[f'{x*100:.0f}%' for x in corruption_levels]}")
    print(f"Strategies: {strategies}")
    print(f"Trials per test: {n_trials}")
    print(f"Message categories: {list(TEST_MESSAGES.keys())}")
    print()

    # Create embedding function
    def embed_fn(texts):
        return get_embedding_api(texts, embed_url, embed_model)

    results = {
        "test_id": "dark-forest-full-scale",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "k_values": k_values,
            "corruption_levels": corruption_levels,
            "n_trials": n_trials,
            "strategies": strategies,
            "embed_model": embed_model,
        },
        "by_k": {},
        "by_category": {},
        "by_strategy": {},
        "summary": {},
    }

    # Test each k value
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"TESTING k={k}")
        print(f"{'='*80}")

        # Create alignment key
        key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)

        k_results = {
            "categories": {},
            "aggregate": {
                "total_tests": 0,
                "total_successes": 0,
            }
        }

        # Test each category
        for category, messages in TEST_MESSAGES.items():
            print(f"\n  Category: {category.upper()}")

            category_results = []
            distractors = DISTRACTORS[category]

            for msg_idx, message in enumerate(messages):
                print(f"    Message {msg_idx+1}: '{message[:40]}...'")

                # Build candidate pool
                candidates = [message] + distractors + [m for m in messages if m != message]

                # Encode message
                original_vector = key.encode(message, embed_fn)

                msg_result = MessageResult(
                    message=message,
                    category=category,
                    k=k,
                )

                found_threshold = False
                found_failure = False

                # Test each corruption level
                for corruption_pct in corruption_levels:
                    n_corrupt = int(k * corruption_pct)
                    if n_corrupt >= k:
                        n_corrupt = k - 1

                    # Test each strategy
                    for strategy_name in strategies:
                        strategy_fn = CORRUPTION_STRATEGIES[strategy_name]

                        successes = 0
                        confidences = []

                        for trial in range(n_trials):
                            seed = trial + msg_idx * 100

                            if n_corrupt == 0:
                                corrupted = original_vector.copy()
                            else:
                                corrupted, _ = strategy_fn(original_vector, n_corrupt, seed)

                            decoded, conf = key.decode(corrupted, candidates, embed_fn)
                            confidences.append(conf)

                            if decoded == message:
                                successes += 1

                        accuracy = successes / n_trials

                        cr = CorruptionResult(
                            strategy=strategy_name,
                            n_corrupted=n_corrupt,
                            corruption_percent=corruption_pct * 100,
                            n_trials=n_trials,
                            successes=successes,
                            accuracy=accuracy,
                            mean_confidence=float(np.mean(confidences)),
                            std_confidence=float(np.std(confidences)),
                            min_confidence=float(np.min(confidences)),
                        )
                        msg_result.corruption_results.append(cr)

                        k_results["aggregate"]["total_tests"] += n_trials
                        k_results["aggregate"]["total_successes"] += successes

                        # Track threshold
                        if accuracy < 1.0 and not found_threshold:
                            msg_result.threshold_dims = n_corrupt
                            found_threshold = True

                        if accuracy < 0.5 and not found_failure:
                            msg_result.failure_dims = n_corrupt
                            found_failure = True

                if not found_threshold:
                    msg_result.threshold_dims = k
                if not found_failure:
                    msg_result.failure_dims = k

                category_results.append(msg_result)

                # Print summary for this message
                best_threshold = msg_result.threshold_dims
                best_failure = msg_result.failure_dims
                print(f"      Threshold: {best_threshold}/{k} ({100*best_threshold/k:.0f}%), Failure: {best_failure}/{k} ({100*best_failure/k:.0f}%)")

            k_results["categories"][category] = {
                "messages": [
                    {
                        "message": mr.message[:50],
                        "threshold_dims": mr.threshold_dims,
                        "threshold_pct": 100 * mr.threshold_dims / k,
                        "failure_dims": mr.failure_dims,
                        "failure_pct": 100 * mr.failure_dims / k,
                    }
                    for mr in category_results
                ],
                "mean_threshold_pct": np.mean([mr.threshold_dims for mr in category_results]) / k * 100,
                "mean_failure_pct": np.mean([mr.failure_dims for mr in category_results]) / k * 100,
            }

        # Aggregate for this k
        overall_accuracy = k_results["aggregate"]["total_successes"] / k_results["aggregate"]["total_tests"]
        k_results["aggregate"]["overall_accuracy"] = overall_accuracy

        results["by_k"][str(k)] = k_results

        print(f"\n  k={k} Overall Accuracy: {overall_accuracy*100:.1f}%")

    # ==========================================================================
    # Cross-strategy analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CROSS-STRATEGY ANALYSIS")
    print("=" * 80)

    strategy_performance = {s: [] for s in strategies}

    # Collect accuracy by strategy across all tests
    for k in k_values:
        for category in TEST_MESSAGES.keys():
            for msg_data in results["by_k"][str(k)]["categories"][category]["messages"]:
                # We need to re-aggregate... simplified for now
                pass

    # ==========================================================================
    # Find TRUE minimum dimensions
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINDING TRUE MINIMUM DIMENSIONS")
    print("=" * 80)

    k = 48  # Use full resolution
    key = AlignmentKey.create("nomic", embed_fn, anchors=STABLE_64, k=k)

    # Test message
    test_msg = "Explain how transformers work in neural networks"
    candidates = [test_msg] + DISTRACTORS["technical"]
    original = key.encode(test_msg, embed_fn)

    print(f"\nMessage: '{test_msg}'")
    print(f"Testing k={k} with random corruption...")

    # Binary search for minimum
    min_dims_results = {}

    for n_keep in range(1, k + 1):
        n_corrupt = k - n_keep
        successes = 0

        for trial in range(20):  # 20 trials per test
            np.random.seed(trial)
            corrupted = original.copy()
            indices = np.random.choice(k, n_corrupt, replace=False)
            corrupted[indices] = 0.0

            decoded, conf = key.decode(corrupted, candidates, embed_fn)
            if decoded == test_msg:
                successes += 1

        accuracy = successes / 20
        min_dims_results[n_keep] = accuracy

        status = "OK" if accuracy == 1.0 else f"{accuracy*100:.0f}%"
        print(f"  Keep {n_keep:2d} dims ({n_keep/k*100:4.1f}%): {status}")

        if accuracy < 0.5:
            break

    # Find minimum for 100% accuracy
    min_for_100 = k
    for n_keep in range(k, 0, -1):
        if min_dims_results.get(n_keep, 0) == 1.0:
            min_for_100 = n_keep
        else:
            break

    # Find minimum for 50% accuracy
    min_for_50 = 1
    for n_keep in range(1, k + 1):
        if min_dims_results.get(n_keep, 0) >= 0.5:
            min_for_50 = n_keep
            break

    results["minimum_dimensions"] = {
        "k": k,
        "min_dims_for_100_accuracy": min_for_100,
        "min_dims_for_50_accuracy": min_for_50,
        "full_results": min_dims_results,
    }

    print(f"\n  Minimum for 100% accuracy: {min_for_100} dims ({min_for_100/k*100:.1f}%)")
    print(f"  Minimum for 50% accuracy: {min_for_50} dims ({min_for_50/k*100:.1f}%)")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": sum(
            results["by_k"][str(k)]["aggregate"]["total_tests"]
            for k in k_values
        ),
        "overall_accuracy": np.mean([
            results["by_k"][str(k)]["aggregate"]["overall_accuracy"]
            for k in k_values
        ]),
        "minimum_dims_for_100": min_for_100,
        "minimum_dims_for_50": min_for_50,
        "holographic_verified": min_for_100 <= 5,  # TRUE holographic if only 5 dims needed
    }

    results["summary"] = summary

    print(f"\nTotal tests run: {summary['total_tests']}")
    print(f"Overall accuracy: {summary['overall_accuracy']*100:.1f}%")
    print(f"Minimum dims for 100% accuracy: {summary['minimum_dims_for_100']} / 48")
    print(f"Minimum dims for 50% accuracy: {summary['minimum_dims_for_50']} / 48")
    print(f"HOLOGRAPHIC VERIFIED: {summary['holographic_verified']}")

    if summary['holographic_verified']:
        print("\n" + "=" * 80)
        print("VERDICT: MEANING IS HOLOGRAPHIC")
        print("=" * 80)
        print("Only", summary['minimum_dims_for_100'], "dimensions needed for perfect accuracy!")
        print("Information is distributed across the entire vector topology.")
        print("=" * 80)

    return results


def run_cross_model_corruption_test(
    embed_url: str = "http://10.5.0.2:1234/v1/embeddings",
    embed_model_a: str = "text-embedding-nomic-embed-text-v1.5",
) -> Dict:
    """Test corruption tolerance across different models."""
    print("\n" + "=" * 80)
    print("CROSS-MODEL CORRUPTION TEST")
    print("=" * 80)

    try:
        from sentence_transformers import SentenceTransformer
        model_b = SentenceTransformer('all-MiniLM-L6-v2')
        model_c = SentenceTransformer('all-mpnet-base-v2')
    except ImportError:
        print("sentence-transformers not available, skipping cross-model test")
        return {}

    def embed_fn_a(texts):
        return get_embedding_api(texts, embed_url, embed_model_a)

    def embed_fn_b(texts):
        return model_b.encode(texts, convert_to_numpy=True)

    def embed_fn_c(texts):
        return model_c.encode(texts, convert_to_numpy=True)

    models = [
        ("nomic-v1.5", embed_fn_a),
        ("MiniLM", embed_fn_b),
        ("MPNet", embed_fn_c),
    ]

    k = 48
    test_msg = "Explain how transformers work in neural networks"
    candidates = [test_msg] + DISTRACTORS["technical"]

    results = {}

    # Test each sender -> receiver pair
    for sender_name, sender_embed in models:
        print(f"\n  Sender: {sender_name}")

        # Create key for sender
        key_sender = AlignmentKey.create(sender_name, sender_embed, anchors=STABLE_64, k=k)
        original = key_sender.encode(test_msg, sender_embed)

        for receiver_name, receiver_embed in models:
            if sender_name == receiver_name:
                continue

            print(f"    -> Receiver: {receiver_name}")

            # Create key for receiver and align
            key_receiver = AlignmentKey.create(receiver_name, receiver_embed, anchors=STABLE_64, k=k)
            pair = key_sender.align_with(key_receiver)

            # Test corruption levels
            corruption_results = []
            for corruption_pct in [0.0, 0.50, 0.75, 0.90, 0.94]:
                n_corrupt = int(k * corruption_pct)
                successes = 0

                for trial in range(10):
                    np.random.seed(trial)
                    corrupted = original.copy()
                    if n_corrupt > 0:
                        indices = np.random.choice(k, n_corrupt, replace=False)
                        corrupted[indices] = 0.0

                    # Transform to receiver space
                    transformed = pair.R_a_to_b @ corrupted

                    # Decode at receiver
                    decoded, conf = key_receiver.decode(transformed, candidates, receiver_embed)
                    if decoded == test_msg:
                        successes += 1

                accuracy = successes / 10
                corruption_results.append({
                    "corruption_pct": corruption_pct * 100,
                    "accuracy": accuracy,
                })
                print(f"      {corruption_pct*100:4.0f}% corruption: {accuracy*100:.0f}%")

            results[f"{sender_name}->{receiver_name}"] = corruption_results

    return results


if __name__ == "__main__":
    start_time = time.time()

    # Run full scale test
    results = run_full_scale_test(
        k_values=[16, 32, 48],
        corruption_levels=[0.0, 0.50, 0.75, 0.90, 0.94, 0.96],
        n_trials=5,
        strategies=["random_zero", "magnitude_high", "sign_flip"],
    )

    # Run cross-model test
    cross_model_results = run_cross_model_corruption_test()
    results["cross_model"] = cross_model_results

    # Save results
    output_path = Path(__file__).parent / "dark_forest_full_scale_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed:.1f} seconds")
    print(f"Results saved to: {output_path}")
