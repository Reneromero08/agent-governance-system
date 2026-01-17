#!/usr/bin/env python3
"""Test 1: Code Distance Determination.

Proves R-gating implements QECC with measurable code distance d = 2t + 1.

Hypothesis:
    Semantic embeddings have higher error tolerance (t_max) than random
    embeddings at the same dimensionality.

Protocol:
    1. For 1000 phrases, compute baseline R_0
    2. Inject n=1,2,...,20 errors per type
    3. Find t_max where gate passes >50%
    4. Compute d = 2*t_max + 1
    5. Compare semantic vs random baseline

Success Criteria:
    - Semantic t_max > Random t_max with p < 0.01
    - For Df~22, expect d ~ 4-7 (sqrt scaling)

Usage:
    python test_code_distance.py [--n-samples 100] [--max-errors 20]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

# Local imports
from core import (
    inject_n_errors,
    r_gate,
    generate_random_embeddings,
    compute_effective_dimensionality,
    cohens_d,
    CodeDistanceResult,
    DEFAULT_R_THRESHOLD,
)

# Try to import sentence transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Using synthetic data.")


# =============================================================================
# Test Phrases
# =============================================================================

TEST_PHRASES = [
    # Factual statements
    "Water freezes at zero degrees Celsius.",
    "The Earth orbits the Sun once per year.",
    "Light travels at approximately 300000 kilometers per second.",
    "DNA contains the genetic instructions for living organisms.",
    "Gravity causes objects to fall toward the ground.",

    # Conceptual statements
    "Democracy requires an informed citizenry to function well.",
    "Language shapes the way we think about the world.",
    "Mathematics describes patterns found in nature.",
    "Consciousness emerges from complex neural activity.",
    "Information can be neither created nor destroyed.",

    # Technical statements
    "Hash functions map inputs to fixed-length outputs.",
    "Neural networks learn by adjusting connection weights.",
    "Encryption protects data from unauthorized access.",
    "Algorithms solve problems through step-by-step procedures.",
    "Databases store and retrieve structured information.",

    # Narrative statements
    "The hero overcame great obstacles to achieve victory.",
    "Trust must be earned through consistent actions.",
    "Change often brings both challenges and opportunities.",
    "Knowledge grows through questioning assumptions.",
    "Communication bridges the gap between minds.",
]


def expand_test_phrases(base_phrases: List[str], n_target: int) -> List[str]:
    """Expand test phrases to target count through variations."""
    expanded = list(base_phrases)

    # Simple variations
    prefixes = ["Indeed, ", "Clearly, ", "In fact, ", "Obviously, ", ""]
    suffixes = ["", " This is important.", " This is fundamental.", ""]

    while len(expanded) < n_target:
        base = base_phrases[len(expanded) % len(base_phrases)]
        prefix = prefixes[len(expanded) % len(prefixes)]
        suffix = suffixes[len(expanded) % len(suffixes)]
        expanded.append(f"{prefix}{base}{suffix}")

    return expanded[:n_target]


# =============================================================================
# Embedding Functions
# =============================================================================

def get_semantic_embeddings(
    phrases: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Get semantic embeddings for phrases using sentence transformer."""
    if not HAS_TRANSFORMERS:
        # Fallback: generate structured random embeddings
        # that simulate semantic clustering
        np.random.seed(42)
        dim = 384
        n = len(phrases)
        embeddings = np.random.randn(n, dim)

        # Add structure: similar phrases should cluster
        for i in range(n):
            cluster_id = i % 5  # 5 clusters based on phrase type
            cluster_center = np.random.randn(dim)
            cluster_center = cluster_center / np.linalg.norm(cluster_center)
            embeddings[i] = 0.7 * cluster_center + 0.3 * embeddings[i]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases, convert_to_numpy=True)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# =============================================================================
# Code Distance Test
# =============================================================================

def find_t_max(
    embeddings: np.ndarray,
    error_type: str,
    max_errors: int = 20,
    n_trials: int = 100,
    threshold: float = DEFAULT_R_THRESHOLD,
    pass_rate_threshold: float = 0.5,
    **error_kwargs
) -> Tuple[int, List[float]]:
    """Find maximum errors where gate still passes >50%.

    Args:
        embeddings: (n, d) semantic or random embeddings
        error_type: Type of error to inject
        max_errors: Maximum errors to test
        n_trials: Number of trials per error count
        threshold: R-gate threshold
        pass_rate_threshold: Rate above which gate is considered working
        **error_kwargs: Additional args for error injection

    Returns:
        Tuple of (t_max, gate_pass_rates)
    """
    dim = embeddings.shape[1]
    pass_rates = []

    for n_errors in range(max_errors + 1):
        passes = 0

        for trial in range(n_trials):
            # Pick a random embedding
            idx = np.random.randint(len(embeddings))
            original = embeddings[idx]

            # Create "observations" by adding small noise to original
            n_obs = 5
            observations = np.array([
                original + np.random.randn(dim) * 0.01
                for _ in range(n_obs)
            ])
            # Normalize
            observations = observations / np.linalg.norm(observations, axis=1, keepdims=True)

            # Inject errors into observations
            if n_errors > 0:
                for i in range(n_obs):
                    result = inject_n_errors(
                        observations[i],
                        n_errors,
                        error_type,
                        **error_kwargs
                    )
                    observations[i] = result.corrupted

            # Apply R-gate
            gate_result = r_gate(observations, threshold)
            if gate_result.passed:
                passes += 1

        pass_rate = passes / n_trials
        pass_rates.append(pass_rate)

        print(f"  n_errors={n_errors:2d}: pass_rate={pass_rate:.2%}")

    # Find t_max: last n_errors where pass_rate > threshold
    t_max = 0
    for n_errors, rate in enumerate(pass_rates):
        if rate >= pass_rate_threshold:
            t_max = n_errors

    return t_max, pass_rates


def run_code_distance_test(
    n_samples: int = 100,
    max_errors: int = 20,
    n_trials: int = 50,
    error_types: Optional[List[str]] = None,
    dim: int = 384,
    n_random_seeds: int = 5
) -> Dict:
    """Run full code distance test.

    Args:
        n_samples: Number of test phrases
        max_errors: Maximum errors to test
        n_trials: Trials per error count
        error_types: List of error types to test
        dim: Embedding dimension
        n_random_seeds: Number of random baselines

    Returns:
        Complete test results dict
    """
    if error_types is None:
        error_types = ["dimension_flip", "gaussian_noise", "random_direction"]

    print("=" * 70)
    print("TEST 1: CODE DISTANCE DETERMINATION")
    print("=" * 70)
    print()

    # Get semantic embeddings
    print(f"Generating {n_samples} semantic embeddings...")
    phrases = expand_test_phrases(TEST_PHRASES, n_samples)
    semantic_emb = get_semantic_embeddings(phrases)
    semantic_df = compute_effective_dimensionality(semantic_emb)
    print(f"  Semantic Df: {semantic_df:.2f}")
    print()

    # Generate random baselines
    print(f"Generating {n_random_seeds} random baselines...")
    random_embs = [
        generate_random_embeddings(n_samples, dim, seed=42 + i)
        for i in range(n_random_seeds)
    ]
    random_dfs = [compute_effective_dimensionality(r) for r in random_embs]
    print(f"  Random Df mean: {np.mean(random_dfs):.2f}")
    print()

    results = {
        "test_id": "q40-code-distance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_samples": n_samples,
            "max_errors": max_errors,
            "n_trials": n_trials,
            "dim": dim,
            "n_random_seeds": n_random_seeds,
        },
        "semantic_df": float(semantic_df),
        "random_df_mean": float(np.mean(random_dfs)),
        "error_types": {},
    }

    for error_type in error_types:
        print("-" * 70)
        print(f"ERROR TYPE: {error_type}")
        print("-" * 70)

        # Semantic test
        print("\nSemantic embeddings:")
        semantic_t_max, semantic_rates = find_t_max(
            semantic_emb, error_type, max_errors, n_trials,
            sigma=0.1, epsilon=0.1
        )
        semantic_d = 2 * semantic_t_max + 1
        print(f"  t_max = {semantic_t_max}, code distance d = {semantic_d}")

        # Random baselines
        print("\nRandom baselines:")
        random_t_maxes = []
        for i, random_emb in enumerate(random_embs):
            print(f"  Random seed {i}:")
            t_max, _ = find_t_max(
                random_emb, error_type, max_errors, n_trials,
                sigma=0.1, epsilon=0.1
            )
            random_t_maxes.append(t_max)
            print(f"    t_max = {t_max}")

        random_t_max_mean = np.mean(random_t_maxes)
        random_d_mean = 2 * random_t_max_mean + 1

        # Statistical test
        # Compare semantic t_max to distribution of random t_maxes
        if len(set(random_t_maxes)) > 1:
            # Use one-sample t-test variant
            diff = semantic_t_max - random_t_max_mean
            std = np.std(random_t_maxes, ddof=1) or 1.0
            z_score = diff / (std / np.sqrt(n_random_seeds))
            from scipy.stats import norm
            p_value = 1 - norm.cdf(z_score)
        else:
            p_value = 0.5 if semantic_t_max <= random_t_max_mean else 0.01

        is_significant = (semantic_t_max > random_t_max_mean) and (p_value < 0.01)

        print()
        print(f"RESULTS for {error_type}:")
        print(f"  Semantic t_max: {semantic_t_max} (d = {semantic_d})")
        print(f"  Random t_max mean: {random_t_max_mean:.2f} (d = {random_d_mean:.2f})")
        print(f"  Difference: {semantic_t_max - random_t_max_mean:.2f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  SIGNIFICANT: {is_significant}")

        results["error_types"][error_type] = {
            "semantic_t_max": semantic_t_max,
            "semantic_code_distance": semantic_d,
            "semantic_pass_rates": semantic_rates,
            "random_t_maxes": random_t_maxes,
            "random_t_max_mean": float(random_t_max_mean),
            "random_code_distance_mean": float(random_d_mean),
            "p_value": float(p_value),
            "is_significant": is_significant,
        }

    # Overall verdict
    n_significant = sum(
        1 for et in results["error_types"].values() if et["is_significant"]
    )
    overall_pass = n_significant >= len(error_types) // 2 + 1

    results["verdict"] = {
        "significant_error_types": n_significant,
        "total_error_types": len(error_types),
        "overall_pass": overall_pass,
        "interpretation": (
            "PASS: Semantic embeddings show superior error correction "
            "compared to random baseline. Code distance is a learned property."
            if overall_pass else
            "FAIL: Semantic embeddings do not show significantly better "
            "error correction than random baseline."
        )
    }

    print()
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    print(f"Significant error types: {n_significant}/{len(error_types)}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(results["verdict"]["interpretation"])
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Test 1: Code Distance')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of test phrases')
    parser.add_argument('--max-errors', type=int, default=20,
                        help='Maximum errors to test')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Trials per error count')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_code_distance_test(
        n_samples=args.n_samples,
        max_errors=args.max_errors,
        n_trials=args.n_trials,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "code_distance.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
