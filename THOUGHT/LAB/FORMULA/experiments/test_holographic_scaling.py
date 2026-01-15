#!/usr/bin/env python3
"""
Test Prediction 3: Linear Holographic Scaling

Claim: D/Df ~ constant ~ 35 across embedding dimensions (not exponential)

CORRECTED: Uses covariance eigenspectrum Df (Q43 method), not per-vector IPR.

The Bloch Sphere report claims:
- BERT 768d -> Df = 22 -> ratio = 35
- This should be constant across model dimensions

Df is computed as participation ratio of covariance eigenvalues:
  Df = (sum lambda_i)^2 / sum(lambda_i^2)
"""

import sys
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
sys.path.insert(0, str(CAPABILITY_PATH))

import numpy as np

# Test corpus - same texts for all models (expanded for better covariance estimate)
TEST_TEXTS = [
    # Scientific concepts
    "quantum mechanics describes the behavior of particles at atomic scales",
    "neural networks learn patterns from data through gradient descent",
    "thermodynamics governs energy transfer in physical systems",
    "information theory quantifies uncertainty and communication",
    "relativity shows that space and time are interconnected",

    # Abstract concepts
    "consciousness emerges from complex information processing",
    "meaning is constructed through relationships between symbols",
    "truth is correspondence between statements and reality",
    "beauty arises from harmony and proportion",
    "justice requires fair distribution of resources",

    # Technical descriptions
    "the algorithm iterates until convergence is achieved",
    "embeddings map discrete tokens to continuous vectors",
    "attention mechanisms weight relevant context",
    "transformers process sequences in parallel",
    "optimization minimizes the loss function",

    # Everyday concepts
    "cats are independent pets that sleep most of the day",
    "coffee provides caffeine to increase alertness",
    "weather changes based on atmospheric conditions",
    "music combines rhythm melody and harmony",
    "books preserve knowledge across generations",

    # Additional for better covariance estimate
    "mathematics describes abstract structures and relationships",
    "language enables complex communication between humans",
    "evolution shapes species through natural selection",
    "economics studies resource allocation decisions",
    "philosophy examines fundamental questions about existence",
    "chemistry explores molecular bonds and reactions",
    "biology investigates living organisms and their processes",
    "psychology studies mind and behavior patterns",
    "sociology examines human social structures",
    "art expresses human creativity and emotion",
    "history records past events and civilizations",
    "geography maps physical features of earth",
    "astronomy observes celestial bodies and phenomena",
    "engineering applies science to build solutions",
    "medicine treats diseases and promotes health",
    "law establishes rules for social order",
    "politics governs collective decision making",
    "ethics determines right and wrong actions",
    "logic formalizes valid reasoning patterns",
    "statistics analyzes data to find patterns",
]


def compute_covariance_Df(vectors: np.ndarray) -> dict:
    """
    Compute Df from covariance eigenspectrum (Q43 method).

    This is the CORRECT Df for the holographic scaling claim.

    Returns:
    - Df: participation ratio of eigenvalues
    - eigenvalues: the spectrum
    - top_k_variance: cumulative variance of top k components
    """
    # Center the data
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean

    # Compute covariance (or use SVD for numerical stability)
    # C = X^T @ X / N
    N = len(vectors)
    cov = centered.T @ centered / N

    # Get eigenvalues (only need values, not vectors)
    eigenvalues = np.linalg.eigvalsh(cov)  # Hermitian, so real
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Filter out numerical noise
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    if sum_lambda_sq < 1e-20:
        Df = 0.0
    else:
        Df = (sum_lambda ** 2) / sum_lambda_sq

    # Cumulative variance explained
    total_var = sum_lambda
    cumulative = np.cumsum(eigenvalues) / total_var if total_var > 0 else eigenvalues

    return {
        'Df': float(Df),
        'eigenvalues': eigenvalues,
        'top_10_variance': float(cumulative[9]) if len(cumulative) >= 10 else float(cumulative[-1]),
        'top_22_variance': float(cumulative[21]) if len(cumulative) >= 22 else float(cumulative[-1]),
    }


def test_single_model(model_name: str, expected_dim: int):
    """Test a single model and return D/Df ratio using covariance Df."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        actual_dim = model.get_sentence_embedding_dimension()

        if actual_dim != expected_dim:
            print(f"  Warning: Expected {expected_dim}d, got {actual_dim}d")

        # Embed all test texts
        vectors = model.encode(TEST_TEXTS, convert_to_numpy=True)

        # Normalize to unit sphere (as per Q43)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        # Compute covariance Df (correct method)
        cov_result = compute_covariance_Df(vectors)
        Df = cov_result['Df']

        ratio = actual_dim / Df if Df > 0 else float('inf')

        return {
            'model': model_name,
            'D': actual_dim,
            'Df': Df,
            'ratio': ratio,
            'top_10_var': cov_result['top_10_variance'],
            'top_22_var': cov_result['top_22_variance'],
            'n_samples': len(TEST_TEXTS)
        }

    except Exception as e:
        return {
            'model': model_name,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("PREDICTION 3: Linear Holographic Scaling (CORRECTED)")
    print("Claim: D/Df ~ constant ~ 35 across embedding dimensions")
    print("Using: Covariance eigenspectrum Df (Q43 method)")
    print("=" * 70)
    print()

    # Models to test (different dimensions)
    models = [
        ('all-MiniLM-L6-v2', 384),
        ('all-mpnet-base-v2', 768),
        ('paraphrase-MiniLM-L6-v2', 384),
        ('multi-qa-MiniLM-L6-cos-v1', 384),
    ]

    results = []
    ratios = []

    for model_name, expected_dim in models:
        print(f"Testing: {model_name} ({expected_dim}d)...")
        result = test_single_model(model_name, expected_dim)
        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  D={result['D']}, Df={result['Df']:.2f}")
            print(f"  D/Df ratio = {result['ratio']:.2f}")
            print(f"  Top-10 variance: {result['top_10_var']:.1%}")
            print(f"  Top-22 variance: {result['top_22_var']:.1%}")
            ratios.append(result['ratio'])
        print()

    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    if len(ratios) >= 2:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        cv = std_ratio / mean_ratio * 100  # Coefficient of variation

        print(f"Mean D/Df ratio: {mean_ratio:.2f} +/- {std_ratio:.2f}")
        print(f"Coefficient of Variation: {cv:.1f}%")
        print()

        # Check against Bloch Sphere prediction (D/Df ~ 35)
        print(f"Bloch Sphere prediction: D/Df ~ 35")
        print(f"Measured mean: {mean_ratio:.1f}")
        deviation_from_35 = abs(mean_ratio - 35) / 35 * 100
        print(f"Deviation from 35: {deviation_from_35:.1f}%")
        print()

        # Verdict
        if cv < 20:
            print("[PASS] D/Df ratio is approximately constant across models")
            print(f"  CV = {cv:.1f}% < 20% threshold")

            if deviation_from_35 < 30:
                print(f"  Ratio ~ {mean_ratio:.0f} is close to predicted 35")
                print("  Holographic compression claim SUPPORTED")
            else:
                print(f"  BUT ratio ~ {mean_ratio:.0f} differs from predicted 35")
                print("  Scaling is linear but constant differs from prediction")
        else:
            print("[FAIL] D/Df ratio varies significantly across models")
            print(f"  CV = {cv:.1f}% > 20% threshold")

        print()
        print("Individual results:")
        for r in results:
            if 'ratio' in r:
                deviation = (r['ratio'] - mean_ratio) / mean_ratio * 100
                print(f"  {r['model']}: D={r['D']}, Df={r['Df']:.1f}, ratio={r['ratio']:.1f} ({deviation:+.1f}%)")
    else:
        print("Not enough successful tests to analyze")

    print()
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
