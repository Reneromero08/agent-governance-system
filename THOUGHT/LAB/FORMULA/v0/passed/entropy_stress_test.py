#!/usr/bin/env python3
"""
F.7.6: Entropy Injection Stress Test

Tests if R degrades exactly as 1/nabla_S predicts.

Formula predicts: R = k × (E / nabla_S) × sigma^Df
If we fix E, sigma, Df, then: R ∝ 1/nabla_S
This means: R × nabla_S = constant

Prediction: R × nabla_S has coefficient of variation < 0.2.
Falsification: CV > 0.5 (relationship is not inverse).
"""

import numpy as np


def entropy_injection_test():
    """
    Systematically inject entropy and measure R degradation.

    Formula predicts: R = k × (E / nabla_S) × sigma^Df

    If we fix E, sigma, Df, then: R ∝ 1/nabla_S

    This means: R × nabla_S = constant
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Fixed symbol set
        symbols = ['dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep']
        embeddings = model.encode(symbols)
    except ImportError:
        print("sentence-transformers not installed, using random embeddings")
        symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        np.random.seed(42)
        embeddings = np.random.randn(8, 384)

    # Vary entropy levels
    entropy_levels = np.linspace(0.01, 2.0, 50)

    results = []
    for nabla_S in entropy_levels:
        # Inject Gaussian noise scaled by entropy level
        R_trials = []

        for _ in range(100):
            # Add noise to embeddings (simulating context corruption)
            noisy_embeddings = embeddings + np.random.randn(*embeddings.shape) * nabla_S

            # Measure retrieval accuracy
            correct = 0
            for i in range(len(symbols)):
                query = embeddings[i]  # Clean query
                distances = np.linalg.norm(noisy_embeddings - query, axis=1)
                if np.argmin(distances) == i:
                    correct += 1

            R_trials.append(correct / len(symbols))

        R_mean = np.mean(R_trials)
        R_std = np.std(R_trials)

        results.append({
            'nabla_S': nabla_S,
            'R': R_mean,
            'R_std': R_std,
            'R_times_nabla_S': R_mean * nabla_S  # Should be constant if formula correct
        })

    # Test: Is R × nabla_S approximately constant?
    products = [r['R_times_nabla_S'] for r in results]
    cv = np.std(products) / np.mean(products) if np.mean(products) > 0 else float('inf')

    return results, cv


if __name__ == '__main__':
    print("F.7.6: Entropy Injection Stress Test")
    print("=" * 50)

    results, cv = entropy_injection_test()

    print(f"\nResults across {len(results)} entropy levels:")
    print("-" * 50)
    print(f"{'nabla_S':>8s} | {'R':>8s} | {'R×nabla_S':>10s}")
    print("-" * 50)

    # Show sample of results
    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]
    for i in indices:
        r = results[i]
        print(f"{r['nabla_S']:8.3f} | {r['R']:8.4f} | {r['R_times_nabla_S']:10.4f}")

    print("-" * 50)

    # R × nabla_S stats
    products = [r['R_times_nabla_S'] for r in results]
    print(f"\nR × nabla_S statistics:")
    print(f"  Mean: {np.mean(products):.4f}")
    print(f"  Std:  {np.std(products):.4f}")
    print(f"  CV:   {cv:.4f}")

    # Also check 1/nabla_S relationship directly
    nabla_S_values = [r['nabla_S'] for r in results]
    R_values = [r['R'] for r in results]
    inv_nabla_S = [1/n for n in nabla_S_values]

    corr_inverse = np.corrcoef(inv_nabla_S, R_values)[0, 1]
    print(f"\n1/nabla_S correlation with R: {corr_inverse:.4f}")

    if cv < 0.2:
        print("\n** VALIDATED: R × nabla_S is nearly constant (CV < 0.2)")
    elif cv < 0.5:
        print("\n*  PASS: R × nabla_S is approximately constant (CV < 0.5)")
    else:
        print("\nX  FALSIFIED: R × nabla_S is not constant (CV > 0.5)")
