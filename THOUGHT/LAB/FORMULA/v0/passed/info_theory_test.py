#!/usr/bin/env python3
"""
F.7.2: Information-Theoretic Validation

Tests if the formula respects Shannon's laws.
Formula claims: R ∝ E/nabla_S where nabla_S ~ H(X|Y)

Test: Does maximizing E/nabla_S maximize mutual information?

Prediction: Correlation > 0.8 between R and mutual information.
Falsification: Correlation < 0.5 or negative.
"""

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


def test_shannon_consistency(n_trials=1000):
    """
    Shannon: I(X;Y) = H(X) - H(X|Y)
    Formula claims: R ∝ E/nabla_S where nabla_S ~ H(X|Y)

    Test: Does maximizing E/nabla_S maximize mutual information?
    """
    results = []

    for trial in range(n_trials):
        # Generate correlated signals
        noise_level = np.random.uniform(0.1, 2.0)
        signal = np.random.randn(1000)
        noise = np.random.randn(1000) * noise_level
        received = signal + noise

        # Discretize for MI calculation
        bins = np.linspace(-3, 3, 50)
        signal_d = np.digitize(signal, bins=bins)
        received_d = np.digitize(received, bins=bins)

        # Shannon metrics
        H_signal = entropy(np.bincount(signal_d, minlength=51) / len(signal_d))
        H_received = entropy(np.bincount(received_d, minlength=51) / len(received_d))
        MI = mutual_info_score(signal_d, received_d)

        # Formula metrics
        E = np.var(signal)  # Signal strength as essence
        nabla_S = np.var(noise)  # Noise as entropy
        R_formula = E / nabla_S if nabla_S > 0 else float('inf')

        results.append({
            'MI': MI,
            'R_formula': R_formula,
            'H_signal': H_signal,
            'SNR': E / nabla_S if nabla_S > 0 else float('inf')
        })

    # Correlation between MI and R_formula
    MI_values = [r['MI'] for r in results]
    R_values = [min(r['R_formula'], 1e6) for r in results]  # Cap infinities

    correlation = np.corrcoef(MI_values, R_values)[0, 1]

    return correlation, results


if __name__ == '__main__':
    print("F.7.2: Information-Theoretic Validation")
    print("=" * 50)

    correlation, results = test_shannon_consistency()

    print(f"\nResults from {len(results)} trials:")
    print(f"  MI-R correlation: {correlation:.4f}")

    if correlation > 0.8:
        print("\n** VALIDATED: Strong correlation (>0.8)")
    elif correlation > 0.5:
        print("\n*  PASS: Moderate correlation (>0.5)")
    else:
        print("\nX  FALSIFIED: Weak or negative correlation (<0.5)")

    # Show sample of results
    print("\nSample results:")
    for i in [0, len(results)//4, len(results)//2, -1]:
        r = results[i]
        print(f"  Trial {i}: MI={r['MI']:.4f}, R={r['R_formula']:.4f}, SNR={r['SNR']:.4f}")
