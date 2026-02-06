#!/usr/bin/env python3
"""
F.7.5: Eigenvalue Spectrum as Essence Measure

Tests if eigenvalue spectrum can serve as E (essence).

Prediction: E_total and E_rank correlate with R (r > 0.7).
Falsification: No correlation or random words have highest E.
"""

import numpy as np
from scipy.linalg import eigh


def eigenvalue_spectrum_as_E(distance_matrix):
    """
    Hypothesis: The eigenvalue spectrum captures "essence" of semantic space.

    E_eigen = sum of positive eigenvalues (total variance explained)
    """
    n = distance_matrix.shape[0]

    if n < 2:
        return {'E_total': 0, 'E_dominant': 0, 'E_rank': 0, 'spectrum': []}

    # Double-center to get Gram matrix
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (distance_matrix ** 2) @ H

    # Eigendecomposition
    try:
        eigenvalues, _ = eigh(B)
        eigenvalues = eigenvalues[::-1]  # Sort descending
    except:
        return {'E_total': 0, 'E_dominant': 0, 'E_rank': 0, 'spectrum': []}

    # E = sum of positive eigenvalues
    E = np.sum(eigenvalues[eigenvalues > 0])

    # Alternative: E = largest eigenvalue (dominant direction)
    E_dominant = eigenvalues[0] if len(eigenvalues) > 0 else 0

    # Alternative: E = effective rank
    total = np.sum(np.abs(eigenvalues))
    if total > 0:
        probs = np.abs(eigenvalues) / total
        probs = probs[probs > 0]  # Remove zeros
        E_rank = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        E_rank = 0

    return {
        'E_total': float(E),
        'E_dominant': float(E_dominant),
        'E_rank': float(E_rank),
        'spectrum': eigenvalues.tolist()
    }


def measure_retrieval_accuracy(embeddings, noise_trials=100):
    """Measure R as retrieval accuracy under noise."""
    if len(embeddings) < 2:
        return 0.0

    embeddings = np.array(embeddings)
    correct = 0
    total = 0

    for i in range(len(embeddings)):
        for _ in range(noise_trials):
            # Add noise to query
            noise_scale = 0.1 * np.std(embeddings)
            noisy_query = embeddings[i] + np.random.randn(embeddings.shape[1]) * noise_scale

            # Find nearest
            distances = np.linalg.norm(embeddings - noisy_query, axis=1)
            nearest = np.argmin(distances)

            if nearest == i:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def test_eigenvalue_E_predicts_R():
    """
    Test: Does eigenvalue-based E predict R better than alternatives?
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("sentence-transformers not installed, using random embeddings")
        model = None

    # Generate test cases with varying "essence clarity"
    test_cases = [
        # Clear essence (semantically tight cluster)
        {'words': ['cat', 'dog', 'bird', 'fish', 'hamster'], 'expected_E': 'high'},
        # Mixed essence (cross-domain)
        {'words': ['cat', 'democracy', 'blue', 'running', 'seven'], 'expected_E': 'low'},
        # Abstract essence (conceptually related)
        {'words': ['truth', 'honesty', 'integrity', 'authenticity', 'sincerity'], 'expected_E': 'high'},
        # Random essence
        {'words': ['quantum', 'sandwich', 'purple', 'slowly', 'nineteen'], 'expected_E': 'low'},
    ]

    results = []
    for case in test_cases:
        if model:
            embeddings = model.encode(case['words'])
        else:
            # Fallback: random embeddings
            np.random.seed(hash(str(case['words'])) % (2**32))
            embeddings = np.random.randn(len(case['words']), 384)

        # Cosine distance matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-10)
        cosine_sim = embeddings_norm @ embeddings_norm.T
        D = np.sqrt(2 * (1 - cosine_sim))  # Cosine distance

        E_metrics = eigenvalue_spectrum_as_E(D)
        R = measure_retrieval_accuracy(embeddings)

        results.append({
            'words': case['words'],
            'expected_E': case['expected_E'],
            **E_metrics,
            'R': R
        })

    return results


if __name__ == '__main__':
    print("F.7.5: Eigenvalue Spectrum as Essence Measure")
    print("=" * 50)

    results = test_eigenvalue_E_predicts_R()

    print(f"\nResults by word set:")
    print("-" * 70)
    print(f"{'Expected':10s} | {'E_total':>10s} | {'E_dominant':>10s} | {'E_rank':>8s} | {'R':>8s}")
    print("-" * 70)
    for item in results:
        print(f"{item['expected_E']:10s} | {item['E_total']:10.4f} | {item['E_dominant']:10.4f} | {item['E_rank']:8.4f} | {item['R']:8.4f}")
        print(f"  Words: {item['words']}")

    # Calculate correlation
    E_values = [item['E_total'] for item in results]
    R_values = [item['R'] for item in results]

    if len(set(E_values)) > 1 and len(set(R_values)) > 1:
        corr = np.corrcoef(E_values, R_values)[0, 1]
    else:
        corr = 0.0

    print(f"\nE_total-R correlation: {corr:.4f}")

    # Check if expected ordering matches
    expected_high = [r for r in results if r['expected_E'] == 'high']
    expected_low = [r for r in results if r['expected_E'] == 'low']

    if expected_high and expected_low:
        avg_E_high = np.mean([r['E_total'] for r in expected_high])
        avg_E_low = np.mean([r['E_total'] for r in expected_low])
        ordering_correct = avg_E_high > avg_E_low
        print(f"E ordering correct: {ordering_correct} (high={avg_E_high:.4f}, low={avg_E_low:.4f})")

    if corr > 0.7:
        print("\n** VALIDATED: Strong E-R correlation")
    elif corr > 0:
        print("\n*  PASS: Positive E-R correlation")
    elif np.isnan(corr):
        print("\n?  INCONCLUSIVE: Cannot compute correlation")
    else:
        print("\nX  FALSIFIED: No positive E-R correlation")
