#!/usr/bin/env python3
"""
F.7.4: Fractal Dimension Measurement

Tests if we can measure Df and if it correlates with R.

Prediction: Df ordering matches R ordering across symbol sets.
Falsification: No correlation between Df and R, or inverse correlation.
"""

import numpy as np
from scipy.spatial.distance import pdist


def box_counting_dimension(points, eps_range=None):
    """
    Estimate fractal dimension via box-counting.

    For symbol embeddings:
    - High Df = embeddings spread across many dimensions
    - Low Df = embeddings clustered
    """
    if len(points) < 2:
        return 0.0

    # Normalize points to [0, 1] range
    points = np.array(points)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    points_norm = (points - min_vals) / range_vals

    if eps_range is None:
        eps_range = np.logspace(-1, 0, 10)

    counts = []
    for eps in eps_range:
        # Count boxes needed to cover points
        boxes = set()
        for p in points_norm:
            box = tuple((p / eps).astype(int))
            boxes.add(box)
        counts.append(len(boxes))

    # Df = -slope of log(N) vs log(eps)
    log_eps = np.log(eps_range)
    log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)

    # Linear regression
    if len(log_eps) > 1:
        slope, _ = np.polyfit(log_eps, log_counts, 1)
        Df = -slope
    else:
        Df = 0

    return max(0, Df)  # Df should be positive


def information_dimension(embeddings):
    """
    Alternative: Information dimension from embedding entropy.

    Df_info = lim (H(eps) / log(1/eps)) as eps -> 0
    """
    if len(embeddings) < 2:
        return 0.0

    distances = pdist(embeddings)

    if len(distances) == 0:
        return 0.0

    eps_range = np.percentile(distances[distances > 0], [10, 25, 50, 75, 90])
    eps_range = eps_range[eps_range > 0]

    if len(eps_range) == 0:
        return 0.0

    H_values = []
    for eps in eps_range:
        # Probability of being within eps of each point
        probs = np.mean(distances < eps)
        if probs > 0 and probs < 1:
            H_values.append(-probs * np.log(probs))
        else:
            H_values.append(0)

    if len(H_values) < 2:
        return 0.0

    # Estimate Df from scaling
    log_eps_inv = np.log(1 / eps_range)
    try:
        slope, _ = np.polyfit(log_eps_inv, H_values, 1)
        return max(0, slope)
    except:
        return 0.0


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


def test_Df_R_correlation():
    """
    Test: Does measured Df correlate with R?

    Use different symbol sets with varying polysemy.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("sentence-transformers not installed, using random embeddings")
        model = None

    # Symbol sets with different expected Df
    symbol_sets = {
        'monosemic': ['cat', 'dog', 'tree', 'house', 'car'],  # Low Df
        'polysemic_low': ['bank', 'bat', 'ring', 'spring', 'light'],  # Medium Df
        'polysemic_high': ['set', 'run', 'get', 'take', 'make'],  # High Df (most meanings)
        'abstract': ['truth', 'justice', 'beauty', 'freedom', 'love'],  # High Df (conceptual)
    }

    results = {}
    for name, symbols in symbol_sets.items():
        # Get embeddings
        if model:
            embeddings = model.encode(symbols)
        else:
            # Fallback: random embeddings with structure
            np.random.seed(hash(name) % (2**32))
            embeddings = np.random.randn(len(symbols), 384)

        # Measure Df
        Df_box = box_counting_dimension(embeddings)
        Df_info = information_dimension(embeddings)

        # Measure R (retrieval accuracy with noisy queries)
        R = measure_retrieval_accuracy(embeddings)

        results[name] = {
            'Df_box': Df_box,
            'Df_info': Df_info,
            'R': R,
            'expected_order': ['monosemic', 'polysemic_low', 'polysemic_high', 'abstract'].index(name)
        }

    return results


if __name__ == '__main__':
    print("F.7.4: Fractal Dimension Measurement")
    print("=" * 50)

    results = test_Df_R_correlation()

    print(f"\nResults by symbol set:")
    print("-" * 50)
    print(f"{'Set':18s} | {'Df_box':>8s} | {'Df_info':>8s} | {'R':>8s}")
    print("-" * 50)
    for name, vals in sorted(results.items(), key=lambda x: x[1]['expected_order']):
        print(f"{name:18s} | {vals['Df_box']:8.4f} | {vals['Df_info']:8.4f} | {vals['R']:8.4f}")

    # Calculate correlation
    Df_values = [v['Df_box'] for v in results.values()]
    R_values = [v['R'] for v in results.values()]

    if len(set(Df_values)) > 1 and len(set(R_values)) > 1:
        corr = np.corrcoef(Df_values, R_values)[0, 1]
    else:
        corr = 0.0

    print(f"\nDf-R correlation: {corr:.4f}")

    if corr > 0.5:
        print("\n** VALIDATED: Strong positive correlation")
    elif corr > 0:
        print("\n*  PASS: Positive correlation")
    elif np.isnan(corr):
        print("\n?  INCONCLUSIVE: Cannot compute correlation")
    else:
        print("\nX  FALSIFIED: No positive correlation between Df and R")
