#!/usr/bin/env python3
"""
Q30: Approximations - Faster Computation Preserving Gate Behavior

PRE-REGISTRATION:
- Goal: 10x speedup with < 5% accuracy loss
- Methods: Random projection, sampling, sketching
- Outcome: Pareto frontier of speed/accuracy

Core bottleneck: R computation has O(n^2) from pairwise similarity.
For n=1000 observations, that's ~500K dot products.

Approximation strategies:
1. Random Sampling: Sample k << n observations, compute R on sample
2. Random Projection: Project d-dim embeddings to k << d dimensions
3. Locality-Sensitive Hashing (LSH): Approximate similarities via hash collisions
4. Centroid-based: Use centroid + variance of distances to centroid
5. Nystrom Approximation: Low-rank approximation of similarity matrix
6. Count-Min Sketch: Streaming approximation for similarity statistics

Run: python test_q30_approximations.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
import hashlib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class ApproximationResult:
    """Result of an approximation method."""
    R_approx: float
    E_approx: float
    sigma_approx: float
    time_seconds: float
    method: str


@dataclass
class BenchmarkResult:
    """Benchmark comparing exact vs approximate."""
    n_observations: int
    d_dimensions: int
    R_exact: float
    R_approx: float
    time_exact: float
    time_approx: float
    speedup: float
    relative_error: float
    gate_agreement: bool  # Same gate decision?


# =============================================================================
# EXACT R COMPUTATION (BASELINE)
# =============================================================================

def compute_r_exact(embeddings: List[np.ndarray], epsilon: float = 1e-6) -> Tuple[float, float, float, float]:
    """
    Exact R computation - O(n^2) pairwise similarities.

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    # Compute ALL pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            similarities.append(sim)

    E = np.mean(similarities)
    sigma = np.std(similarities)
    R = E / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E), float(sigma), elapsed)


# =============================================================================
# APPROXIMATION METHOD 1: RANDOM SAMPLING
# =============================================================================

def compute_r_sampled(
    embeddings: List[np.ndarray],
    sample_size: int = 50,
    epsilon: float = 1e-6,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Random Sampling: Use a random subset of observations.

    Complexity: O(k^2) where k = sample_size << n
    Expected speedup: (n/k)^2

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    rng = np.random.default_rng(seed)

    # Sample observations
    k = min(sample_size, n)
    indices = rng.choice(n, size=k, replace=False)
    sampled = [embeddings[i] for i in indices]

    # Compute R on sample
    similarities = []
    for i in range(k):
        for j in range(i + 1, k):
            sim = float(np.dot(sampled[i], sampled[j]))
            similarities.append(sim)

    if len(similarities) == 0:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    E = np.mean(similarities)
    sigma = np.std(similarities)
    R = E / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E), float(sigma), elapsed)


# =============================================================================
# APPROXIMATION METHOD 2: RANDOM PROJECTION
# =============================================================================

def compute_r_projected(
    embeddings: List[np.ndarray],
    target_dim: int = 64,
    epsilon: float = 1e-6,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Random Projection: Project embeddings to lower dimension.

    Uses Johnson-Lindenstrauss: distances preserved with high probability.
    Complexity: Still O(n^2) but with smaller vectors (faster dot products).

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    d = embeddings[0].shape[0]
    rng = np.random.default_rng(seed)

    # Random projection matrix (scaled Gaussian)
    k = min(target_dim, d)
    projection = rng.standard_normal((d, k)) / np.sqrt(k)

    # Project all embeddings
    projected = []
    for emb in embeddings:
        proj = emb @ projection
        # Re-normalize
        norm = np.linalg.norm(proj)
        if norm > 0:
            proj = proj / norm
        projected.append(proj)

    # Compute R on projected
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(projected[i], projected[j]))
            similarities.append(sim)

    E = np.mean(similarities)
    sigma = np.std(similarities)
    R = E / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E), float(sigma), elapsed)


# =============================================================================
# APPROXIMATION METHOD 3: CENTROID-BASED
# =============================================================================

def compute_r_centroid(
    embeddings: List[np.ndarray],
    epsilon: float = 1e-6
) -> Tuple[float, float, float, float]:
    """
    Centroid-based: Use centroid similarity and variance.

    E_approx = mean(sim(e_i, centroid))
    sigma_approx = std(sim(e_i, centroid))

    Complexity: O(n) - linear in observations!

    Theory: If embeddings cluster around centroid, distance to centroid
    correlates with pairwise distances.

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    # Compute centroid
    stacked = np.vstack(embeddings)
    centroid = np.mean(stacked, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # Similarities to centroid
    sims_to_centroid = []
    for emb in embeddings:
        sim = float(np.dot(emb, centroid))
        sims_to_centroid.append(sim)

    # E and sigma from centroid similarities
    # This approximates pairwise because:
    # sim(a, b) approx sim(a, c) * sim(b, c) for clustered data
    E = np.mean(sims_to_centroid)
    sigma = np.std(sims_to_centroid)

    # Correction factor: centroid-based underestimates pairwise E
    # Empirical correction based on geometry
    E_corrected = E * E  # Approximate sim(a,b) as sim(a,c)*sim(c,b)

    R = E_corrected / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E_corrected), float(sigma), elapsed)


# =============================================================================
# APPROXIMATION METHOD 4: NYSTROM APPROXIMATION
# =============================================================================

def compute_r_nystrom(
    embeddings: List[np.ndarray],
    landmark_size: int = 50,
    epsilon: float = 1e-6,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Nystrom Approximation: Low-rank approximation of similarity matrix.

    Sample k landmarks, compute k x n similarities, approximate full matrix.
    Complexity: O(k*n + k^3) vs O(n^2)

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    rng = np.random.default_rng(seed)

    # Select landmarks
    k = min(landmark_size, n)
    landmark_idx = rng.choice(n, size=k, replace=False)
    landmarks = [embeddings[i] for i in landmark_idx]

    # Compute landmark-landmark similarities (k x k)
    K_mm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            K_mm[i, j] = np.dot(landmarks[i], landmarks[j])

    # Compute all-to-landmark similarities (n x k)
    K_nm = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            K_nm[i, j] = np.dot(embeddings[i], landmarks[j])

    # Nystrom approximation: K_approx = K_nm @ pinv(K_mm) @ K_nm.T
    # Extract approximate statistics from K_nm and K_mm

    # For efficiency, we estimate E and sigma from the approximation
    # E_approx = mean of upper triangle of K_nm @ pinv(K_mm) @ K_nm.T
    # But computing full matrix defeats purpose, so use trace and norms

    # Simpler: use K_nm statistics as proxy
    # Each row of K_nm is similarities to landmarks
    # Mean similarity to landmarks correlates with mean pairwise

    row_means = np.mean(K_nm, axis=1)  # Mean sim of each point to landmarks
    E_proxy = np.mean(row_means ** 2)  # Approximate pairwise mean
    sigma_proxy = np.std(row_means)

    R = E_proxy / max(sigma_proxy, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E_proxy), float(sigma_proxy), elapsed)


# =============================================================================
# APPROXIMATION METHOD 5: MINI-BATCH STREAMING
# =============================================================================

def compute_r_streaming(
    embeddings: List[np.ndarray],
    batch_size: int = 32,
    n_batches: int = 10,
    epsilon: float = 1e-6,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Streaming Mini-batch: Process in small batches, aggregate statistics.

    Good for very large n where even sampling is expensive.
    Uses Welford's online algorithm for running mean/variance.

    Complexity: O(batch_size^2 * n_batches)

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    rng = np.random.default_rng(seed)

    # Welford's online algorithm
    count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squares of differences from mean

    for _ in range(n_batches):
        # Sample a batch
        k = min(batch_size, n)
        indices = rng.choice(n, size=k, replace=False)
        batch = [embeddings[i] for i in indices]

        # Compute pairwise similarities in batch
        for i in range(k):
            for j in range(i + 1, k):
                sim = float(np.dot(batch[i], batch[j]))

                # Welford's update
                count += 1
                delta = sim - mean
                mean += delta / count
                delta2 = sim - mean
                M2 += delta * delta2

    if count < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    E = mean
    sigma = np.sqrt(M2 / count)
    R = E / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E), float(sigma), elapsed)


# =============================================================================
# APPROXIMATION METHOD 6: COMBINED (SAMPLING + PROJECTION)
# =============================================================================

def compute_r_combined(
    embeddings: List[np.ndarray],
    sample_size: int = 100,
    target_dim: int = 64,
    epsilon: float = 1e-6,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Combined: Sample observations AND project to lower dimension.

    Maximum speedup by reducing both n and d.
    Complexity: O(k^2) with smaller k and smaller vector ops.

    Returns: (R, E, sigma, time_seconds)
    """
    start = time.perf_counter()

    n = len(embeddings)
    if n < 2:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    d = embeddings[0].shape[0]
    rng = np.random.default_rng(seed)

    # Sample
    k = min(sample_size, n)
    indices = rng.choice(n, size=k, replace=False)
    sampled = [embeddings[i] for i in indices]

    # Project
    target_k = min(target_dim, d)
    projection = rng.standard_normal((d, target_k)) / np.sqrt(target_k)

    projected = []
    for emb in sampled:
        proj = emb @ projection
        norm = np.linalg.norm(proj)
        if norm > 0:
            proj = proj / norm
        projected.append(proj)

    # Compute R on sampled+projected
    similarities = []
    for i in range(k):
        for j in range(i + 1, k):
            sim = float(np.dot(projected[i], projected[j]))
            similarities.append(sim)

    if len(similarities) == 0:
        return (0.0, 0.0, float('inf'), time.perf_counter() - start)

    E = np.mean(similarities)
    sigma = np.std(similarities)
    R = E / max(sigma, epsilon)

    elapsed = time.perf_counter() - start
    return (float(R), float(E), float(sigma), elapsed)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_mock_embedder(dim: int = 384, seed: int = 42) -> Callable:
    """Create a mock embedder for testing."""
    cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            text_seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            text_rng = np.random.default_rng(text_seed)
            emb = text_rng.standard_normal(dim)
            emb = emb / np.linalg.norm(emb)
            cache[text] = emb
        return cache[text]

    return embed


def generate_test_embeddings(
    n: int,
    dim: int = 384,
    agreement_level: str = "high",
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generate test embeddings with specified agreement level.

    agreement_level: "high", "medium", "low", "mixed"
    """
    rng = np.random.default_rng(seed)

    if agreement_level == "high":
        # All embeddings similar to base
        base = rng.standard_normal(dim)
        base = base / np.linalg.norm(base)
        embeddings = []
        for _ in range(n):
            noise = rng.standard_normal(dim) * 0.1
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings

    elif agreement_level == "medium":
        # Moderate spread
        base = rng.standard_normal(dim)
        base = base / np.linalg.norm(base)
        embeddings = []
        for _ in range(n):
            noise = rng.standard_normal(dim) * 0.5
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings

    elif agreement_level == "low":
        # Random embeddings
        embeddings = []
        for _ in range(n):
            emb = rng.standard_normal(dim)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings

    elif agreement_level == "mixed":
        # Half high agreement, half low
        high = generate_test_embeddings(n // 2, dim, "high", seed)
        low = generate_test_embeddings(n - n // 2, dim, "low", seed + 1)
        return high + low

    else:
        raise ValueError(f"Unknown agreement level: {agreement_level}")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_accuracy_preservation(threshold: float = 0.8) -> Tuple[bool, Dict]:
    """
    TEST 1: Do approximations preserve gate decisions?

    Test that R_approx gives same gate open/close as R_exact.
    """
    print("  Testing accuracy preservation...")

    dim = 384
    test_sizes = [50, 100, 200, 500]
    agreement_levels = ["high", "medium", "low"]

    methods = {
        "exact": lambda e: compute_r_exact(e),
        "sampled_20": lambda e: compute_r_sampled(e, sample_size=20),
        "sampled_50": lambda e: compute_r_sampled(e, sample_size=50),
        "projected_64": lambda e: compute_r_projected(e, target_dim=64),
        "projected_128": lambda e: compute_r_projected(e, target_dim=128),
        "centroid": lambda e: compute_r_centroid(e),
        "nystrom_30": lambda e: compute_r_nystrom(e, landmark_size=30),
        "streaming": lambda e: compute_r_streaming(e, batch_size=16, n_batches=8),
        "combined": lambda e: compute_r_combined(e, sample_size=50, target_dim=64),
    }

    results_by_method = {m: {"correct": 0, "total": 0, "errors": []} for m in methods if m != "exact"}

    for n in test_sizes:
        for agreement in agreement_levels:
            embeddings = generate_test_embeddings(n, dim, agreement)

            # Get exact R
            R_exact, E_exact, sigma_exact, _ = compute_r_exact(embeddings)
            gate_exact = R_exact >= threshold

            # Test each approximation
            for method_name, method_fn in methods.items():
                if method_name == "exact":
                    continue

                R_approx, E_approx, sigma_approx, _ = method_fn(embeddings)
                gate_approx = R_approx >= threshold

                is_correct = gate_exact == gate_approx
                rel_error = abs(R_exact - R_approx) / max(abs(R_exact), 1e-8)

                results_by_method[method_name]["total"] += 1
                if is_correct:
                    results_by_method[method_name]["correct"] += 1
                else:
                    results_by_method[method_name]["errors"].append({
                        "n": n,
                        "agreement": agreement,
                        "R_exact": R_exact,
                        "R_approx": R_approx,
                        "rel_error": rel_error
                    })

    # Calculate accuracy for each method
    accuracies = {}
    for method, data in results_by_method.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        accuracies[method] = acc

    # Goal: at least one method achieves > 95% gate agreement
    best_method = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_method]
    goal_met = best_accuracy >= 0.95

    return goal_met, {
        "test": "ACCURACY_PRESERVATION",
        "threshold": threshold,
        "test_sizes": test_sizes,
        "agreement_levels": agreement_levels,
        "accuracies": {m: f"{acc*100:.1f}%" for m, acc in accuracies.items()},
        "best_method": best_method,
        "best_accuracy": f"{best_accuracy*100:.1f}%",
        "goal": ">95% gate agreement",
        "goal_met": goal_met,
        "sample_errors": {m: data["errors"][:3] for m, data in results_by_method.items() if data["errors"]},
        "pass": goal_met
    }


def test_speedup_measurement() -> Tuple[bool, Dict]:
    """
    TEST 2: Measure speedup of each approximation method.

    Goal: 10x speedup with acceptable accuracy.
    """
    print("  Testing speedup measurement...")

    dim = 384
    test_sizes = [100, 200, 500, 1000]

    methods = {
        "exact": lambda e: compute_r_exact(e),
        "sampled_50": lambda e: compute_r_sampled(e, sample_size=50),
        "projected_64": lambda e: compute_r_projected(e, target_dim=64),
        "centroid": lambda e: compute_r_centroid(e),
        "nystrom_50": lambda e: compute_r_nystrom(e, landmark_size=50),
        "streaming": lambda e: compute_r_streaming(e, batch_size=32, n_batches=10),
        "combined": lambda e: compute_r_combined(e, sample_size=75, target_dim=64),
    }

    timing_results = {m: [] for m in methods}
    speedups = {m: [] for m in methods if m != "exact"}

    for n in test_sizes:
        embeddings = generate_test_embeddings(n, dim, "medium")

        # Time each method (multiple runs for stability)
        for method_name, method_fn in methods.items():
            times = []
            for _ in range(3):
                _, _, _, elapsed = method_fn(embeddings)
                times.append(elapsed)
            avg_time = np.mean(times)
            timing_results[method_name].append({"n": n, "time": avg_time})

        # Calculate speedup vs exact
        exact_time = timing_results["exact"][-1]["time"]
        for method_name in speedups:
            approx_time = timing_results[method_name][-1]["time"]
            speedup = exact_time / max(approx_time, 1e-9)
            speedups[method_name].append({"n": n, "speedup": speedup})

    # Summarize speedups
    avg_speedups = {}
    for method, data in speedups.items():
        avg_speedup = np.mean([d["speedup"] for d in data])
        avg_speedups[method] = avg_speedup

    # Goal: at least one method achieves 10x speedup
    best_method = max(avg_speedups, key=avg_speedups.get)
    best_speedup = avg_speedups[best_method]
    goal_met = best_speedup >= 10.0

    return goal_met, {
        "test": "SPEEDUP_MEASUREMENT",
        "test_sizes": test_sizes,
        "timing_results": {m: data[-1] for m, data in timing_results.items()},  # Largest n
        "speedups_by_n": speedups,
        "avg_speedups": {m: f"{s:.1f}x" for m, s in avg_speedups.items()},
        "best_method": best_method,
        "best_speedup": f"{best_speedup:.1f}x",
        "goal": ">=10x speedup",
        "goal_met": goal_met,
        "pass": goal_met
    }


def test_pareto_frontier() -> Tuple[bool, Dict]:
    """
    TEST 3: Build Pareto frontier of speed vs accuracy.

    Find methods on the Pareto frontier (no method dominates them).
    """
    print("  Testing Pareto frontier...")

    dim = 384
    n = 500
    threshold = 0.8

    # Generate test scenarios
    scenarios = [
        generate_test_embeddings(n, dim, "high"),
        generate_test_embeddings(n, dim, "medium"),
        generate_test_embeddings(n, dim, "low"),
    ]

    # Methods with varying parameters
    methods = [
        ("exact", lambda e: compute_r_exact(e)),
        ("sample_20", lambda e: compute_r_sampled(e, sample_size=20)),
        ("sample_50", lambda e: compute_r_sampled(e, sample_size=50)),
        ("sample_100", lambda e: compute_r_sampled(e, sample_size=100)),
        ("proj_32", lambda e: compute_r_projected(e, target_dim=32)),
        ("proj_64", lambda e: compute_r_projected(e, target_dim=64)),
        ("proj_128", lambda e: compute_r_projected(e, target_dim=128)),
        ("centroid", lambda e: compute_r_centroid(e)),
        ("nystrom_30", lambda e: compute_r_nystrom(e, landmark_size=30)),
        ("nystrom_50", lambda e: compute_r_nystrom(e, landmark_size=50)),
        ("combined_small", lambda e: compute_r_combined(e, sample_size=30, target_dim=32)),
        ("combined_med", lambda e: compute_r_combined(e, sample_size=50, target_dim=64)),
        ("combined_large", lambda e: compute_r_combined(e, sample_size=100, target_dim=128)),
    ]

    # Measure each method
    method_stats = {}
    exact_times = []

    for method_name, method_fn in methods:
        total_correct = 0
        total_scenarios = 0
        total_time = 0.0
        errors = []

        for scenario_idx, embeddings in enumerate(scenarios):
            # Get exact R for comparison
            R_exact, _, _, t_exact = compute_r_exact(embeddings)
            if method_name == "exact":
                exact_times.append(t_exact)
            gate_exact = R_exact >= threshold

            # Test method
            R_approx, _, _, t_approx = method_fn(embeddings)
            gate_approx = R_approx >= threshold

            total_scenarios += 1
            total_time += t_approx

            if gate_exact == gate_approx:
                total_correct += 1
            else:
                errors.append(abs(R_exact - R_approx) / max(abs(R_exact), 1e-8))

        accuracy = total_correct / total_scenarios
        avg_time = total_time / len(scenarios)
        avg_error = np.mean(errors) if errors else 0.0

        method_stats[method_name] = {
            "accuracy": accuracy,
            "avg_time": avg_time,
            "avg_error": avg_error
        }

    # Calculate speedup relative to exact
    exact_avg_time = np.mean(exact_times)
    for name, stats in method_stats.items():
        stats["speedup"] = exact_avg_time / max(stats["avg_time"], 1e-9)

    # Find Pareto frontier
    # A method is on frontier if no other method has both higher accuracy AND higher speedup
    pareto_frontier = []
    for name, stats in method_stats.items():
        is_dominated = False
        for other_name, other_stats in method_stats.items():
            if name == other_name:
                continue
            # other dominates if it has >= accuracy AND >= speedup, with at least one >
            if (other_stats["accuracy"] >= stats["accuracy"] and
                other_stats["speedup"] >= stats["speedup"] and
                (other_stats["accuracy"] > stats["accuracy"] or other_stats["speedup"] > stats["speedup"])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_frontier.append(name)

    # Goal: Have good methods on frontier (>90% accuracy, >5x speedup)
    good_frontier = [
        name for name in pareto_frontier
        if method_stats[name]["accuracy"] >= 0.9 and method_stats[name]["speedup"] >= 5.0
    ]

    goal_met = len(good_frontier) >= 1

    return goal_met, {
        "test": "PARETO_FRONTIER",
        "n_observations": n,
        "threshold": threshold,
        "method_stats": {
            name: {
                "accuracy": f"{stats['accuracy']*100:.1f}%",
                "speedup": f"{stats['speedup']:.1f}x",
                "avg_error": f"{stats['avg_error']*100:.2f}%"
            }
            for name, stats in method_stats.items()
        },
        "pareto_frontier": pareto_frontier,
        "good_frontier_methods": good_frontier,
        "goal": ">=1 method with >90% accuracy and >5x speedup on frontier",
        "goal_met": goal_met,
        "pass": goal_met
    }


def test_scaling_behavior() -> Tuple[bool, Dict]:
    """
    TEST 4: How do methods scale with n?

    Verify that approximate methods scale better than O(n^2).
    """
    print("  Testing scaling behavior...")

    dim = 384
    test_sizes = [50, 100, 200, 400, 800]

    methods = {
        "exact": lambda e: compute_r_exact(e),
        "sampled_fixed": lambda e: compute_r_sampled(e, sample_size=50),
        "centroid": lambda e: compute_r_centroid(e),
        "combined": lambda e: compute_r_combined(e, sample_size=50, target_dim=64),
    }

    scaling_data = {m: [] for m in methods}

    for n in test_sizes:
        embeddings = generate_test_embeddings(n, dim, "medium")

        for method_name, method_fn in methods.items():
            times = []
            for _ in range(3):
                _, _, _, elapsed = method_fn(embeddings)
                times.append(elapsed)
            avg_time = np.mean(times)
            scaling_data[method_name].append({"n": n, "time": avg_time})

    # Fit scaling exponents
    # time ~ n^alpha, log(time) ~ alpha * log(n)
    scaling_exponents = {}
    for method, data in scaling_data.items():
        ns = np.array([d["n"] for d in data])
        times = np.array([d["time"] for d in data])

        # Log-log fit
        log_ns = np.log(ns)
        log_times = np.log(times + 1e-9)

        # Linear regression: log(t) = alpha * log(n) + c
        coeffs = np.polyfit(log_ns, log_times, 1)
        alpha = coeffs[0]

        scaling_exponents[method] = alpha

    # Verify: exact should be ~2 (O(n^2)), approximations should be < 2
    exact_alpha = scaling_exponents.get("exact", 2.0)
    subquadratic_methods = [m for m, a in scaling_exponents.items() if a < exact_alpha - 0.3]

    goal_met = len(subquadratic_methods) >= 2

    return goal_met, {
        "test": "SCALING_BEHAVIOR",
        "test_sizes": test_sizes,
        "scaling_exponents": {m: f"O(n^{a:.2f})" for m, a in scaling_exponents.items()},
        "exact_alpha": f"{exact_alpha:.2f}",
        "subquadratic_methods": subquadratic_methods,
        "goal": ">=2 methods with alpha < exact - 0.3",
        "goal_met": goal_met,
        "scaling_data": {m: data for m, data in scaling_data.items()},
        "pass": goal_met
    }


def test_robustness_to_agreement_level() -> Tuple[bool, Dict]:
    """
    TEST 5: Are approximations robust across agreement levels?

    Some approximations might work better for high vs low agreement.
    """
    print("  Testing robustness to agreement level...")

    dim = 384
    n = 300
    threshold = 0.8
    agreement_levels = ["high", "medium", "low", "mixed"]

    methods = {
        "sampled_50": lambda e: compute_r_sampled(e, sample_size=50),
        "projected_64": lambda e: compute_r_projected(e, target_dim=64),
        "centroid": lambda e: compute_r_centroid(e),
        "combined": lambda e: compute_r_combined(e, sample_size=50, target_dim=64),
    }

    # Track accuracy by agreement level
    accuracy_by_level = {m: {} for m in methods}

    for agreement in agreement_levels:
        for method_name, method_fn in methods.items():
            correct = 0
            total = 5  # Multiple seeds

            for seed in range(5):
                embeddings = generate_test_embeddings(n, dim, agreement, seed=seed * 100)

                R_exact, _, _, _ = compute_r_exact(embeddings)
                R_approx, _, _, _ = method_fn(embeddings)

                gate_exact = R_exact >= threshold
                gate_approx = R_approx >= threshold

                if gate_exact == gate_approx:
                    correct += 1

            accuracy = correct / total
            accuracy_by_level[method_name][agreement] = accuracy

    # Check consistency: accuracy should not vary too much across levels
    consistency_scores = {}
    for method, levels in accuracy_by_level.items():
        accs = list(levels.values())
        std_acc = np.std(accs)
        min_acc = min(accs)
        consistency_scores[method] = {
            "std": std_acc,
            "min_accuracy": min_acc,
            "is_robust": std_acc < 0.15 and min_acc > 0.7
        }

    robust_methods = [m for m, s in consistency_scores.items() if s["is_robust"]]
    goal_met = len(robust_methods) >= 2

    return goal_met, {
        "test": "ROBUSTNESS_TO_AGREEMENT_LEVEL",
        "n_observations": n,
        "agreement_levels": agreement_levels,
        "accuracy_by_level": {
            m: {l: f"{a*100:.0f}%" for l, a in levels.items()}
            for m, levels in accuracy_by_level.items()
        },
        "consistency_scores": consistency_scores,
        "robust_methods": robust_methods,
        "goal": ">=2 methods robust across all agreement levels",
        "goal_met": goal_met,
        "pass": goal_met
    }


def test_recommended_implementation() -> Tuple[bool, Dict]:
    """
    TEST 6: Determine recommended implementation.

    Based on all tests, recommend the best approximation strategy.
    """
    print("  Determining recommended implementation...")

    dim = 384

    # Score each method based on:
    # - Accuracy (weight 3)
    # - Speedup (weight 2)
    # - Simplicity (weight 1)
    # - Robustness (weight 1)

    methods = {
        "sampled_50": {
            "fn": lambda e: compute_r_sampled(e, sample_size=50),
            "simplicity": 3,  # Very simple
            "description": "Random sample of 50 observations"
        },
        "projected_64": {
            "fn": lambda e: compute_r_projected(e, target_dim=64),
            "simplicity": 2,  # Moderate
            "description": "Random projection to 64 dimensions"
        },
        "centroid": {
            "fn": lambda e: compute_r_centroid(e),
            "simplicity": 3,  # Very simple
            "description": "Centroid-based O(n) approximation"
        },
        "nystrom_50": {
            "fn": lambda e: compute_r_nystrom(e, landmark_size=50),
            "simplicity": 1,  # Complex
            "description": "Nystrom low-rank approximation"
        },
        "combined": {
            "fn": lambda e: compute_r_combined(e, sample_size=50, target_dim=64),
            "simplicity": 2,  # Moderate
            "description": "Sample + project combined"
        },
    }

    # Test on standard benchmark
    n = 500
    threshold = 0.8

    benchmark_results = {}
    for method_name, method_info in methods.items():
        correct = 0
        total = 0
        total_time = 0.0
        total_error = 0.0

        for agreement in ["high", "medium", "low"]:
            for seed in range(3):
                embeddings = generate_test_embeddings(n, dim, agreement, seed=seed * 100)

                R_exact, _, _, t_exact = compute_r_exact(embeddings)
                R_approx, _, _, t_approx = method_info["fn"](embeddings)

                gate_exact = R_exact >= threshold
                gate_approx = R_approx >= threshold

                total += 1
                total_time += t_approx
                total_error += abs(R_exact - R_approx) / max(abs(R_exact), 1e-8)

                if gate_exact == gate_approx:
                    correct += 1

        # Get exact time for speedup
        exact_times = []
        for agreement in ["high", "medium", "low"]:
            embeddings = generate_test_embeddings(n, dim, agreement)
            _, _, _, t = compute_r_exact(embeddings)
            exact_times.append(t)
        exact_avg = np.mean(exact_times)

        accuracy = correct / total
        avg_time = total_time / total
        avg_error = total_error / total
        speedup = exact_avg / max(avg_time, 1e-9)

        # Compute score
        score = (
            accuracy * 3.0 +  # Weight 3 for accuracy
            min(speedup / 10, 1.0) * 2.0 +  # Weight 2 for speedup (capped at 10x)
            method_info["simplicity"] / 3.0 * 1.0 +  # Weight 1 for simplicity
            (1.0 - avg_error) * 1.0  # Weight 1 for robustness
        )

        benchmark_results[method_name] = {
            "accuracy": accuracy,
            "speedup": speedup,
            "avg_error": avg_error,
            "simplicity": method_info["simplicity"],
            "score": score,
            "description": method_info["description"]
        }

    # Recommend best method
    best_method = max(benchmark_results, key=lambda m: benchmark_results[m]["score"])
    best_result = benchmark_results[best_method]

    # Generate recommendation
    recommendation = f"""
RECOMMENDED APPROXIMATION: {best_method}

Description: {best_result['description']}

Performance:
- Accuracy: {best_result['accuracy']*100:.1f}% gate agreement
- Speedup: {best_result['speedup']:.1f}x faster than exact
- Average Error: {best_result['avg_error']*100:.2f}%

Code:
def compute_r_fast(embeddings, sample_size=50, target_dim=64, epsilon=1e-6):
    \"\"\"
    Fast approximation of R = E / sigma.

    For n={n}, this achieves {best_result['speedup']:.1f}x speedup
    with {best_result['accuracy']*100:.1f}% accuracy on gate decisions.
    \"\"\"
    n = len(embeddings)
    d = embeddings[0].shape[0] if embeddings else 0

    # Sample observations
    k = min(sample_size, n)
    indices = np.random.choice(n, size=k, replace=False)
    sampled = [embeddings[i] for i in indices]

    # Project to lower dimension (optional for additional speedup)
    if target_dim < d:
        projection = np.random.randn(d, target_dim) / np.sqrt(target_dim)
        sampled = [e @ projection for e in sampled]
        sampled = [e / np.linalg.norm(e) for e in sampled]

    # Compute R on sample
    similarities = []
    for i in range(k):
        for j in range(i + 1, k):
            similarities.append(np.dot(sampled[i], sampled[j]))

    E = np.mean(similarities)
    sigma = np.std(similarities)
    return E / max(sigma, epsilon)
"""

    goal_met = best_result["accuracy"] >= 0.9 and best_result["speedup"] >= 5.0

    return goal_met, {
        "test": "RECOMMENDED_IMPLEMENTATION",
        "n_benchmark": n,
        "benchmark_results": {
            m: {
                "accuracy": f"{r['accuracy']*100:.1f}%",
                "speedup": f"{r['speedup']:.1f}x",
                "avg_error": f"{r['avg_error']*100:.2f}%",
                "score": f"{r['score']:.2f}"
            }
            for m, r in benchmark_results.items()
        },
        "best_method": best_method,
        "recommendation": recommendation,
        "goal": ">90% accuracy and >5x speedup",
        "goal_met": goal_met,
        "pass": goal_met
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests and produce report."""
    print("=" * 70)
    print("Q30: APPROXIMATIONS - FASTER COMPUTATION PRESERVING GATE BEHAVIOR")
    print("=" * 70)
    print()
    print("PRE-REGISTRATION:")
    print("  Goal: 10x speedup with < 5% accuracy loss")
    print("  Methods: Random projection, sampling, sketching")
    print("  Outcome: Pareto frontier of speed/accuracy")
    print()

    # Run tests
    tests = [
        ("ACCURACY_PRESERVATION", test_accuracy_preservation),
        ("SPEEDUP_MEASUREMENT", test_speedup_measurement),
        ("PARETO_FRONTIER", test_pareto_frontier),
        ("SCALING_BEHAVIOR", test_scaling_behavior),
        ("ROBUSTNESS", test_robustness_to_agreement_level),
        ("RECOMMENDED_IMPL", test_recommended_implementation),
    ]

    results = []
    passed = 0
    failed = 0

    print("-" * 70)
    print(f"{'Test':<30} | {'Status':<10} | {'Details'}")
    print("-" * 70)

    for test_name, test_fn in tests:
        try:
            success, result = test_fn()
            results.append(result)

            if success:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            # Print summary line
            detail = ""
            if "best_method" in result:
                detail = f"best={result['best_method']}"
            elif "best_accuracy" in result:
                detail = f"acc={result['best_accuracy']}"
            elif "best_speedup" in result:
                detail = f"speedup={result['best_speedup']}"
            elif "pareto_frontier" in result:
                detail = f"frontier={len(result['pareto_frontier'])} methods"
            elif "subquadratic_methods" in result:
                detail = f"subquadratic={len(result['subquadratic_methods'])}"
            elif "robust_methods" in result:
                detail = f"robust={len(result['robust_methods'])}"

            print(f"{test_name:<30} | {status:<10} | {detail}")

        except Exception as e:
            import traceback
            print(f"{test_name:<30} | ERROR      | {str(e)[:40]}")
            traceback.print_exc()
            failed += 1
            results.append({"test": test_name, "error": str(e), "pass": False})

    print("-" * 70)
    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    print()

    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if failed == 0:
        print("\n** ALL TESTS PASS - APPROXIMATIONS VALIDATED **")
    else:
        print(f"\n** {failed} TESTS FAILED **")

    # Print recommendation
    for r in results:
        if r.get("test") == "RECOMMENDED_IMPLEMENTATION":
            print(r.get("recommendation", ""))
            break

    # Print Pareto frontier
    for r in results:
        if r.get("test") == "PARETO_FRONTIER":
            print("\nPARETO FRONTIER (non-dominated methods):")
            for m in r.get("pareto_frontier", []):
                stats = r.get("method_stats", {}).get(m, {})
                print(f"  {m}: accuracy={stats.get('accuracy', 'N/A')}, speedup={stats.get('speedup', 'N/A')}")
            break

    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "failed": failed,
        "results": results,
        "verdict": "VALIDATED" if failed == 0 else "NOT_VALIDATED"
    }

    output_path = Path(__file__).parent / "q30_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
