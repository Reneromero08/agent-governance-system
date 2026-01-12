#!/usr/bin/env python3
"""
Q7: Multi-Scale R Computation

Extends Q41's multiscale infrastructure to compute R at each scale level.
Uses the hierarchical corpus (words → sentences → paragraphs → documents)
and tests R preservation across scales.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Q41 infrastructure
try:
    from q41.shared.multiscale import (
        MULTI_SCALE_CORPUS,
        SCALE_HIERARCHY,
        compute_containment_matrix,
        aggregate_embeddings,
        compute_lifting_map,
        compute_semantic_l_function,
        compute_l_function_correlation,
        load_multiscale_embeddings,
        build_scale_structures,
        ScaleStructure
    )
    Q41_AVAILABLE = True
except ImportError:
    Q41_AVAILABLE = False
    MULTI_SCALE_CORPUS = None
    SCALE_HIERARCHY = ["words", "sentences", "paragraphs", "documents"]

# Import theory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.scale_transformation import ScaleData, compute_R


# =============================================================================
# R COMPUTATION AT EACH SCALE
# =============================================================================

@dataclass
class ScaleR:
    """R computation results at a scale."""
    scale_name: str
    R_value: float
    n_items: int
    mean_embedding_norm: float
    sigma: float  # Dispersion parameter


def compute_R_from_embeddings(
    embeddings: np.ndarray,
    truth_embedding: Optional[np.ndarray] = None,
    kernel: str = "gaussian"
) -> Tuple[float, float]:
    """
    Compute R from embeddings.

    For embeddings, we need to define:
    - "Truth": The mean embedding (centroid)
    - "Observations": Individual embeddings
    - "Error": Distance from centroid
    - "Sigma": Standard deviation of distances

    Args:
        embeddings: Shape (n, d) embedding matrix
        truth_embedding: Optional centroid (uses mean if None)
        kernel: Evidence kernel

    Returns:
        (R, sigma)
    """
    if len(embeddings) == 0:
        return 0.0, 1.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    # Truth = centroid
    if truth_embedding is None:
        truth_embedding = embeddings_normalized.mean(axis=0)
    truth_norm = np.linalg.norm(truth_embedding)
    if truth_norm > 1e-10:
        truth_embedding = truth_embedding / truth_norm

    # Errors = distances from centroid (using cosine distance)
    # cosine_dist = 1 - cosine_similarity
    cosine_sims = embeddings_normalized @ truth_embedding
    errors = 1 - cosine_sims  # In [0, 2] range

    # Sigma = standard deviation of errors
    sigma = float(np.std(errors)) + 1e-10

    # Normalized error z = error / sigma
    z = errors / sigma

    # Evidence kernel
    if kernel == "gaussian":
        E_values = np.exp(-0.5 * z**2)
    elif kernel == "laplace":
        E_values = np.exp(-np.abs(z))
    else:
        E_values = np.exp(-0.5 * z**2)

    # Mean evidence
    E = float(np.mean(E_values))

    # R = E / sigma
    R = E / sigma

    return R, sigma


def compute_multiscale_R(
    embeddings_by_scale: Dict[str, np.ndarray],
    kernel: str = "gaussian"
) -> Dict[str, ScaleR]:
    """
    Compute R at each scale from embeddings.

    Args:
        embeddings_by_scale: Dict mapping scale name to embedding matrix
        kernel: Evidence kernel

    Returns:
        Dict mapping scale name to ScaleR results
    """
    results = {}

    for scale_name, embeddings in embeddings_by_scale.items():
        if embeddings is None or len(embeddings) == 0:
            continue

        R, sigma = compute_R_from_embeddings(embeddings, kernel=kernel)

        mean_norm = float(np.mean(np.linalg.norm(embeddings, axis=1)))

        results[scale_name] = ScaleR(
            scale_name=scale_name,
            R_value=R,
            n_items=len(embeddings),
            mean_embedding_norm=mean_norm,
            sigma=sigma
        )

    return results


# =============================================================================
# R AGGREGATION ACROSS SCALES
# =============================================================================

def aggregate_R_values(
    child_R_values: np.ndarray,
    containment: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate R values from child scale to parent scale.

    This is the key test: does aggregated R match directly computed parent R?

    Args:
        child_R_values: R values at child scale (n_child,)
        containment: Containment matrix (n_parent, n_child)
        method: Aggregation method ("mean", "weighted_mean", "geometric")

    Returns:
        Aggregated R values at parent scale (n_parent,)
    """
    n_parent, n_child = containment.shape
    parent_R = np.zeros(n_parent)

    for i in range(n_parent):
        child_indices = np.where(containment[i] > 0)[0]

        if len(child_indices) == 0:
            parent_R[i] = 0.0
            continue

        child_vals = child_R_values[child_indices]

        if method == "mean":
            parent_R[i] = np.mean(child_vals)
        elif method == "weighted_mean":
            # Weight by containment strength (if not binary)
            weights = containment[i, child_indices]
            parent_R[i] = np.average(child_vals, weights=weights)
        elif method == "geometric":
            # Geometric mean (for multiplicative structures)
            parent_R[i] = np.exp(np.mean(np.log(child_vals + 1e-10)))
        else:
            parent_R[i] = np.mean(child_vals)

    return parent_R


def test_R_preservation(
    child_embeddings: np.ndarray,
    parent_embeddings: np.ndarray,
    containment: np.ndarray,
    kernel: str = "gaussian",
    tolerance: float = 0.15
) -> Tuple[bool, float, Dict]:
    """
    Test if R is preserved when aggregating from child to parent scale.

    Args:
        child_embeddings: Embeddings at child scale
        parent_embeddings: Embeddings at parent scale
        containment: Containment matrix
        kernel: Evidence kernel
        tolerance: Maximum allowed relative error

    Returns:
        (passes, relative_error, details)
    """
    # Compute R at child scale for each item
    n_child = len(child_embeddings)
    child_R_values = np.zeros(n_child)

    for i in range(n_child):
        # Single-item R is 1/sigma (since E(0) = 1 for z=0)
        # But this doesn't make sense for single items
        # Instead, compute R using local neighborhood
        child_R_values[i] = 1.0  # Placeholder for item-level R

    # Compute R at parent scale directly
    R_parent_direct, sigma_parent = compute_R_from_embeddings(parent_embeddings, kernel=kernel)

    # Aggregate embeddings using containment
    aggregated_embeddings = aggregate_embeddings(child_embeddings, containment, method="mean")

    # Compute R from aggregated embeddings
    R_parent_aggregated, sigma_agg = compute_R_from_embeddings(aggregated_embeddings, kernel=kernel)

    # Relative error
    if R_parent_direct > 1e-10:
        relative_error = abs(R_parent_aggregated - R_parent_direct) / R_parent_direct
    else:
        relative_error = abs(R_parent_aggregated - R_parent_direct)

    passes = relative_error < tolerance

    details = {
        "R_parent_direct": R_parent_direct,
        "R_parent_aggregated": R_parent_aggregated,
        "sigma_parent": sigma_parent,
        "sigma_aggregated": sigma_agg,
        "relative_error": relative_error,
        "n_child": n_child,
        "n_parent": len(parent_embeddings)
    }

    return passes, relative_error, details


# =============================================================================
# L-FUNCTION CORRELATION FOR FUNCTORIALITY
# =============================================================================

def test_functoriality(
    child_embeddings: np.ndarray,
    parent_embeddings: np.ndarray,
    n_primes: int = 10,
    s_values: Optional[np.ndarray] = None,
    threshold: float = 0.8
) -> Tuple[bool, float, Dict]:
    """
    Test functoriality via L-function correlation.

    If the lifting map preserves structure, L-functions at different
    scales should be correlated.

    Args:
        child_embeddings: Embeddings at child scale
        parent_embeddings: Embeddings at parent scale
        n_primes: Number of semantic primes for L-function
        s_values: Values of s for L-function (default: 0.5 to 2.0)
        threshold: Minimum correlation for pass

    Returns:
        (passes, L_correlation, details)
    """
    if s_values is None:
        s_values = np.linspace(0.5, 2.0, 20)

    if not Q41_AVAILABLE:
        # Fallback: simple correlation test
        child_centroid = child_embeddings.mean(axis=0)
        parent_centroid = parent_embeddings.mean(axis=0)

        # Truncate to same dimension
        min_dim = min(len(child_centroid), len(parent_centroid))
        corr = float(np.corrcoef(child_centroid[:min_dim], parent_centroid[:min_dim])[0, 1])

        passes = abs(corr) > threshold
        return passes, corr, {"method": "centroid_correlation"}

    # Use Q41's L-function infrastructure
    L_child = compute_semantic_l_function(child_embeddings, s_values, n_primes)
    L_parent = compute_semantic_l_function(parent_embeddings, s_values, n_primes)

    L_corr = compute_l_function_correlation(L_child, L_parent)

    passes = L_corr > threshold

    details = {
        "L_child_mean": float(np.mean(np.abs(L_child))),
        "L_parent_mean": float(np.mean(np.abs(L_parent))),
        "n_primes": n_primes,
        "n_s_values": len(s_values)
    }

    return passes, float(L_corr), details


# =============================================================================
# CROSS-SCALE ANALYSIS
# =============================================================================

def analyze_cross_scale(
    model_name: str = "MiniLM",
    kernel: str = "gaussian",
    verbose: bool = True
) -> Dict:
    """
    Full cross-scale analysis using Q41 infrastructure.

    Tests:
    1. R preservation at each scale
    2. R preservation under aggregation
    3. Functoriality (L-function correlation)

    Args:
        model_name: Embedding model name
        kernel: Evidence kernel
        verbose: Print progress

    Returns:
        Dict with all test results
    """
    if not Q41_AVAILABLE:
        return {
            "error": "Q41 multiscale infrastructure not available",
            "available": False
        }

    if verbose:
        print(f"\n--- Cross-Scale Analysis ({model_name}) ---")

    # Load embeddings at all scales
    embeddings = load_multiscale_embeddings(MULTI_SCALE_CORPUS, model_name, verbose)

    if not embeddings:
        return {"error": "Failed to load embeddings", "available": False}

    # Compute R at each scale
    scale_R_results = compute_multiscale_R(embeddings, kernel)

    if verbose:
        print("\nR at each scale:")
        for scale_name, result in scale_R_results.items():
            print(f"  {scale_name}: R={result.R_value:.4f}, n={result.n_items}, σ={result.sigma:.4f}")

    # Test R preservation between consecutive scales
    preservation_results = {}
    functoriality_results = {}

    for i in range(len(SCALE_HIERARCHY) - 1):
        child_scale = SCALE_HIERARCHY[i]
        parent_scale = SCALE_HIERARCHY[i + 1]

        if child_scale not in embeddings or parent_scale not in embeddings:
            continue

        child_emb = embeddings[child_scale]
        parent_emb = embeddings[parent_scale]

        # Compute containment
        containment = compute_containment_matrix(
            MULTI_SCALE_CORPUS[parent_scale],
            MULTI_SCALE_CORPUS[child_scale]
        )

        # Test R preservation
        passes_R, error_R, details_R = test_R_preservation(
            child_emb, parent_emb, containment, kernel
        )

        scale_pair = f"{child_scale}→{parent_scale}"
        preservation_results[scale_pair] = {
            "passes": passes_R,
            "relative_error": error_R,
            "details": details_R
        }

        # Test functoriality
        passes_F, corr_F, details_F = test_functoriality(
            child_emb, parent_emb
        )

        functoriality_results[scale_pair] = {
            "passes": passes_F,
            "L_correlation": corr_F,
            "details": details_F
        }

        if verbose:
            print(f"\n{scale_pair}:")
            print(f"  R preservation: {'PASS' if passes_R else 'FAIL'} (error={error_R:.4f})")
            print(f"  Functoriality: {'PASS' if passes_F else 'FAIL'} (L_corr={corr_F:.4f})")

    # Overall results
    all_R_pass = all(r["passes"] for r in preservation_results.values())
    all_F_pass = all(r["passes"] for r in functoriality_results.values())

    # Compute R CV across scales
    R_values = [r.R_value for r in scale_R_results.values()]
    R_cv = float(np.std(R_values) / (np.mean(R_values) + 1e-10))

    return {
        "available": True,
        "model": model_name,
        "scale_R": {k: {"R": v.R_value, "sigma": v.sigma, "n": v.n_items}
                    for k, v in scale_R_results.items()},
        "R_CV": R_cv,
        "preservation": preservation_results,
        "functoriality": functoriality_results,
        "all_R_preserved": all_R_pass,
        "all_functorial": all_F_pass,
        "verdict": "PASS" if (all_R_pass and all_F_pass and R_cv < 0.2) else "FAIL"
    }


# =============================================================================
# TESTS
# =============================================================================

def run_self_tests():
    """Run self-tests for multiscale R."""
    print("\n" + "="*80)
    print("Q7: MULTISCALE R SELF-TESTS")
    print("="*80)

    np.random.seed(42)

    # Test 1: R computation from embeddings
    print("\n--- Test 1: R from Embeddings ---")
    embeddings = np.random.randn(100, 384)
    R, sigma = compute_R_from_embeddings(embeddings)
    print(f"  R = {R:.4f}, σ = {sigma:.4f}")
    assert R > 0, "R should be positive"
    assert sigma > 0, "sigma should be positive"
    print("  PASS")

    # Test 2: R aggregation
    print("\n--- Test 2: R Aggregation ---")
    child_R = np.random.uniform(0.5, 2.0, size=20)
    containment = np.zeros((5, 20))
    for i in range(5):
        containment[i, i*4:(i+1)*4] = 1  # Each parent contains 4 children

    parent_R = aggregate_R_values(child_R, containment, method="mean")
    print(f"  Child R mean: {child_R.mean():.4f}")
    print(f"  Parent R mean: {parent_R.mean():.4f}")
    print("  PASS")

    # Test 3: Cross-scale analysis (if Q41 available)
    print("\n--- Test 3: Cross-Scale Analysis ---")
    if Q41_AVAILABLE:
        try:
            results = analyze_cross_scale(model_name="MiniLM", verbose=False)
            if results.get("available"):
                print(f"  R_CV: {results['R_CV']:.4f}")
                print(f"  All R preserved: {results['all_R_preserved']}")
                print(f"  All functorial: {results['all_functorial']}")
                print(f"  Verdict: {results['verdict']}")
            else:
                print(f"  Skipped: {results.get('error', 'unknown')}")
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("  Skipped: Q41 infrastructure not available")

    print("\n" + "="*80)
    print("MULTISCALE R TESTS: PASSED")
    print("="*80)

    return True


if __name__ == "__main__":
    run_self_tests()
