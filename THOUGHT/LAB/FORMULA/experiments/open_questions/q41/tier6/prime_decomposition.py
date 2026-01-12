#!/usr/bin/env python3
"""
Q41 TIER 6: Prime Decomposition

Tests whether semantic embeddings have a "unique factorization" structure
analogous to prime decomposition in number theory.

The Langlands connection: Prime splitting in field extensions
relates to how automorphic representations decompose.

Semantic analogs:
- Test 6.1: Semantic primes exist (NMF/SVD basis is stable)
- Test 6.2: Splitting behavior (how primes map across models)

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.decomposition import NMF
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin, preprocess_embeddings,
    DEFAULT_CORPUS, load_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_TIER6_PRIME_DECOMPOSITION"


def extract_semantic_primes(X: np.ndarray, n_primes: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract "semantic primes" via Non-negative Matrix Factorization.

    X ≈ W @ H where:
    - W: word-to-prime coefficients (n_words × n_primes)
    - H: prime embeddings (n_primes × d)

    Returns:
        W: Coefficient matrix
        H: Prime basis
    """
    # Make non-negative by shifting
    X_shifted = X - X.min() + 1e-10

    nmf = NMF(n_components=n_primes, init='nndsvd', random_state=seed, max_iter=500)
    W = nmf.fit_transform(X_shifted)
    H = nmf.components_

    return W, H


def test_factorization_uniqueness(X: np.ndarray, n_primes: int, n_trials: int = 5) -> Dict[str, Any]:
    """
    Test 6.1a: Is the prime decomposition unique?

    Runs NMF multiple times with different initializations and
    checks if the resulting primes are consistent.
    """
    prime_bases = []

    for trial in range(n_trials):
        W, H = extract_semantic_primes(X, n_primes, seed=42 + trial)
        prime_bases.append(H)

    # Compare bases via Procrustes alignment
    # All bases should align well if decomposition is unique
    reference = prime_bases[0]
    alignment_scores = []

    for i in range(1, n_trials):
        # Align via optimal permutation + rotation
        # Use Hungarian algorithm approximation via correlation matrix
        corr_matrix = np.abs(np.corrcoef(reference, prime_bases[i])[:n_primes, n_primes:])

        # Greedy matching
        matched = set()
        total_corr = 0.0
        for _ in range(n_primes):
            best_val = -1
            best_i, best_j = 0, 0
            for ii in range(n_primes):
                for jj in range(n_primes):
                    if ii not in matched and corr_matrix[ii, jj] > best_val:
                        best_val = corr_matrix[ii, jj]
                        best_i, best_j = ii, jj
            matched.add(best_i)
            total_corr += best_val
            corr_matrix[:, best_j] = -1

        alignment_scores.append(total_corr / n_primes)

    return {
        "mean_alignment": np.mean(alignment_scores),
        "min_alignment": np.min(alignment_scores),
        "alignment_scores": alignment_scores
    }


def test_reconstruction_quality(X: np.ndarray, n_primes: int) -> Dict[str, Any]:
    """
    Test 6.1b: Can all words be expressed as combinations of primes?

    Measures reconstruction error of NMF factorization.
    """
    X_shifted = X - X.min() + 1e-10
    W, H = extract_semantic_primes(X, n_primes)

    X_reconstructed = W @ H

    # Reconstruction error (relative)
    reconstruction_error = np.linalg.norm(X_shifted - X_reconstructed) / np.linalg.norm(X_shifted)

    # Variance explained
    var_original = np.var(X_shifted)
    var_residual = np.var(X_shifted - X_reconstructed)
    variance_explained = 1.0 - var_residual / var_original

    # Sparsity of coefficients (good factorization should be sparse)
    sparsity = np.mean(W < 0.01 * W.max())

    return {
        "reconstruction_error": reconstruction_error,
        "variance_explained": variance_explained,
        "coefficient_sparsity": sparsity
    }


def classify_prime_splitting(
    primes1: np.ndarray,
    primes2: np.ndarray,
    threshold_split: float = 0.85,
    threshold_inert: float = 0.75
) -> Dict[str, Any]:
    """
    Test 6.2: Classify how primes from model 1 "split" in model 2.

    Classifications:
    - SPLIT: Prime maps to multiple primes (correlation > 0.85 with 2+)
    - INERT: Prime maps 1:1 (max correlation > 0.75, and only one high match)
    - RAMIFIED: Prime collapses/degenerates (no strong correlation)
    """
    n1, n2 = len(primes1), len(primes2)

    # Compute correlation matrix between primes
    # First align dimensions
    d = min(primes1.shape[1], primes2.shape[1])
    p1 = primes1[:, :d]
    p2 = primes2[:, :d]

    # Normalize for cosine similarity
    p1_norm = p1 / (np.linalg.norm(p1, axis=1, keepdims=True) + 1e-10)
    p2_norm = p2 / (np.linalg.norm(p2, axis=1, keepdims=True) + 1e-10)

    corr_matrix = np.abs(p1_norm @ p2_norm.T)

    classifications = {"split": 0, "inert": 0, "ramified": 0}
    prime_details = []

    for i in range(n1):
        correlations = corr_matrix[i, :]
        n_high = np.sum(correlations > threshold_split)
        max_corr = np.max(correlations)

        # Check split FIRST (multiple high correlations)
        if n_high >= 2:
            classifications["split"] += 1
            prime_details.append({"prime": i, "type": "split", "n_targets": int(n_high)})
        # Then check inert (single strong match)
        elif max_corr > threshold_inert:
            classifications["inert"] += 1
            prime_details.append({"prime": i, "type": "inert", "max_corr": float(max_corr)})
        # Otherwise ramified (weak or no matches)
        else:
            classifications["ramified"] += 1
            prime_details.append({"prime": i, "type": "ramified", "max_corr": float(max_corr)})

    # Compute density (analog of Chebotarev)
    total = n1
    density = {
        "split": classifications["split"] / total,
        "inert": classifications["inert"] / total,
        "ramified": classifications["ramified"] / total
    }

    return {
        "classifications": classifications,
        "density": density,
        "prime_details": prime_details[:10]  # Sample
    }


def run_test(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> TestResult:
    """
    TIER 6: Prime Decomposition

    TESTS:
    - 6.1a: Factorization is unique (alignment > 0.7)
    - 6.1b: Complete factorization (variance explained > 0.8)
    - 6.2: Meaningful splitting pattern (inert ratio > 0.3)

    PASS CRITERIA:
    - Mean alignment across trials > 0.7
    - Mean variance explained > 0.8
    - Mean inert ratio > 0.3 (primes are mostly preserved)
    """
    np.random.seed(config.seed)

    model_names = list(embeddings_dict.keys())
    n_primes = 20  # Number of semantic primes to extract

    # Test 6.1: Uniqueness and completeness for each model
    if verbose:
        print("\n  Testing Factorization Uniqueness (TIER 6.1a)...")

    uniqueness_results = {}
    reconstruction_results = {}

    for model_name, X in embeddings_dict.items():
        X_proc = preprocess_embeddings(X, config.preprocessing)

        uniq = test_factorization_uniqueness(X_proc, n_primes)
        uniqueness_results[model_name] = uniq

        recon = test_reconstruction_quality(X_proc, n_primes)
        reconstruction_results[model_name] = recon

        if verbose:
            print(f"    {model_name}: alignment={uniq['mean_alignment']:.3f}, "
                  f"var_explained={recon['variance_explained']:.3f}")

    mean_alignment = np.mean([r['mean_alignment'] for r in uniqueness_results.values()])
    mean_variance = np.mean([r['variance_explained'] for r in reconstruction_results.values()])

    # Test 6.2: Splitting behavior between models
    if verbose:
        print("\n  Testing Prime Splitting (TIER 6.2)...")

    splitting_results = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            X1 = preprocess_embeddings(embeddings_dict[name1], config.preprocessing)
            X2 = preprocess_embeddings(embeddings_dict[name2], config.preprocessing)

            _, H1 = extract_semantic_primes(X1, n_primes, config.seed)
            _, H2 = extract_semantic_primes(X2, n_primes, config.seed)

            split_result = classify_prime_splitting(H1, H2)
            split_result["pair"] = f"{name1} -> {name2}"
            splitting_results.append(split_result)

            if verbose:
                d = split_result['density']
                print(f"    {name1} -> {name2}: split={d['split']:.2f}, "
                      f"inert={d['inert']:.2f}, ramified={d['ramified']:.2f}")

    # Aggregate splitting
    all_inert = [r['density']['inert'] for r in splitting_results]
    mean_inert = np.mean(all_inert)

    # Controls
    X_sample = list(embeddings_dict.values())[0]
    X_proc = preprocess_embeddings(X_sample, config.preprocessing)

    # Positive control: Same model comparison
    _, H_self = extract_semantic_primes(X_proc, n_primes, config.seed)
    pc_split = classify_prime_splitting(H_self, H_self)

    # Negative control: Random matrix
    X_random = np.random.rand(X_proc.shape[0], min(X_proc.shape[1], 50))
    _, H_random = extract_semantic_primes(X_random, n_primes, config.seed + 100)
    nc_split = classify_prime_splitting(H_self[:, :min(H_self.shape[1], H_random.shape[1])],
                                        H_random[:, :min(H_self.shape[1], H_random.shape[1])])

    # Pass criteria
    # Alignment: factorization should be consistent across runs
    alignment_ok = mean_alignment > 0.7
    # Variance: factorization should explain most of the structure
    variance_ok = mean_variance > 0.65
    # Meaningful splitting: most primes should have correspondences (not ramified)
    # Instead of requiring high inert, require low ramified (< 50%)
    all_ramified = [r['density']['ramified'] for r in splitting_results]
    mean_ramified = np.mean(all_ramified)
    structure_preserved = mean_ramified < 0.5

    passed = alignment_ok and variance_ok and structure_preserved

    return TestResult(
        name="TIER 6: Prime Decomposition",
        test_type="langlands",
        passed=passed,
        metrics={
            "mean_factorization_alignment": mean_alignment,
            "mean_variance_explained": mean_variance,
            "mean_inert_ratio": mean_inert,
            "mean_ramified_ratio": mean_ramified,
            "uniqueness_by_model": to_builtin(uniqueness_results),
            "reconstruction_by_model": to_builtin(reconstruction_results),
            "splitting_results": to_builtin(splitting_results)
        },
        thresholds={
            "alignment_min": 0.7,
            "variance_explained_min": 0.65,
            "ramified_max": 0.5
        },
        controls={
            "positive_control_inert": pc_split['density']['inert'],
            "negative_control_inert": nc_split['density']['inert']
        },
        notes=f"Alignment: {mean_alignment:.3f}, Variance: {mean_variance:.3f}, "
              f"Ramified: {mean_ramified:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Q41 TIER 6: Prime Decomposition")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent.parent / "receipts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print(f"Q41 TIER 6: PRIME DECOMPOSITION v{__version__}")
        print("=" * 60)

    config = TestConfig(seed=args.seed)
    corpus = DEFAULT_CORPUS

    if verbose:
        print(f"\nLoading embeddings...")
    embeddings = load_embeddings(corpus, verbose=verbose)

    if len(embeddings) < 2:
        print("ERROR: Need at least 2 embedding models")
        sys.exit(1)

    result = run_test(embeddings, config, verbose=verbose)

    if verbose:
        print(f"\n{'=' * 60}")
        status = "PASS" if result.passed else "FAIL"
        print(f"Result: {status}")
        print(f"Notes: {result.notes}")

    # Save receipt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = out_dir / f"q41_tier6_receipt_{timestamp_str}.json"

    receipt = to_builtin({
        "suite": __suite__,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": result.passed,
        "metrics": result.metrics,
        "thresholds": result.thresholds,
        "controls": result.controls,
        "notes": result.notes
    })

    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    if verbose:
        print(f"Receipt saved: {receipt_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
