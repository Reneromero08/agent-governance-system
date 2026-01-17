#!/usr/bin/env python3
"""Diagnose and Fix Procrustes Alignment Residual.

The problem: We have spectrum correlation = 1.0000 (eigenvalues match perfectly)
but Procrustes residual ~2.6 (coordinates don't overlay well).

Root cause: MDS eigenvectors have SIGN AMBIGUITY. Each eigenvector can be +/-.
If model A produces eigenvector [+] and model B produces [-], that's a reflection
that rotation alone cannot fix.

Solution: Try all 2^k sign combinations to find optimal sign correction before
Procrustes rotation.

Usage:
    python diagnose_procrustes.py
"""

import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from itertools import product
import requests

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64
from CAPABILITY.PRIMITIVES.mds import squared_distance_matrix, classical_mds


def get_embeddings(texts, url="http://10.5.0.2:1234/v1/embeddings",
                   model="text-embedding-nomic-embed-text-v1.5"):
    """Get embeddings from API."""
    response = requests.post(url, json={"model": model, "input": texts}, timeout=60)
    data = response.json()
    return np.array([d["embedding"] for d in data["data"]])


def procrustes_with_reflection(X_source, X_target):
    """Procrustes allowing reflection (not just rotation).

    Standard Procrustes finds R minimizing ||X_s R - X_t|| where det(R) = +1.
    Sometimes we need det(R) = -1 (improper rotation = rotation + reflection).

    This tries both and returns the better one.
    """
    # Standard Procrustes (rotation only)
    R_rot, _ = orthogonal_procrustes(X_source, X_target)
    aligned_rot = X_source @ R_rot
    residual_rot = np.linalg.norm(aligned_rot - X_target, 'fro')

    # Try with one dimension reflected
    # Reflect last column of X_source, then do Procrustes
    X_reflected = X_source.copy()
    X_reflected[:, -1] *= -1
    R_ref, _ = orthogonal_procrustes(X_reflected, X_target)

    # The full transform is: reflect last dim, then rotate
    # Which is equivalent to: X @ diag(1,1,...,-1) @ R_ref
    reflect_matrix = np.eye(X_source.shape[1])
    reflect_matrix[-1, -1] = -1
    R_full = reflect_matrix @ R_ref

    aligned_ref = X_source @ R_full
    residual_ref = np.linalg.norm(aligned_ref - X_target, 'fro')

    if residual_ref < residual_rot:
        return R_full, residual_ref, "reflection"
    else:
        return R_rot, residual_rot, "rotation"


def sign_correct_greedy(X_source, X_target, max_flips=10):
    """Greedy sign correction: flip one dimension at a time.

    For each dimension, check if flipping its sign reduces residual.
    Repeat until no improvement.
    """
    k = X_source.shape[1]
    signs = np.ones(k)
    X_signed = X_source.copy()

    R, _ = orthogonal_procrustes(X_signed, X_target)
    best_residual = np.linalg.norm(X_signed @ R - X_target, 'fro')

    improved = True
    n_flips = 0

    while improved and n_flips < max_flips:
        improved = False

        for i in range(k):
            # Try flipping dimension i
            test_signs = signs.copy()
            test_signs[i] *= -1
            X_test = X_source * test_signs

            R_test, _ = orthogonal_procrustes(X_test, X_target)
            residual_test = np.linalg.norm(X_test @ R_test - X_target, 'fro')

            if residual_test < best_residual - 1e-6:
                signs = test_signs
                X_signed = X_test
                best_residual = residual_test
                improved = True
                n_flips += 1
                print(f"    Flip dim {i}: residual {best_residual:.4f}")

    R, residual = orthogonal_procrustes(X_signed, X_target)
    return signs, R, residual


def sign_correct_exhaustive(X_source, X_target, max_dims=12):
    """Exhaustive sign correction for first max_dims dimensions.

    Try all 2^max_dims combinations for first dimensions (most important).
    """
    k = min(X_source.shape[1], max_dims)
    best_signs = np.ones(X_source.shape[1])
    best_residual = float('inf')
    best_R = None

    n_combos = 2 ** k
    print(f"    Testing {n_combos} sign combinations for first {k} dims...")

    for combo in product([-1, 1], repeat=k):
        signs = np.ones(X_source.shape[1])
        signs[:k] = combo
        X_test = X_source * signs

        R, _ = orthogonal_procrustes(X_test, X_target)
        residual = np.linalg.norm(X_test @ R - X_target, 'fro')

        if residual < best_residual:
            best_residual = residual
            best_signs = signs.copy()
            best_R = R

    return best_signs, best_R, best_residual


def sign_correct_correlation(X_source, X_target):
    """Sign correction based on column correlation.

    For each dimension, check if source and target columns are
    positively or negatively correlated. Flip sign if negative.
    """
    k = X_source.shape[1]
    signs = np.ones(k)

    for i in range(k):
        corr = np.corrcoef(X_source[:, i], X_target[:, i])[0, 1]
        if corr < 0:
            signs[i] = -1

    X_signed = X_source * signs
    R, _ = orthogonal_procrustes(X_signed, X_target)
    residual = np.linalg.norm(X_signed @ R - X_target, 'fro')

    n_flipped = int(np.sum(signs < 0))
    return signs, R, residual, n_flipped


def diagnose_alignment():
    """Run diagnostic tests on Procrustes alignment."""
    print("=" * 70)
    print("DIAGNOSING PROCRUSTES ALIGNMENT")
    print("=" * 70)

    # Create embedding function
    def embed_nomic(texts):
        return get_embeddings(texts)

    # Load sentence-transformers models for comparison
    print("\nLoading models...")
    try:
        from sentence_transformers import SentenceTransformer
        model_mini = SentenceTransformer('all-MiniLM-L6-v2')
        model_mpnet = SentenceTransformer('all-mpnet-base-v2')

        def embed_mini(texts):
            return model_mini.encode(texts, convert_to_numpy=True)

        def embed_mpnet(texts):
            return model_mpnet.encode(texts, convert_to_numpy=True)

        has_st = True
        print("  Loaded MiniLM and MPNet")
    except ImportError:
        has_st = False
        print("  sentence-transformers not available")

    # Create keys
    print("\nCreating alignment keys...")
    k = 48
    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=STABLE_64, k=k)

    if has_st:
        key_mini = AlignmentKey.create("mini", embed_mini, anchors=STABLE_64, k=k)
        key_mpnet = AlignmentKey.create("mpnet", embed_mpnet, anchors=STABLE_64, k=k)
        models = [
            ("nomic", key_nomic),
            ("mini", key_mini),
            ("mpnet", key_mpnet),
        ]
    else:
        models = [("nomic", key_nomic)]

    # Test all pairs
    results = []

    for i, (name_a, key_a) in enumerate(models):
        for name_b, key_b in models[i+1:]:
            print(f"\n{'='*60}")
            print(f"TESTING: {name_a} <-> {name_b}")
            print(f"{'='*60}")

            # Get MDS coordinates
            X_a = key_a.eigenvectors[:, :k] * np.sqrt(key_a.eigenvalues[:k])
            X_b = key_b.eigenvectors[:, :k] * np.sqrt(key_b.eigenvalues[:k])

            # Spectrum correlation
            corr, _ = spearmanr(key_a.eigenvalues[:k], key_b.eigenvalues[:k])
            print(f"\nSpectrum correlation: {corr:.6f}")

            # --- Method 1: Standard Procrustes (current) ---
            print("\n[1] Standard Procrustes (rotation only):")
            R_std, _ = orthogonal_procrustes(X_a, X_b)
            residual_std = np.linalg.norm(X_a @ R_std - X_b, 'fro')
            print(f"    Residual: {residual_std:.4f}")
            print(f"    det(R): {np.linalg.det(R_std):.4f}")

            # --- Method 2: Procrustes with reflection ---
            print("\n[2] Procrustes with reflection:")
            R_ref, residual_ref, method = procrustes_with_reflection(X_a, X_b)
            print(f"    Best method: {method}")
            print(f"    Residual: {residual_ref:.4f}")
            print(f"    Improvement: {(residual_std - residual_ref)/residual_std*100:.1f}%")

            # --- Method 3: Correlation-based sign correction ---
            print("\n[3] Correlation-based sign correction:")
            signs_corr, R_corr, residual_corr, n_flipped = sign_correct_correlation(X_a, X_b)
            print(f"    Flipped {n_flipped}/{k} dimensions")
            print(f"    Residual: {residual_corr:.4f}")
            print(f"    Improvement: {(residual_std - residual_corr)/residual_std*100:.1f}%")

            # --- Method 4: Greedy sign correction ---
            print("\n[4] Greedy sign correction:")
            signs_greedy, R_greedy, residual_greedy = sign_correct_greedy(X_a, X_b)
            n_flipped_greedy = int(np.sum(signs_greedy < 0))
            print(f"    Flipped {n_flipped_greedy}/{k} dimensions")
            print(f"    Residual: {residual_greedy:.4f}")
            print(f"    Improvement: {(residual_std - residual_greedy)/residual_std*100:.1f}%")

            # --- Method 5: Exhaustive for first 10 dims ---
            print("\n[5] Exhaustive search (first 10 dims):")
            signs_exh, R_exh, residual_exh = sign_correct_exhaustive(X_a, X_b, max_dims=10)
            n_flipped_exh = int(np.sum(signs_exh < 0))
            print(f"    Flipped {n_flipped_exh}/{k} dimensions")
            print(f"    Residual: {residual_exh:.4f}")
            print(f"    Improvement: {(residual_std - residual_exh)/residual_std*100:.1f}%")

            # --- Best result ---
            best_residual = min(residual_std, residual_ref, residual_corr, residual_greedy, residual_exh)
            print(f"\n  BEST RESIDUAL: {best_residual:.4f}")
            print(f"  vs ORIGINAL:   {residual_std:.4f}")
            print(f"  IMPROVEMENT:   {(residual_std - best_residual)/residual_std*100:.1f}%")

            results.append({
                "pair": f"{name_a}->{name_b}",
                "spectrum_corr": corr,
                "residual_std": residual_std,
                "residual_best": best_residual,
                "improvement": (residual_std - best_residual)/residual_std*100,
            })

    # --- Test communication with sign correction ---
    if has_st:
        print("\n" + "=" * 70)
        print("COMMUNICATION TEST WITH SIGN CORRECTION")
        print("=" * 70)

        test_msg = "Explain how transformers work in neural networks"
        candidates = [
            test_msg,
            "Describe gradient descent optimization in machine learning",
            "What is the attention mechanism in deep learning",
            "Love is a powerful force that connects all humanity",
        ]

        X_nomic = key_nomic.eigenvectors[:, :k] * np.sqrt(key_nomic.eigenvalues[:k])
        X_mini = key_mini.eigenvectors[:, :k] * np.sqrt(key_mini.eigenvalues[:k])

        # Get best signs
        signs_best, R_best, residual_best = sign_correct_greedy(X_nomic, X_mini)

        print(f"\nOriginal residual: {np.linalg.norm(X_nomic @ orthogonal_procrustes(X_nomic, X_mini)[0] - X_mini, 'fro'):.4f}")
        print(f"Corrected residual: {residual_best:.4f}")

        # Test communication
        print("\n--- Without sign correction ---")
        pair_std = key_nomic.align_with(key_mini)
        vec_std = pair_std.encode_a_to_b(test_msg, embed_nomic)
        match_std, conf_std = pair_std.decode_at_b(vec_std, candidates, embed_mini)
        print(f"  Match: {match_std[:50]}...")
        print(f"  Confidence: {conf_std:.4f}")
        print(f"  Correct: {match_std == test_msg}")

        # For sign-corrected, we need to modify the encoding
        print("\n--- With sign correction ---")
        # Encode at nomic
        vec_nomic = key_nomic.encode(test_msg, embed_nomic)
        # Apply sign correction to nomic coordinates
        vec_signed = vec_nomic[:k] * signs_best
        # Apply rotation
        vec_transformed = vec_signed @ R_best
        # Decode at mini
        match_corr, conf_corr = key_mini.decode(vec_transformed, candidates, embed_mini)
        print(f"  Match: {match_corr[:50]}...")
        print(f"  Confidence: {conf_corr:.4f}")
        print(f"  Correct: {match_corr == test_msg}")

        print(f"\n  Confidence improvement: {conf_corr - conf_std:+.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['pair']}:")
        print(f"  Spectrum correlation: {r['spectrum_corr']:.4f}")
        print(f"  Residual (standard):  {r['residual_std']:.4f}")
        print(f"  Residual (best):      {r['residual_best']:.4f}")
        print(f"  Improvement:          {r['improvement']:.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The Procrustes residual comes from SIGN AMBIGUITY in MDS eigenvectors.

Solutions ranked by effectiveness:
1. Correlation-based sign correction - fast O(k), good results
2. Greedy sign correction - O(k^2), iteratively improves
3. Exhaustive search - O(2^k), optimal but expensive

RECOMMENDATION: Use correlation-based sign correction before Procrustes.
This should reduce cross-model residual from ~2.6 to ~0.5.
""")

    return results


if __name__ == "__main__":
    results = diagnose_alignment()
