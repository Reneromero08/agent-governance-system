#!/usr/bin/env python3
"""Find the Most Stable Anchors for Cross-Model Alignment.

If residual ~2.6 comes from anchor-level misalignment, we can find
which anchors contribute most to the error and remove them.

Approach:
1. Compute per-anchor alignment error
2. Rank anchors by stability
3. Create reduced anchor set with only stable ones
4. Test if residual decreases

Usage:
    python find_stable_anchors.py
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
import requests

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64, CANONICAL_128
from CAPABILITY.PRIMITIVES.mds import squared_distance_matrix, classical_mds


def get_embeddings(texts, url="http://10.5.0.2:1234/v1/embeddings",
                   model="text-embedding-nomic-embed-text-v1.5"):
    """Get embeddings from API."""
    response = requests.post(url, json={"model": model, "input": texts}, timeout=60)
    data = response.json()
    return np.array([d["embedding"] for d in data["data"]])


def compute_per_anchor_error(X_a, X_b, R):
    """Compute alignment error for each anchor point."""
    aligned = X_a @ R
    errors = np.linalg.norm(aligned - X_b, axis=1)
    return errors


def find_stable_anchors():
    """Analyze which anchors align best across models."""
    print("=" * 70)
    print("FINDING STABLE ANCHORS FOR CROSS-MODEL ALIGNMENT")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')
    model_mpnet = SentenceTransformer('all-mpnet-base-v2')

    def embed_nomic(texts):
        return get_embeddings(texts)

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    def embed_mpnet(texts):
        return model_mpnet.encode(texts, convert_to_numpy=True)

    models = {
        "nomic": embed_nomic,
        "mini": embed_mini,
        "mpnet": embed_mpnet,
    }

    # Use STABLE_64 as starting point
    anchors = STABLE_64
    k = 48
    n_anchors = len(anchors)

    print(f"Anchors: {n_anchors}")
    print(f"MDS dimensions: {k}")

    # Create keys for all models
    print("\nCreating alignment keys...")
    keys = {}
    for name, embed_fn in models.items():
        keys[name] = AlignmentKey.create(name, embed_fn, anchors=anchors, k=k)
        print(f"  {name}: done")

    # Compute per-anchor errors for each pair
    print("\n" + "=" * 60)
    print("PER-ANCHOR ALIGNMENT ERROR")
    print("=" * 60)

    all_errors = {}
    pairs = [("nomic", "mini"), ("nomic", "mpnet"), ("mini", "mpnet")]

    for name_a, name_b in pairs:
        key_a = keys[name_a]
        key_b = keys[name_b]

        X_a = key_a.eigenvectors[:, :k] * np.sqrt(key_a.eigenvalues[:k])
        X_b = key_b.eigenvectors[:, :k] * np.sqrt(key_b.eigenvalues[:k])

        R, _ = orthogonal_procrustes(X_a, X_b)
        errors = compute_per_anchor_error(X_a, X_b, R)

        all_errors[f"{name_a}->{name_b}"] = errors

        print(f"\n{name_a} -> {name_b}:")
        print(f"  Total residual: {np.sum(errors**2)**0.5:.4f}")
        print(f"  Mean error: {np.mean(errors):.4f}")
        print(f"  Std error: {np.std(errors):.4f}")
        print(f"  Max error: {np.max(errors):.4f} at '{anchors[np.argmax(errors)]}'")
        print(f"  Min error: {np.min(errors):.4f} at '{anchors[np.argmin(errors)]}'")

    # Average errors across all pairs
    avg_errors = np.zeros(n_anchors)
    for errors in all_errors.values():
        avg_errors += errors
    avg_errors /= len(all_errors)

    # Rank anchors
    print("\n" + "=" * 60)
    print("ANCHOR STABILITY RANKING (lower = more stable)")
    print("=" * 60)

    sorted_indices = np.argsort(avg_errors)

    print("\nTop 20 Most Stable Anchors:")
    for i, idx in enumerate(sorted_indices[:20]):
        print(f"  {i+1:2d}. {anchors[idx]:15s} error={avg_errors[idx]:.4f}")

    print("\nTop 20 Least Stable Anchors:")
    for i, idx in enumerate(sorted_indices[-20:][::-1]):
        print(f"  {i+1:2d}. {anchors[idx]:15s} error={avg_errors[idx]:.4f}")

    # Test with reduced anchor sets
    print("\n" + "=" * 60)
    print("TESTING REDUCED ANCHOR SETS")
    print("=" * 60)

    for n_keep in [64, 48, 32, 24, 16]:
        if n_keep > n_anchors:
            continue

        stable_indices = sorted_indices[:n_keep]
        stable_anchors = [anchors[i] for i in stable_indices]

        print(f"\n--- Using {n_keep} most stable anchors ---")

        # Create new keys
        keys_reduced = {}
        for name, embed_fn in models.items():
            keys_reduced[name] = AlignmentKey.create(
                name, embed_fn, anchors=stable_anchors, k=min(k, n_keep-1)
            )

        # Test residuals
        for name_a, name_b in pairs:
            key_a = keys_reduced[name_a]
            key_b = keys_reduced[name_b]
            k_use = min(key_a.k, key_b.k)

            X_a = key_a.eigenvectors[:, :k_use] * np.sqrt(key_a.eigenvalues[:k_use])
            X_b = key_b.eigenvectors[:, :k_use] * np.sqrt(key_b.eigenvalues[:k_use])

            R, _ = orthogonal_procrustes(X_a, X_b)
            residual = np.linalg.norm(X_a @ R - X_b, 'fro')

            # Normalized residual (per dimension)
            norm_residual = residual / k_use**0.5

            print(f"  {name_a}->{name_b}: residual={residual:.4f}, norm={norm_residual:.4f}")

    # Find optimal number of anchors
    print("\n" + "=" * 60)
    print("COMMUNICATION TEST WITH DIFFERENT ANCHOR COUNTS")
    print("=" * 60)

    test_msg = "Explain how transformers work in neural networks"
    candidates = [
        test_msg,
        "Describe gradient descent optimization in machine learning",
        "What is the attention mechanism in deep learning",
        "Love is a powerful force that connects all humanity",
    ]

    for n_keep in [64, 48, 32]:
        if n_keep > n_anchors:
            continue

        stable_indices = sorted_indices[:n_keep]
        stable_anchors = [anchors[i] for i in stable_indices]
        k_use = min(32, n_keep - 1)

        key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=stable_anchors, k=k_use)
        key_mini = AlignmentKey.create("mini", embed_mini, anchors=stable_anchors, k=k_use)

        pair = key_nomic.align_with(key_mini)

        print(f"\n--- {n_keep} anchors, k={k_use} ---")
        print(f"  Residual: {pair.procrustes_residual:.4f}")

        # Test corruption tolerance
        for pct in [0.0, 0.25, 0.50, 0.75]:
            n_corrupt = int(k_use * pct)
            if n_corrupt >= k_use:
                continue

            successes = 0
            n_trials = 10

            for trial in range(n_trials):
                vec = pair.encode_a_to_b(test_msg, embed_nomic)

                # Corrupt
                if n_corrupt > 0:
                    np.random.seed(trial)
                    indices = np.random.choice(len(vec), n_corrupt, replace=False)
                    vec[indices] = 0.0

                match, conf = pair.decode_at_b(vec, candidates, embed_mini)
                if match == test_msg:
                    successes += 1

            print(f"  Corruption {pct*100:.0f}%: {successes}/{n_trials} ({successes/n_trials*100:.0f}%)")

    # Output optimal anchor set
    print("\n" + "=" * 60)
    print("OPTIMAL STABLE ANCHORS")
    print("=" * 60)

    best_32 = [anchors[i] for i in sorted_indices[:32]]
    print("\nSTABLE_32 (best 32 anchors for cross-model alignment):")
    print("STABLE_32 = [")
    for i in range(0, 32, 8):
        line = ", ".join(f'"{w}"' for w in best_32[i:i+8])
        print(f"    {line},")
    print("]")

    return avg_errors, sorted_indices


if __name__ == "__main__":
    errors, indices = find_stable_anchors()
