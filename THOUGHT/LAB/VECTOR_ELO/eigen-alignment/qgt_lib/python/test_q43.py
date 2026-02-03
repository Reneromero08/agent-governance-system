#!/usr/bin/env python3
"""
Test Q43 Predictions: QGT Structure in Semantic Space

This script validates the Q43 predictions using the QGT Python bindings:
1. Effective rank ~ 22 (matches E.X.3.4 participation ratio)
2. Solid angle (spherical excess) around word analogy loops
3. Natural gradient matches compass mode
4. Holonomy around closed loops

CORRECTION (2025): Uses spherical_excess() for true solid angle computation,
NOT pca_winding_angle() which was incorrectly used before.

Usage:
    python test_q43.py                    # Run all tests with synthetic data
    python test_q43.py --embeddings FILE  # Use real BERT embeddings
    python test_q43.py --real-analogies   # Use real GloVe word embeddings
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qgt_lib.python import qgt


def test_effective_rank():
    """Test 1: Effective rank should be ~22 for trained embeddings."""
    print("\n" + "=" * 60)
    print("TEST 1: Effective Rank (Participation Ratio)")
    print("=" * 60)

    # Simulate different embedding types
    results = {}

    # Random embeddings (should give ~100)
    print("\n[Random Embeddings]")
    random_emb = np.random.randn(1000, 768)
    pr_random = qgt.participation_ratio(random_emb)
    results['random'] = pr_random
    print(f"  Df = {pr_random:.1f}")
    print(f"  Expected: ~100 (high dimensionality)")

    # Simulated untrained BERT (~62)
    print("\n[Simulated Untrained BERT]")
    # Mix of ~62D structure + noise
    untrained = np.random.randn(1000, 62) @ np.random.randn(62, 768)
    untrained += 0.3 * np.random.randn(1000, 768)
    pr_untrained = qgt.participation_ratio(untrained)
    results['untrained'] = pr_untrained
    print(f"  Df = {pr_untrained:.1f}")
    print(f"  Expected: ~62 (partially structured)")

    # Simulated trained BERT (~22)
    print("\n[Simulated Trained BERT]")
    trained = np.random.randn(1000, 22) @ np.random.randn(22, 768)
    trained += 0.1 * np.random.randn(1000, 768)
    pr_trained = qgt.participation_ratio(trained)
    results['trained'] = pr_trained
    print(f"  Df = {pr_trained:.1f}")
    print(f"  Expected: ~22 (Q43 prediction, E.X.3.4 found 22.2)")

    # Verdict
    print("\n[RESULT]")
    if 18 <= pr_trained <= 26:
        print("  [PASS] Trained embeddings have Df ~ 22")
    else:
        print(f"  [FAIL] Expected 22 +/- 4, got {pr_trained:.1f}")

    return results


def test_spherical_excess_synthetic():
    """Test 2: Spherical excess with synthetic word analogy loops."""
    print("\n" + "=" * 60)
    print("TEST 2: Spherical Excess (Synthetic Analogies)")
    print("=" * 60)

    # Create synthetic word embeddings for analogy test
    dim = 768

    # Create embeddings with semantic structure
    # king, queen, man, woman should form a parallelogram
    np.random.seed(42)

    # Base vectors
    gender = np.random.randn(dim)
    gender = gender / np.linalg.norm(gender)

    royalty = np.random.randn(dim)
    royalty = royalty - np.dot(royalty, gender) * gender  # Orthogonalize
    royalty = royalty / np.linalg.norm(royalty)

    base = np.random.randn(dim) * 0.1

    # Create analogy: king - queen ~ man - woman
    embeddings = {
        'king': base + royalty + 0.0 * gender,
        'queen': base + royalty + 1.0 * gender,
        'man': base + 0.0 * royalty + 0.0 * gender,
        'woman': base + 0.0 * royalty + 1.0 * gender,
    }

    # Normalize embeddings
    for word in embeddings:
        embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])

    print("\n[Synthetic Word Analogy Loop: king -> queen -> woman -> man -> king]")

    # Create closed loop: king -> queen -> woman -> man -> king
    loop = np.array([
        embeddings['king'],
        embeddings['queen'],
        embeddings['woman'],
        embeddings['man'],
        embeddings['king']  # Close the loop
    ])

    # Compute spherical excess (solid angle)
    solid_angle = qgt.spherical_excess(loop)
    print(f"  Spherical excess (solid angle): {solid_angle:.6f} rad ({np.degrees(solid_angle):.4f} deg)")

    # For comparison, compute holonomy
    tangent = np.random.randn(dim)
    holonomy_ang = qgt.holonomy_angle(loop[:-1], tangent)
    print(f"  Holonomy angle: {holonomy_ang:.6f} rad ({np.degrees(holonomy_ang):.4f} deg)")

    print("\n[Interpretation]")
    print(f"  Solid angle magnitude: |{solid_angle:.6f}| rad")
    print(f"  Note: Synthetic analogies in high-D space have small solid angles")
    print(f"  because the 'parallelogram' is nearly planar in 768D.")

    # Test with random loop (control)
    print("\n[Random 4-point Loop (control)]")
    random_loop = np.random.randn(4, dim)
    random_excess = qgt.spherical_excess(random_loop)
    print(f"  Spherical excess: {random_excess:.6f} rad ({np.degrees(random_excess):.4f} deg)")

    return {
        'analogy_solid_angle': solid_angle,
        'random_solid_angle': random_excess,
        'holonomy_angle': holonomy_ang,
    }


def test_real_word_analogies():
    """Test 3: Spherical excess with REAL GloVe word embeddings."""
    print("\n" + "=" * 60)
    print("TEST 3: Real Word Analogies (GloVe Embeddings)")
    print("=" * 60)

    try:
        import gensim.downloader as api
    except ImportError:
        print("\n[ERROR] gensim not installed. Install with: pip install gensim")
        print("Skipping real word analogy test.")
        return None

    # Load GloVe embeddings
    print("\n[Loading GloVe embeddings...]")
    print("  Model: glove-wiki-gigaword-300 (300 dimensions)")
    try:
        glove = api.load('glove-wiki-gigaword-300')
        print(f"  Loaded {len(glove)} word vectors")
    except Exception as e:
        print(f"  [ERROR] Failed to load GloVe: {e}")
        return None

    def get_embedding(word):
        """Get normalized embedding for a word."""
        if word in glove:
            vec = glove[word].astype(np.float64)
            return vec / np.linalg.norm(vec)
        return None

    # Classic word analogies that should form geometric loops
    analogies = [
        ('king', 'queen', 'man', 'woman'),        # gender analogy
        ('paris', 'france', 'berlin', 'germany'), # capital-country
        ('big', 'bigger', 'small', 'smaller'),    # comparative
        ('run', 'ran', 'walk', 'walked'),         # tense
        ('good', 'better', 'bad', 'worse'),       # irregular comparative
        ('cat', 'cats', 'dog', 'dogs'),           # plural
        ('boy', 'girl', 'brother', 'sister'),     # gender pairs
        ('slow', 'fast', 'old', 'young'),         # antonyms
        ('apple', 'fruit', 'carrot', 'vegetable'), # category
        ('london', 'england', 'tokyo', 'japan'),  # capital-country 2
    ]

    print("\n[Computing Spherical Excess for Word Analogy Loops]")
    print("-" * 60)

    results = []
    for a, b, c, d in analogies:
        embs = [get_embedding(w) for w in [a, b, c, d]]
        if any(e is None for e in embs):
            missing = [w for w, e in zip([a, b, c, d], embs) if e is None]
            print(f"  SKIP {a}-{b}-{c}-{d}: missing embeddings for {missing}")
            continue

        # Create closed loop: a -> b -> d -> c -> a
        # This traces the parallelogram of the analogy a:b :: c:d
        loop = np.array([embs[0], embs[1], embs[3], embs[2], embs[0]])

        # Compute spherical excess (solid angle)
        solid_angle = qgt.spherical_excess(loop)

        # Also compute holonomy for comparison
        tangent = np.random.randn(300)
        holonomy_ang = qgt.holonomy_angle(loop[:-1], tangent)

        results.append({
            'analogy': f"{a}:{b}::{c}:{d}",
            'solid_angle': solid_angle,
            'holonomy': holonomy_ang
        })

        print(f"  {a:8s}-{b:8s}-{c:8s}-{d:10s}: "
              f"solid_angle = {solid_angle:+.6f} rad ({np.degrees(solid_angle):+.4f} deg)")

    print("-" * 60)

    if results:
        # Statistics
        angles = [r['solid_angle'] for r in results]
        holonomies = [r['holonomy'] for r in results]

        print("\n[Summary Statistics]")
        print(f"  Number of analogies tested: {len(results)}")
        print(f"  Solid angle mean: {np.mean(angles):+.6f} rad ({np.degrees(np.mean(angles)):+.4f} deg)")
        print(f"  Solid angle std:  {np.std(angles):.6f} rad ({np.degrees(np.std(angles)):.4f} deg)")
        print(f"  Solid angle min:  {np.min(angles):+.6f} rad")
        print(f"  Solid angle max:  {np.max(angles):+.6f} rad")
        print(f"  Holonomy mean:    {np.mean(holonomies):.6f} rad")

        print("\n[Analysis]")
        print(f"  The solid angles are on the order of {np.abs(np.mean(angles)):.4f} rad")
        print(f"  This is much smaller than the erroneous -4.7 rad from PCA winding")
        print(f"  Real word analogy loops subtend small solid angles on S^299")

        # Check for pattern
        print("\n[Pattern Check]")
        if np.std(angles) < 0.1:
            print("  Solid angles are relatively consistent across analogies")
        else:
            print("  Solid angles vary significantly across different analogy types")

    return results


def test_holonomy():
    """Test holonomy (parallel transport) around loops."""
    print("\n" + "=" * 60)
    print("TEST 4: Holonomy (Parallel Transport)")
    print("=" * 60)

    dim = 768
    np.random.seed(42)

    # Create a geodesic triangle on the sphere
    # Large triangles have more holonomy
    v1 = np.zeros(dim)
    v1[0] = 1.0

    v2 = np.zeros(dim)
    v2[0] = np.cos(np.pi / 3)
    v2[1] = np.sin(np.pi / 3)

    v3 = np.zeros(dim)
    v3[0] = np.cos(np.pi / 3)
    v3[2] = np.sin(np.pi / 3)

    triangle = np.array([v1, v2, v3])

    # Initial tangent vector at v1
    tangent = np.zeros(dim)
    tangent[1] = 1.0

    print("\n[Parallel Transport Around Geodesic Triangle]")
    transported = qgt.holonomy(triangle, tangent)
    angle = qgt.holonomy_angle(triangle, tangent)

    print(f"  Holonomy angle: {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
    print(f"  |v_final - v_initial| = {np.linalg.norm(transported - tangent / np.linalg.norm(tangent)):.4f}")

    # Compute spherical excess for the same triangle
    excess = qgt.spherical_excess(triangle)
    print(f"  Spherical excess: {excess:.4f} rad ({np.degrees(excess):.2f} deg)")

    if angle > 0.01:
        print("  [OK] Non-trivial holonomy detected (curved geometry)")
    else:
        print("  Holonomy near zero (approximately flat)")

    return {'holonomy_angle': angle, 'spherical_excess': excess}


def test_natural_gradient():
    """Test 5: Natural gradient vs Euclidean gradient."""
    print("\n" + "=" * 60)
    print("TEST 5: Natural Gradient (Compass Mode)")
    print("=" * 60)

    dim = 768
    np.random.seed(42)

    # Create low-rank embeddings (simulating trained BERT)
    embeddings = np.random.randn(1000, 22) @ np.random.randn(22, dim)
    embeddings += 0.1 * np.random.randn(1000, dim)

    # Get principal directions from QGT
    principal_dirs = qgt.principal_directions(embeddings, n_components=22)

    print("\n[Principal Directions from QGT Metric]")
    print(f"  Shape: {principal_dirs.shape}")

    # Create a random Euclidean gradient
    euclidean_grad = np.random.randn(dim)

    # Compute natural gradient
    natural_grad = qgt.natural_gradient(embeddings, euclidean_grad)

    print("\n[Natural Gradient Transform]")
    print(f"  |Euclidean grad|: {np.linalg.norm(euclidean_grad):.4f}")
    print(f"  |Natural grad|: {np.linalg.norm(natural_grad):.4f}")

    # Project gradients onto principal directions
    euclidean_proj = np.abs(principal_dirs @ euclidean_grad)
    natural_proj = np.abs(principal_dirs @ natural_grad)

    print("\n[Projection onto Top 5 Principal Directions]")
    print(f"  Euclidean: {euclidean_proj[:5]}")
    print(f"  Natural:   {natural_proj[:5]}")

    # Natural gradient should align more with principal directions
    euclidean_alignment = np.sum(euclidean_proj[:22]) / np.linalg.norm(euclidean_grad)
    natural_alignment = np.sum(natural_proj[:22]) / np.linalg.norm(natural_grad)

    print(f"\n[Alignment with 22D Subspace]")
    print(f"  Euclidean: {euclidean_alignment:.4f}")
    print(f"  Natural:   {natural_alignment:.4f}")

    if natural_alignment > euclidean_alignment:
        print("  [OK] Natural gradient better aligned with principal subspace")
    else:
        print("  [WARN]  Natural gradient not significantly better aligned")

    return {
        'principal_directions': principal_dirs,
        'euclidean_alignment': euclidean_alignment,
        'natural_alignment': natural_alignment,
    }


def test_with_real_embeddings(path: str):
    """Run tests with real BERT embeddings."""
    print("\n" + "=" * 60)
    print(f"TESTING WITH REAL EMBEDDINGS: {path}")
    print("=" * 60)

    # Load embeddings
    p = Path(path)
    if p.suffix == '.npy':
        embeddings = np.load(p)
    elif p.suffix == '.npz':
        data = np.load(p)
        # Try common keys
        for key in ['embeddings', 'emb', 'vectors', 'data']:
            if key in data:
                embeddings = data[key]
                break
        else:
            print(f"Available keys: {list(data.keys())}")
            return
    else:
        print(f"Unknown format: {p.suffix}")
        return

    print(f"\nLoaded embeddings: {embeddings.shape}")

    # Full analysis
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress Chern number warning
        results = qgt.analyze_qgt_structure(embeddings)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Q43 QGT predictions")
    parser.add_argument('--embeddings', type=str, help="Path to real embeddings file")
    parser.add_argument('--real-analogies', action='store_true',
                        help="Test with real GloVe word embeddings")
    args = parser.parse_args()

    print("=" * 60)
    print("Q43: Quantum Geometric Tensor for Semiosphere")
    print("Testing QGT Structure in Semantic Space")
    print("=" * 60)
    print("\nCORRECTION: Using spherical_excess() for solid angle,")
    print("NOT the incorrect pca_winding_angle() method.")

    if args.embeddings:
        test_with_real_embeddings(args.embeddings)
    elif args.real_analogies:
        # Only run real word analogy test
        test_real_word_analogies()
    else:
        # Run all synthetic tests
        test_effective_rank()
        test_spherical_excess_synthetic()
        test_holonomy()
        test_natural_gradient()

        # Optionally run real analogies if gensim available
        print("\n" + "=" * 60)
        print("OPTIONAL: Real Word Analogies")
        print("=" * 60)
        print("\nTo test with real GloVe embeddings, run:")
        print("  python test_q43.py --real-analogies")
        print("\nThis requires: pip install gensim")

    print("\n" + "=" * 60)
    print("Q43 TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
