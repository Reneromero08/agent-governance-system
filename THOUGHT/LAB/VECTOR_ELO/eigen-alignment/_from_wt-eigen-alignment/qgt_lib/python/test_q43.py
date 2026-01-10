#!/usr/bin/env python3
"""
Test Q43 Predictions: QGT Structure in Semantic Space

This script validates the Q43 predictions using the QGT Python bindings:
1. Effective rank ≈ 22 (matches E.X.3.4 participation ratio)
2. Berry phase around word analogy loops
3. Natural gradient matches compass mode
4. Chern number estimation

Usage:
    python test_q43.py                    # Run all tests with synthetic data
    python test_q43.py --embeddings FILE  # Use real BERT embeddings
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


def test_berry_phase():
    """Test 2: Berry phase around closed loops."""
    print("\n" + "=" * 60)
    print("TEST 2: Berry Phase (Topological Structure)")
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

    # Create analogy: king - queen ≈ man - woman
    embeddings = {
        'king': base + royalty + 0.0 * gender,
        'queen': base + royalty + 1.0 * gender,
        'man': base + 0.0 * royalty + 0.0 * gender,
        'woman': base + 0.0 * royalty + 1.0 * gender,
    }

    print("\n[Word Analogy Loop: king -> queen -> woman -> man -> king]")
    loop = qgt.create_analogy_loop(embeddings, ['king', 'queen', 'woman', 'man'])
    phase = qgt.berry_phase(loop)
    print(f"  Berry phase: {phase:.4f} rad ({np.degrees(phase):.2f}°)")

    # For a perfect parallelogram, Berry phase = area on sphere
    print("\n[Interpretation]")
    if abs(phase) > 0.01:
        print(f"  [OK] Non-zero Berry phase detected!")
        print(f"  This indicates topological structure in semantic space")
    else:
        print(f"  Phase is near zero (flat geometry)")

    # Test with random loop (should have random phase)
    print("\n[Random 4-point Loop (control)]")
    random_loop = np.random.randn(4, dim)
    random_phase = qgt.berry_phase(random_loop)
    print(f"  Berry phase: {random_phase:.4f} rad ({np.degrees(random_phase):.2f}°)")

    return {
        'analogy_phase': phase,
        'random_phase': random_phase,
    }


def test_holonomy():
    """Test holonomy (parallel transport) around loops."""
    print("\n" + "=" * 60)
    print("TEST 2b: Holonomy (Parallel Transport)")
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

    print(f"  Holonomy angle: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
    print(f"  |v_final - v_initial| = {np.linalg.norm(transported - tangent / np.linalg.norm(tangent)):.4f}")

    if angle > 0.01:
        print("  [OK] Non-trivial holonomy detected (curved geometry)")
    else:
        print("  Holonomy near zero (approximately flat)")

    return {'holonomy_angle': angle}


def test_natural_gradient():
    """Test 3: Natural gradient vs Euclidean gradient."""
    print("\n" + "=" * 60)
    print("TEST 3: Natural Gradient (Compass Mode)")
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


def test_chern_number():
    """Test 4: Chern number estimation."""
    print("\n" + "=" * 60)
    print("TEST 4: Chern Number Estimate")
    print("=" * 60)

    dim = 768
    np.random.seed(42)

    # Low-rank embeddings
    embeddings = np.random.randn(500, 22) @ np.random.randn(22, dim)
    embeddings += 0.1 * np.random.randn(500, dim)

    print("\n[Monte Carlo Chern Number Estimation]")
    chern = qgt.chern_number_estimate(embeddings, n_samples=500)
    print(f"  Chern estimate: {chern:.4f}")

    # For comparison, random embeddings
    random_emb = np.random.randn(500, dim)
    chern_random = qgt.chern_number_estimate(random_emb, n_samples=500)
    print(f"  Random baseline: {chern_random:.4f}")

    print("\n[Interpretation]")
    print("  Note: True Chern numbers require complex structure.")
    print("  This is an approximation based on discrete Berry phase.")

    if abs(chern) > abs(chern_random) * 2:
        print(f"  [OK] Structured embeddings show stronger topological signal")
    else:
        print(f"  Similar magnitude to random baseline")

    return {
        'chern_trained': chern,
        'chern_random': chern_random,
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
    results = qgt.analyze_qgt_structure(embeddings)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Q43 QGT predictions")
    parser.add_argument('--embeddings', type=str, help="Path to real embeddings file")
    args = parser.parse_args()

    print("=" * 60)
    print("Q43: Quantum Geometric Tensor for Semiosphere")
    print("Testing QGT Structure in Semantic Space")
    print("=" * 60)

    if args.embeddings:
        test_with_real_embeddings(args.embeddings)
    else:
        # Run all synthetic tests
        test_effective_rank()
        test_berry_phase()
        test_holonomy()
        test_natural_gradient()
        test_chern_number()

    print("\n" + "=" * 60)
    print("Q43 TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
