#!/usr/bin/env python3
"""
Q42 Test 1: Semantic CHSH - Main Non-Locality Test

PURPOSE: Determine if semantic space violates Bell inequality.

This test:
1. Creates "entangled" concept pairs (particle/wave, hot/cold, etc.)
2. Measures semantic correlations across multiple projection directions
3. Computes CHSH statistic for each pair
4. Checks if S > 2 (Bell violation)

Pass criteria:
- H0 (Local): S ≤ 2.0 for ALL concept pairs
- H1 (Non-local): S > 2.1 for at least one pair with p < 0.001

Run: python test_q42_semantic_chsh.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from bell import (
    CLASSICAL_BOUND,
    QUANTUM_BOUND,
    VIOLATION_THRESHOLD,
    CHSHResult,
    compute_chsh,
    semantic_chsh,
    get_projection_directions,
    semantic_correlation,
    ENTANGLED_PAIRS,
    bootstrap_chsh_confidence,
)


# =============================================================================
# EMBEDDING GENERATION (Synthetic for now)
# =============================================================================

def generate_synthetic_embeddings(
    concept: str,
    n_contexts: int = 100,
    dim: int = 768,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic embeddings for a concept across contexts.

    This simulates how a concept's embedding varies with context.

    Args:
        concept: Concept name (used to seed generator for reproducibility)
        n_contexts: Number of context variations
        dim: Embedding dimension
        seed: Optional random seed

    Returns:
        (n_contexts, dim) array of embeddings
    """
    if seed is None:
        # Use concept hash for reproducibility
        seed = hash(concept) % (2**31)

    np.random.seed(seed)

    # Base embedding for concept
    base = np.random.randn(dim)
    base = base / np.linalg.norm(base)

    # Generate context variations
    embeddings = []
    for _ in range(n_contexts):
        # Add context-dependent noise
        noise = np.random.randn(dim) * 0.3
        emb = base + noise
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    return np.array(embeddings)


def generate_entangled_pair_embeddings(
    concept_A: str,
    concept_B: str,
    n_contexts: int = 100,
    dim: int = 768,
    correlation_strength: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated embeddings for an "entangled" concept pair.

    The embeddings are designed to be correlated but not identical,
    simulating semantic entanglement.

    Args:
        concept_A: First concept
        concept_B: Second concept
        n_contexts: Number of contexts
        dim: Embedding dimension
        correlation_strength: How correlated the pairs should be [0,1]

    Returns:
        (embeddings_A, embeddings_B) each (n_contexts, dim)
    """
    seed_A = hash(concept_A) % (2**31)
    seed_B = hash(concept_B) % (2**31)

    np.random.seed(seed_A)
    base_A = np.random.randn(dim)
    base_A = base_A / np.linalg.norm(base_A)

    np.random.seed(seed_B)
    base_B = np.random.randn(dim)
    base_B = base_B / np.linalg.norm(base_B)

    # Make B partially aligned with A for correlation
    base_B = correlation_strength * base_A + (1 - correlation_strength) * base_B
    base_B = base_B / np.linalg.norm(base_B)

    embeddings_A = []
    embeddings_B = []

    np.random.seed(seed_A + seed_B)  # Shared context randomness

    for i in range(n_contexts):
        # Shared context influence
        shared_context = np.random.randn(dim) * 0.2

        # Concept-specific noise
        np.random.seed(seed_A + i)
        noise_A = np.random.randn(dim) * 0.2

        np.random.seed(seed_B + i)
        noise_B = np.random.randn(dim) * 0.2

        # Final embeddings
        emb_A = base_A + shared_context + noise_A
        emb_A = emb_A / np.linalg.norm(emb_A)

        emb_B = base_B + shared_context + noise_B
        emb_B = emb_B / np.linalg.norm(emb_B)

        embeddings_A.append(emb_A)
        embeddings_B.append(emb_B)

    return np.array(embeddings_A), np.array(embeddings_B)


def generate_uncorrelated_pair_embeddings(
    concept_A: str,
    concept_B: str,
    n_contexts: int = 100,
    dim: int = 768
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uncorrelated embeddings for control pairs.

    Args:
        concept_A: First concept
        concept_B: Second concept
        n_contexts: Number of contexts
        dim: Embedding dimension

    Returns:
        (embeddings_A, embeddings_B) each (n_contexts, dim)
    """
    # Completely independent generation
    emb_A = generate_synthetic_embeddings(concept_A, n_contexts, dim)
    emb_B = generate_synthetic_embeddings(concept_B, n_contexts, dim)

    return emb_A, emb_B


# =============================================================================
# SEMANTIC CHSH TESTS
# =============================================================================

@dataclass
class SemanticCHSHResult:
    """Result for a single concept pair."""
    concept_A: str
    concept_B: str
    category: str
    S: float
    S_std: float
    is_violation: bool
    is_significant: bool
    correlations: Dict[str, float]


def test_concept_pair(
    concept_A: str,
    concept_B: str,
    category: str,
    n_contexts: int = 100,
    n_bootstrap: int = 50,
    dim: int = 768
) -> SemanticCHSHResult:
    """
    Test CHSH for a single concept pair.

    Args:
        concept_A: First concept
        concept_B: Second concept
        category: Category name (for reporting)
        n_contexts: Number of contexts to sample
        n_bootstrap: Number of bootstrap samples for confidence
        dim: Embedding dimension

    Returns:
        SemanticCHSHResult
    """
    # Generate embeddings based on category
    if category == 'control_uncorrelated':
        emb_A, emb_B = generate_uncorrelated_pair_embeddings(
            concept_A, concept_B, n_contexts, dim
        )
    else:
        # Entangled pairs get correlated embeddings
        emb_A, emb_B = generate_entangled_pair_embeddings(
            concept_A, concept_B, n_contexts, dim,
            correlation_strength=0.7 if category != 'emergent' else 0.5
        )

    # Compute CHSH
    result = semantic_chsh(emb_A, emb_B)

    # Bootstrap for confidence
    S_values = []
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_contexts, n_contexts, replace=True)
        boot_result = semantic_chsh(emb_A[idx], emb_B[idx])
        S_values.append(boot_result.S)

    S_mean = np.mean(S_values)
    S_std = np.std(S_values)

    # Statistical test: is S > 2 significant at 3σ?
    is_significant = (S_mean - 3 * S_std) > CLASSICAL_BOUND

    return SemanticCHSHResult(
        concept_A=concept_A,
        concept_B=concept_B,
        category=category,
        S=result.S,
        S_std=S_std,
        is_violation=result.S > CLASSICAL_BOUND,
        is_significant=is_significant,
        correlations={
            'E_ab': result.E_ab,
            'E_ab_prime': result.E_ab_prime,
            'E_a_prime_b': result.E_a_prime_b,
            'E_a_prime_b_prime': result.E_a_prime_b_prime
        }
    )


def test_category(
    category: str,
    pairs: List[Tuple[str, str]],
    n_contexts: int = 100
) -> Dict:
    """
    Test all pairs in a category.

    Args:
        category: Category name
        pairs: List of (concept_A, concept_B) tuples
        n_contexts: Number of contexts per pair

    Returns:
        Dict with category results
    """
    print(f"\n--- Testing Category: {category} ---")

    results = []
    for concept_A, concept_B in pairs:
        result = test_concept_pair(
            concept_A, concept_B, category, n_contexts
        )
        results.append(result)

        status = '⚠️' if result.is_violation else '✓'
        sig = '(significant)' if result.is_significant else ''
        print(f"  {status} {concept_A}/{concept_B}: S = {result.S:.4f} ± {result.S_std:.4f} {sig}")

    # Category summary
    S_values = [r.S for r in results]
    n_violations = sum(1 for r in results if r.is_violation)
    n_significant = sum(1 for r in results if r.is_significant)

    return {
        'category': category,
        'pairs': [
            {
                'concept_A': r.concept_A,
                'concept_B': r.concept_B,
                'S': r.S,
                'S_std': r.S_std,
                'is_violation': r.is_violation,
                'is_significant': r.is_significant,
                'correlations': r.correlations
            }
            for r in results
        ],
        'summary': {
            'n_pairs': len(pairs),
            'n_violations': n_violations,
            'n_significant': n_significant,
            'mean_S': float(np.mean(S_values)),
            'max_S': float(np.max(S_values)),
            'min_S': float(np.min(S_values)),
        }
    }


def test_optimal_projection_scan(
    concept_A: str = 'particle',
    concept_B: str = 'wave',
    n_angles: int = 36,
    n_contexts: int = 200
) -> Dict:
    """
    Scan over projection angles to find maximum CHSH.

    This tests whether ANY choice of projections gives S > 2.

    Args:
        concept_A: First concept
        concept_B: Second concept
        n_angles: Number of angles to scan
        n_contexts: Number of contexts

    Returns:
        Dict with scan results
    """
    print(f"\n--- Projection Angle Scan: {concept_A}/{concept_B} ---")

    emb_A, emb_B = generate_entangled_pair_embeddings(
        concept_A, concept_B, n_contexts, 768,
        correlation_strength=0.9  # Maximum correlation for best chance
    )

    # Get PCA directions
    all_emb = np.vstack([emb_A, emb_B])
    from scipy.linalg import svd
    centered = all_emb - all_emb.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)

    # Scan over rotation angles in principal plane
    angles = np.linspace(0, np.pi, n_angles)
    best_S = 0
    best_angles = (0, 0, 0, 0)

    S_values = []

    for i, angle_a in enumerate(angles):
        for j, angle_b in enumerate(angles):
            # Create rotated directions
            a = np.cos(angle_a) * Vt[0] + np.sin(angle_a) * Vt[1]
            a_prime = np.cos(angle_a + np.pi/2) * Vt[0] + np.sin(angle_a + np.pi/2) * Vt[1]
            b = np.cos(angle_b) * Vt[0] + np.sin(angle_b) * Vt[1]
            b_prime = np.cos(angle_b + np.pi/2) * Vt[0] + np.sin(angle_b + np.pi/2) * Vt[1]

            # Compute correlations
            E_ab = semantic_correlation(emb_A, emb_B, a, b)
            E_ab_prime = semantic_correlation(emb_A, emb_B, a, b_prime)
            E_a_prime_b = semantic_correlation(emb_A, emb_B, a_prime, b)
            E_a_prime_b_prime = semantic_correlation(emb_A, emb_B, a_prime, b_prime)

            result = compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)
            S_values.append(result.S)

            if result.S > best_S:
                best_S = result.S
                best_angles = (angle_a, angle_a + np.pi/2, angle_b, angle_b + np.pi/2)

    print(f"  Scanned {n_angles}² = {n_angles**2} angle combinations")
    print(f"  Best S: {best_S:.4f}")
    print(f"  Best angles: a={np.degrees(best_angles[0]):.1f}°, a'={np.degrees(best_angles[1]):.1f}°, "
          f"b={np.degrees(best_angles[2]):.1f}°, b'={np.degrees(best_angles[3]):.1f}°")
    print(f"  Mean S: {np.mean(S_values):.4f}")
    print(f"  S > 2: {sum(1 for s in S_values if s > 2)} / {len(S_values)}")

    return {
        'concept_A': concept_A,
        'concept_B': concept_B,
        'best_S': float(best_S),
        'best_angles_deg': [float(np.degrees(a)) for a in best_angles],
        'mean_S': float(np.mean(S_values)),
        'max_S': float(np.max(S_values)),
        'n_above_2': sum(1 for s in S_values if s > 2),
        'n_total': len(S_values),
        'is_violation': best_S > CLASSICAL_BOUND
    }


def test_correlation_strength_sweep(
    n_strengths: int = 10,
    n_contexts: int = 100
) -> Dict:
    """
    Test how correlation strength affects CHSH.

    If semantic entanglement exists, stronger correlation should
    give higher CHSH (up to quantum bound).

    Returns:
        Dict with sweep results
    """
    print("\n--- Correlation Strength Sweep ---")

    strengths = np.linspace(0, 1, n_strengths)
    S_values = []

    for strength in strengths:
        emb_A, emb_B = generate_entangled_pair_embeddings(
            'particle', 'wave', n_contexts, 768,
            correlation_strength=strength
        )
        result = semantic_chsh(emb_A, emb_B)
        S_values.append(result.S)
        print(f"  ρ = {strength:.2f}: S = {result.S:.4f}")

    return {
        'strengths': strengths.tolist(),
        'S_values': S_values,
        'max_S': float(np.max(S_values)),
        'correlation_S': float(np.corrcoef(strengths, S_values)[0, 1])
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> Dict:
    """Run all semantic CHSH tests."""

    print("\n" + "=" * 70)
    print("Q42 TEST 1: SEMANTIC CHSH - BELL INEQUALITY IN SEMANTIC SPACE")
    print("=" * 70)
    print(f"\nHypotheses:")
    print(f"  H0 (Local): S ≤ {CLASSICAL_BOUND} for all pairs (A1 correct)")
    print(f"  H1 (Non-local): S > {VIOLATION_THRESHOLD} for some pairs (A1 violated)")

    results = {}

    # Test each category of pairs
    for category, pairs in ENTANGLED_PAIRS.items():
        results[category] = test_category(category, pairs)

    # Projection angle scan
    results['angle_scan'] = test_optimal_projection_scan()

    # Correlation strength sweep
    results['strength_sweep'] = test_correlation_strength_sweep()

    # Overall analysis
    print("\n" + "=" * 70)
    print("SUMMARY: Semantic CHSH Results")
    print("=" * 70)

    all_S = []
    all_violations = 0
    all_significant = 0

    for category in ENTANGLED_PAIRS.keys():
        cat_result = results[category]
        all_S.extend([p['S'] for p in cat_result['pairs']])
        all_violations += cat_result['summary']['n_violations']
        all_significant += cat_result['summary']['n_significant']

        print(f"\n{category}:")
        print(f"  Mean S: {cat_result['summary']['mean_S']:.4f}")
        print(f"  Max S: {cat_result['summary']['max_S']:.4f}")
        print(f"  Violations: {cat_result['summary']['n_violations']}/{cat_result['summary']['n_pairs']}")

    # Final verdict
    print("\n" + "-" * 70)

    max_S = max(all_S)
    mean_S = np.mean(all_S)

    if all_significant > 0:
        verdict = "H1 CONFIRMED: Semantic space shows Bell inequality violation"
        h0_status = "REJECTED"
        h1_status = "CONFIRMED"
    elif all_violations > 0:
        verdict = "INCONCLUSIVE: Violations present but not significant"
        h0_status = "UNCERTAIN"
        h1_status = "UNCERTAIN"
    else:
        verdict = "H0 CONFIRMED: No Bell inequality violations detected"
        h0_status = "CONFIRMED"
        h1_status = "REJECTED"

    print(f"\nMax S across all pairs: {max_S:.4f}")
    print(f"Mean S across all pairs: {mean_S:.4f}")
    print(f"Total violations: {all_violations}")
    print(f"Significant violations: {all_significant}")
    print(f"\nVERDICT: {verdict}")
    print(f"H0 (Locality): {h0_status}")
    print(f"H1 (Non-locality): {h1_status}")

    results['summary'] = {
        'max_S': float(max_S),
        'mean_S': float(mean_S),
        'total_pairs': len(all_S),
        'total_violations': all_violations,
        'significant_violations': all_significant,
        'verdict': verdict,
        'h0_status': h0_status,
        'h1_status': h1_status,
        'classical_bound': CLASSICAL_BOUND,
        'timestamp': datetime.now().isoformat()
    }

    return results


if __name__ == '__main__':
    results = run_all_tests()

    # Save results
    output_path = Path(__file__).parent / 'q42_test1_results.json'
    with open(output_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")

    # Exit code: 0 if H0 confirmed, 1 if H1 confirmed, 2 if inconclusive
    if results['summary']['h0_status'] == 'CONFIRMED':
        sys.exit(0)
    elif results['summary']['h1_status'] == 'CONFIRMED':
        sys.exit(1)
    else:
        sys.exit(2)
