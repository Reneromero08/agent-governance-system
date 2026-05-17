#!/usr/bin/env python3
"""
Universal Semantic Anchor Hypothesis Test

Tests whether different embedding models converge on the same relative
distances for simple anchor words (Platonic Representation Hypothesis).

If models converge, it enables cross-model semantic alignment via
distance matrix calibration.

Usage:
    python semantic_anchor_test.py                    # Run all tests
    python semantic_anchor_test.py --quick            # Quick test (2 models)
    python semantic_anchor_test.py --model MODEL_NAME # Test specific model
    python semantic_anchor_test.py --extended         # Extended anchor set

Open Source Models Tested:
    - all-MiniLM-L6-v2 (reference, 384d)
    - e5-large-v2 (Microsoft, 1024d)
    - bge-large-en-v1.5 (BAAI, 1024d)
    - gte-large (Alibaba, 1024d)
    - all-mpnet-base-v2 (sentence-transformers, 768d)

Dependencies:
    pip install sentence-transformers numpy scipy
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Anchor word sets
ANCHOR_WORDS_CORE = ['dog', 'love', 'up', 'true', 'king']

ANCHOR_WORDS_EXTENDED = [
    # Concrete objects
    'dog', 'cat', 'car', 'tree', 'water', 'fire',
    # Abstract concepts
    'love', 'hate', 'truth', 'beauty', 'justice',
    # Spatial relations
    'up', 'down', 'near', 'far',
    # Logical concepts
    'true', 'false',
    # Social roles
    'king', 'queen', 'parent', 'child',
    # Colors
    'red', 'blue', 'green',
]

# Open source models to test (HuggingFace model IDs)
OPEN_SOURCE_MODELS = {
    'all-MiniLM-L6-v2': {
        'id': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'license': 'Apache-2.0',
        'is_reference': True,
    },
    'all-mpnet-base-v2': {
        'id': 'sentence-transformers/all-mpnet-base-v2',
        'dim': 768,
        'license': 'Apache-2.0',
        'is_reference': False,
    },
    'e5-large-v2': {
        'id': 'intfloat/e5-large-v2',
        'dim': 1024,
        'license': 'Apache-2.0',
        'is_reference': False,
        'prefix': 'query: ',  # E5 requires prefix
    },
    'bge-large-en-v1.5': {
        'id': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'license': 'MIT',
        'is_reference': False,
    },
    'gte-large': {
        'id': 'thenlper/gte-large',
        'dim': 1024,
        'license': 'MIT',
        'is_reference': False,
    },
}

# Global model cache
_model_cache: Dict[str, any] = {}


def load_model(model_name: str):
    """Load a sentence-transformers model (with caching)."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    model_info = OPEN_SOURCE_MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  Loading {model_name}...", end=" ", flush=True)
    start = time.time()
    model = SentenceTransformer(model_info['id'])
    elapsed = time.time() - start
    print(f"({elapsed:.1f}s)")

    _model_cache[model_name] = model
    return model


def get_embeddings(model_name: str, words: List[str]) -> np.ndarray:
    """Get embeddings for a list of words."""
    model = load_model(model_name)
    model_info = OPEN_SOURCE_MODELS[model_name]

    # Some models require prefixes
    prefix = model_info.get('prefix', '')
    if prefix:
        words = [prefix + w for w in words]

    embeddings = model.encode(words, normalize_embeddings=True)
    return embeddings


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (embeddings assumed normalized)."""
    return embeddings @ embeddings.T


def compare_matrices(D1: np.ndarray, D2: np.ndarray) -> Dict:
    """Compare two distance matrices."""
    # Flatten for correlation (exclude diagonal which is always 1.0)
    n = D1.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flat1 = D1[mask]
    flat2 = D2[mask]

    # Metrics
    frobenius = np.linalg.norm(D1 - D2, 'fro')
    pearson_r, pearson_p = pearsonr(flat1, flat2)
    spearman_r, spearman_p = spearmanr(flat1, flat2)
    max_diff = np.max(np.abs(D1 - D2))
    mean_diff = np.mean(np.abs(D1 - D2))

    return {
        'frobenius_norm': float(frobenius),
        'pearson_correlation': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
    }


def format_matrix(matrix: np.ndarray, labels: List[str], precision: int = 4) -> str:
    """Format a matrix for display."""
    n = len(labels)
    max_label = max(len(l) for l in labels)

    # Header
    header = " " * (max_label + 1) + "  ".join(f"{l:>{precision+2}}" for l in labels)
    lines = [header]

    # Rows
    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{matrix[i,j]:.{precision}f}" for j in range(n))
        lines.append(f"{label:>{max_label}} {row_vals}")

    return "\n".join(lines)


def test_model(
    model_name: str,
    words: List[str],
    reference_matrix: Optional[np.ndarray] = None
) -> Dict:
    """Test a model and compare to reference."""
    embeddings = get_embeddings(model_name, words)
    matrix = compute_distance_matrix(embeddings)

    result = {
        'model': model_name,
        'info': OPEN_SOURCE_MODELS[model_name],
        'words': words,
        'matrix': matrix.tolist(),
    }

    if reference_matrix is not None:
        comparison = compare_matrices(reference_matrix, matrix)
        result['comparison'] = comparison

    return result


def run_cross_model_test(
    models: List[str],
    words: List[str],
    verbose: bool = True
) -> Dict:
    """Run cross-model comparison test."""
    results = {}
    matrices = {}

    if verbose:
        print(f"\nTesting {len(models)} models with {len(words)} anchor words")
        print("=" * 60)

    # Get all matrices
    for model_name in models:
        embeddings = get_embeddings(model_name, words)
        matrices[model_name] = compute_distance_matrix(embeddings)

    # Reference model
    ref_model = 'all-MiniLM-L6-v2'
    ref_matrix = matrices[ref_model]

    if verbose:
        print(f"\nReference Model: {ref_model}")
        print(f"Reference Distance Matrix:")
        print(format_matrix(ref_matrix, words))

    # Compare all models to reference
    comparisons = {}
    for model_name, matrix in matrices.items():
        if model_name == ref_model:
            comparisons[model_name] = {
                'pearson_correlation': 1.0,
                'spearman_correlation': 1.0,
                'frobenius_norm': 0.0,
            }
        else:
            comparisons[model_name] = compare_matrices(ref_matrix, matrix)

    if verbose:
        print(f"\n{'Model':<25} {'Pearson r':>12} {'Spearman r':>12} {'Frobenius':>12}")
        print("-" * 63)
        for model_name in models:
            c = comparisons[model_name]
            suffix = " (ref)" if model_name == ref_model else ""
            print(f"{model_name:<25} {c['pearson_correlation']:>12.4f} {c['spearman_correlation']:>12.4f} {c['frobenius_norm']:>12.4f}{suffix}")

    # Cross-model correlation matrix
    n_models = len(models)
    cross_correlation = np.zeros((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                cross_correlation[i, j] = 1.0
            else:
                comp = compare_matrices(matrices[m1], matrices[m2])
                cross_correlation[i, j] = comp['pearson_correlation']

    if verbose:
        print(f"\nCross-Model Correlation Matrix:")
        short_names = [m[:12] for m in models]
        print(format_matrix(cross_correlation, short_names, precision=3))

    # Convergence assessment
    correlations = [c['pearson_correlation'] for m, c in comparisons.items() if m != ref_model]
    strong = sum(1 for r in correlations if r > 0.95)
    moderate = sum(1 for r in correlations if 0.8 <= r <= 0.95)
    weak = sum(1 for r in correlations if r < 0.8)

    assessment = {
        'strong_convergence': strong,
        'moderate_convergence': moderate,
        'weak_convergence': weak,
        'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
        'min_correlation': float(np.min(correlations)) if correlations else 0.0,
        'max_correlation': float(np.max(correlations)) if correlations else 0.0,
    }

    if verbose:
        print(f"\nPlatonic Convergence Assessment:")
        print(f"  Strong convergence (r > 0.95): {strong} models")
        print(f"  Moderate convergence (0.8 <= r <= 0.95): {moderate} models")
        print(f"  Weak convergence (r < 0.8): {weak} models")
        print(f"  Mean correlation: {assessment['mean_correlation']:.4f}")
        print(f"  Min correlation: {assessment['min_correlation']:.4f}")

    return {
        'models': models,
        'words': words,
        'reference_model': ref_model,
        'matrices': {k: v.tolist() for k, v in matrices.items()},
        'comparisons': comparisons,
        'cross_correlation': cross_correlation.tolist(),
        'assessment': assessment,
    }


def test_analogy(model_name: str, a: str, b: str, c: str, expected: str) -> Dict:
    """Test classic word analogy: a - b + c ≈ expected."""
    words = [a, b, c, expected]
    embeddings = get_embeddings(model_name, words)

    # a - b + c
    analogy_vec = embeddings[0] - embeddings[1] + embeddings[2]
    analogy_vec = analogy_vec / np.linalg.norm(analogy_vec)

    # Similarity to expected
    similarity = float(analogy_vec @ embeddings[3])

    return {
        'analogy': f"{a} - {b} + {c} ≈ {expected}",
        'similarity': similarity,
    }


def main():
    parser = argparse.ArgumentParser(description="Universal Semantic Anchor Hypothesis Test")
    parser.add_argument('--quick', action='store_true', help="Quick test (2 models)")
    parser.add_argument('--model', type=str, help="Test specific model")
    parser.add_argument('--extended', action='store_true', help="Use extended anchor set")
    parser.add_argument('--output', type=str, help="Output JSON file")
    parser.add_argument('--verbose', action='store_true', default=True, help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("Universal Semantic Anchor Hypothesis Test")
    print("Testing Platonic Representation Convergence")
    print("=" * 60)

    # Select words
    words = ANCHOR_WORDS_EXTENDED if args.extended else ANCHOR_WORDS_CORE
    print(f"\nAnchor words: {words}")

    # Select models
    if args.model:
        models = ['all-MiniLM-L6-v2', args.model]
    elif args.quick:
        models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
    else:
        models = list(OPEN_SOURCE_MODELS.keys())

    print(f"Models to test: {models}")

    # Run test
    print("\nLoading models...")
    results = run_cross_model_test(models, words, verbose=args.verbose)

    # Classic analogy tests
    print("\n" + "=" * 60)
    print("Classic Analogy Tests: king - man + woman ≈ queen")
    print("-" * 60)
    for model_name in models:
        analogy = test_analogy(model_name, 'king', 'man', 'woman', 'queen')
        print(f"  {model_name}: {analogy['similarity']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("HYPOTHESIS VERDICT")
    print("=" * 60)

    mean_corr = results['assessment']['mean_correlation']
    if mean_corr > 0.95:
        verdict = "STRONGLY SUPPORTED"
        explanation = "Models show strong convergence on anchor distances"
    elif mean_corr > 0.8:
        verdict = "SUPPORTED"
        explanation = "Models show moderate convergence - calibration may help"
    else:
        verdict = "WEAK SUPPORT"
        explanation = "Models diverge significantly - transformation needed"

    print(f"\n  Platonic Convergence: {verdict}")
    print(f"  Mean correlation: {mean_corr:.4f}")
    print(f"  Interpretation: {explanation}")

    if mean_corr > 0.8:
        print("\n  Implication: Universal Semantic Anchors are VIABLE")
        print("  Cross-model alignment via distance matrix calibration is possible")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    main()
