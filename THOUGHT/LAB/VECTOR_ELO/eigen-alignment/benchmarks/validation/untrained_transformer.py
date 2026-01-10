#!/usr/bin/env python3
"""E.X.3.3: Untrained Transformer Test

Goal: Test if training creates the J coupling structure, or if architecture alone provides it.

Hypothesis:
- If untrained transformer J ≈ random (0.09) → Training creates the structure
- If untrained transformer J ≈ trained (0.39) → Architecture alone provides structure

This is a critical falsification test for the claim that trained models have
special semantic structure.

Usage:
    python -m benchmarks.validation.untrained_transformer
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.validation.held_out_resistance import (
    ANCHOR_WORDS,
    HELD_OUT_WORDS,
    compute_held_out_alignment,
    generate_random_embeddings,
)

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from collections import Counter


def compute_effective_dimensionality(embeddings: dict) -> dict:
    """
    Compute effective dimensionality metrics from embedding covariance eigenvalues.

    For a hypersphere, eigenvalues should be roughly uniform.
    For trained embeddings, we expect structure (non-uniform eigenvalues).

    Returns:
        dict with:
        - participation_ratio: (Σλ)² / Σλ² - how many dimensions are "active"
        - top_k_variance: fraction of variance in top k eigenvalues
        - eigenvalue_entropy: normalized entropy of eigenvalue distribution
    """
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Center the data
    vecs_centered = vecs - vecs.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid negative/zero

    # Participation ratio: (Σλ)² / Σλ²
    # = n for uniform distribution, = 1 for single dominant eigenvalue
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    participation_ratio = (sum_lambda ** 2) / sum_lambda_sq

    # Top-k variance (how much variance in first k dims)
    total_var = np.sum(eigenvalues)
    top_10_var = np.sum(eigenvalues[:10]) / total_var
    top_50_var = np.sum(eigenvalues[:50]) / total_var

    # Eigenvalue entropy (normalized)
    probs = eigenvalues / sum_lambda
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(eigenvalues))
    normalized_entropy = entropy / max_entropy

    return {
        'participation_ratio': float(participation_ratio),
        'top_10_variance': float(top_10_var),
        'top_50_variance': float(top_50_var),
        'eigenvalue_entropy': float(normalized_entropy),
        'n_dims': len(eigenvalues),
    }


def compute_embedding_phi(embeddings: dict, n_dims: int = 20, n_bins: int = 8) -> float:
    """
    Compute IIT-style Phi (Multi-Information) over embedding dimensions.

    Phi = Sum(H(dim_i)) - H(joint)

    This measures whether embedding dimensions coordinate together (high Phi)
    or are independent (low Phi = 0).

    Args:
        embeddings: dict of word -> embedding vector
        n_dims: number of dimensions to use (first n_dims for efficiency)
        n_bins: bins for discretization

    Returns:
        Phi in bits
    """
    # Build data matrix: words x dimensions
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Use first n_dims dimensions (full 768 dims makes joint entropy intractable)
    n_dims = min(n_dims, vecs.shape[1])
    data = vecs[:, :n_dims]

    n_samples, n_vars = data.shape

    # Bin edges from data range
    data_min = data.min()
    data_max = data.max()
    bins = np.linspace(data_min - 0.1, data_max + 0.1, n_bins + 1)

    # Sum of individual entropies
    sum_h_parts = 0
    for i in range(n_vars):
        counts, _ = np.histogram(data[:, i], bins=bins)
        probs = counts[counts > 0] / n_samples
        h = -np.sum(probs * np.log2(probs + 1e-10))
        sum_h_parts += h

    # Joint entropy via digitization
    digitized = np.zeros_like(data, dtype=int)
    for i in range(n_vars):
        digitized[:, i] = np.digitize(data[:, i], bins)

    rows = [tuple(row) for row in digitized]
    counts = Counter(rows)
    probs = np.array([c / n_samples for c in counts.values()])
    h_joint = -np.sum(probs * np.log2(probs + 1e-10))

    return float(max(0.0, sum_h_parts - h_joint))


def get_untrained_bert_embeddings(words: list, model_name: str = "bert-base-uncased") -> tuple:
    """Get embeddings from an UNTRAINED BERT model (random weights).

    This loads the architecture but reinitializes all weights randomly.
    """
    # Load config but create model with random weights
    config = AutoConfig.from_pretrained(model_name)

    # Create model with random initialization (not from pretrained)
    model = AutoModel.from_config(config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            # Use CLS token embedding
            vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings[word] = vec

    return embeddings, config.hidden_size


def get_trained_bert_embeddings(words: list, model_name: str = "bert-base-uncased") -> tuple:
    """Get embeddings from a TRAINED BERT model (pretrained weights)."""
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings[word] = vec

    return embeddings, model.config.hidden_size


def main():
    parser = argparse.ArgumentParser(description='E.X.3.3: Untrained Transformer Test')
    parser.add_argument('--n-random', type=int, default=5,
                        help='Number of random pairs for baseline')
    parser.add_argument('--n-untrained', type=int, default=3,
                        help='Number of untrained model initializations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    print("=" * 70)
    print("E.X.3.3: UNTRAINED TRANSFORMER TEST")
    print("=" * 70)
    print()
    print("Goal: Does TRAINING create J coupling, or does architecture alone?")
    print()

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available")
        return 1

    all_words = list(set(ANCHOR_WORDS + HELD_OUT_WORDS))
    print(f"Using {len(ANCHOR_WORDS)} anchor words, {len(HELD_OUT_WORDS)} held-out words")
    print()

    # Test 1: Random embeddings baseline
    print("-" * 70)
    print("Test 1: Random Embeddings (baseline)")
    print("-" * 70)

    random_results = []
    random_phis = []
    dim = 768  # BERT hidden size

    for i in range(args.n_random):
        rand_a = generate_random_embeddings(all_words, dim, args.seed + i * 2)
        rand_b = generate_random_embeddings(all_words, dim, args.seed + i * 2 + 1)
        result = compute_held_out_alignment(rand_a, rand_b, ANCHOR_WORDS, HELD_OUT_WORDS)
        random_results.append(result)
        # Compute proper IIT Phi on embedding dimensions
        phi_a = compute_embedding_phi(rand_a)
        phi_b = compute_embedding_phi(rand_b)
        phi_iit = (phi_a + phi_b) / 2
        random_phis.append(phi_iit)
        print(f"  Pair {i+1}: J={result['j_coupling_mean']:.4f}, Phi_IIT={phi_iit:.4f}, held_out={result['held_out_aligned_similarity']:.4f}")

    mean_random_j = float(np.mean([r['j_coupling_mean'] for r in random_results]))
    mean_random_phi_iit = float(np.mean(random_phis))
    mean_random_held_out = float(np.mean([r['held_out_aligned_similarity'] for r in random_results]))
    print(f"\nRandom mean: J={mean_random_j:.4f}, Phi_IIT={mean_random_phi_iit:.4f}, held_out={mean_random_held_out:.4f}")
    print()

    # Test 2: Untrained BERT
    print("-" * 70)
    print("Test 2: Untrained BERT (random weights, transformer architecture)")
    print("-" * 70)

    untrained_results = []
    untrained_phis = []

    for i in range(args.n_untrained):
        print(f"  Initialization {i+1}...")
        # Set different seeds for different random initializations
        torch.manual_seed(args.seed + i * 100)
        np.random.seed(args.seed + i * 100)

        untrained_a, _ = get_untrained_bert_embeddings(all_words)

        torch.manual_seed(args.seed + i * 100 + 50)
        np.random.seed(args.seed + i * 100 + 50)

        untrained_b, _ = get_untrained_bert_embeddings(all_words)

        result = compute_held_out_alignment(untrained_a, untrained_b, ANCHOR_WORDS, HELD_OUT_WORDS)
        untrained_results.append(result)
        # Compute proper IIT Phi on embedding dimensions
        phi_a = compute_embedding_phi(untrained_a)
        phi_b = compute_embedding_phi(untrained_b)
        phi_iit = (phi_a + phi_b) / 2
        untrained_phis.append(phi_iit)
        print(f"    J={result['j_coupling_mean']:.4f}, Phi_IIT={phi_iit:.4f}, held_out={result['held_out_aligned_similarity']:.4f}")

    mean_untrained_j = float(np.mean([r['j_coupling_mean'] for r in untrained_results]))
    mean_untrained_phi_iit = float(np.mean(untrained_phis))
    mean_untrained_held_out = float(np.mean([r['held_out_aligned_similarity'] for r in untrained_results]))
    print(f"\nUntrained mean: J={mean_untrained_j:.4f}, Phi_IIT={mean_untrained_phi_iit:.4f}, held_out={mean_untrained_held_out:.4f}")
    print()

    # Test 3: Trained BERT
    print("-" * 70)
    print("Test 3: Trained BERT (pretrained weights)")
    print("-" * 70)

    print("  Loading bert-base-uncased (trained)...")
    trained_a, _ = get_trained_bert_embeddings(all_words, "bert-base-uncased")

    # Compare to a different trained model
    if ST_AVAILABLE:
        print("  Loading all-MiniLM-L6-v2 (trained)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vectors = model.encode(all_words, convert_to_numpy=True)
        trained_b = {}
        for i, word in enumerate(all_words):
            vec = vectors[i]
            vec = vec / np.linalg.norm(vec)
            trained_b[word] = vec
    else:
        print("  Loading bert-base-cased (trained)...")
        trained_b, _ = get_trained_bert_embeddings(all_words, "bert-base-cased")

    trained_result = compute_held_out_alignment(trained_a, trained_b, ANCHOR_WORDS, HELD_OUT_WORDS)
    # Compute proper IIT Phi on embedding dimensions
    trained_phi_a = compute_embedding_phi(trained_a)
    trained_phi_b = compute_embedding_phi(trained_b)
    trained_phi_iit = (trained_phi_a + trained_phi_b) / 2
    print(f"  J={trained_result['j_coupling_mean']:.4f}, Phi_IIT={trained_phi_iit:.4f}, held_out={trained_result['held_out_aligned_similarity']:.4f}")
    print()

    # Test 4: Effective Dimensionality (Spherical Geometry Test)
    print("-" * 70)
    print("Test 4: Effective Dimensionality (Riemann Sphere Geometry)")
    print("-" * 70)
    print()

    # Compute for one random embedding
    rand_sample = generate_random_embeddings(all_words, dim, args.seed + 999)
    rand_eff_dim = compute_effective_dimensionality(rand_sample)

    # Compute for one untrained
    torch.manual_seed(args.seed + 999)
    np.random.seed(args.seed + 999)
    untrained_sample, _ = get_untrained_bert_embeddings(all_words)
    untrained_eff_dim = compute_effective_dimensionality(untrained_sample)

    # Compute for trained
    trained_eff_dim = compute_effective_dimensionality(trained_a)

    print(f"Participation Ratio (Df = effective dims):")
    print(f"  Random:    {rand_eff_dim['participation_ratio']:.1f} / {rand_eff_dim['n_dims']}")
    print(f"  Untrained: {untrained_eff_dim['participation_ratio']:.1f} / {untrained_eff_dim['n_dims']}")
    print(f"  Trained:   {trained_eff_dim['participation_ratio']:.1f} / {trained_eff_dim['n_dims']}")
    print()
    print(f"Top-10 Variance (low = spherical, high = concentrated):")
    print(f"  Random:    {rand_eff_dim['top_10_variance']:.3f}")
    print(f"  Untrained: {untrained_eff_dim['top_10_variance']:.3f}")
    print(f"  Trained:   {trained_eff_dim['top_10_variance']:.3f}")
    print()
    print(f"Eigenvalue Entropy (1.0 = uniform/spherical, 0.0 = concentrated):")
    print(f"  Random:    {rand_eff_dim['eigenvalue_entropy']:.3f}")
    print(f"  Untrained: {untrained_eff_dim['eigenvalue_entropy']:.3f}")
    print(f"  Trained:   {trained_eff_dim['eigenvalue_entropy']:.3f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"                    | Random      | Untrained   | Trained")
    print(f"--------------------|-------------|-------------|----------")
    print(f"J coupling          | {mean_random_j:.4f}      | {mean_untrained_j:.4f}      | {trained_result['j_coupling_mean']:.4f}")
    print(f"Phi_IIT (multi-info)| {mean_random_phi_iit:.4f}      | {mean_untrained_phi_iit:.4f}      | {trained_phi_iit:.4f}")
    print(f"Held-out aligned    | {mean_random_held_out:.4f}      | {mean_untrained_held_out:.4f}      | {trained_result['held_out_aligned_similarity']:.4f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Is untrained closer to random or trained?
    j_gap_to_random = abs(mean_untrained_j - mean_random_j)
    j_gap_to_trained = abs(mean_untrained_j - trained_result['j_coupling_mean'])

    held_out_gap_to_random = abs(mean_untrained_held_out - mean_random_held_out)
    held_out_gap_to_trained = abs(mean_untrained_held_out - trained_result['held_out_aligned_similarity'])

    print(f"Untrained J is closer to: ", end="")
    if j_gap_to_random < j_gap_to_trained:
        j_verdict = "RANDOM"
        print(f"RANDOM (gap={j_gap_to_random:.4f} vs {j_gap_to_trained:.4f} to trained)")
    else:
        j_verdict = "TRAINED"
        print(f"TRAINED (gap={j_gap_to_trained:.4f} vs {j_gap_to_random:.4f} to random)")

    print(f"Untrained held-out is closer to: ", end="")
    if held_out_gap_to_random < held_out_gap_to_trained:
        held_out_verdict = "RANDOM"
        print(f"RANDOM (gap={held_out_gap_to_random:.4f} vs {held_out_gap_to_trained:.4f} to trained)")
    else:
        held_out_verdict = "TRAINED"
        print(f"TRAINED (gap={held_out_gap_to_trained:.4f} vs {held_out_gap_to_random:.4f} to random)")

    print()
    if j_verdict == "RANDOM" and held_out_verdict == "RANDOM":
        final_verdict = "TRAINING CREATES STRUCTURE"
        explanation = (
            "Untrained transformer behaves like random embeddings. "
            "TRAINING induces the semantic manifold structure (high J, generalization)."
        )
    elif j_verdict == "TRAINED" and held_out_verdict == "TRAINED":
        final_verdict = "ARCHITECTURE PROVIDES STRUCTURE"
        explanation = (
            "Untrained transformer already has structure similar to trained. "
            "The transformer ARCHITECTURE alone provides semantic manifold structure."
        )
    else:
        final_verdict = "MIXED"
        explanation = (
            f"J verdict={j_verdict}, held-out verdict={held_out_verdict}. "
            "Partial structure from architecture, enhanced by training."
        )

    print(f"VERDICT: {final_verdict}")
    print()
    print(explanation)
    print()

    # Save results
    result = {
        'test_id': 'untrained-transformer-E.X.3.3',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'random': {
            'mean_j': mean_random_j,
            'mean_phi_iit': mean_random_phi_iit,
            'mean_held_out': mean_random_held_out,
        },
        'untrained': {
            'mean_j': mean_untrained_j,
            'mean_phi_iit': mean_untrained_phi_iit,
            'mean_held_out': mean_untrained_held_out,
        },
        'trained': {
            'j': trained_result['j_coupling_mean'],
            'phi_iit': trained_phi_iit,
            'held_out': trained_result['held_out_aligned_similarity'],
        },
        'gaps': {
            'untrained_to_random_j': j_gap_to_random,
            'untrained_to_trained_j': j_gap_to_trained,
            'untrained_to_random_held_out': held_out_gap_to_random,
            'untrained_to_trained_held_out': held_out_gap_to_trained,
        },
        'interpretation': {
            'j_verdict': j_verdict,
            'held_out_verdict': held_out_verdict,
            'final_verdict': final_verdict,
            'explanation': explanation,
        },
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'untrained_transformer.json'

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
