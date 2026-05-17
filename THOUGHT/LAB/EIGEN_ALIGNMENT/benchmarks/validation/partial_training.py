#!/usr/bin/env python3
"""
E.X.3.3b: Partial Training Checkpoint Analysis

Tests when semantic structure emerges during training by interpolating
between untrained and trained BERT weights.

Key questions:
1. Phase transition? Does generalization appear suddenly or gradually?
2. J-generalization decoupling? When does high J start predicting generalization?
3. Effective dimensionality trajectory? 99 → 62 → 22 linear or step?
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.mds import squared_distance_matrix, classical_mds
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel, BertConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# Anchor and held-out words (same as other tests)
ANCHOR_WORDS = [
    "time", "space", "energy", "matter", "light",
    "force", "motion", "wave", "particle", "field",
    "truth", "beauty", "justice", "freedom", "power",
    "knowledge", "wisdom", "love", "fear", "hope",
    "water", "fire", "earth", "air", "stone",
    "tree", "river", "mountain", "ocean", "star",
    "music", "art", "science", "math", "language",
    "thought", "memory", "dream", "reason", "emotion",
    "birth", "death", "growth", "change", "balance",
    "order", "chaos", "harmony", "conflict", "peace",
    "king", "queen", "soldier", "priest", "farmer",
    "gold", "silver", "iron", "copper", "salt",
    "bread", "wine", "milk", "honey", "grain",
]

HELD_OUT_WORDS = [
    "gravity", "velocity", "friction", "momentum", "inertia",
    "honor", "courage", "loyalty", "mercy", "virtue",
    "thunder", "lightning", "rainbow", "eclipse", "comet",
    "bridge", "tower", "castle", "temple", "garden",
    "sword", "shield", "arrow", "spear", "armor",
    "winter", "summer", "spring", "autumn", "harvest",
    "dragon", "phoenix", "unicorn", "griffin", "serpent",
    "prophet", "oracle", "sage", "wizard", "knight",
    "diamond", "ruby", "emerald", "sapphire", "pearl",
    "sunrise", "sunset", "midnight", "twilight", "dawn",
]


def compute_effective_dimensionality(embeddings: dict) -> dict:
    """Compute effective dimensionality metrics."""
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])
    vecs_centered = vecs - vecs.mean(axis=0)

    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    participation_ratio = (sum_lambda ** 2) / sum_lambda_sq

    total_var = np.sum(eigenvalues)
    top_10_var = np.sum(eigenvalues[:10]) / total_var

    probs = eigenvalues / sum_lambda
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(eigenvalues))
    normalized_entropy = entropy / max_entropy

    return {
        'participation_ratio': float(participation_ratio),
        'top_10_variance': float(top_10_var),
        'eigenvalue_entropy': float(normalized_entropy),
    }


def compute_neighbor_coupling(held_out_embs: np.ndarray, anchor_embs: np.ndarray, k: int = 5) -> float:
    """J-style metric: Mean cosine similarity to k nearest anchors."""
    sim_matrix = held_out_embs @ anchor_embs.T
    neighbor_sims = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        top_k_indices = np.argsort(-row)[:k]
        top_k_sims = row[top_k_indices]
        neighbor_sims.append(float(np.mean(top_k_sims)))
    return float(np.mean(neighbor_sims))


def compute_held_out_alignment(emb_a: dict, emb_b: dict, anchor_words: list, held_out_words: list) -> dict:
    """Compute alignment metrics including generalization to held-out words."""
    anchor_words = [w for w in anchor_words if w in emb_a and w in emb_b]
    held_out_words = [w for w in held_out_words if w in emb_a and w in emb_b]

    n_anchors = len(anchor_words)
    n_held_out = len(held_out_words)

    X_anchor_a = np.array([emb_a[w] for w in anchor_words])
    X_anchor_b = np.array([emb_b[w] for w in anchor_words])
    X_held_a = np.array([emb_a[w] for w in held_out_words])
    X_held_b = np.array([emb_b[w] for w in held_out_words])

    # MDS on anchors
    D2_a = squared_distance_matrix(X_anchor_a)
    D2_b = squared_distance_matrix(X_anchor_b)
    coords_anchor_a, eigenvalues_a, eigenvectors_a = classical_mds(D2_a)
    coords_anchor_b, eigenvalues_b, eigenvectors_b = classical_mds(D2_b)

    # Match dimensions
    k = min(coords_anchor_a.shape[1], coords_anchor_b.shape[1])
    coords_anchor_a = coords_anchor_a[:, :k]
    coords_anchor_b = coords_anchor_b[:, :k]
    eigenvalues_a = eigenvalues_a[:k]
    eigenvalues_b = eigenvalues_b[:k]
    eigenvectors_a = eigenvectors_a[:, :k]
    eigenvectors_b = eigenvectors_b[:, :k]

    # Procrustes alignment
    R, residual = procrustes_align(coords_anchor_a, coords_anchor_b)

    # Compute squared distances from held-out to anchors
    d2_held_to_anchor_a = np.zeros((n_held_out, n_anchors))
    d2_held_to_anchor_b = np.zeros((n_held_out, n_anchors))
    for i in range(n_held_out):
        for j in range(n_anchors):
            d2_held_to_anchor_a[i, j] = np.sum((X_held_a[i] - X_anchor_a[j])**2)
            d2_held_to_anchor_b[i, j] = np.sum((X_held_b[i] - X_anchor_b[j])**2)

    # Project held-out
    coords_held_a = out_of_sample_mds(d2_held_to_anchor_a, D2_a, eigenvectors_a, eigenvalues_a)
    coords_held_b = out_of_sample_mds(d2_held_to_anchor_b, D2_b, eigenvectors_b, eigenvalues_b)

    # Align held-out
    coords_held_a_aligned = coords_held_a @ R

    # Compute similarities
    aligned_sims = []
    for i in range(n_held_out):
        aligned_sims.append(cosine_similarity(coords_held_a_aligned[i], coords_held_b[i]))

    # J coupling
    j_coupling = compute_neighbor_coupling(X_held_a, X_anchor_a, k=5)

    return {
        'held_out_aligned_similarity': float(np.mean(aligned_sims)),
        'j_coupling': float(j_coupling),
        'n_anchors': n_anchors,
        'n_held_out': n_held_out,
    }


def get_bert_embeddings(words: list, model, tokenizer) -> dict:
    """Get embeddings from a BERT model."""
    embeddings = {}
    model.eval()

    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            vec = outputs.last_hidden_state[0, 0, :].numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings[word] = vec

    return embeddings


def interpolate_models(untrained_model, trained_model, alpha: float):
    """
    Create interpolated model: weights = alpha * trained + (1-alpha) * untrained

    alpha = 0: fully untrained
    alpha = 1: fully trained
    """
    interpolated_state = {}
    untrained_state = untrained_model.state_dict()
    trained_state = trained_model.state_dict()

    for key in untrained_state:
        interpolated_state[key] = alpha * trained_state[key] + (1 - alpha) * untrained_state[key]

    # Create new model with interpolated weights
    config = BertConfig.from_pretrained('bert-base-uncased')
    interpolated_model = BertModel(config)
    interpolated_model.load_state_dict(interpolated_state)

    return interpolated_model


def main():
    parser = argparse.ArgumentParser(description='E.X.3.3b: Partial Training Analysis')
    parser.add_argument('--checkpoints', type=str, default='0,0.1,0.25,0.5,0.75,0.9,1.0',
                        help='Comma-separated training fractions (0=untrained, 1=trained)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers not available")
        return 1

    checkpoints = [float(x) for x in args.checkpoints.split(',')]

    print("=" * 70)
    print("E.X.3.3b: Partial Training Checkpoint Analysis")
    print("=" * 70)
    print()
    print(f"Checkpoints: {checkpoints}")
    print(f"Words: {len(ANCHOR_WORDS)} anchors, {len(HELD_OUT_WORDS)} held-out")
    print()

    all_words = ANCHOR_WORDS + HELD_OUT_WORDS

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Untrained model (random weights)
    print("  Creating untrained BERT...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = BertConfig.from_pretrained('bert-base-uncased')
    untrained_model = BertModel(config)  # Random weights

    # Trained model
    print("  Loading trained BERT...")
    trained_model = BertModel.from_pretrained('bert-base-uncased')

    print()

    results = {
        'test_id': 'partial-training-E.X.3.3b',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'checkpoints': [],
    }

    # Test each checkpoint
    print("-" * 70)
    print("Testing checkpoints...")
    print("-" * 70)
    print()
    print(f"{'Alpha':>6} | {'Df':>6} | {'Top10':>6} | {'J':>6} | {'Held-out':>8}")
    print("-" * 50)

    for alpha in checkpoints:
        # Create interpolated model
        if alpha == 0:
            model = untrained_model
        elif alpha == 1:
            model = trained_model
        else:
            model = interpolate_models(untrained_model, trained_model, alpha)

        # Get embeddings
        embeddings = get_bert_embeddings(all_words, model, tokenizer)

        # Compute metrics
        eff_dim = compute_effective_dimensionality(embeddings)

        # For alignment, we need a second model - use trained as reference
        trained_embeddings = get_bert_embeddings(all_words, trained_model, tokenizer)
        alignment = compute_held_out_alignment(embeddings, trained_embeddings, ANCHOR_WORDS, HELD_OUT_WORDS)

        checkpoint_result = {
            'alpha': alpha,
            'effective_dimensionality': eff_dim,
            'alignment': alignment,
        }
        results['checkpoints'].append(checkpoint_result)

        print(f"{alpha:6.2f} | {eff_dim['participation_ratio']:6.1f} | {eff_dim['top_10_variance']:6.3f} | "
              f"{alignment['j_coupling']:6.3f} | {alignment['held_out_aligned_similarity']:8.3f}")

    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Extract trajectories
    alphas = [c['alpha'] for c in results['checkpoints']]
    dfs = [c['effective_dimensionality']['participation_ratio'] for c in results['checkpoints']]
    top10s = [c['effective_dimensionality']['top_10_variance'] for c in results['checkpoints']]
    js = [c['alignment']['j_coupling'] for c in results['checkpoints']]
    held_outs = [c['alignment']['held_out_aligned_similarity'] for c in results['checkpoints']]

    # Find phase transition (if any)
    # Look for largest single-step change in held-out generalization
    max_jump = 0
    jump_alpha = None
    for i in range(1, len(held_outs)):
        jump = held_outs[i] - held_outs[i-1]
        if jump > max_jump:
            max_jump = jump
            jump_alpha = (alphas[i-1], alphas[i])

    if max_jump > 0.1:
        print(f"[PHASE TRANSITION?] Largest generalization jump: +{max_jump:.3f}")
        print(f"  Between alpha={jump_alpha[0]:.2f} and alpha={jump_alpha[1]:.2f}")
    else:
        print(f"[GRADUAL] No large phase transition detected (max jump: {max_jump:.3f})")

    print()

    # Df trajectory
    print(f"Effective Dimensionality trajectory:")
    print(f"  Untrained (a=0): {dfs[0]:.1f}")
    print(f"  Midpoint (a=0.5): {dfs[len(dfs)//2]:.1f}" if len(dfs) > 2 else "")
    print(f"  Trained (a=1):   {dfs[-1]:.1f}")

    # Check if linear or step
    if len(dfs) >= 3:
        # Linear fit
        from scipy.stats import linregress
        slope, intercept, r, p, se = linregress(alphas, dfs)
        print(f"  Linear fit: R^2 = {r**2:.3f}")
        if r**2 > 0.9:
            print("  -> Df changes LINEARLY with training")
        else:
            print("  -> Df changes NON-LINEARLY (possible phase transition)")

    print()

    # J vs generalization correlation
    from scipy.stats import spearmanr
    j_gen_corr, _ = spearmanr(js, held_outs)
    print(f"J-Generalization correlation: rho = {j_gen_corr:.3f}")
    if j_gen_corr > 0.8:
        print("  -> J predicts generalization well during training")
    elif j_gen_corr < 0.3:
        print("  -> J does NOT predict generalization (as expected from E.X.3.3)")

    print()

    results['analysis'] = {
        'phase_transition': {
            'max_jump': max_jump,
            'jump_between': jump_alpha,
            'detected': max_jump > 0.1,
        },
        'df_linearity': r**2 if len(dfs) >= 3 else None,
        'j_gen_correlation': j_gen_corr,
    }

    # Save results
    output_path = args.output or str(Path(__file__).parent / 'results' / 'partial_training.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
