#!/usr/bin/env python3
"""
Q34: Platonic Convergence Test - Sentence Transformers

Tests whether sentence-transformer models (trained for semantic similarity)
converge better than base language models.

These models are specifically optimized for embedding quality, so they
should show stronger Platonic convergence if the hypothesis is correct.
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'benchmarks' / 'validation'))

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from untrained_transformer import ANCHOR_WORDS, HELD_OUT_WORDS


def get_st_embeddings(words: list, model_name: str) -> tuple:
    """Get embeddings from a sentence-transformer model."""
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)

    # Get embeddings for single words
    embeddings = model.encode(words, normalize_embeddings=True)

    # Convert to dict
    emb_dict = {word: embeddings[i] for i, word in enumerate(words)}
    dim = embeddings.shape[1]

    return emb_dict, dim


def compute_eigenspectrum(embeddings: dict) -> np.ndarray:
    """Compute covariance eigenspectrum."""
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Center and compute covariance
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    return eigenvalues


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute participation ratio Df."""
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    return (sum_lambda ** 2) / sum_lambda_sq


def spectrum_correlation(spec1: np.ndarray, spec2: np.ndarray, k: int = 50) -> float:
    """Compute correlation between two eigenspectra."""
    k = min(k, len(spec1), len(spec2))
    return np.corrcoef(spec1[:k], spec2[:k])[0, 1]


def main():
    print("=" * 70)
    print("Q34: PLATONIC CONVERGENCE - SENTENCE TRANSFORMERS")
    print("Testing models specifically trained for semantic similarity")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    if not ST_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        print("Install with: pip install sentence-transformers")
        return 1

    # Sentence transformer models to test
    models = [
        "all-MiniLM-L6-v2",           # 384D, very popular
        "all-mpnet-base-v2",          # 768D, high quality
        "paraphrase-MiniLM-L6-v2",    # 384D, paraphrase trained
        "multi-qa-MiniLM-L6-cos-v1",  # 384D, QA trained
    ]

    all_words = sorted(list(set(ANCHOR_WORDS + HELD_OUT_WORDS)))
    print(f"Words: {len(all_words)}")
    print()

    # Collect embeddings and spectra
    print("-" * 70)
    print("Loading models and computing eigenspectra...")
    print("-" * 70)

    spectra = {}
    dfs = {}
    dims = {}

    for model_name in models:
        try:
            embeddings, dim = get_st_embeddings(all_words, model_name)
            eigenvalues = compute_eigenspectrum(embeddings)
            df = participation_ratio(eigenvalues)

            spectra[model_name] = eigenvalues
            dfs[model_name] = df
            dims[model_name] = dim

            print(f"    {model_name}: dim={dim}, Df={df:.2f}")
        except Exception as e:
            print(f"    {model_name}: FAILED ({e})")

    print()

    if len(spectra) < 2:
        print("Not enough models loaded for comparison")
        return 1

    # Compare spectra
    print("-" * 70)
    print("Cross-Model Eigenvalue Correlations")
    print("-" * 70)
    print()

    model_names = list(spectra.keys())
    n_models = len(model_names)

    # Need to handle different dimensions - compare normalized spectra
    # Normalize each spectrum to sum to 1
    normalized_spectra = {}
    for name, spec in spectra.items():
        normalized_spectra[name] = spec / np.sum(spec)

    # Correlation matrix
    corr_matrix = np.zeros((n_models, n_models))

    print(f"{'':25}", end="")
    for name in model_names:
        short_name = name[:12]
        print(f"{short_name:>14}", end="")
    print()

    for i, name1 in enumerate(model_names):
        short_name1 = name1[:25]
        print(f"{short_name1:25}", end="")
        for j, name2 in enumerate(model_names):
            # Use minimum length for correlation
            k = min(50, len(spectra[name1]), len(spectra[name2]))
            n1 = normalized_spectra[name1][:k]
            n2 = normalized_spectra[name2][:k]
            corr = np.corrcoef(n1, n2)[0, 1]
            corr_matrix[i, j] = corr
            print(f"{corr:14.4f}", end="")
        print()

    print()

    # Summary
    off_diagonal = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            off_diagonal.append(corr_matrix[i, j])

    mean_corr = np.mean(off_diagonal)
    min_corr = np.min(off_diagonal)
    std_corr = np.std(off_diagonal)

    print("-" * 70)
    print("Summary")
    print("-" * 70)
    print()
    print(f"Mean cross-model correlation: {mean_corr:.4f}")
    print(f"Min cross-model correlation:  {min_corr:.4f}")
    print(f"Std cross-model correlation:  {std_corr:.4f}")
    print()

    print("Participation Ratios (Df):")
    for name in model_names:
        print(f"  {name:30} dim={dims[name]:3}, Df = {dfs[name]:.2f}")

    df_values = list(dfs.values())
    df_mean = np.mean(df_values)
    df_std = np.std(df_values)
    print()
    print(f"Df mean: {df_mean:.2f} +/- {df_std:.2f}")
    print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if mean_corr > 0.95:
        print(f"[STRONG] Mean correlation = {mean_corr:.4f} (>0.95)")
        print("         Sentence transformers show STRONG spectral convergence!")
        status = "STRONG"
    elif mean_corr > 0.9:
        print(f"[PASS] Mean correlation = {mean_corr:.4f} (>0.9)")
        print("       Sentence transformers converge to similar spectral structure!")
        status = "CONFIRMED"
    elif mean_corr > 0.7:
        print(f"[PARTIAL] Mean correlation = {mean_corr:.4f} (0.7-0.9)")
        print("          Moderate spectral convergence.")
        status = "PARTIAL"
    else:
        print(f"[FAIL] Mean correlation = {mean_corr:.4f} (<0.7)")
        print("       Models have different spectral structures.")
        status = "NOT_CONFIRMED"

    print()

    # Compare to base models (from previous test)
    print("Comparison to base language models (from earlier test):")
    print("  Base models (BERT/RoBERTa/DistilBERT/ALBERT): mean corr = 0.852")
    print(f"  Sentence transformers: mean corr = {mean_corr:.3f}")
    if mean_corr > 0.852:
        print("  -> Sentence transformers converge BETTER than base models")
    else:
        print("  -> Base models showed similar or better convergence")
    print()

    # Receipt
    receipt = {
        "test": "Q34_SENTENCE_TRANSFORMERS",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "dimensions": dims,
        "participation_ratios": {k: float(v) for k, v in dfs.items()},
        "mean_correlation": float(mean_corr),
        "min_correlation": float(min_corr),
        "status": status,
    }

    receipt_json = json.dumps(receipt, indent=2)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")

    return receipt


if __name__ == '__main__':
    main()
