#!/usr/bin/env python3
"""
Q34: Platonic Convergence Test

Tests whether different trained models converge to the same covariance spectrum.

Hypothesis: If independent models converge to similar eigenvalue spectra,
this is evidence for Platonic convergence (unique underlying structure).

This replaces the invalid "Chern number" approach from Q43.
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
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from untrained_transformer import ANCHOR_WORDS, HELD_OUT_WORDS


def get_model_embeddings(words: list, model_name: str) -> tuple:
    """Get embeddings from a trained model."""
    print(f"  Loading {model_name}...")

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

    dim = model.config.hidden_size
    return embeddings, dim


def compute_eigenspectrum(embeddings: dict) -> np.ndarray:
    """Compute covariance eigenspectrum."""
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Center and compute covariance
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
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
    print("Q34: PLATONIC CONVERGENCE TEST")
    print("Do different models converge to the same covariance spectrum?")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available")
        return 1

    # Models to test
    models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2",
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

    for model_name in models:
        try:
            embeddings, dim = get_model_embeddings(all_words, model_name)
            eigenvalues = compute_eigenspectrum(embeddings)
            df = participation_ratio(eigenvalues)

            spectra[model_name] = eigenvalues
            dfs[model_name] = df

            print(f"    {model_name}: dim={dim}, Df={df:.2f}")
        except Exception as e:
            print(f"    {model_name}: FAILED ({e})")

    print()

    # Compare spectra
    print("-" * 70)
    print("Cross-Model Eigenvalue Correlations")
    print("-" * 70)
    print()

    model_names = list(spectra.keys())
    n_models = len(model_names)

    # Correlation matrix
    corr_matrix = np.zeros((n_models, n_models))

    print(f"{'':20}", end="")
    for name in model_names:
        short_name = name.split('/')[-1][:12]
        print(f"{short_name:>14}", end="")
    print()

    for i, name1 in enumerate(model_names):
        short_name1 = name1.split('/')[-1][:20]
        print(f"{short_name1:20}", end="")
        for j, name2 in enumerate(model_names):
            corr = spectrum_correlation(spectra[name1], spectra[name2])
            corr_matrix[i, j] = corr
            print(f"{corr:14.4f}", end="")
        print()

    print()

    # Summary statistics
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
    for name, df in dfs.items():
        short_name = name.split('/')[-1]
        print(f"  {short_name:25} Df = {df:.2f}")

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

    if mean_corr > 0.9:
        print(f"[PASS] Mean correlation = {mean_corr:.4f} (>0.9)")
        print("       Different models converge to SAME spectral structure!")
        print("       This is evidence for PLATONIC CONVERGENCE.")
        status = "CONFIRMED"
    elif mean_corr > 0.7:
        print(f"[PARTIAL] Mean correlation = {mean_corr:.4f} (0.7-0.9)")
        print("          Strong but not perfect spectral convergence.")
        status = "PARTIAL"
    else:
        print(f"[FAIL] Mean correlation = {mean_corr:.4f} (<0.7)")
        print("       Models have different spectral structures.")
        status = "NOT_CONFIRMED"

    print()

    # Receipt
    receipt = {
        "test": "Q34_PLATONIC_CONVERGENCE",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "participation_ratios": {k: float(v) for k, v in dfs.items()},
        "mean_correlation": float(mean_corr),
        "min_correlation": float(min_corr),
        "correlation_matrix": corr_matrix.tolist(),
        "status": status,
    }

    receipt_json = json.dumps(receipt, indent=2)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")
    print()

    return receipt


if __name__ == '__main__':
    main()
