#!/usr/bin/env python3
"""
Q34: Platonic Convergence - Cross-Architecture Test

Tests whether fundamentally different embedding architectures converge
to the same spectral structure:
- Count-based: GloVe
- Prediction-based: Word2Vec (skip-gram)
- Transformer-based: BERT, Sentence Transformers

If different architectures converge, this is STRONG evidence for
Platonic convergence being a property of the TASK, not the MODEL.
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

from untrained_transformer import ANCHOR_WORDS, HELD_OUT_WORDS

# Check available libraries
GENSIM_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
ST_AVAILABLE = False

try:
    import gensim.downloader as api
    GENSIM_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    pass


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
    """Compute correlation between normalized eigenspectra."""
    # Normalize to sum to 1
    n1 = spec1 / np.sum(spec1)
    n2 = spec2 / np.sum(spec2)

    k = min(k, len(n1), len(n2))
    return np.corrcoef(n1[:k], n2[:k])[0, 1]


def load_glove(words: list) -> tuple:
    """Load GloVe embeddings for given words."""
    print("  Loading GloVe (glove-wiki-gigaword-300)...")
    model = api.load("glove-wiki-gigaword-300")

    embeddings = {}
    missing = []
    for word in words:
        if word in model:
            embeddings[word] = model[word]
        else:
            missing.append(word)

    if missing:
        print(f"    Missing {len(missing)} words: {missing[:5]}...")

    return embeddings, 300


def load_word2vec(words: list) -> tuple:
    """Load Word2Vec embeddings for given words."""
    print("  Loading Word2Vec (word2vec-google-news-300)...")
    model = api.load("word2vec-google-news-300")

    embeddings = {}
    missing = []
    for word in words:
        if word in model:
            embeddings[word] = model[word]
        else:
            missing.append(word)

    if missing:
        print(f"    Missing {len(missing)} words: {missing[:5]}...")

    return embeddings, 300


def load_fasttext(words: list) -> tuple:
    """Load FastText embeddings for given words."""
    print("  Loading FastText (fasttext-wiki-news-subwords-300)...")
    model = api.load("fasttext-wiki-news-subwords-300")

    embeddings = {}
    missing = []
    for word in words:
        if word in model:
            embeddings[word] = model[word]
        else:
            missing.append(word)

    if missing:
        print(f"    Missing {len(missing)} words: {missing[:5]}...")

    return embeddings, 300


def load_bert(words: list) -> tuple:
    """Load BERT embeddings for given words."""
    print("  Loading BERT (bert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            emb = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings[word] = emb

    return embeddings, 768


def load_sentence_transformer(words: list, model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """Load sentence transformer embeddings."""
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)

    embs = model.encode(words, normalize_embeddings=True)
    embeddings = {word: embs[i] for i, word in enumerate(words)}

    return embeddings, embs.shape[1]


def main():
    print("=" * 70)
    print("Q34: PLATONIC CONVERGENCE - CROSS-ARCHITECTURE TEST")
    print("Testing: GloVe vs Word2Vec vs FastText vs BERT vs Sentence-T")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    # Check dependencies
    print("Dependencies:")
    print(f"  gensim: {'YES' if GENSIM_AVAILABLE else 'NO'}")
    print(f"  transformers: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    print(f"  sentence-transformers: {'YES' if ST_AVAILABLE else 'NO'}")
    print()

    if not GENSIM_AVAILABLE:
        print("ERROR: gensim required for GloVe/Word2Vec")
        print("Install with: pip install gensim")
        return 1

    all_words = sorted(list(set(ANCHOR_WORDS + HELD_OUT_WORDS)))
    print(f"Words: {len(all_words)}")
    print()

    # Load all models
    print("-" * 70)
    print("Loading models...")
    print("-" * 70)

    models = {}
    spectra = {}
    dfs = {}
    dims = {}

    # GloVe
    try:
        emb, dim = load_glove(all_words)
        if len(emb) >= 50:
            spec = compute_eigenspectrum(emb)
            df = participation_ratio(spec)
            models["GloVe"] = emb
            spectra["GloVe"] = spec
            dfs["GloVe"] = df
            dims["GloVe"] = dim
            print(f"    GloVe: {len(emb)} words, dim={dim}, Df={df:.2f}")
    except Exception as e:
        print(f"    GloVe: FAILED ({e})")

    # Word2Vec
    try:
        emb, dim = load_word2vec(all_words)
        if len(emb) >= 50:
            spec = compute_eigenspectrum(emb)
            df = participation_ratio(spec)
            models["Word2Vec"] = emb
            spectra["Word2Vec"] = spec
            dfs["Word2Vec"] = df
            dims["Word2Vec"] = dim
            print(f"    Word2Vec: {len(emb)} words, dim={dim}, Df={df:.2f}")
    except Exception as e:
        print(f"    Word2Vec: FAILED ({e})")

    # FastText
    try:
        emb, dim = load_fasttext(all_words)
        if len(emb) >= 50:
            spec = compute_eigenspectrum(emb)
            df = participation_ratio(spec)
            models["FastText"] = emb
            spectra["FastText"] = spec
            dfs["FastText"] = df
            dims["FastText"] = dim
            print(f"    FastText: {len(emb)} words, dim={dim}, Df={df:.2f}")
    except Exception as e:
        print(f"    FastText: FAILED ({e})")

    # BERT
    if TRANSFORMERS_AVAILABLE:
        try:
            emb, dim = load_bert(all_words)
            spec = compute_eigenspectrum(emb)
            df = participation_ratio(spec)
            models["BERT"] = emb
            spectra["BERT"] = spec
            dfs["BERT"] = df
            dims["BERT"] = dim
            print(f"    BERT: {len(emb)} words, dim={dim}, Df={df:.2f}")
        except Exception as e:
            print(f"    BERT: FAILED ({e})")

    # Sentence Transformer
    if ST_AVAILABLE:
        try:
            emb, dim = load_sentence_transformer(all_words)
            spec = compute_eigenspectrum(emb)
            df = participation_ratio(spec)
            models["SentenceT"] = emb
            spectra["SentenceT"] = spec
            dfs["SentenceT"] = df
            dims["SentenceT"] = dim
            print(f"    SentenceT: {len(emb)} words, dim={dim}, Df={df:.2f}")
        except Exception as e:
            print(f"    SentenceT: FAILED ({e})")

    print()

    if len(spectra) < 2:
        print("Not enough models loaded for comparison")
        return 1

    # Compute cross-model correlations
    print("-" * 70)
    print("Cross-Architecture Eigenvalue Correlations")
    print("-" * 70)
    print()

    model_names = list(spectra.keys())
    n_models = len(model_names)

    # Architecture categories
    arch_type = {
        "GloVe": "count-based",
        "Word2Vec": "prediction",
        "FastText": "prediction",
        "BERT": "transformer",
        "SentenceT": "transformer"
    }

    # Correlation matrix
    corr_matrix = np.zeros((n_models, n_models))

    print(f"{'':15}", end="")
    for name in model_names:
        print(f"{name:>12}", end="")
    print()

    for i, name1 in enumerate(model_names):
        print(f"{name1:15}", end="")
        for j, name2 in enumerate(model_names):
            corr = spectrum_correlation(spectra[name1], spectra[name2])
            corr_matrix[i, j] = corr
            print(f"{corr:12.4f}", end="")
        print()

    print()

    # Analyze by architecture type
    print("-" * 70)
    print("Analysis by Architecture Type")
    print("-" * 70)
    print()

    # Within-type correlations
    within_type = {}
    cross_type = []

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                type1 = arch_type.get(name1, "unknown")
                type2 = arch_type.get(name2, "unknown")
                corr = corr_matrix[i, j]

                if type1 == type2:
                    key = type1
                    if key not in within_type:
                        within_type[key] = []
                    within_type[key].append((name1, name2, corr))
                else:
                    cross_type.append((name1, name2, corr))

    print("Within-architecture correlations:")
    for arch, pairs in within_type.items():
        for name1, name2, corr in pairs:
            print(f"  {name1} <-> {name2} ({arch}): {corr:.4f}")

    print()
    print("Cross-architecture correlations:")
    for name1, name2, corr in cross_type:
        type1 = arch_type.get(name1, "?")
        type2 = arch_type.get(name2, "?")
        print(f"  {name1} <-> {name2} ({type1} vs {type2}): {corr:.4f}")

    # Summary statistics
    all_corrs = [corr_matrix[i,j] for i in range(n_models) for j in range(i+1, n_models)]
    cross_corrs = [c for _, _, c in cross_type]

    mean_all = np.mean(all_corrs)
    mean_cross = np.mean(cross_corrs) if cross_corrs else 0

    print()
    print("-" * 70)
    print("Summary")
    print("-" * 70)
    print()
    print(f"Mean overall correlation: {mean_all:.4f}")
    print(f"Mean cross-architecture:  {mean_cross:.4f}")
    print()

    print("Participation Ratios by Architecture:")
    for name in model_names:
        arch = arch_type.get(name, "?")
        print(f"  {name:15} ({arch:12}): dim={dims[name]:3}, Df={dfs[name]:.2f}")

    print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if mean_cross > 0.9:
        print(f"[STRONG] Cross-architecture correlation = {mean_cross:.4f} (>0.9)")
        print("         Different architectures converge to SAME spectral structure!")
        print("         Platonic convergence is ARCHITECTURE-INDEPENDENT!")
        status = "STRONG"
    elif mean_cross > 0.7:
        print(f"[PARTIAL] Cross-architecture correlation = {mean_cross:.4f} (0.7-0.9)")
        print("          Moderate cross-architecture convergence.")
        status = "PARTIAL"
    elif mean_cross > 0.5:
        print(f"[WEAK] Cross-architecture correlation = {mean_cross:.4f} (0.5-0.7)")
        print("       Weak cross-architecture similarity.")
        status = "WEAK"
    else:
        print(f"[FAIL] Cross-architecture correlation = {mean_cross:.4f} (<0.5)")
        print("       Different architectures have DIFFERENT spectral structures.")
        print("       Platonic convergence may be architecture-dependent.")
        status = "DIVERGENT"

    print()

    # Compare to earlier results
    print("Comparison to earlier tests:")
    print("  Base language models (same arch):     0.852")
    print("  Sentence transformers (same obj):     0.989")
    print(f"  Cross-architecture (this test):       {mean_cross:.3f}")
    print()

    # Receipt
    receipt = {
        "test": "Q34_CROSS_ARCHITECTURE",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "dimensions": dims,
        "participation_ratios": {k: float(v) for k, v in dfs.items()},
        "mean_overall_correlation": float(mean_all),
        "mean_cross_architecture": float(mean_cross),
        "status": status,
    }

    receipt_json = json.dumps(receipt, indent=2)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")

    return receipt


if __name__ == '__main__':
    main()
