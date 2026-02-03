#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q34: Platonic Convergence - Invariant Identification

What EXACTLY is preserved when spectral correlation = 0.97 but Df varies?

Candidates:
1. Eigenvalue RATIOS (relative shape)
2. Eigenvalue RANKS (ordering)
3. Cumulative variance curve (how fast eigenvalues decay)
4. Spectral entropy (distribution flatness)

Goal: Identify the invariant that makes Platonic convergence work.
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, entropy

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check libraries
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


WORDS = [
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "snake", "wolf", "bear", "eagle", "whale", "spider", "ant", "bee",
    "heart", "eye", "hand", "head", "foot", "brain", "blood", "bone",
    "mother", "father", "child", "brother", "sister", "friend", "king", "queen",
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "sword", "light", "shadow", "music", "word", "name", "law",
    "work", "sleep", "play", "fight", "dance", "song", "story", "game",
    "good", "bad", "big", "small", "old", "new", "high", "low",
    "hot", "cold", "dark", "bright", "strong", "weak", "fast", "slow",
]


def get_eigenspectrum(embeddings: dict) -> np.ndarray:
    """Get normalized eigenspectrum from embeddings."""
    words = sorted(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvalues


def compute_invariants(eigenvalues: np.ndarray, k: int = 50) -> dict:
    """Compute candidate invariants from eigenspectrum."""
    ev = eigenvalues[:k]
    total = np.sum(ev)

    # Normalized spectrum (sums to 1)
    normalized = ev / total

    # Cumulative variance
    cumulative = np.cumsum(normalized)

    # Log ratios (consecutive)
    log_ratios = np.log(ev[:-1] / ev[1:])

    # Spectral entropy
    spec_entropy = entropy(normalized)

    # Decay rate (linear fit to log eigenvalues)
    x = np.arange(len(ev))
    log_ev = np.log(ev)
    decay_rate = np.polyfit(x, log_ev, 1)[0]

    # Participation ratio
    df = (np.sum(ev) ** 2) / np.sum(ev ** 2)

    return {
        "normalized": normalized,
        "cumulative": cumulative,
        "log_ratios": log_ratios,
        "spectral_entropy": spec_entropy,
        "decay_rate": decay_rate,
        "df": df,
    }


def load_gensim_model(model_name: str, words: list) -> dict:
    model = api.load(model_name)
    return {w: model[w] for w in words if w in model}


def load_bert_embeddings(words: list, model_name: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            embeddings[word] = outputs.last_hidden_state[0, 0, :].numpy()
    return embeddings


def load_sentence_transformer(words: list, model_name: str) -> dict:
    model = SentenceTransformer(model_name)
    embs = model.encode(words, normalize_embeddings=True)
    return {word: embs[i] for i, word in enumerate(words)}


def main():
    print("=" * 70)
    print("Q34: INVARIANT IDENTIFICATION")
    print("What is preserved when spectral correlation is high but Df varies?")
    print("=" * 70)
    print()

    # Load models
    models = {}
    spectra = {}
    invariants = {}

    print("Loading models...")

    if GENSIM_AVAILABLE:
        for model_id, name in [
            ("glove-wiki-gigaword-300", "GloVe"),
            ("word2vec-google-news-300", "Word2Vec"),
        ]:
            try:
                emb = load_gensim_model(model_id, WORDS)
                if len(emb) >= 50:
                    spectra[name] = get_eigenspectrum(emb)
                    invariants[name] = compute_invariants(spectra[name])
                    print(f"  {name}: Df={invariants[name]['df']:.1f}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    if TRANSFORMERS_AVAILABLE:
        for model_id, name in [
            ("bert-base-uncased", "BERT"),
            ("bert-base-multilingual-cased", "mBERT"),
        ]:
            try:
                emb = load_bert_embeddings(WORDS, model_id)
                spectra[name] = get_eigenspectrum(emb)
                invariants[name] = compute_invariants(spectra[name])
                print(f"  {name}: Df={invariants[name]['df']:.1f}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    if ST_AVAILABLE:
        for model_id, name in [
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet"),
        ]:
            try:
                emb = load_sentence_transformer(WORDS, model_id)
                spectra[name] = get_eigenspectrum(emb)
                invariants[name] = compute_invariants(spectra[name])
                print(f"  {name}: Df={invariants[name]['df']:.1f}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    print()

    if len(spectra) < 2:
        print("Not enough models")
        return

    model_names = list(spectra.keys())
    n = len(model_names)

    # Test each candidate invariant
    print("-" * 70)
    print("TESTING CANDIDATE INVARIANTS")
    print("-" * 70)
    print()

    # 1. Raw normalized spectrum correlation
    print("1. NORMALIZED SPECTRUM (eigenvalues / sum)")
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            n1, n2 = invariants[model_names[i]]['normalized'], invariants[model_names[j]]['normalized']
            k = min(len(n1), len(n2))
            r = np.corrcoef(n1[:k], n2[:k])[0,1]
            corrs.append(r)
            print(f"   {model_names[i]} vs {model_names[j]}: {r:.4f}")
    print(f"   Mean: {np.mean(corrs):.4f}")
    norm_mean = np.mean(corrs)
    print()

    # 2. Cumulative variance curve
    print("2. CUMULATIVE VARIANCE CURVE")
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            c1, c2 = invariants[model_names[i]]['cumulative'], invariants[model_names[j]]['cumulative']
            k = min(len(c1), len(c2))
            r = np.corrcoef(c1[:k], c2[:k])[0,1]
            corrs.append(r)
            print(f"   {model_names[i]} vs {model_names[j]}: {r:.4f}")
    print(f"   Mean: {np.mean(corrs):.4f}")
    cumul_mean = np.mean(corrs)
    print()

    # 3. Log ratios (consecutive eigenvalue ratios)
    print("3. LOG RATIOS (ln(λ_i / λ_{i+1}))")
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            r1, r2 = invariants[model_names[i]]['log_ratios'], invariants[model_names[j]]['log_ratios']
            k = min(len(r1), len(r2))
            r = np.corrcoef(r1[:k], r2[:k])[0,1]
            corrs.append(r)
            print(f"   {model_names[i]} vs {model_names[j]}: {r:.4f}")
    print(f"   Mean: {np.mean(corrs):.4f}")
    ratio_mean = np.mean(corrs)
    print()

    # 4. Spectral entropy comparison
    print("4. SPECTRAL ENTROPY")
    entropies = [(name, invariants[name]['spectral_entropy']) for name in model_names]
    for name, ent in entropies:
        print(f"   {name}: {ent:.4f}")
    ent_values = [e for _, e in entropies]
    ent_cv = np.std(ent_values) / np.mean(ent_values)
    print(f"   CV: {ent_cv:.2%}")
    print()

    # 5. Decay rate comparison
    print("5. DECAY RATE (slope of log spectrum)")
    rates = [(name, invariants[name]['decay_rate']) for name in model_names]
    for name, rate in rates:
        print(f"   {name}: {rate:.4f}")
    rate_values = [r for _, r in rates]
    rate_cv = np.std(rate_values) / np.mean(rate_values)
    print(f"   CV: {rate_cv:.2%}")
    print()

    # Summary
    print("=" * 70)
    print("INVARIANT RANKING (higher = more invariant)")
    print("=" * 70)
    print()

    invariant_scores = [
        ("Normalized Spectrum", norm_mean),
        ("Cumulative Variance", cumul_mean),
        ("Log Ratios", ratio_mean),
        ("Spectral Entropy", 1 - ent_cv),  # Lower CV = more invariant
        ("Decay Rate", 1 - rate_cv),
    ]

    for name, score in sorted(invariant_scores, key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"  {name:<25}: {score:.3f} {bar}")

    print()

    # Determine the invariant
    best = max(invariant_scores, key=lambda x: x[1])
    print(f"BEST INVARIANT: {best[0]} (score={best[1]:.3f})")
    print()

    if best[0] == "Normalized Spectrum":
        print("The SHAPE of the eigenvalue distribution is the invariant.")
        print("Different Df = different 'zoom level' on the same shape.")
    elif best[0] == "Cumulative Variance":
        print("The VARIANCE ACCUMULATION RATE is the invariant.")
        print("All models accumulate variance in the same pattern.")
    elif best[0] == "Log Ratios":
        print("The EIGENVALUE RATIOS are the invariant.")
        print("λ₁/λ₂, λ₂/λ₃, etc. are preserved across models.")

    # Save receipt
    receipt = {
        "test": "Q34_INVARIANT",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "invariant_scores": dict(invariant_scores),
        "best_invariant": best[0],
        "best_score": best[1],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "q34_invariant.json", 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt saved to results/q34_invariant.json")


if __name__ == '__main__':
    main()
