#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q34: Platonic Convergence - Df Attractor Characterization

Tests whether Df ~22 is a universal attractor or varies by:
- Architecture (count-based vs prediction vs transformer)
- Training objective (MLM vs semantic similarity vs skip-gram)
- Language (English vs Chinese)
- Model size (small vs large)

If Df converges to a consistent value across all these variations,
it's strong evidence for a universal semantic attractor.
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

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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


# Word list - use consistent set across all models
WORDS = [
    # Nature
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    # Animals
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "snake", "wolf", "bear", "eagle", "whale", "spider", "ant", "bee",
    # Body
    "heart", "eye", "hand", "head", "foot", "brain", "blood", "bone",
    # Family/People
    "mother", "father", "child", "brother", "sister", "friend", "king", "queen",
    # Abstract
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    # Objects
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "sword", "fire", "light", "shadow", "music", "word", "name", "law",
    # Actions (as nouns)
    "work", "sleep", "play", "fight", "dance", "song", "story", "game",
    # Qualities
    "good", "bad", "big", "small", "old", "new", "high", "low",
    "hot", "cold", "dark", "bright", "strong", "weak", "fast", "slow",
]


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute participation ratio Df = (sum(lambda))^2 / sum(lambda^2)."""
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    return (sum_lambda ** 2) / sum_lambda_sq


def compute_df_from_embeddings(embeddings: dict) -> dict:
    """Compute Df and related metrics from embeddings."""
    words = sorted(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Center
    vecs_centered = vecs - vecs.mean(axis=0)

    # Covariance eigenspectrum
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    df = participation_ratio(eigenvalues)

    # Top-k variance explained
    total_var = np.sum(eigenvalues)
    top10_var = np.sum(eigenvalues[:10]) / total_var
    top22_var = np.sum(eigenvalues[:22]) / total_var

    return {
        "df": df,
        "dim": vecs.shape[1],
        "n_words": len(words),
        "top10_variance": top10_var,
        "top22_variance": top22_var,
        "eigenvalue_ratio_1_22": eigenvalues[0] / eigenvalues[21] if len(eigenvalues) > 21 else None,
    }


def load_gensim_model(model_name: str, words: list) -> dict:
    """Load gensim model embeddings."""
    model = api.load(model_name)
    embeddings = {}
    for word in words:
        if word in model:
            embeddings[word] = model[word]
    return embeddings


def load_bert_embeddings(words: list, model_name: str) -> dict:
    """Load BERT embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings[word] = emb
    return embeddings


def load_sentence_transformer(words: list, model_name: str) -> dict:
    """Load sentence transformer embeddings."""
    model = SentenceTransformer(model_name)
    embs = model.encode(words, normalize_embeddings=True)
    return {word: embs[i] for i, word in enumerate(words)}


def main():
    print("=" * 70)
    print("Q34: PLATONIC CONVERGENCE - Df ATTRACTOR CHARACTERIZATION")
    print("Testing if Df ~22 is a universal attractor")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Word list: {len(WORDS)} words")
    print()

    # Check dependencies
    print("Dependencies:")
    print(f"  gensim: {'YES' if GENSIM_AVAILABLE else 'NO'}")
    print(f"  transformers: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    print(f"  sentence-transformers: {'YES' if ST_AVAILABLE else 'NO'}")
    print()

    results = []

    # === GENSIM MODELS (Count-based and Prediction) ===
    if GENSIM_AVAILABLE:
        gensim_models = [
            ("glove-wiki-gigaword-300", "GloVe", "count-based", "co-occurrence"),
            ("word2vec-google-news-300", "Word2Vec", "prediction", "skip-gram"),
            ("fasttext-wiki-news-subwords-300", "FastText", "prediction", "subword"),
        ]

        for model_id, name, arch, objective in gensim_models:
            print(f"Loading {name}...")
            try:
                emb = load_gensim_model(model_id, WORDS)
                if len(emb) >= 50:
                    metrics = compute_df_from_embeddings(emb)
                    metrics.update({
                        "model": name,
                        "model_id": model_id,
                        "architecture": arch,
                        "objective": objective,
                        "language": "en",
                    })
                    results.append(metrics)
                    print(f"  {name}: Df={metrics['df']:.2f}, dim={metrics['dim']}, words={metrics['n_words']}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    # === TRANSFORMER MODELS ===
    if TRANSFORMERS_AVAILABLE:
        bert_models = [
            ("bert-base-uncased", "BERT-base", "transformer", "MLM", "en"),
            ("bert-base-chinese", "BERT-Chinese", "transformer", "MLM", "zh"),
            ("bert-base-multilingual-cased", "mBERT", "transformer", "MLM", "multi"),
        ]

        for model_id, name, arch, objective, lang in bert_models:
            print(f"Loading {name}...")
            try:
                emb = load_bert_embeddings(WORDS, model_id)
                metrics = compute_df_from_embeddings(emb)
                metrics.update({
                    "model": name,
                    "model_id": model_id,
                    "architecture": arch,
                    "objective": objective,
                    "language": lang,
                })
                results.append(metrics)
                print(f"  {name}: Df={metrics['df']:.2f}, dim={metrics['dim']}, words={metrics['n_words']}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    # === SENTENCE TRANSFORMERS ===
    if ST_AVAILABLE:
        st_models = [
            ("all-MiniLM-L6-v2", "MiniLM", "transformer", "similarity", "en", 384),
            ("all-mpnet-base-v2", "MPNet", "transformer", "similarity", "en", 768),
            ("paraphrase-multilingual-MiniLM-L12-v2", "mMiniLM", "transformer", "similarity", "multi", 384),
        ]

        for model_id, name, arch, objective, lang, dim in st_models:
            print(f"Loading {name}...")
            try:
                emb = load_sentence_transformer(WORDS, model_id)
                metrics = compute_df_from_embeddings(emb)
                metrics.update({
                    "model": name,
                    "model_id": model_id,
                    "architecture": arch,
                    "objective": objective,
                    "language": lang,
                })
                results.append(metrics)
                print(f"  {name}: Df={metrics['df']:.2f}, dim={metrics['dim']}, words={metrics['n_words']}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    print()

    if len(results) < 2:
        print("Not enough models loaded for analysis")
        return 1

    # === ANALYSIS ===
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()

    # Sort by Df
    results_sorted = sorted(results, key=lambda x: x['df'])

    print(f"{'Model':<20} {'Arch':<12} {'Objective':<12} {'Dim':>5} {'Df':>8} {'Top22%':>8}")
    print("-" * 70)
    for r in results_sorted:
        top22 = r.get('top22_variance', 0) * 100
        print(f"{r['model']:<20} {r['architecture']:<12} {r['objective']:<12} {r['dim']:>5} {r['df']:>8.2f} {top22:>7.1f}%")

    print()

    # Statistics
    dfs = [r['df'] for r in results]
    df_mean = np.mean(dfs)
    df_std = np.std(dfs)
    df_min = np.min(dfs)
    df_max = np.max(dfs)

    print("-" * 70)
    print("Df STATISTICS")
    print("-" * 70)
    print()
    print(f"Mean Df:   {df_mean:.2f}")
    print(f"Std Df:    {df_std:.2f}")
    print(f"Range:     {df_min:.2f} - {df_max:.2f}")
    print(f"CV:        {df_std/df_mean:.2%}")
    print()

    # Group by architecture
    print("-" * 70)
    print("Df BY ARCHITECTURE")
    print("-" * 70)
    print()

    arch_groups = {}
    for r in results:
        arch = r['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(r['df'])

    for arch, dfs_arch in arch_groups.items():
        print(f"  {arch:<15}: mean={np.mean(dfs_arch):.2f}, std={np.std(dfs_arch):.2f}, n={len(dfs_arch)}")

    print()

    # Group by objective
    print("-" * 70)
    print("Df BY TRAINING OBJECTIVE")
    print("-" * 70)
    print()

    obj_groups = {}
    for r in results:
        obj = r['objective']
        if obj not in obj_groups:
            obj_groups[obj] = []
        obj_groups[obj].append(r['df'])

    for obj, dfs_obj in obj_groups.items():
        print(f"  {obj:<15}: mean={np.mean(dfs_obj):.2f}, std={np.std(dfs_obj):.2f}, n={len(dfs_obj)}")

    print()

    # === VERDICT ===
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    # Check if Df is consistent (CV < 50%)
    cv = df_std / df_mean

    if cv < 0.3:
        print(f"[STRONG] Df is CONSISTENT across models (CV={cv:.1%})")
        print(f"         Mean Df = {df_mean:.1f} appears to be a universal attractor")
        status = "UNIVERSAL_ATTRACTOR"
    elif cv < 0.5:
        print(f"[PARTIAL] Df shows MODERATE consistency (CV={cv:.1%})")
        print(f"          Mean Df = {df_mean:.1f}, but with notable variation")
        status = "PARTIAL_ATTRACTOR"
    else:
        print(f"[WEAK] Df varies SIGNIFICANTLY across models (CV={cv:.1%})")
        print(f"       No universal attractor detected")
        status = "NO_ATTRACTOR"

    print()

    # Check if objective matters more than architecture
    arch_variance = np.var([np.mean(v) for v in arch_groups.values()])
    obj_variance = np.var([np.mean(v) for v in obj_groups.values()])

    if obj_variance > arch_variance * 2:
        print("Training OBJECTIVE explains more Df variance than architecture")
    elif arch_variance > obj_variance * 2:
        print("ARCHITECTURE explains more Df variance than training objective")
    else:
        print("Architecture and objective contribute similarly to Df variance")

    print()

    # Receipt
    receipt = {
        "test": "Q34_DF_ATTRACTOR",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_models": len(results),
        "n_words": len(WORDS),
        "df_mean": float(df_mean),
        "df_std": float(df_std),
        "df_min": float(df_min),
        "df_max": float(df_max),
        "cv": float(cv),
        "status": status,
        "results": results,
    }

    receipt_json = json.dumps(receipt, indent=2)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")

    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    receipt_path = results_dir / "q34_df_attractor.json"
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"Results saved to: {receipt_path}")

    return receipt


if __name__ == '__main__':
    main()
