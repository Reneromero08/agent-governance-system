#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q34: Platonic Convergence - Cross-Lingual Test

Tests whether models trained on DIFFERENT LANGUAGES converge to the
same spectral structure:
- English BERT (bert-base-uncased)
- Chinese BERT (bert-base-chinese)
- Multilingual BERT (bert-base-multilingual-cased)

If different languages converge, this is the STRONGEST evidence for
Platonic convergence - the underlying semantic structure is UNIVERSAL,
not language-specific.

Approach:
1. Use parallel word lists (English words + Chinese translations)
2. Embed each language's words with its native model
3. Compare eigenvalue spectra of the distance matrices
4. If spectra correlate, semantics is language-independent
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

# Fix Windows console encoding for Chinese characters
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check available libraries
TRANSFORMERS_AVAILABLE = False
ST_AVAILABLE = False

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


# Bilingual anchor words: (English, Chinese)
# Selected for concrete concepts that translate unambiguously
BILINGUAL_ANCHORS = [
    # Nature
    ("water", "水"),
    ("fire", "火"),
    ("earth", "土"),
    ("sky", "天"),
    ("sun", "太阳"),
    ("moon", "月亮"),
    ("star", "星"),
    ("mountain", "山"),
    ("river", "河"),
    ("tree", "树"),
    ("flower", "花"),
    ("rain", "雨"),
    ("wind", "风"),
    ("snow", "雪"),
    ("cloud", "云"),
    # Animals
    ("dog", "狗"),
    ("cat", "猫"),
    ("bird", "鸟"),
    ("fish", "鱼"),
    ("horse", "马"),
    ("tiger", "虎"),
    ("dragon", "龙"),
    ("snake", "蛇"),
    ("elephant", "大象"),
    ("lion", "狮子"),
    # Body
    ("heart", "心"),
    ("eye", "眼"),
    ("hand", "手"),
    ("head", "头"),
    ("foot", "脚"),
    # Family
    ("mother", "母亲"),
    ("father", "父亲"),
    ("child", "孩子"),
    ("brother", "兄弟"),
    ("sister", "姐妹"),
    # Abstract
    ("love", "爱"),
    ("hate", "恨"),
    ("truth", "真"),
    ("life", "生命"),
    ("death", "死"),
    ("time", "时间"),
    ("space", "空间"),
    ("power", "力量"),
    ("peace", "和平"),
    ("war", "战争"),
    # Objects
    ("book", "书"),
    ("door", "门"),
    ("house", "房子"),
    ("road", "路"),
    ("food", "食物"),
    ("water", "水"),
    ("money", "钱"),
    ("king", "王"),
    ("god", "神"),
    ("man", "人"),
    ("woman", "女人"),
    # Actions (as nouns)
    ("work", "工作"),
    ("sleep", "睡眠"),
    ("dream", "梦"),
    ("thought", "思想"),
    ("word", "词"),
    # Qualities
    ("good", "好"),
    ("bad", "坏"),
    ("big", "大"),
    ("small", "小"),
    ("old", "老"),
    ("new", "新"),
    ("high", "高"),
    ("low", "低"),
]

# Remove duplicates
seen_en = set()
BILINGUAL_ANCHORS_UNIQUE = []
for en, zh in BILINGUAL_ANCHORS:
    if en not in seen_en:
        seen_en.add(en)
        BILINGUAL_ANCHORS_UNIQUE.append((en, zh))

BILINGUAL_ANCHORS = BILINGUAL_ANCHORS_UNIQUE


def compute_distance_matrix(embeddings: dict) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    words = sorted(embeddings.keys())
    n = len(words)
    vecs = np.array([embeddings[w] for w in words])

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / (norms + 1e-10)

    # Cosine similarity -> distance
    sim = vecs_norm @ vecs_norm.T
    dist = 1 - sim

    return dist, words


def compute_eigenspectrum(dist_matrix: np.ndarray) -> np.ndarray:
    """Compute eigenspectrum of distance matrix (MDS-style)."""
    n = dist_matrix.shape[0]

    # Double-center (Gram matrix from distances)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist_matrix ** 2) @ H

    # Eigendecompose
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    return eigenvalues


def compute_covariance_spectrum(embeddings: dict) -> np.ndarray:
    """Compute covariance eigenspectrum."""
    words = sorted(embeddings.keys())
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


def load_bert_embeddings(words: list, model_name: str, lang: str) -> tuple:
    """Load BERT embeddings for given words."""
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            emb = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings[word] = emb

    dim = next(iter(embeddings.values())).shape[0]
    print(f"    Loaded {len(embeddings)} {lang} words, dim={dim}")
    return embeddings, dim


def load_sentence_transformer(words: list, model_name: str, lang: str) -> tuple:
    """Load sentence transformer embeddings."""
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)

    embs = model.encode(words, normalize_embeddings=True)
    embeddings = {word: embs[i] for i, word in enumerate(words)}

    print(f"    Loaded {len(embeddings)} {lang} words, dim={embs.shape[1]}")
    return embeddings, embs.shape[1]


def main():
    print("=" * 70)
    print("Q34: PLATONIC CONVERGENCE - CROSS-LINGUAL TEST")
    print("Testing: English BERT vs Chinese BERT vs Multilingual BERT")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    # Check dependencies
    print("Dependencies:")
    print(f"  transformers: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    print(f"  sentence-transformers: {'YES' if ST_AVAILABLE else 'NO'}")
    print()

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers required")
        print("Install with: pip install transformers torch")
        return 1

    # Extract word lists
    english_words = [en for en, zh in BILINGUAL_ANCHORS]
    chinese_words = [zh for en, zh in BILINGUAL_ANCHORS]

    print(f"Bilingual anchors: {len(BILINGUAL_ANCHORS)} word pairs")
    print(f"Examples: {BILINGUAL_ANCHORS[:5]}")
    print()

    # Load models
    print("-" * 70)
    print("Loading models...")
    print("-" * 70)

    models = {}
    spectra = {}
    dfs = {}
    dims = {}

    # English BERT
    try:
        emb, dim = load_bert_embeddings(english_words, "bert-base-uncased", "English")
        dist, _ = compute_distance_matrix(emb)
        spec = compute_eigenspectrum(dist)
        df = participation_ratio(spec)
        models["English-BERT"] = emb
        spectra["English-BERT"] = spec
        dfs["English-BERT"] = df
        dims["English-BERT"] = dim
        print(f"    English-BERT: Df={df:.2f}")
    except Exception as e:
        print(f"    English-BERT: FAILED ({e})")

    # Chinese BERT
    try:
        emb, dim = load_bert_embeddings(chinese_words, "bert-base-chinese", "Chinese")
        dist, _ = compute_distance_matrix(emb)
        spec = compute_eigenspectrum(dist)
        df = participation_ratio(spec)
        models["Chinese-BERT"] = emb
        spectra["Chinese-BERT"] = spec
        dfs["Chinese-BERT"] = df
        dims["Chinese-BERT"] = dim
        print(f"    Chinese-BERT: Df={df:.2f}")
    except Exception as e:
        print(f"    Chinese-BERT: FAILED ({e})")

    # Multilingual BERT - English words
    try:
        emb, dim = load_bert_embeddings(english_words, "bert-base-multilingual-cased", "English")
        dist, _ = compute_distance_matrix(emb)
        spec = compute_eigenspectrum(dist)
        df = participation_ratio(spec)
        models["mBERT-EN"] = emb
        spectra["mBERT-EN"] = spec
        dfs["mBERT-EN"] = df
        dims["mBERT-EN"] = dim
        print(f"    mBERT-EN: Df={df:.2f}")
    except Exception as e:
        print(f"    mBERT-EN: FAILED ({e})")

    # Multilingual BERT - Chinese words
    try:
        emb, dim = load_bert_embeddings(chinese_words, "bert-base-multilingual-cased", "Chinese")
        dist, _ = compute_distance_matrix(emb)
        spec = compute_eigenspectrum(dist)
        df = participation_ratio(spec)
        models["mBERT-ZH"] = emb
        spectra["mBERT-ZH"] = spec
        dfs["mBERT-ZH"] = df
        dims["mBERT-ZH"] = dim
        print(f"    mBERT-ZH: Df={df:.2f}")
    except Exception as e:
        print(f"    mBERT-ZH: FAILED ({e})")

    # Sentence transformers (multilingual)
    if ST_AVAILABLE:
        # Multilingual sentence transformer - English
        try:
            emb, dim = load_sentence_transformer(
                english_words,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "English"
            )
            dist, _ = compute_distance_matrix(emb)
            spec = compute_eigenspectrum(dist)
            df = participation_ratio(spec)
            models["mST-EN"] = emb
            spectra["mST-EN"] = spec
            dfs["mST-EN"] = df
            dims["mST-EN"] = dim
            print(f"    mST-EN: Df={df:.2f}")
        except Exception as e:
            print(f"    mST-EN: FAILED ({e})")

        # Multilingual sentence transformer - Chinese
        try:
            emb, dim = load_sentence_transformer(
                chinese_words,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Chinese"
            )
            dist, _ = compute_distance_matrix(emb)
            spec = compute_eigenspectrum(dist)
            df = participation_ratio(spec)
            models["mST-ZH"] = emb
            spectra["mST-ZH"] = spec
            dfs["mST-ZH"] = df
            dims["mST-ZH"] = dim
            print(f"    mST-ZH: Df={df:.2f}")
        except Exception as e:
            print(f"    mST-ZH: FAILED ({e})")

    print()

    if len(spectra) < 2:
        print("Not enough models loaded for comparison")
        return 1

    # Compute cross-model correlations
    print("-" * 70)
    print("Cross-Lingual Eigenvalue Correlations (Distance Matrix Spectrum)")
    print("-" * 70)
    print()

    model_names = list(spectra.keys())
    n_models = len(model_names)

    # Language/model categories
    lang_type = {
        "English-BERT": "en-mono",
        "Chinese-BERT": "zh-mono",
        "mBERT-EN": "multi-en",
        "mBERT-ZH": "multi-zh",
        "mST-EN": "multi-en",
        "mST-ZH": "multi-zh",
    }

    # Correlation matrix
    corr_matrix = np.zeros((n_models, n_models))

    print(f"{'':15}", end="")
    for name in model_names:
        short_name = name[:10]
        print(f"{short_name:>12}", end="")
    print()

    for i, name1 in enumerate(model_names):
        print(f"{name1:15}", end="")
        for j, name2 in enumerate(model_names):
            corr = spectrum_correlation(spectra[name1], spectra[name2])
            corr_matrix[i, j] = corr
            print(f"{corr:12.4f}", end="")
        print()

    print()

    # Analyze key comparisons
    print("-" * 70)
    print("Key Cross-Lingual Comparisons")
    print("-" * 70)
    print()

    key_pairs = [
        ("English-BERT", "Chinese-BERT", "Monolingual EN vs Monolingual ZH"),
        ("mBERT-EN", "mBERT-ZH", "Same model, different language input"),
        ("English-BERT", "mBERT-EN", "Monolingual vs Multilingual (same lang)"),
        ("Chinese-BERT", "mBERT-ZH", "Monolingual vs Multilingual (same lang)"),
    ]

    if ST_AVAILABLE:
        key_pairs.extend([
            ("mST-EN", "mST-ZH", "Sentence-T: EN vs ZH"),
            ("English-BERT", "mST-EN", "BERT vs Sentence-T (EN)"),
        ])

    cross_lingual_corrs = []

    for name1, name2, desc in key_pairs:
        if name1 in spectra and name2 in spectra:
            corr = spectrum_correlation(spectra[name1], spectra[name2])
            print(f"  {desc}:")
            print(f"    {name1} <-> {name2}: {corr:.4f}")

            # Track cross-lingual specifically
            if "EN" in name1 and "ZH" in name2:
                cross_lingual_corrs.append(corr)
            elif "ZH" in name1 and "EN" in name2:
                cross_lingual_corrs.append(corr)
            elif name1 == "English-BERT" and name2 == "Chinese-BERT":
                cross_lingual_corrs.append(corr)

    print()

    # Summary statistics
    all_corrs = [corr_matrix[i,j] for i in range(n_models) for j in range(i+1, n_models)]

    mean_all = np.mean(all_corrs)
    mean_cross_lingual = np.mean(cross_lingual_corrs) if cross_lingual_corrs else 0

    print("-" * 70)
    print("Summary")
    print("-" * 70)
    print()
    print(f"Mean overall correlation:     {mean_all:.4f}")
    print(f"Mean cross-lingual (EN<->ZH): {mean_cross_lingual:.4f}")
    print()

    print("Participation Ratios:")
    for name in model_names:
        print(f"  {name:15}: dim={dims[name]:3}, Df={dfs[name]:.2f}")

    print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if mean_cross_lingual > 0.9:
        print(f"[STRONG] Cross-lingual correlation = {mean_cross_lingual:.4f} (>0.9)")
        print("         English and Chinese converge to SAME spectral structure!")
        print("         Semantic geometry is LANGUAGE-INDEPENDENT!")
        print("         This is the STRONGEST evidence for Platonic convergence!")
        status = "STRONG"
    elif mean_cross_lingual > 0.7:
        print(f"[PARTIAL] Cross-lingual correlation = {mean_cross_lingual:.4f} (0.7-0.9)")
        print("          Moderate cross-lingual convergence.")
        print("          Some shared structure across languages.")
        status = "PARTIAL"
    elif mean_cross_lingual > 0.5:
        print(f"[WEAK] Cross-lingual correlation = {mean_cross_lingual:.4f} (0.5-0.7)")
        print("       Weak cross-lingual similarity.")
        print("       Languages may have different semantic organizations.")
        status = "WEAK"
    else:
        print(f"[FAIL] Cross-lingual correlation = {mean_cross_lingual:.4f} (<0.5)")
        print("       English and Chinese have DIFFERENT spectral structures.")
        print("       Platonic convergence may be language-specific.")
        status = "DIVERGENT"

    print()

    # Compare to earlier results
    print("Comparison to earlier tests:")
    print("  Same architecture, same lang:     0.852")
    print("  Same training objective:          0.989")
    print("  Cross-architecture (same lang):   0.971")
    print(f"  Cross-lingual (this test):        {mean_cross_lingual:.3f}")
    print()

    # Receipt
    receipt = {
        "test": "Q34_CROSS_LINGUAL",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "dimensions": dims,
        "participation_ratios": {k: float(v) for k, v in dfs.items()},
        "bilingual_anchors": len(BILINGUAL_ANCHORS),
        "mean_overall_correlation": float(mean_all),
        "mean_cross_lingual": float(mean_cross_lingual),
        "status": status,
        "correlation_matrix": corr_matrix.tolist(),
    }

    receipt_json = json.dumps(receipt, indent=2)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"Receipt hash: {receipt_hash[:16]}...")

    # Save receipt
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    receipt_path = results_dir / "q34_cross_lingual.json"
    with open(receipt_path, 'w', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {receipt_path}")

    return receipt


if __name__ == '__main__':
    main()
