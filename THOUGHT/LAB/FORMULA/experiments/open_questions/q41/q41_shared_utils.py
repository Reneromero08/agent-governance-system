#!/usr/bin/env python3
"""
Q41 Phase 2: Shared Utilities

Common utilities for TIER 2 and TIER 5 Langlands tests.
Import this module in individual test files.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import hashlib
import math
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for Langlands tests."""
    seed: int = 42
    k_neighbors: int = 10
    preprocessing: str = "l2"
    distance_metric: str = "euclidean"

@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    test_type: str
    passed: bool
    metrics: Dict[str, Any]
    thresholds: Dict[str, float]
    controls: Dict[str, Any]
    notes: str
    skipped: bool = False
    skip_reason: Optional[str] = None

# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def to_builtin(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj
    return obj

# =============================================================================
# EMBEDDING PREPROCESSING
# =============================================================================

def preprocess_embeddings(X: np.ndarray, method: str = "l2") -> np.ndarray:
    """Preprocess embeddings."""
    if method == "raw":
        return X.copy()
    elif method == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return X / norms
    elif method == "centered":
        return X - X.mean(axis=0)
    else:
        return X.copy()

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    """Build symmetric k-NN adjacency matrix."""
    n = X.shape[0]
    dists = squareform(pdist(X, 'euclidean'))

    A = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:k+1]
        A[i, neighbors] = 1

    A = np.maximum(A, A.T)
    return A


def build_graph_laplacian(A: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Build graph Laplacian from adjacency matrix."""
    D = np.diag(A.sum(axis=1))
    L = D - A

    if normalized:
        d_inv_sqrt = 1.0 / np.sqrt(np.diag(D) + 1e-10)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt

    return L

# =============================================================================
# CORPUS
# =============================================================================

DEFAULT_CORPUS = [
    "king", "queen", "man", "woman", "prince", "princess",
    "father", "mother", "son", "daughter", "brother", "sister",
    "cat", "dog", "bird", "fish", "tree", "flower", "sky", "earth",
    "happy", "sad", "angry", "calm", "love", "hate", "fear", "hope",
    "run", "walk", "jump", "fly", "think", "feel", "see", "hear",
    "science", "art", "music", "math", "history", "future", "past", "present",
    "hot", "cold", "light", "dark", "big", "small", "fast", "slow",
    "good", "bad", "right", "wrong", "true", "false", "real", "fake",
    "water", "fire", "air", "stone", "metal", "wood", "glass", "paper"
]


def compute_corpus_hash(corpus: List[str]) -> str:
    """Compute deterministic hash of corpus."""
    return hashlib.sha256("|".join(corpus).encode()).hexdigest()

# =============================================================================
# EMBEDDING LOADERS
# =============================================================================

def load_embeddings(corpus: List[str], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Load embeddings from available models."""
    embeddings = {}
    import warnings
    warnings.filterwarnings('ignore')

    try:
        from sentence_transformers import SentenceTransformer

        st_models = [
            ('all-MiniLM-L6-v2', 'MiniLM'),
            ('all-mpnet-base-v2', 'MPNet'),
            ('paraphrase-MiniLM-L6-v2', 'Paraphrase'),
        ]

        for model_id, short_name in st_models:
            try:
                if verbose:
                    print(f"  Loading {short_name}...")
                model = SentenceTransformer(model_id)
                embs = model.encode(corpus, normalize_embeddings=False, show_progress_bar=False)
                embeddings[short_name] = embs
                if verbose:
                    print(f"    {short_name}: n={len(embs)}, d={embs.shape[1]}")
            except Exception as e:
                if verbose:
                    print(f"    {short_name}: FAILED ({e})")
    except ImportError:
        if verbose:
            print("  sentence-transformers not available")

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        if verbose:
            print("  Loading BERT...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.eval()

        bert_embs = []
        with torch.no_grad():
            for word in corpus:
                inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[0, 0, :].numpy()
                bert_embs.append(emb)

        embeddings['BERT'] = np.array(bert_embs)
        if verbose:
            print(f"    BERT: n={len(bert_embs)}, d={bert_embs[0].shape[0]}")
    except ImportError:
        if verbose:
            print("  transformers not available")
    except Exception as e:
        if verbose:
            print(f"  BERT: FAILED ({e})")

    return embeddings
