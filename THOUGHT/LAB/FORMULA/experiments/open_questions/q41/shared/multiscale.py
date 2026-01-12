#!/usr/bin/env python3
"""
Q41: Multi-Scale Embedding Infrastructure

Provides hierarchical corpus and embedding capabilities for:
- TIER 3: Functoriality Tower (multi-scale lifting)
- TIER 3: Base Change (cross-lingual)

Scales:
- Word: Individual words (64 items)
- Sentence: Sentences containing words (20 items)
- Paragraph: Paragraphs containing sentences (5 items)
- Document: Documents containing paragraphs (2 items)

The key insight: Langlands functoriality requires structure-preserving
maps between representation spaces at different scales.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# MULTI-SCALE CORPUS
# =============================================================================

# Word-level (same as DEFAULT_CORPUS)
WORDS = [
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

# Sentence-level (20 sentences, each uses ~3-4 words from WORDS)
SENTENCES = [
    "The king and queen ruled the kingdom with love and hope.",
    "A man and woman walked through the dark forest together.",
    "The prince loved his sister and feared his brother.",
    "Father and mother watched their son run fast.",
    "The daughter felt happy when she saw the light.",
    "A big cat jumped over the small dog in the past.",
    "The bird flew through the hot air above the earth.",
    "Science and math help us think about true things.",
    "Art and music make people feel calm and happy.",
    "The future holds hope while the present feels real.",
    "Water and fire are good for different purposes.",
    "Cold stone sits next to hot metal in history.",
    "The princess could see the sky from her tower.",
    "A slow fish swims through dark water at night.",
    "Wood and paper burn fast in the fire.",
    "Bad things feel wrong while good things feel right.",
    "The tree and flower grow under the light sky.",
    "Fear and hate are false emotions that feel real.",
    "A fast bird can fly and hear better than a slow cat.",
    "Glass and metal shine bright in the present light."
]

# Paragraph-level (5 paragraphs, each contains ~4 sentences)
PARAGRAPHS = [
    """The king and queen ruled the kingdom with love and hope. A man and woman walked through the dark forest together. The prince loved his sister and feared his brother. Father and mother watched their son run fast.""",

    """The daughter felt happy when she saw the light. A big cat jumped over the small dog in the past. The bird flew through the hot air above the earth. Science and math help us think about true things.""",

    """Art and music make people feel calm and happy. The future holds hope while the present feels real. Water and fire are good for different purposes. Cold stone sits next to hot metal in history.""",

    """The princess could see the sky from her tower. A slow fish swims through dark water at night. Wood and paper burn fast in the fire. Bad things feel wrong while good things feel right.""",

    """The tree and flower grow under the light sky. Fear and hate are false emotions that feel real. A fast bird can fly and hear better than a slow cat. Glass and metal shine bright in the present light."""
]

# Document-level (2 documents, each contains paragraphs)
DOCUMENTS = [
    """The king and queen ruled the kingdom with love and hope. A man and woman walked through the dark forest together. The prince loved his sister and feared his brother. Father and mother watched their son run fast.

The daughter felt happy when she saw the light. A big cat jumped over the small dog in the past. The bird flew through the hot air above the earth. Science and math help us think about true things.

Art and music make people feel calm and happy. The future holds hope while the present feels real. Water and fire are good for different purposes.""",

    """Cold stone sits next to hot metal in history. The princess could see the sky from her tower. A slow fish swims through dark water at night. Wood and paper burn fast in the fire. Bad things feel wrong while good things feel right.

The tree and flower grow under the light sky. Fear and hate are false emotions that feel real. A fast bird can fly and hear better than a slow cat. Glass and metal shine bright in the present light."""
]

# Chinese translations for base change test
WORDS_ZH = [
    "国王", "女王", "男人", "女人", "王子", "公主",
    "父亲", "母亲", "儿子", "女儿", "兄弟", "姐妹",
    "猫", "狗", "鸟", "鱼", "树", "花", "天空", "地球",
    "快乐", "悲伤", "愤怒", "平静", "爱", "恨", "恐惧", "希望",
    "跑", "走", "跳", "飞", "思考", "感觉", "看", "听",
    "科学", "艺术", "音乐", "数学", "历史", "未来", "过去", "现在",
    "热", "冷", "光", "暗", "大", "小", "快", "慢",
    "好", "坏", "对", "错", "真", "假", "真实", "虚假",
    "水", "火", "空气", "石头", "金属", "木", "玻璃", "纸"
]

SENTENCES_ZH = [
    "国王和女王带着爱和希望统治着王国。",
    "一个男人和一个女人一起走过黑暗的森林。",
    "王子爱他的姐妹，害怕他的兄弟。",
    "父亲和母亲看着儿子跑得很快。",
    "女儿看到光时感到快乐。",
    "一只大猫过去跳过了小狗。",
    "鸟儿飞过地球上空的热空气。",
    "科学和数学帮助我们思考真实的事情。",
    "艺术和音乐让人感到平静和快乐。",
    "未来充满希望，现在感觉真实。",
    "水和火有不同的用途。",
    "历史上，冷石头放在热金属旁边。",
    "公主可以从塔上看到天空。",
    "一条慢鱼在夜晚的黑暗水中游泳。",
    "木头和纸在火中燃烧得很快。",
    "坏事感觉是错的，好事感觉是对的。",
    "树和花在明亮的天空下生长。",
    "恐惧和恨是感觉真实的虚假情感。",
    "快鸟比慢猫飞得更好，听得更清。",
    "玻璃和金属在现在的光中闪闪发亮。"
]

# Multi-scale corpus structure
MULTI_SCALE_CORPUS = {
    "words": WORDS,
    "sentences": SENTENCES,
    "paragraphs": PARAGRAPHS,
    "documents": DOCUMENTS
}

MULTI_SCALE_CORPUS_ZH = {
    "words": WORDS_ZH,
    "sentences": SENTENCES_ZH,
}

# Scale hierarchy for functoriality
SCALE_HIERARCHY = ["words", "sentences", "paragraphs", "documents"]

# Containment mapping: which words appear in which sentences, etc.
def compute_containment_matrix(parent_texts: List[str], child_texts: List[str]) -> np.ndarray:
    """
    Compute binary containment matrix.
    M[i,j] = 1 if child_texts[j] appears in parent_texts[i].

    NOTE ON LANGLANDS INTERPRETATION:
    This tests LINGUISTIC containment (word ⊂ sentence), not representation-theoretic
    functoriality in the strict Langlands sense. The Langlands functor is an
    equivalence of categories preserving L-functions.

    What we test is "embedding hierarchy preservation" - whether L-functions
    correlate across scales. This is an ANALOG of functoriality, demonstrating
    that semantic structure is preserved under hierarchical aggregation.
    The L-function correlation IS meaningful as a test of structural coherence.
    """
    n_parent = len(parent_texts)
    n_child = len(child_texts)
    M = np.zeros((n_parent, n_child))

    for i, parent in enumerate(parent_texts):
        parent_lower = parent.lower()
        for j, child in enumerate(child_texts):
            if child.lower() in parent_lower:
                M[i, j] = 1.0

    return M


@dataclass
class ScaleStructure:
    """Structure at a given scale."""
    name: str
    texts: List[str]
    embeddings: Optional[np.ndarray] = None
    parent_scale: Optional[str] = None
    child_scale: Optional[str] = None
    containment_from_children: Optional[np.ndarray] = None


# =============================================================================
# EMBEDDING AT DIFFERENT SCALES
# =============================================================================

def load_multiscale_embeddings(
    corpus: Dict[str, List[str]],
    model_name: str = "MiniLM",
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load embeddings at all scales for a given model.

    Returns dict mapping scale name to embedding matrix.
    """
    embeddings = {}

    try:
        from sentence_transformers import SentenceTransformer

        model_map = {
            "MiniLM": "all-MiniLM-L6-v2",
            "MPNet": "all-mpnet-base-v2",
            "Paraphrase": "paraphrase-MiniLM-L6-v2",
        }

        model_id = model_map.get(model_name, model_name)

        if verbose:
            print(f"  Loading {model_name} for multi-scale...")

        model = SentenceTransformer(model_id)

        for scale_name, texts in corpus.items():
            embs = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
            embeddings[scale_name] = embs
            if verbose:
                print(f"    {scale_name}: n={len(embs)}, d={embs.shape[1]}")

    except ImportError:
        if verbose:
            print("  sentence-transformers not available")

    return embeddings


def load_bilingual_embeddings(
    corpus_en: Dict[str, List[str]],
    corpus_zh: Dict[str, List[str]],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load bilingual embeddings for base change test.

    Uses multilingual model to embed both English and Chinese.
    """
    embs_en = {}
    embs_zh = {}

    try:
        from sentence_transformers import SentenceTransformer

        if verbose:
            print(f"  Loading multilingual model...")

        model = SentenceTransformer(model_name)

        for scale_name in corpus_en.keys():
            if scale_name in corpus_zh:
                texts_en = corpus_en[scale_name]
                texts_zh = corpus_zh[scale_name]

                embs_en[scale_name] = model.encode(texts_en, normalize_embeddings=False, show_progress_bar=False)
                embs_zh[scale_name] = model.encode(texts_zh, normalize_embeddings=False, show_progress_bar=False)

                if verbose:
                    print(f"    {scale_name}: EN={len(texts_en)}, ZH={len(texts_zh)}, d={embs_en[scale_name].shape[1]}")

    except ImportError:
        if verbose:
            print("  sentence-transformers not available")

    return embs_en, embs_zh


# =============================================================================
# AGGREGATION FUNCTIONS (LIFTING MAPS)
# =============================================================================

def aggregate_embeddings(
    child_embeddings: np.ndarray,
    containment: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate child embeddings to parent scale using containment matrix.

    Methods:
    - "mean": Weighted mean of contained children
    - "max": Max pooling over contained children
    - "attention": Attention-weighted aggregation

    This is the lifting map φ: Rep(G_child) → Rep(G_parent)
    """
    n_parent, n_child = containment.shape
    d = child_embeddings.shape[1]

    parent_embeddings = np.zeros((n_parent, d))

    for i in range(n_parent):
        child_indices = np.where(containment[i] > 0)[0]

        if len(child_indices) == 0:
            continue

        children = child_embeddings[child_indices]

        if method == "mean":
            parent_embeddings[i] = children.mean(axis=0)
        elif method == "max":
            parent_embeddings[i] = children.max(axis=0)
        elif method == "attention":
            # Simple dot-product attention
            query = children.mean(axis=0)
            scores = children @ query
            weights = np.exp(scores - scores.max())
            weights = weights / (weights.sum() + 1e-10)
            parent_embeddings[i] = (children.T @ weights)
        else:
            parent_embeddings[i] = children.mean(axis=0)

    return parent_embeddings


def compute_lifting_map(
    child_embeddings: np.ndarray,
    parent_embeddings: np.ndarray,
    containment: np.ndarray
) -> np.ndarray:
    """
    Compute the linear lifting map φ such that φ(child) ≈ parent.

    Uses least squares: φ = parent^T @ child @ (child^T @ child)^{-1}
    But constrained by containment structure.
    """
    # Build effective child representations for each parent
    n_parent = parent_embeddings.shape[0]
    n_child, d_child = child_embeddings.shape
    d_parent = parent_embeddings.shape[1]

    # For each parent, compute mean of contained children
    effective_child = aggregate_embeddings(child_embeddings, containment, method="mean")

    # Now find linear map from effective_child to parent
    # φ: R^{d_child} → R^{d_parent}
    # Using pseudo-inverse: φ = parent^T @ effective_child^+

    # Regularized pseudo-inverse
    reg = 1e-6 * np.eye(d_child)
    lifting_map = parent_embeddings.T @ effective_child @ np.linalg.inv(effective_child.T @ effective_child + reg)

    return lifting_map.T  # Shape: (d_child, d_parent)


# =============================================================================
# L-FUNCTION COMPUTATION
# =============================================================================

def compute_semantic_l_function(
    embeddings: np.ndarray,
    s_values: np.ndarray,
    n_primes: int = 10
) -> np.ndarray:
    """
    Compute semantic L-function L(s) from embeddings.

    L(s) = Σ_p (local factor at prime p)

    The "primes" are cluster centers found via K-means.
    Local factors are based on distances to these primes.
    """
    from scipy.cluster.vq import kmeans2

    # Find "semantic primes" via K-means
    centroids, labels = kmeans2(embeddings, n_primes, minit='++')

    # Compute local factors
    # L_p(s) = (1 - a_p / p^s)^{-1} for number-theoretic L-functions
    # Semantic analog: L_p(s) = exp(d_p / s) where d_p is mean distance to prime p

    L_values = np.ones(len(s_values), dtype=complex)

    for p_idx, centroid in enumerate(centroids):
        # Distance from all points to this "prime"
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        mean_dist = distances.mean()

        # Local factor contribution
        for i, s in enumerate(s_values):
            if s.real > 0:
                local_factor = np.exp(-mean_dist * s.real)
                L_values[i] *= (1.0 / (1.0 - local_factor + 1e-10))

    return L_values


def compute_l_function_correlation(
    L1: np.ndarray,
    L2: np.ndarray
) -> float:
    """
    Compute correlation between two L-functions.

    For complex values, use magnitude correlation.
    """
    mag1 = np.abs(L1)
    mag2 = np.abs(L2)

    # Normalize
    mag1 = (mag1 - mag1.mean()) / (mag1.std() + 1e-10)
    mag2 = (mag2 - mag2.mean()) / (mag2.std() + 1e-10)

    return float(np.corrcoef(mag1, mag2)[0, 1])


# =============================================================================
# SCALE STRUCTURE BUILDER
# =============================================================================

def build_scale_structures(
    corpus: Dict[str, List[str]],
    model_name: str = "MiniLM",
    verbose: bool = True
) -> Dict[str, ScaleStructure]:
    """
    Build complete scale structure with embeddings and containment matrices.
    """
    structures = {}

    # Load embeddings
    embeddings = load_multiscale_embeddings(corpus, model_name, verbose)

    # Build structures
    for i, scale in enumerate(SCALE_HIERARCHY):
        if scale not in corpus:
            continue

        struct = ScaleStructure(
            name=scale,
            texts=corpus[scale],
            embeddings=embeddings.get(scale),
            parent_scale=SCALE_HIERARCHY[i-1] if i > 0 else None,
            child_scale=SCALE_HIERARCHY[i+1] if i < len(SCALE_HIERARCHY)-1 else None
        )

        # Compute containment from children
        if struct.child_scale and struct.child_scale in corpus:
            struct.containment_from_children = compute_containment_matrix(
                corpus[scale],
                corpus[struct.child_scale]
            )

        structures[scale] = struct

    return structures
