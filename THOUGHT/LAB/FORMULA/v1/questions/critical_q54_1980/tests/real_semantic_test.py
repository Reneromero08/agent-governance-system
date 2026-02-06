#!/usr/bin/env python3
"""
REAL SEMANTIC TEST: R = (E / grad_S) * sigma^Df with REAL Embeddings
====================================================================

This test uses REAL pre-trained word embeddings (not synthetic data) to test
whether the R formula predicts that concrete words have higher R than abstract
words.

DATA SOURCES (in order of preference):
1. GloVe embeddings via gensim (glove-wiki-gigaword-50)
2. Word2Vec embeddings via gensim (word2vec-google-news-300)
3. Sentence-transformers (all-MiniLM-L6-v2)

CONSISTENT DEFINITIONS:
- E: Energy = variance of embedding components (how "spread out" the activations)
- grad_S: Entropy gradient = mean distance to k-nearest neighbors (local density)
- sigma: Phase coherence = cosine similarity to category centroid
- Df: Fractal dimension = participation ratio from eigenspectrum

R = (E / grad_S) * sigma^Df

HYPOTHESIS: Concrete words have higher R than abstract words because:
- Concrete words have consistent meanings (high sigma)
- Concrete words cluster tightly (low grad_S)
- This gives higher R = more "locked" semantic meaning

FALSIFICATION: If R_concrete <= R_abstract, the hypothesis is falsified.

Author: Claude Opus 4.5
Date: 2026-01-30
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# WORD LISTS: Concrete vs Abstract
# =============================================================================

# Concrete words: Physical objects that can be perceived with senses
CONCRETE_WORDS = [
    # Animals
    'dog', 'cat', 'horse', 'elephant', 'lion', 'tiger', 'bird', 'fish',
    # Objects
    'table', 'chair', 'book', 'phone', 'car', 'house', 'tree', 'door',
    'window', 'floor', 'wall', 'cup', 'plate', 'knife', 'bottle', 'key',
    # Body parts
    'hand', 'foot', 'head', 'eye', 'arm', 'leg', 'finger', 'nose',
    # Natural objects
    'rock', 'water', 'fire', 'sun', 'moon', 'mountain', 'river', 'ocean',
]

# Abstract words: Concepts that cannot be directly perceived
ABSTRACT_WORDS = [
    # Emotions/States
    'love', 'hate', 'fear', 'hope', 'joy', 'anger', 'peace', 'chaos',
    # Philosophical concepts
    'truth', 'beauty', 'justice', 'freedom', 'wisdom', 'courage', 'faith',
    'meaning', 'purpose', 'existence', 'consciousness', 'reality',
    # Abstract qualities
    'time', 'space', 'infinity', 'eternity', 'destiny', 'fate', 'luck',
    'honor', 'glory', 'virtue', 'evil', 'good', 'power', 'knowledge',
    # Social concepts
    'democracy', 'liberty', 'equality', 'authority', 'theory', 'concept',
]


# =============================================================================
# EMBEDDING LOADERS
# =============================================================================

def load_glove_embeddings(words: List[str], model_name: str = 'glove-wiki-gigaword-50') -> Tuple[Dict[str, np.ndarray], str]:
    """
    Load GloVe embeddings using gensim.

    Returns: (word_to_embedding dict, source description)
    """
    try:
        import gensim.downloader as api
        print(f"Loading {model_name}...")
        model = api.load(model_name)

        embeddings = {}
        missing = []
        for word in words:
            if word in model:
                embeddings[word] = model[word]
            else:
                missing.append(word)

        if missing:
            print(f"  Missing words: {len(missing)} ({', '.join(missing[:5])}...)")

        return embeddings, f"gensim:{model_name}"
    except Exception as e:
        print(f"Failed to load GloVe: {e}")
        return {}, str(e)


def load_word2vec_embeddings(words: List[str]) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Load Word2Vec embeddings using gensim.
    """
    try:
        import gensim.downloader as api
        print("Loading word2vec-google-news-300...")
        model = api.load('word2vec-google-news-300')

        embeddings = {}
        missing = []
        for word in words:
            if word in model:
                embeddings[word] = model[word]
            else:
                missing.append(word)

        if missing:
            print(f"  Missing words: {len(missing)}")

        return embeddings, "gensim:word2vec-google-news-300"
    except Exception as e:
        print(f"Failed to load Word2Vec: {e}")
        return {}, str(e)


def load_sentence_transformer_embeddings(words: List[str]) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Load embeddings using sentence-transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers:all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        vectors = model.encode(words, normalize_embeddings=True, show_progress_bar=False)
        embeddings = {word: vectors[i] for i, word in enumerate(words)}

        return embeddings, "sentence-transformers:all-MiniLM-L6-v2"
    except Exception as e:
        print(f"Failed to load sentence-transformers: {e}")
        return {}, str(e)


def load_real_embeddings(words: List[str]) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Try to load real embeddings in order of preference.
    """
    # Try GloVe first (fastest)
    embeddings, source = load_glove_embeddings(words)
    if len(embeddings) > len(words) * 0.7:  # Got at least 70% of words
        return embeddings, source

    # Try sentence-transformers (guaranteed coverage)
    embeddings, source = load_sentence_transformer_embeddings(words)
    if embeddings:
        return embeddings, source

    # Try Word2Vec as fallback
    embeddings, source = load_word2vec_embeddings(words)
    if embeddings:
        return embeddings, source

    raise RuntimeError("Could not load any real embeddings!")


# =============================================================================
# R FORMULA COMPONENTS - CONSISTENT DEFINITIONS
# =============================================================================

def compute_E(embeddings: np.ndarray) -> np.ndarray:
    """
    E (Energy) = variance of embedding components.

    For each embedding vector v, E = Var(v) = mean((v - mean(v))^2)

    This measures how "spread out" the activations are across dimensions.
    High E = more information encoded, more "energetic" representation.
    """
    n = embeddings.shape[0]
    E_values = np.zeros(n)

    for i in range(n):
        v = embeddings[i]
        E_values[i] = np.var(v)

    return E_values


def compute_grad_S(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """
    grad_S (Entropy Gradient) = mean distance to k-nearest neighbors.

    High grad_S = sparse region (low local density)
    Low grad_S = dense region (high local density)

    In the R formula, low grad_S increases R (stable, locked meaning).
    """
    n = embeddings.shape[0]
    k = min(k, n - 1)

    # Compute all pairwise distances
    distances = cdist(embeddings, embeddings, metric='euclidean')

    grad_S_values = np.zeros(n)

    for i in range(n):
        # Get distances to all other points
        dists = distances[i].copy()
        dists[i] = np.inf  # Exclude self

        # Mean distance to k nearest neighbors
        nearest_k = np.sort(dists)[:k]
        grad_S_values[i] = np.mean(nearest_k)

    # Ensure no zeros (numerical stability)
    grad_S_values = np.maximum(grad_S_values, 1e-10)

    return grad_S_values


def compute_sigma(embeddings: np.ndarray, category_centroid: np.ndarray) -> np.ndarray:
    """
    sigma (Phase Coherence) = cosine similarity to category centroid.

    High sigma = strongly aligned with category (coherent meaning)
    Low sigma = weak categorical identity (incoherent meaning)

    Mapped to [0, 1] range for use as sigma^Df.
    """
    n = embeddings.shape[0]
    sigma_values = np.zeros(n)

    # Normalize centroid
    centroid_norm = np.linalg.norm(category_centroid)
    if centroid_norm > 1e-10:
        centroid_normalized = category_centroid / centroid_norm
    else:
        centroid_normalized = category_centroid

    for i in range(n):
        v = embeddings[i]
        v_norm = np.linalg.norm(v)

        if v_norm > 1e-10:
            v_normalized = v / v_norm
            cos_sim = np.dot(v_normalized, centroid_normalized)
        else:
            cos_sim = 0.0

        # Map from [-1, 1] to [0, 1]
        sigma_values[i] = (cos_sim + 1) / 2

    # Ensure no zeros for sigma^Df computation
    sigma_values = np.maximum(sigma_values, 1e-10)

    return sigma_values


def compute_Df(embeddings: np.ndarray) -> float:
    """
    Df (Fractal Dimension) = participation ratio from eigenspectrum.

    Df = (sum lambda_i)^2 / sum(lambda_i^2)

    This measures the effective dimensionality of the embedding distribution.
    Higher Df = more dimensions participate equally (complex structure).
    Lower Df = few dimensions dominate (simple structure).
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]

    # Compute covariance matrix
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    return float(Df)


def compute_R(E: np.ndarray, grad_S: np.ndarray, sigma: np.ndarray, Df: float) -> np.ndarray:
    """
    R = (E / grad_S) * sigma^Df

    The unified formula for semantic stability.

    High R = stable, "locked" meaning
    Low R = unstable, context-dependent meaning
    """
    R_values = (E / grad_S) * np.power(sigma, Df)
    return R_values


# =============================================================================
# MAIN TEST
# =============================================================================

@dataclass
class CategoryResult:
    name: str
    n_words: int
    words_used: List[str]
    E_mean: float
    E_std: float
    grad_S_mean: float
    grad_S_std: float
    sigma_mean: float
    sigma_std: float
    Df: float
    R_mean: float
    R_std: float
    R_median: float
    R_values: List[float]


@dataclass
class TestResult:
    timestamp: str
    embedding_source: str
    embedding_dim: int
    n_concrete: int
    n_abstract: int

    concrete_results: dict
    abstract_results: dict

    R_concrete_mean: float
    R_abstract_mean: float
    R_ratio: float

    t_statistic: float
    p_value: float
    effect_size_cohens_d: float

    hypothesis_supported: bool
    verdict: str
    interpretation: str


def run_real_semantic_test(verbose: bool = True) -> TestResult:
    """
    Run the real semantic test with actual word embeddings.
    """
    print("=" * 70)
    print("REAL SEMANTIC TEST: R Formula with Real Embeddings")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Concrete words have higher R than abstract words")
    print()
    print("FORMULA: R = (E / grad_S) * sigma^Df")
    print()
    print("DEFINITIONS:")
    print("  E = variance of embedding components (information content)")
    print("  grad_S = mean distance to k-nearest neighbors (local density)")
    print("  sigma = cosine similarity to category centroid (coherence)")
    print("  Df = participation ratio from eigenspectrum (dimensionality)")
    print()
    print("=" * 70)

    # Load real embeddings
    all_words = CONCRETE_WORDS + ABSTRACT_WORDS
    print(f"\nLoading embeddings for {len(all_words)} words...")
    embeddings_dict, source = load_real_embeddings(all_words)

    # Filter to words that have embeddings
    concrete_words = [w for w in CONCRETE_WORDS if w in embeddings_dict]
    abstract_words = [w for w in ABSTRACT_WORDS if w in embeddings_dict]

    print(f"\nEmbedding source: {source}")
    print(f"Concrete words with embeddings: {len(concrete_words)}/{len(CONCRETE_WORDS)}")
    print(f"Abstract words with embeddings: {len(abstract_words)}/{len(ABSTRACT_WORDS)}")

    if len(concrete_words) < 5 or len(abstract_words) < 5:
        raise RuntimeError("Not enough words with embeddings!")

    # Get embedding dimension
    sample_embedding = list(embeddings_dict.values())[0]
    dim = len(sample_embedding)
    print(f"Embedding dimension: {dim}")

    # Build embedding matrices
    concrete_embeddings = np.array([embeddings_dict[w] for w in concrete_words])
    abstract_embeddings = np.array([embeddings_dict[w] for w in abstract_words])

    # Compute GLOBAL centroid from ALL words (concrete + abstract)
    # This fixes the bias where sigma was computed relative to each category's own centroid
    all_embeddings = np.vstack([concrete_embeddings, abstract_embeddings])
    global_centroid = all_embeddings.mean(axis=0)

    print("\n" + "=" * 70)
    print("COMPUTING GLOBAL CENTROID")
    print("=" * 70)
    print(f"\n  Using global centroid from {len(all_embeddings)} words (concrete + abstract)")

    print("\n" + "=" * 70)
    print("COMPUTING R FOR CONCRETE WORDS")
    print("=" * 70)

    # Compute R components for concrete words
    E_concrete = compute_E(concrete_embeddings)
    grad_S_concrete = compute_grad_S(concrete_embeddings, k=5)
    # Use GLOBAL centroid for sigma (not category-specific centroid)
    sigma_concrete = compute_sigma(concrete_embeddings, global_centroid)
    Df_concrete = compute_Df(concrete_embeddings)
    R_concrete = compute_R(E_concrete, grad_S_concrete, sigma_concrete, Df_concrete)

    print(f"\n  E (energy):       mean={np.mean(E_concrete):.6f}, std={np.std(E_concrete):.6f}")
    print(f"  grad_S (density): mean={np.mean(grad_S_concrete):.6f}, std={np.std(grad_S_concrete):.6f}")
    print(f"  sigma (coherence): mean={np.mean(sigma_concrete):.6f}, std={np.std(sigma_concrete):.6f}")
    print(f"  Df (dimension):   {Df_concrete:.4f}")
    print(f"  R (stability):    mean={np.mean(R_concrete):.6f}, std={np.std(R_concrete):.6f}")

    print("\n" + "=" * 70)
    print("COMPUTING R FOR ABSTRACT WORDS")
    print("=" * 70)

    # Compute R components for abstract words
    E_abstract = compute_E(abstract_embeddings)
    grad_S_abstract = compute_grad_S(abstract_embeddings, k=5)
    # Use GLOBAL centroid for sigma (not category-specific centroid)
    sigma_abstract = compute_sigma(abstract_embeddings, global_centroid)
    Df_abstract = compute_Df(abstract_embeddings)
    R_abstract = compute_R(E_abstract, grad_S_abstract, sigma_abstract, Df_abstract)

    print(f"\n  E (energy):       mean={np.mean(E_abstract):.6f}, std={np.std(E_abstract):.6f}")
    print(f"  grad_S (density): mean={np.mean(grad_S_abstract):.6f}, std={np.std(grad_S_abstract):.6f}")
    print(f"  sigma (coherence): mean={np.mean(sigma_abstract):.6f}, std={np.std(sigma_abstract):.6f}")
    print(f"  Df (dimension):   {Df_abstract:.4f}")
    print(f"  R (stability):    mean={np.mean(R_abstract):.6f}, std={np.std(R_abstract):.6f}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    R_concrete_mean = float(np.mean(R_concrete))
    R_abstract_mean = float(np.mean(R_abstract))
    R_ratio = R_concrete_mean / R_abstract_mean if R_abstract_mean > 0 else float('inf')

    print(f"\n  R_concrete mean: {R_concrete_mean:.6f}")
    print(f"  R_abstract mean: {R_abstract_mean:.6f}")
    print(f"  Ratio (concrete/abstract): {R_ratio:.4f}")

    # T-test
    t_stat, p_value = stats.ttest_ind(R_concrete, R_abstract)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(R_concrete) + np.var(R_abstract)) / 2)
    cohens_d = (R_concrete_mean - R_abstract_mean) / pooled_std if pooled_std > 0 else 0

    print(f"\n  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")

    # Interpretation of effect size
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    print(f"  Effect size interpretation: {effect_interp}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    hypothesis_supported = R_concrete_mean > R_abstract_mean and p_value < 0.05

    if hypothesis_supported:
        verdict = "HYPOTHESIS SUPPORTED"
        interpretation = (
            f"Concrete words have significantly higher R than abstract words "
            f"(ratio={R_ratio:.2f}x, p={p_value:.4f}, d={cohens_d:.2f}). "
            f"This supports the claim that R measures semantic stability."
        )
    elif R_concrete_mean > R_abstract_mean:
        verdict = "TREND IN EXPECTED DIRECTION (NOT SIGNIFICANT)"
        interpretation = (
            f"Concrete words have higher R than abstract words "
            f"(ratio={R_ratio:.2f}x) but difference is not statistically significant "
            f"(p={p_value:.4f}). May need more data or refined definitions."
        )
    elif R_concrete_mean == R_abstract_mean:
        verdict = "NO DIFFERENCE"
        interpretation = (
            f"No difference in R between concrete and abstract words. "
            f"The R formula does not distinguish semantic categories."
        )
    else:
        verdict = "HYPOTHESIS FALSIFIED"
        interpretation = (
            f"Abstract words have HIGHER R than concrete words "
            f"(ratio={1/R_ratio:.2f}x, p={p_value:.4f}). "
            f"This contradicts the prediction."
        )

    print(f"\n  {verdict}")
    print(f"\n  {interpretation}")

    # Show top/bottom R words
    print("\n" + "=" * 70)
    print("TOP 5 CONCRETE WORDS BY R")
    print("=" * 70)
    concrete_sorted = sorted(zip(concrete_words, R_concrete), key=lambda x: x[1], reverse=True)
    for word, r in concrete_sorted[:5]:
        print(f"  {word}: R = {r:.6f}")

    print("\n" + "=" * 70)
    print("TOP 5 ABSTRACT WORDS BY R")
    print("=" * 70)
    abstract_sorted = sorted(zip(abstract_words, R_abstract), key=lambda x: x[1], reverse=True)
    for word, r in abstract_sorted[:5]:
        print(f"  {word}: R = {r:.6f}")

    print("\n" + "=" * 70)
    print("BOTTOM 5 CONCRETE WORDS BY R")
    print("=" * 70)
    for word, r in concrete_sorted[-5:]:
        print(f"  {word}: R = {r:.6f}")

    print("\n" + "=" * 70)
    print("BOTTOM 5 ABSTRACT WORDS BY R")
    print("=" * 70)
    for word, r in abstract_sorted[-5:]:
        print(f"  {word}: R = {r:.6f}")

    # Build result
    concrete_result = CategoryResult(
        name="concrete",
        n_words=len(concrete_words),
        words_used=concrete_words,
        E_mean=float(np.mean(E_concrete)),
        E_std=float(np.std(E_concrete)),
        grad_S_mean=float(np.mean(grad_S_concrete)),
        grad_S_std=float(np.std(grad_S_concrete)),
        sigma_mean=float(np.mean(sigma_concrete)),
        sigma_std=float(np.std(sigma_concrete)),
        Df=float(Df_concrete),
        R_mean=float(np.mean(R_concrete)),
        R_std=float(np.std(R_concrete)),
        R_median=float(np.median(R_concrete)),
        R_values=[float(r) for r in R_concrete]
    )

    abstract_result = CategoryResult(
        name="abstract",
        n_words=len(abstract_words),
        words_used=abstract_words,
        E_mean=float(np.mean(E_abstract)),
        E_std=float(np.std(E_abstract)),
        grad_S_mean=float(np.mean(grad_S_abstract)),
        grad_S_std=float(np.std(grad_S_abstract)),
        sigma_mean=float(np.mean(sigma_abstract)),
        sigma_std=float(np.std(sigma_abstract)),
        Df=float(Df_abstract),
        R_mean=float(np.mean(R_abstract)),
        R_std=float(np.std(R_abstract)),
        R_median=float(np.median(R_abstract)),
        R_values=[float(r) for r in R_abstract]
    )

    result = TestResult(
        timestamp=datetime.now().isoformat(),
        embedding_source=source,
        embedding_dim=dim,
        n_concrete=len(concrete_words),
        n_abstract=len(abstract_words),
        concrete_results=asdict(concrete_result),
        abstract_results=asdict(abstract_result),
        R_concrete_mean=R_concrete_mean,
        R_abstract_mean=R_abstract_mean,
        R_ratio=R_ratio,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        effect_size_cohens_d=float(cohens_d),
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        interpretation=interpretation
    )

    return result


def save_results(result: TestResult, output_dir: Path):
    """Save test results to JSON."""
    output_path = output_dir / "real_semantic_test_results.json"

    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Run the real semantic test."""
    result = run_real_semantic_test(verbose=True)

    # Save results
    output_dir = Path(__file__).parent
    save_results(result, output_dir)

    return result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
