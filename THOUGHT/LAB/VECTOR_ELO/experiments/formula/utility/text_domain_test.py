"""
Text Domain Test - Where the formula was derived

The formula R = (E / nabla_H) * sigma^Df was calibrated for text/semantic space.
alpha = 3^(1/2 - 1) = 1/sqrt(3) for 1D semantic similarity.

Test: Does R correctly identify high-value text chunks?
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_text_corpus(n_docs: int = 100, seed: int = 42) -> Dict:
    """
    Simulate text corpus with:
    - Documents have hidden "relevance" to a query
    - Observable: TF-IDF-like similarity scores
    - Observable: Document entropy (vocabulary diversity)
    """
    np.random.seed(seed)

    # Hidden ground truth: relevance to query
    relevance = np.random.pareto(1.5, n_docs) + 0.1
    relevance = relevance / np.max(relevance)

    # Observable: similarity scores (noisy proxy for relevance)
    # High relevance docs have higher similarity on average
    similarity = relevance * 0.7 + np.random.rand(n_docs) * 0.3
    similarity = np.clip(similarity, 0, 1)

    # Observable: document entropy (vocabulary diversity)
    # Simulate: relevant docs tend to be more focused (lower entropy)
    doc_entropy = 2.0 - relevance * 0.5 + np.random.randn(n_docs) * 0.3
    doc_entropy = np.clip(doc_entropy, 0.5, 3.0)

    # Observable: document length (tokens)
    doc_length = np.random.poisson(100, n_docs) + 20

    return {
        'relevance': relevance,      # Hidden
        'similarity': similarity,    # Observable
        'doc_entropy': doc_entropy,  # Observable
        'doc_length': doc_length,    # Observable
        'n_docs': n_docs
    }


def compute_R_text(doc_idx: int, corpus: Dict, query_entropy: float = 1.5) -> float:
    """
    Compute R for a document using text-domain calibration.

    E = information extracted (similarity score as proxy)
    nabla_H = entropy gradient (|doc_entropy - query_entropy|)
    Df = 5 - H (derived from document entropy)
    alpha = 1/sqrt(3) for 1D text space
    """
    similarity = corpus['similarity'][doc_idx]
    doc_entropy = corpus['doc_entropy'][doc_idx]

    # E: similarity as extracted signal (calibrated with alpha)
    alpha = 3 ** (1/2 - 1)  # 1/sqrt(3) for 1D
    E = similarity ** alpha

    # H: document entropy
    H = max(min(doc_entropy, 4.9), 0.1)

    # nabla_H: entropy gradient (how different from query)
    nabla_H = abs(doc_entropy - query_entropy) + 0.01

    # Df: derived from H
    Df = 5 - H

    # R formula
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R


def compute_R_text_v2(doc_idx: int, corpus: Dict, query_entropy: float = 1.5) -> float:
    """
    Alternative: Use doc_entropy directly as H in E calculation.
    """
    similarity = corpus['similarity'][doc_idx]
    doc_entropy = corpus['doc_entropy'][doc_idx]

    # H from document
    H = max(min(doc_entropy, 4.9), 0.1)

    # E derived from H (per calibration)
    alpha = 3 ** (1/2 - 1)
    E = H ** alpha

    # nabla_H: entropy gradient
    nabla_H = abs(doc_entropy - query_entropy) + 0.01

    # Df: derived from H
    Df = 5 - H

    # Scale by similarity (the actual signal)
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df) * similarity

    return R


def rank_by_R(corpus: Dict, R_func, query_entropy: float = 1.5) -> List[int]:
    """Rank documents by R score."""
    R_scores = [R_func(i, corpus, query_entropy) for i in range(corpus['n_docs'])]
    return list(np.argsort(R_scores)[::-1])


def rank_by_similarity(corpus: Dict) -> List[int]:
    """Baseline: rank by raw similarity."""
    return list(np.argsort(corpus['similarity'])[::-1])


def rank_by_entropy(corpus: Dict) -> List[int]:
    """Baseline: rank by low entropy (focused docs)."""
    return list(np.argsort(corpus['doc_entropy']))


def evaluate_ranking(ranking: List[int], corpus: Dict, k: int = 10) -> float:
    """Evaluate top-k precision using hidden relevance."""
    top_k = ranking[:k]
    relevance = corpus['relevance']
    return np.mean([relevance[i] for i in top_k])


def run_text_comparison(n_trials: int = 20, n_docs: int = 100) -> Dict:
    """Compare R-based ranking vs baselines across multiple corpora."""
    results = {
        'R_v1': [],
        'R_v2': [],
        'similarity': [],
        'entropy': [],
        'random': []
    }

    for seed in range(n_trials):
        corpus = create_text_corpus(n_docs=n_docs, seed=seed * 100 + 42)

        # Rankings
        R_v1_ranking = rank_by_R(corpus, compute_R_text)
        R_v2_ranking = rank_by_R(corpus, compute_R_text_v2)
        sim_ranking = rank_by_similarity(corpus)
        ent_ranking = rank_by_entropy(corpus)
        random_ranking = list(np.random.permutation(n_docs))

        # Evaluate top-10
        results['R_v1'].append(evaluate_ranking(R_v1_ranking, corpus, k=10))
        results['R_v2'].append(evaluate_ranking(R_v2_ranking, corpus, k=10))
        results['similarity'].append(evaluate_ranking(sim_ranking, corpus, k=10))
        results['entropy'].append(evaluate_ranking(ent_ranking, corpus, k=10))
        results['random'].append(evaluate_ranking(random_ranking, corpus, k=10))

    return results


def analyze_R_components(corpus: Dict, top_n: int = 10):
    """Analyze what R is picking up in top-ranked docs."""
    R_scores = [compute_R_text(i, corpus) for i in range(corpus['n_docs'])]
    ranking = np.argsort(R_scores)[::-1]

    print("\nTop-10 documents by R:")
    print(f"{'Rank':>4} | {'R':>10} | {'Sim':>6} | {'H':>6} | {'Relevance':>10}")
    print("-" * 50)

    for rank, idx in enumerate(ranking[:top_n]):
        print(f"{rank+1:>4} | {R_scores[idx]:>10.2f} | {corpus['similarity'][idx]:>6.3f} | {corpus['doc_entropy'][idx]:>6.3f} | {corpus['relevance'][idx]:>10.4f}")

    # Compare to similarity ranking
    sim_ranking = rank_by_similarity(corpus)
    print("\nTop-10 documents by raw similarity:")
    print(f"{'Rank':>4} | {'Sim':>6} | {'H':>6} | {'Relevance':>10}")
    print("-" * 40)

    for rank, idx in enumerate(sim_ranking[:top_n]):
        print(f"{rank+1:>4} | {corpus['similarity'][idx]:>6.3f} | {corpus['doc_entropy'][idx]:>6.3f} | {corpus['relevance'][idx]:>10.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("TEXT DOMAIN TEST - Formula's native territory")
    print("=" * 70)
    print()
    print("Testing R = (E / nabla_H) * sigma^Df in text/semantic space")
    print("alpha = 1/sqrt(3) for 1D semantic similarity")
    print()

    # Single corpus analysis
    print("-" * 70)
    print("Single Corpus Analysis")
    print("-" * 70)

    corpus = create_text_corpus(n_docs=100, seed=42)
    analyze_R_components(corpus)

    # Multi-corpus comparison
    print("\n" + "-" * 70)
    print("Multi-Corpus Comparison (20 corpora, top-10 precision)")
    print("-" * 70)

    results = run_text_comparison(n_trials=20, n_docs=100)

    print(f"\n{'Method':>15} | {'Mean Precision':>15} | {'Std':>8}")
    print("-" * 45)
    for method in ['random', 'entropy', 'similarity', 'R_v1', 'R_v2']:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"{method:>15} | {mean:>15.4f} | {std:>8.4f}")

    # Win rates
    print("\n" + "-" * 70)
    print("Win Rates (method beats similarity baseline)")
    print("-" * 70)

    R_v1_wins = sum(1 for r, s in zip(results['R_v1'], results['similarity']) if r > s)
    R_v2_wins = sum(1 for r, s in zip(results['R_v2'], results['similarity']) if r > s)
    n_trials = len(results['similarity'])

    print(f"  R_v1 beats similarity: {R_v1_wins}/{n_trials} corpora")
    print(f"  R_v2 beats similarity: {R_v2_wins}/{n_trials} corpora")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    R_v1_mean = np.mean(results['R_v1'])
    R_v2_mean = np.mean(results['R_v2'])
    sim_mean = np.mean(results['similarity'])

    best_R = max(R_v1_mean, R_v2_mean)
    best_R_name = "R_v1" if R_v1_mean >= R_v2_mean else "R_v2"

    if best_R > sim_mean * 1.05:  # 5% improvement threshold
        print(f"\n** TEXT DOMAIN: R VALIDATED")
        print(f"   {best_R_name} precision: {best_R:.4f}")
        print(f"   Similarity precision: {sim_mean:.4f}")
        print(f"   Improvement: {best_R/sim_mean:.2f}x")
    elif best_R > sim_mean:
        print(f"\n*  TEXT DOMAIN: MARGINAL IMPROVEMENT")
        print(f"   {best_R_name}: {best_R:.4f} vs similarity: {sim_mean:.4f}")
    else:
        print(f"\nX  TEXT DOMAIN: NO IMPROVEMENT")
        print(f"   R does not beat raw similarity in its native domain")
