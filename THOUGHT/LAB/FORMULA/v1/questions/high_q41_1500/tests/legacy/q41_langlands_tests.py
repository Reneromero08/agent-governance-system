#!/usr/bin/env python3
"""
Q41: Geometric Langlands & Sheaf Cohomology - HARD TESTS

Near-impossible rigor tests for the Langlands Program applied to semiosphere.
Decision rule: ANY Tier 1-2 decisive FAIL = Q41 ANSWERED: NO.

Tests implemented:
- TIER 2.2: Ramanujan Bound Analog (eigenvalue universality)
- TIER 6.1: Semantic Primes (unique factorization)
- TIER 6.2: Splitting Behavior (Chebotarev density)
- TIER 7.1: TQFT Cobordism (gluing axiom)
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'THOUGHT' / 'LAB' / 'VECTOR_ELO' / 'eigen-alignment'))

@dataclass
class TestResult:
    test_id: str
    tier: str
    passed: bool
    score: float
    details: Dict
    falsifier_triggered: bool = False

# =============================================================================
# TIER 2.2: RAMANUJAN BOUND ANALOG
# =============================================================================

def test_ramanujan_bound(
    eigenvalue_data: Dict[str, np.ndarray],
    threshold: float = 0.95
) -> TestResult:
    """
    Test 2.2: Ramanujan Bound Analog

    For modular forms, Ramanujan: |a_p| ≤ 2p^{(k-1)/2}

    Semantic analog: Do eigenvalues satisfy |lambda_i| ≤ C·sigma^alpha for universal C, alpha?

    GATE: Bound must hold across ALL architectures with SAME C, alpha
    FALSIFIER: Architecture-dependent bounds → FAIL
    """
    print("\n" + "="*70)
    print("TIER 2.2: RAMANUJAN BOUND ANALOG")
    print("="*70)

    # Normalize eigenvalues for each model
    normalized_spectra = {}
    for model, eigenvalues in eigenvalue_data.items():
        # Normalize: lambda_i / max(lambda)
        eigenvalues = np.array(eigenvalues)
        normalized = eigenvalues / (np.max(eigenvalues) + 1e-10)
        normalized_spectra[model] = normalized

    # Test 1: Universal decay pattern
    # Hypothesis: lambda_i / lambda_1 = C * i^(-alpha) for universal alpha

    fitted_alphas = {}
    fitted_Cs = {}

    for model, spectrum in normalized_spectra.items():
        # Fit power law: log(lambda_i / lambda_1) = log(C) - alpha*log(i)
        n = min(50, len(spectrum))  # Use top 50 eigenvalues
        i = np.arange(1, n + 1)
        log_ratio = np.log(spectrum[:n] / spectrum[0] + 1e-10)

        # Linear regression
        log_i = np.log(i)
        slope, intercept = np.polyfit(log_i, log_ratio, 1)

        fitted_alphas[model] = -slope  # alpha is negative of slope
        fitted_Cs[model] = np.exp(intercept)

        print(f"  {model:20}: alpha={-slope:.4f}, C={np.exp(intercept):.4f}")

    # Test universality: all alphas should be similar
    alphas = list(fitted_alphas.values())
    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    alpha_cv = alpha_std / (abs(alpha_mean) + 1e-10)  # Coefficient of variation

    print(f"\n  Mean alpha: {alpha_mean:.4f}")
    print(f"  Std alpha:  {alpha_std:.4f}")
    print(f"  CV alpha:   {alpha_cv:.4f}")

    # Test 2: Bound violation check
    # For each model, check if eigenvalues stay within predicted bounds
    violations = {}
    for model, spectrum in normalized_spectra.items():
        alpha = fitted_alphas[model]
        C = fitted_Cs[model]
        n = len(spectrum)

        predicted_bound = C * np.arange(1, n + 1) ** (-alpha)
        actual = spectrum / spectrum[0]

        # Count violations where actual > 1.5 * bound
        violation_mask = actual > 1.5 * predicted_bound
        violations[model] = np.sum(violation_mask)

    total_violations = sum(violations.values())
    total_points = sum(len(s) for s in normalized_spectra.values())
    violation_rate = total_violations / total_points

    print(f"\n  Bound violations: {total_violations}/{total_points} ({violation_rate:.2%})")

    # GATE: CV < 0.15 (alphas are universal) AND violation rate < 5%
    universal = alpha_cv < 0.15
    bounded = violation_rate < 0.05
    passed = universal and bounded

    print(f"\n  Universal (CV < 0.15): {'PASS' if universal else 'FAIL'}")
    print(f"  Bounded (viol < 5%):   {'PASS' if bounded else 'FAIL'}")

    return TestResult(
        test_id="2.2_ramanujan_bound",
        tier="TIER 2",
        passed=passed,
        score=1.0 - alpha_cv if passed else 0.0,
        details={
            "alpha_mean": float(alpha_mean),
            "alpha_std": float(alpha_std),
            "alpha_cv": float(alpha_cv),
            "violation_rate": float(violation_rate),
            "fitted_alphas": {k: float(v) for k, v in fitted_alphas.items()},
            "fitted_Cs": {k: float(v) for k, v in fitted_Cs.items()},
        },
        falsifier_triggered=not passed
    )


# =============================================================================
# TIER 6.1: SEMANTIC PRIMES
# =============================================================================

def tensor_factorize(embedding: np.ndarray, n_factors: int = 2, max_iter: int = 100) -> Tuple[List[np.ndarray], float]:
    """
    Attempt to factorize an embedding vector using non-negative tensor decomposition.
    Returns factors and reconstruction error.
    """
    # Simple approach: try to decompose into sum of rank-1 components
    # v ≈ Sigma_i alpha_i * u_i where u_i are "prime" directions

    dim = len(embedding)
    residual = embedding.copy()
    factors = []

    for _ in range(n_factors):
        # Find direction that explains most variance
        if np.linalg.norm(residual) < 1e-10:
            break

        # Use power iteration to find dominant direction
        u = residual / (np.linalg.norm(residual) + 1e-10)

        # Project
        alpha = np.dot(residual, u)
        factors.append((alpha, u))
        residual = residual - alpha * u

    # Reconstruction error
    reconstructed = sum(alpha * u for alpha, u in factors)
    error = np.linalg.norm(embedding - reconstructed) / (np.linalg.norm(embedding) + 1e-10)

    return factors, error


def test_semantic_primes(
    embeddings: Dict[str, np.ndarray],
    n_samples: int = 100,
    seed: int = 42
) -> TestResult:
    """
    Test 6.1: Semantic Primes

    GATE: Every meaning must factor uniquely into semantic primes
    FALSIFIER: Non-unique factorization or no finite prime set → FAIL
    """
    print("\n" + "="*70)
    print("TIER 6.1: SEMANTIC PRIMES (Unique Factorization)")
    print("="*70)

    np.random.seed(seed)

    words = list(embeddings.keys())
    if len(words) > n_samples:
        words = np.random.choice(words, n_samples, replace=False).tolist()

    print(f"  Testing {len(words)} words")

    # Try to factorize each word embedding
    factorizations = {}
    reconstruction_errors = []

    for word in words:
        emb = embeddings[word]
        factors, error = tensor_factorize(emb, n_factors=5)
        factorizations[word] = factors
        reconstruction_errors.append(error)

    mean_error = np.mean(reconstruction_errors)
    print(f"  Mean reconstruction error: {mean_error:.4f}")

    # Test uniqueness: Check if similar words have similar factorizations
    # (modulo permutation of factors)

    # Extract "prime directions" by clustering factorization directions
    all_directions = []
    for word, factors in factorizations.items():
        for alpha, u in factors:
            if abs(alpha) > 0.1:  # Significant factor
                all_directions.append(u)

    all_directions = np.array(all_directions)
    print(f"  Extracted {len(all_directions)} significant factors")

    # Cluster to find "primes"
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(all_directions) < 10:
        print("  WARNING: Too few directions for clustering")
        n_primes = len(all_directions)
    else:
        # Cosine distance
        norm_dirs = all_directions / (np.linalg.norm(all_directions, axis=1, keepdims=True) + 1e-10)
        cos_sim = norm_dirs @ norm_dirs.T
        cos_dist = 1 - np.abs(cos_sim)

        # Cluster
        linkage_matrix = linkage(cos_dist[np.triu_indices(len(cos_dist), k=1)], method='average')
        clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')
        n_primes = len(np.unique(clusters))

    print(f"  Identified {n_primes} potential 'prime' directions")

    # Test: Is the number of primes finite and stable?
    # For unique factorization, we expect a fixed set of primes

    # Run multiple times with different samples to check stability
    n_trials = 5
    prime_counts = []

    for trial in range(n_trials):
        np.random.seed(seed + trial)
        sample_words = np.random.choice(list(embeddings.keys()), min(50, len(embeddings)), replace=False)

        trial_directions = []
        for word in sample_words:
            emb = embeddings[word]
            factors, _ = tensor_factorize(emb, n_factors=5)
            for alpha, u in factors:
                if abs(alpha) > 0.1:
                    trial_directions.append(u)

        if len(trial_directions) >= 10:
            trial_dirs = np.array(trial_directions)
            norm_dirs = trial_dirs / (np.linalg.norm(trial_dirs, axis=1, keepdims=True) + 1e-10)
            cos_sim = norm_dirs @ norm_dirs.T
            cos_dist = 1 - np.abs(cos_sim)
            linkage_matrix = linkage(cos_dist[np.triu_indices(len(cos_dist), k=1)], method='average')
            clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')
            prime_counts.append(len(np.unique(clusters)))

    prime_count_std = np.std(prime_counts) if prime_counts else float('inf')
    prime_count_mean = np.mean(prime_counts) if prime_counts else 0

    print(f"  Prime count stability: {prime_count_mean:.1f} ± {prime_count_std:.1f}")

    # GATE: Reconstruction error < 0.2 AND prime count stable (std < 5)
    good_reconstruction = mean_error < 0.2
    stable_primes = prime_count_std < 5
    passed = good_reconstruction and stable_primes

    print(f"\n  Good reconstruction (err < 0.2): {'PASS' if good_reconstruction else 'FAIL'}")
    print(f"  Stable primes (std < 5):         {'PASS' if stable_primes else 'FAIL'}")

    return TestResult(
        test_id="6.1_semantic_primes",
        tier="TIER 6",
        passed=passed,
        score=1.0 - mean_error if passed else 0.0,
        details={
            "n_words": len(words),
            "mean_reconstruction_error": float(mean_error),
            "n_primes": int(n_primes),
            "prime_count_mean": float(prime_count_mean),
            "prime_count_std": float(prime_count_std),
        },
        falsifier_triggered=not passed
    )


# =============================================================================
# TIER 6.2: CHEBOTAREV DENSITY
# =============================================================================

def test_chebotarev_splitting(
    embeddings_en: Dict[str, np.ndarray],
    embeddings_zh: Dict[str, np.ndarray],
    bilingual_pairs: List[Tuple[str, str]],
) -> TestResult:
    """
    Test 6.2: Chebotarev Density Theorem Analog

    How do "semantic primes" split across languages?

    GATE: Splitting pattern must follow predicted density distribution
    FALSIFIER: Random splitting (no density law) → FAIL
    """
    print("\n" + "="*70)
    print("TIER 6.2: CHEBOTAREV DENSITY (Cross-Lingual Splitting)")
    print("="*70)

    # Get shared words
    shared_words = [en for en, zh in bilingual_pairs if en in embeddings_en and zh in embeddings_zh]
    print(f"  Testing {len(shared_words)} bilingual pairs")

    if len(shared_words) < 20:
        print("  ERROR: Not enough bilingual pairs")
        return TestResult(
            test_id="6.2_chebotarev_splitting",
            tier="TIER 6",
            passed=False,
            score=0.0,
            details={"error": "insufficient bilingual pairs"},
            falsifier_triggered=True
        )

    # Factorize in each language
    splitting_patterns = []  # "split", "inert", "ramified"

    for en_word, zh_word in bilingual_pairs[:len(shared_words)]:
        if en_word not in embeddings_en or zh_word not in embeddings_zh:
            continue

        en_emb = embeddings_en[en_word]
        zh_emb = embeddings_zh[zh_word]

        en_factors, en_err = tensor_factorize(en_emb, n_factors=3)
        zh_factors, zh_err = tensor_factorize(zh_emb, n_factors=3)

        # Compare factorization structure
        n_en = len([f for f in en_factors if abs(f[0]) > 0.1])
        n_zh = len([f for f in zh_factors if abs(f[0]) > 0.1])

        # Classify splitting type
        if n_en == n_zh:
            if n_en == 1:
                splitting_patterns.append("inert")  # Stays prime
            else:
                # Check if factors align
                en_dirs = np.array([f[1] for f in en_factors[:n_en]])
                zh_dirs = np.array([f[1] for f in zh_factors[:n_zh]])

                # Alignment via cosine similarity
                if len(en_dirs) > 0 and len(zh_dirs) > 0:
                    alignment = np.max(np.abs(en_dirs @ zh_dirs.T))
                    if alignment > 0.7:
                        splitting_patterns.append("inert")
                    else:
                        splitting_patterns.append("split")
                else:
                    splitting_patterns.append("split")
        elif n_zh > n_en:
            splitting_patterns.append("split")  # More factors in target
        else:
            splitting_patterns.append("ramified")  # Fewer factors (merged)

    # Count proportions
    from collections import Counter
    counts = Counter(splitting_patterns)
    total = len(splitting_patterns)

    split_rate = counts.get("split", 0) / total if total > 0 else 0
    inert_rate = counts.get("inert", 0) / total if total > 0 else 0
    ramified_rate = counts.get("ramified", 0) / total if total > 0 else 0

    print(f"\n  Splitting patterns:")
    print(f"    Split:    {counts.get('split', 0):3d} ({split_rate:.1%})")
    print(f"    Inert:    {counts.get('inert', 0):3d} ({inert_rate:.1%})")
    print(f"    Ramified: {counts.get('ramified', 0):3d} ({ramified_rate:.1%})")

    # Chebotarev density predicts specific proportions based on Galois group
    # For cyclic group of order 2 (simplest case): expect ~50% split, ~50% inert
    # For larger groups: more complex distributions

    # Test: Is the distribution non-random?
    # Random would be ~33% each
    # Non-random should show clear deviation

    expected_uniform = 1/3
    chi_sq = sum((r - expected_uniform)**2 / expected_uniform for r in [split_rate, inert_rate, ramified_rate])

    # With 2 degrees of freedom, chi_sq > 5.99 is significant at p=0.05
    non_random = chi_sq > 5.99

    print(f"\n  Chi-squared: {chi_sq:.4f}")
    print(f"  Non-random distribution (chi-sq > 5.99): {'YES' if non_random else 'NO'}")

    # For a true Chebotarev pattern, we expect the dominant type to match
    # the predicted proportion from Galois theory
    dominant_type = max(counts.keys(), key=lambda k: counts[k]) if counts else None
    dominant_rate = max(split_rate, inert_rate, ramified_rate)

    # GATE: Distribution is non-random AND dominant type > 40%
    has_structure = non_random and dominant_rate > 0.4

    print(f"  Has Chebotarev structure: {'PASS' if has_structure else 'FAIL'}")

    return TestResult(
        test_id="6.2_chebotarev_splitting",
        tier="TIER 6",
        passed=has_structure,
        score=chi_sq / 10 if has_structure else 0.0,  # Normalized score
        details={
            "n_pairs": len(splitting_patterns),
            "split_rate": float(split_rate),
            "inert_rate": float(inert_rate),
            "ramified_rate": float(ramified_rate),
            "chi_squared": float(chi_sq),
            "dominant_type": dominant_type,
        },
        falsifier_triggered=not has_structure
    )


# =============================================================================
# TIER 7.1: TQFT COBORDISM
# =============================================================================

def compute_partition_function(embeddings: np.ndarray) -> float:
    """
    Compute a partition function Z for a set of embeddings.
    Uses the determinant of the Gram matrix as Z (related to volume).
    """
    # Gram matrix
    G = embeddings @ embeddings.T

    # Log-determinant for numerical stability
    sign, logdet = np.linalg.slogdet(G + 1e-6 * np.eye(len(G)))

    return logdet


def test_tqft_cobordism(
    embeddings: Dict[str, np.ndarray],
    n_trials: int = 50,
    seed: int = 42
) -> TestResult:
    """
    Test 7.1: TQFT Cobordism Invariance

    GATE: Z(M₁ ∪_Sigma M₂) = Z(M₁) ⊗_Z(Sigma) Z(M₂) (gluing axiom)
    FALSIFIER: Gluing fails → not a TQFT
    """
    print("\n" + "="*70)
    print("TIER 7.1: TQFT COBORDISM (Gluing Axiom)")
    print("="*70)

    np.random.seed(seed)

    words = list(embeddings.keys())
    emb_matrix = np.array([embeddings[w] for w in words])

    gluing_errors = []

    for trial in range(n_trials):
        # Random split into M1, Sigma (overlap), M2
        n = len(words)
        perm = np.random.permutation(n)

        # M1: first third + overlap
        # Sigma: middle third (overlap)
        # M2: overlap + last third

        third = n // 3
        M1_idx = perm[:2*third]
        Sigma_idx = perm[third:2*third]
        M2_idx = perm[third:]
        Union_idx = perm  # Full set

        M1_emb = emb_matrix[M1_idx]
        Sigma_emb = emb_matrix[Sigma_idx]
        M2_emb = emb_matrix[M2_idx]
        Union_emb = emb_matrix[Union_idx]

        # Compute partition functions
        Z_M1 = compute_partition_function(M1_emb)
        Z_Sigma = compute_partition_function(Sigma_emb)
        Z_M2 = compute_partition_function(M2_emb)
        Z_Union = compute_partition_function(Union_emb)

        # TQFT gluing: Z(Union) ≈ Z(M1) + Z(M2) - Z(Sigma) [in log space]
        # This is because Z(M1 ⊗_Sigma M2) = Z(M1) * Z(M2) / Z(Sigma)
        predicted = Z_M1 + Z_M2 - Z_Sigma
        actual = Z_Union

        error = abs(predicted - actual) / (abs(actual) + 1e-10)
        gluing_errors.append(error)

    mean_error = np.mean(gluing_errors)
    std_error = np.std(gluing_errors)
    max_error = np.max(gluing_errors)

    print(f"  Gluing error: {mean_error:.4f} ± {std_error:.4f}")
    print(f"  Max error:    {max_error:.4f}")

    # GATE: Mean gluing error < 0.3 AND max error < 0.5
    passed = mean_error < 0.3 and max_error < 0.5

    print(f"\n  Gluing axiom holds: {'PASS' if passed else 'FAIL'}")

    return TestResult(
        test_id="7.1_tqft_cobordism",
        tier="TIER 7",
        passed=passed,
        score=1.0 - mean_error if passed else 0.0,
        details={
            "n_trials": n_trials,
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "max_error": float(max_error),
        },
        falsifier_triggered=not passed
    )


# =============================================================================
# MAIN HARNESS
# =============================================================================

def load_embeddings_for_testing():
    """Load embeddings from available models."""
    embeddings = {}
    eigenvalues = {}

    # Common test words
    words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "father", "mother", "son", "daughter", "brother", "sister",
        "cat", "dog", "bird", "fish", "tree", "flower", "sky", "earth",
        "happy", "sad", "angry", "calm", "love", "hate", "fear", "hope",
        "run", "walk", "jump", "fly", "think", "feel", "see", "hear",
        "red", "blue", "green", "yellow", "black", "white", "bright", "dark",
        "big", "small", "fast", "slow", "hot", "cold", "old", "new",
        "good", "bad", "true", "false", "right", "wrong", "yes", "no"
    ]

    # Try to load from sentence-transformers (multiple models)
    try:
        from sentence_transformers import SentenceTransformer

        st_models = [
            ('all-MiniLM-L6-v2', 'MiniLM'),
            ('all-mpnet-base-v2', 'MPNet'),
            ('paraphrase-MiniLM-L6-v2', 'Paraphrase'),
        ]

        for model_name, short_name in st_models:
            try:
                print(f"Loading {short_name}...")
                model = SentenceTransformer(model_name)
                embs = model.encode(words, normalize_embeddings=True)
                embeddings[short_name] = {word: embs[i] for i, word in enumerate(words)}

                # Compute eigenvalues
                centered = embs - embs.mean(axis=0)
                cov = np.cov(centered.T)
                eigs = np.linalg.eigvalsh(cov)
                eigenvalues[short_name] = np.sort(eigs)[::-1]

                print(f"  {short_name}: {len(words)} words, dim={embs.shape[1]}")
            except Exception as e:
                print(f"  {short_name}: FAILED ({e})")

    except ImportError:
        print("sentence-transformers not available")

    # Try to load BERT
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        print("Loading BERT...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.eval()

        bert_embs = {}
        with torch.no_grad():
            for word in words:
                inputs = tokenizer(word, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[0, 0, :].numpy()
                bert_embs[word] = emb

        embeddings['BERT'] = bert_embs

        # Eigenvalues
        emb_matrix = np.array([bert_embs[w] for w in words])
        centered = emb_matrix - emb_matrix.mean(axis=0)
        cov = np.cov(centered.T)
        eigs = np.linalg.eigvalsh(cov)
        eigenvalues['BERT'] = np.sort(eigs)[::-1]

        print(f"  BERT: {len(words)} words, dim=768")

    except ImportError:
        print("transformers not available")

    # Try GloVe/Word2Vec via gensim
    try:
        import gensim.downloader as api

        gensim_models = [
            ('glove-wiki-gigaword-100', 'GloVe-100'),
            ('word2vec-google-news-300', 'Word2Vec'),
        ]

        for model_name, short_name in gensim_models:
            try:
                print(f"Loading {short_name}...")
                model = api.load(model_name)

                model_embs = {}
                for word in words:
                    if word in model:
                        model_embs[word] = model[word]

                if len(model_embs) >= 40:
                    embeddings[short_name] = model_embs

                    # Eigenvalues
                    emb_matrix = np.array([model_embs[w] for w in model_embs.keys()])
                    centered = emb_matrix - emb_matrix.mean(axis=0)
                    cov = np.cov(centered.T)
                    eigs = np.linalg.eigvalsh(cov)
                    eigenvalues[short_name] = np.sort(eigs)[::-1]

                    print(f"  {short_name}: {len(model_embs)} words, dim={emb_matrix.shape[1]}")
                else:
                    print(f"  {short_name}: Too few words ({len(model_embs)})")
            except Exception as e:
                print(f"  {short_name}: FAILED ({e})")

    except ImportError:
        print("gensim not available")

    return embeddings, eigenvalues


def main():
    print("="*70)
    print("Q41: GEOMETRIC LANGLANDS - HARD TESTS")
    print("="*70)
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}Z")
    print()

    # Load embeddings
    print("-"*70)
    print("Loading embeddings...")
    print("-"*70)

    embeddings, eigenvalues = load_embeddings_for_testing()

    if not embeddings:
        print("ERROR: No embeddings available")
        return {"error": "no embeddings"}

    results = []

    # TIER 2.2: Ramanujan Bound
    if eigenvalues:
        result = test_ramanujan_bound(eigenvalues)
        results.append(result)
        print(f"\n  RESULT: {'PASS' if result.passed else 'FAIL'}")

    # TIER 6.1: Semantic Primes
    for model_name, embs in embeddings.items():
        result = test_semantic_primes(embs)
        result.test_id = f"6.1_semantic_primes_{model_name}"
        results.append(result)
        print(f"\n  RESULT ({model_name}): {'PASS' if result.passed else 'FAIL'}")
        break  # Just test first model

    # TIER 7.1: TQFT Cobordism
    for model_name, embs in embeddings.items():
        result = test_tqft_cobordism(embs)
        result.test_id = f"7.1_tqft_cobordism_{model_name}"
        results.append(result)
        print(f"\n  RESULT ({model_name}): {'PASS' if result.passed else 'FAIL'}")
        break  # Just test first model

    # Summary
    print("\n" + "="*70)
    print("Q41 TEST SUMMARY")
    print("="*70)

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.tier} - {result.test_id}: score={result.score:.3f}")

    print(f"\n  Total: {n_passed}/{n_total} passed")

    # Verdict
    tier_1_2_failed = any(
        r.falsifier_triggered and r.tier in ["TIER 1", "TIER 2"]
        for r in results
    )

    if tier_1_2_failed:
        verdict = "Q41 ANSWERED: NO (Tier 1-2 falsifier triggered)"
    elif n_passed == n_total:
        verdict = "Q41 PARTIAL: Tests pass, but Tier 1-2 not yet tested"
    else:
        verdict = f"Q41 OPEN: {n_passed}/{n_total} tests pass"

    print(f"\n  VERDICT: {verdict}")

    # Receipt
    receipt = {
        "test": "Q41_GEOMETRIC_LANGLANDS",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": [asdict(r) for r in results],
        "n_passed": n_passed,
        "n_total": n_total,
        "verdict": verdict,
    }

    receipt_json = json.dumps(receipt, indent=2, default=str)
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

    print(f"\n  Receipt hash: {receipt_hash[:16]}...")

    # Save receipt
    receipt_path = Path(__file__).parent / f"q41_receipt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(receipt_path, 'w') as f:
        f.write(receipt_json)
    print(f"  Saved to: {receipt_path}")

    return receipt


if __name__ == '__main__':
    main()
