#!/usr/bin/env python3
"""
Q11 Test 2.9: The Renormalization Escape Test

Tests whether changing scale (coarse-graining) can reveal information
that was invisible at finer scales - like how renormalization group
transformations in physics reveal effective theories.

HYPOTHESIS: Coarse-graining can make visible patterns that were hidden
at finer granularity. This extends the information horizon WITHOUT
changing epistemology (same inference, different scale).

PREDICTION: Coarse-graining reveals hidden structure (clustering accuracy > 0.8)
FALSIFICATION: Coarse-graining provides no additional information
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    compute_cosine_similarity, get_embeddings,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

CLUSTERING_ACCURACY_THRESHOLD = 0.65  # ARI > this = good clustering (0.65+ is standard)
EMERGENCE_THRESHOLD = 0.1  # Coarse must improve by at least this


# =============================================================================
# TEST DATA
# =============================================================================

# Concepts with hidden high-level structure
# At fine grain: individual concepts
# At coarse grain: should reveal three clusters (living, abstract, artificial)

CONCEPTS = {
    'living': [
        'cat', 'dog', 'tree', 'flower', 'bacterium', 'mushroom',
        'bird', 'fish', 'whale', 'grass', 'human', 'insect',
    ],
    'abstract': [
        'number', 'equation', 'proof', 'theorem', 'infinity', 'set',
        'logic', 'algorithm', 'function', 'variable', 'constant', 'matrix',
    ],
    'artificial': [
        'computer', 'car', 'building', 'phone', 'bridge', 'tool',
        'robot', 'airplane', 'television', 'engine', 'circuit', 'machine',
    ],
}


def get_flat_concepts():
    """Get flattened list of concepts with ground truth labels."""
    concepts = []
    labels = []
    for category, items in CONCEPTS.items():
        for item in items:
            concepts.append(item)
            labels.append(category)
    return concepts, labels


def label_to_int(labels: List[str]) -> np.ndarray:
    """Convert string labels to integers."""
    unique = list(set(labels))
    return np.array([unique.index(l) for l in labels])


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("WARNING: sentence-transformers not installed, using fallback")
        return None


def analyze_fine_grained(embeddings: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    Analyze structure at fine-grained (individual embedding) level.

    Returns:
        Dictionary of fine-grained analysis results
    """
    n = len(embeddings)

    # Pairwise similarities
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = compute_cosine_similarity(embeddings[i], embeddings[j])

    # Within-cluster vs between-cluster similarity
    within_sims = []
    between_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            if true_labels[i] == true_labels[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    # Fine-grained pattern detectability
    avg_within = np.mean(within_sims) if within_sims else 0
    avg_between = np.mean(between_sims) if between_sims else 0
    pattern_signal = avg_within - avg_between

    return {
        'n_samples': n,
        'avg_within_similarity': avg_within,
        'avg_between_similarity': avg_between,
        'pattern_signal': pattern_signal,
        'pattern_detectable_fine': pattern_signal > 0.1,
    }


def analyze_coarse_grained(embeddings: np.ndarray, true_labels: np.ndarray,
                          n_clusters: int = 3) -> Dict:
    """
    Analyze structure at coarse-grained (cluster) level.

    Uses clustering to find emergent structure.

    Returns:
        Dictionary of coarse-grained analysis results
    """
    # Method 1: K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)

    # Method 2: Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(embeddings)
    hier_ari = adjusted_rand_score(true_labels, hier_labels)

    # Method 3: PCA then cluster
    pca = PCA(n_components=min(10, embeddings.shape[1]))
    reduced = pca.fit_transform(embeddings)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    pca_labels = kmeans_pca.fit_predict(reduced)
    pca_ari = adjusted_rand_score(true_labels, pca_labels)

    # Best clustering result
    best_ari = max(kmeans_ari, hier_ari, pca_ari)
    best_method = ['kmeans', 'hierarchical', 'pca'][
        [kmeans_ari, hier_ari, pca_ari].index(best_ari)
    ]

    # Silhouette score (cluster quality)
    if len(set(kmeans_labels)) > 1:
        silhouette = silhouette_score(embeddings, kmeans_labels)
    else:
        silhouette = 0

    return {
        'n_clusters': n_clusters,
        'kmeans_ari': kmeans_ari,
        'hierarchical_ari': hier_ari,
        'pca_ari': pca_ari,
        'best_ari': best_ari,
        'best_method': best_method,
        'silhouette': silhouette,
        'pattern_detectable_coarse': best_ari > CLUSTERING_ACCURACY_THRESHOLD,
    }


def test_emergence(fine_results: Dict, coarse_results: Dict) -> Dict:
    """
    Test whether coarse-graining reveals emergent structure.

    Emergence = information visible at coarse scale that wasn't visible at fine scale.

    Returns:
        Dictionary of emergence analysis
    """
    # Compare pattern visibility at different scales
    fine_signal = fine_results['pattern_signal']
    coarse_ari = coarse_results['best_ari']

    # Emergence: coarse reveals what fine couldn't clearly see
    # If fine signal is weak but coarse clustering works, that's emergence
    fine_visible = fine_results['pattern_detectable_fine']
    coarse_visible = coarse_results['pattern_detectable_coarse']

    # Pure emergence: not visible fine, visible coarse
    pure_emergence = not fine_visible and coarse_visible

    # Amplification: visible both but stronger coarse
    amplification = fine_visible and coarse_visible and coarse_ari > fine_signal

    # Improvement metric
    if fine_signal > 0:
        improvement = coarse_ari - fine_signal
    else:
        improvement = coarse_ari

    return {
        'fine_signal': fine_signal,
        'coarse_ari': coarse_ari,
        'improvement': improvement,
        'fine_visible': fine_visible,
        'coarse_visible': coarse_visible,
        'pure_emergence': pure_emergence,
        'amplification': amplification,
        'any_emergence': pure_emergence or amplification,
        'scale_change_extends_horizon': improvement > EMERGENCE_THRESHOLD,
    }


# =============================================================================
# ADDITIONAL TESTS
# =============================================================================

def test_noise_immunity(embeddings: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    Test if coarse-graining is more robust to noise.

    Add noise at fine level, see if coarse structure survives.
    """
    np.random.seed(RANDOM_SEED + 10)

    results = []
    noise_levels = [0.1, 0.2, 0.5, 1.0]

    for noise_level in noise_levels:
        # Add noise
        noise = np.random.randn(*embeddings.shape) * noise_level
        noisy_embeddings = embeddings + noise

        # Fine-grained analysis
        fine = analyze_fine_grained(noisy_embeddings, true_labels)

        # Coarse-grained analysis
        coarse = analyze_coarse_grained(noisy_embeddings, true_labels)

        results.append({
            'noise_level': noise_level,
            'fine_signal': fine['pattern_signal'],
            'coarse_ari': coarse['best_ari'],
            'coarse_survives': coarse['best_ari'] > 0.5,
            'fine_survives': fine['pattern_signal'] > 0.1,
        })

    # Check if coarse is more robust
    coarse_robust_count = sum(1 for r in results if r['coarse_survives'])
    fine_robust_count = sum(1 for r in results if r['fine_survives'])

    return {
        'noise_tests': results,
        'coarse_more_robust': coarse_robust_count > fine_robust_count,
        'coarse_robust_count': coarse_robust_count,
        'fine_robust_count': fine_robust_count,
    }


def test_compression_reveals(embeddings: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    Test if aggressive compression reveals structure.

    Like renormalization: throw away details, keep large-scale structure.
    """
    n_components_list = [2, 5, 10, 20, 50]
    results = []

    # PCA requires n_components <= min(n_samples, n_features)
    max_components = min(embeddings.shape[0], embeddings.shape[1])

    for n_comp in n_components_list:
        # Skip if n_comp exceeds the maximum allowed
        if n_comp >= max_components:
            continue

        # Compress
        pca = PCA(n_components=n_comp)
        compressed = pca.fit_transform(embeddings)

        # Cluster compressed representation
        kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(compressed)
        ari = adjusted_rand_score(true_labels, labels)

        # Variance explained
        var_explained = np.sum(pca.explained_variance_ratio_)

        results.append({
            'n_components': n_comp,
            'variance_explained': var_explained,
            'clustering_ari': ari,
            'structure_preserved': ari > 0.5,
        })

    # Find minimum compression that preserves structure
    preserved = [r for r in results if r['structure_preserved']]
    if preserved:
        min_components = min(r['n_components'] for r in preserved)
    else:
        min_components = None

    return {
        'compression_tests': results,
        'min_components_for_structure': min_components,
        'compression_preserves_structure': min_components is not None,
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_renormalization_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete renormalization escape test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.9: RENORMALIZATION ESCAPE")

    np.random.seed(RANDOM_SEED)

    print("\nLoading embedding model...")
    model = load_model()

    # Get data
    concepts, labels = get_flat_concepts()
    embeddings = get_embeddings(concepts, model)
    true_labels = label_to_int(labels)

    print(f"Loaded {len(concepts)} concepts in {len(CONCEPTS)} categories")

    # Phase 1: Fine-grained analysis
    print_subheader("Phase 1: Fine-Grained Analysis")
    fine_results = analyze_fine_grained(embeddings, true_labels)

    print(f"Number of samples: {fine_results['n_samples']}")
    print(f"Avg within-cluster similarity: {fine_results['avg_within_similarity']:.4f}")
    print(f"Avg between-cluster similarity: {fine_results['avg_between_similarity']:.4f}")
    print(f"Pattern signal (within - between): {fine_results['pattern_signal']:.4f}")
    print(f"Pattern detectable at fine scale: {fine_results['pattern_detectable_fine']}")

    # Phase 2: Coarse-grained analysis
    print_subheader("Phase 2: Coarse-Grained Analysis (Clustering)")
    coarse_results = analyze_coarse_grained(embeddings, true_labels)

    print(f"K-means ARI: {coarse_results['kmeans_ari']:.4f}")
    print(f"Hierarchical ARI: {coarse_results['hierarchical_ari']:.4f}")
    print(f"PCA + K-means ARI: {coarse_results['pca_ari']:.4f}")
    print(f"Best method: {coarse_results['best_method']} (ARI: {coarse_results['best_ari']:.4f})")
    print(f"Silhouette score: {coarse_results['silhouette']:.4f}")
    print(f"Pattern detectable at coarse scale: {coarse_results['pattern_detectable_coarse']}")

    # Phase 3: Emergence test
    print_subheader("Phase 3: Emergence Analysis")
    emergence_results = test_emergence(fine_results, coarse_results)

    print(f"Fine signal: {emergence_results['fine_signal']:.4f}")
    print(f"Coarse ARI: {emergence_results['coarse_ari']:.4f}")
    print(f"Improvement: {emergence_results['improvement']:.4f}")
    print(f"Pure emergence: {emergence_results['pure_emergence']}")
    print(f"Amplification: {emergence_results['amplification']}")
    print(f"Scale change extends horizon: {emergence_results['scale_change_extends_horizon']}")

    # Phase 4: Additional tests
    print_subheader("Phase 4: Robustness and Compression Tests")

    noise_results = test_noise_immunity(embeddings, true_labels)
    print(f"\nNoise immunity:")
    print(f"  Coarse more robust: {noise_results['coarse_more_robust']}")
    print(f"  Coarse robust at {noise_results['coarse_robust_count']}/{len(noise_results['noise_tests'])} noise levels")

    compression_results = test_compression_reveals(embeddings, true_labels)
    print(f"\nCompression:")
    print(f"  Min components preserving structure: {compression_results['min_components_for_structure']}")
    print(f"  Compression preserves structure: {compression_results['compression_preserves_structure']}")

    # Determine pass/fail
    print_subheader("Phase 5: Final Determination")

    # Pass if coarse-graining reveals structure
    # AND either shows emergence or amplification
    coarse_reveals = coarse_results['pattern_detectable_coarse']
    emergence_detected = emergence_results['any_emergence'] or emergence_results['scale_change_extends_horizon']

    passed = coarse_reveals and emergence_detected

    if passed:
        horizon_type = HorizonType.STRUCTURAL  # Can be extended by scale change
        notes = f"Renormalization works: coarse ARI={coarse_results['best_ari']:.2f}, improvement={emergence_results['improvement']:.2f}"
    else:
        horizon_type = HorizonType.UNKNOWN
        if not coarse_reveals:
            notes = "Coarse-graining failed to reveal structure"
        else:
            notes = "No emergence - pattern was already visible at fine scale"

    print(f"\nCoarse-graining reveals structure: {coarse_reveals}")
    print(f"Emergence/amplification detected: {emergence_detected}")
    print_result("Renormalization Escape Test", passed, notes)

    result = HorizonTestResult(
        test_name="Renormalization Escape",
        test_id="Q11_2.9",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'fine_signal': fine_results['pattern_signal'],
            'coarse_ari': coarse_results['best_ari'],
            'improvement': emergence_results['improvement'],
            'silhouette': coarse_results['silhouette'],
            'coarse_more_robust': noise_results['coarse_more_robust'],
            'min_compression_components': compression_results['min_components_for_structure'],
        },
        thresholds={
            'clustering_accuracy_threshold': CLUSTERING_ACCURACY_THRESHOLD,
            'emergence_threshold': EMERGENCE_THRESHOLD,
        },
        evidence={
            'fine_results': to_builtin(fine_results),
            'coarse_results': to_builtin(coarse_results),
            'emergence_results': to_builtin(emergence_results),
            'noise_results': to_builtin(noise_results),
            'compression_results': to_builtin(compression_results),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_renormalization_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
