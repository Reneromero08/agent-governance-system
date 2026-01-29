#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q37: Semiotic Evolution Dynamics - Core Utilities

Infrastructure for testing whether meanings evolve on the M field following
dynamics analogous to biological evolution (competition, speciation,
convergence, selection pressure).

This module provides:
- Eigenspectrum analysis (reusing Q21/Q48-50 foundations)
- Phylogenetic reconstruction metrics
- Semantic distance measurements
- R-gate fitness calculations
- Conservation law tracking (Df x alpha = 8e)

All tests use REAL DATA ONLY. No synthetic simulations.

Key References:
- Q21: Alpha drift detection (temporal dynamics)
- Q38: Noether conservation (CV=6e-7 standard)
- Q48-50: Conservation law Df x alpha = 8e
- Q34: Platonic convergence (cross-lingual)
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Constants from Q48-50 conservation law
EPS = 1e-12
CRITICAL_ALPHA = 0.5  # Riemann critical line
TARGET_DF_ALPHA = 8 * np.e  # ~21.746, conservation law


# =============================================================================
# Eigenspectrum Functions (from Q21/Q48-50)
# =============================================================================

def get_eigenspectrum(embeddings: np.ndarray) -> np.ndarray:
    """
    Get eigenvalues from covariance matrix of embeddings.

    Args:
        embeddings: (n_samples, dim) array

    Returns:
        Sorted eigenvalues (descending order)
    """
    if len(embeddings) < 2:
        return np.array([1.0])
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        return np.array([float(cov)])
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, EPS)


def compute_df(eigenvalues: np.ndarray) -> float:
    """
    Participation ratio Df = (sum(lambda))^2 / sum(lambda^2)
    Measures effective dimensionality.
    """
    ev = eigenvalues[eigenvalues > EPS]
    if len(ev) == 0:
        return 0.0
    return float((np.sum(ev) ** 2) / np.sum(ev ** 2))


def compute_alpha(eigenvalues: np.ndarray) -> float:
    """
    Power law decay exponent alpha where lambda_k ~ k^(-alpha)
    Healthy semantic structure: alpha ~ 0.5 (Riemann critical line)
    """
    ev = eigenvalues[eigenvalues > EPS]
    if len(ev) < 10:
        return 0.0

    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0.0

    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return float(-slope)


def compute_df_alpha(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Df, alpha, and their product for conservation law check.

    Returns:
        (Df, alpha, Df * alpha)
    """
    eigenvalues = get_eigenspectrum(embeddings)
    df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    return df, alpha, df * alpha


# =============================================================================
# R-Gate Fitness Functions
# =============================================================================

def compute_R(embeddings: np.ndarray) -> float:
    """
    Compute R = E / sigma (basic R-gate formula).
    E = mean pairwise similarity, sigma = std of similarities

    This is the fitness measure for semiotic evolution.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + EPS)

    n = len(embeddings)
    if n < 2:
        return 0.0

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)

    if not similarities:
        return 0.0

    E = np.mean(similarities)
    sigma = np.std(similarities)
    return float(E / (sigma + EPS))


def compute_R_between_groups(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute R between two groups of embeddings.
    Used for measuring cross-domain/cross-population fitness.
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0

    # Normalize
    norms1 = np.linalg.norm(group1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(group2, axis=1, keepdims=True)
    norm1 = group1 / (norms1 + EPS)
    norm2 = group2 / (norms2 + EPS)

    # Cross similarities
    similarities = []
    for v1 in norm1:
        for v2 in norm2:
            similarities.append(np.dot(v1, v2))

    if not similarities:
        return 0.0

    E = np.mean(similarities)
    sigma = np.std(similarities)
    return float(E / (sigma + EPS))


# =============================================================================
# Phylogenetic Reconstruction
# =============================================================================

def build_embedding_tree(embeddings: np.ndarray,
                         labels: List[str],
                         method: str = 'average') -> np.ndarray:
    """
    Build hierarchical clustering tree from embeddings.

    Args:
        embeddings: (n_samples, dim) array
        labels: List of n_samples labels
        method: Linkage method ('average', 'ward', 'complete', 'single')

    Returns:
        Linkage matrix Z for scipy.cluster.hierarchy
    """
    # Use cosine distance
    distances = pdist(embeddings, metric='cosine')
    # Handle any NaN from zero vectors
    distances = np.nan_to_num(distances, nan=1.0)
    linkage_matrix = linkage(distances, method=method)
    return linkage_matrix


def get_cluster_assignments(linkage_matrix: np.ndarray,
                           n_clusters: int) -> np.ndarray:
    """
    Get cluster assignments from linkage matrix.
    """
    return fcluster(linkage_matrix, n_clusters, criterion='maxclust')


def compute_phylogeny_metrics(embedding_clusters: np.ndarray,
                             ground_truth_clusters: np.ndarray) -> Dict[str, float]:
    """
    Compare embedding-based phylogeny to ground truth.

    Returns:
        Dict with FMI (Fowlkes-Mallows Index) and ARI (Adjusted Rand Index)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmi = fowlkes_mallows_score(ground_truth_clusters, embedding_clusters)
        ari = adjusted_rand_score(ground_truth_clusters, embedding_clusters)

    return {
        'fowlkes_mallows_index': fmi,
        'adjusted_rand_index': ari
    }


# =============================================================================
# Semantic Distance Measurements
# =============================================================================

def semantic_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute semantic distance as 1 - cosine_similarity.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + EPS)
    v2_norm = v2 / (np.linalg.norm(v2) + EPS)
    return 1.0 - np.dot(v1_norm, v2_norm)


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """
    Compute mean pairwise semantic distance within a group.
    """
    if len(embeddings) < 2:
        return 0.0
    distances = pdist(embeddings, metric='cosine')
    distances = np.nan_to_num(distances, nan=1.0)
    return float(np.mean(distances))


def inter_group_distance(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute mean semantic distance between two groups.
    """
    if len(group1) == 0 or len(group2) == 0:
        return 1.0

    distances = []
    for v1 in group1:
        for v2 in group2:
            d = semantic_distance(v1, v2)
            distances.append(d)

    return float(np.mean(distances))


# =============================================================================
# Evolution Tracking
# =============================================================================

@dataclass
class EvolutionSnapshot:
    """Snapshot of evolutionary state at a point in time."""
    timestamp: str
    embeddings: np.ndarray
    labels: List[str]
    R: float
    Df: float
    alpha: float
    df_alpha: float
    mean_distance: float


@dataclass
class EvolutionTrajectory:
    """Track evolution over time."""
    snapshots: List[EvolutionSnapshot] = field(default_factory=list)

    def add_snapshot(self, timestamp: str, embeddings: np.ndarray, labels: List[str]):
        """Add a new snapshot to the trajectory."""
        R = compute_R(embeddings)
        Df, alpha, df_alpha = compute_df_alpha(embeddings)
        mean_dist = mean_pairwise_distance(embeddings)

        snapshot = EvolutionSnapshot(
            timestamp=timestamp,
            embeddings=embeddings,
            labels=labels,
            R=R,
            Df=Df,
            alpha=alpha,
            df_alpha=df_alpha,
            mean_distance=mean_dist
        )
        self.snapshots.append(snapshot)

    def get_R_trajectory(self) -> List[float]:
        return [s.R for s in self.snapshots]

    def get_df_alpha_trajectory(self) -> List[float]:
        return [s.df_alpha for s in self.snapshots]

    def get_conservation_cv(self) -> float:
        """Compute CV of Df*alpha across trajectory."""
        values = self.get_df_alpha_trajectory()
        if len(values) < 2:
            return 0.0
        return float(np.std(values) / (np.mean(values) + EPS))


# =============================================================================
# Speciation Detection
# =============================================================================

def detect_speciation(population1: np.ndarray,
                      population2: np.ndarray,
                      threshold: float = 0.7) -> Dict[str, any]:
    """
    Detect if two populations have speciated (diverged beyond reconciliation).

    Speciation criteria:
    - Inter-population distance > threshold
    - Cross-population R < 0.3

    Returns:
        Dict with speciation metrics
    """
    inter_dist = inter_group_distance(population1, population2)
    intra_dist1 = mean_pairwise_distance(population1)
    intra_dist2 = mean_pairwise_distance(population2)
    cross_R = compute_R_between_groups(population1, population2)

    speciated = inter_dist > threshold and cross_R < 0.3

    return {
        'speciated': speciated,
        'inter_population_distance': inter_dist,
        'intra_population_distance_1': intra_dist1,
        'intra_population_distance_2': intra_dist2,
        'cross_population_R': cross_R,
        'distance_ratio': inter_dist / (max(intra_dist1, intra_dist2) + EPS)
    }


# =============================================================================
# Hyponymy / Hierarchy Prediction
# =============================================================================

def predict_hyponymy(embeddings: np.ndarray,
                     labels: List[str],
                     k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Predict is-a (hyponymy) relations from embedding distances.

    Heuristic: Hypernyms (more general) tend to be centroid-adjacent,
    hyponyms (more specific) cluster around them.

    Returns:
        List of (hyponym, hypernym, confidence) predictions
    """
    if len(embeddings) < 2:
        return []

    # Compute centroid
    centroid = np.mean(embeddings, axis=0)

    # Distance from centroid (more general = closer to centroid)
    distances_to_centroid = []
    for i, emb in enumerate(embeddings):
        d = semantic_distance(emb, centroid)
        distances_to_centroid.append((labels[i], d))

    # Sort by distance (closest = most general)
    distances_to_centroid.sort(key=lambda x: x[1])

    # Predict: items further from centroid are hyponyms of items closer
    predictions = []
    for i, (hypo_label, hypo_dist) in enumerate(distances_to_centroid[k:], k):
        # Find k nearest that are closer to centroid
        for hyper_label, hyper_dist in distances_to_centroid[:i]:
            # Confidence based on distance difference
            confidence = (hypo_dist - hyper_dist) / (hypo_dist + EPS)
            predictions.append((hypo_label, hyper_label, confidence))
            if len([p for p in predictions if p[0] == hypo_label]) >= k:
                break

    return predictions


# =============================================================================
# Test Result Structures
# =============================================================================

@dataclass
class TierResult:
    """Result of a single tier test."""
    tier: int
    test_name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'tier': self.tier,
            'test': self.test_name,
            'passed': self.passed,
            'metric': self.metric_name,
            'value': self.metric_value,
            'threshold': self.threshold,
            'details': self.details
        }


@dataclass
class Q37TestSuite:
    """Aggregated results from all Q37 tests."""
    results: List[TierResult] = field(default_factory=list)

    def add_result(self, result: TierResult):
        self.results.append(result)

    def get_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def get_tier_summary(self) -> Dict[int, Dict]:
        summary = {}
        for r in self.results:
            if r.tier not in summary:
                summary[r.tier] = {'passed': 0, 'failed': 0, 'tests': []}
            if r.passed:
                summary[r.tier]['passed'] += 1
            else:
                summary[r.tier]['failed'] += 1
            summary[r.tier]['tests'].append(r.test_name)
        return summary

    def print_summary(self):
        print("\n" + "=" * 60)
        print("Q37 SEMIOTIC EVOLUTION DYNAMICS - TEST RESULTS")
        print("=" * 60)

        tier_summary = self.get_tier_summary()
        for tier in sorted(tier_summary.keys()):
            info = tier_summary[tier]
            status = "PASS" if info['failed'] == 0 else "PARTIAL" if info['passed'] > 0 else "FAIL"
            print(f"\nTIER {tier}: {status} ({info['passed']}/{info['passed'] + info['failed']})")
            for r in self.results:
                if r.tier == tier:
                    mark = "[PASS]" if r.passed else "[FAIL]"
                    print(f"  {mark} {r.test_name}: {r.metric_value:.4f} (threshold: {r.threshold})")

        print("\n" + "-" * 60)
        total_passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"OVERALL: {total_passed}/{total} tests passed ({self.get_pass_rate()*100:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    # Quick smoke test
    print("Q37 Evolution Utils - Smoke Test")

    # Generate some test embeddings
    np.random.seed(42)
    test_embeddings = np.random.randn(100, 384)
    test_labels = [f"word_{i}" for i in range(100)]

    # Test basic functions
    R = compute_R(test_embeddings)
    Df, alpha, df_alpha = compute_df_alpha(test_embeddings)

    print(f"R = {R:.4f}")
    print(f"Df = {Df:.4f}")
    print(f"alpha = {alpha:.4f}")
    print(f"Df * alpha = {df_alpha:.4f} (target: {TARGET_DF_ALPHA:.4f})")

    # Test phylogeny
    linkage_matrix = build_embedding_tree(test_embeddings, test_labels)
    clusters = get_cluster_assignments(linkage_matrix, n_clusters=10)
    print(f"Built tree with {len(set(clusters))} clusters")

    print("\nSmoke test passed!")
