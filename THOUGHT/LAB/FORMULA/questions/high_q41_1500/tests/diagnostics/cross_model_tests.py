#!/usr/bin/env python3
"""
Q41 Diagnostic Tests: Cross-Model Comparisons

These tests compare geometric properties across different embedding models.
They are heuristics with positive/negative controls for validation.

Diagnostic Tests:
1. Spectral Signature - Graph Laplacian eigenvalue comparison
2. Heat Trace Fingerprint - Multi-scale geometry via heat trace curves
3. Distance Correlation - Agreement on relative distances
4. Covariance Spectrum - Eigenvalue decay profiles
5. Sparse Coding Stability - SVD basis stability across subsamples
6. Multiscale Connectivity - Component merging at different scales

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import sys
import json
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin,
    DEFAULT_CORPUS, load_embeddings, preprocess_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_DIAGNOSTIC_TESTS"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(val: Any) -> float:
    """Convert to float safely."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except:
        return 0.0


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """Compute correlation safely."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    try:
        if method == "pearson":
            corr, _ = pearsonr(x, y)
        else:
            corr, _ = spearmanr(x, y)
        if math.isnan(corr) or math.isinf(corr):
            return 0.0
        return float(corr)
    except:
        return 0.0


def normalized_l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute normalized L2 distance."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    dist = np.linalg.norm(x - y)
    scale = np.linalg.norm(x) + np.linalg.norm(y) + 1e-10
    return float(dist / scale)


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix."""
    if metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        sim = X_norm @ X_norm.T
        D = 1.0 - sim
        np.fill_diagonal(D, 0)
    else:
        D = squareform(pdist(X, metric="euclidean"))
    return D


def build_mutual_knn_graph(D: np.ndarray, k: int) -> np.ndarray:
    """Build mutual k-NN graph."""
    n = len(D)
    k = min(k, n - 1)
    knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    A_asym = np.zeros((n, n), dtype=int)
    for i in range(n):
        A_asym[i, knn_idx[i]] = 1
    A = (A_asym * A_asym.T).astype(float)
    return A


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Compute normalized graph Laplacian."""
    n = len(A)
    degrees = np.sum(A, axis=1)
    degrees[degrees == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L = (L + L.T) / 2.0
    return L


def heat_trace_from_laplacian(L: np.ndarray, t_grid: List[float]) -> np.ndarray:
    """Compute heat trace for multiple t values."""
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.maximum(eigenvalues, 0)
    return np.array([np.sum(np.exp(-t * eigenvalues)) for t in t_grid])


def random_orthogonal_matrix(d: int, seed: int) -> np.ndarray:
    """Generate random orthogonal matrix."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def generate_controls(X: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate control embeddings."""
    rng = np.random.RandomState(seed)
    n, d = X.shape

    Q = random_orthogonal_matrix(d, seed)
    rotated = X @ Q

    shuffled = X.copy()
    for j in range(d):
        shuffled[:, j] = rng.permutation(shuffled[:, j])

    gaussian = rng.randn(n, d)
    U, _, Vt = np.linalg.svd(gaussian, full_matrices=False)
    flat_spectrum = np.ones(min(n, d))
    gaussian = U @ np.diag(flat_spectrum[:min(n, d)]) @ Vt[:min(n, d), :]
    gaussian = gaussian / (np.linalg.norm(gaussian, axis=1, keepdims=True) + 1e-10)
    gaussian = gaussian * np.linalg.norm(X, axis=1, keepdims=True).mean()

    noise_scale = np.std(X) * 2.0
    noise_corrupted = X + rng.randn(n, d) * noise_scale

    return {
        "rotated": rotated,
        "shuffled": shuffled,
        "gaussian": gaussian,
        "noise_corrupted": noise_corrupted
    }


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def test_spectral_signature(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC A: Spectral Signature of Graph Laplacian"""
    heat_t_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results_per_model = {}
    spectra = {}
    heat_traces = {}

    for name, X in embeddings_dict.items():
        D = pairwise_distances(X, config.distance_metric)
        A = build_mutual_knn_graph(D, config.k_neighbors)
        L = normalized_laplacian(A)

        eigs = np.sort(np.linalg.eigvalsh(L))
        heat = heat_trace_from_laplacian(L, heat_t_grid)

        n_components, _ = connected_components(A > 0, directed=False)
        mean_degree = np.mean(np.sum(A, axis=1))

        spectra[name] = eigs
        heat_traces[name] = heat

        results_per_model[name] = {
            "spectral_gap": safe_float(eigs[1]) if len(eigs) > 1 else 0.0,
            "max_eigenvalue": safe_float(eigs[-1]),
            "n_components": int(n_components),
            "mean_degree": safe_float(mean_degree)
        }

    model_names = list(spectra.keys())
    spectral_distances = []
    heat_distances = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            min_len = min(len(spectra[m1]), len(spectra[m2]))
            spec_dist = normalized_l2_distance(spectra[m1][:min_len], spectra[m2][:min_len])
            spectral_distances.append((m1, m2, safe_float(spec_dist)))

            heat_dist = normalized_l2_distance(heat_traces[m1], heat_traces[m2])
            heat_distances.append((m1, m2, safe_float(heat_dist)))

    mean_spectral_dist = safe_float(np.mean([c[2] for c in spectral_distances])) if spectral_distances else 0
    mean_heat_dist = safe_float(np.mean([c[2] for c in heat_distances])) if heat_distances else 0

    # Controls
    first_model = model_names[0]
    X_first = embeddings_dict[first_model]
    controls = generate_controls(X_first, config.seed)

    X_rot = controls["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)
    A_rot = build_mutual_knn_graph(D_rot, config.k_neighbors)
    L_rot = normalized_laplacian(A_rot)
    eigs_rot = np.sort(np.linalg.eigvalsh(L_rot))

    rot_error = float(np.max(np.abs(spectra[first_model][:len(eigs_rot)] - eigs_rot[:len(spectra[first_model])])))
    positive_control_pass = bool(rot_error < config.identity_tolerance * 100)

    X_noise = controls["noise_corrupted"]
    D_noise = pairwise_distances(X_noise, config.distance_metric)
    A_noise = build_mutual_knn_graph(D_noise, config.k_neighbors)
    L_noise = normalized_laplacian(A_noise)
    eigs_noise = np.sort(np.linalg.eigvalsh(L_noise))

    noise_spec_dist = normalized_l2_distance(spectra[first_model], eigs_noise)
    negative_control_pass = bool(noise_spec_dist > 0.03)

    passed = bool(positive_control_pass and negative_control_pass)

    return TestResult(
        name="spectral_signature",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "per_model": results_per_model,
            "spectral_l2_distances": spectral_distances,
            "heat_l2_distances": heat_distances,
            "mean_spectral_distance": mean_spectral_dist,
            "mean_heat_distance": mean_heat_dist
        },
        thresholds={
            "positive_control_error": config.identity_tolerance * 100,
            "negative_l2_distance_min": 0.03
        },
        controls={
            "positive_rotated_error": safe_float(rot_error),
            "positive_control_pass": positive_control_pass,
            "negative_noise_l2_distance": safe_float(noise_spec_dist),
            "negative_control_pass": negative_control_pass
        },
        notes="Graph Laplacian spectral signatures"
    )


def test_heat_trace_fingerprint(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC B: Heat Trace Curves as Shape Fingerprints"""
    heat_t_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    heat_curves = {}

    for name, X in embeddings_dict.items():
        D = pairwise_distances(X, config.distance_metric)
        A = build_mutual_knn_graph(D, config.k_neighbors)
        L = normalized_laplacian(A)
        heat = heat_trace_from_laplacian(L, heat_t_grid)
        heat_curves[name] = heat / (np.linalg.norm(heat) + 1e-10)

    model_names = list(heat_curves.keys())
    curve_distances = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            dist = np.linalg.norm(heat_curves[m1] - heat_curves[m2])
            curve_distances.append((m1, m2, safe_float(dist)))

    mean_distance = safe_float(np.mean([d[2] for d in curve_distances])) if curve_distances else 0

    # Controls
    first_model = model_names[0]
    X_first = embeddings_dict[first_model]
    controls = generate_controls(X_first, config.seed)

    X_rot = controls["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)
    A_rot = build_mutual_knn_graph(D_rot, config.k_neighbors)
    L_rot = normalized_laplacian(A_rot)
    heat_rot = heat_trace_from_laplacian(L_rot, heat_t_grid)
    heat_rot_norm = heat_rot / (np.linalg.norm(heat_rot) + 1e-10)

    pos_dist = float(np.linalg.norm(heat_curves[first_model] - heat_rot_norm))
    positive_control_pass = bool(pos_dist < config.diagnostic_threshold)

    X_noise = controls["noise_corrupted"]
    D_noise = pairwise_distances(X_noise, config.distance_metric)
    A_noise = build_mutual_knn_graph(D_noise, config.k_neighbors)
    L_noise = normalized_laplacian(A_noise)
    heat_noise = heat_trace_from_laplacian(L_noise, heat_t_grid)
    heat_noise_norm = heat_noise / (np.linalg.norm(heat_noise) + 1e-10)

    neg_dist = float(np.linalg.norm(heat_curves[first_model] - heat_noise_norm))
    neg_dist_min = config.diagnostic_threshold * 0.2
    negative_control_pass = bool(neg_dist >= neg_dist_min or neg_dist >= pos_dist * 5.0)

    passed = bool(positive_control_pass and negative_control_pass)

    return TestResult(
        name="heat_trace_fingerprint",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "heat_curve_distances": curve_distances,
            "mean_distance": mean_distance
        },
        thresholds={"diagnostic_threshold": config.diagnostic_threshold},
        controls={
            "positive_rotated_distance": safe_float(pos_dist),
            "positive_control_pass": positive_control_pass,
            "negative_noise_distance": safe_float(neg_dist),
            "negative_control_pass": negative_control_pass
        },
        notes="Heat trace curves encode multi-scale geometry"
    )


def test_distance_correlation(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC C: Distance Matrix Correlation"""
    dist_matrices = {}
    for name, X in embeddings_dict.items():
        D = pairwise_distances(X, config.distance_metric)
        idx = np.triu_indices(len(D), k=1)
        dist_matrices[name] = D[idx]

    model_names = list(dist_matrices.keys())
    correlations = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            d1, d2 = dist_matrices[m1], dist_matrices[m2]
            min_len = min(len(d1), len(d2))
            corr = safe_correlation(d1[:min_len], d2[:min_len], method="spearman")
            correlations.append((m1, m2, safe_float(corr)))

    mean_corr = safe_float(np.mean([c[2] for c in correlations])) if correlations else 0

    # Controls
    first_model = model_names[0]
    X_first = embeddings_dict[first_model]
    controls = generate_controls(X_first, config.seed)

    X_rot = controls["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)
    idx = np.triu_indices(len(D_rot), k=1)
    d_rot = D_rot[idx]

    pos_corr = safe_correlation(dist_matrices[first_model], d_rot, method="spearman")
    positive_control_pass = bool(pos_corr > 0.999)

    X_shuf = controls["shuffled"]
    D_shuf = pairwise_distances(X_shuf, config.distance_metric)
    d_shuf = D_shuf[idx]

    neg_corr = safe_correlation(dist_matrices[first_model][:len(d_shuf)], d_shuf[:len(dist_matrices[first_model])], method="spearman")
    negative_control_pass = bool(abs(neg_corr) < 0.5)

    passed = bool(positive_control_pass and negative_control_pass)

    return TestResult(
        name="distance_correlation",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "pairwise_correlations": correlations,
            "mean_correlation": mean_corr
        },
        thresholds={"positive_threshold": 0.999, "negative_threshold": 0.5},
        controls={
            "positive_rotated_correlation": safe_float(pos_corr),
            "positive_control_pass": positive_control_pass,
            "negative_shuffled_correlation": safe_float(neg_corr),
            "negative_control_pass": negative_control_pass
        },
        notes="Distance matrix correlation measures agreement on relative distances"
    )


def test_covariance_spectrum(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC D: Covariance Eigenspectrum Comparison"""
    spectra = {}
    stats = {}

    for name, X in embeddings_dict.items():
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigs_norm = eigs / (eigs[0] + 1e-10)
        spectra[name] = eigs_norm

        pr = (np.sum(eigs)**2) / (np.sum(eigs**2) + 1e-10)

        k_fit = np.arange(1, min(31, len(eigs_norm)))
        log_k = np.log(k_fit)
        log_eig = np.log(eigs_norm[:len(k_fit)] + 1e-10)

        if len(k_fit) >= 3:
            slope, intercept = np.polyfit(log_k, log_eig, 1)
            alpha = -slope
        else:
            alpha = 0

        stats[name] = {
            "participation_ratio": safe_float(pr),
            "decay_exponent_alpha": safe_float(alpha)
        }

    model_names = list(spectra.keys())
    spectral_l2_distances = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            min_len = min(len(spectra[m1]), len(spectra[m2]), 30)
            l2_dist = normalized_l2_distance(spectra[m1][:min_len], spectra[m2][:min_len])
            spectral_l2_distances.append((m1, m2, safe_float(l2_dist)))

    alphas = [s["decay_exponent_alpha"] for s in stats.values()]
    alpha_mean = safe_float(np.mean(alphas))
    alpha_cv = safe_float(np.std(alphas) / (abs(alpha_mean) + 1e-10))

    # Controls
    first_model = model_names[0]
    X_first = embeddings_dict[first_model]
    controls = generate_controls(X_first, config.seed)

    X_rot = controls["rotated"]
    X_rot_centered = X_rot - X_rot.mean(axis=0)
    cov_rot = np.cov(X_rot_centered.T)
    eigs_rot = np.sort(np.linalg.eigvalsh(cov_rot))[::-1]
    eigs_rot_norm = eigs_rot / (eigs_rot[0] + 1e-10)

    min_len = min(len(spectra[first_model]), len(eigs_rot_norm))
    pos_error = float(np.max(np.abs(spectra[first_model][:min_len] - eigs_rot_norm[:min_len])))
    positive_control_pass = bool(pos_error < config.identity_tolerance * 100)

    X_gauss = controls["gaussian"]
    X_gauss_centered = X_gauss - X_gauss.mean(axis=0)
    cov_gauss = np.cov(X_gauss_centered.T)
    eigs_gauss = np.sort(np.linalg.eigvalsh(cov_gauss))[::-1]

    pr_gauss = (np.sum(eigs_gauss)**2) / (np.sum(eigs_gauss**2) + 1e-10)
    pr_orig = stats[first_model]["participation_ratio"]
    pr_diff_ratio = abs(pr_gauss - pr_orig) / (pr_orig + 1e-10)

    negative_control_pass = bool(pr_diff_ratio > 0.3)

    passed = bool(positive_control_pass and negative_control_pass and alpha_cv < 0.3)

    return TestResult(
        name="covariance_spectrum",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "per_model": stats,
            "spectral_l2_distances": spectral_l2_distances,
            "alpha_mean": alpha_mean,
            "alpha_cv": alpha_cv
        },
        thresholds={"alpha_cv_threshold": 0.3},
        controls={
            "positive_rotated_error": safe_float(pos_error),
            "positive_control_pass": positive_control_pass,
            "negative_pr_diff_ratio": safe_float(pr_diff_ratio),
            "negative_control_pass": negative_control_pass
        },
        notes="Covariance eigenspectrum comparison"
    )


def test_sparse_coding_stability(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC E: Sparse Coding Stability"""
    rng = np.random.RandomState(config.seed)

    first_model = list(embeddings_dict.keys())[0]
    X = embeddings_dict[first_model]
    n, d = X.shape

    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_components = min(30, len(S))
    basis_ref = Vt[:n_components]

    stability_scores = []
    n_trials = 10

    for _ in range(n_trials):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        X_sub = X[idx]
        X_sub_centered = X_sub - X_sub.mean(axis=0)

        _, _, Vt_sub = np.linalg.svd(X_sub_centered, full_matrices=False)
        basis_sub = Vt_sub[:n_components]

        alignment = np.abs(basis_ref @ basis_sub.T)
        max_alignment = np.max(alignment, axis=1)
        stability_scores.append(float(np.mean(max_alignment)))

    mean_stability = safe_float(np.mean(stability_scores))

    reconstruction_errors = []
    for i in range(min(50, n)):
        x = X_centered[i]
        coeffs = basis_ref @ x
        x_recon = coeffs @ basis_ref
        err = np.linalg.norm(x - x_recon) / (np.linalg.norm(x) + 1e-10)
        reconstruction_errors.append(float(err))

    mean_recon_error = safe_float(np.mean(reconstruction_errors))

    # Controls
    controls_data = generate_controls(X, config.seed)

    X_rot = controls_data["rotated"]
    X_rot_centered = X_rot - X_rot.mean(axis=0)
    _, S_rot, _ = np.linalg.svd(X_rot_centered, full_matrices=False)

    sv_rel_error = float(np.max(np.abs(S[:n_components] - S_rot[:n_components])) / (np.max(S) + 1e-10))
    positive_control_pass = bool(sv_rel_error < 1e-6)

    X_noise = controls_data["noise_corrupted"]
    X_noise_centered = X_noise - X_noise.mean(axis=0)
    _, S_noise, Vt_noise = np.linalg.svd(X_noise_centered, full_matrices=False)

    basis_noise = Vt_noise[:n_components]
    alignment_noise = np.abs(basis_ref @ basis_noise.T)
    max_alignment_noise = safe_float(np.mean(np.max(alignment_noise, axis=1)))

    S_norm = S[:n_components] / (S[0] + 1e-10)
    S_noise_norm = S_noise[:n_components] / (S_noise[0] + 1e-10)
    log_S = np.log(S_norm + 1e-10)
    log_S_noise = np.log(S_noise_norm[:len(S_norm)] + 1e-10)
    sv_log_l2_distance = normalized_l2_distance(log_S, log_S_noise)

    negative_control_pass = bool(max_alignment_noise < 0.7 or sv_log_l2_distance > 0.1)

    passed = bool(positive_control_pass and negative_control_pass and mean_stability > 0.5)

    return TestResult(
        name="sparse_coding_stability",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "n_components": n_components,
            "mean_stability": mean_stability,
            "mean_reconstruction_error": mean_recon_error
        },
        thresholds={"stability_threshold": 0.5},
        controls={
            "positive_sv_rel_error": safe_float(sv_rel_error),
            "positive_control_pass": positive_control_pass,
            "negative_subspace_alignment": max_alignment_noise,
            "negative_control_pass": negative_control_pass
        },
        notes="Sparse coding stability test"
    )


def test_multiscale_connectivity(embeddings_dict: Dict[str, np.ndarray], config: TestConfig) -> TestResult:
    """DIAGNOSTIC F: Multiscale Connectivity Analysis"""
    results = {}

    for name, X in embeddings_dict.items():
        D = pairwise_distances(X, config.distance_metric)
        upper_tri = D[np.triu_indices(len(D), k=1)]
        scales = np.percentile(upper_tri, [10, 25, 50, 75, 90])
        components_at_scale = []

        for eps in scales:
            adj = (D < eps).astype(int)
            np.fill_diagonal(adj, 0)
            n_comp, _ = connected_components(adj, directed=False)
            components_at_scale.append(int(n_comp))

        results[name] = {
            "scales": [safe_float(s) for s in scales],
            "components": components_at_scale
        }

    model_names = list(results.keys())
    component_l2_distances = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            c1 = np.array(results[m1]["components"], dtype=float)
            c2 = np.array(results[m2]["components"], dtype=float)
            l2_dist = normalized_l2_distance(c1, c2)
            component_l2_distances.append((m1, m2, safe_float(l2_dist)))

    # Controls
    first_model = model_names[0]
    X_first = embeddings_dict[first_model]
    controls_data = generate_controls(X_first, config.seed)

    X_rot = controls_data["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)

    orig_components = results[first_model]["components"]
    scales = results[first_model]["scales"]
    rot_components = []
    for eps in scales:
        adj = (D_rot < eps).astype(int)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(adj, directed=False)
        rot_components.append(int(n_comp))

    pos_dist = normalized_l2_distance(np.array(orig_components), np.array(rot_components))
    positive_control_pass = bool(pos_dist < 1e-6)

    X_noise = controls_data["noise_corrupted"]
    D_noise = pairwise_distances(X_noise, config.distance_metric)
    noise_components = []
    for eps in scales:
        adj = (D_noise < eps).astype(int)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(adj, directed=False)
        noise_components.append(int(n_comp))

    neg_dist = normalized_l2_distance(np.array(orig_components), np.array(noise_components))
    negative_control_pass = bool(neg_dist > 0.1)

    passed = bool(positive_control_pass and negative_control_pass)

    return TestResult(
        name="multiscale_connectivity",
        test_type="diagnostic",
        passed=passed,
        metrics={
            "per_model": results,
            "component_l2_distances": component_l2_distances
        },
        thresholds={
            "positive_control_threshold": 1e-6,
            "negative_control_threshold": 0.1
        },
        controls={
            "positive_rotated_l2_distance": safe_float(pos_dist),
            "positive_control_pass": positive_control_pass,
            "negative_noise_l2_distance": safe_float(neg_dist),
            "negative_control_pass": negative_control_pass
        },
        notes="Multiscale connectivity analysis"
    )


def run_diagnostic_tests(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> List[TestResult]:
    """Run all diagnostic tests."""
    results = []

    tests = [
        ("Spectral Signature", test_spectral_signature),
        ("Heat Trace Fingerprint", test_heat_trace_fingerprint),
        ("Distance Correlation", test_distance_correlation),
        ("Covariance Spectrum", test_covariance_spectrum),
        ("Sparse Coding Stability", test_sparse_coding_stability),
        ("Multiscale Connectivity", test_multiscale_connectivity),
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC TESTS (Cross-Model Comparisons)")
        print("=" * 60)

    for test_name, test_fn in tests:
        if verbose:
            print(f"\n  Running: {test_name}...")

        result = test_fn(embeddings_dict, config)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"    {test_name}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Q41 Diagnostic Tests")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent.parent / "receipts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print(f"Q41 DIAGNOSTIC TESTS v{__version__}")
        print("=" * 60)

    config = TestConfig(seed=args.seed)
    corpus = DEFAULT_CORPUS

    if verbose:
        print(f"\nLoading embeddings...")
    embeddings = load_embeddings(corpus, verbose=verbose)

    if len(embeddings) < 2:
        print("ERROR: Need at least 2 embedding models")
        sys.exit(1)

    # Preprocess
    embeddings_proc = {name: preprocess_embeddings(X, config.preprocessing) for name, X in embeddings.items()}

    results = run_diagnostic_tests(embeddings_proc, config, verbose=verbose)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Diagnostic Tests: {passed}/{total} passed")

    # Save receipt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = out_dir / f"q41_diagnostic_receipt_{timestamp_str}.json"

    receipt = {
        "suite": __suite__,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": total,
        "all_pass": passed == total,
        "models": list(embeddings.keys()),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "metrics": to_builtin(r.metrics),
                "controls": to_builtin(r.controls),
                "notes": r.notes
            }
            for r in results
        ]
    }

    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    if verbose:
        print(f"Receipt saved: {receipt_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
