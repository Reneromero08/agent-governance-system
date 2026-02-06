#!/usr/bin/env python3
"""
Q18 Investigation: IMPROVED 8e Novelty Detection via Multi-Metric Approach

PROBLEM: Original 8e detection has limitations:
- Noise injection: 40% (at 30%+ noise)
- Pattern redundancy: 60% (at 20%+ duplication)
- Value corruption: 0% (invisible - covariance invariant to sign)
- Semantic shuffle: 0% (invisible - distribution unchanged by relabeling)

HYPOTHESIS: Combining 8e with complementary metrics can achieve >50% detection
for ALL perturbation types.

APPROACH:
1. 8e (Df x alpha) - Detects distributional changes (noise, redundancy)
2. R-Embedding Correlation - Detects semantic mapping issues (shuffle)
3. Higher-order statistics (kurtosis, skewness) - Detects value corruption
4. Sign consistency - Directly targets sign-flip detection
5. Df and alpha separately - Provides directional diagnostic information

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 2.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
RANDOM_BASELINE = 14.5
THRESHOLD_15_PERCENT = 0.15


@dataclass
class MultiMetricResult:
    """Results from multi-metric novelty detection."""
    name: str
    description: str
    perturbation_type: str
    perturbation_strength: float
    n_samples: int
    n_dims: int

    # 8e metrics (original)
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float

    # R-Embedding correlation metrics
    r_embedding_correlation: float
    r_prediction_error: float

    # Higher-order statistics
    mean_kurtosis: float
    mean_skewness: float
    kurtosis_deviation: float
    skewness_deviation: float

    # Sign consistency metric
    sign_consistency: float
    sign_flip_score: float

    # Magnitude distribution
    magnitude_mean: float
    magnitude_std: float
    magnitude_cv: float

    # Individual metric detections
    detected_by_8e: bool
    detected_by_r_correlation: bool
    detected_by_kurtosis: bool
    detected_by_sign: bool
    detected_by_magnitude: bool

    # Combined detection
    n_metrics_triggered: int
    is_detected: bool
    detection_confidence: float

    # Classification
    classification: str
    anomaly_type: str


def to_builtin(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
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


def load_gene_expression_data(filepath: str) -> Dict[str, Any]:
    """Load gene expression data from JSON cache."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# METRIC 1: 8e (Df x alpha) - Original spectral metric
# =============================================================================

def compute_spectral_8e(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute Df (participation ratio) and alpha (spectral decay).

    Returns: (Df, alpha, eigenvalues)
    """
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


# =============================================================================
# METRIC 2: R-Embedding Correlation - Detects semantic mapping issues
# =============================================================================

def compute_r_embedding_correlation(R_values: np.ndarray, embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Compute correlation between R values and embedding properties.

    This detects semantic shuffle: if R-embedding correspondence is broken,
    the predicted R from embeddings will not match actual R.

    Returns: (correlation, prediction_error)
    """
    # Compute embedding magnitudes (norm of each embedding)
    magnitudes = np.linalg.norm(embeddings, axis=1)

    # Compute first principal component score
    centered = embeddings - embeddings.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1_scores = centered @ Vt[0]

    # Combine features for R prediction
    # The baseline embedding creates inverse relationship: high R -> low magnitude
    features = np.column_stack([magnitudes, np.abs(pc1_scores)])

    # Simple linear prediction of R from features
    X = np.column_stack([np.ones(len(R_values)), features])
    try:
        # Least squares fit
        beta = np.linalg.lstsq(X, R_values, rcond=None)[0]
        R_predicted = X @ beta

        # Correlation between predicted and actual R
        correlation = np.corrcoef(R_values, R_predicted)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Prediction error (normalized RMSE)
        rmse = np.sqrt(np.mean((R_predicted - R_values) ** 2))
        prediction_error = rmse / (np.std(R_values) + 1e-10)

    except np.linalg.LinAlgError:
        correlation = 0.0
        prediction_error = 1.0

    return correlation, prediction_error


# =============================================================================
# METRIC 3: Higher-Order Statistics - Detects value corruption
# =============================================================================

def compute_higher_order_stats(embeddings: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute kurtosis and skewness across embedding dimensions.

    Sign flips and value inversions change the distribution shape,
    which is captured by these higher-order moments.

    Returns: (mean_kurtosis, mean_skewness, kurtosis_deviation, skewness_deviation)
    """
    # Compute kurtosis and skewness for each dimension
    kurtosis_vals = []
    skewness_vals = []

    for dim in range(embeddings.shape[1]):
        k = stats.kurtosis(embeddings[:, dim])
        s = stats.skew(embeddings[:, dim])
        if not np.isnan(k):
            kurtosis_vals.append(k)
        if not np.isnan(s):
            skewness_vals.append(s)

    mean_kurtosis = np.mean(kurtosis_vals) if kurtosis_vals else 0.0
    mean_skewness = np.mean(skewness_vals) if skewness_vals else 0.0

    # For normal distribution, kurtosis should be ~0 (excess kurtosis)
    # and skewness should be ~0
    kurtosis_deviation = np.std(kurtosis_vals) if kurtosis_vals else 0.0
    skewness_deviation = np.std(skewness_vals) if skewness_vals else 0.0

    return mean_kurtosis, mean_skewness, kurtosis_deviation, skewness_deviation


# =============================================================================
# METRIC 4: Sign Consistency - Directly targets sign-flip detection
# =============================================================================

def compute_sign_consistency(R_values: np.ndarray, embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Compute sign consistency between R-ordered samples.

    Insight: In the baseline embedding, adjacent R values should have
    similar embedding directions. Sign flips break this consistency.

    Returns: (sign_consistency, sign_flip_score)
    """
    # Sort by R values
    sorted_idx = np.argsort(R_values)
    sorted_embeddings = embeddings[sorted_idx]

    # Compute sign agreement between consecutive samples
    sign_agreements = []
    direction_changes = []

    for i in range(len(sorted_embeddings) - 1):
        e1 = sorted_embeddings[i]
        e2 = sorted_embeddings[i + 1]

        # Sign agreement: fraction of dimensions with same sign
        same_sign = np.sum(np.sign(e1) == np.sign(e2)) / len(e1)
        sign_agreements.append(same_sign)

        # Direction change: cosine similarity
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            cos_sim = np.dot(e1, e2) / (norm1 * norm2)
            direction_changes.append(cos_sim)

    sign_consistency = np.mean(sign_agreements) if sign_agreements else 0.5

    # Sign flip score: detect abrupt sign changes
    if direction_changes:
        # High variance in direction changes indicates sign flips
        direction_std = np.std(direction_changes)
        # Count negative cosine similarities (opposing directions)
        n_negative = np.sum(np.array(direction_changes) < 0)
        sign_flip_score = n_negative / len(direction_changes) + direction_std
    else:
        sign_flip_score = 0.0

    return sign_consistency, sign_flip_score


# =============================================================================
# METRIC 5: Magnitude Distribution - Detects structural changes
# =============================================================================

def compute_magnitude_stats(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute statistics about embedding magnitudes.

    Returns: (mean, std, coefficient_of_variation)
    """
    magnitudes = np.linalg.norm(embeddings, axis=1)

    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    cv_mag = std_mag / (mean_mag + 1e-10)

    return mean_mag, std_mag, cv_mag


# =============================================================================
# BASELINE EMBEDDING CREATION
# =============================================================================

def create_baseline_embedding(R_values: np.ndarray, n_dims: int = 50, seed: int = 42) -> np.ndarray:
    """Create the baseline sinusoidal R embedding that produces ~8e."""
    np.random.seed(seed)
    n_genes = len(R_values)
    embeddings = np.zeros((n_genes, n_dims))

    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    return embeddings


# =============================================================================
# PERTURBATIONS (Same as original)
# =============================================================================

def perturb_noise_injection(R_values: np.ndarray, embeddings: np.ndarray,
                            noise_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Add random noise samples."""
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    n_noise = int(n_genes * noise_fraction)
    noise_indices = np.random.choice(n_genes, n_noise, replace=False)

    for idx in noise_indices:
        perturbed_R[idx] = np.random.uniform(R_values.min(), R_values.max())
        perturbed_emb[idx] = np.random.randn(n_dims)

    return perturbed_R, perturbed_emb


def perturb_value_corruption(R_values: np.ndarray, embeddings: np.ndarray,
                             corruption_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Invert values (sign flips)."""
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    n_corrupt = int(n_genes * corruption_fraction)
    corrupt_indices = np.random.choice(n_genes, n_corrupt, replace=False)

    for idx in corrupt_indices:
        perturbed_R[idx] = 1.0 / (R_values[idx] + 0.1)
        perturbed_emb[idx] = -perturbed_emb[idx]
        flip_mask = np.random.random(n_dims) < 0.3
        perturbed_emb[idx, flip_mask] = -perturbed_emb[idx, flip_mask]

    return perturbed_R, perturbed_emb


def perturb_redundancy(R_values: np.ndarray, embeddings: np.ndarray,
                       redundancy_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate high-R patterns."""
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    n_duplicate = int(n_genes * redundancy_fraction)
    sorted_indices = np.argsort(R_values)[::-1]
    source_indices = sorted_indices[:max(1, n_duplicate // 5)]
    target_indices = np.random.choice(n_genes, n_duplicate, replace=False)

    for i, target_idx in enumerate(target_indices):
        source_idx = source_indices[i % len(source_indices)]
        perturbed_R[target_idx] = R_values[source_idx] * (1 + np.random.randn() * 0.01)
        perturbed_emb[target_idx] = embeddings[source_idx] + np.random.randn(n_dims) * 0.01

    return perturbed_R, perturbed_emb


def perturb_semantic_shuffle(R_values: np.ndarray, embeddings: np.ndarray,
                             shuffle_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle R-embedding correspondence."""
    np.random.seed(seed)
    n_genes = len(R_values)
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    n_shuffle = int(n_genes * shuffle_fraction)
    shuffle_indices = np.random.choice(n_genes, n_shuffle, replace=False)

    shuffled_Rs = perturbed_R[shuffle_indices].copy()
    np.random.shuffle(shuffled_Rs)
    perturbed_R[shuffle_indices] = shuffled_Rs

    shuffled_embs = perturbed_emb[shuffle_indices].copy()
    np.random.shuffle(shuffled_embs)
    perturbed_emb[shuffle_indices] = shuffled_embs

    return perturbed_R, perturbed_emb


# =============================================================================
# BASELINE REFERENCE VALUES (computed from unperturbed data)
# =============================================================================

@dataclass
class BaselineReference:
    """Reference values from baseline (unperturbed) data."""
    Df: float
    alpha: float
    Df_x_alpha: float
    r_correlation: float
    r_prediction_error: float
    mean_kurtosis: float
    mean_skewness: float
    kurtosis_std: float
    skewness_std: float
    sign_consistency: float
    sign_flip_score: float
    magnitude_mean: float
    magnitude_std: float
    magnitude_cv: float


def compute_baseline_reference(R_values: np.ndarray, embeddings: np.ndarray) -> BaselineReference:
    """Compute reference values from baseline data for anomaly detection."""
    Df, alpha, _ = compute_spectral_8e(embeddings)
    r_corr, r_error = compute_r_embedding_correlation(R_values, embeddings)
    mean_kurt, mean_skew, kurt_std, skew_std = compute_higher_order_stats(embeddings)
    sign_cons, sign_flip = compute_sign_consistency(R_values, embeddings)
    mag_mean, mag_std, mag_cv = compute_magnitude_stats(embeddings)

    return BaselineReference(
        Df=Df,
        alpha=alpha,
        Df_x_alpha=Df * alpha,
        r_correlation=r_corr,
        r_prediction_error=r_error,
        mean_kurtosis=mean_kurt,
        mean_skewness=mean_skew,
        kurtosis_std=kurt_std,
        skewness_std=skew_std,
        sign_consistency=sign_cons,
        sign_flip_score=sign_flip,
        magnitude_mean=mag_mean,
        magnitude_std=mag_std,
        magnitude_cv=mag_cv
    )


# =============================================================================
# MULTI-METRIC DETECTION
# =============================================================================

def run_multi_metric_test(
    name: str,
    description: str,
    perturbation_type: str,
    R_values: np.ndarray,
    embeddings: np.ndarray,
    strength: float,
    baseline: BaselineReference,
    thresholds: Dict[str, float]
) -> MultiMetricResult:
    """
    Run multi-metric novelty detection test.

    Thresholds:
    - 8e_threshold: deviation from 8e (default 0.15)
    - r_corr_threshold: minimum R-embedding correlation (default 0.6)
    - kurtosis_threshold: deviation in kurtosis (default 0.3)
    - sign_threshold: deviation in sign consistency (default 0.15)
    - magnitude_threshold: deviation in magnitude CV (default 0.3)
    """
    # Compute all metrics
    Df, alpha, _ = compute_spectral_8e(embeddings)
    Df_x_alpha = Df * alpha
    deviation_8e = abs(Df_x_alpha - EIGHT_E) / EIGHT_E

    r_corr, r_error = compute_r_embedding_correlation(R_values, embeddings)
    mean_kurt, mean_skew, kurt_std, skew_std = compute_higher_order_stats(embeddings)
    sign_cons, sign_flip = compute_sign_consistency(R_values, embeddings)
    mag_mean, mag_std, mag_cv = compute_magnitude_stats(embeddings)

    # Compute deviations from baseline
    kurtosis_deviation = abs(mean_kurt - baseline.mean_kurtosis) / (abs(baseline.mean_kurtosis) + 0.1)
    skewness_deviation = abs(mean_skew - baseline.mean_skewness) / (abs(baseline.mean_skewness) + 0.1)

    # Individual metric detections
    detected_by_8e = deviation_8e >= thresholds.get('8e_threshold', 0.15)

    # R-correlation: relative drop from baseline indicates semantic issues
    # Use relative change only, not absolute threshold, to avoid false positives on baseline
    r_corr_drop = baseline.r_correlation - r_corr
    r_corr_relative_drop = r_corr_drop / (baseline.r_correlation + 1e-10)
    detected_by_r_correlation = r_corr_relative_drop > thresholds.get('r_corr_threshold', 0.10)

    # Kurtosis: significant change indicates distributional shift (relative to baseline)
    kurtosis_abs_change = abs(mean_kurt - baseline.mean_kurtosis)
    detected_by_kurtosis = kurtosis_abs_change > thresholds.get('kurtosis_threshold', 0.5) or \
                           skewness_deviation > thresholds.get('skewness_threshold', 0.5)

    # Sign consistency: drop from baseline indicates sign flips
    sign_drop = baseline.sign_consistency - sign_cons
    sign_flip_increase = sign_flip - baseline.sign_flip_score
    detected_by_sign = sign_drop > thresholds.get('sign_threshold', 0.02) or \
                       sign_flip_increase > thresholds.get('sign_flip_threshold', 0.05)

    # Magnitude: significant CV change indicates structural perturbation
    mag_cv_change = abs(mag_cv - baseline.magnitude_cv) / (baseline.magnitude_cv + 1e-10)
    detected_by_magnitude = mag_cv_change > thresholds.get('magnitude_threshold', 0.5)

    # Combined detection
    detections = [
        detected_by_8e,
        detected_by_r_correlation,
        detected_by_kurtosis,
        detected_by_sign,
        detected_by_magnitude
    ]
    n_triggered = sum(detections)

    # Detection rule: any 1+ metric triggered counts as detection
    # For high confidence, use 2+ metrics
    is_detected = n_triggered >= 1
    detection_confidence = n_triggered / len(detections)

    # Classification based on which metrics triggered
    if not is_detected:
        classification = 'normal'
        anomaly_type = 'none'
    else:
        classification = 'anomaly'
        # Determine likely anomaly type based on pattern
        if detected_by_8e and not detected_by_r_correlation:
            if Df_x_alpha < EIGHT_E:
                anomaly_type = 'noise_dilution'
            else:
                anomaly_type = 'redundancy'
        elif detected_by_r_correlation and not detected_by_8e:
            anomaly_type = 'semantic_shuffle'
        elif detected_by_sign or detected_by_kurtosis:
            if not detected_by_8e:
                anomaly_type = 'value_corruption'
            else:
                anomaly_type = 'mixed_corruption'
        else:
            anomaly_type = 'unclassified'

    return MultiMetricResult(
        name=name,
        description=description,
        perturbation_type=perturbation_type,
        perturbation_strength=strength,
        n_samples=len(R_values),
        n_dims=embeddings.shape[1],
        Df=Df,
        alpha=alpha,
        Df_x_alpha=Df_x_alpha,
        deviation_from_8e=deviation_8e,
        r_embedding_correlation=r_corr,
        r_prediction_error=r_error,
        mean_kurtosis=mean_kurt,
        mean_skewness=mean_skew,
        kurtosis_deviation=kurtosis_deviation,
        skewness_deviation=skewness_deviation,
        sign_consistency=sign_cons,
        sign_flip_score=sign_flip,
        magnitude_mean=mag_mean,
        magnitude_std=mag_std,
        magnitude_cv=mag_cv,
        detected_by_8e=detected_by_8e,
        detected_by_r_correlation=detected_by_r_correlation,
        detected_by_kurtosis=detected_by_kurtosis,
        detected_by_sign=detected_by_sign,
        detected_by_magnitude=detected_by_magnitude,
        n_metrics_triggered=n_triggered,
        is_detected=is_detected,
        detection_confidence=detection_confidence,
        classification=classification,
        anomaly_type=anomaly_type
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_improved_novelty_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run comprehensive multi-metric novelty detection tests."""

    # Load data
    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: IMPROVED 8e NOVELTY DETECTION (MULTI-METRIC)")
        print("=" * 80)
        print("\nGoal: Achieve >50% detection rate for ALL perturbation types")
        print("\nMetrics used:")
        print("  1. 8e (Df x alpha) - distributional changes")
        print("  2. R-Embedding Correlation - semantic mapping issues")
        print("  3. Higher-Order Stats (kurtosis) - value corruption")
        print("  4. Sign Consistency - sign-flip detection")
        print("  5. Magnitude Distribution - structural changes")
        print(f"\nLoading data from: {cache_path}")

    data = load_gene_expression_data(str(cache_path))
    genes_data = data['genes']
    R_values = np.array([g['R'] for g in genes_data.values()])
    n_genes = len(R_values)

    if verbose:
        print(f"Loaded {n_genes} genes")
        print(f"R range: {R_values.min():.2f} - {R_values.max():.2f}")

    # Create baseline embedding
    baseline_embeddings = create_baseline_embedding(R_values, n_dims=50, seed=42)

    # Compute baseline reference values
    baseline_ref = compute_baseline_reference(R_values, baseline_embeddings)

    if verbose:
        print(f"\nBaseline reference values:")
        print(f"  Df x alpha: {baseline_ref.Df_x_alpha:.2f}")
        print(f"  R-Embedding correlation: {baseline_ref.r_correlation:.3f}")
        print(f"  Mean kurtosis: {baseline_ref.mean_kurtosis:.3f}")
        print(f"  Sign consistency: {baseline_ref.sign_consistency:.3f}")
        print(f"  Magnitude CV: {baseline_ref.magnitude_cv:.3f}")
        print("\n" + "=" * 80)

    # Detection thresholds (tuned for sensitivity)
    thresholds = {
        '8e_threshold': 0.15,
        'r_corr_threshold': 0.10,  # More sensitive to R-correlation drops
        'kurtosis_threshold': 0.25,
        'skewness_threshold': 0.25,
        'sign_threshold': 0.03,   # Very sensitive to sign changes
        'sign_flip_threshold': 0.05,
        'magnitude_threshold': 0.15
    }

    results = []

    # =========================================================================
    # BASELINE TEST
    # =========================================================================
    if verbose:
        print("\n[BASELINE] Testing normal R-structured embedding...")

    baseline_result = run_multi_metric_test(
        name="baseline_normal",
        description="Normal R-structured embedding",
        perturbation_type="none",
        R_values=R_values,
        embeddings=baseline_embeddings,
        strength=0.0,
        baseline=baseline_ref,
        thresholds=thresholds
    )
    results.append(baseline_result)

    if verbose:
        print(f"    Df x alpha: {baseline_result.Df_x_alpha:.2f}")
        print(f"    R-corr: {baseline_result.r_embedding_correlation:.3f}")
        print(f"    Metrics triggered: {baseline_result.n_metrics_triggered}/5")
        print(f"    Classification: {baseline_result.classification}")

    # =========================================================================
    # PERTURBATION A: NOISE INJECTION
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION A] Noise injection...")

    for noise_frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        perturbed_R, perturbed_emb = perturb_noise_injection(
            R_values, baseline_embeddings, noise_frac, seed=42
        )

        result = run_multi_metric_test(
            name=f"noise_{int(noise_frac*100)}pct",
            description=f"Noise injection: {int(noise_frac*100)}%",
            perturbation_type="noise_injection",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=noise_frac,
            baseline=baseline_ref,
            thresholds=thresholds
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected else "MISSED"
            triggered = f"[8e:{int(result.detected_by_8e)} R:{int(result.detected_by_r_correlation)} K:{int(result.detected_by_kurtosis)} S:{int(result.detected_by_sign)} M:{int(result.detected_by_magnitude)}]"
            print(f"    {int(noise_frac*100)}%: {triggered} -> {detected}")

    # =========================================================================
    # PERTURBATION B: VALUE CORRUPTION
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION B] Value corruption (sign flips)...")

    for corrupt_frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        perturbed_R, perturbed_emb = perturb_value_corruption(
            R_values, baseline_embeddings, corrupt_frac, seed=42
        )

        result = run_multi_metric_test(
            name=f"corruption_{int(corrupt_frac*100)}pct",
            description=f"Value corruption: {int(corrupt_frac*100)}%",
            perturbation_type="corruption",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=corrupt_frac,
            baseline=baseline_ref,
            thresholds=thresholds
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected else "MISSED"
            triggered = f"[8e:{int(result.detected_by_8e)} R:{int(result.detected_by_r_correlation)} K:{int(result.detected_by_kurtosis)} S:{int(result.detected_by_sign)} M:{int(result.detected_by_magnitude)}]"
            print(f"    {int(corrupt_frac*100)}%: {triggered} -> {detected}")

    # =========================================================================
    # PERTURBATION C: REDUNDANCY
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION C] Pattern redundancy...")

    for redund_frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        perturbed_R, perturbed_emb = perturb_redundancy(
            R_values, baseline_embeddings, redund_frac, seed=42
        )

        result = run_multi_metric_test(
            name=f"redundancy_{int(redund_frac*100)}pct",
            description=f"Redundancy: {int(redund_frac*100)}%",
            perturbation_type="redundancy",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=redund_frac,
            baseline=baseline_ref,
            thresholds=thresholds
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected else "MISSED"
            triggered = f"[8e:{int(result.detected_by_8e)} R:{int(result.detected_by_r_correlation)} K:{int(result.detected_by_kurtosis)} S:{int(result.detected_by_sign)} M:{int(result.detected_by_magnitude)}]"
            print(f"    {int(redund_frac*100)}%: {triggered} -> {detected}")

    # =========================================================================
    # PERTURBATION D: SEMANTIC SHUFFLE
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION D] Semantic shuffle...")

    for shuffle_frac in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        perturbed_R, perturbed_emb = perturb_semantic_shuffle(
            R_values, baseline_embeddings, shuffle_frac, seed=42
        )

        result = run_multi_metric_test(
            name=f"shuffle_{int(shuffle_frac*100)}pct",
            description=f"Semantic shuffle: {int(shuffle_frac*100)}%",
            perturbation_type="semantic_shuffle",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=shuffle_frac,
            baseline=baseline_ref,
            thresholds=thresholds
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected else "MISSED"
            triggered = f"[8e:{int(result.detected_by_8e)} R:{int(result.detected_by_r_correlation)} K:{int(result.detected_by_kurtosis)} S:{int(result.detected_by_sign)} M:{int(result.detected_by_magnitude)}]"
            print(f"    {int(shuffle_frac*100)}%: {triggered} -> {detected}")

    # =========================================================================
    # RANDOM CONTROL
    # =========================================================================
    if verbose:
        print("\n[CONTROL] Pure random embedding...")

    np.random.seed(42)
    random_embeddings = np.random.randn(n_genes, 50)

    random_result = run_multi_metric_test(
        name="pure_random",
        description="Pure random embeddings",
        perturbation_type="random_control",
        R_values=R_values,
        embeddings=random_embeddings,
        strength=1.0,
        baseline=baseline_ref,
        thresholds=thresholds
    )
    results.append(random_result)

    if verbose:
        detected = "DETECTED" if random_result.is_detected else "MISSED"
        print(f"    Random: {random_result.n_metrics_triggered}/5 triggered -> {detected}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: IMPROVED DETECTION PERFORMANCE")
        print("=" * 80)

    # Calculate detection rates by type
    type_results = {}
    for r in results:
        if r.perturbation_type not in type_results:
            type_results[r.perturbation_type] = []
        type_results[r.perturbation_type].append(r)

    type_detection_rates = {}
    for ptype, presults in type_results.items():
        if ptype in ['none', 'random_control']:
            continue
        detected = sum(1 for r in presults if r.is_detected)
        type_detection_rates[ptype] = detected / len(presults)

    # Overall detection rate (excluding baseline and random control)
    perturbation_results = [r for r in results if r.perturbation_type not in ['none', 'random_control']]
    overall_detected = sum(1 for r in perturbation_results if r.is_detected)
    overall_rate = overall_detected / len(perturbation_results)

    # Check if baseline is correctly NOT detected
    baseline_correct = not baseline_result.is_detected

    # Check if random is correctly detected
    random_correct = random_result.is_detected

    if verbose:
        print(f"\n  Baseline correctly classified: {'YES' if baseline_correct else 'NO'}")
        print(f"  Random control detected: {'YES' if random_correct else 'NO'}")
        print(f"\n  Detection rates by perturbation type:")
        for ptype, rate in type_detection_rates.items():
            status = "PASS" if rate >= 0.5 else "FAIL"
            print(f"    - {ptype}: {rate*100:.0f}% [{status}]")
        print(f"\n  Overall detection rate: {overall_rate*100:.1f}%")

        # Goal check
        all_above_50 = all(rate >= 0.5 for rate in type_detection_rates.values())
        print(f"\n  GOAL (>50% for all types): {'ACHIEVED' if all_above_50 else 'NOT YET ACHIEVED'}")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Multi-metric approach achieves >50% detection for all perturbation types",
        "thresholds": thresholds,
        "baseline_reference": {
            "Df_x_alpha": baseline_ref.Df_x_alpha,
            "r_correlation": baseline_ref.r_correlation,
            "mean_kurtosis": baseline_ref.mean_kurtosis,
            "sign_consistency": baseline_ref.sign_consistency,
            "magnitude_cv": baseline_ref.magnitude_cv
        },
        "results": [],
        "summary": {
            "baseline_correct": baseline_correct,
            "random_control_correct": random_correct,
            "type_detection_rates": {k: float(v) for k, v in type_detection_rates.items()},
            "overall_detection_rate": overall_rate,
            "goal_achieved": all_above_50 if 'all_above_50' in dir() else all(rate >= 0.5 for rate in type_detection_rates.values())
        },
        "comparison_with_original": {
            "original_noise_rate": 0.40,
            "original_corruption_rate": 0.0,
            "original_redundancy_rate": 0.60,
            "original_shuffle_rate": 0.0,
            "improved_noise_rate": type_detection_rates.get('noise_injection', 0),
            "improved_corruption_rate": type_detection_rates.get('corruption', 0),
            "improved_redundancy_rate": type_detection_rates.get('redundancy', 0),
            "improved_shuffle_rate": type_detection_rates.get('semantic_shuffle', 0)
        }
    }

    # Add individual results
    for r in results:
        output["results"].append({
            "name": r.name,
            "perturbation_type": r.perturbation_type,
            "perturbation_strength": r.perturbation_strength,
            "Df_x_alpha": r.Df_x_alpha,
            "r_embedding_correlation": r.r_embedding_correlation,
            "mean_kurtosis": r.mean_kurtosis,
            "sign_consistency": r.sign_consistency,
            "magnitude_cv": r.magnitude_cv,
            "detected_by_8e": r.detected_by_8e,
            "detected_by_r_correlation": r.detected_by_r_correlation,
            "detected_by_kurtosis": r.detected_by_kurtosis,
            "detected_by_sign": r.detected_by_sign,
            "detected_by_magnitude": r.detected_by_magnitude,
            "n_metrics_triggered": r.n_metrics_triggered,
            "is_detected": r.is_detected,
            "anomaly_type": r.anomaly_type
        })

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_improved_novelty_tests(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "8e_novelty_improved_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()
