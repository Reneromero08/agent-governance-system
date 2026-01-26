#!/usr/bin/env python3
"""
Q18 Investigation: 8e Deviation as Novelty Detection Signal

HYPOTHESIS: 8e deviation can reliably detect out-of-distribution or corrupted data.

From theory review:
- Normal data (trained embeddings): Df x alpha ~ 8e (21.75)
- Novel/anomalous data: Df x alpha DEVIATES from 8e
- Random/noise data: Df x alpha ~ 14.5 (random baseline)

TEST PROTOCOL:
1. Create "normal" R-structured embedding (baseline - should show ~8e)
2. Create anomalous embeddings via:
   a) Noise injection - adding random genes
   b) Corruption - inverting gene values
   c) Redundancy - duplicating patterns
   d) Semantic destruction - shuffling labels
3. Measure Df x alpha for each perturbation
4. Evaluate if anomalies show reliable deviation from 8e

PRACTICAL APPLICATIONS:
- Detecting dataset shift in ML models
- Quality control for biological datasets
- Identifying novel biological patterns vs noise

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
RANDOM_BASELINE = 14.5  # Expected for unstructured data
THRESHOLD_15_PERCENT = 0.15  # Standard threshold for 8e detection


@dataclass
class NoveltyResult:
    """Results from a single novelty detection test."""
    name: str
    description: str
    perturbation_type: str
    perturbation_strength: float
    n_samples: int
    n_dims: int
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    deviation_percent: float
    passes_8e: bool  # Within 15% of 8e
    is_detected_as_anomaly: bool  # Deviation > threshold
    classification: str  # 'normal', 'novel', 'chaotic', 'compressed'


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


def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from embedding matrix.

    Args:
        embeddings: (n_samples, n_features) array

    Returns:
        (Df, alpha, eigenvalues)
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio - measures effective dimensionality
    # PR = (sum(lambda))^2 / sum(lambda^2)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent
    # Fit power law: lambda_k ~ k^(-alpha)
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    # Linear regression for slope
    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope  # Negative slope = positive alpha

    return Df, alpha, eigenvalues


def classify_8e_deviation(Df_x_alpha: float, threshold: float = THRESHOLD_15_PERCENT) -> str:
    """
    Classify the type of deviation based on Df x alpha value.

    From theory:
    - Normal semiotic: ~8e (21.75)
    - Random baseline: ~14.5 (no structure)
    - Compressed: below 8e but above random
    - Expanded: above 8e
    - Chaotic: near or below random baseline
    """
    deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E

    if deviation < threshold:
        return 'normal'
    elif Df_x_alpha < RANDOM_BASELINE * 1.1:  # Near or below random baseline
        return 'chaotic'
    elif Df_x_alpha < EIGHT_E:
        return 'compressed'
    else:
        return 'expanded'


# =============================================================================
# BASELINE EMBEDDING (Should show ~8e)
# =============================================================================

def create_baseline_embedding(R_values: np.ndarray, n_dims: int = 50, seed: int = 42) -> np.ndarray:
    """
    Create the baseline sinusoidal R embedding that produces ~8e.

    This is the "normal" reference - embeddings that properly encode the R structure.
    """
    np.random.seed(seed)
    n_genes = len(R_values)
    embeddings = np.zeros((n_genes, n_dims))

    for i, r in enumerate(R_values):
        np.random.seed(i + seed)

        # Scale factor based on R (inverse relationship)
        scale = 1.0 / (r + 0.1)

        # Random direction modulated by R
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Position: base + R-modulated spread
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    return embeddings


# =============================================================================
# PERTURBATION A: NOISE INJECTION
# =============================================================================

def perturb_noise_injection(R_values: np.ndarray, embeddings: np.ndarray,
                            noise_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturbation A: Add random genes (noise injection).

    This simulates:
    - Batch effects in biological data
    - Sensor noise in measurements
    - Contaminated samples

    Args:
        R_values: Original R values
        embeddings: Original embeddings
        noise_fraction: Fraction of samples to replace with random noise (0.0 to 1.0)
        seed: Random seed

    Returns:
        (perturbed_R, perturbed_embeddings)
    """
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape

    # Create copies
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    # Number of samples to replace with noise
    n_noise = int(n_genes * noise_fraction)
    noise_indices = np.random.choice(n_genes, n_noise, replace=False)

    # Replace selected samples with pure random noise
    for idx in noise_indices:
        # Random R value (no biological meaning)
        perturbed_R[idx] = np.random.uniform(R_values.min(), R_values.max())
        # Random embedding (no structure)
        perturbed_emb[idx] = np.random.randn(n_dims)

    return perturbed_R, perturbed_emb


# =============================================================================
# PERTURBATION B: VALUE CORRUPTION
# =============================================================================

def perturb_value_corruption(R_values: np.ndarray, embeddings: np.ndarray,
                             corruption_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturbation B: Invert some gene values (corruption).

    This simulates:
    - Data entry errors
    - Measurement artifacts
    - Systematic biases

    Args:
        R_values: Original R values
        embeddings: Original embeddings
        corruption_fraction: Fraction of values to corrupt (0.0 to 1.0)
        seed: Random seed

    Returns:
        (perturbed_R, perturbed_embeddings)
    """
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape

    # Create copies
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    # Number of genes to corrupt
    n_corrupt = int(n_genes * corruption_fraction)
    corrupt_indices = np.random.choice(n_genes, n_corrupt, replace=False)

    for idx in corrupt_indices:
        # Invert R value (1/R instead of R)
        perturbed_R[idx] = 1.0 / (R_values[idx] + 0.1)

        # Invert embedding direction
        perturbed_emb[idx] = -perturbed_emb[idx]

        # Add sign-flip noise to some dimensions
        flip_mask = np.random.random(n_dims) < 0.3
        perturbed_emb[idx, flip_mask] = -perturbed_emb[idx, flip_mask]

    return perturbed_R, perturbed_emb


# =============================================================================
# PERTURBATION C: PATTERN REDUNDANCY
# =============================================================================

def perturb_redundancy(R_values: np.ndarray, embeddings: np.ndarray,
                       redundancy_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturbation C: Duplicate patterns (redundancy).

    This simulates:
    - Technical replicates mistaken for biological replicates
    - Copy-paste errors in datasets
    - Collapsed gene families

    Args:
        R_values: Original R values
        embeddings: Original embeddings
        redundancy_fraction: Fraction of samples to duplicate (0.0 to 1.0)
        seed: Random seed

    Returns:
        (perturbed_R, perturbed_embeddings)
    """
    np.random.seed(seed)
    n_genes, n_dims = embeddings.shape

    # Create copies
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    # Number of duplications
    n_duplicate = int(n_genes * redundancy_fraction)

    # Select source samples (top R values - most important genes)
    sorted_indices = np.argsort(R_values)[::-1]
    source_indices = sorted_indices[:max(1, n_duplicate // 5)]  # Take from top quintile

    # Select target samples to replace (random)
    target_indices = np.random.choice(n_genes, n_duplicate, replace=False)

    for i, target_idx in enumerate(target_indices):
        source_idx = source_indices[i % len(source_indices)]

        # Copy R value with tiny perturbation
        perturbed_R[target_idx] = R_values[source_idx] * (1 + np.random.randn() * 0.01)

        # Copy embedding with tiny perturbation
        perturbed_emb[target_idx] = embeddings[source_idx] + np.random.randn(n_dims) * 0.01

    return perturbed_R, perturbed_emb


# =============================================================================
# PERTURBATION D: SEMANTIC DESTRUCTION (Label Shuffling)
# =============================================================================

def perturb_semantic_shuffle(R_values: np.ndarray, embeddings: np.ndarray,
                             shuffle_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturbation D: Shuffle labels (semantics destroyed).

    This simulates:
    - Mislabeled samples
    - File mix-ups
    - Completely wrong annotations

    The key insight: this breaks the R-embedding correspondence.
    R values are shuffled relative to embeddings, destroying the semiotic structure.

    Args:
        R_values: Original R values
        embeddings: Original embeddings
        shuffle_fraction: Fraction of labels to shuffle (0.0 to 1.0)
        seed: Random seed

    Returns:
        (perturbed_R, perturbed_embeddings)
    """
    np.random.seed(seed)
    n_genes = len(R_values)

    # Create copies
    perturbed_emb = embeddings.copy()
    perturbed_R = R_values.copy()

    # Number of labels to shuffle
    n_shuffle = int(n_genes * shuffle_fraction)
    shuffle_indices = np.random.choice(n_genes, n_shuffle, replace=False)

    # Shuffle the R values among selected indices
    # This breaks the R-embedding correspondence
    shuffled_Rs = perturbed_R[shuffle_indices].copy()
    np.random.shuffle(shuffled_Rs)
    perturbed_R[shuffle_indices] = shuffled_Rs

    # Also shuffle embedding positions (both R and embedding are now mismatched)
    shuffled_embs = perturbed_emb[shuffle_indices].copy()
    np.random.shuffle(shuffled_embs)
    perturbed_emb[shuffle_indices] = shuffled_embs

    return perturbed_R, perturbed_emb


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_perturbation_test(name: str, description: str, perturbation_type: str,
                          R_values: np.ndarray, embeddings: np.ndarray,
                          strength: float, detection_threshold: float = THRESHOLD_15_PERCENT) -> NoveltyResult:
    """
    Run a single perturbation test and evaluate 8e deviation.
    """
    Df, alpha, _ = compute_spectral_properties(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E
    passes_8e = deviation < detection_threshold
    is_anomaly = deviation >= detection_threshold
    classification = classify_8e_deviation(product, detection_threshold)

    return NoveltyResult(
        name=name,
        description=description,
        perturbation_type=perturbation_type,
        perturbation_strength=strength,
        n_samples=len(R_values),
        n_dims=embeddings.shape[1],
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        deviation_percent=deviation * 100,
        passes_8e=passes_8e,
        is_detected_as_anomaly=is_anomaly,
        classification=classification
    )


def run_novelty_detection_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive novelty detection tests using 8e deviation.
    """
    # Load data
    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: 8e DEVIATION AS NOVELTY DETECTION SIGNAL")
        print("=" * 80)
        print(f"\nHypothesis: 8e deviation reliably detects anomalous/corrupted data")
        print(f"\nTheoretical targets:")
        print(f"  - Normal (in-distribution): Df x alpha ~ {EIGHT_E:.2f}")
        print(f"  - Random (no structure): Df x alpha ~ {RANDOM_BASELINE:.1f}")
        print(f"  - Anomaly threshold: {THRESHOLD_15_PERCENT*100:.0f}% deviation from 8e")
        print(f"\nLoading data from: {cache_path}")

    data = load_gene_expression_data(str(cache_path))
    genes_data = data['genes']
    R_values = np.array([g['R'] for g in genes_data.values()])
    n_genes = len(R_values)

    if verbose:
        print(f"Loaded {n_genes} genes")
        print(f"R range: {R_values.min():.2f} - {R_values.max():.2f}")
        print("\n" + "=" * 80)

    results = []

    # =========================================================================
    # BASELINE TEST
    # =========================================================================
    if verbose:
        print("\n[BASELINE] Creating normal R-structured embedding...")

    baseline_embeddings = create_baseline_embedding(R_values, n_dims=50, seed=42)
    baseline_result = run_perturbation_test(
        name="baseline_normal",
        description="Normal R-structured embedding (should show 8e)",
        perturbation_type="none",
        R_values=R_values,
        embeddings=baseline_embeddings,
        strength=0.0
    )
    results.append(baseline_result)

    if verbose:
        status = "PASS (normal)" if baseline_result.passes_8e else "UNEXPECTED"
        print(f"       Df x alpha = {baseline_result.Df_x_alpha:.2f}")
        print(f"       Deviation from 8e: {baseline_result.deviation_percent:.1f}%")
        print(f"       Classification: {baseline_result.classification} [{status}]")

    # =========================================================================
    # PERTURBATION A: NOISE INJECTION
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION A] Testing noise injection at various levels...")

    noise_levels = [0.05, 0.10, 0.20, 0.30, 0.50]

    for noise_frac in noise_levels:
        perturbed_R, perturbed_emb = perturb_noise_injection(
            R_values, baseline_embeddings, noise_frac, seed=42
        )

        result = run_perturbation_test(
            name=f"noise_injection_{int(noise_frac*100)}pct",
            description=f"Noise injection: {int(noise_frac*100)}% random samples",
            perturbation_type="noise_injection",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=noise_frac
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected_as_anomaly else "MISSED"
            print(f"    {int(noise_frac*100)}% noise: Df x alpha = {result.Df_x_alpha:.2f} "
                  f"({result.deviation_percent:.1f}% dev) - {result.classification} [{detected}]")

    # =========================================================================
    # PERTURBATION B: VALUE CORRUPTION
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION B] Testing value corruption (inversion)...")

    corruption_levels = [0.05, 0.10, 0.20, 0.30, 0.50]

    for corrupt_frac in corruption_levels:
        perturbed_R, perturbed_emb = perturb_value_corruption(
            R_values, baseline_embeddings, corrupt_frac, seed=42
        )

        result = run_perturbation_test(
            name=f"value_corruption_{int(corrupt_frac*100)}pct",
            description=f"Value corruption: {int(corrupt_frac*100)}% inverted values",
            perturbation_type="corruption",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=corrupt_frac
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected_as_anomaly else "MISSED"
            print(f"    {int(corrupt_frac*100)}% corrupt: Df x alpha = {result.Df_x_alpha:.2f} "
                  f"({result.deviation_percent:.1f}% dev) - {result.classification} [{detected}]")

    # =========================================================================
    # PERTURBATION C: REDUNDANCY
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION C] Testing pattern redundancy (duplication)...")

    redundancy_levels = [0.05, 0.10, 0.20, 0.30, 0.50]

    for redund_frac in redundancy_levels:
        perturbed_R, perturbed_emb = perturb_redundancy(
            R_values, baseline_embeddings, redund_frac, seed=42
        )

        result = run_perturbation_test(
            name=f"redundancy_{int(redund_frac*100)}pct",
            description=f"Redundancy: {int(redund_frac*100)}% duplicated patterns",
            perturbation_type="redundancy",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=redund_frac
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected_as_anomaly else "MISSED"
            print(f"    {int(redund_frac*100)}% redundant: Df x alpha = {result.Df_x_alpha:.2f} "
                  f"({result.deviation_percent:.1f}% dev) - {result.classification} [{detected}]")

    # =========================================================================
    # PERTURBATION D: SEMANTIC SHUFFLE
    # =========================================================================
    if verbose:
        print("\n[PERTURBATION D] Testing semantic destruction (label shuffling)...")

    shuffle_levels = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

    for shuffle_frac in shuffle_levels:
        perturbed_R, perturbed_emb = perturb_semantic_shuffle(
            R_values, baseline_embeddings, shuffle_frac, seed=42
        )

        result = run_perturbation_test(
            name=f"semantic_shuffle_{int(shuffle_frac*100)}pct",
            description=f"Semantic destruction: {int(shuffle_frac*100)}% labels shuffled",
            perturbation_type="semantic_shuffle",
            R_values=perturbed_R,
            embeddings=perturbed_emb,
            strength=shuffle_frac
        )
        results.append(result)

        if verbose:
            detected = "DETECTED" if result.is_detected_as_anomaly else "MISSED"
            print(f"    {int(shuffle_frac*100)}% shuffled: Df x alpha = {result.Df_x_alpha:.2f} "
                  f"({result.deviation_percent:.1f}% dev) - {result.classification} [{detected}]")

    # =========================================================================
    # PURE RANDOM CONTROL
    # =========================================================================
    if verbose:
        print("\n[CONTROL] Testing pure random embedding (should be chaotic)...")

    np.random.seed(42)
    random_embeddings = np.random.randn(n_genes, 50)

    random_result = run_perturbation_test(
        name="pure_random_control",
        description="Pure random embeddings (negative control - no structure)",
        perturbation_type="random_control",
        R_values=R_values,
        embeddings=random_embeddings,
        strength=1.0
    )
    results.append(random_result)

    if verbose:
        detected = "DETECTED" if random_result.is_detected_as_anomaly else "MISSED"
        print(f"    Random: Df x alpha = {random_result.Df_x_alpha:.2f} "
              f"({random_result.deviation_percent:.1f}% dev) - {random_result.classification} [{detected}]")

    # =========================================================================
    # SUMMARY ANALYSIS
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: NOVELTY DETECTION PERFORMANCE")
        print("=" * 80)

    # Calculate detection statistics
    perturbation_results = [r for r in results if r.perturbation_type != "none" and r.perturbation_type != "random_control"]

    true_positives = sum(1 for r in perturbation_results if r.is_detected_as_anomaly)
    false_negatives = sum(1 for r in perturbation_results if not r.is_detected_as_anomaly)

    total_perturbations = len(perturbation_results)
    detection_rate = true_positives / total_perturbations if total_perturbations > 0 else 0

    # Check baseline (should NOT be detected)
    baseline_correct = not baseline_result.is_detected_as_anomaly

    # Check random control (should be detected)
    random_correct = random_result.is_detected_as_anomaly

    # Group by perturbation type
    by_type = {}
    for r in results:
        ptype = r.perturbation_type
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(r)

    # Calculate sensitivity by type
    type_sensitivity = {}
    for ptype, type_results in by_type.items():
        if ptype in ['none', 'random_control']:
            continue
        detected = sum(1 for r in type_results if r.is_detected_as_anomaly)
        type_sensitivity[ptype] = detected / len(type_results)

    # Find detection threshold (minimum perturbation level that triggers detection)
    detection_thresholds = {}
    for ptype in ['noise_injection', 'corruption', 'redundancy', 'semantic_shuffle']:
        type_sorted = sorted([r for r in by_type.get(ptype, [])], key=lambda x: x.perturbation_strength)
        for r in type_sorted:
            if r.is_detected_as_anomaly:
                detection_thresholds[ptype] = r.perturbation_strength
                break
        else:
            detection_thresholds[ptype] = None  # Never detected

    if verbose:
        print(f"\n  Baseline (normal) correctly classified: {'YES' if baseline_correct else 'NO'}")
        print(f"  Random control correctly detected: {'YES' if random_correct else 'NO'}")
        print(f"\n  Overall anomaly detection rate: {detection_rate*100:.1f}% ({true_positives}/{total_perturbations})")
        print(f"\n  Sensitivity by perturbation type:")
        for ptype, sensitivity in type_sensitivity.items():
            threshold = detection_thresholds.get(ptype, 'Never')
            if threshold is not None:
                threshold_str = f"{int(threshold*100)}%"
            else:
                threshold_str = "Never detected"
            print(f"    - {ptype}: {sensitivity*100:.0f}% (first detected at {threshold_str})")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "8e deviation can reliably detect out-of-distribution or corrupted data",
        "theoretical_values": {
            "8e": float(EIGHT_E),
            "random_baseline": RANDOM_BASELINE,
            "detection_threshold_percent": THRESHOLD_15_PERCENT * 100
        },
        "data_info": {
            "n_genes": n_genes,
            "R_min": float(R_values.min()),
            "R_max": float(R_values.max()),
            "R_mean": float(R_values.mean()),
            "R_std": float(R_values.std())
        },
        "results": [],
        "summary": {
            "baseline_correct": baseline_correct,
            "random_control_correct": random_correct,
            "total_perturbations_tested": total_perturbations,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "overall_detection_rate": detection_rate,
            "type_sensitivity": {k: float(v) for k, v in type_sensitivity.items()},
            "detection_thresholds": {k: (float(v) if v is not None else None) for k, v in detection_thresholds.items()}
        },
        "conclusion": ""
    }

    # Add individual results
    for r in results:
        output["results"].append({
            "name": r.name,
            "description": r.description,
            "perturbation_type": r.perturbation_type,
            "perturbation_strength": r.perturbation_strength,
            "n_samples": r.n_samples,
            "n_dims": r.n_dims,
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_percent,
            "passes_8e": r.passes_8e,
            "is_detected_as_anomaly": r.is_detected_as_anomaly,
            "classification": r.classification
        })

    # Generate conclusion
    if detection_rate >= 0.7 and baseline_correct and random_correct:
        conclusion = (
            "STRONG SUPPORT: 8e deviation provides reliable novelty detection. "
            f"Detection rate of {detection_rate*100:.0f}% with correct baseline/random classification. "
            "This method can identify corrupted or out-of-distribution data across multiple perturbation types."
        )
    elif detection_rate >= 0.5:
        conclusion = (
            "PARTIAL SUPPORT: 8e deviation provides moderate novelty detection capability. "
            f"Detection rate of {detection_rate*100:.0f}% - some perturbations are missed at low levels. "
            "May need to be combined with other methods for robust anomaly detection."
        )
    elif detection_rate >= 0.3:
        conclusion = (
            "WEAK SUPPORT: 8e deviation shows some sensitivity to perturbations "
            f"(detection rate: {detection_rate*100:.0f}%) but misses many anomalies. "
            "Not recommended as a standalone novelty detector."
        )
    else:
        conclusion = (
            "INSUFFICIENT SUPPORT: 8e deviation does not reliably detect perturbations "
            f"(detection rate: {detection_rate*100:.0f}%). The hypothesis is not supported by this test."
        )

    output["conclusion"] = conclusion

    if verbose:
        print(f"\n  CONCLUSION: {conclusion}")
        print("=" * 80)

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_novelty_detection_tests(verbose=True)

    # Save JSON results
    output_path = Path(__file__).parent / "8e_novelty_detection_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
