"""
Q23 Phase 1: Discover Where sqrt(3) Matters

This test uses REAL embeddings from REAL models with REAL vocabulary.
No synthetic data.

Goal: Find where sqrt(3) provides advantage in the R formula.
- Test 1A: Threshold discovery
- Test 1B: Scaling factor discovery
- Test 1C: Dimensional exponent discovery
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
from scipy import stats

# Import utilities
from q23_utils import (
    SQRT_2, SQRT_3, SQRT_5, PHI, E,
    ALPHA_CANDIDATES, ALPHA_NAMES,
    TestResult, cohens_d, bootstrap_ci, cv, save_result,
    run_all_validations
)

# =============================================================================
# REAL VOCABULARY DATASETS
# =============================================================================

# Semantically related word clusters (should have HIGH R - high agreement)
# BALANCED: 10 clusters to match UNRELATED_CLUSTERS (fixes class imbalance bug)
RELATED_CLUSTERS = [
    # Emotions
    ["happy", "joyful", "delighted", "pleased", "content", "cheerful", "elated"],
    ["sad", "unhappy", "sorrowful", "melancholy", "gloomy", "dejected", "depressed"],

    # Animals by category
    ["dog", "puppy", "hound", "canine", "terrier", "retriever", "spaniel"],
    ["cat", "kitten", "feline", "tabby", "siamese", "persian", "calico"],

    # Professions
    ["doctor", "physician", "surgeon", "nurse", "medic", "clinician", "practitioner"],
    ["teacher", "professor", "instructor", "educator", "tutor", "lecturer", "mentor"],

    # Foods
    ["apple", "orange", "banana", "grape", "strawberry", "blueberry", "cherry"],
    ["bread", "toast", "bagel", "croissant", "muffin", "biscuit", "roll"],

    # Vehicles
    ["car", "automobile", "vehicle", "sedan", "coupe", "hatchback", "wagon"],
    ["plane", "airplane", "aircraft", "jet", "airliner", "jumbo", "propeller"],
]

# Semantically UNRELATED word clusters (should have LOW R - low agreement)
UNRELATED_CLUSTERS = [
    ["quantum", "banana", "democracy", "purple", "saxophone", "algorithm", "umbrella"],
    ["philosophy", "refrigerator", "penguin", "calculus", "lighthouse", "symphony", "tornado"],
    ["keyboard", "waterfall", "monarchy", "cinnamon", "telescope", "metaphor", "volcano"],
    ["microscope", "hamburger", "constitution", "lavender", "accordion", "hypothesis", "hurricane"],
    ["astronomy", "sandwich", "parliament", "turquoise", "xylophone", "theorem", "earthquake"],
    ["biology", "spaghetti", "democracy", "magenta", "harmonica", "equation", "tsunami"],
    ["chemistry", "croissant", "monarchy", "chartreuse", "clarinet", "axiom", "avalanche"],
    ["physics", "burrito", "republic", "burgundy", "trombone", "postulate", "blizzard"],
    ["mathematics", "lasagna", "oligarchy", "periwinkle", "oboe", "corollary", "monsoon"],
    ["economics", "pretzel", "autocracy", "vermillion", "bassoon", "lemma", "cyclone"],
]


@dataclass
class SweepResult:
    """Result of a parameter sweep."""
    parameter_name: str
    values: List[float]
    value_names: List[str]
    f1_scores: List[float]
    precision_scores: List[float]
    recall_scores: List[float]
    optimal_idx: int
    optimal_value: float
    optimal_name: str
    optimal_f1: float

    def to_dict(self) -> Dict:
        return {
            "parameter_name": self.parameter_name,
            "values": self.values,
            "value_names": self.value_names,
            "f1_scores": self.f1_scores,
            "precision_scores": self.precision_scores,
            "recall_scores": self.recall_scores,
            "optimal_idx": self.optimal_idx,
            "optimal_value": self.optimal_value,
            "optimal_name": self.optimal_name,
            "optimal_f1": self.optimal_f1,
        }


# =============================================================================
# CORE R COMPUTATION (matches r_gate.py)
# =============================================================================

def compute_R(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute R = E / sigma for a set of embeddings.
    Returns (R, E, sigma).

    E = mean of pairwise cosine similarities
    sigma = std of pairwise cosine similarities
    """
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = embeddings / norms

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)

    similarities = np.array(similarities)

    E = np.mean(similarities)
    sigma = np.std(similarities)

    # R = E / sigma (with stability epsilon)
    R = E / (sigma + 1e-8)

    return R, E, sigma


def compute_R_with_exponent(embeddings: np.ndarray, alpha: float) -> Tuple[float, float, float]:
    """
    Compute R = E^alpha / sigma (alpha as EXPONENT on E).

    This is how alpha appears in the full formula:
    R = (H^alpha / grad_H) where H is entropy/agreement.

    Returns (R, E, sigma).
    """
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = embeddings / norms

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)

    similarities = np.array(similarities)

    E = np.mean(similarities)
    sigma = np.std(similarities)

    # Handle E <= 0 (can happen with normalized embeddings)
    if E <= 0:
        E = 1e-10

    # R = E^alpha / sigma (alpha as EXPONENT)
    R = (E ** alpha) / (sigma + 1e-8)

    return R, E, sigma


def compute_R_scaled(embeddings: np.ndarray, scale_factor: float) -> float:
    """Compute R with a scaling factor applied (for backward compatibility)."""
    R, E, sigma = compute_R(embeddings)
    return R * scale_factor


def compute_R_with_alpha(embeddings: np.ndarray, alpha_base: float, dim: int) -> float:
    """
    Compute R with dimensional alpha scaling as exponent.
    alpha(d) = alpha_base^(d-2)
    """
    alpha = alpha_base ** (dim - 2)
    R, _, _ = compute_R_with_exponent(embeddings, alpha)
    return R


# =============================================================================
# LOAD REAL EMBEDDINGS
# =============================================================================

def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load a real sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install with: pip install sentence-transformers"
        )


def get_cluster_embeddings(model, cluster: List[str]) -> np.ndarray:
    """Get embeddings for a word cluster."""
    return model.encode(cluster, convert_to_numpy=True)


# =============================================================================
# TEST 1A: THRESHOLD DISCOVERY
# =============================================================================

def test_threshold_discovery(model, verbose: bool = True) -> SweepResult:
    """
    Sweep thresholds to find what value best separates related from unrelated clusters.

    For each threshold t:
    - Predict "related" if R > t
    - Compute F1, precision, recall
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1A: THRESHOLD DISCOVERY")
        print("=" * 60)

    # Compute R for all clusters
    related_Rs = []
    for cluster in RELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)
        R, _, _ = compute_R(emb)
        related_Rs.append(R)

    unrelated_Rs = []
    for cluster in UNRELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)
        R, _, _ = compute_R(emb)
        unrelated_Rs.append(R)

    related_Rs = np.array(related_Rs)
    unrelated_Rs = np.array(unrelated_Rs)

    if verbose:
        print(f"\nRelated clusters: mean R = {np.mean(related_Rs):.3f}, std = {np.std(related_Rs):.3f}")
        print(f"Unrelated clusters: mean R = {np.mean(unrelated_Rs):.3f}, std = {np.std(unrelated_Rs):.3f}")
        print(f"Cohen's d = {cohens_d(related_Rs, unrelated_Rs):.2f}")

    # Threshold candidates
    thresholds = [0.5, 1.0, SQRT_2, 1.5, SQRT_3, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
    threshold_names = ["0.5", "1.0", "sqrt(2)", "1.5", "sqrt(3)", "1.8", "2.0", "2.5", "3.0", "4.0", "5.0"]

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        # Predictions: R > t means "related"
        tp = np.sum(related_Rs > t)  # True positives
        fn = np.sum(related_Rs <= t)  # False negatives
        fp = np.sum(unrelated_Rs > t)  # False positives
        tn = np.sum(unrelated_Rs <= t)  # True negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Find optimal
    optimal_idx = np.argmax(f1_scores)

    result = SweepResult(
        parameter_name="threshold",
        values=thresholds,
        value_names=threshold_names,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        optimal_idx=optimal_idx,
        optimal_value=thresholds[optimal_idx],
        optimal_name=threshold_names[optimal_idx],
        optimal_f1=f1_scores[optimal_idx],
    )

    if verbose:
        print(f"\nThreshold sweep results:")
        for i, (t, name, f1) in enumerate(zip(thresholds, threshold_names, f1_scores)):
            marker = " <-- OPTIMAL" if i == optimal_idx else ""
            print(f"  {name:>8}: F1 = {f1:.3f}{marker}")

        print(f"\nOptimal threshold: {result.optimal_name} = {result.optimal_value:.4f}")
        print(f"Optimal F1: {result.optimal_f1:.3f}")

        # Check if sqrt(3) is optimal or near-optimal
        sqrt3_idx = threshold_names.index("sqrt(3)")
        sqrt3_f1 = f1_scores[sqrt3_idx]
        print(f"\nsqrt(3) F1: {sqrt3_f1:.3f}")
        if optimal_idx == sqrt3_idx:
            print("RESULT: sqrt(3) IS optimal threshold!")
        elif abs(sqrt3_f1 - result.optimal_f1) < 0.05:
            print(f"RESULT: sqrt(3) is NEAR optimal (within 5%)")
        else:
            print(f"RESULT: sqrt(3) is NOT optimal. Best is {result.optimal_name}")

    return result


# =============================================================================
# TEST 1B: SCALING FACTOR DISCOVERY
# =============================================================================

def test_scaling_discovery(model, verbose: bool = True) -> SweepResult:
    """
    Sweep scaling factors to find what value maximizes separation.

    Apply R_scaled = R * factor and measure separation quality.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1B: SCALING FACTOR DISCOVERY")
        print("=" * 60)

    # Compute base R for all clusters
    related_Rs = []
    for cluster in RELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)
        R, _, _ = compute_R(emb)
        related_Rs.append(R)

    unrelated_Rs = []
    for cluster in UNRELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)
        R, _, _ = compute_R(emb)
        unrelated_Rs.append(R)

    related_Rs = np.array(related_Rs)
    unrelated_Rs = np.array(unrelated_Rs)

    # Scaling factors to test
    factors = [1.0, SQRT_2, 1.5, PHI, SQRT_3, 1.8, 2.0, E, SQRT_5]
    factor_names = ["1.0", "sqrt(2)", "1.5", "phi", "sqrt(3)", "1.8", "2.0", "e", "sqrt(5)"]

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for factor in factors:
        # Scale R values
        scaled_related = related_Rs * factor
        scaled_unrelated = unrelated_Rs * factor

        # Find optimal threshold for this scaling (use median of all R values)
        all_scaled = np.concatenate([scaled_related, scaled_unrelated])
        threshold = np.median(all_scaled)

        # Predictions
        tp = np.sum(scaled_related > threshold)
        fn = np.sum(scaled_related <= threshold)
        fp = np.sum(scaled_unrelated > threshold)
        tn = np.sum(scaled_unrelated <= threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    optimal_idx = np.argmax(f1_scores)

    result = SweepResult(
        parameter_name="scaling_factor",
        values=factors,
        value_names=factor_names,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        optimal_idx=optimal_idx,
        optimal_value=factors[optimal_idx],
        optimal_name=factor_names[optimal_idx],
        optimal_f1=f1_scores[optimal_idx],
    )

    if verbose:
        print(f"\nScaling factor sweep results:")
        for i, (f, name, f1) in enumerate(zip(factors, factor_names, f1_scores)):
            marker = " <-- OPTIMAL" if i == optimal_idx else ""
            print(f"  {name:>8}: F1 = {f1:.3f}{marker}")

        print(f"\nOptimal scaling: {result.optimal_name} = {result.optimal_value:.4f}")

        # Note: Scaling doesn't change relative order, so F1 may be same for all
        if max(f1_scores) - min(f1_scores) < 0.01:
            print("\nNOTE: Scaling factor has NO effect on classification (expected)")
            print("This is because scaling preserves relative ordering.")

    return result


# =============================================================================
# TEST 1C: DIMENSIONAL EXPONENT DISCOVERY
# =============================================================================

def test_dimensional_discovery(model, verbose: bool = True) -> Dict[str, SweepResult]:
    """
    For different dimensionalities (via PCA projection), find optimal alpha base.
    alpha(d) = base^(d-2)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1C: DIMENSIONAL EXPONENT DISCOVERY")
        print("=" * 60)

    from sklearn.decomposition import PCA

    # Alpha base candidates
    bases = [SQRT_2, PHI, SQRT_3, 1.8, 2.0, E, SQRT_5]
    base_names = ["sqrt(2)", "phi", "sqrt(3)", "1.8", "2.0", "e", "sqrt(5)"]

    # Dimensions limited by cluster size (7 words = max 6 dimensions after mean-centering)
    dimensions = [2, 3, 5, 6]

    results = {}

    for dim in dimensions:
        if verbose:
            print(f"\n--- Dimension {dim} ---")

        related_Rs = []
        for cluster in RELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)

            # Project to target dimension if needed
            if emb.shape[1] > dim:
                pca = PCA(n_components=dim)
                emb = pca.fit_transform(emb)

            R, _, _ = compute_R(emb)
            related_Rs.append(R)

        unrelated_Rs = []
        for cluster in UNRELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)

            if emb.shape[1] > dim:
                pca = PCA(n_components=dim)
                emb = pca.fit_transform(emb)

            R, _, _ = compute_R(emb)
            unrelated_Rs.append(R)

        related_Rs = np.array(related_Rs)
        unrelated_Rs = np.array(unrelated_Rs)

        # Test each alpha base
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for base in bases:
            alpha = base ** (dim - 2)

            scaled_related = related_Rs * alpha
            scaled_unrelated = unrelated_Rs * alpha

            all_scaled = np.concatenate([scaled_related, scaled_unrelated])
            threshold = np.median(all_scaled)

            tp = np.sum(scaled_related > threshold)
            fn = np.sum(scaled_related <= threshold)
            fp = np.sum(scaled_unrelated > threshold)
            tn = np.sum(scaled_unrelated <= threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)

        optimal_idx = np.argmax(f1_scores)

        result = SweepResult(
            parameter_name=f"alpha_base_dim{dim}",
            values=bases,
            value_names=base_names,
            f1_scores=f1_scores,
            precision_scores=precision_scores,
            recall_scores=recall_scores,
            optimal_idx=optimal_idx,
            optimal_value=bases[optimal_idx],
            optimal_name=base_names[optimal_idx],
            optimal_f1=f1_scores[optimal_idx],
        )

        results[f"dim_{dim}"] = result

        if verbose:
            print(f"  Optimal base at dim={dim}: {result.optimal_name}")
            print(f"  F1 scores: " + ", ".join(f"{name}={f1:.3f}" for name, f1 in zip(base_names, f1_scores)))

    return results


# =============================================================================
# TEST 1D: ALPHA AS EXPONENT DISCOVERY
# =============================================================================

def test_exponent_discovery(model, verbose: bool = True) -> SweepResult:
    """
    Test alpha as an EXPONENT on E: R = E^alpha / sigma

    This is how alpha appears in the full formula.
    Find which alpha maximizes separation between related and unrelated clusters.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1D: ALPHA AS EXPONENT DISCOVERY")
        print("=" * 60)
        print("Formula: R = E^alpha / sigma")

    # Alpha values to test (as exponents)
    alphas = [0.5, 1/SQRT_3, 0.7, 0.8, 1.0, SQRT_2, 1.5, SQRT_3, 2.0, E]
    alpha_names = ["0.5", "1/sqrt(3)", "0.7", "0.8", "1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "e"]

    f1_scores = []
    precision_scores = []
    recall_scores = []
    cohens_ds = []

    for alpha in alphas:
        # Compute R with alpha as exponent for all clusters
        related_Rs = []
        for cluster in RELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            related_Rs.append(R)

        unrelated_Rs = []
        for cluster in UNRELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            unrelated_Rs.append(R)

        related_Rs = np.array(related_Rs)
        unrelated_Rs = np.array(unrelated_Rs)

        # Compute Cohen's d (effect size)
        d = cohens_d(related_Rs, unrelated_Rs)
        cohens_ds.append(d)

        # Use median as threshold
        all_Rs = np.concatenate([related_Rs, unrelated_Rs])
        threshold = np.median(all_Rs)

        tp = np.sum(related_Rs > threshold)
        fn = np.sum(related_Rs <= threshold)
        fp = np.sum(unrelated_Rs > threshold)
        tn = np.sum(unrelated_Rs <= threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    optimal_idx = np.argmax(f1_scores)

    result = SweepResult(
        parameter_name="alpha_exponent",
        values=alphas,
        value_names=alpha_names,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        optimal_idx=optimal_idx,
        optimal_value=alphas[optimal_idx],
        optimal_name=alpha_names[optimal_idx],
        optimal_f1=f1_scores[optimal_idx],
    )

    if verbose:
        print(f"\nAlpha exponent sweep results:")
        for i, (a, name, f1, d) in enumerate(zip(alphas, alpha_names, f1_scores, cohens_ds)):
            marker = " <-- OPTIMAL" if i == optimal_idx else ""
            print(f"  {name:>10}: F1 = {f1:.3f}, Cohen's d = {d:.2f}{marker}")

        print(f"\nOptimal alpha exponent: {result.optimal_name} = {result.optimal_value:.4f}")
        print(f"Optimal F1: {result.optimal_f1:.3f}")

        # Check if sqrt(3)-related values are optimal
        sqrt3_related = ["sqrt(3)", "1/sqrt(3)"]
        for name in sqrt3_related:
            if name in alpha_names:
                idx = alpha_names.index(name)
                f1 = f1_scores[idx]
                print(f"\n{name} F1: {f1:.3f}")
                if idx == optimal_idx:
                    print(f"RESULT: {name} IS optimal exponent!")
                elif abs(f1 - result.optimal_f1) < 0.05:
                    print(f"RESULT: {name} is NEAR optimal (within 5%)")
                else:
                    print(f"RESULT: {name} is NOT optimal")

    return result


# =============================================================================
# CROSS-MODEL VALIDATION
# =============================================================================

def run_cross_model_validation(model_names: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Run all Phase 1 tests across multiple models."""
    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-MODEL VALIDATION")
        print("=" * 60)

    results = {}
    threshold_optima = []

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*40}")
            print(f"Model: {model_name}")
            print("=" * 40)

        try:
            model = load_model(model_name)

            threshold_result = test_threshold_discovery(model, verbose=verbose)
            scaling_result = test_scaling_discovery(model, verbose=verbose)

            results[model_name] = {
                "threshold": threshold_result.to_dict(),
                "scaling": scaling_result.to_dict(),
            }

            threshold_optima.append(threshold_result.optimal_value)

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue

    # Compute cross-model statistics
    if threshold_optima:
        threshold_optima = np.array(threshold_optima)
        results["cross_model_stats"] = {
            "threshold_mean": float(np.mean(threshold_optima)),
            "threshold_std": float(np.std(threshold_optima)),
            "threshold_cv": float(cv(threshold_optima)),
            "n_models": len(threshold_optima),
        }

        if verbose:
            print(f"\n{'='*60}")
            print("CROSS-MODEL SUMMARY")
            print("=" * 60)
            print(f"Optimal threshold mean: {np.mean(threshold_optima):.3f}")
            print(f"Optimal threshold std: {np.std(threshold_optima):.3f}")
            print(f"CV: {cv(threshold_optima)*100:.1f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Q23 PHASE 1: DISCOVER WHERE sqrt(3) MATTERS")
    print("=" * 60)
    print("\nUsing REAL embeddings from trained models")
    print("NO synthetic data")
    print()

    # Phase 0: Validate test utilities
    if not run_all_validations():
        print("\nABORTING: Test validation failed")
        return

    # Load primary model
    print("\nLoading model...")
    model = load_model("all-MiniLM-L6-v2")

    # Run Phase 1 tests
    threshold_result = test_threshold_discovery(model)
    scaling_result = test_scaling_discovery(model)
    exponent_result = test_exponent_discovery(model)
    dim_results = test_dimensional_discovery(model)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)

    print(f"\n1A. Optimal THRESHOLD: {threshold_result.optimal_name} = {threshold_result.optimal_value:.4f}")
    sqrt3_idx = threshold_result.value_names.index("sqrt(3)") if "sqrt(3)" in threshold_result.value_names else -1
    if sqrt3_idx >= 0:
        print(f"    sqrt(3) F1: {threshold_result.f1_scores[sqrt3_idx]:.3f}")
        print(f"    Optimal F1: {threshold_result.optimal_f1:.3f}")

    print(f"\n1B. SCALING FACTOR: No effect (preserves relative ordering)")

    print(f"\n1C. ALPHA AS EXPONENT (R = E^alpha / sigma):")
    print(f"    Optimal alpha: {exponent_result.optimal_name} = {exponent_result.optimal_value:.4f}")
    print(f"    Optimal F1: {exponent_result.optimal_f1:.3f}")

    print(f"\n1D. DIMENSIONAL EXPONENT:")
    for dim_name, result in dim_results.items():
        print(f"    {dim_name}: optimal base = {result.optimal_name}")

    # Check if sqrt(3) appears
    sqrt3_found = False
    sqrt3_near = False

    if threshold_result.optimal_name == "sqrt(3)":
        sqrt3_found = True
        print("\n*** sqrt(3) IS OPTIMAL THRESHOLD ***")
    elif "sqrt(3)" in threshold_result.value_names:
        idx = threshold_result.value_names.index("sqrt(3)")
        if abs(threshold_result.f1_scores[idx] - threshold_result.optimal_f1) < 0.05:
            sqrt3_near = True

    if exponent_result.optimal_name == "sqrt(3)" or exponent_result.optimal_name == "1/sqrt(3)":
        sqrt3_found = True
        print(f"\n*** {exponent_result.optimal_name} IS OPTIMAL EXPONENT ***")
    elif "sqrt(3)" in exponent_result.value_names:
        idx = exponent_result.value_names.index("sqrt(3)")
        if abs(exponent_result.f1_scores[idx] - exponent_result.optimal_f1) < 0.05:
            sqrt3_near = True
        idx2 = exponent_result.value_names.index("1/sqrt(3)") if "1/sqrt(3)" in exponent_result.value_names else -1
        if idx2 >= 0 and abs(exponent_result.f1_scores[idx2] - exponent_result.optimal_f1) < 0.05:
            sqrt3_near = True

    for dim_name, result in dim_results.items():
        if result.optimal_name == "sqrt(3)":
            sqrt3_found = True
            print(f"\n*** sqrt(3) IS OPTIMAL BASE at {dim_name} ***")

    if sqrt3_found:
        print("\n*** sqrt(3) CONFIRMED in at least one test ***")
    elif sqrt3_near:
        print("\n*** sqrt(3) is NEAR optimal (within 5%) but not best ***")
    else:
        print("\n*** sqrt(3) was NOT found to be optimal in any test ***")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "all-MiniLM-L6-v2",
        "threshold_sweep": threshold_result.to_dict(),
        "scaling_sweep": scaling_result.to_dict(),
        "exponent_sweep": exponent_result.to_dict(),
        "dimensional_sweeps": {k: v.to_dict() for k, v in dim_results.items()},
        "sqrt3_found_optimal": sqrt3_found,
        "sqrt3_near_optimal": sqrt3_near,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, f"q23_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
