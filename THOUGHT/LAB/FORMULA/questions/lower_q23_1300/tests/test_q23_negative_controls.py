"""
Q23 Negative Controls and Cross-Model Validation

Negative controls verify our measurements are meaningful:
1. Random embeddings should NOT show sqrt(3) advantage
2. Shuffled embeddings should lose any structure
3. Alternative constants should perform worse than sqrt(3)

Cross-model validation ensures findings are universal, not model-specific.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
from scipy import stats

from q23_utils import (
    SQRT_2, SQRT_3, SQRT_5, PHI, E,
    TestResult, cohens_d, cv, run_all_validations
)

# Import test functions from other modules
from test_q23_alpha_sweep import (
    RELATED_CLUSTERS, UNRELATED_CLUSTERS,
    compute_R, compute_R_with_exponent, get_cluster_embeddings, load_model
)


@dataclass
class NegativeControlResult:
    """Result of a negative control test."""
    control_name: str
    expected_to_fail: bool  # Should this control show NO sqrt(3) signal?
    actually_failed: bool  # Did it fail as expected?
    metric_trained: float  # Metric for trained embeddings
    metric_control: float  # Metric for control condition
    cohens_d: float  # Effect size
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "control_name": self.control_name,
            "expected_to_fail": self.expected_to_fail,
            "actually_failed": self.actually_failed,
            "passed_validation": self.expected_to_fail == self.actually_failed,
            "metric_trained": self.metric_trained,
            "metric_control": self.metric_control,
            "cohens_d": self.cohens_d,
            "interpretation": self.interpretation,
        }


# =============================================================================
# NEGATIVE CONTROL 1: RANDOM EMBEDDINGS
# =============================================================================

def test_random_embeddings(model, verbose: bool = True) -> NegativeControlResult:
    """
    Test that the PATTERN of alpha-F1 relationship differs between trained and random embeddings.

    This control tests: does the alpha optimization come from SEMANTIC structure?

    For trained embeddings, we expect a meaningful relationship between alpha and F1.
    For random embeddings, we expect NO meaningful relationship (flat or random pattern).

    Implementation:
    1. Compute F1 scores for trained embeddings at various alphas
    2. Compute F1 scores for random embeddings at the same alphas
    3. Compare the correlation between alpha and F1 for each

    Control PASSES if: trained embeddings show stronger alpha-F1 relationship than random
    Control FAILS if: random embeddings show similar alpha-F1 relationship as trained
    """
    if verbose:
        print("\n" + "=" * 60)
        print("NEGATIVE CONTROL 1: RANDOM EMBEDDINGS")
        print("=" * 60)
        print("Testing: Does alpha-F1 pattern differ between trained and random?")
        print("Expected: Trained shows stronger alpha-F1 relationship than random")

    np.random.seed(42)

    # Test multiple alpha values
    alphas = [1.0, SQRT_2, 1.5, SQRT_3, 2.0, E]
    alpha_names = ["1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "e"]
    alpha_values = np.array(alphas)

    # Helper function to compute F1
    def compute_f1(related_Rs, unrelated_Rs):
        all_Rs = np.concatenate([related_Rs, unrelated_Rs])
        threshold = np.median(all_Rs)
        tp = np.sum(related_Rs > threshold)
        fp = np.sum(unrelated_Rs > threshold)
        fn = np.sum(related_Rs <= threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # --- TRAINED EMBEDDINGS ---
    trained_f1_scores = []
    for alpha in alphas:
        related_Rs = []
        unrelated_Rs = []

        for cluster in RELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            related_Rs.append(R)

        for cluster in UNRELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            unrelated_Rs.append(R)

        f1 = compute_f1(np.array(related_Rs), np.array(unrelated_Rs))
        trained_f1_scores.append(f1)

    trained_f1_scores = np.array(trained_f1_scores)

    # --- RANDOM EMBEDDINGS ---
    # Generate random clusters matching the structure of the real clusters
    n_words = 7  # Same as cluster size
    embedding_dim = 384
    n_related_clusters = len(RELATED_CLUSTERS)
    n_unrelated_clusters = len(UNRELATED_CLUSTERS)

    random_f1_scores = []
    for alpha in alphas:
        related_Rs = []
        unrelated_Rs = []

        # Random "related" clusters (but truly random, so no real structure)
        for _ in range(n_related_clusters):
            emb = np.random.randn(n_words, embedding_dim)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            related_Rs.append(R)

        # Random "unrelated" clusters (also truly random)
        for _ in range(n_unrelated_clusters):
            emb = np.random.randn(n_words, embedding_dim)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            unrelated_Rs.append(R)

        f1 = compute_f1(np.array(related_Rs), np.array(unrelated_Rs))
        random_f1_scores.append(f1)

    random_f1_scores = np.array(random_f1_scores)

    # --- COMPARE PATTERNS ---
    # Compute correlation between alpha and F1 for each
    trained_corr, trained_p = stats.spearmanr(alpha_values, trained_f1_scores)
    random_corr, random_p = stats.spearmanr(alpha_values, random_f1_scores)

    # Compute range (max - min) of F1 scores as measure of alpha sensitivity
    trained_range = np.max(trained_f1_scores) - np.min(trained_f1_scores)
    random_range = np.max(random_f1_scores) - np.min(random_f1_scores)

    # Compute CV of F1 scores
    trained_cv = cv(trained_f1_scores) if np.mean(trained_f1_scores) > 0 else 0
    random_cv = cv(random_f1_scores) if np.mean(random_f1_scores) > 0 else 0

    if verbose:
        print(f"\nTrained embeddings F1 scores by alpha:")
        for name, f1 in zip(alpha_names, trained_f1_scores):
            print(f"  {name:>8}: F1 = {f1:.3f}")
        print(f"  Correlation (alpha vs F1): r = {trained_corr:.3f}, p = {trained_p:.3f}")
        print(f"  F1 range: {trained_range:.3f}")
        print(f"  F1 CV: {trained_cv*100:.1f}%")

        print(f"\nRandom embeddings F1 scores by alpha:")
        for name, f1 in zip(alpha_names, random_f1_scores):
            print(f"  {name:>8}: F1 = {f1:.3f}")
        print(f"  Correlation (alpha vs F1): r = {random_corr:.3f}, p = {random_p:.3f}")
        print(f"  F1 range: {random_range:.3f}")
        print(f"  F1 CV: {random_cv*100:.1f}%")

    # Control PASSES if trained shows stronger pattern than random:
    # 1. Trained has higher correlation (absolute) than random, OR
    # 2. Trained has larger F1 range than random, OR
    # 3. Random F1 range is very small (< 0.1) indicating no sensitivity
    trained_has_stronger_corr = abs(trained_corr) > abs(random_corr) + 0.1
    trained_has_larger_range = trained_range > random_range + 0.05
    random_is_flat = random_range < 0.1

    control_passed = trained_has_stronger_corr or trained_has_larger_range or random_is_flat

    if verbose:
        print(f"\nComparison:")
        print(f"  Trained correlation stronger: {trained_has_stronger_corr} (|{trained_corr:.3f}| vs |{random_corr:.3f}|)")
        print(f"  Trained range larger: {trained_has_larger_range} ({trained_range:.3f} vs {random_range:.3f})")
        print(f"  Random is flat: {random_is_flat} (range {random_range:.3f} < 0.1)")

        if control_passed:
            print("\nVERDICT: CONTROL PASSED - Trained shows different/stronger alpha-F1 pattern")
            print("This confirms alpha optimization comes from semantic structure")
        else:
            print("\nVERDICT: CONTROL FAILED - Random shows similar alpha-F1 pattern")
            print("This suggests effect may be from distribution shape, not semantics")

    return NegativeControlResult(
        control_name="random_embeddings",
        expected_to_fail=False,  # We want this control to PASS (different patterns)
        actually_failed=not control_passed,
        metric_trained=trained_range,
        metric_control=random_range,
        cohens_d=abs(trained_corr) - abs(random_corr),
        interpretation="Trained embeddings should show stronger alpha-F1 relationship than random"
    )


# =============================================================================
# NEGATIVE CONTROL 2: SHUFFLED EMBEDDINGS
# =============================================================================

def test_shuffled_embeddings(model, verbose: bool = True) -> NegativeControlResult:
    """
    Test that shuffling embedding dimensions destroys structure.

    If shuffled embeddings still show sqrt(3) advantage, the structure is
    just from the distribution shape, not semantic meaning.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("NEGATIVE CONTROL 2: SHUFFLED EMBEDDINGS")
        print("=" * 60)

    np.random.seed(42)

    # Get real embeddings
    alphas = [1.0, SQRT_2, 1.5, SQRT_3, 2.0, E]
    alpha_names = ["1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "e"]

    # Compute for normal embeddings
    normal_related_Rs = {a: [] for a in alphas}
    normal_unrelated_Rs = {a: [] for a in alphas}

    shuffled_related_Rs = {a: [] for a in alphas}
    shuffled_unrelated_Rs = {a: [] for a in alphas}

    for cluster in RELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)

        # Shuffle: permute dimensions independently for each embedding
        shuffled_emb = emb.copy()
        for i in range(len(shuffled_emb)):
            np.random.shuffle(shuffled_emb[i])

        for alpha in alphas:
            R_normal, _, _ = compute_R_with_exponent(emb, alpha)
            R_shuffled, _, _ = compute_R_with_exponent(shuffled_emb, alpha)

            normal_related_Rs[alpha].append(R_normal)
            shuffled_related_Rs[alpha].append(R_shuffled)

    for cluster in UNRELATED_CLUSTERS:
        emb = get_cluster_embeddings(model, cluster)

        shuffled_emb = emb.copy()
        for i in range(len(shuffled_emb)):
            np.random.shuffle(shuffled_emb[i])

        for alpha in alphas:
            R_normal, _, _ = compute_R_with_exponent(emb, alpha)
            R_shuffled, _, _ = compute_R_with_exponent(shuffled_emb, alpha)

            normal_unrelated_Rs[alpha].append(R_normal)
            shuffled_unrelated_Rs[alpha].append(R_shuffled)

    # Compare F1 for normal vs shuffled
    def compute_f1(related, unrelated):
        all_Rs = np.concatenate([related, unrelated])
        threshold = np.median(all_Rs)
        tp = np.sum(related > threshold)
        fp = np.sum(unrelated > threshold)
        fn = np.sum(related <= threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    normal_f1_sqrt3 = compute_f1(
        np.array(normal_related_Rs[SQRT_3]),
        np.array(normal_unrelated_Rs[SQRT_3])
    )

    shuffled_f1_sqrt3 = compute_f1(
        np.array(shuffled_related_Rs[SQRT_3]),
        np.array(shuffled_unrelated_Rs[SQRT_3])
    )

    if verbose:
        print(f"\nF1 with alpha=sqrt(3):")
        print(f"  Normal embeddings: {normal_f1_sqrt3:.3f}")
        print(f"  Shuffled embeddings: {shuffled_f1_sqrt3:.3f}")

    # Shuffling should significantly reduce performance
    performance_drop = normal_f1_sqrt3 - shuffled_f1_sqrt3
    actually_failed = shuffled_f1_sqrt3 < normal_f1_sqrt3 - 0.1  # At least 10% drop

    if verbose:
        print(f"  Performance drop: {performance_drop:.3f}")
        if actually_failed:
            print("\nVERDICT: PASS - Shuffling destroys structure (as expected)")
        else:
            print("\nVERDICT: FAIL - Shuffling does NOT destroy structure (bad sign)")

    return NegativeControlResult(
        control_name="shuffled_embeddings",
        expected_to_fail=True,
        actually_failed=actually_failed,
        metric_trained=normal_f1_sqrt3,
        metric_control=shuffled_f1_sqrt3,
        cohens_d=performance_drop,
        interpretation="Shuffling should destroy semantic structure"
    )


# =============================================================================
# NEGATIVE CONTROL 3: ALTERNATIVE CONSTANTS
# =============================================================================

def test_alternative_constants(model, verbose: bool = True) -> NegativeControlResult:
    """
    Test whether sqrt(3) is actually optimal or if other values perform equally well or better.

    This control checks:
    1. Is sqrt(3) in the top 2 performers?
    2. If not, what constants actually perform best?
    3. Are all values essentially equivalent (flat response)?

    The control interpretation reflects the actual findings about sqrt(3) optimality.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("NEGATIVE CONTROL 3: ALTERNATIVE CONSTANTS")
        print("=" * 60)

    # Test sqrt(3) against nearby arbitrary values
    test_values = [1.6, 1.7, SQRT_3, 1.75, 1.8, 1.9]
    test_names = ["1.6", "1.7", "sqrt(3)=1.732", "1.75", "1.8", "1.9"]

    f1_scores = {}

    for alpha, name in zip(test_values, test_names):
        related_Rs = []
        unrelated_Rs = []

        for cluster in RELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            related_Rs.append(R)

        for cluster in UNRELATED_CLUSTERS:
            emb = get_cluster_embeddings(model, cluster)
            R, _, _ = compute_R_with_exponent(emb, alpha)
            unrelated_Rs.append(R)

        related_Rs = np.array(related_Rs)
        unrelated_Rs = np.array(unrelated_Rs)

        all_Rs = np.concatenate([related_Rs, unrelated_Rs])
        threshold = np.median(all_Rs)

        tp = np.sum(related_Rs > threshold)
        fp = np.sum(unrelated_Rs > threshold)
        fn = np.sum(related_Rs <= threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores[name] = f1

    if verbose:
        print(f"\nF1 scores for nearby constants:")
        for name, f1 in f1_scores.items():
            print(f"  {name:>15}: F1 = {f1:.3f}")

    # Rank all values by F1
    sorted_scores = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    top_2_names = [name for name, _ in sorted_scores[:2]]
    best_name, best_f1 = sorted_scores[0]
    second_name, second_f1 = sorted_scores[1]

    sqrt3_f1 = f1_scores["sqrt(3)=1.732"]
    sqrt3_rank = [name for name, _ in sorted_scores].index("sqrt(3)=1.732") + 1

    other_f1s = [f1 for name, f1 in f1_scores.items() if name != "sqrt(3)=1.732"]
    max_other = max(other_f1s)
    min_other = min(other_f1s)

    # Determine sqrt(3) status
    sqrt3_is_best = best_name == "sqrt(3)=1.732"
    sqrt3_in_top_2 = "sqrt(3)=1.732" in top_2_names
    sqrt3_beaten_by = best_f1 - sqrt3_f1 if not sqrt3_is_best else 0.0

    # Check if all values are essentially equivalent (flat response)
    all_f1s = list(f1_scores.values())
    f1_range = max(all_f1s) - min(all_f1s)
    all_equivalent = f1_range < 0.05

    if verbose:
        print(f"\nRanking:")
        for i, (name, f1) in enumerate(sorted_scores):
            marker = " <-- sqrt(3)" if name == "sqrt(3)=1.732" else ""
            print(f"  {i+1}. {name:>15}: F1 = {f1:.3f}{marker}")

        print(f"\nsqrt(3) analysis:")
        print(f"  sqrt(3) F1: {sqrt3_f1:.3f}")
        print(f"  sqrt(3) rank: {sqrt3_rank} of {len(test_names)}")
        print(f"  sqrt(3) in top 2: {sqrt3_in_top_2}")
        print(f"  Best performer: {best_name} (F1 = {best_f1:.3f})")
        if not sqrt3_is_best:
            print(f"  sqrt(3) beaten by: {sqrt3_beaten_by:.3f}")
        print(f"  F1 range across all: {f1_range:.3f}")
        print(f"  All values equivalent: {all_equivalent}")

    # Build interpretation based on actual findings
    if all_equivalent:
        interpretation = f"All values perform equivalently (range={f1_range:.3f}<0.05). sqrt(3) is not special - any value in range works."
        control_passed = False  # sqrt(3) is not uniquely special
    elif sqrt3_is_best:
        interpretation = f"sqrt(3) IS optimal (rank 1, F1={sqrt3_f1:.3f}). Confirms sqrt(3) hypothesis."
        control_passed = True
    elif sqrt3_in_top_2:
        interpretation = f"sqrt(3) is near-optimal (rank {sqrt3_rank}, F1={sqrt3_f1:.3f}). Best is {best_name}={best_f1:.3f}. Within acceptable range."
        control_passed = True  # Near-optimal is still supportive
    else:
        interpretation = f"sqrt(3) is NOT optimal (rank {sqrt3_rank}, F1={sqrt3_f1:.3f}). Best is {best_name}={best_f1:.3f}. sqrt(3) hypothesis NOT supported."
        control_passed = False

    if verbose:
        print(f"\nInterpretation: {interpretation}")
        if control_passed:
            print("\nVERDICT: CONTROL PASSED - sqrt(3) is optimal or near-optimal")
        else:
            print("\nVERDICT: CONTROL FAILED - sqrt(3) is NOT in top performers")

    return NegativeControlResult(
        control_name="alternative_constants",
        expected_to_fail=False,  # We WANT sqrt(3) to be special (control should pass)
        actually_failed=not control_passed,
        metric_trained=sqrt3_f1,
        metric_control=best_f1 if not sqrt3_is_best else second_f1,
        cohens_d=sqrt3_f1 - best_f1,  # Negative if sqrt(3) is not best
        interpretation=interpretation
    )


# =============================================================================
# CROSS-MODEL VALIDATION
# =============================================================================

def run_cross_model_validation(verbose: bool = True) -> Dict[str, Any]:
    """
    Run tests across multiple models to verify findings are universal.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-MODEL VALIDATION")
        print("=" * 60)

    model_names = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ]

    results = {}
    optimal_alphas = []

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*40}")
            print(f"Model: {model_name}")
            print("=" * 40)

        try:
            model = load_model(model_name)

            # Test which alpha is optimal for this model
            alphas = [1.0, SQRT_2, 1.5, SQRT_3, 2.0, E]
            alpha_names = ["1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "e"]

            f1_scores = {}
            for alpha, name in zip(alphas, alpha_names):
                related_Rs = []
                unrelated_Rs = []

                for cluster in RELATED_CLUSTERS:
                    emb = get_cluster_embeddings(model, cluster)
                    R, _, _ = compute_R_with_exponent(emb, alpha)
                    related_Rs.append(R)

                for cluster in UNRELATED_CLUSTERS:
                    emb = get_cluster_embeddings(model, cluster)
                    R, _, _ = compute_R_with_exponent(emb, alpha)
                    unrelated_Rs.append(R)

                related_Rs = np.array(related_Rs)
                unrelated_Rs = np.array(unrelated_Rs)

                all_Rs = np.concatenate([related_Rs, unrelated_Rs])
                threshold = np.median(all_Rs)

                tp = np.sum(related_Rs > threshold)
                fp = np.sum(unrelated_Rs > threshold)
                fn = np.sum(related_Rs <= threshold)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                f1_scores[name] = f1

            best_alpha = max(f1_scores, key=f1_scores.get)
            optimal_alphas.append(best_alpha)

            results[model_name] = {
                "f1_scores": f1_scores,
                "optimal_alpha": best_alpha,
                "sqrt3_f1": f1_scores["sqrt(3)"],
            }

            if verbose:
                print(f"  Optimal alpha: {best_alpha}")
                print(f"  sqrt(3) F1: {f1_scores['sqrt(3)']:.3f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if optimal_alphas:
        sqrt3_count = optimal_alphas.count("sqrt(3)")
        results["summary"] = {
            "n_models": len(optimal_alphas),
            "sqrt3_optimal_count": sqrt3_count,
            "sqrt3_optimal_rate": sqrt3_count / len(optimal_alphas),
            "optimal_alphas": optimal_alphas,
        }

        if verbose:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print("=" * 60)
            print(f"Models where sqrt(3) is optimal: {sqrt3_count}/{len(optimal_alphas)}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all negative controls and cross-model validation."""
    print("=" * 60)
    print("Q23 NEGATIVE CONTROLS AND CROSS-MODEL VALIDATION")
    print("=" * 60)

    # Run validations
    if not run_all_validations():
        print("\nABORTING: Test validation failed")
        return

    # Load model
    print("\nLoading model...")
    model = load_model("all-MiniLM-L6-v2")

    # Run negative controls
    nc1 = test_random_embeddings(model)
    nc2 = test_shuffled_embeddings(model)
    nc3 = test_alternative_constants(model)

    # Cross-model validation
    cross_model = run_cross_model_validation()

    # Summary
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROLS SUMMARY")
    print("=" * 60)

    controls = [nc1, nc2, nc3]
    passed_count = sum(1 for nc in controls if nc.actually_failed == nc.expected_to_fail)

    for nc in controls:
        status = "PASS" if nc.actually_failed == nc.expected_to_fail else "FAIL"
        print(f"\n{nc.control_name}: {status}")
        print(f"  {nc.interpretation}")
        print(f"  Metric (trained): {nc.metric_trained:.3f}")
        print(f"  Metric (control): {nc.metric_control:.3f}")

    print(f"\nNegative controls passed: {passed_count}/{len(controls)}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "negative_controls": {
            "random_embeddings": nc1.to_dict(),
            "shuffled_embeddings": nc2.to_dict(),
            "alternative_constants": nc3.to_dict(),
        },
        "cross_model": cross_model,
        "summary": {
            "negative_controls_passed": passed_count,
            "negative_controls_total": len(controls),
        }
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, f"q23_negative_controls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, 'w') as f:
        # Use default handler for numpy types
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else str(x))

    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
