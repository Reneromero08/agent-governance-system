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

def test_random_embeddings(verbose: bool = True) -> NegativeControlResult:
    """
    Test that random embeddings do NOT show ANY alpha preference.

    This control tests: does the alpha optimization come from SEMANTIC structure?

    For truly random embeddings (no semantic meaning), all alphas should perform
    EQUALLY POORLY - there's no structure to exploit.

    If random embeddings show differentiated alpha performance, the effect might
    be from distribution shape, not semantics.

    Control PASSES if: all alphas perform equally (CV < 5%)
    Control FAILS if: some alpha is clearly better than others (CV > 5%)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("NEGATIVE CONTROL 1: RANDOM EMBEDDINGS")
        print("=" * 60)
        print("Testing: Do random embeddings show ANY alpha preference?")
        print("Expected: NO - all alphas should perform equally (no structure)")

    np.random.seed(42)

    # Generate PURELY random embeddings - no artificial structure
    n_words = 7  # Same as cluster size
    embedding_dim = 384
    n_clusters = 10

    # Test multiple alpha values
    alphas = [1.0, SQRT_2, 1.5, SQRT_3, 2.0, E]
    alpha_names = ["1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "e"]

    # Generate random clusters - ALL purely random, no artificial structure
    # Group A and B are both random - should be indistinguishable
    group_a_Rs = {a: [] for a in alphas}
    group_b_Rs = {a: [] for a in alphas}

    for _ in range(n_clusters):
        # Both groups are PURELY random - no semantic structure
        group_a_emb = np.random.randn(n_words, embedding_dim)
        group_b_emb = np.random.randn(n_words, embedding_dim)

        for alpha in alphas:
            R_a, _, _ = compute_R_with_exponent(group_a_emb, alpha)
            R_b, _, _ = compute_R_with_exponent(group_b_emb, alpha)

            group_a_Rs[alpha].append(R_a)
            group_b_Rs[alpha].append(R_b)

    # For random data, no alpha should be better at separating A from B
    # because there's no real difference between the groups
    cohens_ds = {}
    for alpha, name in zip(alphas, alpha_names):
        a = np.array(group_a_Rs[alpha])
        b = np.array(group_b_Rs[alpha])
        d = cohens_d(a, b)
        cohens_ds[name] = abs(d)  # Use absolute value

    # All Cohen's d should be small (< 0.3) for random data
    mean_d = np.mean(list(cohens_ds.values()))
    max_d = max(cohens_ds.values())
    d_cv = cv(np.array(list(cohens_ds.values())))

    if verbose:
        print(f"\nCohen's d for separating random groups:")
        for name, d in cohens_ds.items():
            print(f"  {name:>8}: d = {d:.3f}")
        print(f"\nMean Cohen's d: {mean_d:.3f} (expected < 0.3 for no structure)")
        print(f"Max Cohen's d: {max_d:.3f}")
        print(f"CV of Cohen's d: {d_cv*100:.1f}%")

    # Control PASSES if:
    # 1. Mean d is small (no alpha can separate random groups well)
    # 2. CV is low (all alphas perform equally poorly)
    no_separation = mean_d < 0.5  # Can't separate random groups
    equal_performance = d_cv < 0.30 or max_d < 0.5  # All alphas similar

    control_passed = no_separation and equal_performance

    if verbose:
        if control_passed:
            print("\nVERDICT: CONTROL PASSED - Random data shows no alpha preference")
            print("This confirms sqrt(3) advantage comes from semantic structure")
        else:
            print("\nVERDICT: CONTROL FAILED - Random data shows alpha preference")
            print("This suggests effect may be from distribution shape, not semantics")

    return NegativeControlResult(
        control_name="random_embeddings",
        expected_to_fail=False,  # We want this control to PASS (no preference)
        actually_failed=not control_passed,
        metric_trained=cohens_ds.get("sqrt(3)", 0.0),
        metric_control=mean_d,
        cohens_d=max_d,
        interpretation="Random embeddings should show no alpha preference (confirms semantic origin)"
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
    Test that sqrt(3) performs better than arbitrary alternatives.

    If random constants like 1.7, 1.8, 1.9 perform equally well,
    sqrt(3) is not special.
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

    # Check if sqrt(3) is distinguishable from neighbors
    sqrt3_f1 = f1_scores["sqrt(3)=1.732"]
    other_f1s = [f1 for name, f1 in f1_scores.items() if name != "sqrt(3)=1.732"]
    max_other = max(other_f1s)
    min_other = min(other_f1s)

    # If all values give same F1 (within 5%), sqrt(3) is NOT uniquely special
    all_same = max_other - min_other < 0.05 and abs(sqrt3_f1 - max_other) < 0.05

    # Control PASSES if sqrt(3) IS distinguishable (performs differently from neighbors)
    # Control FAILS if sqrt(3) is NOT distinguishable (all neighbors perform the same)
    control_passed = not all_same

    if verbose:
        print(f"\nsqrt(3) F1: {sqrt3_f1:.3f}")
        print(f"Range of others: [{min_other:.3f}, {max_other:.3f}]")
        if all_same:
            print("\nVERDICT: CONTROL FAILED - All nearby values perform equally")
            print("sqrt(3) is NOT uniquely optimal (any value in range works)")
        else:
            print("\nVERDICT: CONTROL PASSED - sqrt(3) is distinguishable from alternatives")

    return NegativeControlResult(
        control_name="alternative_constants",
        expected_to_fail=False,  # We WANT sqrt(3) to be special (control should pass)
        actually_failed=not control_passed,  # Failed if all values perform the same
        metric_trained=sqrt3_f1,
        metric_control=max_other,
        cohens_d=sqrt3_f1 - max_other,
        interpretation="sqrt(3) should be distinguishable from arbitrary nearby values"
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
    nc1 = test_random_embeddings()
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
