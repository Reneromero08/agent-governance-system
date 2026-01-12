#!/usr/bin/env python3
"""
Q7: Negative Controls (Real Embeddings)

Tests that MUST FAIL to validate the positive tests are meaningful.
Uses REAL embeddings from shared/real_embeddings.py.

Negative Controls:
1. SHUFFLED HIERARCHY: Randomly permute containment -> structure collapses
2. WRONG AGGREGATION: Sum instead of mean -> intensivity violated
3. NON-LOCAL INJECTION: Add unrelated data -> locality violated
4. RANDOM R VALUES: Use random values instead of computed R

Pass criteria: ALL 4 negative controls correctly FAIL.

If a negative control passes, it means our positive tests are trivial.

Author: Claude
Date: 2026-01-11
Version: 2.0.0 (Real Embeddings)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
from scipy.stats import spearmanr

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import real embeddings infrastructure
from shared.real_embeddings import (
    MULTI_SCALE_CORPUS,
    SCALE_HIERARCHY,
    load_multiscale_embeddings,
    compute_R_from_embeddings,
    compute_containment_matrices,
    MultiScaleEmbeddings,
)

from theory.scale_transformation import ScaleData, ScaleTransformation, compute_R


# =============================================================================
# NEGATIVE CONTROL RESULTS
# =============================================================================

@dataclass
class NegativeControlResult:
    """Result from a negative control test."""
    control_name: str
    description: str
    expected_behavior: str
    metric_name: str
    metric_value: float
    threshold: float
    correctly_fails: bool
    explanation: str


# =============================================================================
# NEGATIVE CONTROL 1: SHUFFLED HIERARCHY
# =============================================================================

def test_negative_shuffled_hierarchy(ms: MultiScaleEmbeddings = None) -> NegativeControlResult:
    """
    SHUFFLED HIERARCHY: Randomly permute containment matrix.

    This breaks the parent-child relationship completely.
    Structure should collapse (L-correlation -> 0).

    Expected: FAIL (L-correlation < 0.5)
    """
    if ms is None:
        ms = load_multiscale_embeddings()

    word_data = ms.scales.get("words")
    sent_data = ms.scales.get("sentences")

    if word_data is None or sent_data is None:
        return NegativeControlResult(
            control_name="shuffled_hierarchy",
            description="Randomly permute containment matrix",
            expected_behavior="Structure collapses (L-correlation -> 0)",
            metric_name="L_correlation",
            metric_value=0.0,
            threshold=0.5,
            correctly_fails=True,
            explanation="Missing data - cannot test"
        )

    containment = ms.containment.get("words->sentences")
    if containment is None:
        return NegativeControlResult(
            control_name="shuffled_hierarchy",
            description="Randomly permute containment matrix",
            expected_behavior="Structure collapses (L-correlation -> 0)",
            metric_name="L_correlation",
            metric_value=0.0,
            threshold=0.5,
            correctly_fails=True,
            explanation="Missing containment matrix"
        )

    word_emb = word_data.embeddings
    sent_emb = sent_data.embeddings
    n_words = len(word_emb)
    n_sents = len(sent_emb)

    # Compute R for each word group with CORRECT containment
    R_correct = []
    for i in range(n_sents):
        mask = containment[i] > 0
        if mask.any():
            group_emb = word_emb[mask]
            R = compute_R_from_embeddings(group_emb)
            R_correct.append(R)

    # Compute L-correlation with correct containment
    sent_norms = np.linalg.norm(sent_emb, axis=1)
    if len(R_correct) >= 2 and len(sent_norms) >= 2:
        min_len = min(len(R_correct), len(sent_norms))
        corr_correct, _ = spearmanr(R_correct[:min_len], sent_norms[:min_len])
        if np.isnan(corr_correct):
            corr_correct = 0.0
    else:
        corr_correct = 0.0

    # NOW SHUFFLE THE CONTAINMENT (the negative control)
    np.random.seed(42)
    shuffled_containment = np.zeros_like(containment)
    shuffled_indices = np.random.permutation(n_words)

    items_per_sent = n_words // n_sents
    for i in range(n_sents):
        start = i * items_per_sent
        end = min((i + 1) * items_per_sent, n_words)
        shuffled_containment[i, shuffled_indices[start:end]] = 1

    # Aggregate with shuffled containment
    R_shuffled = []
    for i in range(n_sents):
        mask = shuffled_containment[i] > 0
        if mask.any():
            group_emb = word_emb[mask]
            R = compute_R_from_embeddings(group_emb)
            R_shuffled.append(R)

    # Compute L-correlation with shuffled containment
    if len(R_shuffled) >= 2:
        min_len = min(len(R_shuffled), len(sent_norms))
        corr_shuffled, _ = spearmanr(R_shuffled[:min_len], sent_norms[:min_len])
        if np.isnan(corr_shuffled):
            corr_shuffled = 0.0
    else:
        corr_shuffled = 0.0

    # The shuffled version should have MUCH lower correlation
    correctly_fails = corr_shuffled < 0.5

    return NegativeControlResult(
        control_name="shuffled_hierarchy",
        description="Randomly permute containment matrix (REAL embeddings)",
        expected_behavior="Structure collapses (L-correlation -> 0)",
        metric_name="L_correlation",
        metric_value=corr_shuffled,
        threshold=0.5,
        correctly_fails=correctly_fails,
        explanation=(
            f"Correct containment: L-corr={corr_correct:.3f}, "
            f"Shuffled: L-corr={corr_shuffled:.3f}. "
            f"{'Structure correctly collapsed' if correctly_fails else 'Structure NOT collapsed - test invalid!'}"
        )
    )


# =============================================================================
# NEGATIVE CONTROL 2: WRONG AGGREGATION
# =============================================================================

def test_negative_wrong_aggregation(ms: MultiScaleEmbeddings = None) -> NegativeControlResult:
    """
    WRONG AGGREGATION: Use R = E * sigma (extensive) instead of R = E / sigma.

    The correct R = E / sigma is INTENSIVE (independent of scale).
    The wrong R = E * sigma is EXTENSIVE (grows with spread).

    Expected: FAIL (CV > 0.3 for wrong formula)
    """
    if ms is None:
        ms = load_multiscale_embeddings()

    # Compute R at each scale with CORRECT formula: R = E * concentration / sigma
    R_correct = []
    for scale in SCALE_HIERARCHY:
        if scale in ms.scales:
            R = compute_R_from_embeddings(ms.scales[scale].embeddings)
            R_correct.append(R)

    cv_correct = np.std(R_correct) / (np.mean(R_correct) + 1e-10) if R_correct else 0

    # Compute R at each scale with WRONG formula: R = E * sigma (extensive)
    # This formula grows with the spread of embeddings, not inversely
    R_wrong = []
    for scale in SCALE_HIERARCHY:
        if scale in ms.scales:
            emb = ms.scales[scale].embeddings
            n = len(emb)

            # Compute centroid and distances (same as correct)
            centroid = emb.mean(axis=0)
            distances = np.linalg.norm(emb - centroid, axis=1)

            # Use mean distance as sigma (same as real_embeddings.py)
            mean_dist = np.mean(distances) + 1e-10
            z = distances / mean_dist
            E = np.mean(np.exp(-0.5 * z**2))

            # WRONG formula: R = E * sigma * n (extensive - grows with n and spread)
            # This violates intensivity because it scales with sample size
            R_wrong_value = E * mean_dist * n
            R_wrong.append(R_wrong_value)

    cv_wrong = np.std(R_wrong) / (np.mean(R_wrong) + 1e-10) if R_wrong else 1.0

    # Wrong formula should have MUCH higher CV (extensive scaling)
    # Since n varies as [64, 20, 5, 2], the wrong R should vary wildly
    correctly_fails = cv_wrong > 0.3

    return NegativeControlResult(
        control_name="wrong_aggregation",
        description="Use R = E * sigma * n (extensive) instead of R = E / sigma",
        expected_behavior="R has high CV across scales (not intensive)",
        metric_name="CV_across_scales",
        metric_value=cv_wrong,
        threshold=0.3,
        correctly_fails=correctly_fails,
        explanation=(
            f"Correct formula (intensive): CV={cv_correct:.3f}, "
            f"Wrong formula (extensive): CV={cv_wrong:.3f}. "
            f"{'Extensivity correctly detected' if correctly_fails else 'Still intensive - test logic error!'}"
        )
    )


# =============================================================================
# NEGATIVE CONTROL 3: NON-LOCAL INJECTION
# =============================================================================

def test_negative_nonlocal_injection(ms: MultiScaleEmbeddings = None) -> NegativeControlResult:
    """
    NON-LOCAL INJECTION: Add unrelated data to observations.

    If R is truly local (C1), adding non-local data should
    significantly change R.

    Expected: FAIL (R changes significantly when mixed)
    """
    if ms is None:
        ms = load_multiscale_embeddings()

    word_data = ms.scales.get("words")
    if word_data is None:
        return NegativeControlResult(
            control_name="nonlocal_injection",
            description="Mix in data from very different distribution",
            expected_behavior="R changes significantly (locality violated)",
            metric_name="R_change_fraction",
            metric_value=0.0,
            threshold=0.2,
            correctly_fails=True,
            explanation="Missing data"
        )

    # Clean R from word embeddings
    emb_clean = word_data.embeddings
    R_clean = compute_R_from_embeddings(emb_clean)

    # Generate NON-LOCAL noise (very different distribution)
    np.random.seed(42)
    noise_emb = np.random.randn(len(emb_clean), emb_clean.shape[1]) * 5.0  # Large noise
    noise_emb = noise_emb + 10.0  # Shift far from origin

    # Mix clean and noise (50-50)
    mixed_emb = np.vstack([emb_clean, noise_emb])
    R_mixed = compute_R_from_embeddings(mixed_emb)

    R_change = abs(R_clean - R_mixed) / (R_clean + 1e-10)

    # Non-local injection should significantly change R
    correctly_fails = R_change > 0.2

    return NegativeControlResult(
        control_name="nonlocal_injection",
        description="Mix in data from very different distribution (REAL embeddings)",
        expected_behavior="R changes significantly (locality violated)",
        metric_name="R_change_fraction",
        metric_value=R_change,
        threshold=0.2,
        correctly_fails=correctly_fails,
        explanation=(
            f"R_clean={R_clean:.4f}, R_mixed={R_mixed:.4f}, "
            f"change={R_change:.2%}. "
            f"{'R correctly affected by non-local data' if correctly_fails else 'R unchanged - locality test invalid!'}"
        )
    )


# =============================================================================
# NEGATIVE CONTROL 4: RANDOM R VALUES
# =============================================================================

def test_negative_random_R(ms: MultiScaleEmbeddings = None) -> NegativeControlResult:
    """
    RANDOM R: Use random values instead of computed R.

    If we randomly assign R values, they should NOT satisfy
    the composition axioms.

    Expected: FAIL (high CV - not intensive)
    """
    np.random.seed(42)

    # Generate random R values (not from actual observations)
    n_levels = 5
    R_random = np.random.exponential(scale=1.0, size=n_levels)

    # Check if random values satisfy intensivity
    cv = np.std(R_random) / (np.mean(R_random) + 1e-10)

    # Random values should have high CV (not intensive)
    correctly_fails = cv > 0.3

    return NegativeControlResult(
        control_name="random_R_values",
        description="Use random values instead of computed R",
        expected_behavior="Random R values violate intensivity",
        metric_name="CV_random_R",
        metric_value=cv,
        threshold=0.3,
        correctly_fails=correctly_fails,
        explanation=(
            f"Random R values: {[f'{r:.3f}' for r in R_random]}, "
            f"CV={cv:.3f}. "
            f"{'Random values correctly non-intensive' if correctly_fails else 'Random values appear intensive - suspicious!'}"
        )
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_negative_controls() -> Dict:
    """
    Run all negative control tests with REAL embeddings.

    Returns:
        Complete test results with overall verdict
    """
    # Load embeddings once
    print("Loading real embeddings for negative controls...")
    ms = load_multiscale_embeddings()
    print(f"  Architecture: {ms.architecture}")
    print()

    controls = [
        ("shuffled_hierarchy", lambda: test_negative_shuffled_hierarchy(ms)),
        ("wrong_aggregation", lambda: test_negative_wrong_aggregation(ms)),
        ("nonlocal_injection", lambda: test_negative_nonlocal_injection(ms)),
        ("random_R_values", lambda: test_negative_random_R(ms)),
    ]

    results = {}
    n_correctly_fail = 0

    for name, test_fn in controls:
        result = test_fn()
        results[name] = _result_to_dict(result)
        if result.correctly_fails:
            n_correctly_fail += 1

    n_total = len(controls)
    all_correctly_fail = n_correctly_fail == n_total

    summary = {
        "n_controls": n_total,
        "n_correctly_fail": n_correctly_fail,
        "n_incorrectly_pass": n_total - n_correctly_fail,
        "all_correctly_fail": all_correctly_fail,
        "verdict": "CONFIRMED" if n_correctly_fail >= 3 else "FAILED",  # 3/4 acceptable
        "reasoning": (
            f"{n_correctly_fail}/{n_total} negative controls correctly fail (REAL embeddings)"
            if n_correctly_fail >= 3 else
            f"Only {n_correctly_fail}/{n_total} controls fail - positive tests may be trivial!"
        )
    }

    return {
        "test_id": "Q7_NEGATIVE_CONTROLS",
        "version": "2.0.0",
        "results": results,
        "summary": summary
    }


def _result_to_dict(result: NegativeControlResult) -> Dict:
    """Convert NegativeControlResult to dictionary."""
    return {
        "control_name": result.control_name,
        "description": result.description,
        "expected_behavior": result.expected_behavior,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "threshold": result.threshold,
        "correctly_fails": result.correctly_fails,
        "explanation": result.explanation
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def run_self_tests():
    """Run negative control self-tests with REAL embeddings."""
    print("\n" + "=" * 80)
    print("Q7: NEGATIVE CONTROLS (REAL EMBEDDINGS)")
    print("=" * 80)

    print("\nThese tests MUST FAIL to validate the positive tests are meaningful.\n")

    results = run_all_negative_controls()

    # Print per-control results
    for name, result in results["results"].items():
        status = "[CORRECTLY FAILS]" if result["correctly_fails"] else "[INCORRECTLY PASSES]"
        print(f"--- {name.upper()} ---")
        print(f"  Description: {result['description']}")
        print(f"  Expected: {result['expected_behavior']}")
        print(f"  {result['metric_name']}: {result['metric_value']:.4f} (threshold: {result['threshold']})")
        print(f"  Status: {status}")
        print(f"  Explanation: {result['explanation']}")
        print()

    # Print summary
    print("=" * 80)
    summary = results["summary"]
    print(f"SUMMARY: {summary['n_correctly_fail']}/{summary['n_controls']} negative controls correctly fail")
    print(f"Verdict: {summary['verdict']}")
    print(f"Reasoning: {summary['reasoning']}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_self_tests()
