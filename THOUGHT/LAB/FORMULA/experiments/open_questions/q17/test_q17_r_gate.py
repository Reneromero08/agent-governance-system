#!/usr/bin/env python3
"""
Q17: Governance Gating - Empirical Validation

Tests the core claims:
1. R computation: High agreement → high R, low agreement → low R
2. Volume resistance: More low-quality obs doesn't increase R
3. Threshold discrimination: Tiers correctly separate scenarios
4. Real embeddings: Works with actual sentence-transformers

Run: python test_q17_r_gate.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from r_gate import RGate, RResult, GateDecision, ActionTier, GateStatus, create_mock_embedder

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[WARN] sentence-transformers not available, using mock embedder")


# =============================================================================
# TEST DATA
# =============================================================================

# High agreement: Same semantic content, different words
HIGH_AGREEMENT_SET = [
    "The capital of France is Paris.",
    "Paris is the capital city of France.",
    "France's capital is Paris.",
    "Paris serves as France's capital.",
    "The French capital is Paris.",
]

# Low agreement: Completely different topics
LOW_AGREEMENT_SET = [
    "The capital of France is Paris.",
    "Python is a programming language.",
    "The weather is sunny today.",
    "Quantum mechanics studies particles.",
    "Pizza originated in Italy.",
]

# Medium agreement: Related but different aspects
MEDIUM_AGREEMENT_SET = [
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "The Eiffel Tower is in Paris.",
    "French cuisine is world-famous.",
    "Paris has many museums.",
]

# Echo chamber: Identical or near-identical
ECHO_CHAMBER_SET = [
    "The answer is 42.",
    "The answer is 42.",
    "The answer is 42.",
    "The answer is 42.",
    "The answer is 42.",
]

# Noisy low-quality observations
NOISY_SET = [
    "asdfgh jklzxc vbnm qwerty",
    "random noise text here xyz",
    "gibberish words foo bar baz",
    "meaningless string abc 123",
    "nonsense placeholder text",
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_r_ordering(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 1: R values should order correctly.

    Expected: R(high_agreement) > R(medium) > R(low_agreement)
    """
    r_high = gate.compute_r(HIGH_AGREEMENT_SET)
    r_medium = gate.compute_r(MEDIUM_AGREEMENT_SET)
    r_low = gate.compute_r(LOW_AGREEMENT_SET)

    # Check ordering
    ordering_correct = (r_high.R > r_medium.R > r_low.R)

    result = {
        "test": "R_ORDERING",
        "R_high_agreement": r_high.R,
        "R_medium_agreement": r_medium.R,
        "R_low_agreement": r_low.R,
        "E_high": r_high.E,
        "E_medium": r_medium.E,
        "E_low": r_low.E,
        "sigma_high": r_high.sigma,
        "sigma_medium": r_medium.sigma,
        "sigma_low": r_low.sigma,
        "ordering_correct": ordering_correct,
        "pass": ordering_correct
    }

    return ordering_correct, result


def test_volume_resistance(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 2: Volume resistance - adding more noisy observations shouldn't increase R.

    This tests the INTENSIVE property (Q15).
    """
    # Start with low-agreement observations
    base_obs = LOW_AGREEMENT_SET[:3]
    r_base = gate.compute_r(base_obs)

    # Add more noisy observations
    r_values = [r_base.R]
    n_values = [len(base_obs)]

    for i in range(5):
        extended = base_obs + NOISY_SET[:i+1]
        r_ext = gate.compute_r(extended)
        r_values.append(r_ext.R)
        n_values.append(len(extended))

    # R should NOT increase with more noisy observations
    # Allow small fluctuations, but overall trend should be flat or decreasing
    r_increased = r_values[-1] > r_values[0] * 1.5  # More than 50% increase = fail

    result = {
        "test": "VOLUME_RESISTANCE",
        "n_observations": n_values,
        "R_values": r_values,
        "R_initial": r_values[0],
        "R_final": r_values[-1],
        "R_change_pct": ((r_values[-1] - r_values[0]) / (r_values[0] + 1e-8)) * 100,
        "volume_resistant": not r_increased,
        "pass": not r_increased
    }

    return not r_increased, result


def test_echo_chamber_detection(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 3: Echo chambers should produce suspiciously high R.

    R > 10 indicates potential echo chamber (all identical).
    """
    r_echo = gate.compute_r(ECHO_CHAMBER_SET)
    r_normal = gate.compute_r(HIGH_AGREEMENT_SET)

    # Echo chamber should have MUCH higher R (sigma → 0)
    echo_detected = r_echo.R > r_normal.R * 2  # At least 2x normal high-agreement

    result = {
        "test": "ECHO_CHAMBER_DETECTION",
        "R_echo_chamber": r_echo.R,
        "R_normal_high_agreement": r_normal.R,
        "ratio": r_echo.R / (r_normal.R + 1e-8),
        "sigma_echo": r_echo.sigma,
        "sigma_normal": r_normal.sigma,
        "echo_detectable": echo_detected,
        "pass": echo_detected
    }

    return echo_detected, result


def test_threshold_discrimination(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 4: Thresholds should correctly discriminate.

    High agreement → passes T2 (0.8)
    Low agreement → fails T2 (0.8)
    """
    # High agreement should pass T2
    decision_high = gate.check(HIGH_AGREEMENT_SET, ActionTier.T2_PERSISTENT)

    # Low agreement should fail T2
    decision_low = gate.check(LOW_AGREEMENT_SET, ActionTier.T2_PERSISTENT)

    # Discrimination works if high passes and low fails
    discriminates = (
        decision_high.status == GateStatus.OPEN and
        decision_low.status == GateStatus.CLOSED
    )

    result = {
        "test": "THRESHOLD_DISCRIMINATION",
        "high_agreement_R": decision_high.R,
        "high_agreement_status": decision_high.status.value,
        "low_agreement_R": decision_low.R,
        "low_agreement_status": decision_low.status.value,
        "threshold_T2": 0.8,
        "discriminates_correctly": discriminates,
        "pass": discriminates
    }

    return discriminates, result


def test_tier_classification(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 5: Action tier classification works correctly.
    """
    test_cases = [
        ("read", "file.txt", ActionTier.T0_READ),
        ("search", "database", ActionTier.T0_READ),
        ("stage", "changes", ActionTier.T1_REVERSIBLE),
        ("draft", "email", ActionTier.T1_REVERSIBLE),
        ("write", "output.txt", ActionTier.T2_PERSISTENT),
        ("commit", "feature-branch", ActionTier.T2_PERSISTENT),
        ("deploy", "production", ActionTier.T3_CRITICAL),
        ("delete", "canon/file.md", ActionTier.T3_CRITICAL),
        ("write", "main", ActionTier.T3_CRITICAL),  # Target escalates
    ]

    correct = 0
    details = []
    for action, target, expected in test_cases:
        actual = gate.classify_tier(action, target)
        is_correct = actual == expected
        correct += int(is_correct)
        details.append({
            "action": action,
            "target": target,
            "expected": expected.name,
            "actual": actual.name,
            "correct": is_correct
        })

    all_correct = correct == len(test_cases)

    result = {
        "test": "TIER_CLASSIFICATION",
        "correct": correct,
        "total": len(test_cases),
        "accuracy": correct / len(test_cases),
        "details": details,
        "pass": all_correct
    }

    return all_correct, result


def test_minimum_observations(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 6: R computation with edge cases.
    """
    # 0 observations
    r_zero = gate.compute_r([])

    # 1 observation
    r_one = gate.compute_r(["single observation"])

    # 2 observations (minimum for pairwise)
    r_two = gate.compute_r(["observation one", "observation two"])

    edge_cases_handled = (
        r_zero.R == 0.0 and
        r_one.R == 0.0 and
        r_two.R != 0.0 and not np.isinf(r_two.R)
    )

    result = {
        "test": "MINIMUM_OBSERVATIONS",
        "R_zero_obs": r_zero.R,
        "R_one_obs": r_one.R,
        "R_two_obs": r_two.R,
        "zero_handled": r_zero.R == 0.0,
        "one_handled": r_one.R == 0.0,
        "two_valid": r_two.R != 0.0 and not np.isinf(r_two.R),
        "pass": edge_cases_handled
    }

    return edge_cases_handled, result


def test_real_embeddings(gate_real: RGate) -> Tuple[bool, dict]:
    """
    TEST 7: Real sentence-transformer embeddings.

    Only runs if sentence-transformers is available.
    """
    if not ST_AVAILABLE:
        return True, {"test": "REAL_EMBEDDINGS", "skipped": True, "reason": "sentence-transformers not available", "pass": True}

    # Test with real embeddings
    r_high = gate_real.compute_r(HIGH_AGREEMENT_SET)
    r_low = gate_real.compute_r(LOW_AGREEMENT_SET)

    # Real embeddings should show clear separation
    separation = r_high.R > r_low.R

    # And high agreement should have high E
    high_E_valid = r_high.E > 0.7  # Semantically similar should have E > 0.7

    result = {
        "test": "REAL_EMBEDDINGS",
        "model": "all-MiniLM-L6-v2",
        "R_high_agreement": r_high.R,
        "R_low_agreement": r_low.R,
        "E_high": r_high.E,
        "E_low": r_low.E,
        "sigma_high": r_high.sigma,
        "sigma_low": r_low.sigma,
        "separation_clear": separation,
        "high_E_valid": high_E_valid,
        "pass": separation and high_E_valid
    }

    return separation and high_E_valid, result


def test_gate_integration(gate: RGate) -> Tuple[bool, dict]:
    """
    TEST 8: Full gate integration test.

    Simulates real agent decision flow.
    """
    scenarios = []

    # Scenario 1: Agent wants to commit with high-quality consensus
    decision = gate.check(HIGH_AGREEMENT_SET, ActionTier.T2_PERSISTENT)
    scenarios.append({
        "scenario": "commit_high_consensus",
        "expected": "OPEN",
        "actual": decision.status.value,
        "R": decision.R,
        "correct": decision.status == GateStatus.OPEN
    })

    # Scenario 2: Agent wants to commit with poor consensus
    decision = gate.check(LOW_AGREEMENT_SET, ActionTier.T2_PERSISTENT)
    scenarios.append({
        "scenario": "commit_poor_consensus",
        "expected": "CLOSED",
        "actual": decision.status.value,
        "R": decision.R,
        "correct": decision.status == GateStatus.CLOSED
    })

    # Scenario 3: Agent wants to deploy (T3) - needs R > 1.0
    decision = gate.check(HIGH_AGREEMENT_SET, ActionTier.T3_CRITICAL)
    # High agreement may or may not pass T3 depending on R value
    t3_makes_sense = (decision.status == GateStatus.OPEN and decision.R >= 1.0) or \
                     (decision.status == GateStatus.CLOSED and decision.R < 1.0)
    scenarios.append({
        "scenario": "deploy_high_consensus",
        "threshold": 1.0,
        "actual": decision.status.value,
        "R": decision.R,
        "correct": t3_makes_sense
    })

    # Scenario 4: Read action always passes
    decision = gate.check([], ActionTier.T0_READ)
    scenarios.append({
        "scenario": "read_no_observations",
        "expected": "OPEN",
        "actual": decision.status.value,
        "correct": decision.status == GateStatus.OPEN
    })

    all_correct = all(s["correct"] for s in scenarios)

    result = {
        "test": "GATE_INTEGRATION",
        "scenarios": scenarios,
        "all_correct": all_correct,
        "pass": all_correct
    }

    return all_correct, result


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests and produce report."""
    print("=" * 70)
    print("Q17: GOVERNANCE GATING - EMPIRICAL VALIDATION")
    print("=" * 70)
    print()

    # Initialize gates
    mock_embed = create_mock_embedder(dim=384, seed=42)
    gate_mock = RGate(embed_fn=mock_embed)

    gate_real = None
    if ST_AVAILABLE:
        print("Loading sentence-transformers model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        gate_real = RGate(embed_fn=lambda x: model.encode(x, normalize_embeddings=True))
        print("Model loaded.\n")
    else:
        print("Using mock embedder (sentence-transformers not available)\n")

    # Run tests
    tests = [
        ("R_ORDERING", lambda: test_r_ordering(gate_real or gate_mock)),
        ("VOLUME_RESISTANCE", lambda: test_volume_resistance(gate_real or gate_mock)),
        ("ECHO_CHAMBER", lambda: test_echo_chamber_detection(gate_real or gate_mock)),
        ("THRESHOLD_DISCRIMINATION", lambda: test_threshold_discrimination(gate_real or gate_mock)),
        ("TIER_CLASSIFICATION", lambda: test_tier_classification(gate_mock)),  # Doesn't need embeddings
        ("MINIMUM_OBSERVATIONS", lambda: test_minimum_observations(gate_real or gate_mock)),
        ("REAL_EMBEDDINGS", lambda: test_real_embeddings(gate_real) if gate_real else (True, {"skipped": True, "pass": True})),
        ("GATE_INTEGRATION", lambda: test_gate_integration(gate_real or gate_mock)),
    ]

    results = []
    passed = 0
    failed = 0

    print("-" * 70)
    print(f"{'Test':<30} | {'Status':<10} | {'Details'}")
    print("-" * 70)

    for test_name, test_fn in tests:
        try:
            success, result = test_fn()
            results.append(result)

            if result.get("skipped"):
                status = "SKIP"
            elif success:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            # Print summary line
            detail = ""
            if "R_high_agreement" in result:
                detail = f"R_high={result['R_high_agreement']:.3f}, R_low={result.get('R_low_agreement', 'N/A'):.3f}" if isinstance(result.get('R_low_agreement'), float) else f"R_high={result['R_high_agreement']:.3f}"
            elif "R_values" in result:
                detail = f"R_init={result['R_initial']:.3f}, R_final={result['R_final']:.3f}"
            elif "accuracy" in result:
                detail = f"accuracy={result['accuracy']:.1%}"
            elif "all_correct" in result:
                detail = f"scenarios={len(result.get('scenarios', []))}"

            print(f"{test_name:<30} | {status:<10} | {detail}")

        except Exception as e:
            print(f"{test_name:<30} | ERROR      | {str(e)[:40]}")
            failed += 1
            results.append({"test": test_name, "error": str(e), "pass": False})

    print("-" * 70)
    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    print()

    # Detailed results
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\n### {result.get('test', 'UNKNOWN')}")
        print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2, default=str))

    # Final verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if failed == 0:
        print("\n** ALL TESTS PASS - R-GATING VALIDATED **")
        print("\nKey findings:")
        for r in results:
            if r.get("test") == "R_ORDERING" and not r.get("skipped"):
                print(f"  - R ordering correct: high({r['R_high_agreement']:.3f}) > med({r['R_medium_agreement']:.3f}) > low({r['R_low_agreement']:.3f})")
            if r.get("test") == "VOLUME_RESISTANCE" and not r.get("skipped"):
                print(f"  - Volume resistant: R change = {r['R_change_pct']:.1f}%")
            if r.get("test") == "ECHO_CHAMBER_DETECTION" and not r.get("skipped"):
                print(f"  - Echo chamber detectable: ratio = {r['ratio']:.1f}x")
    else:
        print(f"\n** {failed} TESTS FAILED - CLAIMS NOT VALIDATED **")

    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "embedder": "sentence-transformers" if ST_AVAILABLE else "mock",
        "model": "all-MiniLM-L6-v2" if ST_AVAILABLE else "hash-based-mock",
        "passed": passed,
        "failed": failed,
        "results": results,
        "verdict": "VALIDATED" if failed == 0 else "NOT_VALIDATED"
    }

    output_path = Path(__file__).parent / "q17_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
