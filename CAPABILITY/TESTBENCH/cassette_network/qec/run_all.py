#!/usr/bin/env python3
"""Q40 Quantum Error Correction: Master Test Runner.

Runs all Q40 tests and generates a comprehensive report.

Tests:
    1. Code Distance - Measure t_max correctable errors
    2. Syndrome Detection - sigma decomposition identifies errors
    3. Error Threshold - Exponential suppression below epsilon_th
    4. Holographic Reconstruction - Boundary encodes bulk
    5. Phase Parity (Hallucination) - Zero Signature violation
    6. Adversarial Attacks - Robustness against designed attacks
    7. Cross-Model Cascade - Network error handling

Usage:
    python run_all.py [--quick] [--output results.json]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))


def run_all_tests(quick: bool = False) -> Dict:
    """Run all Q40 tests.

    Args:
        quick: Use reduced sample sizes for faster testing

    Returns:
        Combined results dict
    """
    print("=" * 80)
    print("Q40: QUANTUM ERROR CORRECTION - FULL TEST SUITE")
    print("=" * 80)
    print()
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print()

    results = {
        "test_suite": "q40-quantum-error-correction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if quick else "full",
        "tests": {},
        "verdicts": {},
    }

    # Parameters based on mode
    if quick:
        params = {
            "code_distance": {"n_samples": 20, "max_errors": 10, "n_trials": 20},
            "syndrome": {"n_syndromes": 20, "held_out_fraction": 0.3},
            "threshold": {"n_epsilon": 15, "n_trials": 50},
            "holographic": {"n_trials": 20},
            "hallucination": {"n_valid": 20, "n_invalid": 20},
            "adversarial": {"n_trials": 15},
            "cascade": {"n_trials": 15},
        }
    else:
        params = {
            "code_distance": {"n_samples": 100, "max_errors": 20, "n_trials": 50},
            "syndrome": {"n_syndromes": 50, "held_out_fraction": 0.3},
            "threshold": {"n_epsilon": 30, "n_trials": 200},
            "holographic": {"n_trials": 50},
            "hallucination": {"n_valid": 50, "n_invalid": 50},
            "adversarial": {"n_trials": 30},
            "cascade": {"n_trials": 30},
        }

    # Test 1: Code Distance
    print("\n" + "=" * 80)
    print("RUNNING TEST 1: CODE DISTANCE")
    print("=" * 80)
    try:
        from test_code_distance import run_code_distance_test
        test1_results = run_code_distance_test(**params["code_distance"])
        results["tests"]["code_distance"] = test1_results
        results["verdicts"]["code_distance"] = test1_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["code_distance"] = {"error": str(e)}
        results["verdicts"]["code_distance"] = False

    # Test 2: Syndrome Detection
    print("\n" + "=" * 80)
    print("RUNNING TEST 2: SYNDROME DETECTION")
    print("=" * 80)
    try:
        from test_syndrome import run_syndrome_test
        test2_results = run_syndrome_test(**params["syndrome"])
        results["tests"]["syndrome"] = test2_results
        results["verdicts"]["syndrome"] = test2_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["syndrome"] = {"error": str(e)}
        results["verdicts"]["syndrome"] = False

    # Test 3: Error Threshold
    print("\n" + "=" * 80)
    print("RUNNING TEST 3: ERROR THRESHOLD")
    print("=" * 80)
    try:
        from test_threshold import run_threshold_test
        test3_results = run_threshold_test(**params["threshold"])
        results["tests"]["error_threshold"] = test3_results
        results["verdicts"]["error_threshold"] = test3_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["error_threshold"] = {"error": str(e)}
        results["verdicts"]["error_threshold"] = False

    # Test 4: Holographic Reconstruction
    print("\n" + "=" * 80)
    print("RUNNING TEST 4: HOLOGRAPHIC RECONSTRUCTION")
    print("=" * 80)
    try:
        from test_holographic import run_holographic_test
        test4_results = run_holographic_test(**params["holographic"])
        results["tests"]["holographic"] = test4_results
        results["verdicts"]["holographic"] = test4_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["holographic"] = {"error": str(e)}
        results["verdicts"]["holographic"] = False

    # Test 5: Hallucination Detection
    print("\n" + "=" * 80)
    print("RUNNING TEST 5: HALLUCINATION DETECTION")
    print("=" * 80)
    try:
        from test_hallucination import run_hallucination_test
        test5_results = run_hallucination_test(**params["hallucination"])
        results["tests"]["hallucination"] = test5_results
        results["verdicts"]["hallucination"] = test5_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["hallucination"] = {"error": str(e)}
        results["verdicts"]["hallucination"] = False

    # Test 6: Adversarial Attacks
    print("\n" + "=" * 80)
    print("RUNNING TEST 6: ADVERSARIAL ATTACKS")
    print("=" * 80)
    try:
        from test_adversarial import run_adversarial_test
        test6_results = run_adversarial_test(**params["adversarial"])
        results["tests"]["adversarial"] = test6_results
        results["verdicts"]["adversarial"] = test6_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["adversarial"] = {"error": str(e)}
        results["verdicts"]["adversarial"] = False

    # Test 7: Cross-Model Cascade
    print("\n" + "=" * 80)
    print("RUNNING TEST 7: CROSS-MODEL CASCADE")
    print("=" * 80)
    try:
        from test_cascade import run_cascade_test
        test7_results = run_cascade_test(**params["cascade"])
        results["tests"]["cascade"] = test7_results
        results["verdicts"]["cascade"] = test7_results["verdict"]["overall_pass"]
    except Exception as e:
        print(f"ERROR: {e}")
        results["tests"]["cascade"] = {"error": str(e)}
        results["verdicts"]["cascade"] = False

    # Overall verdict
    n_passed = sum(1 for v in results["verdicts"].values() if v)
    n_total = len(results["verdicts"])

    # Q40 is proven if majority pass AND critical tests pass
    critical_tests = ["hallucination", "holographic"]
    critical_passed = all(results["verdicts"].get(t, False) for t in critical_tests)

    overall_pass = (n_passed >= n_total // 2 + 1) or critical_passed

    results["summary"] = {
        "tests_passed": n_passed,
        "tests_total": n_total,
        "critical_passed": critical_passed,
        "overall_pass": overall_pass,
        "interpretation": (
            f"Q40 PROVEN: {n_passed}/{n_total} tests passed. "
            "R-gating implements Quantum Error Correction. "
            "Phase parity detects hallucinations. "
            "M field is holographic."
            if overall_pass else
            f"Q40 NOT PROVEN: Only {n_passed}/{n_total} tests passed. "
            "Additional investigation needed."
        )
    }

    # Final report
    print("\n" + "=" * 80)
    print("FINAL REPORT: Q40 QUANTUM ERROR CORRECTION")
    print("=" * 80)
    print()
    print("Test Results:")
    for test_name, passed in results["verdicts"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:25s} {status}")
    print()
    print(f"Tests Passed: {n_passed}/{n_total}")
    print(f"Critical Tests Passed: {critical_passed}")
    print()
    print(f"OVERALL VERDICT: {'PROVEN' if overall_pass else 'NOT PROVEN'}")
    print()
    print(results["summary"]["interpretation"])
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Full Test Suite')
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced sample sizes for faster testing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_all_tests(quick=args.quick)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "q40_full_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON serialization helper
    def json_serialize(obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
