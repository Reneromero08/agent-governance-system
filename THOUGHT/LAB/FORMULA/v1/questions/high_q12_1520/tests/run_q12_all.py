"""
Q12: Phase Transitions in Semantic Systems - Master Test Runner

Runs all 12 HARDCORE tests and generates comprehensive results.

Tests:
    1. Finite-Size Scaling Collapse (Gold Standard)
    2. Universal Critical Exponents
    3. Susceptibility Divergence
    4. Critical Slowing Down
    5. Hysteresis (First vs Second Order)
    6. Order Parameter Jump
    7. Percolation Threshold
    8. Scale Invariance at Criticality
    9. Binder Cumulant Crossing (Most Precise)
    10. Fisher Information Divergence
    11. Spontaneous Symmetry Breaking
    12. Cross-Architecture Universality

Success Criteria:
    - 10+/12 tests PASS: ANSWERED - Phase transition CONFIRMED
    - 7-9/12 tests PASS: PARTIAL - Strong evidence, some tests inconclusive
    - <7/12 tests PASS: FALSIFIED - Not a true phase transition

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
from datetime import datetime
from typing import Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q12_utils import (
    PhaseTransitionTestResult, TestConfig, save_results, print_summary
)

# Import all test modules
from test_q12_01_finite_size_scaling import run_test as run_test_01
from test_q12_02_universality_class import run_test as run_test_02
from test_q12_03_susceptibility import run_test as run_test_03
from test_q12_04_critical_slowing import run_test as run_test_04
from test_q12_05_hysteresis import run_test as run_test_05
from test_q12_06_order_parameter import run_test as run_test_06
from test_q12_07_percolation import run_test as run_test_07
from test_q12_08_scale_invariance import run_test as run_test_08
from test_q12_09_binder_cumulant import run_test as run_test_09
from test_q12_10_fisher_information import run_test as run_test_10
from test_q12_11_symmetry_breaking import run_test as run_test_11
from test_q12_12_cross_architecture import run_test as run_test_12


def run_all_tests(config: TestConfig = None,
                  verbose: bool = True) -> Dict[str, PhaseTransitionTestResult]:
    """
    Run all 12 Q12 tests and return results.
    """
    if config is None:
        config = TestConfig(verbose=verbose)

    results = {}

    tests = [
        ("Q12_TEST_01", "Finite-Size Scaling", run_test_01),
        ("Q12_TEST_02", "Universality Class", run_test_02),
        ("Q12_TEST_03", "Susceptibility", run_test_03),
        ("Q12_TEST_04", "Critical Slowing", run_test_04),
        ("Q12_TEST_05", "Hysteresis", run_test_05),
        ("Q12_TEST_06", "Order Parameter", run_test_06),
        ("Q12_TEST_07", "Percolation", run_test_07),
        ("Q12_TEST_08", "Scale Invariance", run_test_08),
        ("Q12_TEST_09", "Binder Cumulant", run_test_09),
        ("Q12_TEST_10", "Fisher Info", run_test_10),
        ("Q12_TEST_11", "Symmetry Breaking", run_test_11),
        ("Q12_TEST_12", "Cross-Architecture", run_test_12),
    ]

    print("\n" + "=" * 70)
    print("Q12: PHASE TRANSITIONS IN SEMANTIC SYSTEMS")
    print("HARDCORE VALIDATION SUITE")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Running {len(tests)} tests...")

    for i, (test_id, test_name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(tests)}] Running: {test_name}")
        print("=" * 70)

        try:
            result = test_func(config)
            results[test_id] = result
            status = "PASS" if result.passed else "FAIL"
            print(f"\n>>> Result: {status}")
        except Exception as e:
            print(f"\n>>> ERROR: {str(e)}")
            # Create failed result
            results[test_id] = PhaseTransitionTestResult(
                test_name=test_name,
                test_id=test_id,
                passed=False,
                metric_value=0.0,
                threshold=0.0,
                falsification_evidence=f"Test error: {str(e)}"
            )

    return results


def main():
    """Main entry point."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " Q12: PHASE TRANSITIONS - HARDCORE VALIDATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Run all tests
    config = TestConfig(seed=42, verbose=True)
    results = run_all_tests(config)

    # Print summary
    print_summary(results)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Q12_RESULTS.json"
    )
    save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Final verdict
    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if passed >= 10:
        print(f"\n  ** Q12 ANSWERED: Phase transition CONFIRMED **")
        print(f"     {passed}/{total} tests passed")
        print(f"     Truth crystallizes SUDDENLY at critical threshold")
        print(f"     Evidence meets physics-level rigor")
    elif passed >= 7:
        print(f"\n  Q12 PARTIAL: Strong evidence but incomplete")
        print(f"     {passed}/{total} tests passed")
        print(f"     More testing needed for full confirmation")
    else:
        print(f"\n  Q12 FALSIFIED: Not a true phase transition")
        print(f"     {passed}/{total} tests passed")
        print(f"     Semantic change is gradual, not critical")

    print("\n" + "=" * 70)

    # Return exit code
    return 0 if passed >= 7 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
