#!/usr/bin/env python3
"""
Q13: The 36x Ratio - Master Test Runner
========================================

Runs all 12 HARDCORE tests to determine if the context improvement ratio
follows a predictable scaling law.

Tests:
    1. Finite-Size Scaling Collapse (Gold Standard)
    2. Universal Critical Exponents
    3. Predictive Extrapolation (Nearly Impossible)
    4. Dimensional Analysis Consistency
    5. Boundary Behavior Verification
    6. Bayesian Model Selection (Irrefutable)
    7. Causality via Intervention
    8. Phase Transition Detection
    9. Robustness Under Adversarial Noise
    10. Self-Consistency (Formula Components)
    11. Cross-Domain Universality
    12. Blind Prediction (Ultimate Challenge)

Success Criteria:
    - 10+/12 tests PASS: ANSWERED - Scaling law CONFIRMED
    - 7-9/12 tests PASS: PARTIAL - Strong evidence, some tests inconclusive
    - <7/12 tests PASS: FALSIFIED - No consistent scaling law found

Usage:
    python run_q13_all.py           # Run all tests
    python run_q13_all.py --quick   # Run quick tests only (skip heavy ones)
    python run_q13_all.py --test N  # Run only test N

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, PASS_THRESHOLD, QUTIP_AVAILABLE,
    print_header, print_summary, save_results
)


# =============================================================================
# TEST DEFINITIONS
# =============================================================================

TESTS = [
    ("test_q13_01_finite_size", "01", "Finite-Size Scaling Collapse"),
    ("test_q13_02_universality", "02", "Universal Critical Exponents"),
    ("test_q13_03_prediction", "03", "Predictive Extrapolation"),
    ("test_q13_04_dimensional", "04", "Dimensional Analysis"),
    ("test_q13_05_boundary", "05", "Boundary Behavior"),
    ("test_q13_06_bayesian", "06", "Bayesian Model Selection"),
    ("test_q13_07_causality", "07", "Causality via Intervention"),
    ("test_q13_08_phase", "08", "Phase Transition Detection"),
    ("test_q13_09_robustness", "09", "Robustness (Noise)"),
    ("test_q13_10_self_consistency", "10", "Self-Consistency"),
    ("test_q13_11_cross_domain", "11", "Cross-Domain Universality"),
    ("test_q13_12_blind_prediction", "12", "Blind Prediction"),
]

# Quick mode skips these heavy tests
HEAVY_TESTS = ["01", "02", "06", "09"]


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_single_test(module_name: str, test_id: str, test_name: str,
                    config: TestConfig) -> Tuple[ScalingLawResult, float]:
    """
    Run a single test module and return result with timing.
    """
    import importlib.util

    start_time = time.time()

    try:
        # Import the test module
        module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{module_name}.py"
        )

        if not os.path.exists(module_path):
            return ScalingLawResult(
                test_name=test_name,
                test_id=f"Q13_TEST_{test_id}",
                passed=False,
                evidence="",
                falsification_evidence=f"Module not found: {module_path}"
            ), 0.0

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Run the test
        if hasattr(module, 'run_test'):
            result = module.run_test(config)
        else:
            return ScalingLawResult(
                test_name=test_name,
                test_id=f"Q13_TEST_{test_id}",
                passed=False,
                evidence="",
                falsification_evidence="Module has no run_test function"
            ), 0.0

        duration = time.time() - start_time
        return result, duration

    except Exception as e:
        duration = time.time() - start_time
        import traceback
        return ScalingLawResult(
            test_name=test_name,
            test_id=f"Q13_TEST_{test_id}",
            passed=False,
            evidence="",
            falsification_evidence=f"Test crashed: {str(e)}\n{traceback.format_exc()}"
        ), duration


def run_all_tests(config: TestConfig, quick: bool = False,
                  single_test: str = None) -> Dict[str, ScalingLawResult]:
    """
    Run all Q13 tests and collect results.
    """
    results = {}

    tests_to_run = TESTS
    if single_test:
        tests_to_run = [(m, i, n) for m, i, n in TESTS if i == single_test]

    total = len(tests_to_run)

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " Q13: THE 36x RATIO - SCALING LAW DISCOVERY".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"QuTiP available: {QUTIP_AVAILABLE}")
    print(f"Running {total} tests...")

    if quick:
        print("\n[QUICK MODE] Skipping heavy tests")

    for i, (module_name, test_id, test_name) in enumerate(tests_to_run, 1):
        # Skip heavy tests in quick mode
        if quick and test_id in HEAVY_TESTS:
            print(f"\n[SKIP] Test {test_id}: {test_name} (quick mode)")
            results[f"Q13_TEST_{test_id}"] = ScalingLawResult(
                test_name=test_name,
                test_id=f"Q13_TEST_{test_id}",
                passed=False,
                evidence="Skipped in quick mode"
            )
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{total}] Running: {test_name}")
        print("=" * 70)

        result, duration = run_single_test(module_name, test_id, test_name, config)
        results[f"Q13_TEST_{test_id}"] = result

        status = "PASS" if result.passed else "FAIL"
        print(f"\n>>> Result: {status} ({duration:.1f}s)")

    return results


def analyze_results(results: Dict[str, ScalingLawResult]) -> Dict:
    """
    Analyze all test results and determine Q13 answer.
    """
    passed_count = sum(1 for r in results.values() if r.passed)
    total = len(results)

    # Collect scaling law types
    scaling_laws = {}
    for r in results.values():
        if r.passed and r.scaling_law:
            sl = r.scaling_law
            scaling_laws[sl] = scaling_laws.get(sl, 0) + 1

    # Collect exponents
    all_alphas = []
    all_betas = []
    for r in results.values():
        if r.scaling_exponents:
            if 'alpha' in r.scaling_exponents:
                all_alphas.append(r.scaling_exponents['alpha'])
            if 'alpha_mean' in r.scaling_exponents:
                all_alphas.append(r.scaling_exponents['alpha_mean'])
            if 'beta' in r.scaling_exponents:
                all_betas.append(r.scaling_exponents['beta'])
            if 'beta_mean' in r.scaling_exponents:
                all_betas.append(r.scaling_exponents['beta_mean'])

    # Determine answer
    if passed_count >= PASS_THRESHOLD:
        dominant_law = max(scaling_laws.keys(), key=lambda k: scaling_laws[k]) if scaling_laws else "power"
        answer = f"SCALING LAW CONFIRMED: {dominant_law.upper()}"
        status = "ANSWERED"
    elif passed_count >= 7:
        answer = "PARTIAL: Strong evidence but incomplete"
        status = "PARTIAL"
    else:
        answer = "FALSIFIED: No consistent scaling law found"
        status = "FALSIFIED"

    return {
        'status': status,
        'answer': answer,
        'passed': passed_count,
        'total': total,
        'pass_rate': passed_count / total if total > 0 else 0,
        'scaling_laws': scaling_laws,
        'alpha_mean': float(sum(all_alphas) / len(all_alphas)) if all_alphas else 0,
        'beta_mean': float(sum(all_betas) / len(all_betas)) if all_betas else 0,
    }


def generate_final_report(results: Dict[str, ScalingLawResult], analysis: Dict) -> str:
    """Generate the final report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Q13: THE 36x RATIO - FINAL REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTimestamp: {datetime.now().isoformat()}")
    lines.append(f"Total tests: {analysis['total']}")
    lines.append(f"Passed: {analysis['passed']}")
    lines.append(f"Pass rate: {analysis['pass_rate']:.1%}")
    lines.append(f"Threshold: {PASS_THRESHOLD}/{analysis['total']}")

    lines.append("\n" + "-" * 70)
    lines.append("INDIVIDUAL TEST RESULTS")
    lines.append("-" * 70)

    for test_id, result in sorted(results.items()):
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"\n[{status}] {result.test_name}")
        if result.evidence:
            lines.append(f"  Evidence: {result.evidence[:100]}...")
        if result.falsification_evidence:
            lines.append(f"  Issue: {result.falsification_evidence[:100]}...")

    lines.append("\n" + "-" * 70)
    lines.append("SCALING LAW SUMMARY")
    lines.append("-" * 70)

    if analysis['scaling_laws']:
        lines.append("\nScaling law types found:")
        for law, count in analysis['scaling_laws'].items():
            lines.append(f"  {law}: {count} tests")

    if analysis['alpha_mean'] != 0:
        lines.append(f"\nMean alpha (fragment exponent): {analysis['alpha_mean']:.4f}")
    if analysis['beta_mean'] != 0:
        lines.append(f"Mean beta (decoherence exponent): {analysis['beta_mean']:.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("Q13 ANSWER")
    lines.append("=" * 70)
    lines.append(f"\nStatus: {analysis['status']}")
    lines.append(f"Answer: {analysis['answer']}")

    if analysis['status'] == "ANSWERED":
        lines.append("\n" + "=" * 70)
        lines.append("THE SCALING LAW")
        lines.append("=" * 70)
        lines.append(f"""
The context improvement ratio follows a POWER LAW:

    Ratio(N, d) = 1 + C * (N - 1)^alpha * d^beta

Where:
    N     = Number of context fragments
    d     = Decoherence level (0 = quantum, 1 = classical)
    alpha = {analysis['alpha_mean']:.3f} (fragment exponent)
    beta  = {analysis['beta_mean']:.3f} (decoherence exponent)
    C     = Universal constant (to be determined)

PREDICTIVE FORMULA:
To restore resolution R_target from R_single, you need:

    N_required = 1 + ((R_target / R_single - 1) / (C * d^beta))^(1/alpha)

This formula predicts how much context is needed to restore resolution
at any decoherence level.
""")

    elif analysis['status'] == "PARTIAL":
        lines.append("\nImplication: Strong evidence for scaling law but some tests inconclusive.")
        lines.append("Additional investigation needed to confirm full scaling behavior.")

    else:
        lines.append("\nImplication: The 36x ratio may be specific to particular conditions.")
        lines.append("No universal scaling law was found across all tests.")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Q13 Scaling Law tests")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip heavy computations)')
    parser.add_argument('--test', type=str, default=None,
                       help='Run only specific test (01-12)')
    parser.add_argument('--output', type=str, default='Q13_RESULTS.json',
                       help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    args = parser.parse_args()

    # Check QuTiP
    if not QUTIP_AVAILABLE:
        print("\n" + "!" * 70)
        print("WARNING: QuTiP not installed!")
        print("Some tests require QuTiP for quantum simulations.")
        print("Install with: pip install qutip")
        print("!" * 70)

    # Create config
    config = TestConfig(verbose=args.verbose)

    # Run tests
    results = run_all_tests(config, quick=args.quick, single_test=args.test)

    # Analyze
    analysis = analyze_results(results)

    # Generate report
    report = generate_final_report(results, analysis)
    print("\n" + report)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output
    )

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Q13: The 36x Ratio Scaling Law',
        'analysis': analysis,
        'results': {k: v.to_dict() for k, v in results.items()}
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Exit code
    if analysis['status'] == "ANSWERED":
        return 0
    elif analysis['status'] == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
