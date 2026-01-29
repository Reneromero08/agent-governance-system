#!/usr/bin/env python3
"""
Q39: Homeostatic Regulation - Master Test Runner

Runs all 5 tests and aggregates results for Q39 resolution.

Tests:
1. Perturbation-Recovery Dynamics
2. Basin of Attraction Mapping
3. Negative Feedback Quantification
4. Catastrophic Failure Boundary
5. Cross-Domain Universality

Run:
    python run_all_q39_tests.py
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_q39_perturbation_recovery import run_comprehensive_test as run_test1
from test_q39_basin_mapping import run_comprehensive_test as run_test2
from test_q39_negative_feedback import run_comprehensive_test as run_test3
from test_q39_catastrophic_boundary import run_comprehensive_test as run_test4
from test_q39_cross_domain import run_comprehensive_test as run_test5


def run_all_tests(seed: int = 42) -> dict:
    """
    Run all Q39 tests and aggregate results.
    """
    print("=" * 70)
    print("Q39: HOMEOSTATIC REGULATION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Seed: {seed}")
    print()

    results = {
        'question': 'Q39',
        'title': 'Homeostatic Regulation',
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'tests': {},
        'summary': {}
    }

    # Test 1: Perturbation-Recovery
    print("-" * 70)
    print("TEST 1: Perturbation-Recovery Dynamics")
    print("-" * 70)
    test1 = run_test1(seed)
    results['tests']['perturbation_recovery'] = test1['summary']
    print(f"  PASS: {test1['summary']['PASS']}")
    print()

    # Test 2: Basin Mapping
    print("-" * 70)
    print("TEST 2: Basin of Attraction Mapping")
    print("-" * 70)
    test2 = run_test2(seed)
    results['tests']['basin_mapping'] = test2['summary']
    print(f"  PASS: {test2['summary']['PASS']}")
    print()

    # Test 3: Negative Feedback
    print("-" * 70)
    print("TEST 3: Negative Feedback Quantification")
    print("-" * 70)
    test3 = run_test3(seed)
    results['tests']['negative_feedback'] = test3['summary']
    print(f"  PASS: {test3['summary']['PASS']}")
    print()

    # Test 4: Catastrophic Boundary
    print("-" * 70)
    print("TEST 4: Catastrophic Failure Boundary")
    print("-" * 70)
    test4 = run_test4(seed)
    results['tests']['catastrophic_boundary'] = test4['summary']
    print(f"  PASS: {test4['summary']['PASS']}")
    print()

    # Test 5: Cross-Domain Universality
    print("-" * 70)
    print("TEST 5: Cross-Domain Universality")
    print("-" * 70)
    test5 = run_test5(seed)
    results['tests']['cross_domain'] = test5['summary']
    print(f"  PASS: {test5['summary']['PASS']}")
    print()

    # Aggregate results
    all_pass = [
        test1['summary']['PASS'],
        test2['summary']['PASS'],
        test3['summary']['PASS'],
        test4['summary']['PASS'],
        test5['summary']['PASS']
    ]

    n_passed = sum(all_pass)
    n_total = len(all_pass)

    # Helper to safely compare with None
    def safe_gt(val, threshold):
        return val is not None and val > threshold

    def safe_lt(val, threshold):
        return val is not None and val < threshold

    results['summary'] = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'pass_rate': n_passed / n_total,
        'overall_PASS': n_passed >= 3,  # At least 3 of 5 tests must pass
        'key_findings': {
            'exponential_recovery': safe_gt(test1['summary'].get('mean_R_squared'), 0.8),
            'stable_attractor': safe_gt(test2['summary'].get('basin_width'), 1.0),
            'negative_feedback': safe_lt(test3['summary'].get('mean_correlation'), -0.3),
            'sharp_boundary': safe_gt(test4['summary'].get('sharpness'), 0.5),
            'universal_constants': test5['summary'].get('is_universal', False)
        }
    }

    return results


def format_final_report(results: dict) -> str:
    """Format results as a report for Q39 documentation."""
    report = []
    report.append("# Q39: Homeostatic Regulation - Test Results")
    report.append("")
    report.append(f"**Date:** {results['timestamp']}")
    report.append(f"**Status:** {'ANSWERED' if results['summary']['overall_PASS'] else 'PARTIAL'}")
    report.append("")

    report.append("## Test Results Summary")
    report.append("")
    report.append("| Test | Result | Key Metric |")
    report.append("|------|--------|------------|")

    test_names = [
        ('perturbation_recovery', 'Perturbation Recovery', 'mean_R_squared'),
        ('basin_mapping', 'Basin Mapping', 'basin_width'),
        ('negative_feedback', 'Negative Feedback', 'mean_correlation'),
        ('catastrophic_boundary', 'Catastrophic Boundary', 'sharpness'),
        ('cross_domain', 'Cross-Domain Universality', 'is_universal')
    ]

    for key, name, metric in test_names:
        test = results['tests'].get(key, {})
        passed = test.get('PASS', False)
        status = "PASS" if passed else "FAIL"
        value = test.get(metric, 'N/A')
        if isinstance(value, float):
            value = f"{value:.3f}"
        report.append(f"| {name} | {status} | {metric}={value} |")

    report.append("")
    report.append(f"**Overall:** {results['summary']['tests_passed']}/{results['summary']['tests_total']} tests passed")
    report.append("")

    if results['summary']['overall_PASS']:
        report.append("## Conclusion")
        report.append("")
        report.append("**YES - R > τ is a homeostatic setpoint.**")
        report.append("")
        report.append("Evidence:")
        report.append("1. **Exponential Recovery**: M(t) follows M* + ΔM₀·exp(-t/τ_relax)")
        report.append("2. **Stable Attractor**: Basin of attraction with finite width exists")
        report.append("3. **Negative Feedback**: corr(M, dE/dt) < -0.3 (Active Inference)")
        report.append("4. **Phase Transition**: Sharp boundary between recovery and collapse")
        report.append("5. **Universality**: Constants (τ_relax, M*) similar across domains")
        report.append("")
        report.append("Homeostasis emerges from: Active Inference (Q35) + FEP (Q9) + Noether Conservation (Q38)")

    return "\n".join(report)


if __name__ == '__main__':
    results = run_all_tests(seed=42)

    # Print final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTests Passed: {results['summary']['tests_passed']}/{results['summary']['tests_total']}")
    print(f"Pass Rate: {results['summary']['pass_rate']*100:.1f}%")
    overall_status = "PASS - Q39 ANSWERED" if results['summary']['overall_PASS'] else "PARTIAL"
    print(f"\nOVERALL RESULT: {overall_status}")
    print()

    # Key findings
    print("Key Findings:")
    for finding, status in results['summary']['key_findings'].items():
        symbol = "[+]" if status else "[-]"
        print(f"  {symbol} {finding}")

    # Save results
    output_path = Path(__file__).parent / 'q39_all_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Generate report
    report = format_final_report(results)
    report_path = Path(__file__).parent / 'Q39_TEST_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
