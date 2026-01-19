#!/usr/bin/env python3
"""
Q11: Valley Blindness - Complete Test Suite Runner

Executes all 12 tests and compiles results to answer Q11:
"Can we extend the information horizon without changing epistemology?
Or is 'can't know from here' an irreducible limit?"

Usage:
    python run_q11_all.py           # Run all tests
    python run_q11_all.py --quick   # Run quick tests only (no embeddings)
    python run_q11_all.py --phase 1 # Run only Phase 1 tests
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_MODULES = {
    # Phase 1: Structural Horizons
    'phase_1': [
        ('test_q11_semantic_horizon', '2.1 Semantic Event Horizon'),
        ('test_q11_bayesian_prison', '2.2 Bayesian Prison Break'),
        ('test_q11_kolmogorov_ceiling', '2.3 Kolmogorov Ceiling'),
        ('test_q11_incommensurability', '2.4 Incommensurability'),
    ],
    # Phase 2: Detection & Extension
    'phase_2': [
        ('test_q11_unknown_unknowns', '2.5 Unknown Unknowns'),
        ('test_q11_horizon_extension', '2.6 Horizon Extension (CORE)'),
        ('test_q11_entanglement_bridge', '2.7 Entanglement Bridge'),
        ('test_q11_time_asymmetry', '2.8 Time Asymmetry'),
    ],
    # Phase 3: Transcendence
    'phase_3': [
        ('test_q11_renormalization', '2.9 Renormalization Escape'),
        ('test_q11_goedel_construction', '2.10 Goedel Construction'),
        ('test_q11_qualia_horizon', '2.11 Qualia Horizon'),
        ('test_q11_self_detection', '2.12 Self-Detection (ULTIMATE)'),
    ],
}

QUICK_TESTS = [
    'test_q11_bayesian_prison',
    'test_q11_kolmogorov_ceiling',
    'test_q11_horizon_extension',
    'test_q11_goedel_construction',
    'test_q11_time_asymmetry',
    'test_q11_self_detection',
]

PASS_THRESHOLD = 8  # Need 8+ tests to pass for Q11 to be "answered"


# =============================================================================
# TEST EXECUTION
# =============================================================================

@dataclass
class TestResult:
    """Result from running a single test."""
    module_name: str
    test_name: str
    passed: bool
    horizon_type: str
    notes: str
    metrics: Dict
    error: Optional[str] = None
    duration_sec: float = 0.0


def run_single_test(module_name: str, test_name: str) -> TestResult:
    """
    Run a single test module.

    Args:
        module_name: Name of the test module (without .py)
        test_name: Human-readable test name

    Returns:
        TestResult with pass/fail status
    """
    import time

    start_time = time.time()

    try:
        # Import the module
        module_path = Path(__file__).parent / f"{module_name}.py"

        if not module_path.exists():
            return TestResult(
                module_name=module_name,
                test_name=test_name,
                passed=False,
                horizon_type="unknown",
                notes="Module not found",
                metrics={},
                error=f"File not found: {module_path}",
            )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find and run the main test function
        # Look for run_*_test function
        test_func = None
        for name in dir(module):
            if name.startswith('run_') and name.endswith('_test'):
                test_func = getattr(module, name)
                break

        if test_func is None:
            return TestResult(
                module_name=module_name,
                test_name=test_name,
                passed=False,
                horizon_type="unknown",
                notes="No test function found",
                metrics={},
                error="Could not find run_*_test function",
            )

        # Run the test
        passed, result = test_func()

        duration = time.time() - start_time

        return TestResult(
            module_name=module_name,
            test_name=test_name,
            passed=passed,
            horizon_type=result.horizon_type.value if hasattr(result.horizon_type, 'value') else str(result.horizon_type),
            notes=result.notes,
            metrics=result.metrics,
            duration_sec=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        import traceback
        return TestResult(
            module_name=module_name,
            test_name=test_name,
            passed=False,
            horizon_type="error",
            notes=f"Test crashed: {str(e)}",
            metrics={},
            error=traceback.format_exc(),
            duration_sec=duration,
        )


def run_phase(phase_name: str, tests: List[Tuple[str, str]],
              quick_mode: bool = False) -> List[TestResult]:
    """
    Run all tests in a phase.

    Args:
        phase_name: Name of the phase
        tests: List of (module_name, test_name) tuples
        quick_mode: If True, skip embedding-heavy tests

    Returns:
        List of TestResults
    """
    results = []

    print(f"\n{'='*70}")
    print(f"PHASE: {phase_name.upper().replace('_', ' ')}")
    print('='*70)

    for module_name, test_name in tests:
        if quick_mode and module_name not in QUICK_TESTS:
            print(f"\n[SKIP] {test_name} (quick mode)")
            continue

        print(f"\n[RUN] {test_name}")
        print("-" * 50)

        result = run_single_test(module_name, test_name)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {test_name}")
        print(f"  Horizon type: {result.horizon_type}")
        print(f"  Duration: {result.duration_sec:.1f}s")

        if result.error:
            print(f"  Error: {result.notes}")

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results: List[TestResult]) -> Dict:
    """
    Analyze all test results and determine Q11 answer.

    Args:
        results: List of all test results

    Returns:
        Dictionary with analysis and Q11 answer
    """
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    total_tests = len(results)

    # Count horizon types
    horizon_types = {}
    for r in results:
        if r.passed:
            ht = r.horizon_type
            horizon_types[ht] = horizon_types.get(ht, 0) + 1

    # Determine Q11 answer
    if passed_count >= PASS_THRESHOLD:
        structural_count = horizon_types.get('structural', 0) + horizon_types.get('ontological', 0)
        instrumental_count = horizon_types.get('instrumental', 0)

        if structural_count > instrumental_count:
            answer = "STRUCTURAL: Some horizons require epistemology change to extend"
            answer_type = "structural"
        else:
            answer = "INSTRUMENTAL: All tested horizons can be extended without epistemology change"
            answer_type = "instrumental"

        status = "ANSWERED"
    else:
        answer = "INCONCLUSIVE: Insufficient test agreement to answer Q11"
        answer_type = "inconclusive"
        status = "PARTIAL"

    return {
        'status': status,
        'answer': answer,
        'answer_type': answer_type,
        'passed': passed_count,
        'failed': failed_count,
        'total': total_tests,
        'pass_rate': passed_count / total_tests if total_tests > 0 else 0,
        'horizon_types': horizon_types,
        'pass_threshold': PASS_THRESHOLD,
        'meets_threshold': passed_count >= PASS_THRESHOLD,
    }


def generate_report(results: List[TestResult], analysis: Dict) -> str:
    """Generate a text report of all results."""
    lines = []
    lines.append("=" * 70)
    lines.append("Q11: VALLEY BLINDNESS - TEST RESULTS")
    lines.append("=" * 70)
    lines.append(f"\nTimestamp: {datetime.now().isoformat()}")
    lines.append(f"Total tests: {analysis['total']}")
    lines.append(f"Passed: {analysis['passed']}")
    lines.append(f"Failed: {analysis['failed']}")
    lines.append(f"Pass rate: {analysis['pass_rate']:.1%}")
    lines.append(f"\nPass threshold: {analysis['pass_threshold']}")
    lines.append(f"Meets threshold: {analysis['meets_threshold']}")

    lines.append("\n" + "-" * 70)
    lines.append("INDIVIDUAL TEST RESULTS")
    lines.append("-" * 70)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"\n[{status}] {r.test_name}")
        lines.append(f"  Module: {r.module_name}")
        lines.append(f"  Horizon type: {r.horizon_type}")
        lines.append(f"  Duration: {r.duration_sec:.1f}s")
        lines.append(f"  Notes: {r.notes}")
        if r.error:
            lines.append(f"  Error: {r.error[:200]}...")

    lines.append("\n" + "-" * 70)
    lines.append("HORIZON TYPE DISTRIBUTION")
    lines.append("-" * 70)

    for ht, count in analysis['horizon_types'].items():
        lines.append(f"  {ht}: {count}")

    lines.append("\n" + "=" * 70)
    lines.append("Q11 ANSWER")
    lines.append("=" * 70)
    lines.append(f"\nStatus: {analysis['status']}")
    lines.append(f"Answer: {analysis['answer']}")

    if analysis['answer_type'] == 'structural':
        lines.append("\nImplication: Information horizons are HIERARCHICAL.")
        lines.append("Some can be extended without changing epistemology (instrumental),")
        lines.append("but others REQUIRE paradigm shifts (structural).")
        lines.append("'Can't know from here' is sometimes an IRREDUCIBLE LIMIT.")
    elif analysis['answer_type'] == 'instrumental':
        lines.append("\nImplication: Information horizons are primarily INSTRUMENTAL.")
        lines.append("With better tools, more data, or more compute,")
        lines.append("all tested horizons can eventually be extended.")
    else:
        lines.append("\nImplication: More testing needed to determine horizon nature.")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Q11 Valley Blindness tests")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip embedding-heavy tests)')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help='Run only specified phase (1, 2, or 3)')
    parser.add_argument('--output', type=str, default='Q11_RESULTS.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    print("=" * 70)
    print("Q11: VALLEY BLINDNESS - COMPLETE TEST SUITE")
    print("=" * 70)
    print("\nQuestion: Can we extend the information horizon without")
    print("          changing epistemology? Or is 'can't know from here'")
    print("          an irreducible limit?")
    print("\nRunning 12 tests across 3 phases...")

    if args.quick:
        print("\n[QUICK MODE] Skipping embedding-heavy tests")

    # Collect all results
    all_results = []

    # Determine which phases to run
    phases_to_run = []
    if args.phase:
        phases_to_run = [f'phase_{args.phase}']
    else:
        phases_to_run = ['phase_1', 'phase_2', 'phase_3']

    # Run tests
    for phase in phases_to_run:
        if phase in TEST_MODULES:
            results = run_phase(phase, TEST_MODULES[phase], args.quick)
            all_results.extend(results)

    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYZING RESULTS")
    print("=" * 70)

    analysis = analyze_results(all_results)

    # Generate report
    report = generate_report(all_results, analysis)
    print("\n" + report)

    # Save JSON results
    output_path = Path(__file__).parent / args.output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Q11: Valley Blindness',
        'core_question': 'Can we extend information horizon without changing epistemology?',
        'analysis': analysis,
        'results': [asdict(r) for r in all_results],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Exit code based on Q11 answer
    if analysis['meets_threshold']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
