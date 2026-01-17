"""
Q8 Master Test Runner

Executes all Q8 tests and produces a unified verdict on whether
semantic space is a Kahler manifold with c_1 = 1.

Tests:
1. Direct Chern class computation (TEST 1)
2. Kahler structure verification (TEST 2)
3. Holonomy group classification (TEST 3)
4. 50% corruption stress test (TEST 4)

Verdict Criteria:
- Q8 ANSWERED (c_1 = 1 confirmed) if 6/7 tests pass, no catastrophic failures
- Q8 FALSIFIED if c_1 != 1 (p < 0.01), Kahler violated, or CV > 30%
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import traceback

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from q8_test_harness import Q8Logger, get_test_metadata, save_results


def run_all_tests(quick: bool = True, save: bool = True):
    """
    Run all Q8 tests and produce unified verdict.

    Args:
        quick: If True, run quick versions of tests
        save: If True, save combined results
    """
    logger = Q8Logger("Q8-MASTER", verbose=True)

    print("\n" + "="*70)
    print("  Q8: TOPOLOGY CLASSIFICATION - MASTER TEST SUITE")
    print("  TARGET: Prove semantic space is Kahler with c_1 = 1")
    print("="*70 + "\n")

    results = {
        'test_suite': 'Q8_TOPOLOGY_CLASSIFICATION',
        'timestamp': datetime.now().isoformat(),
        'mode': 'QUICK' if quick else 'FULL',
        'tests': {},
        'verdicts': {},
        'metadata': get_test_metadata()
    }

    # Track verdicts
    verdicts = {}

    # ==========================================================================
    # TEST 1: Direct Chern Class Computation
    # ==========================================================================
    logger.section("TEST 1: DIRECT CHERN CLASS COMPUTATION")

    try:
        from test_q8_chern_class import main as run_chern_test
        test1_results = run_chern_test(quick=quick, save=False)
        results['tests']['test1_chern_class'] = test1_results
        verdicts['test1'] = test1_results.get('verdict', {}).get('final_verdict', 'ERROR') == 'PASS'
        logger.info(f"TEST 1 VERDICT: {'PASS' if verdicts['test1'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}")
        traceback.print_exc()
        verdicts['test1'] = False
        results['tests']['test1_chern_class'] = {'error': str(e)}

    # ==========================================================================
    # TEST 2: Kahler Structure Verification
    # ==========================================================================
    logger.section("TEST 2: KAHLER STRUCTURE VERIFICATION")

    try:
        from test_q8_kahler_structure import main as run_kahler_test
        test2_results = run_kahler_test(quick=quick, save=False)
        results['tests']['test2_kahler_structure'] = test2_results
        verdicts['test2'] = test2_results.get('verdict', {}).get('final_verdict', 'ERROR') == 'PASS'
        logger.info(f"TEST 2 VERDICT: {'PASS' if verdicts['test2'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}")
        traceback.print_exc()
        verdicts['test2'] = False
        results['tests']['test2_kahler_structure'] = {'error': str(e)}

    # ==========================================================================
    # TEST 3: Holonomy Group Classification
    # ==========================================================================
    logger.section("TEST 3: HOLONOMY GROUP CLASSIFICATION")

    try:
        from test_q8_holonomy import main as run_holonomy_test
        test3_results = run_holonomy_test(quick=quick, save=False)
        results['tests']['test3_holonomy'] = test3_results
        verdicts['test3'] = test3_results.get('verdict', {}).get('final_verdict', 'ERROR') == 'PASS'
        logger.info(f"TEST 3 VERDICT: {'PASS' if verdicts['test3'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}")
        traceback.print_exc()
        verdicts['test3'] = False
        results['tests']['test3_holonomy'] = {'error': str(e)}

    # ==========================================================================
    # TEST 4: Corruption Stress Test
    # ==========================================================================
    logger.section("TEST 4: 50% CORRUPTION STRESS TEST")

    try:
        from test_q8_corruption import main as run_corruption_test
        test4_results = run_corruption_test(save=False)
        results['tests']['test4_corruption'] = test4_results
        verdicts['test4'] = test4_results.get('verdict', {}).get('final_verdict', 'ERROR') == 'PASS'
        logger.info(f"TEST 4 VERDICT: {'PASS' if verdicts['test4'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"TEST 4 FAILED: {e}")
        traceback.print_exc()
        verdicts['test4'] = False
        results['tests']['test4_corruption'] = {'error': str(e)}

    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    logger.section("FINAL VERDICT")

    results['verdicts'] = verdicts

    n_passed = sum(verdicts.values())
    n_total = len(verdicts)
    pass_rate = n_passed / n_total if n_total > 0 else 0

    # Verdict logic:
    # - 4/4 tests pass = CONFIRMED
    # - 3/4 tests pass = PARTIAL (need more investigation)
    # - <3/4 tests pass = FALSIFIED

    if n_passed == n_total:
        final_verdict = "CONFIRMED"
        q8_status = "ANSWERED"
    elif n_passed >= 3:
        final_verdict = "PARTIAL"
        q8_status = "NEEDS_MORE_TESTS"
    else:
        final_verdict = "FALSIFIED"
        q8_status = "FALSIFIED"

    results['final_verdict'] = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'pass_rate': pass_rate,
        'verdict': final_verdict,
        'q8_status': q8_status,
        'conclusion': get_conclusion(final_verdict, verdicts)
    }

    # Print summary
    print("\n" + "="*70)
    print("  Q8 TOPOLOGY CLASSIFICATION - FINAL RESULTS")
    print("="*70)
    print(f"\n  Tests Passed: {n_passed}/{n_total} ({pass_rate*100:.0f}%)")
    print()

    for test, passed in verdicts.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test.upper()}: {status}")

    print()
    print(f"  FINAL VERDICT: {final_verdict}")
    print(f"  Q8 STATUS: {q8_status}")
    print()
    print(f"  {results['final_verdict']['conclusion']}")
    print("="*70 + "\n")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_master_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


def get_conclusion(verdict: str, verdicts: dict) -> str:
    """Generate human-readable conclusion."""

    if verdict == "CONFIRMED":
        return (
            "SEMANTIC SPACE IS A KAHLER MANIFOLD WITH c_1 = 1.\n"
            "  The first Chern class equals 1, implying alpha = 1/2.\n"
            "  This is a TOPOLOGICAL INVARIANT, not a coincidence.\n"
            "  Q8 can be marked as ANSWERED."
        )
    elif verdict == "PARTIAL":
        failed = [k for k, v in verdicts.items() if not v]
        return (
            f"PARTIAL EVIDENCE for Kahler structure with c_1 = 1.\n"
            f"  Failed tests: {', '.join(failed)}\n"
            f"  Additional investigation needed before final verdict."
        )
    else:
        failed = [k for k, v in verdicts.items() if not v]
        return (
            f"c_1 = 1 HYPOTHESIS NOT SUPPORTED.\n"
            f"  Failed tests: {', '.join(failed)}\n"
            f"  The theory needs revision or the tests need debugging."
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Master Test Runner")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    run_all_tests(quick=not args.full, save=not args.no_save)
