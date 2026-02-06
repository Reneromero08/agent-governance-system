#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q37: Semiotic Evolution Dynamics - Aggregated Test Runner

Runs all implemented tiers and produces a comprehensive summary.

REAL DATA ONLY. No synthetic bullshit.

Current Implementation Status:
- Tier 1: Historical Semantic Drift - REQUIRES REAL HISTWORDS DATA
- Tier 3: Cross-Lingual Convergence - IMPLEMENTED (mBERT/multilingual)
- Tier 4: Phylogenetic Reconstruction - IMPLEMENTED (WordNet)
- Tier 9: Conservation Law Persistence - IMPLEMENTED (multilingual + WordNet)
- Tier 10: Multi-Model Universality - IMPLEMENTED (5 architectures)

Usage:
    python run_all_q37_tests.py [--tier N] [--all]
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from q37_evolution_utils import Q37TestSuite, TierResult


def run_tier_1(histwords_dir: str) -> Q37TestSuite:
    """Run Tier 1: Historical Semantic Drift"""
    from test_q37_historical import run_tier_1_tests
    return run_tier_1_tests(histwords_dir)


def run_tier_3() -> Q37TestSuite:
    """Run Tier 3: Cross-Lingual Convergence"""
    from test_q37_crosslingual import run_tier_3_tests
    return run_tier_3_tests()


def run_tier_4() -> Q37TestSuite:
    """Run Tier 4: Phylogenetic Reconstruction"""
    from test_q37_phylogeny import run_tier_4_tests
    return run_tier_4_tests()


def run_tier_9() -> Q37TestSuite:
    """Run Tier 9: Conservation Law Persistence"""
    from test_q37_conservation import run_tier_9_tests
    return run_tier_9_tests()


def run_tier_10() -> Q37TestSuite:
    """Run Tier 10: Multi-Model Universality"""
    from test_q37_multimodel import run_tier_10_tests
    return run_tier_10_tests()


def print_final_summary(tier_results: Dict[int, Q37TestSuite]):
    """Print comprehensive summary of all tiers."""
    print("\n" + "=" * 70)
    print("Q37 SEMIOTIC EVOLUTION DYNAMICS - FINAL SUMMARY")
    print("=" * 70)
    print("REAL DATA ONLY - No synthetic simulations")
    print("=" * 70)

    total_passed = 0
    total_tests = 0
    tier_status = {}

    for tier_num in sorted(tier_results.keys()):
        suite = tier_results[tier_num]
        if suite is None or len(suite.results) == 0:
            tier_status[tier_num] = "SKIPPED"
            continue

        passed = sum(1 for r in suite.results if r.passed)
        total = len(suite.results)
        total_passed += passed
        total_tests += total

        if passed == total:
            status = "PASS"
        elif passed >= total * 0.66:
            status = "PARTIAL"
        else:
            status = "FAIL"

        tier_status[tier_num] = status

        print(f"\nTIER {tier_num}: {status} ({passed}/{total})")
        for r in suite.results:
            mark = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {mark} {r.test_name}: {r.metric_value:.4f} (threshold: {r.threshold})")

    print("\n" + "-" * 70)
    print("TIER STATUS SUMMARY:")
    print("-" * 70)

    tier_names = {
        1: "Historical Semantic Drift",
        3: "Cross-Lingual Convergence",
        4: "Phylogenetic Reconstruction (WordNet)",
        9: "Conservation Law Persistence",
        10: "Multi-Model Universality"
    }

    for tier_num, name in tier_names.items():
        status = tier_status.get(tier_num, "NOT IMPLEMENTED")
        print(f"  Tier {tier_num}: {name} - {status}")

    print("\n" + "=" * 70)
    if total_tests > 0:
        overall_rate = total_passed / total_tests
        print(f"OVERALL: {total_passed}/{total_tests} tests passed ({overall_rate*100:.1f}%)")

        if overall_rate >= 0.8:
            print("\nVERDICT: Q37 STRONG SUPPORT")
            print("Meanings evolve on M-field following evolutionary dynamics.")
        elif overall_rate >= 0.6:
            print("\nVERDICT: Q37 PARTIAL SUPPORT")
            print("Evidence supports evolutionary dynamics with some caveats.")
        else:
            print("\nVERDICT: Q37 WEAK SUPPORT")
            print("More evidence needed to confirm evolutionary dynamics.")
    else:
        print("No tests run - check data availability")

    print("=" * 70)


def save_aggregated_results(tier_results: Dict[int, Q37TestSuite], output_dir: str):
    """Save all results to a single JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Q37 Semiotic Evolution Dynamics',
        'data_type': 'REAL DATA ONLY',
        'tiers': {}
    }

    total_passed = 0
    total_tests = 0

    for tier_num, suite in tier_results.items():
        if suite is None or len(suite.results) == 0:
            results['tiers'][f'tier_{tier_num}'] = {'status': 'skipped'}
            continue

        passed = sum(1 for r in suite.results if r.passed)
        total = len(suite.results)
        total_passed += passed
        total_tests += total

        results['tiers'][f'tier_{tier_num}'] = {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'tests': [convert_numpy(r.to_dict()) for r in suite.results]
        }

    results['overall'] = {
        'total_passed': total_passed,
        'total_tests': total_tests,
        'pass_rate': total_passed / total_tests if total_tests > 0 else 0
    }

    output_path = os.path.join(output_dir, 'q37_all_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAggregated results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Q37 Semiotic Evolution - Run All Tests')
    parser.add_argument('--tier', type=int, choices=[1, 3, 4, 9, 10],
                        help='Run specific tier only')
    parser.add_argument('--all', action='store_true',
                        help='Run all implemented tiers')
    parser.add_argument('--histwords-dir', type=str, default='data/histwords_data',
                        help='Directory for HistWords data (Tier 1)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')

    args = parser.parse_args()

    tier_results = {}

    if args.tier:
        # Run single tier
        if args.tier == 1:
            data_dir = os.path.join(script_dir, args.histwords_dir)
            if os.path.exists(data_dir) and os.listdir(data_dir):
                tier_results[1] = run_tier_1(data_dir)
            else:
                print(f"Tier 1 requires real HistWords data at {data_dir}")
                print("Download from: https://nlp.stanford.edu/projects/histwords/")
                tier_results[1] = None
        elif args.tier == 3:
            tier_results[3] = run_tier_3()
        elif args.tier == 4:
            tier_results[4] = run_tier_4()
        elif args.tier == 9:
            tier_results[9] = run_tier_9()
        elif args.tier == 10:
            tier_results[10] = run_tier_10()
    elif args.all:
        # Run all implemented tiers
        print("\n" + "=" * 70)
        print("RUNNING ALL IMPLEMENTED TIERS")
        print("=" * 70)

        # Tier 1 - only if HistWords available
        data_dir = os.path.join(script_dir, args.histwords_dir)
        if os.path.exists(data_dir) and os.listdir(data_dir):
            tier_results[1] = run_tier_1(data_dir)
        else:
            print("\n[SKIP] Tier 1: HistWords data not available")
            print("       Download from: https://nlp.stanford.edu/projects/histwords/")
            tier_results[1] = None

        # Tier 3 - Cross-lingual (always available)
        tier_results[3] = run_tier_3()

        # Tier 4 - Phylogeny (always available)
        tier_results[4] = run_tier_4()

        # Tier 9 - Conservation (always available for 9.2, 9.3)
        tier_results[9] = run_tier_9()

        # Tier 10 - Multi-Model Universality (always available)
        tier_results[10] = run_tier_10()

    else:
        # Default: run tiers 3, 4, 9, 10 (always available)
        print("\n" + "=" * 70)
        print("RUNNING AVAILABLE TIERS (3, 4, 9, 10)")
        print("=" * 70)
        print("For Tier 1, download HistWords and use --all")

        tier_results[3] = run_tier_3()
        tier_results[4] = run_tier_4()
        tier_results[9] = run_tier_9()
        tier_results[10] = run_tier_10()

    # Print summary
    print_final_summary(tier_results)

    # Save results
    output_dir = os.path.join(script_dir, args.output_dir)
    save_aggregated_results(tier_results, output_dir)

    # Exit code based on overall pass rate
    total_passed = sum(sum(1 for r in s.results if r.passed) for s in tier_results.values() if s)
    total_tests = sum(len(s.results) for s in tier_results.values() if s)

    if total_tests > 0 and total_passed / total_tests >= 0.6:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
