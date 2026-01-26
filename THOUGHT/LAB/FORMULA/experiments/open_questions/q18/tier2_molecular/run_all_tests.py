#!/usr/bin/env python3
"""
Q18 Tier 2: Molecular Scale Test Runner

Executes all Tier 2 molecular scale tests for R = E/sigma verification.

Tests:
1. Blind Folding Prediction (AUC > 0.75)
2. Binding Causality - Mutations (Spearman rho > 0.5)
3. 8e Conservation Law (CV < 15%)
4. Adversarial Sequences (Survival rate > 70%)

Output: molecular_report.json

Author: Claude
Date: 2026-01-25
Version: 1.0.0
"""

import sys
import json
import hashlib
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from molecular_utils import to_builtin, compute_data_hash

__version__ = "1.0.0"
__suite__ = "Q18_TIER2_MOLECULAR"


def run_blind_folding_test(verbose: bool = True) -> Dict[str, Any]:
    """Run Test 2.1: Blind Folding Prediction."""
    try:
        from test_blind_folding import run_test
        result = run_test(n_families=50, proteins_per_family=15, seed=42, verbose=verbose)
        return {
            "auc": result.auc,
            "correlation": result.correlation,
            "p_value": result.p_value,
            "n_proteins": result.n_proteins,
            "n_families": result.n_families,
            "passed": result.passed,
            "error": None
        }
    except Exception as e:
        if verbose:
            print(f"ERROR in blind folding test: {e}")
            traceback.print_exc()
        return {
            "auc": 0.0,
            "passed": False,
            "error": str(e)
        }


def run_binding_causality_test(verbose: bool = True) -> Dict[str, Any]:
    """Run Test 2.2: Binding Causality (Mutations)."""
    try:
        from test_binding_causality import run_test
        result = run_test(n_proteins=5, mutations_per_protein=80, seed=42, verbose=verbose)
        return {
            "spearman_rho": result.spearman_rho,
            "p_value": result.p_value,
            "n_mutations": result.n_mutations,
            "n_proteins": result.n_proteins,
            "pearson_r": result.pearson_r,
            "passed": result.passed,
            "error": None
        }
    except Exception as e:
        if verbose:
            print(f"ERROR in binding causality test: {e}")
            traceback.print_exc()
        return {
            "spearman_rho": 0.0,
            "p_value": 1.0,
            "passed": False,
            "error": str(e)
        }


def run_8e_conservation_test(verbose: bool = True) -> Dict[str, Any]:
    """Run Test 2.3: 8e Conservation Law."""
    try:
        from test_8e_conservation import run_test
        result = run_test(n_families_per_class=5, proteins_per_family=25, seed=42, verbose=verbose)
        return {
            "df": result.mean_df,
            "alpha": result.mean_alpha,
            "df_x_alpha": result.mean_df_x_alpha,
            "cv_across_families": result.cv_across_families,
            "deviation_from_8e": result.deviation_from_8e,
            "passed": result.passed,
            "error": None
        }
    except Exception as e:
        if verbose:
            print(f"ERROR in 8e conservation test: {e}")
            traceback.print_exc()
        return {
            "df": 0.0,
            "alpha": 0.0,
            "df_x_alpha": 0.0,
            "cv_across_families": 1.0,
            "passed": False,
            "error": str(e)
        }


def run_adversarial_test(verbose: bool = True) -> Dict[str, Any]:
    """Run Test 2.4: Adversarial Sequences."""
    try:
        from test_adversarial import run_test
        result = run_test(n_sequences_per_case=20, seed=42, verbose=verbose)

        case_summary = {
            r.case_name: {"mean_r": r.mean_r, "meaningful": r.meaningful}
            for r in result.case_results
        }

        return {
            "survival_rate": result.survival_rate,
            "n_cases": result.n_cases,
            "cases_passed": result.cases_passed,
            "case_summary": case_summary,
            "passed": result.passed,
            "error": None
        }
    except Exception as e:
        if verbose:
            print(f"ERROR in adversarial test: {e}")
            traceback.print_exc()
        return {
            "survival_rate": 0.0,
            "passed": False,
            "error": str(e)
        }


def generate_key_findings(results: Dict[str, Any]) -> List[str]:
    """Generate key findings from test results."""
    findings = []

    # Blind folding
    bf = results.get("blind_folding", {})
    if bf.get("passed"):
        findings.append(f"R successfully predicts fold quality with AUC={bf.get('auc', 0):.3f}")
    else:
        findings.append(f"Blind folding prediction below threshold (AUC={bf.get('auc', 0):.3f})")

    # Binding causality
    bc = results.get("binding_causality", {})
    rho = abs(bc.get("spearman_rho", 0))
    if bc.get("passed"):
        findings.append(f"Delta-R correlates with mutation effects (|rho|={rho:.3f})")
    else:
        findings.append(f"Mutation effect correlation below threshold (|rho|={rho:.3f})")

    # 8e conservation
    ec = results.get("8e_conservation", {})
    if ec.get("passed"):
        findings.append(f"8e conservation holds at molecular scale (CV={ec.get('cv_across_families', 0)*100:.1f}%)")
    else:
        findings.append(f"8e conservation varies too much (CV={ec.get('cv_across_families', 0)*100:.1f}%)")

    # Adversarial
    adv = results.get("adversarial", {})
    if adv.get("passed"):
        findings.append(f"R robust to adversarial sequences ({adv.get('survival_rate', 0)*100:.0f}% survival)")
    else:
        findings.append(f"R partially fails on adversarial cases ({adv.get('survival_rate', 0)*100:.0f}% survival)")

    return findings


def run_all_tests(verbose: bool = True, output_dir: str = None) -> Dict[str, Any]:
    """
    Run all Tier 2 molecular scale tests and generate report.
    """
    if verbose:
        print("=" * 70)
        print("Q18 TIER 2: MOLECULAR SCALE TESTS")
        print("Testing R = E/sigma at protein/molecular scales")
        print("=" * 70)
        print(f"\nStarted: {datetime.now().isoformat()}")
        print()

    # Run all tests
    results = {}

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING TEST 2.1: BLIND FOLDING PREDICTION")
        print("=" * 70)
    results["blind_folding"] = run_blind_folding_test(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING TEST 2.2: BINDING CAUSALITY (MUTATIONS)")
        print("=" * 70)
    results["binding_causality"] = run_binding_causality_test(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING TEST 2.3: 8e CONSERVATION LAW")
        print("=" * 70)
    results["8e_conservation"] = run_8e_conservation_test(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING TEST 2.4: ADVERSARIAL SEQUENCES")
        print("=" * 70)
    results["adversarial"] = run_adversarial_test(verbose=verbose)

    # Count passed tests
    tests_passed = sum(1 for r in results.values() if r.get("passed", False))
    tests_total = len(results)

    # Generate key findings
    key_findings = generate_key_findings(results)

    # Compute data hash
    data_str = json.dumps(to_builtin(results), sort_keys=True)
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # Build final report
    report = {
        "agent_id": "molecular_tier2",
        "tier": "molecular",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {
            "blind_folding": {
                "auc": results["blind_folding"].get("auc", 0),
                "passed": results["blind_folding"].get("passed", False)
            },
            "binding_causality": {
                "spearman_rho": results["binding_causality"].get("spearman_rho", 0),
                "p_value": results["binding_causality"].get("p_value", 1.0),
                "passed": results["binding_causality"].get("passed", False)
            },
            "8e_conservation": {
                "df": results["8e_conservation"].get("df", 0),
                "alpha": results["8e_conservation"].get("alpha", 0),
                "df_x_alpha": results["8e_conservation"].get("df_x_alpha", 0),
                "cv_across_families": results["8e_conservation"].get("cv_across_families", 1.0),
                "passed": results["8e_conservation"].get("passed", False)
            },
            "adversarial": {
                "survival_rate": results["adversarial"].get("survival_rate", 0),
                "passed": results["adversarial"].get("passed", False)
            }
        },
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "key_findings": key_findings,
        "data_hash": data_hash,
        "detailed_results": results
    }

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: Q18 TIER 2 MOLECULAR SCALE TESTS")
        print("=" * 70)
        print(f"\nTests passed: {tests_passed}/{tests_total}")
        print("\nKey findings:")
        for finding in key_findings:
            print(f"  - {finding}")
        print(f"\nData hash: {data_hash}")

    # Save report
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "molecular_report.json"

    with open(report_path, 'w') as f:
        json.dump(to_builtin(report), f, indent=2)

    if verbose:
        print(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Q18 Tier 2: Molecular Scale Tests")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    verbose = not args.quiet
    report = run_all_tests(verbose=verbose, output_dir=args.output)

    # Exit with appropriate code
    if report["tests_passed"] == report["tests_total"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
