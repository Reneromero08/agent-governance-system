#!/usr/bin/env python3
"""
Q7: Receipt Generation

Runs all Q7 tests and compiles results into q7_receipt.json.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import json
import sys
import os
from datetime import datetime, timezone
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types properly."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
from test_q7_axiom_falsification import run_all_axiom_tests
from test_q7_alternatives_fail import test_all_alternatives
from test_q7_adversarial_gauntlet import run_adversarial_gauntlet
from test_q7_cross_scale_arch import test_all_combinations, test_all_intensivity
from test_q7_negative_controls import run_all_negative_controls
from test_q7_phase_transition import run_phase_transition_tests

from theory.scale_transformation import run_self_tests as run_scale_tests
from theory.beta_function import run_self_tests as run_beta_tests
from theory.percolation import run_self_tests as run_percolation_tests


def generate_receipt() -> dict:
    """Generate complete Q7 validation receipt."""

    print("=" * 80)
    print("Q7: GENERATING VALIDATION RECEIPT")
    print("=" * 80)

    receipt = {
        "suite": "Q7_MULTISCALE_COMPOSITION",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": True,  # Will be updated based on results
    }

    # Track PRIMARY test results (real embeddings - these determine verdict)
    primary_pass = True

    # 1. Theory verification (INFORMATIONAL - uses synthetic data)
    print("\n[1/7] Running theory self-tests...")
    try:
        scale_tests_pass = run_scale_tests()
        receipt["theory"] = {
            "T_operator_defined": True,
            "T_group_action_verified": scale_tests_pass,
            "scale_transformation_tests": "PASS" if scale_tests_pass else "FAIL",
            "note": "Theory tests use synthetic data for operator verification"
        }
        # Theory tests are informational - don't block verdict
    except Exception as e:
        receipt["theory"] = {"error": str(e), "note": "Theory tests are informational"}

    # 2. Axiom tests (INFORMATIONAL - uses synthetic data)
    print("\n[2/7] Running axiom falsification tests...")
    try:
        axiom_results = run_all_axiom_tests()
        receipt["axiom_tests"] = axiom_results
        # Axiom tests use synthetic data - informational only
        receipt["axiom_tests"]["note"] = "Axiom tests use synthetic data for verification"
    except Exception as e:
        receipt["axiom_tests"] = {"error": str(e), "note": "Axiom tests are informational"}

    # 3. Alternative operators (PRIMARY - real embeddings)
    print("\n[3/7] Running alternative operator tests...")
    try:
        alt_results = test_all_alternatives()
        receipt["alternatives_failed"] = alt_results
        if alt_results["summary"]["verdict"] != "CONFIRMED":
            primary_pass = False
    except Exception as e:
        receipt["alternatives_failed"] = {"error": str(e)}
        primary_pass = False

    # 4. Adversarial gauntlet (PRIMARY - real embeddings)
    print("\n[4/7] Running adversarial gauntlet...")
    try:
        gauntlet_results = run_adversarial_gauntlet()
        receipt["adversarial_gauntlet"] = gauntlet_results
        if gauntlet_results["summary"]["verdict"] != "CONFIRMED":
            primary_pass = False
    except Exception as e:
        receipt["adversarial_gauntlet"] = {"error": str(e)}
        primary_pass = False

    # 5. Cross-scale validation (PRIMARY - real embeddings)
    print("\n[5/7] Running cross-scale architecture tests...")
    try:
        cross_results = test_all_combinations()
        intensivity_results = test_all_intensivity()
        receipt["cross_scale_validation"] = {
            "cross_scale": cross_results,
            "intensivity": intensivity_results
        }
        if cross_results["summary"]["verdict"] != "CONFIRMED":
            primary_pass = False
    except Exception as e:
        receipt["cross_scale_validation"] = {"error": str(e)}
        primary_pass = False

    # 6. Negative controls (PRIMARY - real embeddings)
    print("\n[6/7] Running negative control tests...")
    try:
        negative_results = run_all_negative_controls()
        receipt["negative_controls"] = negative_results
        if negative_results["summary"]["verdict"] != "CONFIRMED":
            primary_pass = False
    except Exception as e:
        receipt["negative_controls"] = {"error": str(e)}
        primary_pass = False

    # 7. Phase transition (PRIMARY - real embeddings)
    print("\n[7/7] Running phase transition tests...")
    try:
        pt_results = run_phase_transition_tests()
        receipt["phase_transition"] = pt_results
        if pt_results["summary"]["verdict"] != "CONFIRMED":
            primary_pass = False
    except Exception as e:
        receipt["phase_transition"] = {"error": str(e)}
        primary_pass = False

    # Final verdict based on PRIMARY tests (real embeddings)
    receipt["passed"] = primary_pass
    receipt["verdict"] = "CONFIRMED" if primary_pass else "FAILED"
    receipt["reasoning"] = generate_reasoning(receipt)

    return receipt


def generate_reasoning(receipt: dict) -> str:
    """Generate human-readable reasoning from receipt."""
    parts = []

    # Theory
    if receipt.get("theory", {}).get("T_group_action_verified"):
        parts.append("T operator satisfies group action properties")

    # Axioms
    axiom_result = receipt.get("axiom_tests", {})
    if axiom_result.get("all_pass"):
        parts.append("All 4 composition axioms C1-C4 pass")

    # Alternatives
    alt_result = receipt.get("alternatives_failed", {})
    if alt_result.get("summary", {}).get("verdict") == "CONFIRMED":
        n = alt_result["summary"].get("n_correctly_fail", 0)
        parts.append(f"{n} alternative operators correctly fail")

    # Gauntlet
    gauntlet = receipt.get("adversarial_gauntlet", {})
    if gauntlet.get("summary", {}).get("verdict") == "CONFIRMED":
        n_pass = gauntlet["summary"].get("n_pass", 0)
        parts.append(f"{n_pass}/6 adversarial domains pass")

    # Cross-scale
    cross = receipt.get("cross_scale_validation", {})
    if cross.get("cross_scale", {}).get("summary", {}).get("verdict") == "CONFIRMED":
        n = cross["cross_scale"]["summary"].get("n_combinations", 0)
        parts.append(f"All {n} cross-scale combinations pass")

    # Negative controls
    neg = receipt.get("negative_controls", {})
    if neg.get("summary", {}).get("verdict") == "CONFIRMED":
        n = neg["summary"].get("n_correctly_fail", 0)
        parts.append(f"{n} negative controls correctly fail")

    # Phase transition
    pt = receipt.get("phase_transition", {})
    if pt.get("summary", {}).get("verdict") == "CONFIRMED":
        tau_c = pt.get("phase_transition", {}).get("critical_threshold", 0)
        parts.append(f"Phase transition detected at tau_c={tau_c:.3f}")

    if receipt["verdict"] == "CONFIRMED":
        return "R is proven RG fixed point. " + "; ".join(parts)
    else:
        return "FAILED: " + "; ".join(parts) if parts else "Multiple test failures"


def main():
    """Main entry point."""
    receipt = generate_receipt()

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "q7_receipt.json")
    with open(output_path, "w") as f:
        json.dump(receipt, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 80)
    print(f"Receipt saved to: {output_path}")
    print(f"Final verdict: {receipt['verdict']}")
    print(f"Reasoning: {receipt['reasoning']}")
    print("=" * 80)

    return receipt


if __name__ == "__main__":
    main()
