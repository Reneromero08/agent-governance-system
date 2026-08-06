#!/usr/bin/env python3
"""Independent V3 reconstruction for Candidate M."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw_outputs" / "source_reproduction_v3" / "candidate_m_run1"
RAW = ROOT / "raw_outputs" / "independent_m_v3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cpair(value: list[float]) -> complex:
    return complex(value[0], value[1])


def matrix(value: list[list[list[float]]]) -> list[list[complex]]:
    return [[cpair(cell) for cell in row] for row in value]


def frobenius(left: list[list[complex]], right: list[list[complex]]) -> float:
    return math.sqrt(sum(abs(left[i][j] - right[i][j]) ** 2 for i in range(2) for j in range(2)))


def dagger(value: list[list[complex]]) -> list[list[complex]]:
    return [[value[j][i].conjugate() for j in range(2)] for i in range(2)]


def mmul(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    return [
        [sum(left[i][k] * right[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]


def identity2() -> list[list[complex]]:
    return [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]


def check_m() -> dict:
    result = read_json(SRC / "result.json")
    oracle = read_json(SRC / "oracle.json")
    primary = matrix(result["primary"]["boundary"])
    reordered = matrix(result["reordered_forward"]["boundary"])
    reuse = matrix(result["reuse"]["boundary"])
    primary_formula = matrix(result["primary"]["discrete_formula_boundary"])
    reuse_formula = matrix(result["reuse"]["discrete_formula_boundary"])
    unitary_primary = frobenius(mmul(dagger(primary), primary), identity2())
    unitary_reuse = frobenius(mmul(dagger(reuse), reuse), identity2())
    order_difference = frobenius(primary, reordered)
    formula_primary = frobenius(primary, primary_formula)
    formula_reuse = frobenius(reuse, reuse_formula)
    controls = result["controls"]
    baseline = result["matched_compact_classical"]
    mechanism_survives = (
        result["result"] == "PASS"
        and oracle["result"] == "PASS"
        and oracle["production_backend_imported"] is False
        and order_difference > 1.0
        and oracle["loop_order_noncommutes"] is True
        and unitary_primary < 1e-12
        and unitary_reuse < 1e-12
        and formula_primary < 1e-9
        and formula_reuse < 1e-9
        and result["actual_restored_carrier_reused"] is True
        and result["same_outer_carrier_variable"] is True
        and controls["missing_inverse_restored"] is False
        and controls["wrong_inverse_restored"] is False
        and controls["reordered_inverse_restored"] is False
        and result["intermediate_frame_projected"] is False
    )
    return {
        "candidate": "M",
        "independent_method": "matrix_level_reconstruction_from_reproduced_boundaries_and_formula_parity; no independent CP2 dark-frame geometry rebuild",
        "matrix_checks": {
            "primary_reordered_frobenius": order_difference,
            "source_recorded_order_difference": result["reordered_forward"]["boundary_difference_frobenius"],
            "primary_formula_frobenius": formula_primary,
            "reuse_formula_frobenius": formula_reuse,
            "primary_unitarity_frobenius": unitary_primary,
            "reuse_unitarity_frobenius": unitary_reuse,
        },
        "oracle_checks": {
            "mpmath_precision_digits": oracle["precision_decimal_digits"],
            "production_backend_imported": oracle["production_backend_imported"],
            "loop_order_noncommutes": oracle["loop_order_noncommutes"],
            "matched_compact_classical_2x2_recurrence_identical": oracle["matched_compact_classical_2x2_recurrence_identical"],
            "closed_form_fixed_loop_modules_available": oracle["closed_form_fixed_loop_modules_available"],
        },
        "restoration_and_controls": {
            "actual_restored_carrier_reused": result["actual_restored_carrier_reused"],
            "same_outer_carrier_variable": result["same_outer_carrier_variable"],
            "fresh_restored_reuse_boundary_error": result["fresh_restored_reuse_boundary_error"],
            "stress_maximum_restoration_error": result["stress_maximum_restoration_error"],
            "missing_inverse_restored": controls["missing_inverse_restored"],
            "wrong_inverse_restored": controls["wrong_inverse_restored"],
            "reordered_inverse_restored": controls["reordered_inverse_restored"],
            "intermediate_frame_projected": result["intermediate_frame_projected"],
            "final_boundary_projected_before_inverse": result["final_boundary_projected_before_inverse"],
        },
        "baseline_challenge": {
            "identical_2x2_holonomy_recurrence": baseline["identical_2x2_holonomy_recurrence"],
            "closed_form_fixed_loop_modules_available": baseline["closed_form_fixed_loop_modules_available"],
            "runtime_advantage_claimed": baseline["runtime_advantage_claimed"],
            "distinct_phase_resource_established": result["distinct_phase_resource_established"],
            "computational_advantage": result["computational_advantage"],
        },
        "mechanism_survives": mechanism_survives,
        "scope_caveat": "finite-edge numerical CP2/U2 source package plus matrix-level parity; not independent Wilczek-Zee geometry, not CATVM custody, not a resource separation.",
        "supported_split": {
            "source_package": "SOURCE_WILCZEK_ZEE_PACKAGE_REPRODUCED_STRICT_SCOPE",
            "branch_local_abstraction": "BRANCH_LOCAL_NONCOMMUTING_HIDDEN_FRAME_TRANSACTION_DEMONSTRATED",
            "not_established": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_WILCZEK_ZEE_CARRIER_MECHANISM",
        },
        "missing_independent_controls": [
            "independent CP2 bright-ray reconstruction",
            "independent dark-frame basis construction",
            "edge polar decomposition rebuild",
            "gauge transformation test",
            "alternate segment counts",
            "alternate loop parameters",
            "third loop",
            "chart-boundary behavior",
        ],
        "classification": "SOURCE_WILCZEK_ZEE_PACKAGE_REPRODUCED_STRICT_SCOPE" if mechanism_survives else "REJECTED_HOLONOMY_OR_RESTORATION_DEFECT",
    }


def render(report: dict) -> str:
    return f"""# Non-Abelian Holonomy Report

Candidate M: Wilczek-Zee shared phase-frame holonomy.

Mechanism survived: `{report['mechanism_survives']}`
Classification: `{report['classification']}`

V3 reconstruction actually performed:

- Recomputed the primary-vs-reordered boundary Frobenius separation from reproduced 2×2 matrices: `{report['matrix_checks']['primary_reordered_frobenius']}`.
- Recomputed primary/reuse unitarity defects: `{report['matrix_checks']['primary_unitarity_frobenius']}`, `{report['matrix_checks']['reuse_unitarity_frobenius']}`.
- Recomputed discrete formula parity errors: `{report['matrix_checks']['primary_formula_frobenius']}`, `{report['matrix_checks']['reuse_formula_frobenius']}`.
- Confirmed mpmath oracle did not import the production backend and agreed on noncommutation.

Finding:

The source Wilczek-Zee package reproduced in strict scope, and the branch-local toy harness separately demonstrates a noncommuting reversible hidden-frame transaction. V3 did not independently rebuild the Wilczek-Zee geometry, so this report no longer claims independently verified Wilczek-Zee transfer.

Baseline discipline:

The strongest compact classical baseline is the identical 2×2 matrix recurrence, with closed-form fixed-loop modules available. Therefore this is not a non-collapse resource separation, not a Small Wall result, and not physical Family 10h evidence.
"""


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    report = check_m()
    (RAW / "m_independent_data.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "NONABELIAN_HOLONOMY_REPORT.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"M": report["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
