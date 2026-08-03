#!/usr/bin/env python3
"""Test global cyclotomic-unit balancing on the M141 shared-scale state.

The accepted diagnostic uses the existing exact, log-free 49-direction line
descent.  A separately implemented 65,536-bit log-embedding lattice proposal
is retained only as a parity control.  Both act on the actual exact M140 final
state after M141 shared-denominator and common-pi extraction.

This is a bounded test of two declared balancing methods, not a global unit-
lattice optimum or a lower bound for every exact representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f17_anisotropic_radial_shared_scale_height_diagnostic as shared
import f17_cubic_chain_period17_pi_unit_exact_coordinate_descent as descent
import f17_cubic_chain_period17_pi_unit_lattice_center as lattice
import f17_matrix_free_anisotropic_radial_phase_fourier as matrix_free
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_LOG_FREE_49_DIRECTION_AND_FIXED_PRECISION_LOG_EMBEDDING_"
    "GLOBAL_CYCLOTOMIC_UNIT_BALANCERS_AGREE_ON_ALL_21_COMMON_PI_FACTORED_"
    "ANISOTROPIC_F17_RADIAL_FINAL_STATES_REPAIR_THE_DEPTH1_PI_RESIDUAL_"
    "BLOWUP_BUT_SELECT_THE_IDENTITY_AT_EVERY_DEPTH2_PLUS_CASE_THROUGH_"
    "DEPTH64_WHILE_EXACT_RECONSTRUCTION_RESTORATION_REUSE_AND_THE_"
    "IDENTICAL_CLASSICAL_BALANCING_PATH_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def reconstruct_balanced(
    alg: backend.Algebra,
    balanced: list[tuple[int, ...]],
    unit_ledger: list[int],
    denominator_exponent: int,
    common_pi_exponent: int,
) -> list[Any]:
    unit_scale = descent.tracked_ledger_scale(unit_ledger)
    pi_power = descent.cyclo.ring_one()
    for _ in range(common_pi_exponent):
        pi_power = descent.cyclo.ring_multiply(pi_power, shared.height.PI)
    denominator = shared.field_integer(alg, P ** denominator_exponent)
    result = []
    for residual in balanced:
        original_residual = descent.cyclo.ring_multiply(unit_scale, residual)
        numerator = descent.cyclo.ring_multiply(pi_power, original_residual)
        result.append(
            alg.divide(shared.ring_to_field(alg, numerator), denominator)
        )
    return result


def balance_signature(
    scale: shared.SharedScale,
) -> tuple[dict[str, Any], list[tuple[int, ...]], list[int]]:
    source = list(scale.residuals)
    exact_stats = descent.ExactSearchStats()
    exact_residual, exact_ledger = descent.balance_vector(
        source,
        [0 for _ in range(descent.UNIT_RANK)],
        exact_stats,
    )
    proposal_stats = lattice.BalanceStats()
    proposal_residual, proposal_ledger = lattice.balance_vector(
        source,
        [0 for _ in range(lattice.UNIT_RANK)],
        proposal_stats,
    )
    if (
        exact_residual != proposal_residual
        or exact_ledger != proposal_ledger
    ):
        fail("exact and fixed-precision unit balancers disagree")

    residual_bits = descent.base.vector_payload_bits(source)
    balanced_bits = descent.base.vector_payload_bits(exact_residual)
    unit_ledger_bits = descent.base.ledger_payload_bits(exact_ledger)
    scale_ledger_bits = (
        shared.signed_bits(scale.denominator_exponent)
        + shared.signed_bits(scale.common_pi_exponent)
    )
    return (
        {
            "pi_residual_payload_bits": residual_bits,
            "balanced_residual_payload_bits": balanced_bits,
            "unit_ledger_payload_bits": unit_ledger_bits,
            "scale_ledger_payload_bits": scale_ledger_bits,
            "balanced_all_ledgers_payload_bits": (
                balanced_bits + unit_ledger_bits + scale_ledger_bits
            ),
            "pi_residual_maximum_signed_bits": (
                descent.base.vector_width(source)
            ),
            "balanced_residual_maximum_signed_bits": (
                descent.base.vector_width(exact_residual)
            ),
            "unit_ledger": exact_ledger,
            "unit_ledger_is_identity": not any(exact_ledger),
            "exact_coordinate_sweeps": (
                exact_stats.coordinate_sweeps_completed
            ),
            "exact_coordinate_moves": exact_stats.coordinate_moves_accepted,
            "exact_coordinatewise_certified": (
                exact_stats.coordinatewise_certified_calls == 1
            ),
            "exact_sweep_cap_hits": exact_stats.coordinate_sweep_cap_hits,
            "exact_bracket_cap_hits": exact_stats.coordinate_bracket_cap_hits,
            "exact_candidate_evaluations": (
                exact_stats.balance_candidate_evaluations
            ),
            "fixed_precision_selected_steps": (
                proposal_stats.balance_selected_steps
            ),
            "fixed_precision_candidate_evaluations": (
                proposal_stats.balance_candidate_evaluations
            ),
            "fixed_precision_nonpositive_embedding_rejections": (
                proposal_stats.fixed_precision_nonpositive_embedding_rejections
            ),
            "balancer_residuals_and_ledgers_equal": True,
        },
        exact_residual,
        exact_ledger,
    )


def execute_case(
    geometry: matrix_free.MatrixFreeGeometry,
    depth: int,
    family: str,
) -> dict[str, Any]:
    carrier = matrix_free.MatrixFreeCarrier.create(geometry)
    program = matrix_free.compile_program(depth, family)
    backing = carrier.backing_identity()
    matrix_free.begin_forward(carrier, program)
    matrix_free.forward(carrier, program)
    commitment, _, _ = matrix_free.state_commitment(carrier)
    scale, scale_metrics = shared.compile_shared_scale(
        carrier.alg, carrier.cells
    )
    metrics, balanced, unit_ledger = balance_signature(scale)
    if reconstruct_balanced(
        carrier.alg,
        balanced,
        unit_ledger,
        scale.denominator_exponent,
        scale.common_pi_exponent,
    ) != carrier.cells:
        fail("balanced representation failed exact carrier reconstruction")
    boundary = matrix_free.project(carrier, program)
    matrix_free.inverse(carrier, program)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "final_state_commitment": commitment,
        "final_boundary": carrier.alg.serialize(boundary),
        "denominator_power_of_17": scale.denominator_exponent,
        "common_pi_exponent": scale.common_pi_exponent,
        "m141_common_pi_normalized_payload_bits": (
            scale_metrics["common_pi_normalized_payload_bits"]
        ),
        **metrics,
        "exact_balanced_reconstruction_equal": True,
        "same_backing": carrier.backing_identity() == backing,
        "restored_exact_zero": carrier.exact_zero(),
        "package_local_restoration_count": (
            carrier.package_local_restoration_count
        ),
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "resident_carrier_restoration_class": (
            "EXACT_ALGEBRAIC_RESTORATION"
        ),
        "unit_balance_buffer_restoration_class": "NO_RESTORATION_CLAIM",
    }


def reuse_control(
    geometry: matrix_free.MatrixFreeGeometry,
) -> dict[str, Any]:
    carrier = matrix_free.MatrixFreeCarrier.create(geometry)
    backing = carrier.backing_identity()

    def transaction(depth: int, family: str) -> tuple[Any, dict[str, Any]]:
        program = matrix_free.compile_program(depth, family)
        matrix_free.begin_forward(carrier, program)
        matrix_free.forward(carrier, program)
        scale, _ = shared.compile_shared_scale(carrier.alg, carrier.cells)
        signature, balanced, ledger = balance_signature(scale)
        if reconstruct_balanced(
            carrier.alg,
            balanced,
            ledger,
            scale.denominator_exponent,
            scale.common_pi_exponent,
        ) != carrier.cells:
            fail("reuse balance reconstruction changed resident state")
        boundary = matrix_free.project(carrier, program)
        matrix_free.inverse(carrier, program)
        return boundary, signature

    transaction(8, "PRIMARY")
    restored_boundary, restored_signature = transaction(16, "REUSE")
    fresh = matrix_free.MatrixFreeCarrier.create(geometry)
    program = matrix_free.compile_program(16, "REUSE")
    matrix_free.begin_forward(fresh, program)
    matrix_free.forward(fresh, program)
    scale, _ = shared.compile_shared_scale(fresh.alg, fresh.cells)
    fresh_signature, balanced, ledger = balance_signature(scale)
    if reconstruct_balanced(
        fresh.alg,
        balanced,
        ledger,
        scale.denominator_exponent,
        scale.common_pi_exponent,
    ) != fresh.cells:
        fail("fresh balance reconstruction changed resident state")
    fresh_boundary = matrix_free.project(fresh, program)
    matrix_free.inverse(fresh, program)
    return {
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_boundary_equal": restored_boundary == fresh_boundary,
        "fresh_restored_balance_signature_equal": (
            restored_signature == fresh_signature
        ),
        "restored_exact_zero": carrier.exact_zero(),
        "package_local_restoration_count": (
            carrier.package_local_restoration_count
        ),
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def run(m141_path: Path) -> dict[str, Any]:
    predecessor = json.loads(m141_path.read_text(encoding="utf-8"))
    geometry = matrix_free.MatrixFreeGeometry.compile(
        backend.Algebra("Q_ZETA17")
    )
    cases = [
        execute_case(geometry, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    predecessor_cases = {
        (case["family"], case["depth"]): case
        for case in predecessor["cases"]
    }
    for case in cases:
        prior = predecessor_cases[(case["family"], case["depth"])]
        for field in (
            "program_fingerprint",
            "final_state_commitment",
            "final_boundary",
            "denominator_power_of_17",
            "common_pi_exponent",
        ):
            if case[field] != prior[field]:
                fail(f"unit diagnostic diverged from M141 field {field}")
        if not (
            case["exact_balanced_reconstruction_equal"]
            and case["balancer_residuals_and_ledgers_equal"]
            and case["exact_coordinatewise_certified"]
            and case["exact_sweep_cap_hits"] == 0
            and case["exact_bracket_cap_hits"] == 0
            and case["fixed_precision_nonpositive_embedding_rejections"] == 0
            and case["same_backing"]
            and case["restored_exact_zero"]
            and case["package_local_restoration_count"] == 1
            and not case["snapshot_reload_used"]
        ):
            fail("unit diagnostic case control failed")

    depth1 = [case for case in cases if case["depth"] == 1]
    depth2_plus = [case for case in cases if case["depth"] >= 2]
    if not all(
        not case["unit_ledger_is_identity"]
        and case["balanced_all_ledgers_payload_bits"]
        < case["m141_common_pi_normalized_payload_bits"]
        for case in depth1
    ):
        fail("declared depth1 unit repair did not reproduce")
    if not all(case["unit_ledger_is_identity"] for case in depth2_plus):
        fail("a depth2-plus case selected a nonidentity unit")

    endpoints = {
        family: {
            "depth1_all_ledgers_payload_bits": next(
                case["balanced_all_ledgers_payload_bits"]
                for case in cases
                if case["family"] == family and case["depth"] == 1
            ),
            "depth64_all_ledgers_payload_bits": next(
                case["balanced_all_ledgers_payload_bits"]
                for case in cases
                if case["family"] == family and case["depth"] == 64
            ),
        }
        for family in FAMILIES
    }
    if not all(
        values["depth64_all_ledgers_payload_bits"]
        > values["depth1_all_ledgers_payload_bits"]
        for values in endpoints.values()
    ):
        fail("unit-balanced endpoint payload did not grow")

    reuse = reuse_control(geometry)
    if not (
        reuse["same_original_backing"]
        and reuse["fresh_restored_boundary_equal"]
        and reuse["fresh_restored_balance_signature_equal"]
        and reuse["restored_exact_zero"]
        and reuse["package_local_restoration_count"] == 2
        and not reuse["snapshot_reload_used"]
        and reuse["inverse_history_cells"] == 0
    ):
        fail("unit diagnostic reuse failed")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_GLOBAL_UNIT_BALANCE_NO_GO_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m141_result_sha256": hashlib.sha256(m141_path.read_bytes()).hexdigest(),
        "source_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "exact_balancer": "LOG_FREE_49_DIRECTION_EXACT_LINE_DESCENT",
            "separate_parity_balancer": (
                "65536_BIT_LOG_EMBEDDING_LATTICE_CENTER_WITH_EXACT_ACCEPTANCE"
            ),
        },
        "cases": cases,
        "endpoints": endpoints,
        "all_three_depth1_cases_repaired": True,
        "all_18_depth2_plus_cases_select_identity_unit": True,
        "all_21_balancer_outputs_equal": True,
        "all_21_exact_reconstructions_equal": True,
        "reuse": reuse,
        "resource_law": {
            "resident_radial_exact_field_cells": 17,
            "represented_cyclotomic_integer_coefficients": 272,
            "unit_ledger_integers": 7,
            "shared_denominator_ledger_integers": 1,
            "common_pi_ledger_integers": 1,
            "compiled_exact_direction_count": len(descent.SEARCH_DIRECTIONS),
            "compiled_exact_unit_table_payload_bits": (
                descent.compiled_unit_table_payload_bits()
            ),
            "fixed_precision_proposal_bits": lattice.EMBEDDING_PRECISION_BITS,
            "fixed_precision_path_is_control_not_accepted_balancer": True,
            "python_container_allocator_sympy_mpmath_native_bigint_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_classical_baseline": {
            "method": "IDENTICAL_SHARED_SCALE_AND_EXACT49_DIRECTION_GLOBAL_UNIT_BALANCING_RECURRENCE",
            "same_representation_search_ledgers_and_payload_law": True,
            "universal_optimality_or_lower_bound_claimed": False,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "restoration": {
            "resident_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "unit_search_and_scale_buffers": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "fresh_restored_reuse_equal": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "restoration_count_is_package_local_not_catvm_generation": True,
        },
        "claim_boundary": {
            "established": [
                "BOUNDED_DEPTH1_COMMON_PI_RESIDUAL_PAYLOAD_REPAIR",
                "BOUNDED_IDENTITY_SELECTION_BY_TWO_BALANCERS_ON18_DEPTH2_PLUS_CASES",
                "EXACT_BALANCED_STATE_RECONSTRUCTION_BEFORE_FINAL_PROJECTION_AND_INVERSE",
                "EXACT_SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE",
            ],
            "not_established": [
                "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
                "LOWER_BOUND_FOR_ALL_EXACT_REPRESENTATIONS",
                "ASYMPTOTIC_HEIGHT_LOWER_BOUND",
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_EXECUTION",
                "PHYSICAL_BIT_REPLACEMENT",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "THE_DECLARED_GLOBAL_UNIT_BALANCERS_REPAIR_ONLY_THE_DEPTH1_PI_"
            "EXTRACTION_ARTIFACT_AND_SELECT_IDENTITY_FROM_DEPTH2_ONWARD_"
            "WHILE_EXACT_RESIDUAL_HEIGHT_AND_THE_IDENTICAL_CLASSICAL_PATH_REMAIN"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m141", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frontier = Path(__file__).resolve().parent
    m141_path = args.m141 or (
        frontier / "F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_DIAGNOSTIC_RESULTS.json"
    )
    result = run(m141_path)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
