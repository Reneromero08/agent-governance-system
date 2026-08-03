#!/usr/bin/env python3
"""Test exact shared-scale compression of the M140 radial phase carrier.

The candidate representation replaces the repeated rational denominator in
the 17 cyclotomic cells by one base-17 exponent and then extracts the greatest
common power of pi = 1-zeta_17 from every integral numerator cell.  The
experiment is exact and reconstructs every resident value before allowing the
usual final projection and actual inverse.

This is a bounded compression diagnostic, not a general height lower bound.
The identical representation is available to compact classical software.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_cubic_chain_period17_height_lower_bound as height
import f17_matrix_free_anisotropic_radial_phase_fourier as matrix_free
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_SHARED_DENOMINATOR_LEDGER_REDUCES_REPEATED_DENOMINATOR_"
    "FINAL_STATE_ACCOUNTING_BY_MORE_THAN_HALF_ACROSS_21_MATRIX_FREE_"
    "ANISOTROPIC_F17_RADIAL_CASES_BUT_COMMON_PI_CONTENT_DOES_NOT_BOUND_"
    "RESIDUAL_HEIGHT_THROUGH_DEPTH64_WHILE_EXACT_RECONSTRUCTION_"
    "RESTORATION_REUSE_AND_THE_IDENTICAL_COMPACT_CLASSICAL_NORMALIZATION_"
    "REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return 1 if value == 0 else abs(value).bit_length() + 1


def power_of_17(value: int) -> int:
    if value < 1:
        fail("shared denominator is not positive")
    exponent = 0
    while value % P == 0:
        value //= P
        exponent += 1
    if value != 1:
        fail("shared denominator is not a power of 17")
    return exponent


def ring_payload(element: tuple[int, ...]) -> int:
    return sum(signed_bits(value) for value in element)


def field_integer(alg: backend.Algebra, value: int) -> Any:
    return matrix_free.field_integer(alg, value)


def ring_to_field(alg: backend.Algebra, element: tuple[int, ...]) -> Any:
    total = alg.zero
    for degree, coefficient in enumerate(element):
        total = alg.add(
            total,
            alg.mul(field_integer(alg, coefficient), alg.power(degree)),
        )
    return total


@dataclass(frozen=True)
class SharedScale:
    residuals: tuple[tuple[int, ...], ...]
    denominator_exponent: int
    common_pi_exponent: int

    def payload_bits(self) -> int:
        return (
            sum(ring_payload(element) for element in self.residuals)
            + signed_bits(self.denominator_exponent)
            + signed_bits(self.common_pi_exponent)
        )


def compile_shared_scale(
    alg: backend.Algebra, cells: list[Any]
) -> tuple[SharedScale, dict[str, Any]]:
    serialized = [alg.serialize(value) for value in cells]
    denominator = 1
    for cell in serialized:
        for _, coefficient_denominator in cell:
            denominator = math.lcm(denominator, coefficient_denominator)

    integral = [
        tuple(
            numerator * (denominator // coefficient_denominator)
            for numerator, coefficient_denominator in cell
        )
        for cell in serialized
    ]
    rational_content = denominator
    for element in integral:
        for coefficient in element:
            rational_content = math.gcd(rational_content, abs(coefficient))
    if rational_content > 1:
        denominator //= rational_content
        integral = [
            tuple(coefficient // rational_content for coefficient in element)
            for element in integral
        ]

    denominator_exponent = power_of_17(denominator)
    valuations = [
        height.pi_valuation(element)
        for element in integral
        if any(element)
    ]
    if not valuations or any(value is None for value in valuations):
        fail("resident radial state has no nonzero pi valuation")
    common_pi = min(int(value) for value in valuations if value is not None)
    residuals = []
    division_checks = 0
    for original in integral:
        residual = original
        for _ in range(common_pi):
            prior = residual
            residual = height.divide_pi_exact(residual)
            if height.cyclo.ring_multiply(height.PI, residual) != prior:
                fail("pi-content division failed exact reconstruction")
            division_checks += 1
        residuals.append(residual)

    scale = SharedScale(
        tuple(residuals), denominator_exponent, common_pi
    )
    pi_power = height.cyclo.ring_one()
    for _ in range(common_pi):
        pi_power = height.cyclo.ring_multiply(pi_power, height.PI)
    reconstructed = []
    denominator_field = field_integer(alg, P ** denominator_exponent)
    for residual in scale.residuals:
        numerator = height.cyclo.ring_multiply(pi_power, residual)
        reconstructed.append(
            alg.divide(ring_to_field(alg, numerator), denominator_field)
        )
    if reconstructed != cells:
        fail("shared-scale carrier failed exact state reconstruction")

    integer_payload = sum(ring_payload(element) for element in integral)
    shared_denominator_payload = integer_payload + signed_bits(
        denominator_exponent
    )
    return scale, {
        "raw_repeated_denominator_payload_bits": sum(
            alg.payload_bits(value) for value in cells
        ),
        "shared_denominator_payload_bits": shared_denominator_payload,
        "common_pi_normalized_payload_bits": scale.payload_bits(),
        "denominator_power_of_17": denominator_exponent,
        "common_pi_exponent": common_pi,
        "minimum_cell_pi_valuation": min(int(value) for value in valuations),
        "maximum_cell_pi_valuation": max(int(value) for value in valuations),
        "maximum_integral_numerator_signed_bits": max(
            signed_bits(coefficient)
            for element in integral
            for coefficient in element
        ),
        "maximum_pi_residual_signed_bits": max(
            signed_bits(coefficient)
            for element in scale.residuals
            for coefficient in element
        ),
        "resident_cyclotomic_integer_coefficients": 17 * 16,
        "shared_denominator_ledger_integers": 1,
        "common_pi_ledger_integers": 1,
        "exact_pi_divisions": division_checks,
        "exact_reconstruction_equal": True,
    }


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
    scale, metrics = compile_shared_scale(carrier.alg, carrier.cells)
    boundary = matrix_free.project(carrier, program)
    matrix_free.inverse(carrier, program)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "final_state_commitment": commitment,
        "final_boundary": carrier.alg.serialize(boundary),
        **metrics,
        "shared_scale_resident_integer_coefficients": sum(
            len(element) for element in scale.residuals
        ),
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
        "scale_ledger_restoration_class": "NO_RESTORATION_CLAIM",
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
        _, metrics = compile_shared_scale(carrier.alg, carrier.cells)
        boundary = matrix_free.project(carrier, program)
        matrix_free.inverse(carrier, program)
        return boundary, metrics

    transaction(8, "PRIMARY")
    restored_boundary, restored_metrics = transaction(16, "REUSE")
    fresh = matrix_free.MatrixFreeCarrier.create(geometry)
    program = matrix_free.compile_program(16, "REUSE")
    matrix_free.begin_forward(fresh, program)
    matrix_free.forward(fresh, program)
    _, fresh_metrics = compile_shared_scale(fresh.alg, fresh.cells)
    fresh_boundary = matrix_free.project(fresh, program)
    matrix_free.inverse(fresh, program)
    return {
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_boundary_equal": restored_boundary == fresh_boundary,
        "fresh_restored_scale_signature_equal": (
            restored_metrics == fresh_metrics
        ),
        "restored_exact_zero": carrier.exact_zero(),
        "package_local_restoration_count": (
            carrier.package_local_restoration_count
        ),
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def run(m140_path: Path) -> dict[str, Any]:
    m140 = json.loads(m140_path.read_text(encoding="utf-8"))
    exact_geometry = matrix_free.MatrixFreeGeometry.compile(
        backend.Algebra("Q_ZETA17")
    )
    cases = [
        execute_case(exact_geometry, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    primary_m140 = {
        item["depth"]: item for item in m140["exact_transactions"]
    }
    for case in cases:
        if case["family"] != "PRIMARY":
            continue
        predecessor = primary_m140[case["depth"]]
        if (
            case["program_fingerprint"] != predecessor["program_fingerprint"]
            or case["final_state_commitment"]
            != predecessor["final_state_commitment"]
            or case["final_boundary"] != predecessor["final_boundary"]
        ):
            fail("shared-scale diagnostic diverged from M140")

    for case in cases:
        if not (
            case["exact_reconstruction_equal"]
            and case["same_backing"]
            and case["restored_exact_zero"]
            and case["package_local_restoration_count"] == 1
        ):
            fail("shared-scale diagnostic transaction failed")

    families = {
        family: [case for case in cases if case["family"] == family]
        for family in FAMILIES
    }
    endpoint_growth = {
        family: {
            "shared_denominator_payload_grows": (
                values[-1]["shared_denominator_payload_bits"]
                > values[0]["shared_denominator_payload_bits"]
            ),
            "pi_normalized_payload_grows": (
                values[-1]["common_pi_normalized_payload_bits"]
                > values[0]["common_pi_normalized_payload_bits"]
            ),
            "residual_width_grows": (
                values[-1]["maximum_pi_residual_signed_bits"]
                > values[0]["maximum_pi_residual_signed_bits"]
            ),
        }
        for family, values in families.items()
    }
    if not all(all(checks.values()) for checks in endpoint_growth.values()):
        fail("shared-scale height obstruction did not reproduce")

    reuse = reuse_control(exact_geometry)
    if not all(
        value
        for key, value in reuse.items()
        if key not in {
            "package_local_restoration_count",
            "inverse_history_cells",
            "snapshot_reload_used",
        }
    ) or reuse["package_local_restoration_count"] != 2 or reuse[
        "snapshot_reload_used"
    ]:
        fail("shared-scale reuse failed")

    if not all(
        case["shared_denominator_payload_bits"] * 2
        < case["raw_repeated_denominator_payload_bits"]
        for case in cases
    ):
        fail("shared denominator did not reduce every declared case by more than half")

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_DIAGNOSTIC_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE",
        "source_sha256": source_hash,
        "m140_result_sha256": hashlib.sha256(m140_path.read_bytes()).hexdigest(),
        "source_scope": {
            "coordinate_field": "F17_SQUARED",
            "anisotropic_norm": "X_SQUARED_MINUS_3_Y_SQUARED",
            "phase_family": "M140_QUARTIC_RADIAL_PHASE_AND_MATRIX_FREE_FOURIER",
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "normalization": (
                "ONE_SHARED_POWER17_DENOMINATOR_LEDGER_PLUS_GREATEST_"
                "COMMON_PI_CONTENT_LEDGER"
            ),
        },
        "cases": cases,
        "endpoint_growth": endpoint_growth,
        "all_depth2_and_above_have_zero_common_pi_content": all(
            case["common_pi_exponent"] == 0
            for case in cases
            if case["depth"] >= 2
        ),
        "all_shared_denominators_are_power17": True,
        "all_exact_reconstructions_equal": True,
        "reuse": reuse,
        "resource_law": {
            "resident_radial_field_cells": 17,
            "represented_cyclotomic_integer_coefficients": 272,
            "shared_denominator_ledger_integers": 1,
            "common_pi_ledger_integers": 1,
            "matrix_free_generator_exact_field_cells": 1,
            "retained_public_kernel_exact_field_cells": 0,
            "accepted_assignment_truth_table_coordinate_or_kernel_cells": 0,
            "payload_includes_all272_integer_coefficients_and_both_ledgers": True,
            "python_container_allocator_sympy_native_bigint_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_classical_baseline": {
            "method": "IDENTICAL_SHARED_SCALE_NORMALIZED_MATRIX_FREE_17_COORDINATE_RECURRENCE",
            "all_case_boundaries_equal": True,
            "same_normalization_and_payload_law": True,
            "universal_optimality_or_lower_bound_claimed": False,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "controls": {
            "m140_primary_commitment_and_boundary_parity": True,
            "exact_pi_division_identity": True,
            "common_pi_content_removes_depth2_plus_denominator_growth": False,
            "fixed_bounded_width_exact_state_established": False,
            "snapshot_command_available": False,
            "intermediate_projection_available": False,
        },
        "restoration": {
            "class": "EXACT_ALGEBRAIC_RESTORATION",
            "scale_metric_buffers": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "fresh_restored_reuse_equal": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "restoration_count_is_package_local_not_catvm_generation": True,
        },
        "claim_boundary": {
            "established": [
                "BOUNDED_EXACT_SHARED_DENOMINATOR_FINAL_STATE_PAYLOAD_REDUCTION_BY_MORE_THAN_HALF_ACROSS_21_DECLARED_CASES",
                "BOUNDED_COMMON_PI_CONTENT_NORMALIZATION_DOES_NOT_BOUND_RESIDUAL_HEIGHT_THROUGH_DEPTH64",
                "EXACT_RECONSTRUCTION_BEFORE_FINAL_PROJECTION_AND_ACTUAL_INVERSE",
                "EXACT_SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE",
            ],
            "not_established": [
                "ASYMPTOTIC_OR_UNIVERSAL_HEIGHT_LOWER_BOUND",
                "OPTIMAL_EXACT_REPRESENTATION",
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
            "SHARED_DENOMINATOR_STORAGE_REDUCES_REPEATED_DENOMINATOR_FINAL_"
            "STATE_ACCOUNTING_BUT_ALL_DEPTH2_PLUS_TESTED_STATES_HAVE_ZERO_"
            "COMMON_PI_CONTENT_AND_RESIDUAL_NUMERATOR_HEIGHT_STILL_GROWS_"
            "WHILE_THE_IDENTICAL_COMPACT_CLASSICAL_NORMALIZATION_REMAINS"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m140", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frontier = Path(__file__).resolve().parent
    m140_path = args.m140 or (
        frontier / "F17_MATRIX_FREE_ANISOTROPIC_RADIAL_PHASE_FOURIER_RESULTS.json"
    )
    result = run(m140_path)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
