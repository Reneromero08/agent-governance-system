#!/usr/bin/env python3
"""Resource-capped exact lift probe for period-17 Krylov dependencies.

The modular period-block diagnostic establishes seed-image dimensions but
does not lift its dependence polynomials to Z[zeta_17].  This probe tracks
the modular dependence coefficients, combines them by CRT, and tests the
candidate directly against a streamed exact fixed-basis integer recurrence.
No rational matrix or dense period-block operator is materialized.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_adaptive_gauge as adaptive
import f17_cubic_chain_period17_krylov as period17


PRIME = 17
DIMENSION = 16
MESSAGE_CELLS = PRIME * DIMENSION
PERIOD = 17
CRT_MODULI = (
    41,
    73,
    65537,
    65539,
    65543,
    65551,
    65557,
    65563,
)
MAX_DEPENDENCE_DEGREE = MESSAGE_CELLS
MAX_EXACT_VECTOR_SIGNED_BITS = 16384


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    if value == 0:
        return 1
    return abs(value).bit_length() + 1


@dataclass
class DependencyStats:
    row_elimination_updates: int = 0
    polynomial_elimination_updates: int = 0
    normalization_updates: int = 0
    period_applications: int = 0


def modular_dependency(
    family: str,
    modulus: int,
) -> dict[str, Any]:
    program = adaptive.compile_program(PERIOD + 1, family)
    current = period17.fixed_seed(program, modulus)
    basis: dict[int, tuple[list[int], list[int]]] = {}
    stats = DependencyStats()
    polynomial_cells = MAX_DEPENDENCE_DEGREE + 1
    for step in range(MAX_DEPENDENCE_DEGREE + 1):
        work = current[:]
        polynomial = [0] * polynomial_cells
        polynomial[step] = 1
        for pivot in sorted(basis):
            factor = work[pivot]
            if factor == 0:
                continue
            row, row_polynomial = basis[pivot]
            for index in range(pivot, MESSAGE_CELLS):
                work[index] = (
                    work[index] - factor * row[index]
                ) % modulus
                stats.row_elimination_updates += 1
            for index in range(step + 1):
                polynomial[index] = (
                    polynomial[index]
                    - factor * row_polynomial[index]
                ) % modulus
                stats.polynomial_elimination_updates += 1
        pivot = next(
            (
                index
                for index, value in enumerate(work)
                if value
            ),
            None,
        )
        if pivot is None:
            coefficients = polynomial[:step + 1]
            if coefficients[-1] != 1:
                fail("modular dependence was not monic")
            return {
                "modulus": modulus,
                "degree": step,
                "coefficients": coefficients,
                "coefficient_sha256": hashlib.sha256(
                    json.dumps(
                        coefficients,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "basis_rows": len(basis),
                "basis_message_field_cells": (
                    len(basis) * MESSAGE_CELLS
                ),
                "basis_polynomial_field_cells": (
                    len(basis) * polynomial_cells
                ),
                "current_work_and_polynomial_field_cells": (
                    2 * MESSAGE_CELLS + polynomial_cells
                ),
                "combined_explicit_peak_field_cells": (
                    len(basis)
                    * (MESSAGE_CELLS + polynomial_cells)
                    + 2 * MESSAGE_CELLS
                    + polynomial_cells
                ),
                "operation_counts": {
                    "row_elimination_updates": (
                        stats.row_elimination_updates
                    ),
                    "polynomial_elimination_updates": (
                        stats.polynomial_elimination_updates
                    ),
                    "normalization_updates": (
                        stats.normalization_updates
                    ),
                    "period_applications": stats.period_applications,
                },
            }
        inverse = pow(work[pivot], modulus - 2, modulus)
        for index in range(pivot, MESSAGE_CELLS):
            work[index] = work[index] * inverse % modulus
            stats.normalization_updates += 1
        for index in range(step + 1):
            polynomial[index] = polynomial[index] * inverse % modulus
            stats.normalization_updates += 1
        basis[pivot] = (work, polynomial)
        current = period17.apply_period(
            current,
            program,
            modulus,
        )
        stats.period_applications += 1
    fail("no modular dependence within the declared message dimension")


def crt_pair(
    left: int,
    left_modulus: int,
    right: int,
    right_modulus: int,
) -> int:
    correction = (
        (right - left)
        * pow(left_modulus, -1, right_modulus)
    ) % right_modulus
    return left + left_modulus * correction


def combine_dependencies(
    dependencies: list[dict[str, Any]],
) -> dict[str, Any]:
    degrees = {entry["degree"] for entry in dependencies}
    if len(degrees) != 1:
        fail("modular dependence degrees disagree")
    degree = degrees.pop()
    residues = dependencies[0]["coefficients"][:]
    combined_modulus = dependencies[0]["modulus"]
    centered = [
        value
        if value <= combined_modulus // 2
        else value - combined_modulus
        for value in residues
    ]
    stages = [
        {
            "moduli_used": [dependencies[0]["modulus"]],
            "combined_modulus_bits": combined_modulus.bit_length(),
            "maximum_candidate_coefficient_signed_bits": max(
                signed_bits(value)
                for value in centered
            ),
            "coefficients_unchanged_from_previous_stage": None,
            "candidate_sha256": hashlib.sha256(
                json.dumps(
                    centered,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    ]
    for dependency in dependencies[1:]:
        modulus = dependency["modulus"]
        previous_centered = centered
        residues = [
            crt_pair(
                left,
                combined_modulus,
                right,
                modulus,
            )
            for left, right in zip(
                residues,
                dependency["coefficients"],
                strict=True,
            )
        ]
        combined_modulus *= modulus
        centered = [
            value
            if value <= combined_modulus // 2
            else value - combined_modulus
            for value in residues
        ]
        stages.append(
            {
                "moduli_used": [
                    entry["modulus"]
                    for entry in dependencies[:len(stages) + 1]
                ],
                "combined_modulus_bits": (
                    combined_modulus.bit_length()
                ),
                "maximum_candidate_coefficient_signed_bits": max(
                    signed_bits(value)
                    for value in centered
                ),
                "coefficients_unchanged_from_previous_stage": sum(
                    left == right
                    for left, right in zip(
                        previous_centered,
                        centered,
                        strict=True,
                    )
                ),
                "candidate_sha256": hashlib.sha256(
                    json.dumps(
                        centered,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    if centered[-1] != 1:
        fail("CRT candidate was not monic")
    return {
        "degree": degree,
        "combined_modulus": combined_modulus,
        "combined_modulus_bits": combined_modulus.bit_length(),
        "maximum_candidate_coefficient_signed_bits": max(
            signed_bits(value)
            for value in centered
        ),
        "nonzero_candidate_coefficients": sum(
            value != 0
            for value in centered
        ),
        "crt_stages": stages,
        "candidate_coefficients": centered,
        "candidate_sha256": hashlib.sha256(
            json.dumps(
                centered,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def exact_seed(
    program: adaptive.ChainProgram,
) -> list[int]:
    message = [0] * MESSAGE_CELLS
    for value in range(PRIME):
        phase = adaptive.unary_phase(
            program.unary_coefficients[0],
            value,
        )
        offset = value * DIMENSION
        if phase < DIMENSION:
            message[offset + phase] = 1
        else:
            for basis in range(DIMENSION):
                message[offset + basis] = -1
    return message


def exact_transfer(
    source: list[int],
    program: adaptive.ChainProgram,
    edge_index: int,
) -> tuple[list[int], int]:
    target = [0] * MESSAGE_CELLS
    updates = 0
    unary = program.unary_coefficients[edge_index + 1]
    edge = program.edge_coefficients[edge_index]
    for left in range(PRIME):
        source_offset = left * DIMENSION
        for right in range(PRIME):
            shift = (
                adaptive.unary_phase(unary, right)
                + adaptive.edge_phase(edge, left, right)
            ) % PRIME
            target_offset = right * DIMENSION
            for basis in range(DIMENSION):
                coefficient = source[source_offset + basis]
                if coefficient == 0:
                    continue
                exponent = (basis + shift) % PRIME
                if exponent < DIMENSION:
                    target[target_offset + exponent] += coefficient
                    updates += 1
                else:
                    for output_basis in range(DIMENSION):
                        target[target_offset + output_basis] -= (
                            coefficient
                        )
                        updates += 1
    return target, updates


def exact_period(
    source: list[int],
    program: adaptive.ChainProgram,
) -> tuple[list[int], int]:
    current = source
    updates = 0
    for edge_index in range(PERIOD):
        current, transfer_updates = exact_transfer(
            current,
            program,
            edge_index,
        )
        updates += transfer_updates
    return current, updates


def verify_exact_candidate(
    family: str,
    coefficients: list[int],
) -> dict[str, Any]:
    degree = len(coefficients) - 1
    program = adaptive.compile_program(PERIOD + 1, family)
    current = exact_seed(program)
    residual = [0] * MESSAGE_CELLS
    maximum_iterate_bits = 1
    period_applications = 0
    transfer_integer_updates = 0
    residual_axpy_updates = 0
    for power, coefficient in enumerate(coefficients):
        maximum_iterate_bits = max(
            maximum_iterate_bits,
            max(signed_bits(value) for value in current),
        )
        if maximum_iterate_bits > MAX_EXACT_VECTOR_SIGNED_BITS:
            return {
                "completed": False,
                "resource_cap_reached": True,
                "degree": degree,
                "stopped_at_power": power,
                "maximum_iterate_signed_bits": maximum_iterate_bits,
                "candidate_residual_zero": False,
            }
        if coefficient:
            for index, value in enumerate(current):
                residual[index] += coefficient * value
                residual_axpy_updates += 1
        if power != degree:
            current, updates = exact_period(current, program)
            transfer_integer_updates += updates
            period_applications += 1
    nonzero = [value for value in residual if value]
    return {
        "completed": True,
        "resource_cap_reached": False,
        "degree": degree,
        "period_applications": period_applications,
        "transfer_integer_updates": transfer_integer_updates,
        "residual_axpy_updates": residual_axpy_updates,
        "maximum_iterate_signed_bits": maximum_iterate_bits,
        "residual_nonzero_cells": len(nonzero),
        "maximum_residual_signed_bits": max(
            (signed_bits(value) for value in nonzero),
            default=1,
        ),
        "candidate_residual_zero": not nonzero,
        "streaming_exact_message_integer_cells": (
            2 * MESSAGE_CELLS
        ),
        "residual_integer_cells": MESSAGE_CELLS,
        "combined_explicit_peak_integer_cells": (
            3 * MESSAGE_CELLS
        ),
    }


def family_case(family: str) -> dict[str, Any]:
    dependencies = [
        modular_dependency(family, modulus)
        for modulus in CRT_MODULI
    ]
    candidate = combine_dependencies(dependencies)
    exact = verify_exact_candidate(
        family,
        candidate["candidate_coefficients"],
    )
    candidate_summary = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_coefficients"
    }
    return {
        "family": family,
        "modular_dependencies": dependencies,
        "crt_candidate": candidate_summary,
        "exact_verification": exact,
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_cubic_chain_period17_exact_lift.py")
    cases = [family_case(family) for family in ("PRIMARY", "REUSE")]
    all_degrees_match_prior = (
        cases[0]["crt_candidate"]["degree"] == 241
        and cases[1]["crt_candidate"]["degree"] == 256
    )
    exact_lift_established = all(
        case["exact_verification"]["candidate_residual_zero"]
        for case in cases
    )
    result = {
        "result": "PASS",
        "experiment": (
            "RESOURCE_CAPPED_EXACT_Z_ZETA17_LIFT_OF_PERIOD17_"
            "MODULAR_KRYLOV_DEPENDENCIES"
        ),
        "scope": (
            "TWO_PUBLIC_F17_PERIOD17_CUBIC_CHAIN_FAMILIES_"
            "EIGHT_PRIME_CRT_STREAMED_EXACT_FIXED_BASIS_"
            "VERIFICATION_SOFTWARE_ONLY"
        ),
        "crt_moduli": list(CRT_MODULI),
        "cases": cases,
        "all_degrees_match_prior_modular_dimensions": (
            all_degrees_match_prior
        ),
        "exact_z_zeta17_dependency_lift_established": (
            exact_lift_established
        ),
        "dense_period_block_materialized": False,
        "rational_matrix_materialized": False,
        "assignment_table_materialized": False,
        "relation_table_materialized": False,
        "python_object_overhead_bounded": False,
        "allocator_peak_bounded": False,
        "bit_operation_peak_bounded": False,
        "whole_process_peak_bounded": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    if not all_degrees_match_prior:
        fail("tracked dependence degrees do not match prior modular ranks")
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
