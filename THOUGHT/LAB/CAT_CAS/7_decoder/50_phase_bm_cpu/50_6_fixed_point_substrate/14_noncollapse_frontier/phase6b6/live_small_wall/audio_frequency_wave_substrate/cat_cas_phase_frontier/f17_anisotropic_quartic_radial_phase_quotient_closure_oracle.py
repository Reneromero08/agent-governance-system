#!/usr/bin/env python3
"""Independent oracle for the anisotropic quartic radial quotient package.

This file does not import the production package.  It reconstructs the public
F17 geometry, quotient Fourier matrix, program schedules, exact recurrence,
inverse law, and selected dense 289-coordinate controls directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
D = 3
POINTS = tuple((x, y) for x in range(P) for y in range(P))
FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def qnorm(point: tuple[int, int], coefficient: int = D) -> int:
    x, y = point
    return (x * x - coefficient * y * y) % P


def bilinear(left: tuple[int, int], right: tuple[int, int]) -> int:
    u, v = left
    x, y = right
    return (u * x - D * v * y) % P


def shells(coefficient: int = D) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple(point for point in POINTS if qnorm(point, coefficient) == residue)
        for residue in range(P)
    )


def integer(alg: backend.Algebra, value: int) -> Any:
    return exact.field_integer(alg, value)


def matrix(alg: backend.Algebra) -> tuple[tuple[Any, ...], ...]:
    classes = shells()
    representatives = tuple(group[0] for group in classes)
    inverse17 = alg.inverse(integer(alg, P))
    rows = []
    for target in representatives:
        row = []
        for group in classes:
            value = alg.zero
            for source in group:
                value = alg.add(value, alg.power(bilinear(target, source)))
            row.append(alg.mul(inverse17, value))
        rows.append(tuple(row))
    return tuple(rows)


def matrix_vector(alg: backend.Algebra, transform: tuple[tuple[Any, ...], ...], vector: list[Any]) -> list[Any]:
    output = []
    for row in transform:
        value = alg.zero
        for coefficient, item in zip(row, vector, strict=True):
            value = alg.add(value, alg.mul(coefficient, item))
        output.append(value)
    return output


def gate(index: int, family: str) -> tuple[int, int, int]:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    ternary_weight = 0
    remaining = index
    while remaining:
        ternary_weight += remaining % 3
        remaining //= 3
    if family == "PRIMARY":
        values = (3 * index + 5 * bit_weight + 1, 7 * index + 2 * bit_weight + 2, 11 * index + bit_weight + 4)
    elif family == "REUSE":
        values = (5 * index + 2 * ternary_weight + 3, 4 * index + 3 * ternary_weight + 6, 9 * index + ternary_weight + 8)
    elif family == "ALTERNATE":
        values = (7 * index + 3 * gray_weight + 2, 8 * index + 2 * gray_weight + 5, 6 * index + gray_weight + 1)
    else:
        fail("oracle family outside public schedule")
    first = values[0] % P or 1
    return first, values[1] % P, values[2] % P


def exponent(parameters: tuple[int, int, int], residue: int) -> int:
    first, second, constant = parameters
    return (first * residue * residue + second * residue + constant) % P


def observation(depth: int, family: str) -> tuple[int, int]:
    return (3 * depth + 2 * len(family) + 1) % P or 1, (5 * depth + len(family) + 4) % P


def payload(alg: backend.Algebra, values: list[Any]) -> int:
    return sum(alg.payload_bits(value) for value in values)


def commitment(alg: backend.Algebra, values: list[Any]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        record = json.dumps(alg.serialize(value), separators=(",", ":")).encode()
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest()


def execute(depth: int, family: str, alg: backend.Algebra, transform: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
    state = [alg.one for _ in range(P)]
    maximum_resident_payload = payload(alg, state)
    schedule = [gate(index, family) for index in range(depth)]
    for parameters in schedule:
        state = [
            alg.mul(value, alg.power(exponent(parameters, residue)))
            for residue, value in enumerate(state)
        ]
        maximum_resident_payload = max(maximum_resident_payload, payload(alg, state))
        state = matrix_vector(alg, transform, state)
        maximum_resident_payload = max(maximum_resident_payload, payload(alg, state))
    final_state = list(state)
    first, second = observation(depth, family)
    boundary = alg.zero
    counts = [len(group) for group in shells()]
    for residue, value in enumerate(state):
        phase = alg.power(first * residue * residue + second * residue)
        boundary = alg.add(
            boundary, alg.mul(integer(alg, counts[residue]), alg.mul(phase, value))
        )
    for parameters in reversed(schedule):
        state = matrix_vector(alg, transform, state)
        maximum_resident_payload = max(maximum_resident_payload, payload(alg, state))
        state = [
            alg.mul(value, alg.power(-exponent(parameters, residue)))
            for residue, value in enumerate(state)
        ]
        maximum_resident_payload = max(maximum_resident_payload, payload(alg, state))
    return {
        "depth": depth,
        "family": family,
        "boundary": alg.serialize(boundary),
        "boundary_payload_bits": alg.payload_bits(boundary),
        "maximum_resident_payload_bits": maximum_resident_payload,
        "restored_seed": state == [alg.one for _ in range(P)],
        "final_state": final_state,
        "final_state_commitment": commitment(alg, final_state),
    }


def verify_transform(alg: backend.Algebra, transform: tuple[tuple[Any, ...], ...]) -> dict[str, bool]:
    classes = shells()
    representatives = tuple(group[0] for group in classes)
    invariant = True
    inverse17 = alg.inverse(integer(alg, P))
    for target in POINTS:
        row = []
        for group in classes:
            value = alg.zero
            for source in group:
                value = alg.add(value, alg.power(bilinear(target, source)))
            row.append(alg.mul(inverse17, value))
        invariant &= tuple(row) == transform[qnorm(target)]
    involution = True
    for first in range(P):
        for second in range(P):
            value = alg.zero
            for middle in range(P):
                value = alg.add(
                    value,
                    alg.mul(transform[first][middle], transform[middle][second]),
                )
            involution &= value == (alg.one if first == second else alg.zero)
    return {
        "all_289_target_rows_factor_through_norm": invariant,
        "normalized_quotient_fourier_squares_to_identity": involution,
        "representatives_cover_all_17_norms": all(
            qnorm(representatives[value]) == value for value in range(P)
        ),
    }


def dense(depth: int, family: str, alg: backend.Algebra) -> tuple[Any, bool, list[Any]]:
    state = [[alg.one for _ in range(P)] for _ in range(P)]
    inverse17 = alg.inverse(integer(alg, P))
    for index in range(depth):
        parameters = gate(index, family)
        for x, y in POINTS:
            state[x][y] = alg.mul(
                state[x][y], alg.power(exponent(parameters, qnorm((x, y))))
            )
        first = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for y in range(P):
                for x in range(P):
                    first[u][y] = alg.add(
                        first[u][y], alg.mul(alg.power(u * x), state[x][y])
                    )
        output = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for v in range(P):
                for y in range(P):
                    output[u][v] = alg.add(
                        output[u][v],
                        alg.mul(alg.power(-D * v * y), first[u][y]),
                    )
                output[u][v] = alg.mul(inverse17, output[u][v])
        state = output
    representatives = tuple(group[0] for group in shells())
    radial = all(
        state[x][y] == state[representatives[qnorm((x, y))][0]][representatives[qnorm((x, y))][1]]
        for x, y in POINTS
    )
    first, second = observation(depth, family)
    boundary = alg.zero
    for x, y in POINTS:
        residue = qnorm((x, y))
        boundary = alg.add(
            boundary,
            alg.mul(alg.power(first * residue * residue + second * residue), state[x][y]),
        )
    shell_values = [state[x][y] for x, y in representatives]
    return boundary, radial, shell_values


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(production_path: Path, production_source: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    if production["schema"] != "CAT_CAS_F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_CLOSURE_V1":
        fail("oracle received a different production schema")
    if [len(group) for group in shells()] != [1, *([18] * 16)]:
        fail("independent anisotropic shell law failed")

    exact_alg = backend.Algebra("Q_ZETA17")
    exact_transform = matrix(exact_alg)
    exact_reexecution = [
        execute(item["depth"], item["family"], exact_alg, exact_transform)
        for item in production["exact_transactions"]
    ]
    exact_boundary_equal = all(
        oracle["boundary"] == observed["final_boundary"]
        for oracle, observed in zip(exact_reexecution, production["exact_transactions"], strict=True)
    )
    exact_resident_payload_equal = all(
        oracle["maximum_resident_payload_bits"] == observed["maximum_resident_payload_bits"]
        for oracle, observed in zip(exact_reexecution, production["exact_transactions"], strict=True)
    )
    exact_commitment_equal = all(
        oracle["final_state_commitment"] == observed["final_state_commitment"]
        for oracle, observed in zip(exact_reexecution, production["exact_transactions"], strict=True)
    )

    field_results = []
    structural_equal = True
    dense_controls = []
    for modulus, root in FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        transform = matrix(alg)
        transform_checks = verify_transform(alg, transform)
        selected = [
            item for item in production["structural_transactions"]
            if item["algebra_kind"] == f"F{modulus}"
        ]
        transactions = [
            execute(item["depth"], item["family"], alg, transform) for item in selected
        ]
        equal = all(
            oracle["boundary"] == observed["final_boundary"]
            and oracle["final_state_commitment"] == observed["final_state_commitment"]
            and oracle["restored_seed"]
            for oracle, observed in zip(transactions, selected, strict=True)
        )
        structural_equal &= equal
        for depth, family in ((2, "ALTERNATE"), (4, "PRIMARY")):
            compact = execute(depth, family, alg, transform)
            dense_boundary, radial, dense_shell_values = dense(depth, family, alg)
            dense_controls.append(
                {
                    "field": f"F{modulus}",
                    "depth": depth,
                    "family": family,
                    "boundary_equal": alg.serialize(dense_boundary) == compact["boundary"],
                    "dense_state_remains_radial": radial,
                    "all_17_shell_values_equal": dense_shell_values == compact["final_state"],
                }
            )
        field_results.append(
            {
                "field": f"F{modulus}",
                "transform_checks": transform_checks,
                "all_structural_boundaries_equal": equal,
                "transaction_count": len(transactions),
            }
        )

    isotropic = shells(coefficient=1)
    nonzero_zero = next(point for point in isotropic[0] if point != (0, 0))
    mutation_alg = backend.Algebra("F137", modulus=137, root=16)
    origin = [len(group) % 137 for group in isotropic]
    mutated = []
    for group in isotropic:
        value = mutation_alg.zero
        for x, y in group:
            u, v = nonzero_zero
            value = mutation_alg.add(value, mutation_alg.power(u * x - v * y))
        mutated.append(value)
    mutation_controls = {
        "isotropic_zero_orbit_overmerge_fails": origin != mutated,
        "quartic_phase_rejects_sign_overmerge": any(
            exponent(gate(0, "PRIMARY"), value)
            != exponent(gate(0, "PRIMARY"), (-value) % P)
            for value in range(1, P)
        ),
    }

    passed = (
        exact_boundary_equal
        and exact_resident_payload_equal
        and exact_commitment_equal
        and all(item["restored_seed"] for item in exact_reexecution)
        and structural_equal
        and all(all(item["transform_checks"].values()) for item in field_results)
        and all(
            item["boundary_equal"]
            and item["dense_state_remains_radial"]
            and item["all_17_shell_values_equal"]
            for item in dense_controls
        )
        and all(mutation_controls.values())
    )
    if not passed:
        fail("independent anisotropic radial oracle found a mismatch")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_source_sha256": source_sha(production_source),
        "production_result_sha256": source_sha(production_path),
        "independence": {
            "imports_production_module": False,
            "reconstructs_public_geometry": True,
            "reconstructs_program_schedule": True,
            "reconstructs_exact_forward_inverse_recurrence": True,
            "uses_separable_289_coordinate_dft_for_selected_dense_controls": True,
            "shared_established_generic_exact_arithmetic_backend_only": True,
        },
        "exact_reexecution": {
            "transaction_count": len(exact_reexecution),
            "all_boundaries_equal": exact_boundary_equal,
            "all_resident_payload_maxima_equal": exact_resident_payload_equal,
            "all_final_state_commitments_equal": exact_commitment_equal,
            "all_restore_seed_exactly": all(item["restored_seed"] for item in exact_reexecution),
            "depths": [item["depth"] for item in exact_reexecution],
            "resident_payload_tuple": [
                item["maximum_resident_payload_bits"] for item in exact_reexecution
            ],
            "update_scratch_payload_tuple_independently_reproduced": False,
        },
        "structural_reexecution": field_results,
        "dense_coordinate_controls": dense_controls,
        "mutations": mutation_controls,
        "resource_law": {
            "resident_relation_quotient_exact_field_cells": 17,
            "represented_dense_coordinate_field_cells": 289,
            "compiled_public_quotient_fourier_field_cells": 289,
            "public_kernel_source_coordinate_visits": 4913,
            "exhaustive_invariance_verification_target_coordinate_visits": 289,
            "exhaustive_invariance_verification_source_coordinate_visits": 83521,
            "exhaustive_invariance_verification_total_coordinate_visits": 83810,
            "accepted_assignment_or_truth_table_cells": 0,
            "executed_matched_classical_recurrence_exact_field_cells": 17,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "STRICT_ANISOTROPIC_F17_SQUARED_RADIAL_FUNCTIONS_WITH_"
            "QUARTIC_NORM_PHASES_AND_THE_DECLARED_NORMALIZED_PHASE_FOURIER"
        ),
        "not_established": [
            "CATVM_CUSTODY",
            "GENERAL_NONLINEAR_RELATION_QUOTIENT",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION",
            "UNBOUNDED_COMPUTATION",
        ],
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.production, args.production_source)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
