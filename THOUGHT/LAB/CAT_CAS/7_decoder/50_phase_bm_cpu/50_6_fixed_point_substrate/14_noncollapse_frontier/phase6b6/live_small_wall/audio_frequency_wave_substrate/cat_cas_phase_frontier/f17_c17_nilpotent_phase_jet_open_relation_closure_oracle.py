#!/usr/bin/env python3
"""Independent full-group-algebra oracle for the F17 C17 phase-jet result.

This file imports neither the production package nor a numerical array
library.  It executes the complete 17-coefficient cyclic group algebra and
only then maps the final state to Hasse-jet coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 17
ORDER = 17
RANKS = (2, 4, 8)
DEPTHS = (1, 4, 16, 64)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F17_C17_REPEATED_ROOT_NILPOTENT_PHASE_JET_QUOTIENT_"
    "CLOSES_TRANSLATION_INVARIANT_OPEN_RELATION_COMPOSITION_AND_NONLINEAR_"
    "CONVOLUTION_SHEAR_IN_RANKS2_4_8_THROUGH_DEPTH64_WITH_FINAL_ONLY_"
    "BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_RANK_R_"
    "CLASSICAL_JET_RECURRENCE_REMAINS_AND_HIGHER_HASSE_MOMENTS_ARE_"
    "EXPLICITLY_OUTSIDE_THE_QUOTIENT"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for item in range(1, k + 1):
        numerator = numerator * (n - item + 1) % FIELD
        denominator = denominator * item % FIELD
    return numerator * pow(denominator, FIELD - 2, FIELD) % FIELD


def group_add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def group_subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def group_scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * x) % FIELD for x in value]


def cyclic_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[(i + j) % ORDER] = (result[(i + j) % ORDER] + x * y) % FIELD
    return result


def rotate(value: list[int], shift: int) -> list[int]:
    result = [0] * ORDER
    for index, coefficient in enumerate(value):
        result[(index + shift) % ORDER] = coefficient
    return result


def to_jet(coefficients: list[int], rank: int) -> list[int]:
    return [
        sum(coefficients[position] * binomial(position, degree) for position in range(ORDER))
        % FIELD
        for degree in range(rank)
    ]


def jet17_to_group(jet: list[int]) -> list[int]:
    if len(jet) != ORDER:
        raise RuntimeError("full jet must have rank 17")
    coefficients = [0] * ORDER
    for degree, value in enumerate(jet):
        for power_index in range(degree + 1):
            sign = -1 if (degree - power_index) % 2 else 1
            coefficients[power_index] = (
                coefficients[power_index]
                + value * binomial(degree, power_index) * sign
            ) % FIELD
    return coefficients


def jet_multiply(left: list[int], right: list[int], rank: int) -> list[int]:
    result = [0] * rank
    for degree in range(rank):
        result[degree] = sum(left[index] * right[degree - index] for index in range(degree + 1)) % FIELD
    return result


def jet_power(value: list[int], exponent: int, rank: int) -> list[int]:
    result = [1] + [0] * (rank - 1)
    base = value.copy()
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = jet_multiply(result, base, rank)
        base = jet_multiply(base, base, rank)
        remaining >>= 1
    return result


def jet_inverse(value: list[int], rank: int) -> list[int]:
    if value[0] == 0:
        raise RuntimeError("nonunit")
    inverse = [0] * rank
    inverse[0] = pow(value[0], FIELD - 2, FIELD)
    for degree in range(1, rank):
        subtotal = sum(value[index] * inverse[degree - index] for index in range(1, degree + 1))
        inverse[degree] = (-inverse[0] * subtotal) % FIELD
    return inverse


def phase_jet(rank: int, shift: int) -> list[int]:
    t = [1, 1] + [0] * (rank - 2)
    return jet_power(t, shift % ORDER, rank)


def kernel_jet(rank: int, index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    shift = (3 * index + 5 * code + 1) % ORDER
    slope = (7 * index + 4 * code + 3) % FIELD
    return jet_multiply(phase_jet(rank, shift), [1, slope] + [0] * (rank - 2), rank)


def full_kernel(index: int, family: str) -> list[int]:
    return jet17_to_group(kernel_jet(ORDER, index, family))


def full_kernel_inverse(index: int, family: str) -> list[int]:
    return jet17_to_group(jet_inverse(kernel_jet(ORDER, index, family), ORDER))


def parameters(index: int, family: str) -> tuple[int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    alpha = (5 * index + 3 * code + 1) % FIELD or 1
    beta = (7 * index + 2 * code + 4) % FIELD or 1
    b_shift = (11 * index + 5 * code + 2) % ORDER
    return alpha, beta, b_shift


def seed_coefficient(position: int, family: str, register: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    if register == "A":
        atoms = (
            (code, 2 + code),
            ((3 + 2 * code) % ORDER, 5),
            ((8 + code) % ORDER, 9),
        )
    else:
        atoms = (
            ((2 + code) % ORDER, 4),
            ((7 + 2 * code) % ORDER, 6 + code),
            ((13 - code) % ORDER, 11),
        )
    return sum(weight for location, weight in atoms if location == position) % FIELD


def seed(family: str, register: str) -> list[int]:
    return [seed_coefficient(position, family, register) for position in range(ORDER)]


def weights(rank: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [(3 * degree + 5 * code + rank) % FIELD or 1 for degree in range(rank)]


def descriptor(rank: int, depth: int, family: str) -> dict[str, Any]:
    return {
        "schema": "CAT_CAS_F17_C17_NILPOTENT_PHASE_JET_PROGRAM_V1",
        "field": FIELD,
        "group": "C17",
        "rank": rank,
        "depth": depth,
        "family": family,
        "port_type": f"F17_C17_TRANSLATION_INVARIANT_OPEN_RELATION_JET_R{rank}",
        "state_law": "F17_EPSILON_MOD_EPSILON_TO_RANK",
        "composition": "TRUNCATED_JET_MULTIPLICATION",
        "nonlinear_module": "REVERSIBLE_B_PLUS_BETA_A_CONVOLUTION_SQUARE",
        "projection": "FINAL_ONLY_PUBLIC_LINEAR_B_JET_BOUNDARY",
        "boundary_weights": weights(rank, family),
        "topology_compilation_reads_final_answer": False,
    }


def forward(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a = a.copy()
    current_b = b.copy()
    for index in range(depth):
        alpha, beta, b_shift = parameters(index, family)
        if family == "PRIMARY":
            current_a = group_add(current_a, group_scale(current_b, alpha))
            current_b = group_add(
                current_b,
                group_scale(cyclic_convolution(current_a, current_a), beta),
            )
            current_a = cyclic_convolution(current_a, full_kernel(index, family))
        else:
            current_b = group_add(
                current_b,
                group_scale(cyclic_convolution(current_a, current_a), beta),
            )
            current_a = cyclic_convolution(current_a, full_kernel(index, family))
            current_a = group_add(current_a, group_scale(current_b, alpha))
        current_b = rotate(current_b, b_shift)
    return current_a, current_b


def inverse(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a = a.copy()
    current_b = b.copy()
    for index in reversed(range(depth)):
        alpha, beta, b_shift = parameters(index, family)
        current_b = rotate(current_b, -b_shift)
        if family == "PRIMARY":
            current_a = cyclic_convolution(current_a, full_kernel_inverse(index, family))
            current_b = group_subtract(
                current_b,
                group_scale(cyclic_convolution(current_a, current_a), beta),
            )
            current_a = group_subtract(current_a, group_scale(current_b, alpha))
        else:
            current_a = group_subtract(current_a, group_scale(current_b, alpha))
            current_a = cyclic_convolution(current_a, full_kernel_inverse(index, family))
            current_b = group_subtract(
                current_b,
                group_scale(cyclic_convolution(current_a, current_a), beta),
            )
    return current_a, current_b


def case_key(rank: int, depth: int, family: str) -> tuple[int, int, str]:
    return rank, depth, family


def quotient_law_checks() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    comparisons = 0
    for rank in RANKS:
        for sample in range(6):
            left = [((position + 2) * (sample + 3) + position * position) % FIELD for position in range(ORDER)]
            right = [((3 * position + 1) * (sample + 5) + position) % FIELD for position in range(ORDER)]
            convolution_ok = to_jet(cyclic_convolution(left, right), rank) == jet_multiply(
                to_jet(left, rank), to_jet(right, rank), rank
            )
            rotation_ok = to_jet(rotate(left, sample + 1), rank) == jet_multiply(
                to_jet(left, rank), phase_jet(rank, sample + 1), rank
            )
            addition_ok = to_jet(group_add(left, right), rank) == [
                (x + y) % FIELD
                for x, y in zip(to_jet(left, rank), to_jet(right, rank), strict=True)
            ]
            checks[f"rank{rank}_sample{sample}_convolution"] = convolution_ok
            checks[f"rank{rank}_sample{sample}_rotation"] = rotation_ok
            checks[f"rank{rank}_sample{sample}_addition"] = addition_ok
            comparisons += 3

    quotient_scope: dict[str, bool] = {}
    for rank in RANKS:
        epsilon_rank_jet = [0] * ORDER
        epsilon_rank_jet[rank] = 1
        discarded = jet17_to_group(epsilon_rank_jet)
        zero = [0] * ORDER
        delta_zero = [1] + [0] * (ORDER - 1)
        quotient_scope[f"rank{rank}_epsilon_rank_is_discarded"] = to_jet(discarded, rank) == to_jet(zero, rank)
        hadamard = [(x * y) % FIELD for x, y in zip(discarded, delta_zero, strict=True)]
        quotient_scope[f"rank{rank}_coefficient_hadamard_does_not_descend"] = to_jet(hadamard, rank) != [0] * rank
    return {
        "checks": checks,
        "comparison_count": comparisons,
        "all_pass": all(checks.values()),
        "scope_controls": quotient_scope,
        "all_scope_controls_pass": all(quotient_scope.values()),
    }


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    production_cases = {
        case_key(item["rank"], item["depth"], item["family"]): item
        for item in production["cases"]
    }
    cases = []
    comparisons = 0
    for rank in RANKS:
        for family in FAMILIES:
            for depth in DEPTHS:
                initial_a = seed(family, "A")
                initial_b = seed(family, "B")
                final_a, final_b = forward(initial_a, initial_b, depth, family)
                restored_a, restored_b = inverse(final_a, final_b, depth, family)
                final_jet = to_jet(final_a, rank) + to_jet(final_b, rank)
                boundary = sum(
                    x * y
                    for x, y in zip(weights(rank, family), final_jet[rank:], strict=True)
                ) % FIELD
                fingerprint = digest_json(descriptor(rank, depth, family))
                observed = production_cases[case_key(rank, depth, family)]
                field_matches = {
                    "program_fingerprint": observed["program_fingerprint"] == fingerprint,
                    "boundary": observed["boundary"] == boundary,
                    "final_jet_commitment": observed["final_jet_commitment"] == digest_json(final_jet),
                    "exact_restoration": observed["exact_cells_restored"] and restored_a == initial_a and restored_b == initial_b,
                }
                comparisons += len(field_matches)
                cases.append(
                    {
                        "rank": rank,
                        "depth": depth,
                        "family": family,
                        "program_fingerprint": fingerprint,
                        "boundary": boundary,
                        "final_jet_commitment": digest_json(final_jet),
                        "full17_coefficient_restoration": restored_a == initial_a and restored_b == initial_b,
                        "field_matches": field_matches,
                        "all_fields_match": all(field_matches.values()),
                    }
                )
    quotient_checks = quotient_law_checks()
    all_match = all(item["all_fields_match"] for item in cases)
    if not all_match or not quotient_checks["all_pass"] or not quotient_checks["all_scope_controls_pass"]:
        raise RuntimeError("independent oracle mismatch")
    return {
        "schema": "CAT_CAS_F17_C17_NILPOTENT_PHASE_JET_ORACLE_RESULTS_V1",
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production": False,
        "imports_numpy": False,
        "oracle_state_law": "FULL17_COEFFICIENT_F17_C17_GROUP_ALGEBRA",
        "cases": cases,
        "production_comparison": {
            "cases": len(cases),
            "comparisons": comparisons,
            "all_match": all_match,
        },
        "quotient_law": quotient_checks,
        "resource_scope": {
            "oracle_full_coefficient_carrier_field_cells": 34,
            "accepted_maximum_jet_carrier_field_cells": 16,
            "oracle_dense17_by17_relation_table_cells": 0,
            "oracle_state_is_verification_only": True,
            "production_backing_identity_independently_observed": False,
        },
        "claim_ceiling": "F17_C17_TRANSLATION_INVARIANT_OPEN_RELATION_JETS_RANKS2_4_8_TWO_PUBLIC_PROGRAM_FAMILIES_DEPTHS1_4_16_64_FINAL_LINEAR_B_JET_BOUNDARIES",
        "preserved_subclaims": [
            "FULL17_CYCLIC_CONVOLUTION_MAPS_EXACTLY_TO_TRUNCATED_JET_MULTIPLICATION",
            "FULL17_NONLINEAR_CONVOLUTION_SHEAR_MAPS_EXACTLY_TO_THE_JET_SHEAR",
            "FULL17_REFERENCE_AND_ACCEPTED_JET_BOUNDARIES_MATCH_IN24_CASES",
            "FULL17_AND_JET_CARRIERS_RESTORE_EXACTLY",
            "HIGHER_HASSE_COORDINATES_ARE_DEMONSTRABLY_DISCARDED",
            "COEFFICIENTWISE_HADAMARD_INTERSECTION_DOES_NOT_DESCEND_TO_THIS_QUOTIENT",
        ],
        "rejected_interpretations": [
            "GENERAL_OPEN_RELATION_INTERSECTION_CLOSURE",
            "LOSSLESS_FULL_F17_C17_RELATION_REPRESENTATION_BELOW17_COORDINATES",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production.read_text(encoding="utf-8"))
    result = build_result(production)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
