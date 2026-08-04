#!/usr/bin/env python3
"""Independent full-17 oracle for the C17 Paley relation algebra result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 103
ORDER = 17
ZETA = 72
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


RESIDUES = {value * value % ORDER for value in range(1, ORDER)}
NONRESIDUES = set(range(1, ORDER)) - RESIDUES
CLASSES = ({0}, RESIDUES, NONRESIDUES)


def phase(exponent: int) -> int:
    return pow(ZETA, exponent % ORDER, FIELD)


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * x) % FIELD for x in value]


def hadamard(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[(i + j) % ORDER] = (result[(i + j) % ORDER] + x * y) % FIELD
    return result


def expand(values: list[int]) -> list[int]:
    return [next(values[index] for index, orbit in enumerate(CLASSES) if position in orbit) for position in range(ORDER)]


def contract(full: list[int]) -> list[int]:
    result: list[int] = []
    for orbit in CLASSES:
        values = {full[position] for position in orbit}
        if len(values) != 1:
            raise RuntimeError("full relation is outside the Paley orbit algebra")
        result.append(next(iter(values)))
    return result


def parameters(index: int, family: str) -> tuple[int, int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    return (
        phase(3 * index + code),
        phase(5 * index + 2 * code + 1),
        phase(7 * index + 3 * code + 2),
        phase(11 * index + 4 * code + 3),
    )


def kernel(index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return expand([phase(index + code), phase(3 * index + 2 * code + 1), phase(5 * index + 4 * code + 2)])


def mask(index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return expand([phase(2 * index + code + 1), phase(7 * index + 3 * code + 2), phase(13 * index + 5 * code + 3)])


def seed(family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 5 if register == "A" else 19
    return expand([phase(offset + code * (coordinate + 1) + coordinate * coordinate) for coordinate in range(3)])


def boundary(b_values: list[int], family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [phase(2 * code + 1), phase(5 * code + 2), phase(9 * code + 3)]
    return sum(weight * value for weight, value in zip(weights, b_values, strict=True)) % FIELD


def forward(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a, current_b = a.copy(), b.copy()
    for index in range(depth):
        alpha, beta, gamma, delta = parameters(index, family)
        public_kernel, public_mask = kernel(index, family), mask(index, family)
        if family == "PRIMARY":
            current_a = add(current_a, scale(hadamard(current_b, current_b), alpha))
            current_b = add(current_b, scale(convolution(current_a, current_a), beta))
            current_a = add(current_a, scale(convolution(current_b, public_kernel), gamma))
            current_b = add(current_b, scale(hadamard(current_a, public_mask), delta))
        else:
            current_b = add(current_b, scale(hadamard(current_a, public_mask), delta))
            current_a = add(current_a, scale(convolution(current_b, public_kernel), gamma))
            current_b = add(current_b, scale(convolution(current_a, current_a), beta))
            current_a = add(current_a, scale(hadamard(current_b, current_b), alpha))
    return current_a, current_b


def inverse(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a, current_b = a.copy(), b.copy()
    for index in reversed(range(depth)):
        alpha, beta, gamma, delta = parameters(index, family)
        public_kernel, public_mask = kernel(index, family), mask(index, family)
        if family == "PRIMARY":
            current_b = subtract(current_b, scale(hadamard(current_a, public_mask), delta))
            current_a = subtract(current_a, scale(convolution(current_b, public_kernel), gamma))
            current_b = subtract(current_b, scale(convolution(current_a, current_a), beta))
            current_a = subtract(current_a, scale(hadamard(current_b, current_b), alpha))
        else:
            current_a = subtract(current_a, scale(hadamard(current_b, current_b), alpha))
            current_b = subtract(current_b, scale(convolution(current_a, current_a), beta))
            current_a = subtract(current_a, scale(convolution(current_b, public_kernel), gamma))
            current_b = subtract(current_b, scale(hadamard(current_a, public_mask), delta))
    return current_a, current_b


def intersection_numbers() -> list[list[list[int]]]:
    tensor: list[list[list[int]]] = []
    for left in CLASSES:
        row: list[list[int]] = []
        for right in CLASSES:
            counts = [
                sum(1 for x in left for y in right if (x + y) % ORDER == target)
                for target in range(ORDER)
            ]
            row.append(contract(counts))
        tensor.append(row)
    return tensor


def algebra_oracle() -> dict[str, Any]:
    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    composition_checks = []
    hadamard_checks = []
    for i, left in enumerate(basis):
        for j, right in enumerate(basis):
            composition_checks.append({"left": i, "right": j, "result": contract(convolution(expand(left), expand(right)))})
            hadamard_checks.append({"left": i, "right": j, "result": contract(hadamard(expand(left), expand(right)))})
    delta_one = [0] * ORDER
    delta_one[1] = 1
    outside_rejected = False
    try:
        contract(delta_one)
    except RuntimeError:
        outside_rejected = True
    return {
        "residues": sorted(RESIDUES),
        "nonresidues": sorted(NONRESIDUES),
        "class_sizes": [len(orbit) for orbit in CLASSES],
        "intersection_number_tensor": intersection_numbers(),
        "composition_basis_checks": composition_checks,
        "hadamard_basis_checks": hadamard_checks,
        "composition_checks": len(composition_checks),
        "hadamard_checks": len(hadamard_checks),
        "all_outputs_class_constant": True,
        "single_residue_delta_outside_family_rejected": outside_rejected,
        "zeta_order17": pow(ZETA, ORDER, FIELD) == 1 and all(pow(ZETA, exponent, FIELD) != 1 for exponent in range(1, ORDER)),
    }


def compare_cases(production: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    expected = {(case["depth"], case["family"]): case for case in production["cases"]}
    cases: list[dict[str, Any]] = []
    comparisons = 0
    for family in FAMILIES:
        initial_a, initial_b = seed(family, "A"), seed(family, "B")
        for depth in DEPTHS:
            final_a, final_b = forward(initial_a, initial_b, depth, family)
            a_values, b_values = contract(final_a), contract(final_b)
            restored_a, restored_b = inverse(final_a, final_b, depth, family)
            package = expected[(depth, family)]
            observed = {
                "depth": depth,
                "family": family,
                "forward_a_matches": a_values == package["forward_a"],
                "forward_b_matches": b_values == package["forward_b"],
                "boundary_matches": boundary(b_values, family) == package["boundary"],
                "commitment_matches": digest_json([a_values, b_values]) == package["forward_commitment"],
                "full34_cells_restore_exactly": restored_a == initial_a and restored_b == initial_b,
            }
            comparisons += 5
            observed["all_fields_match"] = all(observed[key] for key in ("forward_a_matches", "forward_b_matches", "boundary_matches", "commitment_matches", "full34_cells_restore_exactly"))
            cases.append(observed)
    return cases, comparisons


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    algebra = algebra_oracle()
    cases, comparisons = compare_cases(production)
    expected_tensor = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [8, 3, 4], [0, 4, 4]],
        [[0, 0, 1], [0, 4, 4], [8, 4, 3]],
    ]
    if algebra["intersection_number_tensor"] != expected_tensor:
        raise RuntimeError("independent Paley intersection-number mismatch")
    if not algebra["single_residue_delta_outside_family_rejected"] or not algebra["zeta_order17"]:
        raise RuntimeError("scope or phase-root control failed")
    if not all(case["all_fields_match"] for case in cases):
        raise RuntimeError("full relation case mismatch")
    return {
        "schema": "CAT_CAS_F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_ORACLE_RESULTS_V1",
        "claim": production["claim"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production": False,
        "imports_numpy": False,
        "oracle_state_law": "FULL17_F103_TRANSLATION_INVARIANT_RELATION_VALUES",
        "algebra_oracle": algebra,
        "production_comparison": {
            "cases": len(cases),
            "field_comparisons": comparisons,
            "all_match": all(case["all_fields_match"] for case in cases),
        },
        "cases": cases,
        "resource_scope": {
            "accepted_relation_coordinates_per_port": 3,
            "accepted_two_register_carrier_field_cells": 6,
            "oracle_full_relation_coordinates_per_port": 17,
            "oracle_full_two_register_carrier_field_cells": 34,
            "dense17_by17_relation_table_cells": 0,
            "full_relation_vectors_are_oracle_only": True,
            "whole_process_memory_not_measured": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "THREE_CLASS_PALEY_RELATIONS_CLOSE_UNDER_FULL17_CYCLIC_CONVOLUTION",
            "THREE_CLASS_PALEY_RELATIONS_CLOSE_UNDER_FULL17_HADAMARD_INTERSECTION",
            "FULL17_AND_THREE_COORDINATE_BOUNDARIES_MATCH_IN12_CASES",
            "FULL34_CELL_AND_ACCEPTED_CARRIERS_RESTORE_EXACTLY",
        ],
        "rejected_interpretations": [
            "GENERAL_C17_RELATION_CLOSURE_IN_THREE_COORDINATES",
            "GENERAL_ASSOCIATION_SCHEME_COMPILER",
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
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_result.read_text(encoding="utf-8"))
    payload = json.dumps(build_result(production), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
