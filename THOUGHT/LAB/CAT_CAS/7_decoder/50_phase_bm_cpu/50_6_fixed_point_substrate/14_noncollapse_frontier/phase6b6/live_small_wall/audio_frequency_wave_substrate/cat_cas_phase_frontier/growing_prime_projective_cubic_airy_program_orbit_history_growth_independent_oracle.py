#!/usr/bin/env python3
"""Independent exact oracle for the bounded projective program-orbit result.

This file imports neither the production implementation nor any predecessor
module.  It independently rebuilds the safe-prime fields, phase fixtures,
Gaussian kernels, dense two-fiber carrier, public layers, inverse traversal,
projective canonicalization, collision subspace, controls, and restored reuse.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CASES = ((5, 1), (5, 2), (5, 3), (5, 4), (11, 1), (11, 2))


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return value == divisor
        divisor += 1
    return True


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    root: int


def make_field(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("not a safe-prime pair")
    primitive = 2
    while primitive < p and (
        pow(primitive, 2, p) == 1 or pow(primitive, q, p) == 1
    ):
        primitive += 1
    if primitive == p:
        fail("primitive root unavailable")
    root = pow(primitive, 2, p)
    if pow(root, q, p) != 1 or len(
        {pow(root, exponent, p) for exponent in range(q)}
    ) != q:
        fail("invalid phase root")
    return Field(q, p, root)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


def kernel(
    payload: tuple[int, ...], target: int, source: int, field: Field
) -> int:
    a, b, c, d, coefficient = payload
    if b % field.q:
        inverse = pow(2 * b % field.q, -1, field.q)
        exponent = (
            d * target * target - 2 * target * source + a * source * source
        ) * inverse
        return coefficient * phase(field, exponent) % field.p
    if source % field.q != d * target % field.q:
        return 0
    return coefficient * phase(
        field,
        c * d * target * target * pow(2, -1, field.q),
    ) % field.p


def matrix(payload: tuple[int, ...], field: Field) -> list[list[int]]:
    return [
        [kernel(payload, target, source, field) for source in range(field.q)]
        for target in range(field.q)
    ]


def matrix_inverse(values: list[list[int]], modulus: int) -> list[list[int]]:
    size = len(values)
    augmented = [
        [value % modulus for value in row]
        + [int(row_index == column) for column in range(size)]
        for row_index, row in enumerate(values)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            fail("singular Gaussian matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = pow(augmented[column][column], -1, modulus)
        augmented[column] = [value * inverse % modulus for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                (left - factor * right) % modulus
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def fixture(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    latent = [
        phase(field, (code + 1) * strength**2 + (2 * code + 1) * strength + code)
        for strength in range(field.q)
    ]
    data = [
        (fiber + 1)
        * phase(
            field,
            (fiber + code + 1) * coordinate**2
            + (3 * fiber + 2 * code + 1) * coordinate,
        )
        % field.p
        for fiber in range(2)
        for coordinate in range(field.q)
    ]
    return [
        latent[strength] * data[fiber * field.q + coordinate] % field.p
        for strength in range(field.q)
        for fiber in range(2)
        for coordinate in range(field.q)
    ]


def offset(q: int, strength: int, fiber: int, coordinate: int) -> int:
    return (2 * strength + fiber) * q + coordinate


def layer_payloads(
    q: int, layer: int, family: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    code = 1 if family == "PRIMARY" else 2
    parameter = (layer + code) % q or 1
    return (
        (1, 1, parameter, parameter + 1, 1),
        (parameter + 1, 1, parameter, 1, 1),
    )


def layer_matrices(
    field: Field, layer: int, family: str
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    data_payload, latent_payload = layer_payloads(field.q, layer, family)
    data = matrix(data_payload, field)
    latent = matrix(latent_payload, field)
    return data, matrix_inverse(data, field.p), latent, matrix_inverse(latent, field.p)


def controlled(state: list[int], field: Field, strength_value: int) -> None:
    for strength in range(field.q):
        for coordinate in range(field.q):
            multiplier = phase(
                field, strength_value * strength * coordinate * coordinate * coordinate
            )
            for fiber in range(2):
                index = offset(field.q, strength, fiber, coordinate)
                state[index] = state[index] * multiplier % field.p


def apply_axis(
    state: list[int], field: Field, values: list[list[int]], axis: str
) -> None:
    q, p = field.q, field.p
    if axis == "DATA":
        for strength in range(q):
            for fiber in range(2):
                start = offset(q, strength, fiber, 0)
                source = state[start : start + q]
                state[start : start + q] = [
                    sum(left * right for left, right in zip(row, source)) % p
                    for row in values
                ]
        return
    if axis == "LATENT":
        for fiber in range(2):
            for coordinate in range(q):
                source = [
                    state[offset(q, strength, fiber, coordinate)]
                    for strength in range(q)
                ]
                output = [
                    sum(left * right for left, right in zip(row, source)) % p
                    for row in values
                ]
                for strength, value in enumerate(output):
                    state[offset(q, strength, fiber, coordinate)] = value
        return
    fail("invalid axis")


def apply_layer(
    state: list[int],
    field: Field,
    chart: tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]],
    first: int,
    second: int,
    family: str,
) -> None:
    data, _, latent, _ = chart
    controlled(state, field, first)
    if family == "PRIMARY":
        apply_axis(state, field, data, "DATA")
        controlled(state, field, second)
        apply_axis(state, field, latent, "LATENT")
    else:
        apply_axis(state, field, latent, "LATENT")
        controlled(state, field, second)
        apply_axis(state, field, data, "DATA")


def inverse_layer(
    state: list[int],
    field: Field,
    chart: tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]],
    first: int,
    second: int,
    family: str,
    mutation: str | None = None,
) -> None:
    _, data_inverse, _, latent_inverse = chart
    inverse_first = (-first + int(mutation == "WRONG")) % field.q
    inverse_second = -second % field.q
    if family == "PRIMARY":
        if mutation == "REORDER":
            controlled(state, field, inverse_first)
        apply_axis(state, field, latent_inverse, "LATENT")
        controlled(state, field, inverse_second)
        apply_axis(state, field, data_inverse, "DATA")
        if mutation not in ("MISSING", "REORDER"):
            controlled(state, field, inverse_first)
    else:
        if mutation == "REORDER":
            controlled(state, field, inverse_first)
        apply_axis(state, field, data_inverse, "DATA")
        controlled(state, field, inverse_second)
        apply_axis(state, field, latent_inverse, "LATENT")
        if mutation not in ("MISSING", "REORDER"):
            controlled(state, field, inverse_first)


def exact_code(values: list[int], base: int) -> int:
    result = 0
    for value in reversed(values):
        result = result * base + value
    return result


def projective_code(values: list[int], field: Field) -> tuple[int, int]:
    pivot = next((value for value in values if value), None)
    if pivot is None:
        fail("zero projective state")
    inverse = pow(pivot, -1, field.p)
    return exact_code([(value * inverse) % field.p for value in values], field.p), pivot


def decode_history(encoded: int, q: int, depth: int) -> list[tuple[int, int]]:
    pairs = []
    for _ in range(depth):
        second = encoded % q
        encoded //= q
        first = encoded % q
        encoded //= q
        pairs.append((first, second))
    return list(reversed(pairs))


def row_reduce(rows: list[list[int]], modulus: int) -> tuple[list[list[int]], list[int]]:
    reduced = [[value % modulus for value in row] for row in rows if any(row)]
    pivots: list[int] = []
    pivot_row = 0
    if not reduced:
        return [], pivots
    for column in range(len(reduced[0])):
        pivot = next(
            (row for row in range(pivot_row, len(reduced)) if reduced[row][column]),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        inverse = pow(reduced[pivot_row][column], -1, modulus)
        reduced[pivot_row] = [value * inverse % modulus for value in reduced[pivot_row]]
        for row, values in enumerate(reduced):
            if row == pivot_row or not values[column]:
                continue
            factor = values[column]
            reduced[row] = [
                (left - factor * right) % modulus
                for left, right in zip(values, reduced[pivot_row])
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return reduced[:pivot_row], pivots


def nullspace_basis(rows: list[list[int]], modulus: int, columns: int) -> list[list[int]]:
    reduced, pivots = row_reduce(rows, modulus)
    basis = []
    for free in (column for column in range(columns) if column not in pivots):
        vector = [0] * columns
        vector[free] = 1
        for row, pivot in zip(reduced, pivots):
            vector[pivot] = -row[free] % modulus
        basis.append(vector)
    return basis


def normalize_line(vector: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    pivot = next(value for value in vector if value)
    inverse = pow(pivot, -1, modulus)
    return tuple(value * inverse % modulus for value in vector)


def orbit_case(q: int, depth: int) -> dict[str, Any]:
    field = make_field(q)
    state = fixture(field, "PRIMARY")
    initial = tuple(state)
    charts = [layer_matrices(field, layer, "PRIMARY") for layer in range(depth)]
    first_by_code: dict[int, int] = {}
    collision_counts: dict[int, int] = {}
    colliding_histories: set[int] = set()
    difference_lines: set[tuple[int, ...]] = set()
    raw_equalities = 0
    restoration_comparisons = 0

    def visit(layer: int, history: int) -> None:
        nonlocal raw_equalities, restoration_comparisons
        if layer == depth:
            code, pivot = projective_code(state, field)
            if code in first_by_code:
                first_history, first_pivot = divmod(first_by_code[code], field.p)
                collision_counts[code] = collision_counts.get(code, 0) + 1
                left = decode_history(first_history, q, depth)
                right = decode_history(history, q, depth)
                scalar = pivot * pow(first_pivot, -1, field.p) % field.p
                raw_equalities += int(scalar == 1)
                colliding_histories.update((first_history, history))
                difference = tuple(
                    (right_value - left_value) % q
                    for left_pair, right_pair in zip(left, right)
                    for left_value, right_value in zip(left_pair, right_pair)
                )
                difference_lines.add(normalize_line(difference, q))
            else:
                first_by_code[code] = history * field.p + pivot
            return
        for first in range(q):
            for second in range(q):
                before = tuple(state)
                apply_layer(state, field, charts[layer], first, second, "PRIMARY")
                visit(layer + 1, (history * q + first) * q + second)
                inverse_layer(state, field, charts[layer], first, second, "PRIMARY")
                restoration_comparisons += 1
                if tuple(state) != before:
                    fail("independent branch restoration failed")

    visit(0, 0)
    if tuple(state) != initial:
        fail("independent root restoration failed")
    expected = q ** (2 * depth)
    distinct = len(first_by_code)
    vectors = [
        [value for pair in decode_history(encoded, q, depth) for value in pair]
        for encoded in colliding_histories
    ]
    basis, _ = row_reduce(vectors, q)
    histogram: dict[str, int] = {}
    for count in collision_counts.values():
        key = str(count + 1)
        histogram[key] = histogram.get(key, 0) + 1
    singletons = distinct - len(collision_counts)
    if singletons:
        histogram["1"] = singletons
    return {
        "q": q,
        "p": field.p,
        "depth": depth,
        "public_program_histories": expected,
        "distinct_projective_full_states": distinct,
        "projective_collisions": expected - distinct,
        "non_singleton_projective_classes": len(collision_counts),
        "projective_collision_class_size_histogram": histogram,
        "histories_in_non_singleton_projective_classes": sum(
            count + 1 for count in collision_counts.values()
        ),
        "minimum_exact_projective_state_identifier_bits": (distinct - 1).bit_length(),
        "minimum_exact_public_history_identifier_bits": (expected - 1).bit_length(),
        "collision_relation_diagnostic": {
            "collision_events_total": expected - distinct,
            "all_collisions_are_raw_state_equal_not_only_projectively_equal": (
                raw_equalities == expected - distinct
            ),
            "colliding_history_vectors": len(colliding_histories),
            "colliding_history_span_rank": len(basis),
            "colliding_histories_form_complete_linear_subspace": bool(
                colliding_histories
            )
            and len(colliding_histories) == q ** len(basis),
            "colliding_history_subspace_rref_basis": basis,
            "colliding_history_subspace_constraint_basis": nullspace_basis(
                basis, q, 2 * depth
            ),
            "normalized_collision_difference_line_generators": [
                list(vector) for vector in sorted(difference_lines)
            ],
            "single_collision_difference_line": len(difference_lines) == 1,
        },
        "exact_root_restored": tuple(state) == initial,
        "exact_branch_restoration_comparisons": restoration_comparisons,
    }


def state_for_history(q: int, history: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    field = make_field(q)
    state = fixture(field, "PRIMARY")
    for layer, (first, second) in enumerate(history):
        apply_layer(
            state,
            field,
            layer_matrices(field, layer, "PRIMARY"),
            first % q,
            second % q,
            "PRIMARY",
        )
    return tuple(state)


def relation_comparisons(q: int) -> list[bool]:
    output = []
    for first0, second0, first1, total, shift in (
        (0, 0, 0, 1, 1),
        (1, 2, 3, 4, 2),
        (2, 4, 1, 0, 3),
    ):
        left = ((first0, second0), (first1, total), (0, 0), (0, 0))
        right = (
            (first0, second0),
            (first1, total - shift),
            (0, 0),
            (0, shift),
        )
        output.append(state_for_history(q, left) == state_for_history(q, right))
    return output


def inverse_attack(mutation: str) -> bool:
    field = make_field(5)
    state = fixture(field, "PRIMARY")
    initial = tuple(state)
    chart = layer_matrices(field, 0, "PRIMARY")
    apply_layer(state, field, chart, 1, 2, "PRIMARY")
    inverse_layer(state, field, chart, 1, 2, "PRIMARY", mutation)
    return tuple(state) != initial


def boundary(state: list[int], field: Field, family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    probes = (
        ((3 * code + 1) % field.q, 0, (5 * code + 2) % field.q, 1),
        ((7 * code + 2) % field.q, 1, (11 * code + 3) % field.q, 2),
        ((13 * code + 1) % field.q, 0, (17 * code + 4) % field.q, 3),
        ((19 * code + 2) % field.q, 1, (23 * code + 5) % field.q, 5),
    )
    return sum(
        weight * state[offset(field.q, strength, fiber, coordinate)]
        for strength, fiber, coordinate, weight in probes
    ) % field.p


def reuse_check() -> dict[str, Any]:
    first = orbit_case(5, 2)
    field = make_field(5)
    restored = fixture(field, "PRIMARY")
    initial = tuple(restored)
    backing = id(restored)
    chart = layer_matrices(field, 0, "ALTERNATE")
    apply_layer(restored, field, chart, 2, 3, "ALTERNATE")
    restored_boundary = boundary(restored, field, "PRIMARY")
    inverse_layer(restored, field, chart, 2, 3, "ALTERNATE")
    fresh = fixture(field, "PRIMARY")
    apply_layer(fresh, field, chart, 2, 3, "ALTERNATE")
    return {
        "first_projective_orbit_size": first["distinct_projective_full_states"],
        "second_matches_fresh": restored_boundary == boundary(fresh, field, "PRIMARY"),
        "exact_payload_restored_after_reuse": tuple(restored) == initial,
        "same_backing_reused": id(restored) == backing,
        "restoration_generation": 2,
        "snapshot_used": False,
    }


def inspect_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "imports": imports,
        "production_source_imports_oracle": any(
            "independent_oracle" in name for name in imports
        ),
        "exact_projective_traversal_present": "traverse_projective_orbit" in source,
        "cryptographic_digest_used_as_state_equality": "hashlib" in imports,
    }


def parity_fields(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "q",
        "p",
        "depth",
        "public_program_histories",
        "distinct_projective_full_states",
        "projective_collisions",
        "non_singleton_projective_classes",
        "projective_collision_class_size_histogram",
        "histories_in_non_singleton_projective_classes",
        "minimum_exact_projective_state_identifier_bits",
        "minimum_exact_public_history_identifier_bits",
        "collision_relation_diagnostic",
    )
    return {key: case[key] for key in keys}


def run(production_source: Path, production_result: Path) -> dict[str, Any]:
    production = json.loads(production_result.read_text(encoding="utf-8"))
    production_cases = {
        (case["q"], case["depth"]): case for case in production["cases"]
    }
    certificates = [orbit_case(q, depth) for q, depth in CASES]
    for certificate in certificates:
        key = certificate["q"], certificate["depth"]
        rebuilt = parity_fields(certificate)
        recorded = parity_fields(production_cases[key])
        if rebuilt != recorded:
            differences = {
                field: {"independent": rebuilt[field], "production": recorded[field]}
                for field in rebuilt
                if rebuilt[field] != recorded[field]
            }
            fail(
                f"independent case differs from production at {key}: "
                + json.dumps(differences, sort_keys=True)
            )
    q5, q11, q23 = (
        relation_comparisons(5),
        relation_comparisons(11),
        relation_comparisons(23),
    )
    controls = {
        "all_six_case_summaries_match_production": True,
        "missing_inverse_fails": inverse_attack("MISSING"),
        "wrong_inverse_fails": inverse_attack("WRONG"),
        "reordered_inverse_fails": inverse_attack("REORDER"),
        "q5_collision_direction_reproduces": all(q5),
        "q5_collision_direction_fails_q11_transfer": not any(q11),
        "q5_collision_direction_fails_q23_transfer": not any(q23),
        "production_controls_all_pass": all(production["controls"].values()),
    }
    restoration = reuse_check()
    source = inspect_source(production_source)
    if (
        not all(controls.values())
        or not all(
            value for key, value in restoration.items() if key != "snapshot_used"
        )
        or restoration["snapshot_used"]
        or source["production_source_imports_oracle"]
        or not source["exact_projective_traversal_present"]
        or source["cryptographic_digest_used_as_state_equality"]
    ):
        fail("independent controls, restoration, or source isolation failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_PROJECTIVE_CUBIC_AIRY_PROGRAM_ORBIT_HISTORY_GROWTH_INDEPENDENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_source": source,
        "production_result_sha256": hashlib.sha256(
            production_result.read_bytes()
        ).hexdigest(),
        "case_certificates": certificates,
        "controls": controls,
        "restoration_and_reuse": restoration,
        "observed_resource_law": {
            "resident_dense_state_field_elements": "2*Q^2",
            "public_program_descriptor_field_elements": "2*DEPTH",
            "diagnostic_exact_code_set_entries": "DISTINCT_PROJECTIVE_FULL_STATES",
            "q5_depth4_exact_code_set_entries": 388125,
            "q5_depth4_program_histories": 390625,
            "q5_depth4_identifier_bits_before_and_after_quotient": 19,
            "accepted_algorithm_uses_enumerated_code_set": False,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "Q5_DEPTHS1_THROUGH3_PROJECTIVE_ORBITS_COLLISION_FREE",
            "Q5_DEPTH4_HAS625_RAW_EQUALITY_CLASSES_OF_SIZE5_ON_A_RANK5_SUBSPACE",
            "Q5_COLLISION_KERNEL_IS_THE_SINGLE_REPORTED_PROGRAM_DIRECTION",
            "Q11_DEPTHS1_AND2_PROJECTIVE_ORBITS_COLLISION_FREE",
            "EXACT_IN_PLACE_BRANCH_RESTORATION_AND_RESTORED_CARRIER_REUSE",
            "IDENTICAL_COMPACT_CLASSICAL_FACTOR_GRAPH_REMAINS",
        ],
        "rejected_interpretations": [
            "TRANSFERABLE_Q5_COLLISION_DIRECTION",
            "FIXED_HISTORY_FREE_NONLINEAR_CLOSURE",
            "PROJECTIVE_IDENTIFIER_BIT_WIDTH_REDUCTION_AT_Q5_DEPTH4",
            "UNIVERSAL_STATE_OR_INFORMATION_LOWER_BOUND",
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
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(
        run(args.production_source, args.production_result), indent=2, sort_keys=True
    ) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
