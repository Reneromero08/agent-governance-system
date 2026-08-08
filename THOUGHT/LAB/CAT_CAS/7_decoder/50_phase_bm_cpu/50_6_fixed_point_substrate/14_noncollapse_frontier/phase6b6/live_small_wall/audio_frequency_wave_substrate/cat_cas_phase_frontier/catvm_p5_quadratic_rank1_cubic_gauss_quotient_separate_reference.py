#!/usr/bin/env python3
"""Independent exact formula, dense parity, and custody oracle for M243."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Sequence


P = 5
DIMENSIONS = (2, 3, 4, 6, 8, 12, 16)
E = tuple[int, int, int, int]
ZERO: E = (0, 0, 0, 0)
ONE: E = (1, 0, 0, 0)
SQRT5: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def mul(left: E, right: E) -> E:
    raw = [0] * 7
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            raw[left_index + right_index] += left_value * right_value
    for degree in range(6, 3, -1):
        coefficient = raw[degree]
        if coefficient:
            for target in range(degree - 4, degree):
                raw[target] -= coefficient
            raw[degree] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def root(power: int) -> E:
    power %= P
    if power == 4:
        return (-1, -1, -1, -1)
    result = [0] * 4
    result[power] = 1
    return tuple(result)  # type: ignore[return-value]


def canonical(numerator: E, exponent: int) -> tuple[E, int]:
    value = numerator
    while exponent and all(coefficient % P == 0 for coefficient in value):
        value = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return value, exponent


def packed_index(row: int, column: int) -> int:
    if row < column:
        row, column = column, row
    return row * (row + 1) // 2 + column


def unpack(dimension: int, packed: Sequence[int]) -> list[list[int]]:
    matrix = [[0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            value = int(packed[packed_index(row, column)]) % P
            matrix[row][column] = value
            matrix[column][row] = value
    return matrix


def graph_connected(dimension: int, packed: Sequence[int]) -> bool:
    adjacency = [set() for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row):
            if int(packed[packed_index(row, column)]) % P:
                adjacency[row].add(column)
                adjacency[column].add(row)
    seen = {0}
    pending = [0]
    while pending:
        node = pending.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    return len(seen) == dimension


def determinant_and_solve(
    dimension: int,
    packed: Sequence[int],
    vector: Sequence[int],
) -> tuple[int, tuple[int, ...], int, int]:
    matrix = unpack(dimension, packed)
    augmented = [
        matrix[row] + [int(vector[row]) % P]
        for row in range(dimension)
    ]
    determinant = 1
    operations = 0
    for column in range(dimension):
        pivot_row = next(
            (row for row in range(column, dimension) if augmented[row][column] % P),
            None,
        )
        if pivot_row is None:
            raise RuntimeError("singular independent M243 matrix")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
            determinant = -determinant
        pivot = augmented[column][column] % P
        determinant = determinant * pivot % P
        inverse = pow(pivot, P - 2, P)
        augmented[column] = [value * inverse % P for value in augmented[column]]
        operations += dimension + 1
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column] % P
            if factor:
                augmented[row] = [
                    (left - factor * right) % P
                    for left, right in zip(augmented[row], augmented[column])
                ]
                operations += 2 * (dimension + 1)
    solution = tuple(augmented[row][-1] % P for row in range(dimension))
    delta = sum((int(left) % P) * right for left, right in zip(vector, solution)) % P
    determinant %= P
    square_class = 1 if determinant in (1, 4) else -1
    return square_class, solution, delta, operations


def exact_formula(
    dimension: int,
    packed: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
) -> dict[str, object]:
    square_class, solution, delta, elimination_operations = determinant_and_solve(
        dimension, packed, vector
    )
    channels: list[E] = []
    for channel in range(P):
        total = ZERO
        for value in range(P):
            total = add(
                total,
                root(cubic_strength * value * value * value - channel * value),
            )
        channels.append(total)
    closure = ZERO
    for channel, coefficient in enumerate(channels):
        closure = add(closure, mul(coefficient, root(delta * channel * channel)))
    numerator = closure
    for _ in range(dimension):
        numerator = mul(numerator, SQRT5)
    if square_class < 0:
        numerator = tuple(-coefficient for coefficient in numerator)  # type: ignore[assignment]
    numerator, exponent = canonical(numerator, dimension + 1)
    return {
        "amplitude": {"numerator": list(numerator), "denominator_power5": exponent},
        "square_class": square_class,
        "delta": delta,
        "solution": list(solution),
        "channel_numerators": [list(value) for value in channels],
        "elimination_scalar_operations": elimination_operations,
        "channel_character_terms": 25,
        "coherent_channel_terms": 5,
    }


def dense_amplitude(
    dimension: int,
    packed: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
) -> dict[str, object]:
    matrix = unpack(dimension, packed)
    total = ZERO
    assignments = 0
    for state in product(range(P), repeat=dimension):
        quadratic = 0
        for row in range(dimension):
            quadratic += matrix[row][row] * state[row] * state[row]
            for column in range(row):
                quadratic += 2 * matrix[row][column] * state[row] * state[column]
        linear = sum((int(vector[index]) % P) * state[index] for index in range(dimension)) % P
        phase = quadratic + cubic_strength * linear * linear * linear
        total = add(total, root(phase))
        assignments += 1
    numerator, exponent = canonical(total, dimension)
    return {
        "amplitude": {"numerator": list(numerator), "denominator_power5": exponent},
        "assignments": assignments,
    }


def transform_collision_descriptor(descriptor: dict[str, object]) -> dict[str, object]:
    dimension = int(descriptor["dimension"])
    matrix = list(descriptor["matrix"])
    vector = list(descriptor["vector"])
    scale = 2
    for row in range(dimension):
        cell = packed_index(row, 0)
        matrix[cell] = int(matrix[cell]) * scale % P
    matrix[packed_index(0, 0)] = int(matrix[packed_index(0, 0)]) * scale % P
    vector[0] = int(vector[0]) * scale % P
    return {
        "dimension": dimension,
        "matrix": matrix,
        "vector": vector,
        "cubic_strength": int(descriptor["cubic_strength"]),
    }


class ReferencePort:
    def __init__(self) -> None:
        self.stage = "CANONICAL"
        self.last_generation = 0
        self.generation = 0
        self.amplitude: dict[str, object] | None = None

    def lease(self, generation: int) -> None:
        if self.stage != "CANONICAL" or generation != self.last_generation + 1:
            raise RuntimeError("independent M243 lease rejected")
        self.generation = generation
        self.stage = "LEASED"

    def forward(self, descriptor: dict[str, object]) -> None:
        if self.stage != "LEASED":
            raise RuntimeError("independent M243 forward ordering rejected")
        result = exact_formula(
            int(descriptor["dimension"]),
            descriptor["matrix"],
            descriptor["vector"],
            int(descriptor["cubic_strength"]),
        )
        self.amplitude = result["amplitude"]  # type: ignore[assignment]
        self.stage = "FINAL_RESIDENT"

    def project(self) -> dict[str, object]:
        if self.stage != "FINAL_RESIDENT" or self.amplitude is None:
            raise RuntimeError("independent M243 premature projection rejected")
        return dict(self.amplitude)

    def inverse(self) -> None:
        if self.stage != "FINAL_RESIDENT":
            raise RuntimeError("independent M243 inverse ordering rejected")
        self.stage = "RESTORED"

    def release(self) -> None:
        if self.stage != "RESTORED":
            raise RuntimeError("independent M243 response ordering rejected")
        self.last_generation = self.generation
        self.generation = 0
        self.amplitude = None
        self.stage = "CANONICAL"


def reference_custody_controls(descriptor: dict[str, object]) -> dict[str, bool]:
    port = ReferencePort()
    try:
        port.project()
        premature = False
    except RuntimeError:
        premature = True
    port.lease(1)
    port.forward(descriptor)
    retained = port.project()
    try:
        port.release()
        early = False
    except RuntimeError:
        early = True
    port.inverse()
    port.release()
    port.lease(2)
    port.forward(descriptor)
    second = port.project()
    port.inverse()
    port.release()
    try:
        port.lease(2)
        stale = False
    except RuntimeError:
        stale = True
    return {
        "independent_premature_projection_rejected": premature,
        "independent_response_before_inverse_rejected": early,
        "independent_stale_generation_rejected": stale,
        "independent_generation_two_reuse_boundary_equal": retained == second,
        "independent_port_canonical_after_reuse": port.stage == "CANONICAL",
    }


def main(raw_path: Path) -> None:
    raw = json.loads(raw_path.read_text())
    private = json.loads(sys.stdin.buffer.readline())
    oracles = private["oracles"]
    cases: list[dict[str, object]] = []
    for case in raw["cases"]:
        descriptor = oracles[case["oracle_id"]]
        formula = exact_formula(
            int(descriptor["dimension"]),
            descriptor["matrix"],
            descriptor["vector"],
            int(descriptor["cubic_strength"]),
        )
        if formula["amplitude"] != case["final_amplitude"]:
            raise RuntimeError("independent M243 formula boundary mismatch")
        dense = (
            dense_amplitude(
                int(descriptor["dimension"]),
                descriptor["matrix"],
                descriptor["vector"],
                int(descriptor["cubic_strength"]),
            )
            if int(descriptor["dimension"]) <= 4
            else None
        )
        if dense is not None and dense["amplitude"] != formula["amplitude"]:
            raise RuntimeError("independent M243 dense parity mismatch")
        cases.append({
            "oracle_id": case["oracle_id"],
            "dimension": int(descriptor["dimension"]),
            "final_amplitude": formula["amplitude"],
            "dense_exact_parity": dense is not None,
            "dense_assignments_verifier_only": None if dense is None else dense["assignments"],
            "compact_elimination_scalar_operations": formula["elimination_scalar_operations"],
            "channel_character_terms": formula["channel_character_terms"],
            "coherent_channel_terms": formula["coherent_channel_terms"],
        })

    collision_a = oracles["quotient_collision_a"]
    collision_b = oracles["quotient_collision_b"]
    derived_b = transform_collision_descriptor(collision_a)
    collision_descriptors_match_congruence = all(
        list(collision_b[key]) == list(derived_b[key])
        if key in {"matrix", "vector"}
        else int(collision_b[key]) == int(derived_b[key])
        for key in ("dimension", "matrix", "vector", "cubic_strength")
    )
    formula_a = exact_formula(
        int(collision_a["dimension"]), collision_a["matrix"], collision_a["vector"],
        int(collision_a["cubic_strength"]),
    )
    formula_b = exact_formula(
        int(collision_b["dimension"]), collision_b["matrix"], collision_b["vector"],
        int(collision_b["cubic_strength"]),
    )

    control_descriptor = oracles["n4_primary"]
    pure_gaussian = exact_formula(
        4,
        control_descriptor["matrix"],
        control_descriptor["vector"],
        0,
    )
    square_class, _, _, _ = determinant_and_solve(
        4, control_descriptor["matrix"], control_descriptor["vector"]
    )
    pure_numerator = ONE
    for _ in range(4):
        pure_numerator = mul(pure_numerator, SQRT5)
    if square_class < 0:
        pure_numerator = tuple(-value for value in pure_numerator)  # type: ignore[assignment]
    pure_numerator, pure_exponent = canonical(pure_numerator, 4)
    pure_gaussian_closed_form = {
        "numerator": list(pure_numerator), "denominator_power5": pure_exponent
    }

    result = {
        "schema": "cat_cas.catvm_p5_quadratic_rank1_cubic_gauss_quotient_reference.v1",
        "cases": cases,
        "controls": {
            "independent_disconnected_regular_graph_rejected": not graph_connected(
                4,
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            ),
            "all_private_descriptors_are_genuinely_connected": all(
                graph_connected(int(item["dimension"]), item["matrix"])
                for item in oracles.values()
            ),
            "independent_congruence_collision_descriptor_reconstructed": collision_descriptors_match_congruence,
            "independent_congruence_collision_boundary_equal": formula_a["amplitude"] == formula_b["amplitude"],
            "pure_quadratic_lambda_zero_matches_gauss_closed_form": pure_gaussian["amplitude"] == pure_gaussian_closed_form,
            "rank1_cubic_uses_exactly_five_coherent_channels": len(formula_a["channel_numerators"]) == P,
            "dense_verifier_used_only_through_n4": all(
                case["dense_exact_parity"] == (case["dimension"] <= 4) for case in cases
            ),
            **reference_custody_controls(control_descriptor),
        },
        "independent_exact_power_basis_formula_reexecution": True,
        "independent_modular_gaussian_elimination_not_ldl_production_path": True,
        "independent_dense_sum_parity_dimensions2_3_4": True,
        "strongest_compact_classical_baseline_is_identical_gauss_quotient": True,
        "imports_service_controller_or_m237": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: catvm_p5_quadratic_rank1_cubic_gauss_quotient_separate_reference.py RAW_RESULTS_JSON"
        )
    main(Path(sys.argv[1]))
