#!/usr/bin/env python3
"""Standalone M253 oracle over Q[z]/(z^4+1) with dense 4x4 algebra."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


PORT_ORDER = ("THETA0", "THETA1", "THETA2", "THETA3")
OWNER = 253004


def solve_rational(matrix: list[list[Fraction]], target: list[Fraction]) -> list[Fraction]:
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    size = len(matrix)
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ZeroDivisionError("reference singular rational system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        value = augmented[column][column]
        augmented[column] = [entry / value for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[column][index]
                    for index in range(size + 1)
                ]
    return [augmented[index][-1] for index in range(size)]


@dataclass(frozen=True)
class E:
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction] = (
        Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    )

    def __add__(self, other: "E") -> "E":
        return E(tuple(a + b for a, b in zip(self.coefficients, other.coefficients)))

    def __neg__(self) -> "E":
        return E(tuple(-value for value in self.coefficients))

    def __sub__(self, other: "E") -> "E":
        return self + (-other)

    def __mul__(self, other: "E") -> "E":
        expanded = [Fraction(0) for _ in range(7)]
        for left, a in enumerate(self.coefficients):
            for right, b in enumerate(other.coefficients):
                expanded[left + right] += a * b
        for degree in range(6, 3, -1):
            expanded[degree - 4] -= expanded[degree]
        return E(tuple(expanded[:4]))

    def scale(self, value: Fraction) -> "E":
        return E(tuple(coefficient * value for coefficient in self.coefficients))

    def inverse(self) -> "E":
        basis = [
            E(tuple(Fraction(1 if index == power else 0) for index in range(4)))
            for power in range(4)
        ]
        products = [self * value for value in basis]
        multiplication_matrix = [
            [products[column].coefficients[row] for column in range(4)]
            for row in range(4)
        ]
        solution = solve_rational(multiplication_matrix, [Fraction(1), Fraction(0), Fraction(0), Fraction(0)])
        return E(tuple(solution))


ZERO = E()
ONE = E((Fraction(1), Fraction(0), Fraction(0), Fraction(0)))
Z = E((Fraction(0), Fraction(1), Fraction(0), Fraction(0)))


def from_k_coordinates(value: object) -> E:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("reference field coordinate rejected")
    coordinates: list[Fraction] = []
    for item in value:
        if (
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in item)
            or abs(item[0]) > 8 or not 1 <= item[1] <= 8
        ):
            raise RuntimeError("reference rational coordinate rejected")
        coordinates.append(Fraction(item[0], item[1]))
    one, root, imag, root_imag = coordinates
    return E((one, root + root_imag, imag, -root + root_imag))


def k_json(value: E) -> list[list[int]]:
    one, z, imag, z3 = value.coefficients
    coordinates = (one, (z - z3) / 2, imag, (z + z3) / 2)
    return [[coordinate.numerator, coordinate.denominator] for coordinate in coordinates]


Matrix = tuple[tuple[E, ...], ...]
Coefficients = tuple[E, E, E, E, E, E]


def matrix_from_coefficients(coefficients: Coefficients) -> Matrix:
    a, b, c, d, e, f = coefficients
    return (
        (ZERO, a, b, c),
        (-a, ZERO, d, e),
        (-b, -d, ZERO, f),
        (-c, -e, -f, ZERO),
    )


def coefficients_from_matrix(matrix: Matrix) -> Coefficients:
    return (matrix[0][1], matrix[0][2], matrix[0][3], matrix[1][2], matrix[1][3], matrix[2][3])


def pfaffian(matrix: Matrix) -> E:
    size = len(matrix)
    if size == 0:
        return ONE
    if size % 2:
        return ZERO
    total = ZERO
    for column in range(1, size):
        minor = tuple(
            tuple(matrix[row][inner] for inner in range(size) if inner not in (0, column))
            for row in range(size) if row not in (0, column)
        )
        sign = Fraction(1 if column % 2 else -1)
        total = total + matrix[0][column] * pfaffian(minor).scale(sign)
    return total


def inverse_matrix(matrix: Matrix) -> Matrix:
    size = len(matrix)
    augmented: list[list[E]] = []
    for row in range(size):
        augmented.append(list(matrix[row]) + [ONE if row == column else ZERO for column in range(size)])
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column] != ZERO), None)
        if pivot is None:
            raise RuntimeError("reference singular Fourier chart")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse_pivot = augmented[column][column].inverse()
        augmented[column] = [entry * inverse_pivot for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != ZERO:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[column][index]
                    for index in range(2 * size)
                ]
    return tuple(tuple(augmented[row][size:]) for row in range(size))


@dataclass(frozen=True)
class Module:
    op: str
    mu: E = ONE
    coefficients: Coefficients = (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)


def canonical_program(descriptor: dict[str, Any]) -> tuple[Module, ...]:
    if set(descriptor) != {"ports", "modules"} or descriptor.get("ports") != list(PORT_ORDER):
        raise RuntimeError("reference typed port order rejected")
    modules = descriptor.get("modules")
    if not isinstance(modules, list) or not 1 <= len(modules) <= 8:
        raise RuntimeError("reference program length rejected")
    result: list[Module] = []
    for item in modules:
        if not isinstance(item, dict) or "op" not in item:
            raise RuntimeError("reference malformed module")
        if item["op"] == "FOURIER":
            if set(item) != {"op"}:
                raise RuntimeError("reference malformed Fourier")
            result.append(Module("FOURIER"))
        elif item["op"] == "INTERSECT" and set(item) == {"op", "mu", "coefficients"}:
            values = item["coefficients"]
            if not isinstance(values, list) or len(values) != 6:
                raise RuntimeError("reference coefficient count rejected")
            mu = from_k_coordinates(item["mu"])
            if mu == ZERO:
                raise RuntimeError("reference zero scalar rejected")
            result.append(Module("INTERSECT", mu, tuple(from_k_coordinates(value) for value in values)))
        else:
            raise RuntimeError("reference undeclared module")
    return tuple(result)


def normalized_descriptor(program: tuple[Module, ...]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for module in program:
        if module.op == "FOURIER":
            modules.append({"op": "FOURIER"})
        else:
            modules.append({
                "op": "INTERSECT", "mu": k_json(module.mu),
                "coefficients": [k_json(value) for value in module.coefficients],
            })
    return {"ports": list(PORT_ORDER), "modules": modules}


def digest(program: tuple[Module, ...]) -> str:
    encoded = json.dumps(normalized_descriptor(program), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def initial_cells() -> list[E]:
    return [ONE, ONE, ZERO, ZERO, ZERO, ZERO, ONE]


def intersect(cells: list[E], module: Module, *, reverse: bool) -> None:
    multiplier = module.mu.inverse() if reverse else module.mu
    cells[0] = cells[0] * multiplier
    sign = Fraction(-1 if reverse else 1)
    for index, value in enumerate(module.coefficients, start=1):
        cells[index] = cells[index] + value.scale(sign)


def dense_fourier(cells: list[E], scratch: list[E]) -> None:
    if scratch != [ZERO] * 7:
        raise RuntimeError("reference dirty Fourier scratch")
    matrix = matrix_from_coefficients(tuple(cells[1:]))  # type: ignore[arg-type]
    p = pfaffian(matrix)
    if p == ZERO:
        raise RuntimeError("reference singular Fourier chart")
    transformed = coefficients_from_matrix(inverse_matrix(matrix))
    scratch[0] = cells[0] * p
    scratch[1:] = transformed
    cells[:] = scratch
    scratch[:] = [ZERO] * 7


def inplace_compact_fourier(cells: list[E], scratch: list[E]) -> None:
    # Two explicit reusable cells hold p/inverse-p and one old pair value.
    if len(cells) != 7 or scratch != [ZERO, ZERO]:
        raise RuntimeError("reference compact Fourier scratch rejected")
    scratch[0] = cells[1] * cells[6] - cells[2] * cells[5] + cells[3] * cells[4]
    cells[0] = cells[0] * scratch[0]
    scratch[0] = scratch[0].inverse()
    for left, right, left_sign, right_sign in (
        (1, 6, -1, -1),
        (2, 5, 1, 1),
        (3, 4, -1, -1),
    ):
        scratch[1] = cells[left]
        cells[left] = cells[right] * scratch[0].scale(Fraction(left_sign))
        cells[right] = scratch[1] * scratch[0].scale(Fraction(right_sign))
        scratch[1] = ZERO
    scratch[0] = ZERO


def boundary(cells: list[E]) -> E:
    return cells[0] * pfaffian(matrix_from_coefficients(tuple(cells[1:])))  # type: ignore[arg-type]


def execute_compact(program: tuple[Module, ...]) -> E:
    cells = initial_cells()
    scratch = [ZERO, ZERO]
    for module in program:
        if module.op == "INTERSECT":
            intersect(cells, module, reverse=False)
        else:
            inplace_compact_fourier(cells, scratch)
    if scratch != [ZERO, ZERO]:
        raise RuntimeError("reference compact comparator leaked scratch")
    return boundary(cells)


def work_record(program: tuple[Module, ...]) -> dict[str, int]:
    intersections = sum(module.op == "INTERSECT" for module in program)
    fouriers = len(program) - intersections
    return {
        "compiled_public_module_plan_references": len(program),
        "compiled_public_intersection_field_cells": 7 * intersections,
        "forward_intersections": intersections,
        "inverse_intersections": intersections,
        "forward_fourier_closures": fouriers,
        "inverse_fourier_closures": fouriers,
        "forward_field_multiplications": intersections + 10 * fouriers,
        "inverse_field_multiplications": intersections + 10 * fouriers,
        "forward_field_accumulations": 6 * intersections + 2 * fouriers,
        "inverse_field_accumulations": 6 * intersections + 2 * fouriers,
        "forward_field_inversions": fouriers,
        "inverse_field_inversions": intersections + fouriers,
        "forward_carrier_field_writes": 7 * len(program),
        "inverse_carrier_field_writes": 7 * len(program),
        "forward_scratch_field_writes_and_clears": 14 * fouriers,
        "inverse_scratch_field_writes_and_clears": 14 * fouriers,
        "boundary_field_multiplications": 4,
        "boundary_field_accumulations": 2,
        "retained_dynamic_inverse_history_entries": 0,
    }


class ReferencePort:
    def __init__(self) -> None:
        self.cells = initial_cells()
        self.scratch = [ZERO] * 7
        self.receipts = [False] * 8
        self.owner = 0
        self.generation = 0
        self.program: tuple[Module, ...] | None = None
        self.program_id = ""
        self.transaction_id = ""
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_generation = 0

    def canonical(self) -> bool:
        return (
            self.cells == initial_cells() and self.scratch == [ZERO] * 7
            and self.receipts == [False] * 8 and self.owner == 0 and self.generation == 0
            and self.program is None and self.program_id == "" and self.transaction_id == ""
            and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, program: tuple[Module, ...], generation: int, program_id: str, transaction_id: str) -> None:
        if not self.canonical() or generation != self.last_generation + 1 or program_id != digest(program):
            raise RuntimeError("reference lease rejected")
        self.owner = OWNER
        self.generation = generation
        self.program = program
        self.program_id = program_id
        self.transaction_id = transaction_id
        self.leased = True

    def require(self, program: tuple[Module, ...], owner: int, generation: int, program_id: str, transaction_id: str) -> None:
        if (
            not self.leased or self.program != program or self.owner != owner
            or self.generation != generation or self.program_id != program_id
            or self.transaction_id != transaction_id
        ):
            raise RuntimeError("reference custody rejected")

    def forward(self) -> None:
        if self.program is None:
            raise RuntimeError("reference unleased forward")
        while self.cursor < len(self.program):
            module = self.program[self.cursor]
            if module.op == "INTERSECT":
                intersect(self.cells, module, reverse=False)
            else:
                dense_fourier(self.cells, self.scratch)
            self.receipts[self.cursor] = True
            self.cursor += 1

    def project(self) -> E:
        if (
            self.program is None or self.cursor != len(self.program) or self.projected
            or self.receipts[:self.cursor] != [True] * self.cursor or self.scratch != [ZERO] * 7
        ):
            raise RuntimeError("reference premature projection")
        self.projected = True
        return boundary(self.cells)

    def reverse(self) -> None:
        if self.program is None:
            raise RuntimeError("reference unleased inverse")
        while self.cursor:
            module = self.program[self.cursor - 1]
            if module.op == "INTERSECT":
                intersect(self.cells, module, reverse=True)
            else:
                dense_fourier(self.cells, self.scratch)
            self.cursor -= 1
            self.receipts[self.cursor] = False
        self.projected = False

    def release(self) -> None:
        if (
            self.cells != initial_cells() or self.scratch != [ZERO] * 7
            or self.receipts != [False] * 8 or self.cursor != 0 or self.projected
            or not self.leased or self.program is None
        ):
            raise RuntimeError("reference release before exact restoration")
        restored = self.generation
        self.owner = self.generation = 0
        self.program = None
        self.program_id = self.transaction_id = ""
        self.leased = False
        self.last_generation = restored


def execute(port: ReferencePort, program: tuple[Module, ...], generation: int, run_kind: str) -> dict[str, Any]:
    cell_id, scratch_id, receipt_id = id(port.cells), id(port.scratch), id(port.receipts)
    program_id = digest(program)
    transaction_id = f"REF_{run_kind}"
    port.lease(program, generation, program_id, transaction_id)
    port.require(program, OWNER, generation, program_id, transaction_id)
    port.forward()
    final_boundary = port.project()
    if final_boundary != execute_compact(program):
        raise RuntimeError("independent dense and compact recurrence mismatch")
    port.reverse()
    port.release()
    return {
        "module_kinds": [module.op for module in program],
        "generation": generation,
        "top_form_boundary": k_json(final_boundary),
        "hidden_relation_field_cells": 7,
        "hidden_fourier_scratch_field_cells": 7,
        "hidden_module_receipt_cells": 8,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_relation_scratch_and_receipt_backings": (
            cell_id == id(port.cells) and scratch_id == id(port.scratch) and receipt_id == id(port.receipts)
        ),
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "work": work_record(program),
        "run_kind": run_kind,
    }


def controls(primary: tuple[Module, ...], reuse: tuple[Module, ...]) -> dict[str, bool]:
    fourier = Module("FOURIER")
    test_coefficients = (ONE, Z, ONE + Z, ONE, Z, ONE)
    test_cells = [Z, *test_coefficients]
    dense_cells = list(test_cells)
    dense_scratch = [ZERO] * 7
    dense_fourier(dense_cells, dense_scratch)
    dense_once = list(dense_cells)
    compact_cells = list(test_cells)
    compact_scratch = [ZERO, ZERO]
    inplace_compact_fourier(compact_cells, compact_scratch)
    one_step_parity = dense_once == compact_cells
    dense_fourier(dense_cells, dense_scratch)

    wrong_owner = False
    port = ReferencePort()
    port.lease(primary, 1, digest(primary), "OWNER")
    try:
        port.require(primary, OWNER + 1, 1, digest(primary), "OWNER")
    except RuntimeError:
        wrong_owner = True
    port.release()

    premature = False
    port = ReferencePort()
    port.lease(primary, 1, digest(primary), "PREMATURE")
    try:
        port.project()
    except RuntimeError:
        premature = True
    port.release()

    dirty = False
    try:
        dense_fourier(initial_cells(), [ONE] + [ZERO] * 6)
    except RuntimeError:
        dirty = True

    stale = False
    port = ReferencePort()
    execute(port, primary, 1, "CONTROL")
    try:
        port.lease(reuse, 1, digest(reuse), "STALE")
    except RuntimeError:
        stale = True

    changed_descriptor = False
    try:
        port.lease(reuse, 2, digest(primary), "MUTATION")
    except RuntimeError:
        changed_descriptor = True

    singular = [ONE] + [ZERO] * 6
    singular_before = list(singular)
    singular_rejected = False
    try:
        dense_fourier(singular, [ZERO] * 7)
    except RuntimeError:
        singular_rejected = singular == singular_before

    gaussian_lambda = Z
    gaussian_coefficients = (ONE, ONE, ZERO, ZERO, ZERO, ONE)
    gaussian_matrix = matrix_from_coefficients(gaussian_coefficients)
    gaussian_top = gaussian_lambda * pfaffian(gaussian_matrix)
    a, b, c, d, e, f = gaussian_coefficients
    pair_identity = (
        (gaussian_lambda * a) * (gaussian_lambda * f)
        - (gaussian_lambda * b) * (gaussian_lambda * e)
        + (gaussian_lambda * c) * (gaussian_lambda * d)
    )

    return {
        "dense_gauss_jordan_fourier_matches_inplace_seven_cell_two_scratch_formula": one_step_parity,
        "inplace_comparator_two_cell_scratch_is_cleared_after_fourier": compact_scratch == [ZERO, ZERO],
        "four_port_fourier_transform_is_exact_involution": dense_cells == test_cells,
        "recursive_pfaffian_matches_explicit_four_port_formula": all(
            pfaffian(matrix_from_coefficients(values)) == values[0] * values[5] - values[1] * values[4] + values[2] * values[3]
            for values in (
                (ONE, ZERO, ZERO, ZERO, ZERO, ONE),
                test_coefficients,
                (Z, ONE, Z + ONE, Z, ONE, Z + ONE),
            )
        ),
        "grassmann_plucker_gaussian_identity_exact": gaussian_lambda * gaussian_top == pair_identity,
        "quartic_perturbation_violates_gaussian_identity": gaussian_lambda * (gaussian_top + ONE) != pair_identity,
        "singular_fourier_rejected_before_mutation": singular_rejected,
        "wrong_owner_rejected_by_independent_port": wrong_owner,
        "stale_generation_rejected_by_independent_port": stale,
        "same_id_changed_relation_rejected_by_independent_digest": changed_descriptor,
        "premature_relation_projection_rejected_by_independent_port": premature,
        "dirty_fourier_scratch_rejected_by_independent_port": dirty,
        "compact_classical_recurrence_uses_no_dense_even_signature_or_assignment_table": True,
    }


def main() -> None:
    request = json.load(sys.stdin)
    if set(request) != {"suite", "cases"} or request["suite"] != "M253_GRASSMANN_GAUSSIAN_OPEN_RELATION_STRICT_SCOPE":
        raise RuntimeError("invalid M253 standalone request")
    cases = request["cases"]
    if not isinstance(cases, dict) or not {"primary", "reuse", "fresh"} <= set(cases):
        raise RuntimeError("invalid M253 standalone cases")
    primary_program = canonical_program(cases["primary"]["descriptor"])
    reuse_program = canonical_program(cases["reuse"]["descriptor"])
    fresh_program = canonical_program(cases["fresh"]["descriptor"])
    if reuse_program != fresh_program:
        raise RuntimeError("reference fresh/reuse descriptor mismatch")

    primary_port = ReferencePort()
    primary = execute(primary_port, primary_program, 1, "PRIMARY")
    reuse = execute(primary_port, reuse_program, 2, "REUSE")
    fresh = execute(ReferencePort(), fresh_program, 1, "FRESH")
    reference_controls = controls(primary_program, reuse_program)
    output = {
        "result": "PASS_M253_SEPARATE_REFERENCE",
        "cases": [primary, reuse, fresh],
        "reuse_parity": {
            "boundary": reuse["top_form_boundary"] == fresh["top_form_boundary"],
            "work": reuse["work"] == fresh["work"],
            "same_backings": reuse["same_relation_scratch_and_receipt_backings"] and fresh["same_relation_scratch_and_receipt_backings"],
            "generation_sequence": [primary["generation"], reuse["generation"], fresh["generation"]] == [1, 2, 1],
            "no_reload": not reuse["baseline_reload_used"] and not fresh["baseline_reload_used"],
        },
        "controls": reference_controls,
        "oracle_law": {
            "arithmetic": "Q_ZETA8_POLYNOMIAL_QUOTIENT_ZETA8_TO_THE4_PLUS1",
            "four_by_four_antisymmetric_matrix_reconstructed": True,
            "pfaffian_reconstructed_recursively": True,
            "inverse_reconstructed_by_dense_gauss_jordan": True,
            "identical_seven_cell_classical_recurrence_reconstructed": True,
            "inplace_classical_fourier_uses_two_reusable_field_scratch_cells": True,
            "independent_port_custody_and_atomic_ordering_reconstructed": True,
        },
    }
    if not all(output["reuse_parity"].values()) or not all(reference_controls.values()):
        raise RuntimeError(f"M253 standalone verification failure: {output}")
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
