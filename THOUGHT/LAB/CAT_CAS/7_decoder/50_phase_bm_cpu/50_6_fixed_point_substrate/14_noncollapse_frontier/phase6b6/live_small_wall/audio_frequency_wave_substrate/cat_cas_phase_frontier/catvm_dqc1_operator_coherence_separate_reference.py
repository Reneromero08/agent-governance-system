#!/usr/bin/env python3
"""Standalone M251 Q[zeta8]/(zeta8^4+1) matrix and custody oracle."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class E:
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction] = (
        Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    )

    def __add__(self, other: "E") -> "E":
        return E(tuple(left + right for left, right in zip(self.coefficients, other.coefficients)))

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

    def bar(self) -> "E":
        a, b, c, d = self.coefficients
        return E((a, -d, -c, -b))


Z = E()
O = E((Fraction(1), Fraction(0), Fraction(0), Fraction(0)))
W = E((Fraction(0), Fraction(1), Fraction(0), Fraction(0)))
J = W * W
INV_ROOT2 = E((Fraction(0), Fraction(1, 2), Fraction(0), Fraction(-1, 2)))


def canonical_json(value: E) -> list[list[int]]:
    a, b, c, d = value.coefficients
    coordinates = (a, (b - d) / 2, c, (b + d) / 2)
    return [[item.numerator, item.denominator] for item in coordinates]


Matrix = tuple[tuple[E, ...], ...]


def freeze(matrix: Any) -> Matrix:
    return tuple(tuple(row) for row in matrix)


def mutable(matrix: Matrix) -> list[list[E]]:
    return [list(row) for row in matrix]


def overwrite(target: list[list[E]], source: Matrix) -> None:
    if len(target) != len(source) or any(len(left) != len(right) for left, right in zip(target, source)):
        raise RuntimeError("reference matrix backing shape changed")
    for row, values in zip(target, source):
        row[:] = values


def eye(size: int) -> Matrix:
    return tuple(tuple(O if row == column else Z for column in range(size)) for row in range(size))


def product(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    inner_size = len(right)
    columns = len(right[0])
    return tuple(
        tuple(sum((left[row][inner] * right[inner][column] for inner in range(inner_size)), Z) for column in range(columns))
        for row in range(rows)
    )


def adjoint(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[column][row].bar() for column in range(len(matrix))) for row in range(len(matrix[0])))


def tensor(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(left[i][j] * right[k][l] for j in range(len(left)) for l in range(len(right)))
        for i in range(len(left)) for k in range(len(right))
    )


I2 = eye(2)
H2: Matrix = ((INV_ROOT2, INV_ROOT2), (INV_ROOT2, -INV_ROOT2))
T2: Matrix = ((O, Z), (Z, W))
C01: Matrix = ((O, Z, Z, Z), (Z, O, Z, Z), (Z, Z, Z, O), (Z, Z, O, Z))
C10: Matrix = ((O, Z, Z, Z), (Z, Z, Z, O), (Z, Z, O, Z), (Z, O, Z, Z))
H0_4 = tensor(H2, I2)
H1_4 = tensor(I2, H2)
T0_4 = tensor(T2, I2)
T1_4 = tensor(I2, T2)
GATE_LIBRARY = {
    "H0": H0_4, "H1": H1_4, "T0": T0_4, "T1": T1_4,
    "CNOT01": C01, "CNOT10": C10,
}


def gate(name: str) -> Matrix:
    if name not in GATE_LIBRARY:
        raise RuntimeError("reference gate rejected")
    return GATE_LIBRARY[name]


def validate_word(word: tuple[str, ...]) -> None:
    if not 1 <= len(word) <= 8:
        raise RuntimeError("reference word length rejected")
    for name in word:
        gate(name)


def direct_matrix(word: tuple[str, ...]) -> Matrix:
    validate_word(word)
    result = eye(4)
    for name in word:
        result = product(gate(name), result)
    return result


def controlled(data_gate: Matrix) -> Matrix:
    return tuple(
        tuple(
            (O if row == column else Z) if row < 4 and column < 4
            else data_gate[row - 4][column - 4] if row >= 4 and column >= 4
            else Z
            for column in range(8)
        )
        for row in range(8)
    )


def initial_density() -> Matrix:
    return tuple(
        tuple(
            E((Fraction(1, 8), Fraction(0), Fraction(0), Fraction(0)))
            if row % 4 == column % 4 else Z
            for column in range(8)
        )
        for row in range(8)
    )


def dephased_density() -> Matrix:
    return tuple(
        tuple(E((Fraction(1, 8), Fraction(0), Fraction(0), Fraction(0))) if row == column else Z for column in range(8))
        for row in range(8)
    )


def basis_density() -> Matrix:
    return tuple(
        tuple(E((Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0))) if row % 4 == 0 and column % 4 == 0 else Z for column in range(8))
        for row in range(8)
    )


def evolve(density: Matrix, word: tuple[str, ...]) -> Matrix:
    result = density
    for name in word:
        operation = controlled(gate(name))
        result = product(product(operation, result), adjoint(operation))
    return result


def reverse_evolve(density: Matrix, word: tuple[str, ...]) -> Matrix:
    result = density
    for name in reversed(word):
        operation = controlled(adjoint(gate(name)))
        result = product(product(operation, result), adjoint(operation))
    return result


def boundary(density: Matrix) -> E:
    return sum((density[4 + data][data].scale(Fraction(2)) for data in range(4)), Z)


def data_marginal_is_maximally_mixed(density: Matrix) -> bool:
    for data_row in range(4):
        for data_column in range(4):
            actual = density[data_row][data_column] + density[4 + data_row][4 + data_column]
            expected = O.scale(Fraction(1, 4)) if data_row == data_column else Z
            if actual != expected:
                return False
    return True


def joint_is_product_with_mixed_data(density: Matrix) -> bool:
    control = tuple(
        tuple(
            sum((density[4 * left + data][4 * right + data] for data in range(4)), Z)
            for right in range(2)
        )
        for left in range(2)
    )
    for left in range(2):
        for right in range(2):
            for data_row in range(4):
                for data_column in range(4):
                    expected = control[left][right].scale(Fraction(1, 4)) if data_row == data_column else Z
                    if density[4 * left + data_row][4 * right + data_column] != expected:
                        return False
    return True


def normalized_trace(matrix: Matrix) -> E:
    return sum((matrix[index][index] for index in range(4)), Z).scale(Fraction(1, 4))


def descriptor_digest(word: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(word, separators=(",", ":")).encode()).hexdigest()


class ReferencePort:
    def __init__(self) -> None:
        self.density = mutable(initial_density())
        self.scratch = [Z, Z, Z, Z]
        self.owner = 0
        self.generation = 0
        self.digest = ""
        self.word: tuple[str, ...] | None = None
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_generation = 0

    def canonical(self) -> bool:
        return (
            freeze(self.density) == initial_density() and self.scratch == [Z] * 4
            and self.owner == 0 and self.generation == 0 and self.digest == ""
            and self.word is None and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, word: tuple[str, ...], owner: int, generation: int, digest: str) -> None:
        validate_word(word)
        if not self.canonical() or owner != 251004 or digest != descriptor_digest(word):
            raise RuntimeError("reference lease rejected")
        if generation != self.last_generation + 1:
            raise RuntimeError("reference generation rejected")
        self.owner, self.generation, self.digest, self.word, self.leased = owner, generation, digest, word, True

    def forward(self, name: str) -> None:
        if not self.leased or self.projected or self.word is None or name != self.word[self.cursor] or any(value != Z for value in self.scratch):
            raise RuntimeError("reference forward custody rejected")
        operation = controlled(gate(name))
        overwrite(self.density, product(product(operation, self.density), adjoint(operation)))
        self.cursor += 1

    def project(self) -> E:
        if self.word is None or self.cursor != len(self.word) or self.projected:
            raise RuntimeError("reference premature projection")
        value = boundary(self.density)
        self.projected = True
        return value

    def reverse(self, name: str) -> None:
        if self.word is None or name != self.word[self.cursor - 1]:
            raise RuntimeError("reference inverse custody rejected")
        operation = controlled(adjoint(gate(name)))
        overwrite(self.density, product(product(operation, self.density), adjoint(operation)))
        self.cursor -= 1
        if self.cursor == 0:
            self.projected = False

    def release(self) -> None:
        if freeze(self.density) != initial_density() or self.scratch != [Z] * 4 or self.cursor or self.projected or not self.leased:
            raise RuntimeError("reference release before restoration")
        self.last_generation = self.generation
        self.owner = self.generation = 0
        self.digest = ""
        self.word = None
        self.leased = False


def work_for_length(length: int) -> dict[str, int]:
    return {
        "compiled_public_gate_plan_matrix_references": length,
        "forward_controlled_gates": length,
        "inverse_controlled_gates": length,
        "forward_field_multiply_terms": 256 * length,
        "inverse_field_multiply_terms": 256 * length,
        "forward_density_field_writes": 64 * length,
        "inverse_density_field_writes": 64 * length,
        "forward_scratch_result_writes": 64 * length,
        "inverse_scratch_result_writes": 64 * length,
        "forward_scratch_clear_writes": 64 * length,
        "inverse_scratch_clear_writes": 64 * length,
        "boundary_field_multiplications": 4,
        "boundary_field_accumulations": 4,
        "retained_dynamic_inverse_history_entries": 0,
    }


def run(port: ReferencePort, word: tuple[str, ...], generation: int, kind: str) -> dict[str, Any]:
    ids = (id(port.density), id(port.scratch))
    port.lease(word, 251004, generation, descriptor_digest(word))
    for name in word:
        port.forward(name)
    value = port.project()
    forward_density = freeze(port.density)
    direct = normalized_trace(direct_matrix(word))
    if value != direct:
        raise RuntimeError("reference density/direct trace mismatch")
    marginal_mixed = data_marginal_is_maximally_mixed(forward_density)
    nonproduct = not joint_is_product_with_mixed_data(forward_density)
    for name in reversed(word):
        port.reverse(name)
    port.release()
    return {
        "word": list(word), "generation": port.last_generation,
        "normalized_trace": canonical_json(value),
        "joint_not_product_after_forward": nonproduct,
        "data_marginal_maximally_mixed_after_forward": marginal_mixed,
        "hidden_density_field_cells": 64, "hidden_scratch_field_cells": 4,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_density_and_scratch_backings": ids == (id(port.density), id(port.scratch)),
        "canonical_after_restoration": port.canonical(), "baseline_reload_used": False,
        "work": work_for_length(len(word)), "run_kind": kind,
    }


PRIMARY = ("H0", "T0", "H0", "CNOT01")
REUSE = ("H1", "CNOT01", "T0", "H0", "T1", "CNOT10")


def oracle_controls() -> dict[str, bool]:
    unit = direct_matrix(PRIMARY)
    full = evolve(initial_density(), PRIMARY)
    expected_block = tuple(
        tuple(
            (O if row == column else Z).scale(Fraction(1, 8)) if row < 4 and column < 4
            else adjoint(unit)[row][column - 4].scale(Fraction(1, 8)) if row < 4 and column >= 4
            else unit[row - 4][column].scale(Fraction(1, 8)) if row >= 4 and column < 4
            else (O if row == column else Z).scale(Fraction(1, 8))
            for column in range(8)
        )
        for row in range(8)
    )
    dephased = evolve(dephased_density(), PRIMARY)
    basis = evolve(basis_density(), PRIMARY)
    reordered = ("T0", "H0", "H0", "CNOT01")
    omitted = PRIMARY[:-1]
    scalar_word_density = evolve(initial_density(), ("H0", "H0"))

    wrong_operation = controlled(gate("T0"))
    wrong_right = product(product(wrong_operation, initial_density()), wrong_operation)

    missing = full
    for name in reversed(PRIMARY[1:]):
        operation = controlled(adjoint(gate(name)))
        missing = product(product(operation, missing), adjoint(operation))
    wrong = full
    names = [PRIMARY[-2], *reversed(PRIMARY[:-1])]
    for name in names:
        operation = controlled(adjoint(gate(name)))
        wrong = product(product(operation, wrong), adjoint(operation))
    reordered_inverse = full
    names = list(reversed(PRIMARY))
    names[-2], names[-1] = names[-1], names[-2]
    for name in names:
        operation = controlled(adjoint(gate(name)))
        reordered_inverse = product(product(operation, reordered_inverse), adjoint(operation))

    premature = ReferencePort()
    premature.lease(PRIMARY, 251004, 1, descriptor_digest(PRIMARY))
    premature_rejected = False
    try:
        premature.project()
    except RuntimeError:
        premature_rejected = True
    wrong_owner = False
    try:
        ReferencePort().lease(PRIMARY, 251005, 1, descriptor_digest(PRIMARY))
    except RuntimeError:
        wrong_owner = True
    wrong_digest = False
    try:
        ReferencePort().lease(PRIMARY, 251004, 1, descriptor_digest(REUSE))
    except RuntimeError:
        wrong_digest = True
    stale_port = ReferencePort()
    run(stale_port, PRIMARY, 1, "CONTROL")
    stale = False
    try:
        stale_port.lease(PRIMARY, 251004, 1, descriptor_digest(PRIMARY))
    except RuntimeError:
        stale = True
    dirty = ReferencePort()
    dirty.lease(PRIMARY, 251004, 1, descriptor_digest(PRIMARY))
    dirty.scratch[0] = O
    dirty_rejected = False
    try:
        dirty.forward(PRIMARY[0])
    except RuntimeError:
        dirty_rejected = True

    return {
        "exact_eight_by_eight_block_identity_reconstructed": full == expected_block,
        "density_boundary_matches_direct_four_by_four_normalized_trace": boundary(full) == normalized_trace(unit),
        "dephased_clean_control_has_zero_xy_boundary": boundary(dephased) == Z,
        "basis_data_input_returns_diagonal_element_not_trace": boundary(basis) == unit[0][0] and boundary(basis) != boundary(full),
        "noncommuting_reorder_changes_direct_trace": normalized_trace(direct_matrix(reordered)) != boundary(full),
        "omitted_gate_changes_direct_trace": normalized_trace(direct_matrix(omitted)) != boundary(full),
        "valid_scalar_public_word_retains_factorized_joint_state_and_trace_boundary": (
            joint_is_product_with_mixed_data(scalar_word_density)
            and boundary(scalar_word_density) == O
        ),
        "wrong_right_action_breaks_hermiticity": wrong_right != adjoint(wrong_right),
        "missing_inverse_fails_reference_density_restoration": missing != initial_density(),
        "wrong_inverse_completed_path_fails_reference_density_restoration": wrong != initial_density(),
        "reordered_inverse_fails_reference_density_restoration": reordered_inverse != initial_density(),
        "reference_premature_projection_rejected": premature_rejected,
        "reference_wrong_owner_rejected": wrong_owner,
        "reference_same_id_changed_word_rejected": wrong_digest,
        "reference_stale_generation_rejected": stale,
        "reference_dirty_scratch_rejected": dirty_rejected,
    }


def main() -> None:
    config = json.load(sys.stdin)
    if config != {"suite": "M251_DQC1_OPERATOR_COHERENCE_STRICT_SCOPE"}:
        raise RuntimeError("invalid M251 reference configuration")
    shared = ReferencePort()
    fresh = ReferencePort()
    cases = [
        run(shared, PRIMARY, 1, "PRIMARY"),
        run(shared, REUSE, 2, "REUSE"),
        run(fresh, REUSE, 1, "FRESH"),
    ]
    controls = oracle_controls()
    if not all(controls.values()):
        raise RuntimeError(f"M251 independent control failure: {controls}")
    output = {
        "result": "PASS_SEPARATE_REFERENCE_DQC1_OPERATOR_COHERENCE_STRICT_SCOPE",
        "cases": cases, "controls": controls,
        "independent_oracle": {
            "qzeta8_polynomial_quotient_arithmetic_reconstructed": True,
            "dense_eight_by_eight_density_oracle_executed": True,
            "direct_four_by_four_trace_oracle_executed": True,
            "block_identity_reconstructed": True,
            "computational_path_enumeration_used": False,
            "production_source_imported": False,
        },
        "oracle_resource_law": {
            "dense_eight_by_eight_oracle_is_verifier_only": True,
            "strongest_fixed_fixture_classical_baseline": "PUBLIC_WORD_VALIDATION_PLUS_FROZEN_EXACT_NORMALIZED_TRACE_IN_O1_WORK",
            "strongest_transferable_descriptor_level_classical_baseline": "DIRECT_EXACT_FOUR_BY_FOUR_QZETA8_PUBLIC_WORD_MATRIX_RECURRENCE_PLUS_TRACE_WITH16_RESIDENT_MATRIX_FIELD_CELLS_PLUS_DECLARED_TRANSIENT_MULTIPLICATION_SCRATCH_AND_NO_CATVM_INVERSE",
            "direct_transferable_baseline_resident_matrix_field_cells": 16,
            "direct_transferable_baseline_transient_peak_complete": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
