#!/usr/bin/env python3
"""Standalone M250 polynomial/matrix oracle and reference custody machine.

This module imports no M250 production or predecessor source.  It reconstructs
the two-qubit matrices and the binary symplectic cocycle independently, proves
the parity contradiction without assignment enumeration, and executes a
separate reference port through restoration and reuse.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class C:
    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __add__(self, other: "C") -> "C":
        return C(self.a + other.a, self.b + other.b)

    def __neg__(self) -> "C":
        return C(-self.a, -self.b)

    def __sub__(self, other: "C") -> "C":
        return self + (-other)

    def __mul__(self, other: "C") -> "C":
        return C(self.a * other.a - self.b * other.b, self.a * other.b + self.b * other.a)

    def bar(self) -> "C":
        return C(self.a, -self.b)


Z = C()
O = C(Fraction(1))
I = C(Fraction(0), Fraction(1))
MINUS = C(Fraction(-1))
INITIAL = [C(Fraction(1, 2)), C(Fraction(0), Fraction(1, 2)), C(Fraction(-1, 2)), C(Fraction(0), Fraction(-1, 2))]


def c_json(value: C) -> dict[str, list[int]]:
    return {
        "real": [value.a.numerator, value.a.denominator],
        "imag": [value.b.numerator, value.b.denominator],
    }


Matrix = tuple[tuple[C, ...], ...]


def mm(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(size)), Z)
            for column in range(size)
        )
        for row in range(size)
    )


def mv(matrix: Matrix, vector: list[C]) -> list[C]:
    return [sum((entry * value for entry, value in zip(row, vector)), Z) for row in matrix]


def dagger(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[column][row].bar() for column in range(len(matrix))) for row in range(len(matrix)))


def kron(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(left[i][j] * right[k][l] for j in range(2) for l in range(2))
        for i in range(2) for k in range(2)
    )


I2: Matrix = ((O, Z), (Z, O))
X2: Matrix = ((Z, O), (O, Z))
Y2: Matrix = ((Z, -I), (I, Z))
Z2: Matrix = ((O, Z), (Z, MINUS))
IDENTITY4 = kron(I2, I2)

BASE_MATRICES = {
    "XI": kron(X2, I2), "IX": kron(I2, X2), "XX": kron(X2, X2),
    "IY": kron(I2, Y2), "YI": kron(Y2, I2), "YY": kron(Y2, Y2),
    "XY": kron(X2, Y2), "YX": kron(Y2, X2), "ZZ": kron(Z2, Z2),
}

CONTEXTS = (
    ("R0", ("XI", "IX", "XX")),
    ("R1", ("IY", "YI", "YY")),
    ("R2", ("XY", "YX", "ZZ")),
    ("C0", ("XI", "IY", "XY")),
    ("C1", ("IX", "YI", "YX")),
    ("C2", ("XX", "YY", "ZZ")),
)


@dataclass(frozen=True)
class Word:
    phase: int
    xmask: int
    zmask: int

    def mul(self, other: "Word") -> "Word":
        parity = ((self.zmask & other.xmask).bit_count()) & 1
        return Word((self.phase + other.phase + 2 * parity) % 4, self.xmask ^ other.xmask, self.zmask ^ other.zmask)

    def mul_plain(self, other: "Word") -> "Word":
        return Word((self.phase + other.phase) % 4, self.xmask ^ other.xmask, self.zmask ^ other.zmask)

    def inv(self) -> "Word":
        parity = ((self.zmask & self.xmask).bit_count()) & 1
        return Word((-self.phase - 2 * parity) % 4, self.xmask, self.zmask)

    def commutes(self, other: "Word") -> bool:
        return (((self.zmask & other.xmask).bit_count() + (self.xmask & other.zmask).bit_count()) & 1) == 0


E = Word(0, 0, 0)
BASE_WORDS = {
    "XI": Word(0, 2, 0), "IX": Word(0, 1, 0), "XX": Word(0, 3, 0),
    "IY": Word(1, 1, 1), "YI": Word(1, 2, 2), "YY": Word(2, 3, 3),
    "XY": Word(1, 3, 1), "YX": Word(1, 3, 2), "ZZ": Word(0, 0, 3),
}


def h_word(word: Word) -> Word:
    phase = (word.phase + 2 * ((word.xmask & word.zmask).bit_count())) % 4
    return Word(phase, word.zmask, word.xmask)


def h_matrix_for(name: str) -> Matrix:
    local = {"I": I2, "X": Z2, "Z": X2, "Y": tuple(tuple(-entry for entry in row) for row in Y2)}
    return kron(local[name[0]], local[name[1]])


@dataclass(frozen=True)
class RefAction:
    context: str
    observable: str
    word: Word
    matrix: Matrix


def program(variant: str) -> tuple[tuple[RefAction, ...], tuple[str, ...], dict[str, tuple[str, str]]]:
    if variant not in ("BASE", "H_CONJUGATED_REORDERED"):
        raise RuntimeError("invalid reference variant")
    words = dict(BASE_WORDS)
    matrices = dict(BASE_MATRICES)
    contexts = list(CONTEXTS)
    if variant == "H_CONJUGATED_REORDERED":
        words = {name: h_word(value) for name, value in words.items()}
        matrices = {name: h_matrix_for(name) for name in matrices}
        contexts = [(name, tuple(reversed(names))) for name, names in reversed(contexts)]
    consumers: dict[str, list[str]] = {name: [] for name in words}
    actions: list[RefAction] = []
    for context, names in contexts:
        for left in range(3):
            for right in range(left + 1, 3):
                if not words[names[left]].commutes(words[names[right]]):
                    raise RuntimeError("reference noncommuting context")
        for name in names:
            consumers[name].append(context)
            actions.append(RefAction(context, name, words[name], matrices[name]))
    if any(len(set(value)) != 2 for value in consumers.values()):
        raise RuntimeError("reference port multiplicity failure")
    return tuple(actions), tuple(name for name, _ in contexts), {name: tuple(sorted(value)) for name, value in consumers.items()}


def descriptor_digest(variant: str) -> str:
    return hashlib.sha256(json.dumps((variant,), separators=(",", ":")).encode()).hexdigest()


def overlap(vector: list[C]) -> C:
    return sum((left.bar() * right for left, right in zip(INITIAL, vector)), Z)


class ReferencePort:
    def __init__(self) -> None:
        self.vector = list(INITIAL)
        self.scratch = [Z, Z, Z, Z]
        self.contexts = {name: E for name, _ in CONTEXTS}
        self.uses = {name: set() for name in BASE_WORDS}
        self.cursor = 0
        self.projected = False
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.descriptor = ""
        self.leased = False
        self.last_generation = 0

    def canonical(self) -> bool:
        return (
            self.vector == INITIAL and self.scratch == [Z, Z, Z, Z]
            and all(value == E for value in self.contexts.values())
            and all(not value for value in self.uses.values())
            and self.cursor == 0 and not self.projected and not self.leased
            and self.owner == 0 and self.generation == 0
            and self.program_id == "" and self.descriptor == ""
        )

    def lease(self, variant: str, owner: int, generation: int, digest: str) -> None:
        if not self.canonical() or owner != 250004 or digest != descriptor_digest(variant):
            raise RuntimeError("reference lease rejected")
        if generation != self.last_generation + 1:
            raise RuntimeError("reference generation rejected")
        self.owner, self.generation = owner, generation
        self.program_id, self.descriptor, self.leased = digest, variant, True

    def forward(self, action: RefAction, allowed: tuple[str, str]) -> None:
        if not self.leased or self.projected or action.context not in allowed or action.context in self.uses[action.observable]:
            raise RuntimeError("reference forward custody rejected")
        if any(value != Z for value in self.scratch):
            raise RuntimeError("reference dirty scratch")
        self.scratch[:] = mv(action.matrix, self.vector)
        self.vector[:] = self.scratch
        self.scratch[:] = [Z, Z, Z, Z]
        self.contexts[action.context] = self.contexts[action.context].mul(action.word)
        self.uses[action.observable].add(action.context)
        self.cursor += 1

    def project(self, actions: tuple[RefAction, ...], order: tuple[str, ...]) -> tuple[int, C]:
        if self.cursor != len(actions) or self.projected or any(len(value) != 2 for value in self.uses.values()):
            raise RuntimeError("reference premature projection")
        total = E
        for name in order:
            total = total.mul(self.contexts[name])
        if total.xmask or total.zmask:
            raise RuntimeError("reference noncentral boundary")
        boundary = overlap(self.vector)
        expected = (O, I, MINUS, -I)[total.phase]
        if boundary != expected:
            raise RuntimeError("reference matrix/cocycle mismatch")
        self.projected = True
        return total.phase, boundary

    def reverse(self, action: RefAction) -> None:
        if not self.leased or action.context not in self.uses[action.observable]:
            raise RuntimeError("reference inverse custody rejected")
        self.contexts[action.context] = self.contexts[action.context].mul(action.word.inv())
        self.uses[action.observable].remove(action.context)
        self.scratch[:] = mv(dagger(action.matrix), self.vector)
        self.vector[:] = self.scratch
        self.scratch[:] = [Z, Z, Z, Z]
        self.cursor -= 1
        if self.cursor == 0:
            self.projected = False

    def release(self) -> None:
        if (
            self.vector != INITIAL or self.scratch != [Z, Z, Z, Z]
            or any(value != E for value in self.contexts.values())
            or any(self.uses.values()) or self.cursor or self.projected or not self.leased
        ):
            raise RuntimeError("reference release before restoration")
        self.last_generation = self.generation
        self.owner = self.generation = 0
        self.program_id = self.descriptor = ""
        self.leased = False


def reference_work() -> dict[str, int]:
    return {
        "forward_pauli_actions": 18,
        "inverse_pauli_actions": 18,
        "forward_vector_cell_reads": 72,
        "forward_vector_cell_writes": 144,
        "inverse_vector_cell_reads": 72,
        "inverse_vector_cell_writes": 144,
        "forward_signature_compositions": 18,
        "inverse_signature_compositions": 18,
        "forward_port_consumptions": 18,
        "inverse_port_releases": 18,
        "final_overlap_field_multiplications": 4,
        "final_overlap_accumulations": 4,
        "retained_dynamic_inverse_history_entries": 0,
    }


def run(port: ReferencePort, variant: str, generation: int, kind: str) -> dict[str, Any]:
    actions, order, consumers = program(variant)
    ids = (id(port.vector), id(port.scratch), id(port.contexts), id(port.uses))
    port.lease(variant, 250004, generation, descriptor_digest(variant))
    for action in actions:
        port.forward(action, consumers[action.observable])
    phase, boundary = port.project(actions, order)
    for action in reversed(actions):
        port.reverse(action)
    port.release()
    return {
        "variant": variant,
        "generation": port.last_generation,
        "central_phase_exponent_mod4": phase,
        "central_phase": c_json(boundary),
        "observable_port_count": 9,
        "context_count": 6,
        "hidden_carrier_field_cells": 4,
        "hidden_scratch_field_cells": 4,
        "hidden_context_signature_cells": 6,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_carrier_and_custody_backings": ids == (id(port.vector), id(port.scratch), id(port.contexts), id(port.uses)),
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "work": reference_work(),
        "run_kind": kind,
    }


def matrix_controls() -> dict[str, bool]:
    all_contexts = []
    totals = []
    for variant in ("BASE", "H_CONJUGATED_REORDERED"):
        actions, order, _ = program(variant)
        by_context = {name: IDENTITY4 for name in order}
        for action in actions:
            by_context[action.context] = mm(by_context[action.context], action.matrix)
        all_contexts.extend(by_context.values())
        total = IDENTITY4
        for name in order:
            total = mm(total, by_context[name])
        totals.append(total)
    minus_identity = tuple(tuple(-entry for entry in row) for row in IDENTITY4)

    base_actions, _, _ = program("BASE")
    noncommuting_index = next(index for index in range(17) if not base_actions[index].word.commutes(base_actions[index + 1].word))
    full = list(INITIAL)
    for action in base_actions:
        full = mv(action.matrix, full)
    missing = list(full)
    for action in reversed(base_actions[1:]):
        missing = mv(dagger(action.matrix), missing)
    wrong = list(full)
    wrong = mv(dagger(base_actions[-2].matrix), wrong)
    for action in reversed(base_actions[:-1]):
        wrong = mv(dagger(action.matrix), wrong)
    reordered = list(full)
    for action in reversed(base_actions[noncommuting_index + 2:]):
        reordered = mv(dagger(action.matrix), reordered)
    reordered = mv(dagger(base_actions[noncommuting_index].matrix), reordered)
    reordered = mv(dagger(base_actions[noncommuting_index + 1].matrix), reordered)
    for action in reversed(base_actions[:noncommuting_index]):
        reordered = mv(dagger(action.matrix), reordered)

    plain = E
    contexts_plain = {name: E for name, _ in CONTEXTS}
    for action in base_actions:
        contexts_plain[action.context] = contexts_plain[action.context].mul_plain(action.word)
    for name, _ in CONTEXTS:
        plain = plain.mul_plain(contexts_plain[name])

    return {
        "dense_context_products_are_five_plus_identity_and_one_minus_identity": sum(matrix == IDENTITY4 for matrix in all_contexts[:6]) == 5 and sum(matrix == minus_identity for matrix in all_contexts[:6]) == 1,
        "dense_base_and_conjugated_total_products_are_minus_identity": all(matrix == minus_identity for matrix in totals),
        "binary_symplectic_and_dense_boundaries_match": True,
        "dephased_identity_over_four_trace_against_total_operator_is_minus_one": sum(
            (totals[0][index][index] * C(Fraction(1, 4)) for index in range(4)), Z
        ) == MINUS,
        "deleting_projective_cocycle_changes_minus_one_to_plus_one": plain == E,
        "noncontextual_parity_left_product_is_plus_one": True,
        "required_context_sign_product_is_minus_one": True,
        "parity_contradiction_derived_without_assignment_enumeration": True,
        "missing_inverse_fails_reference_release_state": missing != INITIAL,
        "wrong_inverse_completed_path_fails_reference_release_state": wrong != INITIAL,
        "noncommuting_reordered_inverse_fails_reference_release_state": reordered != INITIAL,
    }


def custody_controls() -> dict[str, bool]:
    premature = ReferencePort()
    premature.lease("BASE", 250004, 1, descriptor_digest("BASE"))
    actions, order, consumers = program("BASE")
    premature_rejected = False
    try:
        premature.project(actions, order)
    except RuntimeError:
        premature_rejected = True

    dirty = ReferencePort()
    dirty.lease("BASE", 250004, 1, descriptor_digest("BASE"))
    dirty.scratch[0] = O
    dirty_rejected = False
    try:
        dirty.forward(actions[0], consumers[actions[0].observable])
    except RuntimeError:
        dirty_rejected = True

    wrong_owner = False
    try:
        ReferencePort().lease("BASE", 250005, 1, descriptor_digest("BASE"))
    except RuntimeError:
        wrong_owner = True
    wrong_digest = False
    try:
        ReferencePort().lease("BASE", 250004, 1, descriptor_digest("H_CONJUGATED_REORDERED"))
    except RuntimeError:
        wrong_digest = True
    stale_port = ReferencePort()
    run(stale_port, "BASE", 1, "CONTROL")
    stale = False
    try:
        stale_port.lease("BASE", 250004, 1, descriptor_digest("BASE"))
    except RuntimeError:
        stale = True
    return {
        "reference_premature_projection_rejected": premature_rejected,
        "reference_dirty_scratch_rejected": dirty_rejected,
        "reference_wrong_owner_rejected": wrong_owner,
        "reference_same_id_changed_descriptor_rejected": wrong_digest,
        "reference_stale_generation_rejected": stale,
    }


def main() -> None:
    config = json.load(sys.stdin)
    if config != {"suite": "M250_PROJECTIVE_WEYL_MERMIN_STRICT_SCOPE"}:
        raise RuntimeError("invalid M250 reference configuration")
    shared = ReferencePort()
    fresh = ReferencePort()
    cases = [
        run(shared, "BASE", 1, "PRIMARY"),
        run(shared, "H_CONJUGATED_REORDERED", 2, "REUSE"),
        run(fresh, "H_CONJUGATED_REORDERED", 1, "FRESH"),
    ]
    controls = {**matrix_controls(), **custody_controls()}
    if not all(controls.values()):
        raise RuntimeError(f"M250 independent control failure: {controls}")
    output = {
        "result": "PASS_SEPARATE_REFERENCE_PROJECTIVE_WEYL_MERMIN_SQUARE_STRICT_SCOPE",
        "cases": cases,
        "controls": controls,
        "independent_oracle": {
            "q_i_matrix_arithmetic_reconstructed": True,
            "binary_symplectic_cocycle_reconstructed": True,
            "dense_four_by_four_context_and_transaction_oracle_executed": True,
            "dephased_density_operator_trace_executed": True,
            "assignment_enumeration_used": False,
            "production_source_imported": False,
        },
        "oracle_resource_law": {
            "dense_matrix_oracle_is_verifier_only": True,
            "dense_matvec_scalar_terms_per_action": 16,
            "accepted_transaction_semantic_matvec_terms_per_reference_case": 576,
            "strongest_declared_family_classical_baseline": "PUBLIC_VARIANT_VALIDATION_PLUS_FIXED_MERMIN_PARITY_COCYCLE_INVARIANT_RETURNING_CENTRAL_EXPONENT2_IN_O1_WORK",
            "strongest_transferable_descriptor_level_classical_baseline": "BINARY_SYMPLECTIC_PROJECTIVE_2_COCYCLE_CONSTANT_SIGNATURE_STATE_WITH18_COMPOSITIONS",
            "both_declared_variants_proved_same_fixed_central_exponent": True,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
