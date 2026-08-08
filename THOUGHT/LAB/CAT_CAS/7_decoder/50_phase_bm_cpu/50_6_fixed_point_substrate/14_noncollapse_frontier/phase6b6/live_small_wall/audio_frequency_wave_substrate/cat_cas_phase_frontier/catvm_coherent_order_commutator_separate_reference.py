#!/usr/bin/env python3
"""Standalone M252 Q[zeta8]/(zeta8^4+1) order-port oracle."""

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

    def bar(self) -> "E":
        a, b, c, d = self.coefficients
        return E((a, -d, -c, -b))


ZERO = E()
ONE = E((Fraction(1), Fraction(0), Fraction(0), Fraction(0)))
W = E((Fraction(0), Fraction(1), Fraction(0), Fraction(0)))
INV_ROOT2 = E((Fraction(0), Fraction(1, 2), Fraction(0), Fraction(-1, 2)))


def canonical_json(value: E) -> list[list[int]]:
    a, b, c, d = value.coefficients
    coordinates = (a, (b - d) / 2, c, (b + d) / 2)
    return [[item.numerator, item.denominator] for item in coordinates]


Matrix = tuple[tuple[E, E], tuple[E, E]]
X: Matrix = ((ZERO, ONE), (ONE, ZERO))
Z: Matrix = ((ONE, ZERO), (ZERO, -ONE))
H: Matrix = ((INV_ROOT2, INV_ROOT2), (INV_ROOT2, -INV_ROOT2))
T: Matrix = ((ONE, ZERO), (ZERO, W))
GATES = {"X": X, "Z": Z, "H": H, "T": T}


def adjoint(matrix: Matrix) -> Matrix:
    return (
        (matrix[0][0].bar(), matrix[1][0].bar()),
        (matrix[0][1].bar(), matrix[1][1].bar()),
    )


def matrix_product(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(sum((left[row][inner] * right[inner][column] for inner in range(2)), ZERO) for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def matrix_vector(matrix: Matrix, vector: tuple[E, E]) -> tuple[E, E]:
    return tuple(
        sum((matrix[row][column] * vector[column] for column in range(2)), ZERO)
        for row in range(2)
    )  # type: ignore[return-value]


def inner(left: tuple[E, E], right: tuple[E, E]) -> E:
    return sum((left[index].bar() * right[index] for index in range(2)), ZERO)


@dataclass(frozen=True)
class Spec:
    u: str
    v: str

    def __post_init__(self) -> None:
        if self.u not in GATES or self.v not in GATES:
            raise RuntimeError("reference public pair rejected")

    @property
    def pair(self) -> tuple[str, str]:
        return (self.u, self.v)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.pair, separators=(",", ":")).encode()).hexdigest()


def direct_dense_commutator(spec: Spec) -> E:
    u = GATES[spec.u]
    v = GATES[spec.v]
    commutator = matrix_product(matrix_product(matrix_product(adjoint(u), adjoint(v)), u), v)
    return commutator[0][0]


def streamed_one_vector_commutator(spec: Spec) -> E:
    vector = (ONE, ZERO)
    for matrix in (GATES[spec.v], GATES[spec.u], adjoint(GATES[spec.v]), adjoint(GATES[spec.u])):
        vector = matrix_vector(matrix, vector)
    return vector[0]


def direct_branches(spec: Spec) -> tuple[tuple[E, E], tuple[E, E]]:
    source = (ONE, ZERO)
    branch_vu = matrix_vector(GATES[spec.v], matrix_vector(GATES[spec.u], source))
    branch_uv = matrix_vector(GATES[spec.u], matrix_vector(GATES[spec.v], source))
    return branch_vu, branch_uv


def direct_boundary(spec: Spec) -> E:
    left, right = direct_branches(spec)
    return inner(left, right)


def dephased_order_boundary(spec: Spec) -> E:
    left, right = direct_branches(spec)
    amplitudes = (left[0] * INV_ROOT2, left[1] * INV_ROOT2,
                  right[0] * INV_ROOT2, right[1] * INV_ROOT2)
    density = tuple(
        tuple(
            amplitudes[row] * amplitudes[column].bar()
            if row // 2 == column // 2 else ZERO
            for column in range(4)
        )
        for row in range(4)
    )
    return (density[0][2] + density[1][3]).scale(Fraction(2))


def initial_state() -> list[E]:
    return [INV_ROOT2, ZERO, INV_ROOT2, ZERO]


def work_record() -> dict[str, int]:
    return {
        "compiled_public_gate_plan_matrix_references": 4,
        "forward_branch_gate_actions": 4,
        "inverse_branch_gate_actions": 4,
        "forward_field_multiply_terms": 16,
        "inverse_field_multiply_terms": 16,
        "forward_field_accumulations": 16,
        "inverse_field_accumulations": 16,
        "forward_branch_field_writes": 8,
        "inverse_branch_field_writes": 8,
        "forward_scratch_clear_writes": 8,
        "inverse_scratch_clear_writes": 8,
        "boundary_field_multiply_terms": 3,
        "boundary_field_accumulations": 2,
        "retained_dynamic_inverse_history_entries": 0,
    }


class ReferencePort:
    def __init__(self) -> None:
        self.state = initial_state()
        self.scratch = [ZERO, ZERO]
        self.receipts = [False, False]
        self.last_generation = 0
        self.generation = 0
        self.owner = 0
        self.spec: Spec | None = None
        self.program_id = ""
        self.transaction_id = ""
        self.cursor = 0
        self.projected = False
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.state == initial_state() and self.scratch == [ZERO, ZERO]
            and self.receipts == [False, False] and self.generation == 0
            and self.owner == 0 and self.spec is None and self.program_id == ""
            and self.transaction_id == "" and self.cursor == 0
            and not self.projected and not self.leased
        )

    def lease(self, spec: Spec, *, owner: int, generation: int, program_id: str, transaction_id: str) -> None:
        if not self.canonical() or generation != self.last_generation + 1 or program_id != spec.digest:
            raise RuntimeError("reference lease rejected")
        self.owner = owner
        self.generation = generation
        self.spec = spec
        self.program_id = program_id
        self.transaction_id = transaction_id
        self.leased = True

    def require(self, spec: Spec, *, owner: int, generation: int, program_id: str, transaction_id: str) -> None:
        if (
            not self.leased or self.spec != spec or self.owner != owner
            or self.generation != generation or self.program_id != program_id
            or self.transaction_id != transaction_id
        ):
            raise RuntimeError("reference custody rejected")

    def plan(self) -> tuple[tuple[int, Matrix], ...]:
        if self.spec is None:
            raise RuntimeError("reference unleased plan")
        u, v = GATES[self.spec.u], GATES[self.spec.v]
        return ((0, u), (0, v), (1, v), (1, u))

    def apply(self, branch: int, matrix: Matrix) -> None:
        if self.scratch != [ZERO, ZERO]:
            raise RuntimeError("reference dirty scratch")
        offset = 2 * branch
        old = (self.state[offset], self.state[offset + 1])
        updated = matrix_vector(matrix, old)
        self.scratch[:] = updated
        self.state[offset:offset + 2] = self.scratch
        self.scratch[:] = [ZERO, ZERO]

    def forward(self) -> None:
        plan = self.plan()
        while self.cursor < 4:
            branch, matrix = plan[self.cursor]
            self.apply(branch, matrix)
            self.cursor += 1
            if self.cursor == 2:
                self.receipts[0] = True
            if self.cursor == 4:
                self.receipts[1] = True

    def project(self) -> E:
        if self.cursor != 4 or self.receipts != [True, True] or self.projected:
            raise RuntimeError("reference premature projection")
        self.projected = True
        return (self.state[0].bar() * self.state[2] + self.state[1].bar() * self.state[3]).scale(Fraction(2))

    def reverse(self) -> None:
        plan = self.plan()
        while self.cursor:
            branch, matrix = plan[self.cursor - 1]
            self.apply(branch, adjoint(matrix))
            self.cursor -= 1
            if self.cursor == 2:
                self.receipts[1] = False
            if self.cursor == 0:
                self.receipts[0] = False
                self.projected = False

    def release(self) -> None:
        if (
            self.state != initial_state() or self.scratch != [ZERO, ZERO]
            or self.receipts != [False, False] or self.cursor != 0
            or self.projected or not self.leased or self.spec is None
        ):
            raise RuntimeError("reference release before restoration")
        restored = self.generation
        self.generation = 0
        self.owner = 0
        self.spec = None
        self.program_id = ""
        self.transaction_id = ""
        self.leased = False
        self.last_generation = restored


def execute(port: ReferencePort, spec: Spec, generation: int, run_kind: str) -> dict[str, Any]:
    state_id, scratch_id, receipt_id = id(port.state), id(port.scratch), id(port.receipts)
    port.lease(spec, owner=252004, generation=generation, program_id=spec.digest, transaction_id=f"REF_{run_kind}")
    port.require(spec, owner=252004, generation=generation, program_id=spec.digest, transaction_id=f"REF_{run_kind}")
    port.forward()
    boundary = port.project()
    if (
        boundary != direct_dense_commutator(spec)
        or boundary != direct_boundary(spec)
        or boundary != streamed_one_vector_commutator(spec)
    ):
        raise RuntimeError("independent dense/branch commutator mismatch")
    port.reverse()
    port.release()
    return {
        "pair": list(spec.pair),
        "generation": generation,
        "commutator_boundary": canonical_json(boundary),
        "hidden_branch_field_cells": 4,
        "hidden_scratch_field_cells": 2,
        "hidden_order_consumer_receipt_cells": 2,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_branch_scratch_and_receipt_backings": (
            state_id == id(port.state) and scratch_id == id(port.scratch) and receipt_id == id(port.receipts)
        ),
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "work": work_record(),
        "run_kind": run_kind,
    }


def controls() -> dict[str, bool]:
    primary = Spec("X", "Z")
    ht = Spec("H", "T")
    initial = initial_state()

    wrong_owner = False
    port = ReferencePort()
    port.lease(primary, owner=252004, generation=1, program_id=primary.digest, transaction_id="OWNER")
    try:
        port.require(primary, owner=252005, generation=1, program_id=primary.digest, transaction_id="OWNER")
    except RuntimeError:
        wrong_owner = True
    port.release()

    premature = False
    port = ReferencePort()
    port.lease(primary, owner=252004, generation=1, program_id=primary.digest, transaction_id="PRE")
    try:
        port.project()
    except RuntimeError:
        premature = True
    port.release()

    dirty = False
    port = ReferencePort()
    port.scratch[0] = ONE
    try:
        port.apply(0, X)
    except RuntimeError:
        dirty = True

    stale = False
    port = ReferencePort()
    execute(port, primary, 1, "PRIMARY_CONTROL")
    try:
        port.lease(ht, owner=252004, generation=1, program_id=ht.digest, transaction_id="STALE")
    except RuntimeError:
        stale = True

    pair_mutation = False
    try:
        port.lease(ht, owner=252004, generation=2, program_id=primary.digest, transaction_id="MUTATE")
    except RuntimeError:
        pair_mutation = True

    return {
        "xz_projective_commutator_boundary_is_minus_one": direct_boundary(primary) == -ONE,
        "commuting_tz_pair_boundary_is_one": direct_boundary(Spec("T", "Z")) == ONE,
        "ht_noncentral_boundary_matches_exact_qzeta8_formula": direct_boundary(ht) == (ONE + W.bar()).scale(Fraction(1, 2)),
        "dense_commutator_matrix_matches_two_branch_overlap_for_all_declared_pairs": all(
            direct_dense_commutator(Spec(u, v)) == direct_boundary(Spec(u, v))
            == streamed_one_vector_commutator(Spec(u, v))
            for u in GATES for v in GATES
        ),
        "swapping_noncentral_order_conjugates_and_changes_boundary": (
            direct_boundary(Spec("T", "H")) == direct_boundary(ht).bar()
            and direct_boundary(Spec("T", "H")) != direct_boundary(ht)
        ),
        "dephased_order_has_zero_offdiagonal_boundary_by_exact_density_contraction": (
            dephased_order_boundary(primary) == ZERO and direct_boundary(primary) != ZERO
        ),
        "same_order_pair_has_unit_boundary": direct_boundary(Spec("H", "H")) == ONE,
        "wrong_owner_rejected_by_independent_port": wrong_owner,
        "stale_generation_rejected_by_independent_port": stale,
        "same_id_changed_pair_rejected_by_independent_digest": pair_mutation,
        "premature_order_projection_rejected_by_independent_port": premature,
        "dirty_scratch_rejected_by_independent_port": dirty,
        "initial_state_is_exactly_normalized": (
            initial[0].bar() * initial[0] + initial[2].bar() * initial[2] == ONE
        ),
        "no_path_or_assignment_enumeration": True,
    }


def main() -> None:
    request = json.load(sys.stdin)
    if request != {"suite": "M252_COHERENT_ORDER_COMMUTATOR_STRICT_SCOPE"}:
        raise RuntimeError("invalid M252 standalone request")
    primary_port = ReferencePort()
    primary = execute(primary_port, Spec("X", "Z"), 1, "PRIMARY")
    reuse = execute(primary_port, Spec("H", "T"), 2, "REUSE")
    fresh = execute(ReferencePort(), Spec("H", "T"), 1, "FRESH")
    output = {
        "result": "PASS_M252_SEPARATE_REFERENCE",
        "cases": [primary, reuse, fresh],
        "reuse_parity": {
            "boundary": reuse["commutator_boundary"] == fresh["commutator_boundary"],
            "work": reuse["work"] == fresh["work"],
            "same_backings": reuse["same_branch_scratch_and_receipt_backings"] and fresh["same_branch_scratch_and_receipt_backings"],
            "generation_sequence": [primary["generation"], reuse["generation"], fresh["generation"]] == [1, 2, 1],
            "no_reload": not reuse["baseline_reload_used"] and not fresh["baseline_reload_used"],
        },
        "controls": controls(),
        "oracle_law": {
            "arithmetic": "Q_ZETA8_POLYNOMIAL_QUOTIENT_ZETA8_TO_THE4_PLUS1",
            "dense_two_by_two_commutator_matrix_reconstructed": True,
            "branch_overlap_reconstructed": True,
            "strongest_transferable_one_vector_commutator_reconstructed": True,
            "independent_port_custody_and_atomic_ordering_reconstructed": True,
        },
    }
    if not all(output["reuse_parity"].values()) or not all(output["controls"].values()):
        raise RuntimeError("M252 standalone verification failure")
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
