#!/usr/bin/env python3
"""Exact growing-prime cubic deformation of the Weil relation chart.

The production path extends the preceding homogeneous Weil-Gaussian relation
with a packed superposition of Weyl-displaced Gaussian components.  A second
resident port remains Gaussian and conjugates the unresolved deformed port.
Public separable cubic phases act by native left/right component convolution.
No ordinary relation table is retained by the accepted path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import growing_prime_two_fiber_weil_gaussian_phase_kernel_closure as gaussian


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
BASE_DEPTH = 2
DEEPEST_ORDER = 113
DEEPEST_DEPTH = 4
ALTERNATE_ORDER = 41
ALTERNATE_DEPTH = 3
CHECKPOINTS = (1, 2, 3, 4)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_PHASE_"
    "KERNEL_GAUSSIAN_COMPONENT_CHART_HAS_AT_LEAST_Q_MINUS3_SQUARED_DEFORMED_"
    "PORT_COMPONENTS_AFTER_ONE_CUBIC_INTERSECTION_ON_EVERY_DECLARED_ORDER_"
    "WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_AN_EXECUTED_Q_"
    "VECTOR_PUBLIC_WORD_CLASSICAL_RECURRENCE_USES_LINEAR_STATE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class Work:
    streamed_cocycle_calls: int = 0
    streamed_phase_sum_terms: int = 0
    closed_cocycle_calls: int = 0
    component_pair_products: int = 0
    weyl_phase_evaluations: int = 0
    cubic_spectrum_calls: int = 0
    cubic_spectrum_phase_terms: int = 0
    composition_calls: int = 0
    cubic_intersection_calls: int = 0
    gaussian_conjugation_calls: int = 0
    inverse_gaussian_rematerializations: int = 0


@dataclass
class Relation:
    field: gaussian.Field
    symplectic: list[int]
    fiber: list[int]
    coefficients: dict[tuple[int, int], int]

    def canonical(self) -> tuple[Any, ...]:
        return (
            tuple(value % self.field.q for value in self.symplectic),
            tuple(value % self.field.p for value in self.fiber),
            tuple(sorted((r % self.field.q, s % self.field.q, value % self.field.p) for (r, s), value in self.coefficients.items() if value % self.field.p)),
        )

    def backing_ids(self) -> tuple[int, ...]:
        return id(self), id(self.symplectic), id(self.fiber), id(self.coefficients)


def clean(values: dict[tuple[int, int], int], p: int) -> dict[tuple[int, int], int]:
    return {key: value % p for key, value in values.items() if value % p}


def set_relation(target: Relation, source: Relation) -> None:
    target.symplectic[:] = source.symplectic
    target.fiber[:] = source.fiber
    target.coefficients.clear()
    target.coefficients.update(source.coefficients)


def covariance_vector(matrix: list[int], vector: tuple[int, int], q: int) -> tuple[int, int]:
    a, b, c, d = matrix
    r, s = vector
    # D_(r,s) uses f(x+r), so its stored r is the negative of the
    # conventional position displacement.  Metaplectic covariance is the
    # conjugated action diag(-1,1) S diag(-1,1).
    return (a * r - b * s) % q, (-c * r + d * s) % q


def displacement_phase(field: gaussian.Field, left: tuple[int, int], right: tuple[int, int]) -> int:
    inverse_two = pow(2, -1, field.q)
    exponent = (left[0] * right[1] - left[1] * right[0]) * inverse_two
    return gaussian.phase(field, exponent)


def cocycle(left: list[int], right: list[int], field: gaussian.Field, method: str, work: Work) -> int:
    if method == "STREAMED_PHASE_SUM":
        work.streamed_cocycle_calls += 1
        work.streamed_phase_sum_terms += field.q
        return sum(
            gaussian.kernel_value(left, 0, shared, field) * gaussian.kernel_value(right, shared, 0, field)
            for shared in range(field.q)
        ) % field.p
    if method == "CLOSED_CLASSICAL_GAUSS":
        work.closed_cocycle_calls += 1
        return gaussian.cocycle_closed(left, right, field, gaussian.Work())
    fail("unknown cocycle method")


def compose(left: Relation, right: Relation, method: str, work: Work) -> Relation:
    if left.field != right.field:
        fail("field mismatch")
    field = left.field
    q, p = field.q, field.p
    scalar = cocycle(left.symplectic, right.symplectic, field, method, work)
    output: dict[tuple[int, int], int] = {}
    for left_key, left_value in left.coefficients.items():
        for right_key, right_value in right.coefficients.items():
            transported = covariance_vector(left.symplectic, right_key, q)
            key = ((left_key[0] + transported[0]) % q, (left_key[1] + transported[1]) % q)
            phase_value = displacement_phase(field, left_key, transported)
            output[key] = (output.get(key, 0) + scalar * left_value * right_value * phase_value) % p
            work.component_pair_products += 1
            work.weyl_phase_evaluations += 1
    work.composition_calls += 1
    return Relation(
        field,
        gaussian.symplectic_multiply(left.symplectic, right.symplectic, q),
        gaussian.fiber_multiply(left.fiber, right.fiber, p),
        clean(output, p),
    )


def gaussian_inverse(source: Relation, method: str, work: Work) -> Relation:
    if set(source.coefficients) != {(0, 0)}:
        fail("only the resident Gaussian port has a compact inverse")
    field = source.field
    inverse_symplectic = gaussian.symplectic_inverse(source.symplectic, field.q)
    scalar = cocycle(source.symplectic, inverse_symplectic, field, method, work)
    coefficient = source.coefficients[(0, 0)] * scalar % field.p
    work.inverse_gaussian_rematerializations += 1
    return Relation(
        field,
        inverse_symplectic,
        gaussian.fiber_inverse(source.fiber, field.p),
        {(0, 0): pow(coefficient, -1, field.p)},
    )


def cubic_spectrum(field: gaussian.Field, strength: int, work: Work) -> dict[tuple[int, int], int]:
    q, p = field.q, field.p
    inverse_q = pow(q, -1, p)
    work.cubic_spectrum_calls += 1
    work.cubic_spectrum_phase_terms += q * q
    values = {}
    for mode in range(q):
        coefficient = inverse_q * sum(
            gaussian.phase(field, strength * x * x * x - mode * x) for x in range(q)
        ) % p
        if coefficient:
            values[(0, mode)] = coefficient
    return values


def cubic_operator(field: gaussian.Field, strength: int, work: Work) -> Relation:
    return Relation(field, [1, 0, 0, 1], [1, 0, 0, 1], cubic_spectrum(field, strength, work))


def cubic_intersection(source: Relation, row: int, column: int, method: str, work: Work) -> Relation:
    left = cubic_operator(source.field, row, work)
    right = cubic_operator(source.field, column, work)
    work.cubic_intersection_calls += 1
    return compose(compose(left, source, method, work), right, method, work)


def chirp_gaussian(source: Relation, row: int, column: int) -> Relation:
    if set(source.coefficients) != {(0, 0)}:
        fail("chirp update is reserved for the Gaussian port")
    q = source.field.q
    left = [1, 0, 2 * row % q, 1]
    right = [1, 0, 2 * column % q, 1]
    return Relation(
        source.field,
        gaussian.symplectic_multiply(gaussian.symplectic_multiply(left, source.symplectic, q), right, q),
        source.fiber[:],
        dict(source.coefficients),
    )


def conjugate_deformed(deformed: Relation, resident_gaussian: Relation, inverse: bool, method: str, work: Work) -> Relation:
    gaussian_inv = gaussian_inverse(resident_gaussian, method, work)
    work.gaussian_conjugation_calls += 1
    if inverse:
        return compose(compose(gaussian_inv, deformed, method, work), resident_gaussian, method, work)
    return compose(compose(resident_gaussian, deformed, method, work), gaussian_inv, method, work)


def displacement_kernel(relation: Relation, key: tuple[int, int], x: int, y: int) -> int:
    r, s = key
    field = relation.field
    phase_value = gaussian.phase(field, r * s * pow(2, -1, field.q) + s * x)
    return phase_value * gaussian.kernel_value(relation.symplectic, (x + r) % field.q, y, field) % field.p


def kernel_entry(relation: Relation, x: int, y: int) -> int:
    return sum(value * displacement_kernel(relation, key, x, y) for key, value in relation.coefficients.items()) % relation.field.p


def relation_value(relation: Relation, source: int, target: int, x: int, y: int) -> int:
    return relation.fiber[2 * source + target] * kernel_entry(relation, x, y) % relation.field.p


@dataclass(frozen=True)
class Stage:
    operation: str
    index: int
    family: str
    row: int = 0
    column: int = 0


def descriptors(index: int, family: str, q: int) -> tuple[Stage, ...]:
    if family not in FAMILIES:
        fail("invalid family")
    code = 1 if family == "PRIMARY" else 2
    stages = (
        Stage("GAUSSIAN_CHIRP", index, family, (2 * index + code + 1) % q, (3 * index + code + 2) % q),
        Stage("CONJUGATE", index, family),
        Stage("CUBIC_INTERSECTION", index, family, (5 * index + 2 * code + 1) % q, (7 * index + 3 * code + 2) % q),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


@dataclass
class Carrier:
    field: gaussian.Field
    seed_family: str
    deformed: Relation
    resident_gaussian: Relation
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, q: int, family: str = "PRIMARY") -> "Carrier":
        if family not in FAMILIES:
            fail("invalid family")
        field = gaussian.make_field(q)
        code = 1 if family == "PRIMARY" else 2
        deformed = Relation(field, [1, 1, 1, 2], [1, 2 + code, 3, 7 + 3 * code], {(0, 0): 1})
        resident = Relation(field, [2, 1, 1, 1], [2, 1 + code, 5, 3 + 3 * code], {(0, 0): 1})
        gaussian.fiber_inverse(deformed.fiber, field.p)
        gaussian.fiber_inverse(resident.fiber, field.p)
        return cls(field, family, deformed, resident)

    def canonical_state(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.seed_family, self.deformed.canonical(), self.resident_gaussian.canonical(), self.stage

    def backing_ids(self) -> tuple[int, ...]:
        return self.deformed.backing_ids() + self.resident_gaussian.backing_ids()


def apply_stage(carrier: Carrier, stage: Stage, method: str, work: Work) -> None:
    if stage.operation == "GAUSSIAN_CHIRP":
        updated = chirp_gaussian(carrier.resident_gaussian, stage.row, stage.column)
        set_relation(carrier.resident_gaussian, updated)
    elif stage.operation == "CONJUGATE":
        updated = conjugate_deformed(carrier.deformed, carrier.resident_gaussian, False, method, work)
        set_relation(carrier.deformed, updated)
    elif stage.operation == "CUBIC_INTERSECTION":
        updated = cubic_intersection(carrier.deformed, stage.row, stage.column, method, work)
        set_relation(carrier.deformed, updated)
    else:
        fail("unknown stage")


def undo_stage(carrier: Carrier, stage: Stage, method: str, work: Work, wrong: bool = False) -> None:
    if stage.operation == "GAUSSIAN_CHIRP":
        updated = chirp_gaussian(carrier.resident_gaussian, -stage.row + int(wrong), -stage.column)
        set_relation(carrier.resident_gaussian, updated)
    elif stage.operation == "CONJUGATE":
        updated = conjugate_deformed(carrier.deformed, carrier.resident_gaussian, True, method, work)
        if wrong:
            first = next(iter(updated.coefficients))
            updated.coefficients[first] = (updated.coefficients[first] + 1) % carrier.field.p
        set_relation(carrier.deformed, updated)
    elif stage.operation == "CUBIC_INTERSECTION":
        updated = cubic_intersection(carrier.deformed, -stage.row + int(wrong), -stage.column, method, work)
        set_relation(carrier.deformed, updated)
    else:
        fail("unknown inverse stage")


def checkpoint(carrier: Carrier, depth: int) -> dict[str, int]:
    q, p = carrier.field.q, carrier.field.p
    support_a = len(carrier.deformed.coefficients)
    support_b = len(carrier.resident_gaussian.coefficients)
    return {
        "depth": depth,
        "deformed_port_gaussian_component_support": support_a,
        "resident_gaussian_port_component_support": support_b,
        "deformed_port_component_capacity": q * q,
        "packed_component_coordinate_cells": 3 * (support_a + support_b),
        "resident_chart_and_fiber_cells": 16,
        "resident_packed_field_cells": 3 * (support_a + support_b) + 16,
        "logical_resident_payload_bits": (support_a + support_b) * (2 * (q - 1).bit_length() + (p - 1).bit_length()) + 8 * (q - 1).bit_length() + 8 * (p - 1).bit_length(),
        "ordinary_dense_two_port_relation_cells": 8 * q * q,
    }


def forward(carrier: Carrier, depth: int, family: str, method: str, reorder_modules: bool = False) -> tuple[list[dict[str, int]], Work, list[Stage]]:
    if carrier.stage != "IDLE" or depth < 1 or family not in FAMILIES:
        fail("invalid forward request")
    work = Work()
    records = []
    executed = []
    for index in range(depth):
        stages = descriptors(index, family, carrier.field.q)
        if reorder_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply_stage(carrier, stage, method, work)
            executed.append(stage)
        if index + 1 in CHECKPOINTS:
            records.append(checkpoint(carrier, index + 1))
    carrier.stage = "FORWARD_COMPLETE"
    return records, work, executed


def reverse(carrier: Carrier, executed: list[Stage], method: str, work: Work, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier lacks forward state")
    sequence = list(reversed(executed))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, stage in enumerate(sequence):
        undo_stage(carrier, stage, method, work, mutation == "WRONG" and position == 0)
    carrier.stage = "IDLE"


def probes(q: int, family: str) -> tuple[tuple[int, int, int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        (0, 1, (3 * code + 1) % q, (5 * code + 2) % q, 1),
        (1, 0, (7 * code + 2) % q, (11 * code + 3) % q, 2),
        (0, 0, (13 * code + 1) % q, (17 * code + 4) % q, 3),
        (1, 1, (19 * code + 2) % q, (23 * code + 5) % q, 5),
    )


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    return sum(
        weight * relation_value(carrier.deformed, source, target, x, y)
        for source, target, x, y, weight in probes(carrier.field.q, family)
    ) % carrier.field.p


def semantic_digest(carrier: Carrier) -> str:
    digest = hashlib.sha256()
    digest.update(repr(carrier.deformed.canonical()).encode("ascii"))
    digest.update(repr(carrier.resident_gaussian.canonical()).encode("ascii"))
    return digest.hexdigest()


def apply_gaussian_vector(vector: list[int], symplectic: tuple[int, ...], coefficient: int, field: gaussian.Field, work: dict[str, int]) -> list[int]:
    q, p = field.q, field.p
    work["gaussian_matrix_vector_terms"] += q * q
    return [
        coefficient * sum(gaussian.kernel_value(list(symplectic), x, y, field) * vector[y] for y in range(q)) % p
        for x in range(q)
    ]


def apply_cubic_vector(vector: list[int], strength: int, field: gaussian.Field, work: dict[str, int]) -> list[int]:
    work["cubic_vector_phase_multiplications"] += field.q
    return [value * gaussian.phase(field, strength * x * x * x) % field.p for x, value in enumerate(vector)]


def public_word_plan(q: int, depth: int, family: str) -> tuple[Carrier, list[tuple[str, Any]], list[tuple[str, Any]], list[int]]:
    seed = Carrier.seal(q)
    resident = Relation(seed.field, seed.resident_gaussian.symplectic[:], seed.resident_gaussian.fiber[:], dict(seed.resident_gaussian.coefficients))
    left_ops: list[tuple[str, Any]] = []
    right_ops: list[tuple[str, Any]] = []
    fiber = seed.deformed.fiber[:]
    scratch = Work()
    for index in range(depth):
        for stage in descriptors(index, family, q):
            if stage.operation == "GAUSSIAN_CHIRP":
                resident = chirp_gaussian(resident, stage.row, stage.column)
            elif stage.operation == "CONJUGATE":
                inverse = gaussian_inverse(resident, "CLOSED_CLASSICAL_GAUSS", scratch)
                left_ops.append(("GAUSSIAN", (tuple(resident.symplectic), resident.coefficients[(0, 0)])))
                right_ops.append(("GAUSSIAN", (tuple(inverse.symplectic), inverse.coefficients[(0, 0)])))
                fiber = gaussian.fiber_multiply(gaussian.fiber_multiply(resident.fiber, fiber, seed.field.p), inverse.fiber, seed.field.p)
            else:
                left_ops.append(("CUBIC", stage.row))
                right_ops.append(("CUBIC", stage.column))
    return seed, left_ops, right_ops, fiber


def baseline_boundary(q: int, depth: int, family: str) -> dict[str, Any]:
    seed, left_ops, right_ops, fiber = public_word_plan(q, depth, family)
    field = seed.field
    work = {"gaussian_matrix_vector_terms": 0, "cubic_vector_phase_multiplications": 0}
    total = 0
    for source, target, x, y, weight in probes(q, family):
        vector = [0] * q
        vector[y] = 1
        for operation, payload in reversed(right_ops):
            vector = apply_gaussian_vector(vector, payload[0], payload[1], field, work) if operation == "GAUSSIAN" else apply_cubic_vector(vector, payload, field, work)
        vector = apply_gaussian_vector(vector, tuple(seed.deformed.symplectic), 1, field, work)
        for operation, payload in left_ops:
            vector = apply_gaussian_vector(vector, payload[0], payload[1], field, work) if operation == "GAUSSIAN" else apply_cubic_vector(vector, payload, field, work)
        total = (total + weight * fiber[2 * source + target] * vector[x]) % field.p
    return {
        "boundary": total,
        "public_word_plan_cells": 12 * depth,
        "live_vector_field_cells": 2 * q,
        "resident_dynamic_relation_cells": 0,
        "work": work,
        "cold_start_comparison_used": False,
    }


def transaction(carrier: Carrier, depth: int, family: str, method: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    records, work, executed = forward(carrier, depth, family, method)
    projected = boundary(carrier, family)
    commitment = semantic_digest(carrier)
    final_point = checkpoint(carrier, depth)
    baseline = baseline_boundary(carrier.field.q, depth, family)
    reverse(carrier, executed, method, work)
    exact = carrier.canonical_state() == before
    same_backing = carrier.backing_ids() == backing
    generation_stable = carrier.restoration_generation == generation
    if not exact or not same_backing or not generation_stable or projected != baseline["boundary"]:
        fail(
            f"cubic-deformed transaction qualification failed q={carrier.field.q} depth={depth} "
            f"family={family} exact={exact} backing={same_backing} generation={generation_stable} "
            f"boundary={projected} baseline={baseline['boundary']}"
        )
    carrier.restoration_generation += 1
    return {
        "q": carrier.field.q,
        "p": carrier.field.p,
        "family": family,
        "depth": depth,
        "method": method,
        "boundary": projected,
        "semantic_commitment": commitment,
        "checkpoints": records,
        "final_checkpoint": final_point,
        "work": work.__dict__,
        "matched_q_vector_classical": {**baseline, "boundary_matches": True},
        "exact_canonical_state_restored": exact,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_relation_entries_serialized": False,
    }


def dense_relation(relation: Relation) -> list[list[int]]:
    return [[kernel_entry(relation, x, y) for y in range(relation.field.q)] for x in range(relation.field.q)]


def dense_multiply(left: list[list[int]], right: list[list[int]], p: int) -> list[list[int]]:
    q = len(left)
    return [[sum(left[x][z] * right[z][y] for z in range(q)) % p for y in range(q)] for x in range(q)]


def algebra_checks() -> dict[str, bool]:
    field = gaussian.make_field(5)
    symplectics = [
        [a, b, c, d]
        for a in range(5) for b in range(5) for c in range(5) for d in range(5)
        if (a * d - b * c) % 5 == 1
    ]
    covariance = True
    identity_fiber = [1, 0, 0, 1]
    for symplectic in symplectics:
        gaussian_relation = Relation(field, symplectic, identity_fiber, {(0, 0): 1})
        for r in range(5):
            for s in range(5):
                displacement = Relation(field, [1, 0, 0, 1], identity_fiber, {(r, s): 1})
                product = compose(gaussian_relation, displacement, "STREAMED_PHASE_SUM", Work())
                if dense_relation(product) != dense_multiply(dense_relation(gaussian_relation), dense_relation(displacement), field.p):
                    covariance = False
                    break
            if not covariance:
                break
        if not covariance:
            break
    cubic_reconstruction = True
    for q in (5, 11):
        q_field = gaussian.make_field(q)
        for strength in range(q):
            operator = cubic_operator(q_field, strength, Work())
            dense = dense_relation(operator)
            if any(dense[x][y] != (gaussian.phase(q_field, strength * x * x * x) if x == y else 0) for x in range(q) for y in range(q)):
                cubic_reconstruction = False
                break
    source = Relation(field, [1, 1, 1, 2], [1, 0, 0, 1], {(0, 0): 1})
    cubic = cubic_intersection(source, 2, 3, "STREAMED_PHASE_SUM", Work())
    dense_source = dense_relation(source)
    dense_expected = [
        [gaussian.phase(field, 2 * x * x * x + 3 * y * y * y) * dense_source[x][y] % field.p for y in range(5)]
        for x in range(5)
    ]
    return {
        "all_120_sl2_q5_elements_enumerated": len(symplectics) == 120,
        "all_3000_q5_gaussian_displacement_products_match_dense_semantics": covariance,
        "all_q5_q11_cubic_spectra_reconstruct_exact_diagonals": cubic_reconstruction,
        "cubic_hadamard_intersection_matches_dense_semantics": dense_relation(cubic) == dense_expected,
    }


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    expected = Carrier.seal(11).canonical_state()
    failures = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        carrier = Carrier.seal(11)
        _, work, executed = forward(carrier, 1, "PRIMARY", "STREAMED_PHASE_SUM")
        reverse(carrier, executed, "STREAMED_PHASE_SUM", work, mutation)
        failures[mutation] = carrier.canonical_state() != expected
    normal = Carrier.seal(11)
    forward(normal, 1, "PRIMARY", "STREAMED_PHASE_SUM")
    altered = Carrier.seal(11)
    forward(altered, 1, "PRIMARY", "STREAMED_PHASE_SUM", reorder_modules=True)
    idle = Carrier.seal(11)
    null_work = Work()
    null_source = Relation(idle.field, idle.deformed.symplectic[:], idle.deformed.fiber[:], dict(idle.deformed.coefficients))
    null_result = cubic_intersection(null_source, 0, 0, "STREAMED_PHASE_SUM", null_work)
    return {
        "missing_inverse_fails_restoration": failures["MISSING"],
        "wrong_inverse_fails_restoration": failures["WRONG"],
        "reordered_inverse_fails_restoration": failures["REORDER"],
        "module_reordering_changes_final_boundary": boundary(normal, "PRIMARY") != boundary(altered, "PRIMARY"),
        "premature_boundary_projection_rejected": raises(lambda: boundary(idle, "PRIMARY")),
        "null_cubic_intersection_is_identity": null_result.canonical() == null_source.canonical(),
        "null_family_rejected": raises(lambda: Carrier.seal(11, "NULL")),
        "non_safe_prime_order_rejected": raises(lambda: Carrier.seal(17)),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(23)
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 1, "PRIMARY", "STREAMED_PHASE_SUM")
    second = transaction(carrier, 2, "ALTERNATE", "STREAMED_PHASE_SUM")
    fresh = transaction(Carrier.seal(23), 2, "ALTERNATE", "STREAMED_PHASE_SUM")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_canonical_state_restored_after_reuse": carrier.canonical_state() == before,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "unrelated_second_commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "snapshot_used": False,
    }


def case_specs() -> list[tuple[int, int, str]]:
    cases = [(q, BASE_DEPTH, "PRIMARY") for q in ORDERS]
    cases.append((DEEPEST_ORDER, DEEPEST_DEPTH, "PRIMARY"))
    cases.append((ALTERNATE_ORDER, ALTERNATE_DEPTH, "ALTERNATE"))
    return cases


def build_result() -> dict[str, Any]:
    cases = [transaction(Carrier.seal(q), depth, family, "STREAMED_PHASE_SUM") for q, depth, family in case_specs()]
    checks = controls()
    algebra = algebra_checks()
    reuse = reuse_check()
    first_support_law = all(case["checkpoints"][0]["deformed_port_gaussian_component_support"] >= (case["q"] - 3) ** 2 for case in cases)
    if not all(checks.values()) or not all(algebra.values()) or not first_support_law:
        fail(f"cubic-deformed controls, algebra, or support law failed controls={checks} algebra={algebra} first_support_law={first_support_law} first_supports={[(case['q'], case['checkpoints'][0]['deformed_port_gaussian_component_support']) for case in cases]}")
    if not all(case["exact_canonical_state_restored"] and case["same_backing_restored"] and case["matched_q_vector_classical"]["boundary_matches"] for case in cases):
        fail("cubic-deformed case qualification failed")
    if not all(reuse[key] for key in ("same_backing_reused", "exact_canonical_state_restored_after_reuse", "unrelated_second_boundary_matches_fresh", "unrelated_second_commitment_matches_fresh")):
        fail("cubic-deformed reuse failed")
    max_q = max(case["q"] for case in cases)
    max_depth = max(case["depth"] for case in cases)
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "safe_prime_orders_q": list(ORDERS),
            "coefficient_fields_p": [2 * q + 1 for q in ORDERS],
            "deformed_port_state": "PACKED_WEYL_DISPLACEMENT_COEFFICIENTS_TIMES_ONE_COMMON_WEIL_GAUSSIAN_CHART_AND_2X2_FIBER_MATRIX",
            "second_port_state": "ONE_ACTUAL_RESIDENT_WEIL_GAUSSIAN_CHART_AND_2X2_FIBER_MATRIX",
            "native_composition": "EXACT_PROJECTIVE_WEIL_WEYL_TWISTED_COMPONENT_COMPOSITION",
            "native_intersection": "PUBLIC_SEPARABLE_CUBIC_ROW_COLUMN_PHASE_HADAMARD_INTERSECTION",
            "resident_gaussian_port_consumed_by_deformed_port": True,
            "retained_inverse_history_cells": 0,
            "ordinary_relation_table_materialized_on_accepted_path": False,
            "final_boundary_only": True,
            "hidden_relation_entries_serialized": False,
            "restoration_generation_enforcement": "DIRECT_PROCESS_BOOKKEEPING_ONLY",
        },
        "cases": cases,
        "at_least_q_minus_3_squared_deformed_component_support_after_first_cubic_intersection_all_cases": first_support_law,
        "observed_first_cubic_component_supports": [case["checkpoints"][0]["deformed_port_gaussian_component_support"] for case in cases],
        "observed_final_component_supports": [case["final_checkpoint"]["deformed_port_gaussian_component_support"] for case in cases],
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "maximum_q": max_q,
            "maximum_depth": max_depth,
            "accepted_resident_packed_field_cells_at_full_support": f"3*Q^2+19",
            "accepted_cubic_update_conservative_two_map_peak_field_cells": f"6*Q^2+Q+19",
            "accepted_cubic_spectrum_generation_phase_terms_per_side": "Q^2",
            "accepted_component_convolution_pair_products_measured": True,
            "matched_q_vector_live_field_cells": "2*Q",
            "matched_public_word_plan_cells": "12*DEPTH",
            "matched_classical_retained_dynamic_relation_cells": 0,
            "dense_two_port_relation_field_cells": "8*Q^2",
            "snapshot_cells": 0,
            "controller_backend_traffic_bytes": 0,
            "python_dict_list_integer_allocator_interpreter_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest_executed": "EXACT_STREAMED_Q_VECTOR_PUBLIC_OPERATOR_WORD_RECURRENCE_FOR_THE_DECLARED_FINAL_BOUNDARY",
            "all_case_boundaries_match": True,
            "linear_q_dynamic_state": True,
            "public_word_plan_grows_with_depth_and_is_counted": True,
            "dense_relation_matrices_are_QUALIFICATION_ORACLE_NOT_MATCHED_BASELINE": True,
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH2_ALL_PRIMARY_DEPTH4_Q113_ALTERNATE_DEPTH3_Q41_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "COMPACT_NON_GAUSSIAN_CLOSURE",
            "SUBQUADRATIC_GAUSSIAN_COMPONENT_SUPPORT_ACROSS_GROWING_Q",
            "EXACT_FULL_Q2_COMPONENT_SATURATION",
            "GENERAL_CUBIC_WEIL_RELATION_COMPILER",
            "FIXED_BIT_WIDTH_ACROSS_UNBOUNDED_Q",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "ONE_CUBIC_INTERSECTION_FORCES_AT_LEAST_Q_MINUS3_SQUARED_WEYL_DISPLACEMENT_COMPONENTS_ON_EVERY_DECLARED_ORDER_WHILE_THE_EXECUTED_FINAL_BOUNDARY_CLASSICAL_WORD_RECURRENCE_USES_ONLY_LINEAR_Q_DYNAMIC_STATE_SO_A_PHASE_NATIVE_ESCAPE_MUST_AVOID_GENERIC_NON_GAUSSIAN_OPERATOR_SUPPORT_OR_USE_A_RESOURCE_BEYOND_FINITE_FIELD_COMPONENT_ARITHMETIC",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
