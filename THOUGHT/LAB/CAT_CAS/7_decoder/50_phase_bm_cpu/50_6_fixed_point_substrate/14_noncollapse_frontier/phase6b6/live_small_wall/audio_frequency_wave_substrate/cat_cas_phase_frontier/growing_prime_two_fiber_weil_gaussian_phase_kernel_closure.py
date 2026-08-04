#!/usr/bin/env python3
"""Exact growing-prime two-fiber Weil-Gaussian relation chart.

The accepted path retains two unresolved relation ports as one homogeneous
quadratic Weil kernel and one 2 x 2 fiber matrix per port.  It composes ports
through an exact streamed Gauss cocycle and intersects them with public
separable chirp relations without materializing an ordinary relation table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SAFE_PRIME_ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
PRIMARY_DEPTH = 256
DEEPEST_ORDER = 113
DEEPEST_DEPTH = 1024
ALTERNATE_ORDER = 41
ALTERNATE_DEPTH = 128
CHECKPOINTS = (1, 4, 16, 64, 128, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_"
    "RELATION_CHART_CLOSES_NATIVE_NONCOMMUTATIVE_COMPOSITION_AND_SEPARABLE_"
    "CHIRP_HADAMARD_INTERSECTION_IN_FIXED16_TWO_PORT_CELLS_THROUGH_DEPTH1024_"
    "WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL16_"
    "CELL_CLASSICAL_CHART_WITH_CLOSED_GAUSS_COCYCLE_REMAINS"
)


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
    primitive: int
    root: int
    gauss_one: int
    primitive_search_candidates: int


def make_field(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("order is not a declared safe-prime pair")
    primitive = 2
    while primitive < p:
        if pow(primitive, 2, p) != 1 and pow(primitive, q, p) != 1:
            break
        primitive += 1
    if primitive == p:
        fail("primitive root not found")
    root = pow(primitive, 2, p)
    if pow(root, q, p) != 1 or any(pow(root, exponent, p) == 1 for exponent in range(1, q)):
        fail("phase root does not have exact order q")
    gauss_one = sum(pow(root, (x * x) % q, p) for x in range(q)) % p
    if gauss_one == 0:
        fail("quadratic Gauss sum vanished")
    return Field(q, p, primitive, root, gauss_one, primitive - 1)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


def symplectic_multiply(left: list[int], right: list[int], q: int) -> list[int]:
    a, b, c, d = left
    e, f, g, h = right
    return [
        (a * e + b * g) % q,
        (a * f + b * h) % q,
        (c * e + d * g) % q,
        (c * f + d * h) % q,
    ]


def symplectic_inverse(matrix: list[int], q: int) -> list[int]:
    a, b, c, d = matrix
    if (a * d - b * c) % q != 1:
        fail("non-symplectic relation chart")
    return [d % q, -b % q, -c % q, a % q]


def fiber_multiply(left: list[int], right: list[int], p: int) -> list[int]:
    a, b, c, d = left
    e, f, g, h = right
    return [
        (a * e + b * g) % p,
        (a * f + b * h) % p,
        (c * e + d * g) % p,
        (c * f + d * h) % p,
    ]


def fiber_scale(matrix: list[int], scalar: int, p: int) -> list[int]:
    return [scalar * value % p for value in matrix]


def fiber_inverse(matrix: list[int], p: int) -> list[int]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % p
    if determinant == 0:
        fail("noninvertible fiber matrix")
    inverse = pow(determinant, -1, p)
    return [d * inverse % p, -b * inverse % p, -c * inverse % p, a * inverse % p]


def kernel_value(symplectic: list[int], x: int, y: int, field: Field) -> int:
    a, b, c, d = symplectic
    q = field.q
    if b % q:
        inverse = pow(2 * b % q, -1, q)
        exponent = (d * x * x - 2 * x * y + a * y * y) * inverse
        return phase(field, exponent)
    if y % q != d * x % q:
        return 0
    return phase(field, c * d * x * x * pow(2, -1, q))


@dataclass
class Work:
    streamed_cocycle_calls: int = 0
    streamed_phase_sum_terms: int = 0
    closed_cocycle_calls: int = 0
    legendre_symbol_calls: int = 0
    composition_calls: int = 0
    chirp_intersection_calls: int = 0
    inverse_relation_rematerializations: int = 0


def cocycle_streamed(left: list[int], right: list[int], field: Field, work: Work) -> int:
    work.streamed_cocycle_calls += 1
    work.streamed_phase_sum_terms += field.q
    return sum(
        kernel_value(left, 0, shared, field) * kernel_value(right, shared, 0, field)
        for shared in range(field.q)
    ) % field.p


def cocycle_closed(left: list[int], right: list[int], field: Field, work: Work) -> int:
    work.closed_cocycle_calls += 1
    a, b, _, _ = left
    _, f, _, h = right
    if b % field.q == 0 or f % field.q == 0:
        return 1
    coefficient = (
        a * pow(2 * b % field.q, -1, field.q)
        + h * pow(2 * f % field.q, -1, field.q)
    ) % field.q
    if coefficient == 0:
        return field.q % field.p
    work.legendre_symbol_calls += 1
    character = pow(coefficient, (field.q - 1) // 2, field.q)
    return field.gauss_one if character == 1 else -field.gauss_one % field.p


@dataclass
class Relation:
    field: Field
    symplectic: list[int]
    fiber: list[int]

    def canonical(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(value % self.field.q for value in self.symplectic), tuple(
            value % self.field.p for value in self.fiber
        )

    def backing_ids(self) -> tuple[int, int, int]:
        return id(self), id(self.symplectic), id(self.fiber)


def set_relation(target: Relation, source: Relation) -> None:
    target.symplectic[:] = source.symplectic
    target.fiber[:] = source.fiber


def compose_relation(left: Relation, right: Relation, method: str, work: Work) -> Relation:
    if left.field != right.field:
        fail("field mismatch")
    field = left.field
    if method == "STREAMED_PHASE_SUM":
        scalar = cocycle_streamed(left.symplectic, right.symplectic, field, work)
    elif method == "CLOSED_CLASSICAL_GAUSS":
        scalar = cocycle_closed(left.symplectic, right.symplectic, field, work)
    else:
        fail("unknown cocycle method")
    if scalar == 0:
        fail("composition cocycle vanished")
    work.composition_calls += 1
    return Relation(
        field,
        symplectic_multiply(left.symplectic, right.symplectic, field.q),
        fiber_scale(fiber_multiply(left.fiber, right.fiber, field.p), scalar, field.p),
    )


def inverse_relation(source: Relation, method: str, work: Work) -> Relation:
    inverse_symplectic = symplectic_inverse(source.symplectic, source.field.q)
    if method == "STREAMED_PHASE_SUM":
        scalar = cocycle_streamed(source.symplectic, inverse_symplectic, source.field, work)
    else:
        scalar = cocycle_closed(source.symplectic, inverse_symplectic, source.field, work)
    work.inverse_relation_rematerializations += 1
    return Relation(
        source.field,
        inverse_symplectic,
        fiber_scale(fiber_inverse(source.fiber, source.field.p), pow(scalar, -1, source.field.p), source.field.p),
    )


def chirp_intersection(source: Relation, row_strength: int, column_strength: int, work: Work) -> Relation:
    q = source.field.q
    left_chirp = [1, 0, 2 * row_strength % q, 1]
    right_chirp = [1, 0, 2 * column_strength % q, 1]
    work.chirp_intersection_calls += 1
    return Relation(
        source.field,
        symplectic_multiply(
            symplectic_multiply(left_chirp, source.symplectic, q), right_chirp, q
        ),
        source.fiber[:],
    )


def relation_value(relation: Relation, source: int, target: int, x: int, y: int) -> int:
    return relation.fiber[2 * source + target] * kernel_value(
        relation.symplectic, x, y, relation.field
    ) % relation.field.p


@dataclass(frozen=True)
class Stage:
    target: str
    operation: str
    index: int
    family: str
    row_strength: int = 0
    column_strength: int = 0


def descriptors(index: int, family: str, q: int) -> tuple[Stage, ...]:
    code = 1 if family == "PRIMARY" else 2
    stages = (
        Stage("A", "RIGHT_COMPOSE_SOURCE", index, family),
        Stage("B", "CHIRP_INTERSECTION", index, family, (3 * index + code) % q, (5 * index + 2 * code + 1) % q),
        Stage("B", "LEFT_COMPOSE_SOURCE", index, family),
        Stage("A", "CHIRP_INTERSECTION", index, family, (7 * index + 3 * code + 1) % q, (11 * index + code + 2) % q),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


@dataclass
class Carrier:
    field: Field
    seed_family: str
    a: Relation
    b: Relation
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, q: int, family: str = "PRIMARY") -> "Carrier":
        if family not in FAMILIES:
            fail("invalid family")
        field = make_field(q)
        code = 1 if family == "PRIMARY" else 2
        a = Relation(field, [1, 1, 1, 2], [1, 2 + code, 3, 7 + 3 * code])
        b = Relation(field, [2, 1, 1, 1], [2, 1 + code, 5, 3 + 3 * code])
        fiber_inverse(a.fiber, field.p)
        fiber_inverse(b.fiber, field.p)
        return cls(field, family, a, b)

    def canonical_state(self) -> tuple[Any, ...]:
        return (
            self.field.q,
            self.field.p,
            self.seed_family,
            self.a.canonical(),
            self.b.canonical(),
            self.stage,
        )

    def backing_ids(self) -> tuple[int, ...]:
        return self.a.backing_ids() + self.b.backing_ids()


def other(carrier: Carrier, target: str) -> tuple[Relation, Relation]:
    return (carrier.a, carrier.b) if target == "A" else (carrier.b, carrier.a)


def apply_stage(carrier: Carrier, stage: Stage, method: str, work: Work) -> None:
    target, source = other(carrier, stage.target)
    if stage.operation == "RIGHT_COMPOSE_SOURCE":
        updated = compose_relation(target, source, method, work)
    elif stage.operation == "LEFT_COMPOSE_SOURCE":
        updated = compose_relation(source, target, method, work)
    elif stage.operation == "CHIRP_INTERSECTION":
        updated = chirp_intersection(target, stage.row_strength, stage.column_strength, work)
    else:
        fail("unknown stage")
    set_relation(target, updated)


def undo_stage(carrier: Carrier, stage: Stage, method: str, work: Work, wrong: bool = False) -> None:
    target, source = other(carrier, stage.target)
    if stage.operation == "CHIRP_INTERSECTION":
        row = -stage.row_strength + (1 if wrong else 0)
        updated = chirp_intersection(target, row, -stage.column_strength, work)
    else:
        inverse_source = inverse_relation(source, method, work)
        if wrong:
            inverse_source.fiber[0] = (inverse_source.fiber[0] + 1) % source.field.p
        if stage.operation == "RIGHT_COMPOSE_SOURCE":
            updated = compose_relation(target, inverse_source, method, work)
        elif stage.operation == "LEFT_COMPOSE_SOURCE":
            updated = compose_relation(inverse_source, target, method, work)
        else:
            fail("unknown inverse stage")
    set_relation(target, updated)


def checkpoint(carrier: Carrier, depth: int) -> dict[str, int]:
    return {
        "depth": depth,
        "resident_relation_components": 2,
        "resident_symplectic_cells": 8,
        "resident_fiber_cells": 8,
        "resident_two_port_cells": 16,
        "logical_resident_payload_bits": 8 * (carrier.field.q - 1).bit_length()
        + 8 * (carrier.field.p - 1).bit_length(),
        "ordinary_dense_two_port_relation_cells": 8 * carrier.field.q * carrier.field.q,
        "monomial_kernel_ports": sum(relation.symplectic[1] % carrier.field.q == 0 for relation in (carrier.a, carrier.b)),
    }


def forward(carrier: Carrier, depth: int, family: str, method: str, reverse_modules: bool = False) -> tuple[list[dict[str, int]], Work]:
    if carrier.stage != "IDLE" or family not in FAMILIES or depth < 1:
        fail("invalid forward request")
    work = Work()
    records = []
    for index in range(depth):
        stages = descriptors(index, family, carrier.field.q)
        if reverse_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply_stage(carrier, stage, method, work)
        if index + 1 in CHECKPOINTS:
            records.append(checkpoint(carrier, index + 1))
    carrier.stage = "FORWARD_COMPLETE"
    return records, work


def reverse(carrier: Carrier, depth: int, family: str, method: str, work: Work, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier lacks forward state")
    sequence = [stage for index in reversed(range(depth)) for stage in reversed(descriptors(index, family, carrier.field.q))]
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, stage in enumerate(sequence):
        undo_stage(carrier, stage, method, work, wrong=mutation == "WRONG" and position == 0)
    carrier.stage = "IDLE"


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    q = carrier.field.q
    p = carrier.field.p
    code = 1 if family == "PRIMARY" else 2
    probes = (
        (carrier.a, 0, 1, (3 * code + 1) % q, (5 * code + 2) % q, 1),
        (carrier.a, 1, 0, (7 * code + 2) % q, (11 * code + 3) % q, 2),
        (carrier.b, 0, 0, (13 * code + 1) % q, (17 * code + 4) % q, 3),
        (carrier.b, 1, 1, (19 * code + 2) % q, (23 * code + 5) % q, 5),
    )
    return sum(weight * relation_value(relation, source, target, x, y) for relation, source, target, x, y, weight in probes) % p


def semantic_digest(carrier: Carrier) -> str:
    digest = hashlib.sha256()
    for relation in (carrier.a, carrier.b):
        for source in range(2):
            for x in range(carrier.field.q):
                for target in range(2):
                    for y in range(carrier.field.q):
                        digest.update(relation_value(relation, source, target, x, y).to_bytes(2, "big"))
    return digest.hexdigest()


def transaction(carrier: Carrier, depth: int, family: str, method: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    checkpoints, work = forward(carrier, depth, family, method)
    projected = boundary(carrier, family)
    commitment = semantic_digest(carrier)
    final_checkpoint = checkpoint(carrier, depth)
    reverse(carrier, depth, family, method, work)
    exact = carrier.canonical_state() == before
    same_backing = carrier.backing_ids() == backing
    generation_unchanged_during_inverse = carrier.restoration_generation == generation
    if not exact or not same_backing or not generation_unchanged_during_inverse:
        fail("Weil-Gaussian carrier restoration failed")
    carrier.restoration_generation += 1
    return {
        "q": carrier.field.q,
        "p": carrier.field.p,
        "family": family,
        "depth": depth,
        "method": method,
        "boundary": projected,
        "semantic_commitment": commitment,
        "checkpoints": checkpoints,
        "final_checkpoint": final_checkpoint,
        "work": work.__dict__,
        "exact_canonical_state_restored": exact,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_relation_entries_serialized": False,
    }


def dense_kernel(symplectic: list[int], field: Field) -> list[list[int]]:
    return [[kernel_value(symplectic, x, y, field) for y in range(field.q)] for x in range(field.q)]


def dense_multiply(left: list[list[int]], right: list[list[int]], p: int) -> list[list[int]]:
    size = len(left)
    return [[sum(left[x][shared] * right[shared][y] for shared in range(size)) % p for y in range(size)] for x in range(size)]


def algebra_checks() -> dict[str, bool]:
    field = make_field(5)
    all_sl2 = [
        [a, b, c, d]
        for a in range(5)
        for b in range(5)
        for c in range(5)
        for d in range(5)
        if (a * d - b * c) % 5 == 1
    ]
    cocycles_match = True
    projective_dense_match = True
    for left in all_sl2:
        for right in all_sl2:
            streamed = cocycle_streamed(left, right, field, Work())
            closed = cocycle_closed(left, right, field, Work())
            if streamed != closed:
                cocycles_match = False
                break
            output = symplectic_multiply(left, right, field.q)
            dense_product = dense_multiply(dense_kernel(left, field), dense_kernel(right, field), field.p)
            if any(
                dense_product[x][y] != streamed * kernel_value(output, x, y, field) % field.p
                for x in range(field.q)
                for y in range(field.q)
            ):
                projective_dense_match = False
                break
        if not cocycles_match or not projective_dense_match:
            break
    relation = Relation(field, [1, 1, 1, 2], [1, 3, 3, 10])
    chirped = chirp_intersection(relation, 2, 3, Work())
    chirp_dense_match = all(
        kernel_value(chirped.symplectic, x, y, field)
        == kernel_value(relation.symplectic, x, y, field) * phase(field, 2 * x * x + 3 * y * y) % field.p
        for x in range(field.q)
        for y in range(field.q)
    )
    inverse = inverse_relation(relation, "STREAMED_PHASE_SUM", Work())
    identity = compose_relation(relation, inverse, "STREAMED_PHASE_SUM", Work())
    inverse_exact = identity.canonical() == Relation(field, [1, 0, 0, 1], [1, 0, 0, 1]).canonical()
    left = Relation(field, [1, 1, 1, 2], [1, 3, 3, 10])
    right = Relation(field, [2, 1, 1, 1], [2, 2, 5, 6])
    noncommutative = compose_relation(left, right, "STREAMED_PHASE_SUM", Work()).canonical() != compose_relation(
        right, left, "STREAMED_PHASE_SUM", Work()
    ).canonical()
    return {
        "all_120_sl2_q5_elements_enumerated": len(all_sl2) == 120,
        "all_14400_ordered_q5_cocycle_pairs_streamed_equal_closed": cocycles_match,
        "all_14400_ordered_q5_projective_products_match_dense_semantics": projective_dense_match,
        "separable_chirp_hadamard_intersection_matches_dense_semantics": chirp_dense_match,
        "relation_inverse_composes_to_exact_identity": inverse_exact,
        "two_fiber_relation_composition_is_noncommutative": noncommutative,
    }


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    expected = Carrier.seal(23).canonical_state()
    failures: dict[str, bool] = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        carrier = Carrier.seal(23)
        _, work = forward(carrier, 4, "PRIMARY", "STREAMED_PHASE_SUM")
        reverse(carrier, 4, "PRIMARY", "STREAMED_PHASE_SUM", work, mutation)
        failures[mutation] = carrier.canonical_state() != expected
    normal = Carrier.seal(23)
    forward(normal, 4, "PRIMARY", "STREAMED_PHASE_SUM")
    altered = Carrier.seal(23)
    forward(altered, 4, "PRIMARY", "STREAMED_PHASE_SUM", reverse_modules=True)
    idle = Carrier.seal(23)
    return {
        "missing_inverse_fails_restoration": failures["MISSING"],
        "wrong_inverse_fails_restoration": failures["WRONG"],
        "reordered_inverse_fails_restoration": failures["REORDER"],
        "module_reordering_changes_final_boundary": boundary(normal, "PRIMARY") != boundary(altered, "PRIMARY"),
        "premature_boundary_projection_rejected": raises(lambda: boundary(idle, "PRIMARY")),
        "null_family_rejected": raises(lambda: Carrier.seal(23, "NULL")),
        "non_safe_prime_order_rejected": raises(lambda: Carrier.seal(17)),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(53)
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 4, "PRIMARY", "STREAMED_PHASE_SUM")
    second = transaction(carrier, 64, "ALTERNATE", "STREAMED_PHASE_SUM")
    fresh = transaction(Carrier.seal(53), 64, "ALTERNATE", "STREAMED_PHASE_SUM")
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
    cases = [(q, PRIMARY_DEPTH, "PRIMARY") for q in SAFE_PRIME_ORDERS]
    cases.append((DEEPEST_ORDER, DEEPEST_DEPTH, "PRIMARY"))
    cases.append((ALTERNATE_ORDER, ALTERNATE_DEPTH, "ALTERNATE"))
    return cases


def build_result() -> dict[str, Any]:
    cases = []
    for q, depth, family in case_specs():
        accepted = transaction(Carrier.seal(q), depth, family, "STREAMED_PHASE_SUM")
        baseline = transaction(Carrier.seal(q), depth, family, "CLOSED_CLASSICAL_GAUSS")
        cases.append(
            {
                **accepted,
                "matched_classical": {
                    "boundary": baseline["boundary"],
                    "semantic_commitment": baseline["semantic_commitment"],
                    "checkpoints": baseline["checkpoints"],
                    "work": baseline["work"],
                    "boundary_matches": accepted["boundary"] == baseline["boundary"],
                    "semantic_commitment_matches": accepted["semantic_commitment"] == baseline["semantic_commitment"],
                    "checkpoints_match": accepted["checkpoints"] == baseline["checkpoints"],
                    "restoration_matches": baseline["exact_canonical_state_restored"] and baseline["same_backing_restored"],
                },
            }
        )
    checks = controls()
    algebra = algebra_checks()
    reuse = reuse_check()
    if not all(checks.values()) or not all(algebra.values()):
        fail("Weil-Gaussian controls or algebra checks failed")
    if not all(
        case["exact_canonical_state_restored"]
        and case["same_backing_restored"]
        and all(
            case["matched_classical"][key]
            for key in ("boundary_matches", "semantic_commitment_matches", "checkpoints_match", "restoration_matches")
        )
        for case in cases
    ):
        fail("Weil-Gaussian case qualification failed")
    if not all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "exact_canonical_state_restored_after_reuse",
            "unrelated_second_boundary_matches_fresh",
            "unrelated_second_commitment_matches_fresh",
        )
    ):
        fail("Weil-Gaussian reuse failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_CLOSURE_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "safe_prime_orders_q": list(SAFE_PRIME_ORDERS),
            "coefficient_fields_p": [2 * q + 1 for q in SAFE_PRIME_ORDERS],
            "native_state": "ONE_HOMOGENEOUS_QUADRATIC_WEIL_KERNEL_TIMES_ONE_2X2_FIBER_MATRIX_PER_PORT",
            "native_composition": "EXACT_PROJECTIVE_WEIL_COMPOSITION_WITH_STREAMED_PHASE_GAUSS_COCYCLE",
            "native_intersection": "HADAMARD_MULTIPLICATION_BY_PUBLIC_SEPARABLE_ROW_AND_COLUMN_CHIRP_RELATION",
            "resident_relation_components": 2,
            "resident_cells_per_port": 8,
            "resident_two_port_cells": 16,
            "retained_inverse_history_cells": 0,
            "retained_compiled_plan_cells": 0,
            "ordinary_relation_table_materialized_on_accepted_path": False,
            "final_boundary_only": True,
            "public_descriptor_compiler_reads_final_answer": False,
            "hidden_relation_entries_serialized": False,
            "restoration_generation_enforcement": "DIRECT_PROCESS_BOOKKEEPING_ONLY",
        },
        "cases": cases,
        "fixed16_two_port_cells_across_all_orders_and_depths": all(
            case["final_checkpoint"]["resident_two_port_cells"] == 16
            and all(point["resident_two_port_cells"] == 16 for point in case["checkpoints"])
            for case in cases
        ),
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "carrier_creation_field_cells": 16,
            "hidden_carrier_field_cells": 16,
            "public_field_plan_cells_q_p_primitive_root_gauss_one": 5,
            "accepted_creation_also_computes_public_gauss_one_phase_terms_per_q": "Q",
            "accepted_streamed_cocycle_terms_per_composition": "Q",
            "classical_warm_closed_cocycle_state_cells": 1,
            "classical_closed_cocycle_uses_legendre_symbol_and_cached_gauss_one": True,
            "gauss_one_cache_creation_phase_terms_per_q": "Q",
            "conservative_accepted_named_peak_field_cells": 56,
            "conservative_classical_named_peak_field_cells": 55,
            "dense_two_port_relation_field_cells": "8*Q^2",
            "projection_streams_four_public_entries": True,
            "verification_streams_but_does_not_retain_all_8Q2_relation_entries": True,
            "package_algebra_qualification_q5_dense_kernel_peak_field_cells": 75,
            "snapshot_cells": 0,
            "controller_backend_traffic_bytes": 0,
            "python_containers_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest_executed": "IDENTICAL16_CELL_WEIL_GAUSSIAN_RELATION_CHART_WITH_CLOSED_GAUSS_COCYCLE",
            "state_law_identical_to_accepted": True,
            "classical_composition_work_is_lower_after_one_scalar_gauss_cache": True,
            "dense_relation_matrices_are_verification_ORACLE_NOT_MATCHED_BASELINE": True,
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH256_ALL_PRIMARY_DEPTH1024_Q113_ALTERNATE_DEPTH128_Q41_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "ARBITRARY_GAUSSIAN_GAUSSIAN_HADAMARD_INTERSECTION",
            "ADDITIVE_SUPERPOSITION_OF_MULTIPLE_GAUSSIAN_COMPONENTS",
            "NON_GAUSSIAN_PHASE_CLOSURE",
            "FIXED_BIT_WIDTH_ACROSS_UNBOUNDED_Q",
            "GENERAL_RELATION_COMPILER",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "THE_FIXED16_WEIL_GAUSSIAN_CHART_IS_STABILIZER_LIKE_AND_THE_IDENTICAL_CLASSICAL_CHART_HAS_A_LOWER_WORK_CLOSED_GAUSS_COCYCLE_SO_THE_NEXT_PHASE_LAW_MUST_ADD_NON_GAUSSIAN_COUPLING_WITHOUT_COMPONENT_OR_PRECISION_GROWTH",
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
